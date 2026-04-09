from typing import Literal, List
from dataclasses import dataclass, field

import torch
import lightning as L
import torch.nn.functional as F
from transformers import CLIPModel
from peft import LoraConfig, get_peft_model
from torch.optim.lr_scheduler import CosineAnnealingLR


@dataclass
class SchedulerConfig:
    name: Literal["cosine", "none"] = "cosine"
    min_lr: float = 1e-7


@dataclass
class LearningConfig:
    lr: float = 5e-5
    weight_decay: float = 0.01
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass
class LoraTrainConfig:
    enabled: bool = False
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    use_dora: bool = True
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    fully_train_projectors: bool = True


class CLIPLightning(L.LightningModule):
    def __init__(
        self,
        model_id: str,
        learn_config: LearningConfig,
        processor,
        lora_config: LoraTrainConfig = None,  # type: ignore
        freeze_backbones: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["processor"])
        self.learn_config = learn_config
        self.processor = processor

        self.model = CLIPModel.from_pretrained(model_id)

        if lora_config is not None and lora_config.enabled:
            modules_to_save = []
            target_modules = list(lora_config.target_modules)

            if lora_config.fully_train_projectors:
                modules_to_save = ["text_projection", "visual_projection"]
            else:
                target_modules.extend(["text_projection", "visual_projection"])

            peft_config = LoraConfig(
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_config.lora_dropout,
                use_dora=lora_config.use_dora,
                bias="none",
                modules_to_save=modules_to_save,
            )

            self.model = get_peft_model(self.model, peft_config)
            self.model.base_model.model.logit_scale.requires_grad = True

        else:
            if freeze_backbones:
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False

                trainable_modules = [
                    self.model.text_projection,
                    self.model.visual_projection,
                ]

                for module in trainable_modules:
                    module.train()
                    for param in module.parameters():
                        param.requires_grad = True

                self.model.logit_scale.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(
            f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
        )

    def forward(self, pixel_values, input_ids=None, attention_mask=None):
        if input_ids is not None:
            return self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        return self.model.get_image_features(pixel_values=pixel_values)  # type: ignore

    def training_step(self, batch, batch_idx):
        inputs, _, _, _ = batch
        bs = inputs["pixel_values"].shape[0]
        outputs = self(**inputs)

        logits_img = outputs.logits_per_image
        logits_text = outputs.logits_per_text
        labels = torch.arange(bs, device=self.device)

        loss = (
            F.cross_entropy(logits_img, labels) + F.cross_entropy(logits_text, labels)
        ) / 2

        self.log("train_loss", loss, prog_bar=True, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, _, _, _ = batch
        bs = inputs["pixel_values"].shape[0]
        outputs = self(**inputs)

        logits_img = outputs.logits_per_image
        logits_text = outputs.logits_per_text
        labels = torch.arange(bs, device=self.device)

        loss = (
            F.cross_entropy(logits_img, labels) + F.cross_entropy(logits_text, labels)
        ) / 2

        self.log("val_loss", loss, prog_bar=True, batch_size=bs)

        preds = logits_img.argmax(dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_acc", acc, prog_bar=True, batch_size=bs)

    def configure_optimizers(self):  # type: ignore
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learn_config.lr,
            weight_decay=self.learn_config.weight_decay,
        )

        if self.learn_config.scheduler.name == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.estimated_stepping_batches,  # type: ignore
                eta_min=self.learn_config.scheduler.min_lr,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

        return optimizer

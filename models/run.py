import argparse
import torch
import yaml
import lightning as L
from pathlib import Path
from lightning.pytorch.loggers import TensorBoardLogger
from transformers import CLIPProcessor

from data import get_datasets, get_dataloaders
from callback import ModelCheckpointCallback
from train import CLIPLightning, LearningConfig, SchedulerConfig, LoraTrainConfig


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="CLIP Wiki Training Pipeline")
    parser.add_argument("--data_config", type=str, default="configs/data.yaml")
    parser.add_argument("--train_config", type=str, default="configs/train.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    data_cfg = load_yaml(args.data_config)
    train_cfg = load_yaml(args.train_config)

    L.seed_everything(train_cfg["trainer"]["seed"])
    device = get_device()
    print(f"Используем устройство: {device}")

    model_id = train_cfg["model"]["model_id"]
    processor = CLIPProcessor.from_pretrained(model_id)

    learn_config = LearningConfig(
        lr=float(train_cfg["optimizer"]["lr"]),
        weight_decay=float(train_cfg["optimizer"]["weight_decay"]),
        scheduler=SchedulerConfig(
            name=train_cfg["scheduler"]["name"],
            min_lr=float(train_cfg["scheduler"]["min_lr"]),
        ),
    )

    lora_params = train_cfg.get("lora", {})
    lora_config = LoraTrainConfig(
        enabled=lora_params.get("enabled", False),
        r=lora_params.get("r", 8),
        lora_alpha=lora_params.get("lora_alpha", 16),
        lora_dropout=lora_params.get("dropout", 0.05),
        use_dora=lora_params.get("use_dora", True),
        target_modules=lora_params.get("target_modules", ["q_proj", "v_proj"]),
        fully_train_projectors=lora_params.get("fully_train_projectors", True),
    )

    model = CLIPLightning(
        model_id=model_id,
        learn_config=learn_config,
        processor=processor,
        lora_config=lora_config,
        freeze_backbones=train_cfg["model"]["freeze_backbones"],
    )

    d_paths = data_cfg["paths"]
    aug_params = train_cfg["augmentation"]
    dl_params = train_cfg["dataloader"]

    train_ds, val_ds, index_ds = get_datasets(
        train_jsonl_path=d_paths["train_jsonl"],
        val_jsonl_path=d_paths["val_jsonl"],
        aug_map_path=d_paths["aug_map"],
        use_semantic_aug=aug_params["enabled"],
        imagenet_dir=d_paths["imagenet_dir"],
        semantic_aug_prob=aug_params["prob"],
    )

    train_loader, val_loader, _ = get_dataloaders(
        train_dataset=train_ds,
        val_dataset=val_ds,
        index_dataset=index_ds,
        processor=processor,
        batch_size=dl_params["batch_size"],
        num_workers=dl_params["num_workers"],
        pin_memory=(device != "cpu"),
    )

    t_paths = train_cfg["paths"]
    trainer_params = train_cfg["trainer"]

    logger = TensorBoardLogger(save_dir=t_paths["log_dir"], name="clip_wiki_finetuning")

    checkpoint_callback = ModelCheckpointCallback(
        save_dir=t_paths["model_checkpoints_dir"],
        every_n_epochs=trainer_params["epochs_between_updates"],
    )

    precision = trainer_params["precision"]
    if device == "cpu" and precision == "16-mixed":
        precision = "32-true"

    trainer = L.Trainer(
        max_epochs=trainer_params["max_epochs"],
        accelerator="mps",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=trainer_params["log_every_n_steps"],
        check_val_every_n_epoch=1,
        precision=precision,
        accumulate_grad_batches=trainer_params["accumulate_grad_batches"],
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()

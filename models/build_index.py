import json
import yaml
import torch
import faiss
import os
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

from data import WikiDataset

try:
    from peft import PeftModel

    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class IndexCollate:
    def __init__(self, processor: CLIPProcessor):
        self.processor = processor

    def __call__(self, batch):
        titles = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        images = [item[2] for item in batch]
        img_paths = [item[3] for item in batch]

        full_texts = [f"{title}. {text}" for title, text in zip(titles, texts)]

        inputs = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",  # type: ignore
            padding="max_length",  # type: ignore
            truncation=True,  # type: ignore
            max_length=77,  # type: ignore
        )
        return inputs, titles, texts, img_paths


def build_index_for_model(model_cfg, data_cfg, common_settings, device):
    checkpoint_path = model_cfg["checkpoint"]
    base_model_id = "openai/clip-vit-base-patch32"

    if checkpoint_path == base_model_id:
        model = CLIPModel.from_pretrained(base_model_id)
    elif (Path(checkpoint_path) / "adapter_config.json").exists():
        base = CLIPModel.from_pretrained(base_model_id)
        model = PeftModel.from_pretrained(base, checkpoint_path)  # type: ignore
        model = model.merge_and_unload()  # type: ignore
    else:
        model = CLIPModel.from_pretrained(checkpoint_path)

    model = model.to(device).eval()
    processor = CLIPProcessor.from_pretrained(base_model_id)

    dataset = WikiDataset(jsonl_path=data_cfg["all_jsonl"], mode="index")

    for sample in dataset.samples:
        if "images" in sample:
            sample["images"] = [
                os.path.join("./data/", img) for img in sample["images"]
            ]
        if "img_path" in sample:
            sample["img_path"] = os.path.join("./data/", sample["img_path"])

    dataloader = DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        collate_fn=IndexCollate(processor),
    )

    dim = model.config.projection_dim
    index = faiss.IndexFlatIP(dim)

    out_dir = Path(model_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / common_settings["meta_name"]
    target = common_settings["target"]

    idx_counter = 0
    with open(meta_path, "w", encoding="utf-8") as meta_file:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Indexing {target}"):
                inputs, titles, texts, img_paths = batch
                inputs = {k: v.to(device) for k, v in inputs.items()}

                if target == "text":
                    outputs = model.get_text_features(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                    )
                else:
                    outputs = model.get_image_features(
                        pixel_values=inputs["pixel_values"]
                    )

                features = (
                    outputs.pooler_output  # type: ignore
                    if hasattr(outputs, "pooler_output")
                    else outputs
                )
                features = torch.nn.functional.normalize(features, p=2, dim=-1)  # type: ignore

                index.add(features.cpu().numpy().astype("float32"))  # type: ignore

                for title, text, img_path in zip(titles, texts, img_paths):
                    record = {
                        "title": title,
                        "text": text,
                        "image_path": img_path,
                    }
                    meta_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    idx_counter += 1

    faiss.write_index(index, str(out_dir / common_settings["index_name"]))
    print(f"Индекс содержит {idx_counter} записей (картинок).")

    del model
    torch.cuda.empty_cache()


def main():
    cfg = load_yaml("configs/index.yaml")
    device = "cuda" if torch.cuda.is_available() else "mps"

    for m_cfg in cfg["models_to_index"]:
        try:
            build_index_for_model(m_cfg, cfg["data"], cfg["index_settings"], device)
        except Exception as e:
            print(f"Ошибка в {m_cfg['name']}: {e}")


if __name__ == "__main__":
    main()

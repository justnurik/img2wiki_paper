import json
import torch
import faiss
import yaml
from pathlib import Path
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    d_cfg = load_yaml("configs/data.yaml")
    t_cfg = load_yaml("configs/train.yaml")

    aug_cfg = d_cfg["aug_compute"]
    d_paths = d_cfg["paths"]

    model_id = t_cfg["model"]["model_id"]

    device = "cpu"

    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device).eval()  # type: ignore

    index = faiss.read_index(d_paths["faiss_index"])
    faiss.omp_set_num_threads(1)

    with open(d_paths["meta_path"], "r", encoding="utf-8") as f:
        metadata = json.load(f)

    data = []
    with open(d_paths["train_jsonl"], "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    aug_map = {}
    top_k = aug_cfg["top_k"]
    batch_size = aug_cfg["batch_size"]

    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i : i + batch_size]
        queries = [f"{item['title']}. {item['text']}" for item in batch]

        with torch.no_grad():
            inputs = processor(
                text=queries,
                return_tensors="pt",  # type: ignore
                padding=True,  # type: ignore
                truncation=True,  # type: ignore
                max_length=77,  # type: ignore
            ).to(device)

            outputs = model.get_text_features(**inputs)

            features = (
                outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs  # type: ignore
            )
            features /= features.norm(p=2, dim=-1, keepdim=True)  # type: ignore
            queries_np = features.cpu().numpy().astype("float32")

        _, indices = index.search(queries_np, top_k)

        for j, item in enumerate(batch):
            img_path_key = item["image_file"]
            aug_map[img_path_key] = [metadata[str(idx)] for idx in indices[j]]

    output_path = Path(aug_cfg["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aug_map, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

import json
import random
import yaml
from pathlib import Path


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_and_fix_paths(ds_path_str):
    ds_path = Path(ds_path_str)
    data = []
    jsonl_path = ds_path / "dataset.jsonl"
    img_dir_name = "images"

    if not jsonl_path.exists():
        return []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                if "image_file" in item:
                    item["image_file"] = str(
                        ds_path / img_dir_name / item["image_file"]
                    )
                    data.append(item)
            except:
                continue
    return data


def main():
    data_cfg = load_yaml("configs/data.yaml")
    cfg = data_cfg["merger"]

    all_data = []
    for s_dir in cfg["source_dirs"]:
        all_data.extend(load_and_fix_paths(s_dir))

    random.seed(cfg["seed"])
    random.shuffle(all_data)

    train_limit = cfg["train_size"]
    val_limit = cfg["val_size"]

    train_data = all_data[:train_limit]
    val_data = all_data[train_limit : train_limit + val_limit]

    def save_jsonl(data, output_path):
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    save_jsonl(train_data, cfg["train_output"])
    save_jsonl(val_data, cfg["val_output"])

    print(f"Total: {len(all_data)}")
    print(f"Train: {cfg['train_output']} ({len(train_data)})")
    print(f"Val: {cfg['val_output']} ({len(val_data)})")


if __name__ == "__main__":
    main()

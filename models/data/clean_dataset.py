import json
import yaml
import warnings
from pathlib import Path
from PIL import Image, ImageFile
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)


def is_valid(line):
    try:
        item = json.loads(line.strip())
        img_path = item["image_file"]

        with Image.open(img_path) as img:
            img.verify()
        with Image.open(img_path) as img:
            img.load()

        key = (item.get("title"), item.get("text"), item.get("image_file"))
        return line.strip(), key
    except:
        return None


def clean_file(path_str):
    path = Path(path_str)
    if not path.exists():
        return

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Processing {path.name} ({len(lines)} lines)...")

    results = []
    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(executor.map(is_valid, lines), total=len(lines), leave=False)
        )

    final_lines = []
    seen_keys = set()
    corrupted = 0
    duplicates = 0

    for res in results:
        if res is None:
            corrupted += 1
            continue

        line, key = res
        if key in seen_keys:
            duplicates += 1
            continue

        seen_keys.add(key)
        final_lines.append(line)

    with open(path, "w", encoding="utf-8") as f:
        for line in final_lines:
            f.write(line + "\n")

    print(f"Done {path.name}:")
    print(f"  - Valid & Unique: {len(final_lines)}")
    print(f"  - Corrupted removed: {corrupted}")
    print(f"  - Duplicates removed: {duplicates}")
    print(f"  - Total removed: {corrupted + duplicates}\n")


def main():
    with open("configs/data.yaml", "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    paths = data_cfg["paths"]
    clean_file(paths["train_jsonl"])
    clean_file(paths["val_jsonl"])


if __name__ == "__main__":
    main()

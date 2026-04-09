import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import albumentations as A
from PIL import Image, ImageFile
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from transformers import CLIPProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True

# FIXME: лучше наверно было бы сжать датасет
SAFE_PIXEL_LIMIT = 3000 * 3000


import json
import random
from PIL import Image
from torch.utils.data import Dataset


class WikiDataset(Dataset):
    def __init__(self, jsonl_path, mode="train"):
        """
        mode="train" -> Для одной статьи выдает 1 случайную картинку (чтобы не ломать Contrastive Loss).
        mode="index" -> Разворачивает статьи: каждая картинка становится отдельным сэмплом. (для построения индекса)
        """
        self.mode = mode
        self.samples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                title = data.get("title", "")
                text = data.get("text", "")

                img_list = data.get("image_files", [])
                if not img_list and data.get("image_file"):
                    img_list = [data.get("image_file")]

                if not img_list:
                    continue

                if self.mode == "index":
                    for img_path in img_list:
                        self.samples.append(
                            {
                                "title": title,
                                "text": text,
                                "images": [img_path],
                            }
                        )
                else:
                    self.samples.append(
                        {"title": title, "text": text, "images": img_list}
                    )

    def __len__(self):
        return len(self.samples)

    def get_info(self, idx):
        s = self.samples[idx]
        selected_img_path = random.choice(s["images"])
        return s["title"], s["text"], selected_img_path

    def __getitem__(self, idx):
        s = self.samples[idx]

        selected_img_path = random.choice(s["images"])

        try:
            image = Image.open(selected_img_path).convert("RGB")
        except Exception as e:
            # TODO: what the fuck???
            # я вроде решил предобработкой данных, таких случаев нет
            print("=" * 10 + f"ERROR [{selected_img_path}]" + "=" * 10)
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        return s["title"], s["text"], image, selected_img_path


class ComposedWikiDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        aug_map_path=None,
        imagenet_dir=None,
        pixel_transforms=None,
        semantic_aug_prob=0.5,
    ):
        self.base_dataset = base_dataset
        self.imagenet_dir = Path(imagenet_dir) if imagenet_dir else None
        self.pixel_transforms = pixel_transforms
        self.semantic_aug_prob = semantic_aug_prob
        self.aug_map = {}
        if aug_map_path and Path(aug_map_path).exists():
            with open(aug_map_path, "r") as f:
                self.aug_map = json.load(f)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        while True:
            try:
                title, text, img_path = self.base_dataset.get_info(index)

                if (
                    self.aug_map
                    and self.imagenet_dir
                    and np.random.random() < self.semantic_aug_prob
                ):
                    if img_path in self.aug_map:
                        chosen_name = np.random.choice(self.aug_map[img_path])
                        img_path = str(self.imagenet_dir / chosen_name)

                img = Image.open(img_path)

                if img.width * img.height > SAFE_PIXEL_LIMIT:
                    raise ValueError(f"Image too large: {img.width}x{img.height}")

                img = img.convert("RGB")

                if self.pixel_transforms is not None:
                    img = self.pixel_transforms(img)

                return title, text, img

            except Exception:
                index = np.random.randint(0, len(self.base_dataset))


class CLIPCollate:
    def __init__(self, processor: CLIPProcessor):
        self.processor = processor

    def __call__(self, batch):
        titles, texts, images = zip(*batch)

        full_texts = [f"{title}. {text}" for title, text in zip(titles, texts)]

        inputs = self.processor(
            text=full_texts,
            images=list(images),
            return_tensors="pt",  # type: ignore
            padding="max_length",  # type: ignore
            truncation=True,  # type: ignore
            max_length=77,  # type: ignore
        )

        return inputs, titles, texts, images


class AlbuWrapper:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img)
        augmented = self.transform(image=img_np)
        return Image.fromarray(augmented["image"])


def get_datasets(
    train_jsonl_path: str | Path,
    val_jsonl_path: str | Path,
    aug_map_path: Optional[str] = None,
    imagenet_dir: Optional[str | Path] = None,
    use_semantic_aug: bool = False,
    semantic_aug_prob: float = 0.5,
) -> Tuple[Dataset, Dataset, Dataset]:

    train_albu = A.Compose(
        [
            A.RandomResizedCrop(size=(224, 224), scale=(0.85, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.7),
        ]
    )
    train_pixel_transforms = AlbuWrapper(train_albu)

    val_albu = A.Compose(
        [A.SmallestMaxSize(max_size=224), A.CenterCrop(height=224, width=224)]
    )
    val_pixel_transforms = AlbuWrapper(val_albu)

    train_base = WikiDataset(jsonl_path=str(train_jsonl_path))
    val_base = WikiDataset(jsonl_path=str(val_jsonl_path))
    index_base = WikiDataset(jsonl_path=str(train_jsonl_path))

    train_dataset = ComposedWikiDataset(
        base_dataset=train_base,
        aug_map_path=aug_map_path if use_semantic_aug else None,
        imagenet_dir=str(imagenet_dir) if imagenet_dir else None,
        pixel_transforms=train_pixel_transforms,
        semantic_aug_prob=semantic_aug_prob if use_semantic_aug else 0.0,
    )

    val_dataset = ComposedWikiDataset(
        base_dataset=val_base,
        aug_map_path=None,
        imagenet_dir=None,
        pixel_transforms=val_pixel_transforms,
        semantic_aug_prob=0.0,
    )

    index_dataset = ComposedWikiDataset(
        base_dataset=index_base,
        aug_map_path=None,
        imagenet_dir=None,
        pixel_transforms=val_pixel_transforms,
        semantic_aug_prob=0.0,
    )

    return train_dataset, val_dataset, index_dataset


def get_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    index_dataset: Dataset,
    processor: CLIPProcessor,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    collate_fn = CLIPCollate(processor)

    is_multi = num_workers > 0
    p_factor = 2 if is_multi else None
    p_workers = True if is_multi else False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=p_workers,
        prefetch_factor=p_factor,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=p_workers,
        prefetch_factor=p_factor,
    )

    index_loader = DataLoader(
        index_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=p_workers,
        prefetch_factor=p_factor,
    )

    return train_loader, val_loader, index_loader

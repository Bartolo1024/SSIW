import os
from typing import Dict, Optional, List, Tuple, Union

import PIL.Image
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from torchvision import transforms
from src.utils import transforms_utils


class CMPDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "data/",
        img_size: Union[int, Tuple[int, int]] = (512, 512),
    ):
        self.img_size: Tuple[int, int] = (
            (img_size, img_size) if isinstance(img_size, int) else img_size
        )
        mean, std = transforms_utils.get_imagenet_mean_std()
        self.transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.ann_transform = transforms.Compose(
            [
                transforms.Resize(img_size, PIL.Image.NEAREST),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.items = []
        for file in os.listdir(data_dir):
            if file.endswith(".jpg"):
                img_path = os.path.join(data_dir, file)
                ann_path = os.path.join(data_dir, file.replace(".jpg", ".png"))
                assert os.path.isfile(
                    ann_path
                ), f"Corresponding annotation missed {ann_path}"
                self.items.append((img_path, ann_path))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, ann_path = self.items[index]
        img = PIL.Image.open(img_path).convert("RGB")
        ann = PIL.Image.open(ann_path)
        x = self.transform(img)
        target = self.ann_transform(ann)
        return x, target

    @staticmethod
    def categorical_mask_to_boolean() -> Image.Image:
        pass

import os
from typing import Dict, Optional, List, Tuple, Union

import PIL.Image
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import get_dimensions, crop


from src.utils.labels_dict import UNI_UID2UNAME
from src.utils import transforms_utils
from src.utils.get_class_emb import create_embs_from_names
import torch.nn.functional as F


class PairRandomCrop(transforms.RandomCrop):
    def forward(self, img1, img2):
        """
        Args:
            img1 (PIL Image or Tensor): Image to be cropped.
            img2 (PIL Image or Tensor): Image to be cropped.

        Returns:
            Tuple of two PIL Image or Tensor: Cropped image.
        """
        if self.padding is not None:
            img1 = F.pad(img1, self.padding, self.fill, self.padding_mode)
            img2 = F.pad(img2, self.padding, self.fill, self.padding_mode)

        _, height, width = get_dimensions(img1)
        _, height2, width2 = get_dimensions(img2)
        assert width2 == width
        assert height2 == height

        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img1 = F.pad(img1, padding, self.fill, self.padding_mode)
            img2 = F.pad(img2, padding, self.fill, self.padding_mode)

        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img1 = F.pad(img1, padding, self.fill, self.padding_mode)
            img2 = F.pad(img2, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img1, self.size)

        return crop(img1, i, j, h, w), crop(img2, i, j, h, w)


class CMPDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "data/",
        labels_path: str = "src/configs/labels.yaml",
        img_size: Union[int, Tuple[int, int]] = (256, 256),
        num_classes: int = 512
    ):
        self.num_classes = num_classes
        self.img_size: Tuple[int, int] = (
            (img_size, img_size) if isinstance(img_size, int) else img_size
        )
        mean, std = transforms_utils.get_imagenet_mean_std()
        self.pair_crop = PairRandomCrop(img_size)
        self.transform = transforms.Compose(
            [
                # transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.ann_transform = transforms.Compose(
            [
                # transforms.Resize(img_size, PIL.Image.NEAREST),
                transforms.PILToTensor(),
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
        self.labels = self.load_labels(labels_path)
        self.embeddigns = self.load_embeddings(self.labels)
        self.label_map = self.create_cmp_label_map(self.labels)

    def load_labels(self, path):
        with open(path, 'r') as f:
            labels = yaml.load(f, yaml.BaseLoader)
        return labels

    @staticmethod
    def load_embeddings(labels: Dict[str, Dict[str, Union[str, int]]]):
        keys = labels.keys()
        descriptions = {k: v['description'] for k, v in labels.items()}
        embs = create_embs_from_names(keys, descriptions)
        return embs

    def create_cmp_label_map(self, labels: Dict[str, Dict[str, Union[str, int]]], start_idx: int = 194):
        ret = {}
        keys = labels.keys()
        name_to_id = {v: k for k, v in UNI_UID2UNAME.items()}
        background_idx = name_to_id['unlabeled']
        ret[1] = background_idx
        idx = start_idx
        for key in keys:
            if key in name_to_id:
                ret[labels[key]['label']] = name_to_id[key]
            else:
                ret[labels[key]['label']] = idx
                idx += 1
        return ret

    def __len__(self):
        return len(self.items)

    def label_img_to_embedding_space(self, x: torch.Tensor):
        orig_shape = x.shape
        indices = x.view(-1)
        feature_map = self.embeddigns[indices]
        feature_map = feature_map.view(*orig_shape, self.num_classes)
        return feature_map

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path, ann_path = self.items[index]
        img = PIL.Image.open(img_path).convert("RGB")
        ann = PIL.Image.open(ann_path)
        img, ann = self.pair_crop(img, ann)
        x = self.transform(img)
        ann = self.ann_transform(ann)
        target = self.map_pixels(ann)

        one_hot = self.create_one_hot_label(target)
        target = self.label_img_to_embedding_space(target)
        return x, target, one_hot

    def map_pixels(self, x: torch.Tensor):
        c, h, w = x.shape
        ret = torch.zeros((h, w), dtype=torch.long)
        x = x.squeeze(0)
        for label, target in self.label_map.items():
            mask = x == label
            ret[mask] = target
        return ret

    def create_one_hot_label(self, ann: torch.Tensor):
        base_shape = ann.shape
        ann = ann.view(-1)
        one_hot = F.one_hot(ann, num_classes=self.num_classes)
        one_hot.view(*base_shape, self.num_classes)
        return one_hot
from typing import Dict, List

import click
import torch
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from torch.utils.data import random_split
from src.utils.dataflow import CMPDataset
from src.utils.segformer import get_configured_segformer


def create_loaders(data_root: str, train_ratio: float = 0.8, batch_size: int = 4):
    dataset = CMPDataset(data_root)
    train_len = int(len(dataset) * train_ratio)
    train_dataset, valid_dataset = random_split(
        dataset, lengths=[train_len, len(dataset) - train_len]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader


def get_model(num_classes: int, checkpoint_weights: str):
    model = get_configured_segformer(num_classes, load_imagenet_model=False)
    state_dict = torch.load(checkpoint_weights, map_location=torch.device("cpu"))
    state_dict = state_dict["state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    del state_dict["criterion.0.logit_scale"]
    model.load_state_dict(state_dict)
    return model


def create_step_fn(model, optimizer, loss_fn, device):
    def step_fn(batch):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        pass

    return step_fn


@click.command()
@click.option("--batch-size")
@click.option("--device", default="cuda:0")
def train(batch_size, device):
    device = torch.device(device)
    train_loader, val_loader = create_loaders("data/base")
    model = get_model(num_classes=512, checkpoint_weights="segformer_7data.pth").to(
        device
    )
    x = torch.rand(2, 3, 128, 128).to(device)
    breakpoint()
    pass


if __name__ == "__main__":
    train()

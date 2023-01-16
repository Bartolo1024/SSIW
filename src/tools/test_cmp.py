import os.path

import click
import numpy as np
import PIL.Image
import torch
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms

from src.tools.train import get_model, load_embeddings
from src.utils.loss import nxn_cos_sim
from src.utils.transforms_utils import get_imagenet_mean_std


class ImageLoader:
    def __init__(self, device: torch.device):
        mean, std = get_imagenet_mean_std()
        self.normalizer = transforms.Normalize(mean, std)
        self.to_tensor = transforms.ToTensor()
        self.device = device

    def __call__(self, img_path: str):
        img = PIL.Image.open(img_path).convert("RGB")
        x = self.to_tensor(img)
        x = self.normalizer(x).to(self.device)
        return img, x


def predict(x: torch.Tensor, model: nn.Module, embeddings: torch.Tensor):
    with torch.no_grad():
        pred, _, _ = model(x.unsqueeze(0))
    bs, channels, h, w = pred.shape
    output = pred.permute(0, 2, 3, 1).view(-1, channels)
    cos_sim = nxn_cos_sim(output, embeddings)
    _, labels = cos_sim.max(dim=-1)
    labels = labels.view(h, w)
    return labels.cpu().numpy()


def plot(img, labels: np.ndarray, target=None):
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(img)
    axs[0].set_title('Input')
    axs[0].axis('off')
    axs[1].imshow(labels)
    axs[1].axis('off')
    axs[1].set_title('Prediction')
    if target:
        axs[2].imshow(target)
        axs[2].axis('off')
        axs[2].set_title('Target')
    plt.savefig("pred.png")
    plt.show()


@click.command()
@click.argument("img_path")
@click.argument("checkpoint_path")
@click.option("--device", default="cuda:0")
@click.option("--labels-path", default="src/configs/labels.yaml")
def train(img_path, checkpoint_path, device, labels_path):
    device = torch.device(device)
    img_loader = ImageLoader(device)
    device = torch.device(device)
    embs = load_embeddings(labels_path)
    embs = embs.to(device)
    model = get_model(num_classes=512, checkpoint_weights=checkpoint_path).to(device).eval()
    img, x = img_loader(img_path)
    labels = predict(x, model, embs)
    tgt_path = img_path.replace('.jpg', '.png')
    if os.path.isfile(tgt_path):
        target = PIL.Image.open(tgt_path)
        plot(img, labels, target)
    else:
        plot(img, labels)


if __name__ == "__main__":
    train()

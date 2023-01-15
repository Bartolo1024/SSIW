import PIL.Image
import click
import numpy as np
import torch
from torch import nn
from  matplotlib import pyplot as plt
from torchvision import transforms
from src.tools.train import load_labels, load_embeddings, get_model
from src.utils.transforms_utils import get_imagenet_mean_std
from src.utils.loss import nxn_cos_sim


class ImageLoader:
    def __init__(self):
        mean, std = get_imagenet_mean_std()
        self.normalizer = transforms.Normalize(mean, std)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img_path: str):
        img = PIL.Image.open(img_path).convert("RGB")
        x = self.to_tensor(img)
        x = self.normalizer(x)
        return img, x


def predict(x: torch.Tensor, model: nn.Module, embeddings: torch.Tensor):
    pred, _, _ = model(x.unsqueeze(0))
    bs, channels, h, w = pred.shape
    output = pred.permute(0, 2, 3, 1).view(-1, channels)
    cos_sim = nxn_cos_sim(output, embeddings)
    labels = cos_sim.argmax(dim=-1).view(h, w)
    return labels.cpu().numpy()


def plot(labels: np.ndarray):
    plt.imshow(labels)
    plt.show()


@click.command()
@click.argument("img_path")
@click.option("--device", default="cuda:0")
@click.option("--labels-path", default="src/configs/labels.yaml")
def train(img_path, device, labels_path):
    img_loader = ImageLoader()
    device = torch.device(device)
    # labels = load_labels(labels_path)
    embs = load_embeddings(labels_path)
    embs = embs.to(device)
    model = get_model(num_classes=512, checkpoint_weights="out.pth").to(device)
    img, x = img_loader(img_path)
    labels = predict(x, model, embs)
    plot(labels)


if __name__ == "__main__":
    train()

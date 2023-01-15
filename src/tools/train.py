from typing import Any, Callable, Optional, Sequence, Tuple, Union
import yaml
import click
import torch
from ignite.engine import Engine
from torch import optim
from torch.utils.data import DataLoader, random_split

from src.utils.dataflow import CMPDataset
from src.utils.loss import HDLoss
from src.utils.segformer import get_configured_segformer
from src.utils.get_class_emb import create_embs_from_names


def load_labels(path):
    with open(path, 'r') as f:
        labels = yaml.load(f, yaml.BaseLoader)
    return labels


def load_embeddings(labels_path: str):
    labels = load_labels(labels_path)
    keys = labels.keys()
    descriptions = {k: v['description'] for k, v in labels.items()}
    embs = create_embs_from_names(keys, descriptions, device='cpu')
    return embs


def create_loaders(embeddigns: torch.Tensor, data_root: str, train_ratio: float = 0.8, batch_size: int = 4):
    dataset = CMPDataset(embeddigns, data_root)
    train_len = int(len(dataset) * train_ratio)
    train_dataset, valid_dataset = random_split(
        dataset, lengths=[train_len, len(dataset) - train_len]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader


def get_model(num_classes: int, checkpoint_weights: str, freeze: bool = True):
    model = get_configured_segformer(num_classes, None, load_imagenet_model=False)
    state_dict = torch.load(checkpoint_weights, map_location=torch.device("cpu"))
    state_dict = state_dict["state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    del state_dict["criterion.0.logit_scale"]
    model.load_state_dict(state_dict)
    return model


def create_step_fn(model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[Callable, torch.nn.Module],
    device: Optional[Union[str, torch.device]] = None,
    output_transform: Callable[[Any, Any, Any, torch.Tensor], Any] = lambda x, y, y_pred, loss: loss.item(),
    ):
    def update(_: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        optimizer.zero_grad()
        model.train()
        x, target, one_hot = batch
        x = x.to(device)
        target = target.to(device)
        one_hot = one_hot.to(device)
        y_pred, _, _ = model(x)
        breakpoint()
        loss = loss_fn(y_pred, target, one_hot)
        loss.backward()
        optimizer.step()
        return output_transform(x, target, y_pred, loss)
    return update


@click.command()
@click.option("--batch-size", default=2)
@click.option("--device", default="cuda:0")
@click.option("--labels-path", default="src/configs/labels.yaml")
def train(batch_size, device, labels_path):
    device = torch.device(device)
    embs = load_embeddings(labels_path)
    train_loader, val_loader = create_loaders(embs, "data/base/base", batch_size=batch_size)
    model = get_model(num_classes=512, checkpoint_weights="segformer_7data.pth").to(device)
    loss_fn = HDLoss(embs).to(device)
    optimizer = optim.Adam(lr=0.0005, params=model.parameters())
    step_fn = create_step_fn(model, optimizer, loss_fn, device)

    trainer = Engine(step_fn)
    trainer.run(train_loader)


if __name__ == "__main__":
    train()

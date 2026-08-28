"""pytorch-example: A Flower / PyTorch app."""

import json
from datetime import datetime
from pathlib import Path

from matplotlib import image
import torch
import torch.nn as nn
from flwr.common.typing import UserConfig
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
    RandomHorizontalFlip,
    RandomCrop,
)
from torchvision.models import resnet18
import numpy as np
from collections import deque
from typing import Deque, Dict, List, Tuple, Optional, Iterable
from flwr.common import Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters
from dataclasses import dataclass, field
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.serverapp.strategy import FedAvg
import torch.nn.functional as F




class NetCIFAR(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz') for CIFAR-10."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def _replace_bn_with_groupnorm(module: nn.Module, num_groups: int = 32) -> None:
    """Recursively replace every BatchNorm2d in `module` with GroupNorm(num_groups, C).

    GroupNorm avoids BatchNorm running-stat buffers being FedAvg-averaged across
    non-IID / Byzantine clients. All ResNet-18 channel counts (64/128/256/512)
    are divisible by 32, so GroupNorm(32, C) is always valid.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(num_groups, child.num_features))
        else:
            _replace_bn_with_groupnorm(child, num_groups)


def build_resnet18_cifar(num_classes: int = 10, num_groups: int = 32) -> nn.Module:
    """CIFAR-adapted ResNet-18 with GroupNorm (defined here so every app shares it).

    Adapts the ImageNet stem for 32x32 inputs:
      - conv1: 7x7 stride-2  ->  3x3 stride-1 pad-1  (keep spatial resolution)
      - maxpool: 3x3 stride-2  ->  Identity          (no early 4x downsample)
    and replaces all BatchNorm2d with GroupNorm. The stock stem collapses 32x32
    to ~4x4 and caps CIFAR ~75-85%; this adapted stem reaches ~93-95% centrally.
    """
    net = resnet18(weights=None, num_classes=num_classes)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    _replace_bn_with_groupnorm(net, num_groups=num_groups)
    return net


# MNIST and Fashion-MNIST configuration
FM_NORMALIZATION = ((0.1307,), (0.3081,))
EVAL_TRANSFORMS = Compose([ToTensor(), Normalize(*FM_NORMALIZATION)])
TRAIN_TRANSFORMS = Compose(
    [
        RandomCrop(28, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(*FM_NORMALIZATION),
    ]
)

DATASET_REGISTRY = {
    "mnist": {
        "hf_path": "ylecun/mnist",
        "image_key": "image",
        "channels": 1,
        "image_size": 28,
        "normalization": ((0.1307,), (0.3081,)),
    },
    "fashion_mnist": {
        "hf_path": "zalando-datasets/fashion_mnist",
        "image_key": "image",
        "channels": 1,
        "image_size": 28,
        "normalization": ((0.1307,), (0.3081,)),
    },
    "cifar10": {
        "hf_path": "uoft-cs/cifar10",
        "image_key": "img",
        "channels": 3,
        "image_size": 32,
        "normalization": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    },
}


def get_net(dataset_name: str, cifar_model: str = "netcifar") -> nn.Module:
    """Return a fresh model instance matching the dataset's input shape.

    For cifar10, `cifar_model` selects the architecture:
      - "netcifar" (default): the small 60-min-blitz CNN (``NetCIFAR``)
      - "resnet18": CIFAR-adapted ResNet-18 with GroupNorm (``build_resnet18_cifar``)
    The default preserves existing behaviour; the server chooses via run config.
    """
    if dataset_name == "cifar10":
        if cifar_model == "resnet18":
            return build_resnet18_cifar()
        return NetCIFAR()
    return Net()


def build_transforms(dataset_name: str):
    """Build (train_transform, eval_transform) for the given dataset."""
    cfg = DATASET_REGISTRY[dataset_name]
    normalization = cfg["normalization"]
    image_size = cfg["image_size"]

    eval_transform = Compose([ToTensor(), Normalize(*normalization)])
    train_transform = Compose(
        [
            RandomCrop(image_size, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(*normalization),
        ]
    )
    return train_transform, eval_transform

class Net(nn.Module):
    """Model CNN adapted for Fashion-MNIST or MNIST."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"] # for MNIST and Fashion-MNIST
            #images = batch["img"] # for CIFAR-10
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            #images = batch["img"].to(device) # for CIFAR-10
            images = batch["image"].to(device) # for MNIST and Fashion-MNIST
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def apply_train_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    #batch["img"] = [TRAIN_TRANSFORMS(img) for img in batch["img"]]
    batch["image"] = [TRAIN_TRANSFORMS(img) for img in batch["image"]]
    return batch


def apply_eval_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    #batch["img"] = [EVAL_TRANSFORMS(img) for img in batch["img"]]
    batch["image"] = [EVAL_TRANSFORMS(img) for img in batch["image"]]
    return batch


def make_train_transform_fn(transform):
    """Build a `.with_transform()`-compatible fn for a given train transform."""
    def apply(batch):
        batch["image"] = [transform(img) for img in batch["image"]]
        return batch
    return apply


def make_eval_transform_fn(transform):
    """Build a `.with_transform()`-compatible fn for a given eval transform."""
    def apply(batch):
        batch["image"] = [transform(img) for img in batch["image"]]
        return batch
    return apply


fds: FederatedDataset | None = None
_fds_cache: Dict[str, FederatedDataset] = {}

def load_data(partition_id: int, num_partitions: int, dataset_name: str = "fashion_mnist"):
    """Load a client partition for the given dataset."""
    cfg = DATASET_REGISTRY[dataset_name]

    if dataset_name not in _fds_cache:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=0.5,
            seed=42,
        )
        _fds_cache[dataset_name] = FederatedDataset(
            dataset=cfg["hf_path"],
            partitioners={"train": partitioner},
        )
    fds_for_dataset = _fds_cache[dataset_name]

    partition = fds_for_dataset.load_partition(partition_id)
    if cfg["image_key"] != "image":
        partition = partition.rename_column(cfg["image_key"], "image")

    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    train_transform, eval_transform = build_transforms(dataset_name)
    train_partition = partition_train_test["train"].with_transform(
        make_train_transform_fn(train_transform)
    )
    test_partition = partition_train_test["test"].with_transform(
        make_eval_transform_fn(eval_transform)
    )
    trainloader = DataLoader(train_partition, batch_size=32, shuffle=True)
    testloader = DataLoader(test_partition, batch_size=32)
    return trainloader, testloader


def create_run_dir(config: UserConfig) -> tuple[Path, str]:
    """Create a directory where to save results from this run."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    # Save path is based on the current directory
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=False)

    # Save run config as json
    with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
        json.dump(config, fp)

    return save_path, run_dir


def build_fltrust_rootset_loader(n_root: int = 500, batch_size: int = 32, seed: int = 42):
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(
            num_partitions=1,
            partition_by="label",
            alpha=0.5,
            seed=42,
        )
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            #dataset="zalando-datasets/fashion_mnist",
            #ataset="ylecun/mnist",
            partitioners={"train": partitioner},
        )
    test_split = fds.load_split("test").with_transform(apply_eval_transforms)
    idx = np.random.default_rng(seed).choice(len(test_split), size=min(n_root, len(test_split)), replace=False)
    rootset = test_split.select(idx)
    return DataLoader(rootset, batch_size=batch_size, shuffle=True)







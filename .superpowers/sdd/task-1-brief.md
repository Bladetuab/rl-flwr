### Task 1: Dataset/model registry in `task.py`

**Files:**
- Modify: `rl_flwr/task.py:33-55` (commented-out CIFAR block → active `NetCIFAR`)
- Modify: `rl_flwr/task.py:57-67` (add `DATASET_REGISTRY`, `get_net`, `build_transforms` after the existing `FM_NORMALIZATION`/`EVAL_TRANSFORMS`/`TRAIN_TRANSFORMS` block — those stay, untouched, for `build_fltrust_rootset_loader`)
- Modify: `rl_flwr/task.py:128-171` (add transform-closure factories; rewrite `load_data`)

**Interfaces:**
- Produces: `get_net(dataset_name: str) -> nn.Module`, `build_transforms(dataset_name: str) -> tuple[Compose, Compose]` (train, eval), `make_train_transform_fn(transform) -> Callable[[dict], dict]`, `make_eval_transform_fn(transform) -> Callable[[dict], dict]`, `DATASET_REGISTRY: dict`, `NetCIFAR` (class), `load_data(partition_id: int, num_partitions: int, dataset_name: str = "fashion_mnist") -> tuple[DataLoader, DataLoader]`.
- Consumes: nothing from other tasks (this is the foundation task).

- [ ] **Step 1: Replace the commented-out CIFAR block with an active `NetCIFAR` class**

Find this block at `rl_flwr/task.py:33-55`:

```python
# CIFAR-10 configuration
# TRAIN_TRANSFORMS = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# EVAL_TRANSFORMS = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# class Net(nn.Module):
#     """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)

# #MNIST and Fashion-MNIST configuration
```

Replace it with:

```python
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


# MNIST and Fashion-MNIST configuration
```

Note: the flatten dimension `16 * 5 * 5` is correct for 32×32 CIFAR-10 input: `conv1` (kernel 5, no padding) takes 32→28, `pool` takes 28→14, `conv2` takes 14→10, `pool` takes 10→5, so `16 channels × 5 × 5 = 400` matches `fc1 = nn.Linear(16 * 5 * 5, 120)`. This is verified in Step 6's smoke test (a shape mismatch here is a hard crash, not a silent bug).

- [ ] **Step 2: Add `DATASET_REGISTRY`, `get_net`, `build_transforms` after the existing transform constants**

The existing block at `rl_flwr/task.py:57-67` (`FM_NORMALIZATION`, `EVAL_TRANSFORMS`, `TRAIN_TRANSFORMS`) stays exactly as-is — it's still used by `build_fltrust_rootset_loader` (out of scope). Immediately after that block (i.e. after line 67, before `class Net(nn.Module):`), insert:

```python
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


def get_net(dataset_name: str) -> nn.Module:
    """Return a fresh model instance matching the dataset's input shape."""
    if dataset_name == "cifar10":
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
```

`get_net` references `Net`, which is defined right after this block — that's fine, Python resolves the name at call time, not at module-load time, so the forward reference is safe.

- [ ] **Step 3: Run a quick shape sanity check (catches a `NetCIFAR` flatten-dim mistake in seconds instead of mid-sweep)**

Run:
```bash
cd /home/tuan/get-local-model-bt-id/rl-flwr && python -c "
import torch
from rl_flwr.task import get_net, DATASET_REGISTRY

for name, cfg in DATASET_REGISTRY.items():
    net = get_net(name)
    x = torch.randn(2, cfg['channels'], cfg['image_size'], cfg['image_size'])
    out = net(x)
    print(name, '->', tuple(out.shape))
"
```
Expected output (exact shapes, order may vary by dict iteration but Python 3.7+ dicts preserve insertion order so this should match):
```
mnist -> (2, 10)
fashion_mnist -> (2, 10)
cifar10 -> (2, 10)
```
If `cifar10` raises a `RuntimeError` about a shape mismatch in `fc1`, the flatten-dim note in Step 1 was violated — re-check the `Conv2d`/`MaxPool2d` parameters were copied exactly.

- [ ] **Step 4: Add transform-closure factories and rewrite `load_data`**

Find the existing `apply_train_transforms`/`apply_eval_transforms` functions at `rl_flwr/task.py:128-139` — **leave them exactly as they are** (still used by `build_fltrust_rootset_loader`, out of scope). Immediately after them (after line 139, before `fds: FederatedDataset | None = None`), add:

```python
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
```

Then find the existing `fds`/`load_data` block at `rl_flwr/task.py:142-171`:

```python
fds: FederatedDataset | None = None

def load_data(partition_id: int, num_partitions: int):
    """Load partition FashionMNIST data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=0.5,
            seed=42,
        )
        fds = FederatedDataset(
            #dataset="uoft-cs/cifar10",
            # dataset="ylecun/mnist",
            dataset="zalando-datasets/fashion_mnist",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    train_partition = partition_train_test["train"].with_transform(
        apply_train_transforms
    )
    test_partition = partition_train_test["test"].with_transform(apply_eval_transforms)
    trainloader = DataLoader(train_partition, batch_size=32, shuffle=True)
    testloader = DataLoader(test_partition, batch_size=32)
    return trainloader, testloader
```

Replace it with (note: `fds` global stays, untouched, for `build_fltrust_rootset_loader`; this adds a **new**, separate `_fds_cache` dict for the sweep path):

```python
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
```

`Dict` is already imported at the top of `task.py` (`from typing import Deque, Dict, List, Tuple, Optional, Iterable`) — no new import needed.

- [ ] **Step 5: Verify `task.py` still imports cleanly**

Run:
```bash
cd /home/tuan/get-local-model-bt-id/rl-flwr && python -c "import rl_flwr.task; print('ok')"
```
Expected output: `ok` (no `SyntaxError`/`ImportError`/`NameError`).

---


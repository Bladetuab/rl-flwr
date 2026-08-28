### Task 2: Wire dataset selection into `client_app.py`

**Files:**
- Modify: `rl_flwr/client_app.py:1-9` (imports)
- Modify: `rl_flwr/client_app.py:15-63` (`train()` handler)
- Modify: `rl_flwr/client_app.py:66-100` (`evaluate()` handler)

**Interfaces:**
- Consumes: `get_net(dataset_name: str) -> nn.Module` and `load_data(partition_id, num_partitions, dataset_name=...)` from Task 1.
- Produces: nothing new for downstream tasks — this task only makes the existing client handlers dataset-aware.

- [ ] **Step 1: Update the import**

Find:
```python
from rl_flwr.task import Net, load_data
from rl_flwr.task import test as test_fn
from rl_flwr.task import train as train_fn
```
Replace with:
```python
from rl_flwr.task import get_net, load_data
from rl_flwr.task import test as test_fn
from rl_flwr.task import train as train_fn
```

- [ ] **Step 2: Make `train()` dataset-aware**

Find (`rl_flwr/client_app.py:15-28`):
```python
@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
```
Replace with:
```python
@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    dataset_name = msg.content["config"].get("dataset", "fashion_mnist")

    # Load the model and initialize it with the received weights
    model = get_net(dataset_name)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions, dataset_name=dataset_name)
```

- [ ] **Step 3: Make `evaluate()` dataset-aware**

Find (`rl_flwr/client_app.py:66-79`):
```python
@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, valloader = load_data(partition_id, num_partitions)
```
Replace with:
```python
@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    dataset_name = msg.content["config"].get("dataset", "fashion_mnist")

    # Load the model and initialize it with the received weights
    model = get_net(dataset_name)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, valloader = load_data(partition_id, num_partitions, dataset_name=dataset_name)
```

`.get("dataset", "fashion_mnist")` is required (not `["dataset"]`) because `server_app.py` (out of scope, untouched) never sends a `dataset` config key — this default keeps its existing single-run behavior working unchanged.

- [ ] **Step 4: Verify `client_app.py` still imports cleanly**

Run:
```bash
cd /home/tuan/get-local-model-bt-id/rl-flwr && python -c "import rl_flwr.client_app; print('ok')"
```
Expected output: `ok`.

Full end-to-end verification of this task happens in Task 4's smoke test (it needs a running Flower simulation, which this isolated import check can't exercise).

---


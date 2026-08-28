"""rl-flwr: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from rl_flwr.task import get_net, load_data
from rl_flwr.task import test as test_fn
from rl_flwr.task import train as train_fn

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    dataset_name = msg.content["config"].get("dataset", "fashion_mnist")
    cifar_model = msg.content["config"].get("cifar_model", "netcifar")

    # Load the model and initialize it with the received weights
    model = get_net(dataset_name, cifar_model=cifar_model)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions, dataset_name=dataset_name)

    # ---------- NEW: validation BEFORE local training ----------
    val_loss_before, _ = test_fn(
        model,
        valloader,
        device,
    )

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # ---------- NEW: validation AFTER local training ----------
    val_loss_after, _ = test_fn(
        model,
        valloader,
        device,
    )
    # ---------- NEW: client-side validation delta ----------
    val_loss_delta = float(val_loss_after - val_loss_before)

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    dataset_name = msg.content["config"].get("dataset", "fashion_mnist")
    cifar_model = msg.content["config"].get("cifar_model", "netcifar")

    # Load the model and initialize it with the received weights
    model = get_net(dataset_name, cifar_model=cifar_model)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, valloader = load_data(partition_id, num_partitions, dataset_name=dataset_name)



    # ---------- validation BEFORE local training ----------

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)

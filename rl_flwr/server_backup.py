"""rl-flwr: A Flower / PyTorch app."""

from matplotlib.pyplot import grid
import torch
from datasets import load_dataset
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from torch.utils.data import DataLoader
from flwr.serverapp.strategy import FedAvg, Krum



from get_local_model_bt_id import strategy
from rl_flwr.task import Net, apply_eval_transforms, create_run_dir, test, build_fltrust_rootset_loader
# from rl_flwr.strategy import CustomFedAvg, CustomKrum, FLTrustCustomFedAvg, FedAvgWithLBFGSHVP, HVPConfig, CustomFedAvg_adaptive
from rl_flwr.strategy import CustomFedAvg_adaptive, CustomFedAvg, CustomFedAvg_attack, CustomKrum_attack, FedAvgWithLBFGSHVP, HVPConfig, FLTrustCustomFedAvg
# Create ServerApp
from rl_flwr.CustomFedAvg_adaptive_new import CustomFedAvg_adaptive_new

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    # Read run config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_train = context.run_config["fraction-train"]
    fraction_eval = context.run_config["fraction-evaluate"]
    device = context.run_config["server-device"]
    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    #Initialize FedAvg strategy NO ATTACK
    # attack_mode = context.run_config.get("attack-mode", "no_attack")
    # strategy = CustomFedAvg(
    #     fraction_train=fraction_train, fraction_evaluate=fraction_eval
    # )
    
    # Initialize FedAvg strategy WITH ATTACK
    attack_mode = "fang"
    strategy = CustomFedAvg_attack(
        num_malicious=40, warmup_rounds=3, 
        attack_mode=attack_mode,
        fraction_train=fraction_train, 
        fraction_evaluate=fraction_eval
    )


    # attack_mode = "gaussian"
    # strategy = CustomFedAvg_adaptive_new(
    #     num_malicious=40, warmup_rounds=5, attack_mode=attack_mode,
    #     fraction_train=fraction_train, fraction_evaluate=fraction_eval,
    #     stat_decay=0.2, clone_cos_threshold=0.95, anom_decay=0.8, tau_s_quantile=0.8
    # )

    # WITH BEST DEFENSE
    # attack_mode = "fang"
    # anom_decay = 0.6
    # tau_s_quantile = 0.8
    # strategy = CustomFedAvg_adaptive(
    #     num_malicious=40, warmup_rounds=5, anom_decay=anom_decay, attack_mode=attack_mode,
    #     tau_s_quantile=tau_s_quantile,
    #     fraction_train=fraction_train, fraction_evaluate=fraction_eval
    # )

    # WITH Krum defense
    # attack_mode = "fang"
    # strategy = CustomKrum_attack (
    #     num_malicious_nodes=40, 
    #     num_nodes_to_select=60,
    #     warmup_rounds=5,
    #     fraction_train=fraction_train, 
    #     fraction_evaluate=fraction_eval,
    #     attack_mode=attack_mode
    # )




    # attack_mode = "minmax" #context.run_config.get("attack-mode", "no_attack")
    # root_loader = build_fltrust_rootset_loader(n_root=200, batch_size=32, seed=42)
    # strategy = FLTrustCustomFedAvg(
    #     num_malicious=40,
    #     warmup_rounds=5,
    #     attack_mode=attack_mode,
    #     server_lr=1.0,
    #     model_fn=lambda: Net(),     # model giống client
    #     root_loader=root_loader,
    #     loss_fn=torch.nn.CrossEntropyLoss(),
    #     optimizer_fn=lambda m: torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9),
    #     layer_filter=None,          # hoặc : lambda n: ("layer4" in n) or ("fc" in n)

    # )

    # attack_mode = "lie"
    # strategy = FedAvgWithLBFGSHVP(
    #     # your existing FedAvg/base args here
    #     fraction_train=1.0, fraction_evaluate=1.0,

    #     # attack simulation knobs (only for experiments)
    #     num_malicious=40,
    #     warmup_rounds_attack=5,
    #     attack_mode=attack_mode,

    #     # HVP / L-BFGS config
    #     hvp_cfg=HVPConfig(
    #         array_key="arrays",
    #         eta_server=1e-3,
    #         max_hist=5,
    #         warmup_rounds_hvp=5,
    #     ),

    #     # new FLDetector-style knobs
    #     suspicion_window=5,      # N
    #     detect_from_round=5,     # start detection after round 5
    #     gap_B=20,                 # Gap statistics samples
    #     gap_K=10,                 # max clusters
    #     remove_detected_clients=True,
    #     restart_on_detection=False,   # set True if you want FLDetector-style restart
    #     max_restarts=1,
    #     random_state=0,
    # )

    # strategy = Krum(num_malicious_nodes  = 40, fraction_train=fraction_train, fraction_evaluate=fraction_eval)
    # Define directory for results and save config
    save_path, run_dir = create_run_dir(config=context.run_config)
    named_path = save_path.parent / f"{save_path.name}_{attack_mode}"
    save_path.rename(named_path)
    save_path = named_path

    strategy.set_save_path_and_run_dir(save_path, run_dir)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        #train_config=ConfigRecord({"lr": 0.01}), #for CIFAR-10
        train_config=ConfigRecord({"lr": 0.001}), #for MNIST and Fashion-MNIST
        num_rounds=num_rounds,
        evaluate_fn=get_global_evaluate_fn(device=device),
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


def get_global_evaluate_fn(device: str):
    """Return an evaluation function for server-side evaluation."""

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Evaluate model on central data."""

        # This is the exact same dataset as the one downloaded by the clients via
        # FlowerDatasets. However, we don't use FlowerDatasets for the server since
        # partitioning is not needed.
        # We make use of the "test" split only
        # global_test_set = load_dataset("uoft-cs/cifar10")["test"]
        global_test_set = load_dataset("zalando-datasets/fashion_mnist")["test"]
        # global_test_set = load_dataset("ylecun/mnist")["test"]
        testloader = DataLoader(
            global_test_set.with_transform(apply_eval_transforms),
            batch_size=32,
        )

        net = Net()
        net.load_state_dict(arrays.to_torch_state_dict())
        net.to(device)
        loss, accuracy = test(net, testloader, device=device)
        return MetricRecord({"accuracy": accuracy, "loss": loss})

    return global_evaluate

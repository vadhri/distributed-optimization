import wandb
import torch
import os
from flwr.common import Context, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from distributed_minst.task import Net, get_weights, set_weights

class WandbFedAvg(FedAvg):
    def __init__(self, num_rounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = Net()
        self.num_rounds = num_rounds

    def aggregate_evaluate(self, rnd, results, failures):
        aggregated_result = super().aggregate_evaluate(rnd, results, failures)
        if aggregated_result is not None:
            loss, metrics = aggregated_result
            wandb.log({"server_loss": loss, "server_accuracy": metrics.get("accuracy", 0.0), "round": rnd})
        return aggregated_result

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        if aggregated_parameters is not None:
            set_weights(self.model, parameters_to_ndarrays(aggregated_parameters))
            os.makedirs("checkpoints", exist_ok=True)
            filename = f"checkpoints/model_round_{rnd}.pt"
            torch.save(self.model.state_dict(), filename)
            print(f"[Server] Saved aggregated model after round {rnd} to {filename}")

            if rnd == self.num_rounds:
                final_filename = "checkpoints/final_model.pt"
                torch.save(self.model.state_dict(), final_filename)
                print(f"[Server] Saved final model to {final_filename}")
                artifact = wandb.Artifact("final-model", type="model")
                artifact.add_file(final_filename)
                wandb.log_artifact(artifact)

            artifact = wandb.Artifact(f"model-round-{rnd}", type="model")
            artifact.add_file(filename)
            wandb.log_artifact(artifact)
        return aggregated_parameters, aggregated_metrics

def server_fn(context: Context):
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    wandb.init(
        project="distributed-minst",
        name="flower-server",
        config={
            "num_server_rounds": num_rounds,
            "fraction_fit": fraction_fit,
            "strategy": "FedAvg",
            "min_available_clients": 20,
            "fraction_evaluate": 1.0,
        },
    )

    initial_weights = get_weights(Net())
    parameters = ndarrays_to_parameters(initial_weights)

    strategy = WandbFedAvg(
        num_rounds=num_rounds,
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=10,
        initial_parameters=parameters,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)
"""simple-linear-regression: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from simple_linear_regression.task import Net, get_weights
import logging
from logging import INFO, DEBUG
logging.basicConfig(filename='myapp.log', level=logging.DEBUG)
logger = logging.getLogger("flwr-serverapp")

logger.info(f"[server_fn] Starting Flower server!!!")
import wandb 

from flwr.server.strategy import FedAvg

class WandbFedAvg(FedAvg):
    def evaluate(self, server_round, parameters, client_metrics=None, config=None):
        # Handle initial evaluation (no client metrics available)
        if client_metrics is None:
            print(f"[WandbFedAvg] Initial evaluation for round {server_round}")
            return super().evaluate(server_round, parameters)

        # Compute aggregated val_loss from client_metrics
        total_examples = sum(num_examples for num_examples, _ in client_metrics)
        weighted_loss = sum(
            num_examples * metrics["val_loss"]
            for num_examples, metrics in client_metrics
        )
        aggregated_loss = weighted_loss / total_examples if total_examples > 0 else None

        # Log to wandb
        wandb.log({"round_val_loss": aggregated_loss, "round": server_round})

        return aggregated_loss, {"val_loss": aggregated_loss}


def server_fn(context: Context):
    logger.info(f"[server_fn] Starting Flower server with config: {context.run_config}")
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize wandb
    wandb.init(
        project="fl-simple-linear-regression",
        name="server",
        config={"strategy": "FedAvg"},
        reinit=True,
    )

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Use custom strategy with wandb logging
    strategy = WandbFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=0.5,
        min_available_clients=4,
        initial_parameters=parameters,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)
# Create ServerApp
app = ServerApp(server_fn=server_fn)

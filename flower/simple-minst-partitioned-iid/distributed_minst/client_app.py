"""distributed-minst: A Flower / PyTorch app with wandb logging."""

import wandb
import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from distributed_minst.task import Net, get_weights, load_data, set_weights, test, train
from torchvision.transforms import Compose, Normalize, ToTensor

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, partition_id):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.partition_id = partition_id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.net.to(self.device)

        wandb.init(
            project="distributed-minst",
            name=f"client-{partition_id}",
            config={
                "partition_id": partition_id,
                "local_epochs": local_epochs,
                "model": "FL_MNIST",
            },
        )

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train_loss, train_accuracy = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )

        wandb.log({"train_loss": train_loss, "train_accuracy": train_accuracy})

        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss, "train_accuracy": train_accuracy},
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)

        wandb.log({
            "eval_loss": loss,
            "eval_accuracy": accuracy,
        })

        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs, partition_id).to_client()

# Flower ClientApp
app = ClientApp(client_fn)

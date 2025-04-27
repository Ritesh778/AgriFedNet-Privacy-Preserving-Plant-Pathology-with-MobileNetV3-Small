

"""Federated Learning Server with Custom FedAvg Strategy.

This script implements a Flower server using a custom FedAvg strategy to
aggregate client updates, evaluate global and local metrics, and save the best
model based on accuracy. It supports MobileNetV3-Small and runs for a specified
number of rounds.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional
import warnings

# Suppress all warnings immediately after standard imports
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torchvision.models as models
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)
import flwr as fl
from flwr.common import NDArrays, Scalar, Parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from prettytable import PrettyTable
import pickle


class CustomCNN(nn.Module):
    """Custom CNN model based on MobileNetV3-Small."""

    def __init__(self, num_classes: int):
        super(CustomCNN, self).__init__()
        self.backbone = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.features[6:].parameters():
            param.requires_grad = True
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def get_model(model_name: str, num_classes: int) -> nn.Module:
    """Initialize and return the specified model.

    Args:
        model_name (str): Name of the model (e.g., 'mobilenet_v3_small').
        num_classes (int): Number of output classes.

    Returns:
        nn.Module: Initialized model.

    Raises:
        ValueError: If the model name is not supported.
    """
    if model_name.lower() == "mobilenet_v3_small":
        return CustomCNN(num_classes=num_classes)
    raise ValueError(f"Model {model_name} not supported.")


class SaveFedAvg(FedAvg):
    """Custom FedAvg strategy that saves models and tracks metrics."""

    def __init__(
        self,
        num_classes: int,
        model_name: str,
        num_rounds: int,
        *args,
        **kwargs
    ):
        """Initialize the custom FedAvg strategy.

        Args:
            num_classes (int): Number of classes in the dataset.
            model_name (str): Name of the model architecture.
            num_rounds (int): Total number of training rounds.
            *args: Additional arguments for FedAvg.
            **kwargs: Additional keyword arguments for FedAvg.
        """
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.model_name = model_name
        self.num_rounds = num_rounds
        self.global_metrics_history = {
            "fedavg": {
                "loss": [], "accuracy": [], "precision": [],
                "recall": [], "f1": []
            },
            "local": {
                "loss": [], "accuracy": [], "precision": [],
                "recall": [], "f1": []
            },
            "diff": {
                "loss": [], "accuracy": [], "precision": [],
                "recall": [], "f1": []
            }
        }
        self.best_accuracy = 0.0
        self.best_model_params = None
        self.current_params = None  # Store aggregated parameters

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[Tuple[ClientProxy, fl.common.FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate client weights after a training round.

        Args:
            server_round (int): Current server round number.
            results (List[Tuple[ClientProxy, FitRes]]): Successful client updates.
            failures (List[Tuple[ClientProxy, FitRes]]): Failed client updates.

        Returns:
            Tuple[Optional[Parameters], Dict[str, Scalar]]: Aggregated parameters
                and metrics.
        """
        aggregated_params, metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if aggregated_params is not None:
            self.current_params = aggregated_params  # Store the aggregated params
            with open(f"server_model_round_{server_round}.pkl", "wb") as f:
                pickle.dump(aggregated_params, f)
        return aggregated_params, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate client evaluation results.

        Args:
            server_round (int): Current server round number.
            results (List[Tuple[ClientProxy, EvaluateRes]]): Successful client
                evaluations.
            failures (List[Tuple[ClientProxy, EvaluateRes]]): Failed client
                evaluations.

        Returns:
            Tuple[Optional[float], Dict[str, Scalar]]: Aggregated loss and metrics.
        """
        if not results:
            return None, {}

        valid_results = [r for r in results if r[1].metrics is not None]
        if not valid_results:
            return None, {}

        total_examples = sum(r[1].num_examples for r in valid_results)
        if total_examples == 0:
            return None, {}

        fedavg_loss = sum(
            r[1].loss * r[1].num_examples for r in valid_results
        ) / total_examples
        fedavg_metrics = {"loss": fedavg_loss}
        for key in ["accuracy", "precision", "recall", "f1"]:
            fedavg_metrics[key] = sum(
                r[1].metrics.get(key, 0) * r[1].num_examples
                for r in valid_results
            ) / total_examples

        local_metrics = {
            "loss": 0, "accuracy": 0, "precision": 0, "recall": 0, "f1": 0
        }
        for key in local_metrics:
            local_metrics[key] = sum(
                r[1].metrics.get(f"local_{key}", 0) * r[1].num_examples
                for r in valid_results
            ) / total_examples

        diff_metrics = {
            key: fedavg_metrics[key] - local_metrics[key]
            for key in local_metrics
        }

        for key in fedavg_metrics:
            self.global_metrics_history["fedavg"][key].append(fedavg_metrics[key])
            self.global_metrics_history["local"][key].append(local_metrics[key])
            self.global_metrics_history["diff"][key].append(diff_metrics[key])

        current_accuracy = fedavg_metrics["accuracy"]
        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy
            self.best_model_params = self.get_parameters({})

        print(
            f"\nServer Round {server_round} - FedAvg: "
            f"Loss: {fedavg_metrics['loss']:.4f}, "
            f"Acc: {fedavg_metrics['accuracy']:.4f}, "
            f"Prec: {fedavg_metrics['precision']:.4f}, "
            f"Rec: {fedavg_metrics['recall']:.4f}, "
            f"F1: {fedavg_metrics['f1']:.4f}"
        )
        print(
            f"Server Round {server_round} - Local: "
            f"Loss: {local_metrics['loss']:.4f}, "
            f"Acc: {local_metrics['accuracy']:.4f}, "
            f"Prec: {local_metrics['precision']:.4f}, "
            f"Rec: {local_metrics['recall']:.4f}, "
            f"F1: {local_metrics['f1']:.4f}"
        )
        print(
            f"Server Round {server_round} - Diff: "
            f"Loss: {diff_metrics['loss']:.4f}, "
            f"Acc: {diff_metrics['accuracy']:.4f}, "
            f"Prec: {diff_metrics['precision']:.4f}, "
            f"Rec: {diff_metrics['recall']:.4f}, "
            f"F1: {diff_metrics['f1']:.4f}"
        )

        if server_round == self.num_rounds:
            self.print_final_table()

        return fedavg_loss, fedavg_metrics

    def get_parameters(self, config: Dict) -> Optional[Parameters]:
        """Return the current aggregated parameters.

        Args:
            config (Dict): Configuration dictionary (unused here).

        Returns:
            Optional[Parameters]: Current aggregated parameters or None if not set.
        """
        return self.current_params

    def print_final_table(self) -> None:
        """Print a formatted table comparing final FedAvg and Local metrics."""
        table = PrettyTable()
        table.field_names = [
            "Metric", "FedAvg (Round 3)", "Local", "Difference (FedAvg - Local)"
        ]
        for metric in ["loss", "accuracy", "precision", "recall", "f1"]:
            table.add_row([
                metric.capitalize(),
                f"{self.global_metrics_history['fedavg'][metric][-1]:.4f}",
                f"{self.global_metrics_history['local'][metric][-1]:.4f}",
                f"{self.global_metrics_history['diff'][metric][-1]:.4f}"
            ])
        print("\nFinal Metrics Comparison Table:")
        print(table)

        if self.best_model_params:
            with open("best_server_model.pkl", "wb") as f:
                pickle.dump(self.best_model_params, f)
            print(
                f"\nBest model (accuracy: {self.best_accuracy:.4f}) "
                f"saved as 'best_server_model.pkl'"
            )


def run_federated_learning(
    model_name: str,
    num_classes: int,
    num_rounds: int = 3
) -> None:
    """Run the federated learning server.

    Args:
        model_name (str): Name of the model to use.
        num_classes (int): Number of classes in the dataset.
        num_rounds (int): Number of federated learning rounds (default: 3).
    """
    model = get_model(model_name, num_classes)
    initial_params = [
        val.detach().cpu().numpy() for val in model.parameters()
        if val.requires_grad
    ]

    strategy = SaveFedAvg(
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        num_classes=num_classes,
        model_name=model_name,
        num_rounds=num_rounds,
        initial_parameters=fl.common.ndarrays_to_parameters(initial_params),
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    MODEL_NAME = "mobilenet_v3_small"
    NUM_CLASSES = 15  # Adjust based on PlantVillage dataset
    NUM_ROUNDS = 3
    run_federated_learning(MODEL_NAME, NUM_CLASSES, NUM_ROUNDS)
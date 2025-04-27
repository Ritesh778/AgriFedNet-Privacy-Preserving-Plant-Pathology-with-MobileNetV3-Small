

"""Federated Learning Client for PlantVillage Dataset.

This script implements a Flower client using MobileNetV3-Small for federated
learning on the PlantVillage dataset. It supports local training, evaluation,
and metric plotting, with data split across clients (80%, 15%, 5%).
"""

import sys
import os
import numpy as np
from typing import List, Tuple, Dict, Optional
import warnings

# Suppress all warnings immediately after standard imports
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)
import flwr as fl
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from torch.amp import GradScaler, autocast


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


class CustomImageDataset(Dataset):
    """Custom dataset for loading PlantVillage images."""

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = sorted(
            [d for d in os.listdir(root_dir)
             if os.path.isdir(os.path.join(root_dir, d))]
        )
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(self.class_names)
        }

        for cls_name in self.class_names:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class FlowerClient(fl.client.NumPyClient):
    """Flower client implementation for federated learning."""

    def __init__(
        self,
        client_id: int,
        train_loader: Optional[DataLoader],
        global_test_loader: DataLoader,
        num_classes: int,
        class_names: List[str]
    ):
        self.client_id = client_id
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = CustomCNN(num_classes=num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam([
            {"params": self.model.backbone.features[6:].parameters(),
             "lr": 1e-4},
            {"params": self.model.backbone.classifier.parameters(),
             "lr": 1e-3}
        ])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=1
        )
        self.scaler = GradScaler(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.train_loader = train_loader
        self.global_test_loader = global_test_loader
        self.num_classes = num_classes
        self.class_names = class_names
        self.local_metrics = {
            "loss": [], "accuracy": [], "precision": [], "recall": [], "f1": []
        }
        self.local_model_metrics = self.train_local_model()

    def train_local_model(self) -> Dict[str, float]:
        """Train a standalone local model for comparison."""
        print(f"Client {self.client_id}: Training standalone local model...")
        local_model = CustomCNN(num_classes=self.num_classes).to(self.device)
        local_optimizer = optim.Adam([
            {"params": local_model.backbone.features[6:].parameters(),
             "lr": 1e-4},
            {"params": local_model.backbone.classifier.parameters(),
             "lr": 1e-3}
        ])
        local_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            local_optimizer, mode="min", factor=0.5, patience=1
        )
        scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

        if self.train_loader is not None and len(self.train_loader.dataset) > 0:
            local_model.train()
            for epoch in range(3):
                total_loss = 0
                for i, (images, labels) in enumerate(self.train_loader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    local_optimizer.zero_grad()
                    with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                        outputs = local_model(images)
                        loss = self.criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(local_optimizer)
                    scaler.update()
                    total_loss += loss.item()
                    if i % 50 == 0:
                        print(
                            f"Client {self.client_id} Local, Epoch {epoch+1}, "
                            f"Batch {i}/{len(self.train_loader)}, "
                            f"Loss: {loss.item():.4f}"
                        )
                avg_loss = total_loss / len(self.train_loader)
                local_scheduler.step(avg_loss)
                print(
                    f"Client {self.client_id} Local, Epoch {epoch+1} completed, "
                    f"Avg Loss: {avg_loss:.4f}"
                )

        local_model.eval()
        y_true, y_pred = [], []
        total_loss = 0
        with torch.no_grad():
            for images, labels in self.global_test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = local_model(images)
                    loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        metrics = {
            "local_loss": (
                total_loss / len(self.global_test_loader)
                if len(self.global_test_loader) > 0 else float("inf")
            ),
            "local_accuracy": accuracy_score(y_true, y_pred),
            "local_precision": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "local_recall": recall_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "local_f1": f1_score(y_true, y_pred, average="macro", zero_division=0)
        }
        print(
            f"Client {self.client_id} Local Model - Loss: {metrics['local_loss']:.4f}, "
            f"Acc: {metrics['local_accuracy']:.4f}, "
            f"Prec: {metrics['local_precision']:.4f}, "
            f"Rec: {metrics['local_recall']:.4f}, "
            f"F1: {metrics['local_f1']:.4f}"
        )
        return metrics

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return model parameters as NumPy arrays."""
        return [p.cpu().detach().numpy()
                for p in self.model.parameters() if p.requires_grad]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from NumPy arrays."""
        params_dict = zip(
            [p for p in self.model.parameters() if p.requires_grad],
            parameters
        )
        for param, new_param in params_dict:
            param.data = torch.from_numpy(new_param).to(self.device)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        """Train the model on local data."""
        self.set_parameters(parameters)
        if self.train_loader is None or len(self.train_loader.dataset) == 0:
            print(f"Client {self.client_id}: No training data, skipping fit")
            return self.get_parameters(config={}), 0, {"loss": float("inf")}

        self.model.train()
        print(f"Client {self.client_id}: Starting 3 local epochs")
        for epoch in range(3):
            total_loss = 0
            for i, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                total_loss += loss.item()
                if i % 50 == 0:
                    print(
                        f"Client {self.client_id}, Epoch {epoch+1}, "
                        f"Batch {i}/{len(self.train_loader)}, "
                        f"Loss: {loss.item():.4f}"
                    )
            avg_loss = total_loss / len(self.train_loader)
            self.scheduler.step(avg_loss)
            print(
                f"Client {self.client_id}, Epoch {epoch+1} completed, "
                f"Avg Loss: {avg_loss:.4f}"
            )
        return (
            self.get_parameters(config={}),
            len(self.train_loader.dataset),
            {"loss": avg_loss}
        )

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[float, int, Dict[str, float]]:
        """Evaluate the model on the global test set."""
        self.set_parameters(parameters)
        self.model.eval()
        y_true, y_pred = [], []
        total_loss = 0
        with torch.no_grad():
            for images, labels in self.global_test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        avg_loss = total_loss / len(self.global_test_loader)
        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "recall": recall_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "f1": f1_score(y_true, y_pred, average="macro", zero_division=0)
        }
        for m in metrics:
            self.local_metrics[m].append(metrics[m])

        print(
            f"Client {self.client_id} FedAvg Eval - Loss: {metrics['loss']:.4f}, "
            f"Acc: {metrics['accuracy']:.4f}, Prec: {metrics['precision']:.4f}, "
            f"Rec: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}"
        )

        if len(self.local_metrics["accuracy"]) == 3:
            self.plot_metrics()
            self.plot_confusion_matrix(y_true, y_pred)

        return (
            float(avg_loss),
            len(self.global_test_loader.dataset),
            {**metrics, **self.local_model_metrics}
        )

    def plot_metrics(self) -> None:
        """Plot FedAvg vs Local metrics over rounds."""
        metrics = ["loss", "accuracy", "precision", "recall", "f1"]
        for metric in metrics:
            plt.figure(figsize=(8, 5))
            plt.plot(
                range(1, 4),
                self.local_metrics[metric],
                label=f"FedAvg {metric.capitalize()}"
            )
            plt.axhline(
                y=self.local_model_metrics[f"local_{metric}"],
                color="r",
                linestyle="--",
                label=f"Local {metric.capitalize()}"
            )
            plt.xlabel("Round")
            plt.ylabel(metric.capitalize())
            plt.title(f"Client {self.client_id} FedAvg vs Local {metric.capitalize()}")
            plt.legend()
            plt.savefig(f"client_{self.client_id}_{metric}.png")
            plt.close()

    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int]) -> None:
        """Plot confusion matrix for the final round."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Client {self.client_id} Confusion Matrix (Round 3)")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.savefig(f"client_{self.client_id}_confusion_matrix.png")
        plt.close()


def get_random_split(
    dataset: Dataset,
    client_id: int,
    num_clients: int = 3
) -> Subset:
    """Split dataset randomly among clients (80%, 15%, 5%)."""
    total_samples = len(dataset)
    all_indices = np.arange(total_samples)
    np.random.shuffle(all_indices)

    split_proportions = [0.8, 0.15, 0.05]
    split_sizes = [int(p * total_samples) for p in split_proportions]
    split_sizes[-1] = total_samples - sum(split_sizes[:-1])  # Adjust last split

    if client_id == 0:
        indices = all_indices[:split_sizes[0]]
        data_percent = split_proportions[0] * 100
    elif client_id == 1:
        indices = all_indices[split_sizes[0]:split_sizes[0] + split_sizes[1]]
        data_percent = split_proportions[1] * 100
    elif client_id == 2:
        indices = all_indices[split_sizes[0] + split_sizes[1]:]
        data_percent = split_proportions[2] * 100
    else:
        raise ValueError(f"Client ID {client_id} not supported. Use 0, 1, or 2.")

    train_subset = Subset(dataset, indices)
    print(
        f"Client {client_id}: Assigned {len(train_subset)} samples "
        f"({data_percent:.1f}% of total dataset), "
        f"Total dataset size: {total_samples}"
    )
    return train_subset


def get_global_test_set(dataset: Dataset) -> Subset:
    """Create a global test set (20% of dataset)."""
    num_samples = len(dataset)
    test_size = int(0.2 * num_samples)
    indices = np.random.choice(num_samples, test_size, replace=False)
    return Subset(dataset, indices)


if __name__ == "__main__":
    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(f"Starting client with ID: {client_id}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset_path = r"C:\Users\tw9520gi\RJ project\Plant village code\PlantVillage"
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} not found.")
        sys.exit(1)

    full_dataset = CustomImageDataset(root_dir=dataset_path, transform=None)
    num_classes = len(full_dataset.class_names)
    class_names = full_dataset.class_names
    if num_classes < 2:
        print("Dataset must have at least 2 classes.")
        sys.exit(1)

    train_dataset = CustomImageDataset(
        root_dir=dataset_path,
        transform=train_transform
    )
    test_dataset = CustomImageDataset(
        root_dir=dataset_path,
        transform=test_transform
    )

    train_subset = get_random_split(train_dataset, client_id, num_clients=3)
    train_loader = (
        DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=0)
        if len(train_subset) > 0
        else None
    )
    global_test_subset = get_global_test_set(test_dataset)
    global_test_loader = DataLoader(
        global_test_subset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    print(
        f"Client {client_id}: Loaded dataset - Train size: {len(train_subset)}, "
        f"Test size: {len(global_test_subset)}, Classes: {num_classes}"
    )

    client = FlowerClient(
        client_id,
        train_loader,
        global_test_loader,
        num_classes,
        class_names
    )
    fl.client.start_client(
        server_address="localhost:8080",
        client=client.to_client(),
        grpc_max_message_length=1073741824
    )
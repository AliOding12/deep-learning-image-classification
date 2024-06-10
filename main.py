# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.cnn_model import CNNModel
from models.ann_model import ANNModel
from utils.train import train_model
from utils.evaluate import evaluate_model
from utils.plot import plot_metrics

# -----------------------------
# Configuration
# -----------------------------
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Dataset & Dataloaders
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(
    root="data",
    train=True,
    transform=transform,
    download=True
)
test_dataset = datasets.MNIST(
    root="data",
    train=False,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# Initialize Models
# -----------------------------
models = {
    "CNN": CNNModel(),
    "ANN": ANNModel()
}

# -----------------------------
# Training & Evaluation
# -----------------------------
results = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = train_model(
        model, train_loader, criterion, optimizer, DEVICE, epochs=EPOCHS
    )

    print(f"Evaluating {model_name}...")
    accuracy = evaluate_model(model, test_loader, DEVICE)

    results[model_name] = {
        "history": history,
        "accuracy": accuracy
    }

# -----------------------------
# Results & Plots
# -----------------------------
for model_name, res in results.items():
    print(f"{model_name} Accuracy: {res['accuracy']:.2f}%")
    plot_metrics(res['history'], title=f"{model_name} Training History")



# Add main script to orchestrate training and evaluation
# Refactor main script for better modularity
# Integrate early stopping in training

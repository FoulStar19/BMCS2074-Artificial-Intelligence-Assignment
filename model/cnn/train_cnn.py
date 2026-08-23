"""Train the binary CNN used by the sliding-window car detector."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cnn_model import TrafficCNN

IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
REQUIRED_CLASSES = {"background", "car"}


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = correct = samples = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_sum += criterion(outputs, labels).item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            samples += labels.size(0)
    return loss_sum / samples, correct / samples


def save_curves(history, output_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(epochs, history["train_accuracy"], label="Train")
    axes[1].plot(epochs, history["val_accuracy"], label="Validation")
    axes[1].set(title="Binary classification accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    cnn_dir = Path(__file__).resolve().parent
    dataset_dir = cnn_dir.parents[1] / "dataset" / "cnn_car_background"
    output_dir = cnn_dir / "saved_model"
    output_dir.mkdir(exist_ok=True)
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10), transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_data = datasets.ImageFolder(dataset_dir / "train", train_transform)
    val_data = datasets.ImageFolder(dataset_dir / "val", eval_transform)
    if set(train_data.classes) != REQUIRED_CLASSES or train_data.classes != val_data.classes:
        raise ValueError("Run prepare_cnn_dataset.py first. Both car and background classes are required.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; classes: {train_data.classes}; train images: {len(train_data)}; validation images: {len(val_data)}")
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_data, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin_memory)
    model = TrafficCNN(num_classes=2, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    best_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = correct = samples = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            samples += labels.size(0)
        train_loss, train_accuracy = loss_sum / samples, correct / samples
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        print(f"Epoch {epoch:02d}/{args.epochs}: train accuracy={train_accuracy:.4f}, val accuracy={val_accuracy:.4f}")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({"model_state_dict": model.state_dict(), "class_names": train_data.classes,
                        "image_size": IMAGE_SIZE, "mean": MEAN, "std": STD,
                        "best_val_accuracy": best_accuracy}, output_dir / "best_car_detector.pth")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    save_curves(history, output_dir / "training_curves.png")
    print(f"Saved: {output_dir / 'best_car_detector.pth'}")


if __name__ == "__main__":
    main()

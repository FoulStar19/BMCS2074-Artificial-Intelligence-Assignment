"""Evaluate the saved CNN model on the validation crop dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cnn_model import TrafficCNN


def main():
    cnn_dir = Path(__file__).resolve().parent
    checkpoint_path = cnn_dir / "saved_model" / "best_car_detector.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError("Train the model first: best_model.pth was not found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    transform = transforms.Compose([
        transforms.Resize((checkpoint.get("image_size", 224),) * 2),
        transforms.ToTensor(),
        transforms.Normalize(mean=checkpoint["mean"], std=checkpoint["std"]),
    ])
    dataset = datasets.ImageFolder(cnn_dir.parents[1] / "dataset" / "cnn_car_background" / "val", transform)
    if dataset.classes != class_names:
        raise ValueError(f"Validation classes {dataset.classes} differ from checkpoint classes {class_names}.")

    model = TrafficCNN(len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    labels, predictions = [], []
    with torch.no_grad():
        for images, targets in DataLoader(dataset, batch_size=32, shuffle=False):
            outputs = model(images.to(device))
            labels.extend(targets.tolist())
            predictions.extend(outputs.argmax(dim=1).cpu().tolist())

    print(f"Validation accuracy: {sum(a == b for a, b in zip(labels, predictions)) / len(labels):.4f}")
    print(classification_report(labels, predictions, target_names=class_names, zero_division=0))
    matrix = confusion_matrix(labels, predictions, labels=range(len(class_names)))
    display = ConfusionMatrixDisplay(matrix, display_labels=class_names)
    display.plot(cmap="Blues", xticks_rotation=45)
    plt.tight_layout()
    output_path = cnn_dir / "saved_model" / "confusion_matrix.png"
    plt.savefig(output_path, dpi=160)
    print(f"Saved confusion matrix: {output_path}")


if __name__ == "__main__":
    main()

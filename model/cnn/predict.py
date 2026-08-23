"""Predict a vehicle class for one cropped image."""

import argparse
from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms

from cnn_model import TrafficCNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to one cropped vehicle image")
    args = parser.parse_args()
    cnn_dir = Path(__file__).resolve().parent
    checkpoint_path = cnn_dir / "saved_model" / "best_car_detector.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TrafficCNN(len(checkpoint["class_names"]), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((checkpoint.get("image_size", 224),) * 2),
        transforms.ToTensor(),
        transforms.Normalize(mean=checkpoint["mean"], std=checkpoint["std"]),
    ])
    image = Image.open(args.image).convert("RGB")
    with torch.no_grad():
        probabilities = torch.softmax(model(transform(image).unsqueeze(0).to(device)), dim=1)[0]
    confidence, index = probabilities.max(dim=0)
    print(f"Prediction: {checkpoint['class_names'][index.item()]} ({confidence.item() * 100:.2f}%)")


if __name__ == "__main__":
    main()

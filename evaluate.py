import os
import torch
import config
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from microformerx_rtx3050 import MicroFormerX


def main():

    # -----------------------------
    # Device
    # -----------------------------
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # -----------------------------
    # External Test Transform
    # (Harder / Realistic Conditions)
    # -----------------------------
    external_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.IMAGENET_MEAN,
            std=config.IMAGENET_STD
        ),
    ])

    # -----------------------------
    # External Dataset
    # -----------------------------
    external_path = f"{config.DATA_PATH}/external_test"

    external_ds = datasets.ImageFolder(
        root=external_path,
        transform=external_transform
    )

    external_loader = DataLoader(
        external_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    print(f"External samples: {len(external_ds)}")

    # -----------------------------
    # Load Model
    # -----------------------------
    model = MicroFormerX(
        num_classes=config.NUM_CLASSES,
        num_sources=config.NUM_SOURCES
    )

    state_dict = torch.load(config.MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # -----------------------------
    # Evaluation
    # -----------------------------
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in external_loader:

            images = images.to(device)

            outputs = model(images)

            # Handle multi-output models
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # -----------------------------
    # Metrics
    # -----------------------------
    acc = accuracy_score(y_true, y_pred)

    print("\n===== EXTERNAL VALIDATION RESULTS =====")
    print(f"External Accuracy: {acc * 100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=external_ds.classes
    ))

    cm = confusion_matrix(y_true, y_pred)

    print("Confusion Matrix:")
    print(cm)

    # -----------------------------
    # Save Confusion Matrix Image
    # -----------------------------
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest')
    plt.title("External Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(external_ds.classes))
    plt.xticks(tick_marks, external_ds.classes, rotation=45)
    plt.yticks(tick_marks, external_ds.classes)

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig("external_confusion_matrix.png")
    plt.close()

    print("\nConfusion matrix saved as external_confusion_matrix.png")


if __name__ == "__main__":
    main()
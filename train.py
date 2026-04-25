import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import config
from microformerx_rtx3050 import MicroFormerX


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(config.SEED)
    torch.backends.cudnn.benchmark = True
    os.makedirs("outputs", exist_ok=True)

    # --- Separate transforms: augmentation for train, clean eval for val ---
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(f"{config.DATA_PATH}/train", train_transform)
    val_ds   = datasets.ImageFolder(f"{config.DATA_PATH}/val",   val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    model = MicroFormerX(
        config.NUM_CLASSES,
        config.NUM_SOURCES,
    ).to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
    )

    # CosineAnnealing LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS
    )

    # CPU-safe AMP scaler
    use_amp = config.DEVICE == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    def train_epoch():
        model.train()
        losses = []

        for x, y in tqdm(train_loader, desc="Training"):
            x = x.to(config.DEVICE, non_blocking=True)
            y = y.to(config.DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                morph, _, _ = model(x)
                loss = criterion(morph, y)

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            # Gradient clipping for transformer stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())

        return sum(losses) / len(losses)

    def validate():
        model.eval()
        correct = 0
        total   = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(config.DEVICE)
                y = y.to(config.DEVICE)

                pred     = model(x)[0].argmax(1)
                correct += (pred == y).sum().item()
                total   += y.size(0)

        return correct / total

    # Training loop with best-model checkpoint
    best_acc = 0.0
    log_path = "outputs/training_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,loss,val_acc,lr\n")

    for epoch in range(config.EPOCHS):
        loss = train_epoch()
        acc  = validate()
        lr   = scheduler.get_last_lr()[0]
        scheduler.step()

        print(f"\nEpoch {epoch+1}/{config.EPOCHS}  "
              f"Loss {loss:.4f}  Acc {acc:.4f}  LR {lr:.2e}")

        # Save training log
        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{loss:.4f},{acc:.4f},{lr:.6f}\n")

        # Save best checkpoint
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), config.BEST_PATH)
            print(f"  ✅ Best model saved  (val_acc={best_acc:.4f})")

    # Also save final epoch weights
    torch.save(model.state_dict(), config.MODEL_PATH)
    print(f"\nTraining complete. Best val_acc = {best_acc:.4f}")
    print(f"Best checkpoint  → {config.BEST_PATH}")
    print(f"Final checkpoint → {config.MODEL_PATH}")


if __name__ == "__main__":
    main()

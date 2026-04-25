import os
import torch
from torchvision import transforms
from PIL import Image

import config
from microformerx_rtx3050 import MicroFormerX


# Derive class names from dataset folder — stays in sync with training
CLASSES = sorted(os.listdir(os.path.join(config.DATA_PATH, "train")))

# Inference transform — must match validation transform used during training
_transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
])

# Cache the model at module level so it is loaded only once per process
_model = None


def load_model():
    """Load and cache the trained model."""
    global _model
    if _model is None:
        _model = MicroFormerX(config.NUM_CLASSES, config.NUM_SOURCES)
        _model.load_state_dict(
            torch.load(config.MODEL_PATH, map_location=config.DEVICE)
        )
        _model.to(config.DEVICE)
        _model.eval()
    return _model


def predict(image_path: str) -> str:
    """Return the predicted class name for a single image file."""
    model = load_model()

    img = Image.open(image_path).convert("RGB")
    img = _transform(img).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        pred = model(img)[0].argmax(1).item()

    return CLASSES[pred]


if __name__ == "__main__":
    result = predict("dataset/val/algae/1445.jpg")
    print(f"Predicted class: {result}")

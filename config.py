import torch

# ======================
# DATA
# ======================
DATA_PATH    = "dataset"
IMG_SIZE     = 224
NUM_CLASSES  = 4
NUM_SOURCES  = 3

# ImageNet normalization (required by pretrained Swin & ConvNeXt)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ======================
# TRAINING RTX3050 SAFE
# ======================
BATCH_SIZE   = 4
EPOCHS       = 20
LR           = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 2
SEED         = 42

# ======================
# PATHS
# ======================
MODEL_PATH   = "outputs/microformerx_rtx3050.pth"
BEST_PATH    = "outputs/best_model.pth"

# ======================
# DEVICE
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

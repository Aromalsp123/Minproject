import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

import config
from microformerx_rtx3050 import MicroFormerX


# =========================
# LOAD MODEL
# =========================
model = MicroFormerX(config.NUM_CLASSES, config.NUM_SOURCES).to(config.DEVICE)

model.load_state_dict(
    torch.load("outputs/microformerx_rtx3050.pth",
               map_location=config.DEVICE,
               weights_only=True)
)
model.eval()

classes = sorted(os.listdir(os.path.join(config.DATA_PATH, "train")))

transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor()
])


# =========================
# SHAPE DESCRIPTION
# =========================
def describe_shape(ar, circularity):
    if circularity > 0.75:
        circ_desc = "High"
    elif circularity > 0.4:
        circ_desc = "Medium"
    else:
        circ_desc = "Low"

    if ar > 2:
        shape_desc = "Elongated"
    elif ar < 0.5:
        shape_desc = "Flat"
    else:
        shape_desc = "Irregular"

    return shape_desc, circ_desc


# =========================
# VISUAL ANALYSIS
# =========================
def analyze_image(img_path):

    # --- LOAD IMAGE ---
    pil_img = Image.open(img_path).convert("RGB")
    orig = np.array(pil_img)

    # --- CLASSIFICATION ---
    tensor = transform(pil_img).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        morph, _, _ = model(tensor)
        probs = torch.softmax(morph, dim=1)
        pred = probs.argmax(1).item()
        conf = probs.max().item()

    pred_class = classes[pred]

    # --- SEGMENTATION (EDGE BASED) ---
    gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blur, 40, 120)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    detect_img = orig.copy()
    analysis_img = orig.copy()

    # --- FIND LARGEST CONTOUR (particle) ---
    if len(contours) == 0:
        print("No particle detected")
        return

    cnt = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    x,y,w,h = cv2.boundingRect(cnt)
    ar = w/(h+1e-6)

    perimeter = cv2.arcLength(cnt, True)
    circularity = 4*np.pi*area/(perimeter**2+1e-6)

    shape_desc, circ_desc = describe_shape(ar, circularity)

    # =========================
    # DRAW DETECTION
    # =========================
    cv2.drawContours(detect_img, [cnt], -1, (0,255,0), 4)
    cv2.rectangle(detect_img, (x,y), (x+w,y+h), (0,255,0), 3)

    label = f"{pred_class} {conf*100:.0f}%"
    cv2.putText(detect_img, label, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0,255,0), 2, cv2.LINE_AA)

    # =========================
    # DRAW ANALYSIS
    # =========================
    cv2.drawContours(analysis_img, [cnt], -1, (0,255,0), 4)
    cv2.rectangle(analysis_img, (x,y), (x+w,y+h), (0,255,0), 3)

    text_lines = [
        f"Microplastic: {pred_class}",
        f"Confidence: {conf*100:.0f}%",
        f"Shape: {shape_desc}",
        f"Circularity: {circ_desc}",
        "Location: Algae boundary"
    ]

    ty = y-120 if y>120 else y+h+30
    for i, t in enumerate(text_lines):
        cv2.putText(analysis_img, t,
                    (x, ty+i*28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,255,0), 2, cv2.LINE_AA)

    # =========================
    # SAVE 3 PANEL FIGURE
    # =========================
    combined = np.hstack([
        orig,
        detect_img,
        analysis_img
    ])

    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/visual_analysis.jpg",
                cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    print("✅ Saved → outputs/visual_analysis.jpg")
    print("Microplastic:", pred_class)
    print("Confidence:", f"{conf*100:.1f}%")
    print("Shape:", shape_desc)
    print("Circularity:", circ_desc)


# =========================
# RUN
# =========================
if __name__ == "__main__":
    analyze_image("dataset/val/algae/11.jpg")
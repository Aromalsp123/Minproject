"""
MicroPlastic Detection Web App — Flask Backend
==============================================
Endpoints:
  GET  /              → frontend HTML
  GET  /api/status    → model health / device info
  GET  /api/classes   → list of class names
  POST /api/predict   → upload image → JSON result
  POST /api/analyze   → upload image → JSON + annotated base64 image
"""

import os
import io
import sys
import json
import base64
import traceback

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from flask import Flask, request, jsonify, send_from_directory

# ── project imports ──────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(__file__))

import config
from microformerx_rtx3050 import MicroFormerX


# ── app setup ─────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="webapp", static_url_path="")


# ── model (loaded once at startup) ────────────────────────────────────────────

_model = None
_classes = []
_device = config.DEVICE


def _get_classes():
    """Read class names from dataset/train subdirectories."""
    train_dir = os.path.join(config.DATA_PATH, "train")
    if os.path.isdir(train_dir):
        return sorted(os.listdir(train_dir))
    # fallback hardcoded list
    return ["algae", "fiber", "fragment", "pellet"]


def load_model():
    global _model, _classes
    _classes = _get_classes()
    model = MicroFormerX(config.NUM_CLASSES, config.NUM_SOURCES)
    if os.path.exists(config.MODEL_PATH):
        state = torch.load(config.MODEL_PATH, map_location=_device, weights_only=True)
        model.load_state_dict(state)
        print(f"[INFO] Model loaded from {config.MODEL_PATH}")
    else:
        print(f"[WARN] No checkpoint found at {config.MODEL_PATH}. Running without weights.")
    model.to(_device)
    model.eval()
    _model = model


# ── transforms ────────────────────────────────────────────────────────────────

def get_transform():
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def pil_from_request(file_storage) -> Image.Image:
    """Read uploaded file → PIL RGB image."""
    data = file_storage.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return img


# ── morphology helpers (reused from xai.py) ───────────────────────────────────

def extract_morphology(img_rgb: np.ndarray):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (peri ** 2 + 1e-8)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = max(w, h) / (min(w, h) + 1e-8)
    hull = cv2.convexHull(cnt)
    solidity = area / (cv2.contourArea(hull) + 1e-8)
    return {
        "contour": cnt,
        "circularity": round(float(circularity), 4),
        "aspect_ratio": round(float(aspect_ratio), 4),
        "solidity": round(float(solidity), 4),
        "area": round(float(area), 2),
        "bbox": [int(x), int(y), int(w), int(h)],
    }


def shape_label(m):
    if m is None:
        return "Unknown"
    if m["circularity"] > 0.75:
        return "Pellet-like"
    elif m["aspect_ratio"] > 4:
        return "Filament-like"
    elif m["solidity"] < 0.8:
        return "Fragment-like"
    else:
        return "Irregular"


# ── visual analysis (annotated image) ─────────────────────────────────────────

def build_annotated_image(img_rgb: np.ndarray, pred_class: str, conf: float) -> np.ndarray:
    """Draw contour + bounding box + label overlay."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = img_rgb.copy()
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.drawContours(out, [cnt], -1, (0, 255, 180), 3)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 180), 2)
        label = f"{pred_class}  {conf * 100:.1f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        ty = max(y - 10, th + 4)
        cv2.rectangle(out, (x, ty - th - 4), (x + tw + 6, ty + 2), (0, 180, 120), -1)
        cv2.putText(out, label, (x + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)

    return out


def ndarray_to_b64(img_rgb: np.ndarray) -> str:
    """numpy RGB → JPEG base64 string."""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("webapp", "index.html")


@app.route("/api/status")
def api_status():
    model_loaded = _model is not None
    checkpoint_exists = os.path.exists(config.MODEL_PATH)
    return jsonify({
        "model_loaded": model_loaded,
        "checkpoint_exists": checkpoint_exists,
        "device": _device,
        "classes": _classes,
        "num_classes": config.NUM_CLASSES,
        "img_size": config.IMG_SIZE,
    })


@app.route("/api/classes")
def api_classes():
    return jsonify({"classes": _classes})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded (key: 'image')"}), 400

    try:
        pil_img = pil_from_request(request.files["image"])
        transform = get_transform()
        tensor = transform(pil_img).unsqueeze(0).to(_device)

        with torch.no_grad():
            morph_logits, source_logits, _ = _model(tensor)
            morph_probs = torch.softmax(morph_logits, dim=1)[0].cpu().numpy()
            source_probs = torch.softmax(source_logits, dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(morph_probs))
        pred_class = _classes[pred_idx] if pred_idx < len(_classes) else str(pred_idx)
        confidence = float(morph_probs[pred_idx])

        # morphology
        img_rgb = np.array(pil_img.resize((config.IMG_SIZE, config.IMG_SIZE)))
        morph = extract_morphology(img_rgb)
        shape = shape_label(morph)

        # class probabilities list
        class_probs = [
            {"class": cls, "probability": round(float(p), 4)}
            for cls, p in zip(_classes, morph_probs)
        ]

        return jsonify({
            "predicted_class": pred_class,
            "confidence": round(confidence, 4),
            "confidence_pct": round(confidence * 100, 2),
            "class_probabilities": class_probs,
            "shape_analysis": shape,
            "morphology": {
                "circularity": morph["circularity"] if morph else None,
                "aspect_ratio": morph["aspect_ratio"] if morph else None,
                "solidity": morph["solidity"] if morph else None,
                "area_px": morph["area"] if morph else None,
            },
            "source_probabilities": source_probs.tolist(),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded (key: 'image')"}), 400

    try:
        pil_img = pil_from_request(request.files["image"])
        img_rgb = np.array(pil_img.resize((config.IMG_SIZE, config.IMG_SIZE)))

        transform = get_transform()
        tensor = transform(pil_img).unsqueeze(0).to(_device)

        with torch.no_grad():
            morph_logits, _, _ = _model(tensor)
            probs = torch.softmax(morph_logits, dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_class = _classes[pred_idx] if pred_idx < len(_classes) else str(pred_idx)
        confidence = float(probs[pred_idx])

        annotated = build_annotated_image(img_rgb, pred_class, confidence)
        original_b64 = ndarray_to_b64(img_rgb)
        annotated_b64 = ndarray_to_b64(annotated)

        morph = extract_morphology(img_rgb)

        return jsonify({
            "predicted_class": pred_class,
            "confidence": round(confidence, 4),
            "original_image": original_b64,
            "annotated_image": annotated_b64,
            "morphology": {
                "circularity": morph["circularity"] if morph else None,
                "aspect_ratio": morph["aspect_ratio"] if morph else None,
                "solidity": morph["solidity"] if morph else None,
                "area_px": morph["area"] if morph else None,
                "shape_label": shape_label(morph),
            },
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
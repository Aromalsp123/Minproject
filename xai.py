import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import measure, color
from skimage.segmentation import mark_boundaries, slic

from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from lime import lime_image

import config
from microformerx_rtx3050 import MicroFormerX


# ══════════════════════════════════════════════════════════════════════
# MODEL WRAPPER
# ══════════════════════════════════════════════════════════════════════
class MorphHead(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        morph, _, _ = self.model(x)
        return morph


# ══════════════════════════════════════════════════════════════════════
# FIX 1 — SWIN RESHAPE  (handles non-square N safely)
# ══════════════════════════════════════════════════════════════════════
def reshape_swin(tensor):
    if tensor is None:
        return tensor
    if tensor.ndim == 4:
        return tensor
    if tensor.ndim == 3:
        B, N, C = tensor.shape
        H = int(round(N ** 0.5))
        W = N // H
        if H * W < N:
            H += 1
        pad = H * W - N
        if pad > 0:
            tensor = torch.cat([tensor, tensor[:, N - pad:N, :]], dim=1)
        return tensor.permute(0, 2, 1).reshape(B, C, H, W)
    return tensor


def get_swin_target_layer(swin_model):
    last_block = swin_model.layers[-1].blocks[-1]
    if hasattr(last_block, 'attn') and hasattr(last_block.attn, 'proj'):
        return last_block.attn.proj
    if hasattr(last_block, 'attention'):
        return last_block.attention.output.dense
    return last_block


# ══════════════════════════════════════════════════════════════════════
# ALGAE SEGMENTATION
# ══════════════════════════════════════════════════════════════════════
def segment_algae_cells(img_rgb):
    gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh   = clahe.apply(gray)
    blur  = cv2.GaussianBlur(enh, (5, 5), 0)
    _, otsu = cv2.threshold(blur, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    msk = cv2.morphologyEx(otsu, cv2.MORPH_OPEN,  k_open,  iterations=1)
    msk = cv2.morphologyEx(msk,  cv2.MORPH_CLOSE, k_close, iterations=1)
    labeled_raw = measure.label(msk)
    labeled     = np.zeros_like(labeled_raw, dtype=np.uint8)
    nid = 1
    for reg in measure.regionprops(labeled_raw):
        if reg.area > 300:
            labeled[labeled_raw == reg.label] = nid
            nid += 1
    return labeled, msk


# ══════════════════════════════════════════════════════════════════════
# PARTICLE CLASSIFIER
# ══════════════════════════════════════════════════════════════════════
def classify_particle(region):
    area  = region.area
    peri  = max(region.perimeter, 1)
    major = region.major_axis_length
    minor = region.minor_axis_length
    sol   = region.solidity
    ori   = np.degrees(region.orientation)
    circ  = 4 * np.pi * area / (peri ** 2 + 1e-8)
    ar    = major / (minor + 1e-8)
    elong = 1.0 - minor / (major + 1e-8)

    if   circ > 0.70 and ar < 2.0:   mp_type = "Pellet"
    elif ar > 3.5 or elong > 0.68:   mp_type = "Filament"
    elif sol < 0.78:                  mp_type = "Fragment"
    else:                             mp_type = "Fiber/Other"

    return {
        "type":            mp_type,
        "area_px2":        round(area,  1),
        "major_axis_px":   round(major, 1),
        "minor_axis_px":   round(minor, 1),
        "circularity":     round(circ,  3),
        "aspect_ratio":    round(ar,    3),
        "solidity":        round(sol,   3),
        "elongation":      round(elong, 3),
        "orientation_deg": round(ori,   1),
        "centroid":        region.centroid,
        "bbox":            region.bbox,
    }


def _std_map(gray_roi, ksize=5):
    f   = gray_roi.astype(np.float32)
    mu  = cv2.blur(f, (ksize, ksize))
    mu2 = cv2.blur(f ** 2, (ksize, ksize))
    std = np.sqrt(np.maximum(mu2 - mu ** 2, 0))
    return cv2.normalize(std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════
# MICROPLASTIC DETECTION
# FIX 2 — centroid row/col swapped
# FIX 3 — label stacking goes negative; replaced with clamped upward stack
# ══════════════════════════════════════════════════════════════════════
def detect_microplastics_in_algae(img_rgb, algae_labeled):
    COLORS = {
        "Pellet":     (0,   200,  80),
        "Filament":   (220, 100,   0),
        "Fragment":   (210,  20, 190),
        "Fiber/Other":(0,   180, 230),
    }
    CELL_COLOR = (0, 180, 220)
    debug_img  = img_rgb.copy()
    results    = []
    gray       = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    H_img, W_img = img_rgb.shape[:2]

    for cid in np.unique(algae_labeled)[1:]:
        cell_mask = (algae_labeled == cid).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(debug_img, cnts, -1, CELL_COLOR, 2)

        props = measure.regionprops((cell_mask > 0).astype(int))
        if not props:
            continue
        r0, c0, r1, c1 = props[0].bbox
        pad = 6
        r0 = max(0, r0-pad);  c0 = max(0, c0-pad)
        r1 = min(H_img, r1+pad);  c1 = min(W_img, c1+pad)

        roi_gray = gray[r0:r1, c0:c1].copy()
        roi_mask = cell_mask[r0:r1, c0:c1]

        # Signal A
        clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        roi_enh = clahe.apply(roi_gray)
        sig_a   = cv2.adaptiveThreshold(roi_enh, 255,
                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                      cv2.THRESH_BINARY, 11, -5)

        # Signal B
        std_u8  = _std_map(roi_gray)
        vals    = std_u8[roi_mask > 0]
        thr     = int(np.percentile(vals, 75)) if len(vals) else 80
        _, sig_b = cv2.threshold(std_u8, thr, 255, cv2.THRESH_BINARY)

        # Signal C
        roi_m = cv2.bitwise_and(roi_gray, roi_mask)
        _, sig_c = cv2.threshold(roi_m, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        combined = cv2.bitwise_or(
            cv2.bitwise_and(sig_a, sig_b),
            cv2.bitwise_and(sig_b, sig_c)
        )
        combined = cv2.bitwise_and(combined, roi_mask)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k)

        mp_list      = []
        used_label_y = []   # track y positions of placed labels to avoid overlap

        for reg in measure.regionprops(measure.label(combined)):
            if reg.area < 15:
                continue

            info = classify_particle(reg)

            # bbox in full-image coords
            y0, x0, y1, x1 = reg.bbox
            y0 += r0;  y1 += r0
            x0 += c0;  x1 += c0
            info["bbox_global"] = (y0, x0, y1, x1)

            # FIX 2: centroid — reg.centroid is (row, col)
            # row → y axis,  col → x axis
            cy_local, cx_local = reg.centroid
            info["centroid_global"] = (
                int(cx_local) + c0,   # x  (col offset)
                int(cy_local) + r0,   # y  (row offset)
            )
            mp_list.append(info)

            col = COLORS.get(info["type"], (100, 100, 100))

            # Bounding box
            cv2.rectangle(debug_img, (x0, y0), (x1, y1), col, 2)

            # Centroid dot — (x, y) = centroid_global
            cx, cy = info["centroid_global"]
            cv2.circle(debug_img, (cx, cy), 4, col, -1)
            cv2.circle(debug_img, (cx, cy), 4, (255, 255, 255), 1)

            # FIX 3: label placement — stack upward from box top,
            # but clamp to image bounds and avoid collisions
            lbl    = f"{info['type']}  {info['major_axis_px']:.0f}px"
            lx     = min(x0, W_img - 160)
            lx     = max(lx, 2)

            # Desired y = just above bounding box top
            ly_desired = y0 - 6
            # Push up if this slot is already occupied (±10 px)
            ly = ly_desired
            for used in used_label_y:
                if abs(ly - used) < 12:
                    ly = used - 13          # stack further up
            ly = max(ly, 14)               # clamp: never above image top
            used_label_y.append(ly)

            cv2.putText(debug_img, lbl, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(debug_img, lbl, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        col, 1, cv2.LINE_AA)

        cv2.putText(debug_img,
                    f"Cell {cid}  ({len(mp_list)} MP)",
                    (c0, max(r0 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    CELL_COLOR, 2, cv2.LINE_AA)

        results.append({
            "cell_id":         int(cid),
            "n_microplastics": len(mp_list),
            "microplastics":   mp_list,
        })

    return results, debug_img


# ══════════════════════════════════════════════════════════════════════
# LIME
# ══════════════════════════════════════════════════════════════════════
def run_lime(img_rgb, model, device):
    def predict_fn(images):
        t = torch.stack([
            transforms.ToTensor()(Image.fromarray(im.astype(np.uint8)))
            for im in images
        ]).to(device)
        with torch.no_grad():
            return torch.softmax(model(t), dim=1).cpu().numpy()

    segments = slic(img_rgb, n_segments=30, compactness=10, sigma=1)
    explainer = lime_image.LimeImageExplainer()
    exp = explainer.explain_instance(
        img_rgb.astype(np.double), predict_fn,
        top_labels=1, num_samples=1500,         # ← updated
        segmentation_fn=lambda x: segments)
    label = exp.top_labels[0]
    img_lime, mask = exp.get_image_and_mask(
        label, positive_only=True, num_features=6, hide_rest=False)
    return mark_boundaries(img_lime / 255.0,
                           mask * (img_rgb[:, :, 0] < 230),
                           color=(1, 0.8, 0), outline_color=(1, 0.8, 0))


# ══════════════════════════════════════════════════════════════════════
# ANALYZE
# ══════════════════════════════════════════════════════════════════════
def analyze(img_path,
            save_path="outputs/xai_result.png",
            px_per_um=1.0):

    os.makedirs("outputs", exist_ok=True)

    base = MicroFormerX(config.NUM_CLASSES,
                        config.NUM_SOURCES).to(config.DEVICE)
    # FIX 6 — weights_only=False suppresses FutureWarning
    base.load_state_dict(
        torch.load("outputs/microformerx_rtx3050.pth",
                   map_location=config.DEVICE,
                   weights_only=False)
    )
    base.eval()

    model   = MorphHead(base).to(config.DEVICE)
    classes = sorted(os.listdir(os.path.join(config.DATA_PATH, "train")))

    img     = Image.open(img_path).convert("RGB").resize(
                  (config.IMG_SIZE, config.IMG_SIZE))
    img_rgb = np.array(img)
    tensor  = transforms.ToTensor()(img).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    pred_idx   = int(np.argmax(probs))
    pred_class = classes[pred_idx]
    conf       = probs[pred_idx]
    print(f"Prediction: {pred_class}  ({conf*100:.1f}%)")

    # EigenCAM
    target_layer = get_swin_target_layer(base.swin)
    cam = EigenCAM(model=model,
                   target_layers=[target_layer],
                   reshape_transform=reshape_swin)
    cam_map     = cam(input_tensor=tensor, eigen_smooth=True)[0]
    gradcam_img = show_cam_on_image(
        img_rgb.astype(np.float32) / 255.0,
        cam_map, use_rgb=True,
        colormap=cv2.COLORMAP_JET)

    algae_labeled, _ = segment_algae_cells(img_rgb)
    lime_img          = run_lime(img_rgb, model, config.DEVICE)
    mp_results, annotated_img = detect_microplastics_in_algae(
        img_rgb, algae_labeled)

    # ── Stats for summary box ──────────────────────────────────────────
    total_mp    = sum(r["n_microplastics"] for r in mp_results)
    type_counts = {}
    size_list   = []
    for cell in mp_results:
        for mp in cell["microplastics"]:
            type_counts[mp["type"]] = type_counts.get(mp["type"], 0) + 1
            size_list.append(mp["major_axis_px"])

    # ── Figure — white background 2×3 grid matching Image 2 ───────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor("white")

    ax_orig, ax_cam,  ax_lime = axes[0]
    ax_seg,  ax_mp,   ax_sum  = axes[1]

    # Row 1
    ax_orig.imshow(img_rgb)
    ax_orig.set_title("Original Image",   fontsize=11)
    ax_orig.axis("off")

    ax_cam.imshow(gradcam_img)
    ax_cam.set_title("GradCAM (model focus)", fontsize=11)
    ax_cam.axis("off")

    ax_lime.imshow(lime_img)
    ax_lime.set_title("LIME Explanation", fontsize=11)
    ax_lime.axis("off")

    # FIX 7 — Segmentation: salmon/pink tint on cells, dimmed background
    seg_vis = img_rgb.astype(np.float32) / 255.0
    seg_vis[algae_labeled == 0] *= 0.55
    seg_rgb = seg_vis.copy()
    for cid in np.unique(algae_labeled)[1:]:
        mask = algae_labeled == cid
        seg_rgb[mask, 0] = np.clip(seg_vis[mask, 0] * 1.5 + 0.15, 0, 1)
        seg_rgb[mask, 1] = np.clip(seg_vis[mask, 1] * 0.6,        0, 1)
        seg_rgb[mask, 2] = np.clip(seg_vis[mask, 2] * 0.6,        0, 1)

    ax_seg.imshow(np.clip(seg_rgb, 0, 1))
    ax_seg.set_title("Algae Cell Segmentation", fontsize=11)
    ax_seg.axis("off")

    # MP panel
    ax_mp.imshow(annotated_img)
    ax_mp.set_title("Microplastics Inside Algae", fontsize=11)
    ax_mp.axis("off")

    # FIX 5 — Legend on MP panel
    legend_patches = [
        mpatches.Patch(color=(0/255,   200/255,  80/255), label="Pellet"),
        mpatches.Patch(color=(220/255, 100/255,   0/255), label="Filament"),
        mpatches.Patch(color=(210/255,  20/255, 190/255), label="Fragment"),
        mpatches.Patch(color=(0/255,   180/255, 230/255), label="Fiber/Other"),
        mpatches.Patch(color=(0/255,   180/255, 220/255), label="Algae boundary"),
    ]
    ax_mp.legend(handles=legend_patches, loc="lower right",
                 fontsize=7.5, framealpha=0.85,
                 facecolor="white", edgecolor="#aaaaaa")

    # FIX 4 — Full summary text box
    ax_sum.axis("off")
    ax_sum.set_title("Summary", fontsize=11)

    sz_min  = f"{min(size_list):.1f}" if size_list else "–"
    sz_max  = f"{max(size_list):.1f}" if size_list else "–"
    sz_mean = f"{np.mean(size_list):.1f}" if size_list else "–"

    type_lines = "\n".join(
        f"  {t}: {n}"
        for t, n in sorted(type_counts.items())
    ) if type_counts else "  (none)"

    summary = (
        f"Model: {pred_class}  ({conf*100:.1f}%)\n"
        f"Overall shape: {shape_label(None)}\n"
        f"\n"
        f"Algae cells  : {len(mp_results)}\n"
        f"Total MPs    : {total_mp}\n"
        f"\n"
        f"MP types:\n"
        f"{type_lines}\n"
        f"\n"
        f"Size (major axis):\n"
        f"  Min  : {sz_min} px\n"
        f"  Max  : {sz_max} px\n"
        f"  Mean : {sz_mean} px"
    )

    ax_sum.text(
        0.08, 0.92, summary,
        transform=ax_sum.transAxes,
        va="top", ha="left",
        fontsize=9,
        fontfamily="monospace",
        linespacing=1.55,
        bbox=dict(boxstyle="round,pad=0.6",
                  facecolor="#f5f5f5",
                  edgecolor="#aaaaaa",
                  linewidth=1.0)
    )

    fig.suptitle(
        f"Type: {pred_class}  |  Conf: {conf*100:.1f}%  |  "
        f"MPs found: {total_mp}",
        fontsize=11, y=1.005
    )

    plt.tight_layout(pad=1.2)
    plt.savefig(save_path, dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"Saved → {save_path}")
    return mp_results


def shape_label(m):
    if m is None:               return "Unknown"
    if m["circularity"] > 0.75: return "Pellet"
    if m["aspect_ratio"] > 4:   return "Filament"
    if m["solidity"] < 0.8:     return "Fragment"
    return "Algae/Other"


def extract_morphology(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, binary
    cnt  = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    hull = cv2.convexHull(cnt)
    return {
        "contour":      cnt,
        "circularity":  4*np.pi*area/(peri**2+1e-8),
        "aspect_ratio": max(w,h)/(min(w,h)+1e-8),
        "solidity":     area/(cv2.contourArea(hull)+1e-8),
    }, binary


if __name__ == "__main__":
    analyze("dataset/val/algae/18761.jpg",
            save_path="outputs/xai_result12.png",
            px_per_um=1.0)
/**
 * MicroPlastic Detector – Frontend App Logic
 * ===========================================
 * Handles: upload, drag-drop, camera capture, API calls,
 *          result rendering, history management
 */

"use strict";

// ── DOM refs ──────────────────────────────────────────────
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const previewImg = document.getElementById("preview-img");
const previewPh = document.getElementById("preview-ph");
const btnBrowse = document.getElementById("btn-browse");
const btnCamera = document.getElementById("btn-camera");
const btnSample = document.getElementById("btn-sample");
const btnAnalyse = document.getElementById("btn-analyse");
const btnClear = document.getElementById("btn-clear");
const analyseSpinner = document.getElementById("analyse-spinner");
const analyseLabel = document.getElementById("analyse-label");

const statusPill = document.getElementById("status-pill");
const statusText = document.getElementById("status-text");

const resultsPh = document.getElementById("results-placeholder");
const resultsPanel = document.getElementById("results-panel");
const badgeIcon = document.getElementById("badge-icon");
const badgeClass = document.getElementById("badge-class");
const badgeConf = document.getElementById("badge-conf");
const confList = document.getElementById("conf-list");
const morphGrid = document.getElementById("morph-grid");
const detectionImg = document.getElementById("detection-img");
const detectionWrap = document.getElementById("detection-wrap");

const historyGrid = document.getElementById("history-grid");
const historyPh = document.getElementById("history-ph");
const btnClearHistory = document.getElementById("btn-clear-history");

const cameraModal = document.getElementById("camera-modal");
const cameraVideo = document.getElementById("camera-video");
const btnCamSnap = document.getElementById("btn-cam-snap");
const btnCamCancel = document.getElementById("btn-cam-cancel");
const snapCanvas = document.getElementById("snap-canvas");

const toast = document.getElementById("toast");

// ── State ─────────────────────────────────────────────────
let currentFile = null;
let cameraStream = null;
let history = JSON.parse(localStorage.getItem("mp_history") || "[]");
let classes = [];

// ── Class icons ───────────────────────────────────────────
const CLASS_ICONS = {
    algae: "🌿",
    fiber: "🧵",
    fragment: "🔷",
    pellet: "⚪",
};

function getIcon(cls) {
    return CLASS_ICONS[cls?.toLowerCase()] || "🔵";
}

// ── Toast ─────────────────────────────────────────────────
let toastTimer;
function showToast(msg, type = "info") {
    toast.textContent = (type === "success" ? "✅ " : type === "error" ? "❌ " : "ℹ️ ") + msg;
    toast.className = `toast show ${type}`;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { toast.className = "toast"; }, 3500);
}

// ── Status indicator ──────────────────────────────────────
async function checkStatus() {
    statusPill.className = "status-pill loading";
    statusText.textContent = "Connecting…";
    try {
        const res = await fetch("/api/status");
        const data = await res.json();
        classes = data.classes || [];
        if (data.model_loaded) {
            statusPill.className = "status-pill ready";
            statusText.textContent = `Ready · ${data.device.toUpperCase()} · ${data.num_classes} classes`;
        } else {
            statusPill.className = "status-pill loading";
            statusText.textContent = "Model loading…";
        }
    } catch {
        statusPill.className = "status-pill error";
        statusText.textContent = "Server offline";
    }
}

// ── Image load helpers ────────────────────────────────────
function loadPreview(file) {
    currentFile = file;
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    previewImg.classList.add("visible");
    previewPh.style.display = "none";
    btnAnalyse.disabled = false;
    btnClear.disabled = false;
    // reset results
    resultsPanel.classList.add("hidden");
    resultsPh.style.display = "";
}

function loadBlobAsFile(blob, name = "capture.jpg") {
    const file = new File([blob], name, { type: "image/jpeg" });
    loadPreview(file);
}

function clearImage() {
    currentFile = null;
    previewImg.classList.remove("visible");
    previewImg.src = "";
    previewPh.style.display = "";
    btnAnalyse.disabled = true;
    btnClear.disabled = true;
    resultsPanel.classList.add("hidden");
    resultsPh.style.display = "";
}

// ── Upload zone events ────────────────────────────────────
dropZone.addEventListener("click", () => fileInput.click());
btnBrowse.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) loadPreview(fileInput.files[0]);
});

dropZone.addEventListener("dragover", e => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) loadPreview(file);
    else showToast("Please drop an image file", "error");
});

btnClear.addEventListener("click", clearImage);

// ── Sample image ──────────────────────────────────────────
btnSample.addEventListener("click", async () => {
    // Generate a small gradient noise image as demo input
    const canvas = document.createElement("canvas");
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext("2d");
    // Draw a fake microscopy-like background
    const grad = ctx.createRadialGradient(112, 112, 10, 112, 112, 120);
    grad.addColorStop(0, "rgba(180,210,200,1)");
    grad.addColorStop(0.4, "rgba(100,150,130,1)");
    grad.addColorStop(1, "rgba(30,50,45,1)");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 224, 224);
    // Oval particle
    ctx.beginPath();
    ctx.ellipse(112, 112, 30, 10, Math.PI / 5, 0, 2 * Math.PI);
    ctx.fillStyle = "rgba(220,240,200,0.85)";
    ctx.fill();
    canvas.toBlob(blob => loadBlobAsFile(blob, "sample.png"), "image/png");
    showToast("Sample image loaded", "success");
});

// ── Analyse ───────────────────────────────────────────────
btnAnalyse.addEventListener("click", runAnalysis);

async function runAnalysis() {
    if (!currentFile) return;

    // UI: start loading
    btnAnalyse.disabled = true;
    analyseSpinner.classList.add("visible");
    analyseLabel.textContent = "Analysing…";

    try {
        const fd = new FormData();
        fd.append("image", currentFile);

        // call both endpoints in parallel
        const [predRes, analyzeRes] = await Promise.all([
            fetch("/api/predict", { method: "POST", body: fd }),
            fetch("/api/analyze", { method: "POST", body: (() => { const f = new FormData(); f.append("image", currentFile); return f; })() }),
        ]);

        if (!predRes.ok) throw new Error((await predRes.json()).error || "Prediction failed");
        if (!analyzeRes.ok) throw new Error((await analyzeRes.json()).error || "Analysis failed");

        const pred = await predRes.json();
        const analyze = await analyzeRes.json();

        renderResults(pred, analyze);
        addToHistory(pred, previewImg.src);
        showToast(`Classified as ${pred.predicted_class} (${pred.confidence_pct}%)`, "success");

    } catch (err) {
        showToast(err.message || "Analysis failed", "error");
        console.error(err);
    } finally {
        btnAnalyse.disabled = false;
        analyseSpinner.classList.remove("visible");
        analyseLabel.textContent = "🔍 Analyse Image";
    }
}

// ── Render results ────────────────────────────────────────
function renderResults(pred, analyze) {
    // Badge
    const cls = pred.predicted_class;
    const conf = pred.confidence_pct;
    badgeIcon.textContent = getIcon(cls);
    badgeClass.textContent = cls;
    badgeConf.textContent = `Confidence: ${conf}%  ·  Shape: ${pred.shape_analysis}`;

    // Confidence bars
    confList.innerHTML = "";
    const probs = pred.class_probabilities || [];
    probs.sort((a, b) => b.probability - a.probability).forEach(item => {
        const isPred = item.class === cls;
        const pct = (item.probability * 100).toFixed(1);
        confList.insertAdjacentHTML("beforeend", `
      <div class="conf-item">
        <div class="conf-header">
          <span class="conf-label ${isPred ? "predicted" : ""}">${getIcon(item.class)} ${item.class}</span>
          <span class="conf-pct">${pct}%</span>
        </div>
        <div class="conf-bar-bg">
          <div class="conf-bar ${isPred ? "predicted-bar" : "other-bar"}" style="width:0%" data-target="${pct}"></div>
        </div>
      </div>
    `);
    });
    // Animate bars after small delay
    requestAnimationFrame(() => {
        document.querySelectorAll(".conf-bar").forEach(bar => {
            bar.style.width = bar.dataset.target + "%";
        });
    });

    // Morphology
    const m = pred.morphology || {};
    morphGrid.innerHTML = `
    ${morphStat("Circularity", m.circularity != null ? m.circularity.toFixed(3) : "N/A")}
    ${morphStat("Aspect Ratio", m.aspect_ratio != null ? m.aspect_ratio.toFixed(3) : "N/A")}
    ${morphStat("Solidity", m.solidity != null ? m.solidity.toFixed(3) : "N/A")}
    ${morphStat("Area (px²)", m.area_px != null ? Math.round(m.area_px) : "N/A")}
    ${morphStat("Shape Label", pred.shape_analysis || "N/A")}
    ${morphStat("Prediction", cls)}
  `;

    // Detection image
    if (analyze.annotated_image) {
        detectionWrap.querySelector(".preview-placeholder")?.remove();
        detectionImg.src = "data:image/jpeg;base64," + analyze.annotated_image;
        detectionImg.classList.add("visible");
    }

    // Show panel
    resultsPh.style.display = "none";
    resultsPanel.classList.remove("hidden");
}

function morphStat(label, value) {
    return `
    <div class="morph-stat">
      <div class="morph-stat-label">${label}</div>
      <div class="morph-stat-value">${value}</div>
    </div>
  `;
}

// ── Tabs ──────────────────────────────────────────────────
document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        const target = btn.dataset.tab;
        document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
        document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById(`tab-${target}`).classList.add("active");
    });
});

// ── History ───────────────────────────────────────────────
function addToHistory(pred, thumbSrc) {
    history.unshift({ cls: pred.predicted_class, conf: pred.confidence_pct, thumb: thumbSrc, ts: Date.now() });
    if (history.length > 10) history = history.slice(0, 10);
    localStorage.setItem("mp_history", JSON.stringify(history));
    renderHistory();
}

function renderHistory() {
    if (history.length === 0) {
        historyPh.style.display = "";
        return;
    }
    historyPh.style.display = "none";
    // Remove old cards (not the placeholder)
    historyGrid.querySelectorAll(".history-item").forEach(el => el.remove());
    history.forEach((item, i) => {
        const div = document.createElement("div");
        div.className = "history-item";
        div.style.animationDelay = `${i * 0.05}s`;
        div.innerHTML = `
      <img class="history-thumb" src="${item.thumb}" alt="${item.cls}" loading="lazy" />
      <div class="history-meta">
        <div class="history-class">${getIcon(item.cls)} ${item.cls}</div>
        <div class="history-conf">${item.conf}%</div>
      </div>
    `;
        historyGrid.appendChild(div);
    });
}

btnClearHistory.addEventListener("click", () => {
    history = [];
    localStorage.removeItem("mp_history");
    historyGrid.querySelectorAll(".history-item").forEach(el => el.remove());
    historyPh.style.display = "";
    showToast("History cleared");
});

// ── Camera ────────────────────────────────────────────────
btnCamera.addEventListener("click", async () => {
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
        cameraVideo.srcObject = cameraStream;
        cameraModal.classList.add("open");
    } catch {
        showToast("Camera not available", "error");
    }
});

btnCamCancel.addEventListener("click", stopCamera);

btnCamSnap.addEventListener("click", () => {
    const w = cameraVideo.videoWidth;
    const h = cameraVideo.videoHeight;
    snapCanvas.width = w;
    snapCanvas.height = h;
    snapCanvas.getContext("2d").drawImage(cameraVideo, 0, 0, w, h);
    snapCanvas.toBlob(blob => {
        loadBlobAsFile(blob, "camera.jpg");
        stopCamera();
        showToast("Photo captured!", "success");
    }, "image/jpeg", 0.92);
});

function stopCamera() {
    cameraModal.classList.remove("open");
    if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
    }
}

// Close modal on overlay click
cameraModal.addEventListener("click", e => {
    if (e.target === cameraModal) stopCamera();
});

// ── Init ──────────────────────────────────────────────────
(async () => {
    await checkStatus();
    renderHistory();
    // Re-poll status every 10 s
    setInterval(checkStatus, 10_000);
})();

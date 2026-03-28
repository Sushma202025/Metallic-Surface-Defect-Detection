"""
MetalInspect Backend — Enhanced Flask API
  - /predict        → single image: defect + confidence + all class probs + optional GradCAM
  - /predict_batch  → multiple images in one request
  - /health         → model status, device, class list
  - /history        → server-side prediction log (last 100)
"""

import io
import os
import base64
import uuid
from collections import deque
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision import transforms

from model import CNN_ViT

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[MetalInspect] Running on device: {device}")

# ── Load class names from classes.txt (written by train.py) ───────────────────
def _load_classes() -> list:
    if os.path.exists("classes.txt"):
        with open("classes.txt") as f:
            classes = [line.strip() for line in f if line.strip()]
        print(f"[MetalInspect] Loaded {len(classes)} classes from classes.txt: {classes}")
        return classes
    # Fallback: NEU-DET standard 6 classes
    default = ["Crazing", "Inclusion", "Patches", "Pitted", "Rolled", "Scratches"]
    print(f"[MetalInspect] classes.txt not found — using default: {default}")
    return default

CLASSES = _load_classes()

# ── Load model — num_classes comes from classes.txt so it always matches .pth ─
print(f"[MetalInspect] Loading model with {len(CLASSES)} output classes…")
model = CNN_ViT(num_classes=len(CLASSES)).to(device)
model.load_state_dict(torch.load("cnn_vit_metal_defect.pth", map_location=device))
model.eval()
print(f"[MetalInspect] Model ready ✓")

# ── Image transform ────────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Server-side log ────────────────────────────────────────────────────────────
_prediction_log: deque = deque(maxlen=100)

# ── GradCAM ───────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        try:
            target = self.model.vit.blocks[-1].norm1
            target.register_forward_hook(self._save_activation)
            target.register_full_backward_hook(self._save_gradient)
        except Exception:
            pass

    def _save_activation(self, _m, _i, output):
        self.activations = output.detach()

    def _save_gradient(self, _m, _gi, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, tensor, class_idx):
        if self.activations is None:
            return None
        try:
            self.model.zero_grad()
            output = self.model(tensor)
            output[0, class_idx].backward(retain_graph=True)
            if self.gradients is None:
                return None
            weights = self.gradients.mean(dim=1, keepdim=True)
            cam     = F.relu((weights * self.activations).sum(dim=-1))
            n = cam.shape[-1]
            h = w = int(n ** 0.5)
            if h * w != n:
                return None
            cam = cam.reshape(h, w).cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            return cam
        except Exception as e:
            print(f"[GradCAM] {e}")
            return None

gradcam = GradCAM(model)

def cam_to_png_b64(cam, pil_image, alpha=0.5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as mpl_cm
    cam_r  = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(pil_image.size, Image.BILINEAR)) / 255.0
    heatmap = mpl_cm.get_cmap("jet")(cam_r)[:, :, :3]
    orig    = np.array(pil_image.convert("RGB")) / 255.0
    blended = ((1 - alpha) * orig + alpha * heatmap * 255).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

# ── Shared prediction helper ───────────────────────────────────────────────────
def run_prediction(pil_image, include_gradcam=False):
    tensor = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)[0]

    confidence, pred_idx = torch.max(probs, dim=0)
    pred_class = CLASSES[pred_idx.item()]
    all_probs  = {cls: round(probs[i].item() * 100, 2) for i, cls in enumerate(CLASSES)}

    result = {
        "defect":      pred_class,
        "confidence":  round(confidence.item() * 100, 2),
        "all_probs":   all_probs,
        "gradcam_url": None,
    }

    if include_gradcam:
        tensor_g = preprocess(pil_image).unsqueeze(0).to(device).requires_grad_(True)
        cam = gradcam(tensor_g, pred_idx.item())
        if cam is not None:
            result["gradcam_url"] = cam_to_png_b64(cam, pil_image)

    return result

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":   "ok",
        "model":    "CNN_ViT",
        "device":   str(device),
        "classes":  CLASSES,
        "log_size": len(_prediction_log),
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400

    file            = request.files["image"]
    include_gradcam = request.form.get("gradcam", "false").lower() == "true"
    print(f"[/predict] {file.filename}  gradcam={include_gradcam}")

    try:
        pil_image = Image.open(file).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Cannot open image: {e}"}), 400

    result = run_prediction(pil_image, include_gradcam=include_gradcam)

    _prediction_log.appendleft({
        "id":         str(uuid.uuid4())[:8],
        "filename":   file.filename,
        "defect":     result["defect"],
        "confidence": result["confidence"],
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
    })

    print(f"[/predict] → {result['defect']} ({result['confidence']}%)")
    return jsonify(result)


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images provided"}), 400

    results = []
    for f in files:
        try:
            pil_image = Image.open(f).convert("RGB")
            pred = run_prediction(pil_image)
            pred["filename"] = f.filename
            _prediction_log.appendleft({
                "id":         str(uuid.uuid4())[:8],
                "filename":   f.filename,
                "defect":     pred["defect"],
                "confidence": pred["confidence"],
                "timestamp":  datetime.now().isoformat(timespec="seconds"),
            })
            results.append(pred)
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)})

    print(f"[/predict_batch] Processed {len(results)} images")
    return jsonify(results)


@app.route("/history", methods=["GET"])
def history():
    limit = min(int(request.args.get("limit", 50)), 100)
    return jsonify(list(_prediction_log)[:limit])


@app.route("/history", methods=["DELETE"])
def clear_history():
    _prediction_log.clear()
    return jsonify({"status": "cleared"})


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[MetalInspect] Starting server at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)

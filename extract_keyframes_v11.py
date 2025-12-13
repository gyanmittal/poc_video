#!/usr/bin/env python3
import argparse
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, List, Tuple, Optional

import cv2
import duckdb  # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
import timm
from timm.data import resolve_model_data_config, create_transform

import clip  # type: ignore
from PIL import Image
from urllib.request import urlretrieve
import skimage
from skimage import restoration, exposure, color

try:
    import pyiqa  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pyiqa = None

try:
    import piq  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    piq = None

try:
    from insightface.app import FaceAnalysis  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    FaceAnalysis = None


# ============================================================
# Smile detector (OpenCV Haar cascade)
# ============================================================

def _ensure_smile_cascade() -> Optional[cv2.CascadeClassifier]:
    global _SMILE_CASCADE
    if _SMILE_CASCADE is not None:
        return _SMILE_CASCADE
    try:
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_smile.xml")
        if os.path.exists(cascade_path):
            _SMILE_CASCADE = cv2.CascadeClassifier(cascade_path)
    except Exception:
        _SMILE_CASCADE = None
    return _SMILE_CASCADE


def detect_smile(face_img: np.ndarray) -> bool:
    smile_cascade = _ensure_smile_cascade()
    if smile_cascade is None or face_img.size == 0:
        return False
    try:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=20, minSize=(20, 20))
        return len(smiles) > 0
    except Exception:
        return False


def _ensure_secondary_face_detector() -> Optional[cv2.CascadeClassifier]:
    global _SECONDARY_FACE_CASCADE
    if _SECONDARY_FACE_CASCADE is not None:
        return _SECONDARY_FACE_CASCADE
    try:
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if os.path.exists(cascade_path):
            _SECONDARY_FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
    except Exception:
        _SECONDARY_FACE_CASCADE = None
    return _SECONDARY_FACE_CASCADE


def detect_secondary_face(frame_bgr: np.ndarray) -> Optional[dict]:
    cascade = _ensure_secondary_face_detector()
    if cascade is None:
        return None
    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        bbox = (int(x), int(y), int(x + w), int(y + h))
        rel_area = float((w * h) / max(1, frame_bgr.shape[0] * frame_bgr.shape[1]))
        try:
            face_region = frame_bgr[y : y + h, x : x + w]
            face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())
        except Exception:
            sharpness = 0.0
        return {
            "bbox": bbox,
            "rel_area": rel_area,
            "sharpness": sharpness,
        }
    except Exception:
        return None


def _ensure_pedestrian_detector() -> Optional[cv2.HOGDescriptor]:
    global _HOG_PEDESTRIAN
    if _HOG_PEDESTRIAN is not None:
        return _HOG_PEDESTRIAN
    try:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        _HOG_PEDESTRIAN = hog
    except Exception:
        _HOG_PEDESTRIAN = None
    return _HOG_PEDESTRIAN


def estimate_person_ratio(frame_bgr: np.ndarray) -> float:
    hog = _ensure_pedestrian_detector()
    if hog is None:
        return 0.0
    try:
        h, w = frame_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return 0.0
        max_edge = max(h, w)
        scale = 1.0
        resized = frame_bgr
        if max_edge > 640:
            scale = 640.0 / float(max_edge)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        resized_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        rects, _ = hog.detectMultiScale(
            resized_gray,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        total_area = float(h * w)
        if total_area <= 0:
            return 0.0
        area_sum = 0.0
        inv_scale = 1.0 / max(scale, 1e-6)
        for (rx, ry, rw, rh) in rects:
            area_sum += float((rw * inv_scale) * (rh * inv_scale))
        return min(1.0, float(area_sum / total_area))
    except Exception:
        return 0.0


# ============================================================
# Global configuration
# ============================================================

_EMB_MODEL = None
_EMB_TRANSFORM = None
_DEVICE = None

_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_AESTHETIC_HEAD = None
_AESTHETIC_DEVICE = None
_NIMA_METRIC = None
_NIMA_DEVICE = None
_PIQ_METRIC = None
_PIQ_DEVICE = None

_FACE_ANALYZER: Optional[FaceAnalysis] = None
_SMILE_CASCADE: Optional[cv2.CascadeClassifier] = None
_SECONDARY_FACE_CASCADE: Optional[cv2.CascadeClassifier] = None
_HOG_PEDESTRIAN: Optional[cv2.HOGDescriptor] = None

CLIP_SELECTION_THRESHOLD = 5.0
NIMA_SELECTION_THRESHOLD = 4.75
PIQ_SELECTION_THRESHOLD = 0.7
OVERALL_RATING_THRESHOLD = 8.0

VIDEO_EXTENSIONS = (
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".flv",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
)


# ============================================================
# Embedding model (timm)
# ============================================================

def _ensure_embedding_model_loaded(model_name: str = "resnet50") -> None:
    global _EMB_MODEL, _EMB_TRANSFORM, _DEVICE
    if _EMB_MODEL is not None and _EMB_TRANSFORM is not None:
        return

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = timm.create_model(model_name, pretrained=True, num_classes=0)  # feature extractor
    model.eval().to(device)

    config = resolve_model_data_config(model)
    transform = create_transform(**config)

    _EMB_MODEL = model
    _EMB_TRANSFORM = transform
    _DEVICE = device


def compute_embedding(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Compute a feature embedding for a frame using a timm model.
    Returns a 1D numpy array.
    """
    _ensure_embedding_model_loaded()
    assert _EMB_MODEL is not None and _EMB_TRANSFORM is not None
    assert _DEVICE is not None

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    with torch.no_grad():
        x = _EMB_TRANSFORM(pil_img).unsqueeze(0).to(_DEVICE)
        feat = _EMB_MODEL(x)  # (1, D)
        feat = F.normalize(feat, dim=-1)
        feat_np = feat.squeeze(0).cpu().numpy().astype(np.float32)

    return feat_np


# ============================================================
# CLIP + LAION Aesthetic Predictor
# ============================================================


def _get_laion_aesthetic_head(clip_model_short: str = "vit_l_14") -> torch.nn.Module:
    """
    Load the LAION aesthetic linear head for the given CLIP backbone.
    Weights from: https://github.com/LAION-AI/aesthetic-predictor
    """
    home = os.path.expanduser("~")
    cache_folder = os.path.join(home, ".cache", "laion_aesthetic")
    os.makedirs(cache_folder, exist_ok=True)
    path_to_model = os.path.join(cache_folder, f"sa_0_4_{clip_model_short}_linear.pth")

    if not os.path.exists(path_to_model):
        if clip_model_short not in ("vit_l_14", "vit_b_32"):
            raise ValueError(f"Unsupported clip_model_short: {clip_model_short}")
        url_model = (
            f"https://github.com/LAION-AI/aesthetic-predictor/raw/main/"
            f"sa_0_4_{clip_model_short}_linear.pth"
        )
        print(f"Downloading LAION aesthetic head to {path_to_model} ...")
        urlretrieve(url_model, path_to_model)

    if clip_model_short == "vit_l_14":
        head = torch.nn.Linear(768, 1)
    elif clip_model_short == "vit_b_32":
        head = torch.nn.Linear(512, 1)
    else:
        raise ValueError(f"Unsupported clip_model_short: {clip_model_short}")

    state = torch.load(path_to_model, map_location="cpu")
    head.load_state_dict(state)
    head.eval()
    return head


def _ensure_aesthetic_models_loaded() -> None:
    global _CLIP_MODEL, _CLIP_PREPROCESS, _AESTHETIC_HEAD, _AESTHETIC_DEVICE
    if _CLIP_MODEL is not None and _AESTHETIC_HEAD is not None:
        return

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Loading CLIP ViT-L/14 on {device} for aesthetic scoring...")
    clip_model, preprocess = clip.load("ViT-L/14", device=device)  # type: ignore
    aesthetic_head = _get_laion_aesthetic_head("vit_l_14").to(device)

    _CLIP_MODEL = clip_model
    _CLIP_PREPROCESS = preprocess
    _AESTHETIC_HEAD = aesthetic_head
    _AESTHETIC_DEVICE = device


def compute_aesthetic_score(frame_bgr: np.ndarray) -> float:
    """
    Compute an aesthetic score (roughly 0–10) using CLIP embeddings
    and the LAION aesthetic predictor.
    """
    _ensure_aesthetic_models_loaded()
    assert _CLIP_MODEL is not None and _CLIP_PREPROCESS is not None
    assert _AESTHETIC_HEAD is not None and _AESTHETIC_DEVICE is not None

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    with torch.no_grad():
        image_input = _CLIP_PREPROCESS(pil_img).unsqueeze(0).to(_AESTHETIC_DEVICE)
        image_features = _CLIP_MODEL.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        score = _AESTHETIC_HEAD(image_features).squeeze().cpu().item()

    return float(score)


# ============================================================
# Additional aesthetic metrics (NIMA, PIQ, OVERALL_RATING)
# ============================================================

def _score_device() -> str:
    if _AESTHETIC_DEVICE is not None:
        return _AESTHETIC_DEVICE
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _ensure_nima_metric(device: str) -> torch.nn.Module:
    global _NIMA_METRIC, _NIMA_DEVICE
    if pyiqa is None:
        raise RuntimeError("pyiqa is required for NIMA scoring. Install it or disable aesthetic scoring.")
    if _NIMA_METRIC is None or _NIMA_DEVICE != device:
        _NIMA_METRIC = pyiqa.create_metric("nima", device=device)
        _NIMA_DEVICE = device
    return _NIMA_METRIC


def _ensure_piq_metric(device: str) -> torch.nn.Module:
    global _PIQ_METRIC, _PIQ_DEVICE
    if piq is None:
        raise RuntimeError("piq is required for CLIPIQA scoring. Install it or disable aesthetic scoring.")
    if _PIQ_METRIC is None or _PIQ_DEVICE != device:
        _PIQ_METRIC = piq.CLIPIQA().to(device)
        _PIQ_DEVICE = device
    return _PIQ_METRIC


def _frame_to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor


def compute_nima_score(frame_bgr: np.ndarray) -> float:
    device = _score_device()
    metric = _ensure_nima_metric(device)
    tensor = _frame_to_tensor(frame_bgr).to(device)
    with torch.no_grad():
        score = metric(tensor)
        value = float(score.view(-1)[0].detach().cpu().item())
    return value


def compute_piq_score(frame_bgr: np.ndarray) -> float:
    device = _score_device()
    metric = _ensure_piq_metric(device)
    tensor = _frame_to_tensor(frame_bgr).to(device)
    with torch.no_grad():
        score = metric(tensor)
        value = float(score.view(-1)[0].detach().cpu().item())
    return value


def compute_overall_rating(clip_score: Optional[float], nima_score: Optional[float], piq_score: Optional[float]) -> Optional[float]:
    if not all(isinstance(x, (int, float)) for x in (clip_score, nima_score, piq_score)):
        return None
    clip_val = float(clip_score)  # type: ignore[arg-type]
    nima_val = float(nima_score)  # type: ignore[arg-type]
    piq_val = float(piq_score)  # type: ignore[arg-type]
    return (clip_val + nima_val) * piq_val


def _format_score(value: Optional[float]) -> str:
    return "NA" if not isinstance(value, (int, float)) else f"{value:.2f}"


def passes_selection_thresholds(score_info: dict) -> bool:
    clip_score = score_info.get("clip")
    nima_score = score_info.get("nima")
    piq_score = score_info.get("piq")
    overall = score_info.get("overall")
    if not isinstance(clip_score, (int, float)) or clip_score < CLIP_SELECTION_THRESHOLD:
        return False
    if not isinstance(nima_score, (int, float)) or nima_score < NIMA_SELECTION_THRESHOLD:
        return False
    if not isinstance(piq_score, (int, float)) or piq_score < PIQ_SELECTION_THRESHOLD:
        return False
    if not isinstance(overall, (int, float)) or overall < OVERALL_RATING_THRESHOLD:
        return False
    return True


# ============================================================
# Technical Quality Assessment
# ============================================================


def compute_technical_quality(frame_bgr: np.ndarray) -> dict:
    """Compute technical metrics and an overall technical quality score."""
    h, w = frame_bgr.shape[:2]
    total_pixels = int(h * w)

    # Sharpness (variance of Laplacian)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    sharpness_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Prepare RGB float image
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb_f = rgb.astype(np.float32) / 255.0

    # Noise estimation (per channel, mean)
    try:
        sigma = restoration.estimate_sigma(rgb_f, channel_axis=-1)
        noise_sigma = float(np.mean(sigma))
    except Exception:
        noise_sigma = 0.0

    # Exposure histogram & clipping
    gray_f = gray.astype(np.float32) / 255.0
    try:
        hist, _ = exposure.histogram(gray_f, nbins=256)
        total = float(hist.sum()) or 1.0
        clip_pct_shadows = float(hist[0:2].sum() / total)
        clip_pct_highlights = float(hist[-2:].sum() / total)
    except Exception:
        clip_pct_shadows = 0.0
        clip_pct_highlights = 0.0
    mean_luma = float(gray_f.mean()) if gray_f.size else 0.0

    # Color cast using LAB mean a/b magnitude
    try:
        lab = color.rgb2lab(rgb_f)
        mean_a = float(np.mean(lab[..., 1]))
        mean_b = float(np.mean(lab[..., 2]))
        color_cast_score = float(np.sqrt(mean_a ** 2 + mean_b ** 2))
    except Exception:
        color_cast_score = 0.0

    # Component scores (normalized 0-1)
    min_sharpness_var = 100.0
    max_noise_sigma = 0.08
    max_clip_pct = 0.01
    max_color_cast_score = 10.0

    sharp_score = float(min(1.0, sharpness_var / min_sharpness_var))
    noise_score = float(1.0 - min(1.0, noise_sigma / max_noise_sigma))
    clip_score_sh = float(1.0 - min(1.0, clip_pct_shadows / max_clip_pct))
    clip_score_hi = float(1.0 - min(1.0, clip_pct_highlights / max_clip_pct))

    if mean_luma < 0.2:
        exp_score = mean_luma / 0.2
    elif mean_luma > 0.8:
        exp_score = (1.0 - mean_luma) / 0.2
    else:
        exp_score = 1.0
    exp_score = float(np.clip(exp_score, 0.0, 1.0))

    color_score = float(1.0 - min(1.0, color_cast_score / max_color_cast_score))

    tech_score = 100.0 * float(np.mean([
        sharp_score,
        noise_score,
        clip_score_sh,
        clip_score_hi,
        exp_score,
        color_score,
    ]))

    return {
        "sharpness_var": sharpness_var,
        "noise_sigma": noise_sigma,
        "clip_pct_shadows": clip_pct_shadows,
        "clip_pct_highlights": clip_pct_highlights,
        "mean_luma": mean_luma,
        "color_cast_score": color_cast_score,
        "resolution_w": int(w),
        "resolution_h": int(h),
        "total_pixels": total_pixels,
        "tech_score": tech_score,
    }


def is_black_slate(gray: np.ndarray, mean_luma: float, threshold_ratio: float, min_ratio: float) -> bool:
    if gray.size == 0:
        return False
    non_black_ratio = float((gray > threshold_ratio).mean())
    max_luma = float(gray.max())
    return (non_black_ratio < min_ratio and mean_luma < threshold_ratio) or (
        mean_luma < 0.05 and (max_luma - mean_luma) > 0.5
    )


# ============================================================
# Face detection and composition scoring
# ============================================================


def _ensure_face_analyzer_loaded() -> None:
    global _FACE_ANALYZER
    if FaceAnalysis is None:
        return
    if _FACE_ANALYZER is None:
        try:
            _FACE_ANALYZER = FaceAnalysis(name="buffalo_l")
            ctx_id = 0 if torch.cuda.is_available() else -1
            _FACE_ANALYZER.prepare(ctx_id=ctx_id, det_size=(640, 640))
        except Exception:
            _FACE_ANALYZER = None


# ============================================================
# Face detection and composition scoring
# ============================================================


def compute_face_features(frame_bgr: np.ndarray) -> dict:
    if FaceAnalysis is None:
        return {
            "num_faces": 0,
            "has_face": False,
            "main_face_conf": 0.0,
            "main_face_bbox": None,
            "main_face_rel_area": 0.0,
            "face_size_score": 0.0,
            "face_quality_score": 0.0,
            "face_sharpness_var": 0.0,
        }

    _ensure_face_analyzer_loaded()
    if _FACE_ANALYZER is None:
        return {
            "num_faces": 0,
            "has_face": False,
            "main_face_conf": 0.0,
            "main_face_bbox": None,
            "main_face_rel_area": 0.0,
            "face_size_score": 0.0,
            "face_quality_score": 0.0,
            "face_sharpness_var": 0.0,
        }

    h, w = frame_bgr.shape[:2]
    try:
        faces = _FACE_ANALYZER.get(frame_bgr)
    except Exception:
        faces = []

    if not faces:
        return {
            "num_faces": 0,
            "has_face": False,
            "main_face_conf": 0.0,
            "main_face_bbox": None,
            "main_face_rel_area": 0.0,
            "face_size_score": 0.0,
            "face_quality_score": 0.0,
            "face_sharpness_var": 0.0,
        }

    num_faces = len(faces)
    main_face = max(faces, key=lambda f: getattr(f, "det_score", 0.0))
    bbox = getattr(main_face, "bbox", None)
    det_conf = float(getattr(main_face, "det_score", 0.0))

    if bbox is None or len(bbox) < 4:
        return {
            "num_faces": num_faces,
            "has_face": True,
            "main_face_conf": det_conf,
            "main_face_bbox": None,
            "main_face_rel_area": 0.0,
            "face_size_score": 0.0,
            "face_quality_score": 100.0 * max(0.0, min(1.0, det_conf)),
            "face_sharpness_var": 0.0,
        }

    x_min = int(max(0, min(w, bbox[0])))
    y_min = int(max(0, min(h, bbox[1])))
    x_max = int(max(0, min(w, bbox[2])))
    y_max = int(max(0, min(h, bbox[3])))

    face_area = max(0, x_max - x_min) * max(0, y_max - y_min)
    total_area = float(w * h) if w > 0 and h > 0 else 1.0
    main_face_rel_area = float(face_area / total_area) if total_area > 0 else 0.0

    target_min = 0.05
    target_max = 0.40
    if main_face_rel_area <= 0:
        size_score = 0.0
    elif main_face_rel_area < target_min:
        size_score = main_face_rel_area / target_min
    elif main_face_rel_area > target_max:
        size_score = max(0.0, 1.0 - (main_face_rel_area - target_max) / target_max)
    else:
        size_score = 1.0
    size_score = float(np.clip(size_score, 0.0, 1.0))

    main_conf = det_conf
    face_quality_score = 100.0 * float(0.6 * main_conf + 0.4 * size_score)

    face_region = frame_bgr[y_min:y_max, x_min:x_max]
    face_sharpness_var = 0.0
    face_tilt_deg = 0.0
    if face_region.size > 0:
        try:
            face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            face_sharpness_var = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())
        except Exception:
            face_sharpness_var = 0.0
        try:
            kps = getattr(main_face, "kps", None)
            if kps is not None and len(kps) >= 2:
                left_eye = kps[0]
                right_eye = kps[1]
                dy = float(right_eye[1] - left_eye[1])
                dx = float(right_eye[0] - left_eye[0])
                face_tilt_deg = float(np.degrees(np.arctan2(dy, dx)))
        except Exception:
            face_tilt_deg = 0.0

    return {
        "num_faces": num_faces,
        "has_face": True,
        "main_face_conf": main_conf,
        "main_face_bbox": (x_min, y_min, x_max, y_max),
        "main_face_rel_area": main_face_rel_area,
        "face_size_score": size_score,
        "face_quality_score": face_quality_score,
        "face_sharpness_var": face_sharpness_var,
        "face_tilt_deg": face_tilt_deg,
    }


def compute_composition_features(frame_bgr: np.ndarray, face_info: dict | None = None) -> dict:
    """Compute composition-related metrics for the frame."""
    h, w = frame_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return {
            "subject_x_norm": 0.5,
            "subject_y_norm": 0.5,
            "rule_of_thirds_score": 0.0,
            "centered_score": 0.0,
            "edge_proximity_score": 0.0,
            "composition_score": 0.0,
        }

    if face_info and face_info.get("has_face") and face_info.get("main_face_bbox"):
        x_min, y_min, x_max, y_max = face_info["main_face_bbox"]
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
    else:
        cx = w / 2.0
        cy = h / 2.0

    subject_x_norm = float(cx / max(1, w))
    subject_y_norm = float(cy / max(1, h))

    thirds_points = [
        (1.0 / 3.0, 1.0 / 3.0),
        (2.0 / 3.0, 1.0 / 3.0),
        (1.0 / 3.0, 2.0 / 3.0),
        (2.0 / 3.0, 2.0 / 3.0),
    ]
    dists = [
        ((subject_x_norm - tx) ** 2 + (subject_y_norm - ty) ** 2) ** 0.5
        for (tx, ty) in thirds_points
    ]
    d_min = min(dists) if dists else 1.0
    thirds_score = max(0.0, min(1.0, 1.0 - d_min / 0.25))

    dx_center = abs(subject_x_norm - 0.5)
    dy_center = abs(subject_y_norm - 0.5)
    centered_score = max(0.0, min(1.0, 1.0 - (dx_center + dy_center) / 0.2))

    dist_left = subject_x_norm
    dist_right = 1.0 - subject_x_norm
    dist_top = subject_y_norm
    dist_bottom = 1.0 - subject_y_norm
    d_edge = min(dist_left, dist_right, dist_top, dist_bottom)
    edge_proximity_score = max(0.0, min(1.0, d_edge / 0.1))

    composition_score = 100.0 * float(np.mean([
        0.5 * thirds_score + 0.5 * centered_score,
        edge_proximity_score,
    ]))

    return {
        "subject_x_norm": subject_x_norm,
        "subject_y_norm": subject_y_norm,
        "rule_of_thirds_score": thirds_score,
        "centered_score": centered_score,
        "edge_proximity_score": edge_proximity_score,
        "composition_score": composition_score,
    }


#!/usr/bin/env python3
import argparse
import os
import shutil
from typing import List, Tuple, Optional

import cv2
import duckdb  # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
import timm
from timm.data import resolve_model_data_config, create_transform

import clip  # type: ignore
from PIL import Image
from urllib.request import urlretrieve
import skimage
from skimage import restoration, exposure, color

try:
    from insightface.app import FaceAnalysis  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    FaceAnalysis = None

# ============================================================
# Global configuration
# ============================================================

_EMB_MODEL = None
_EMB_TRANSFORM = None
_DEVICE = None

_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_AESTHETIC_HEAD = None
_AESTHETIC_DEVICE = None


# Recommended default technical cutoffs for gallery/stock usage
MIN_TECH_SCORE = 70.0       # overall technical quality 0–100
MIN_RES_LONG_EDGE = 320     # minimum width or height in pixels
MIN_TOTAL_PIXELS = 200_000  # ~0.2 MP
MIN_SHARPNESS_VAR = 80.0    # absolute minimum accept, below = junk
MAX_NOISE_SIGMA = 0.12      # hard upper bound on noise
MAX_CLIP_PCT = 0.05         # >5% clipped is too much

# Composition defaults
MIN_COMPOSITION_SCORE = 60.0

# Face framing defaults
MIN_FACE_MIN_DIM_RATIO = 0.12  # minimum fraction of shorter edge for face min dimension
FACE_ASPECT_MIN = 0.6
FACE_ASPECT_MAX = 1.8
MIN_FACE_SHARPNESS = 60.0
MAX_FACE_TILT_DEGREES = 35.0

# ============================================================
# Embedding model (timm)
# ============================================================

def _ensure_embedding_model_loaded(model_name: str = "resnet50") -> None:
    global _EMB_MODEL, _EMB_TRANSFORM, _DEVICE
    if _EMB_MODEL is not None and _EMB_TRANSFORM is not None:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(model_name, pretrained=True, num_classes=0)  # feature extractor
    model.eval().to(device)

    config = resolve_model_data_config(model)
    transform = create_transform(**config)

    _EMB_MODEL = model
    _EMB_TRANSFORM = transform
    _DEVICE = device


def compute_embedding(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Compute a feature embedding for a frame using a timm model.
    Returns a 1D numpy array.
    """
    _ensure_embedding_model_loaded()
    assert _EMB_MODEL is not None and _EMB_TRANSFORM is not None
    assert _DEVICE is not None

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    with torch.no_grad():
        x = _EMB_TRANSFORM(pil_img).unsqueeze(0).to(_DEVICE)
        feat = _EMB_MODEL(x)  # (1, D)
        feat = F.normalize(feat, dim=-1)
        feat_np = feat.squeeze(0).cpu().numpy().astype(np.float32)

    return feat_np


# ============================================================
# CLIP + LAION Aesthetic Predictor
# ============================================================

def _get_laion_aesthetic_head(clip_model_short: str = "vit_l_14") -> torch.nn.Module:
    """
    Load the LAION aesthetic linear head for the given CLIP backbone.
    Weights from: https://github.com/LAION-AI/aesthetic-predictor
    """
    home = os.path.expanduser("~")
    cache_folder = os.path.join(home, ".cache", "laion_aesthetic")
    os.makedirs(cache_folder, exist_ok=True)
    path_to_model = os.path.join(cache_folder, f"sa_0_4_{clip_model_short}_linear.pth")

    if not os.path.exists(path_to_model):
        if clip_model_short not in ("vit_l_14", "vit_b_32"):
            raise ValueError(f"Unsupported clip_model_short: {clip_model_short}")
        url_model = (
            f"https://github.com/LAION-AI/aesthetic-predictor/raw/main/"
            f"sa_0_4_{clip_model_short}_linear.pth"
        )
        print(f"Downloading LAION aesthetic head to {path_to_model} ...")
        urlretrieve(url_model, path_to_model)

    if clip_model_short == "vit_l_14":
        head = torch.nn.Linear(768, 1)
    elif clip_model_short == "vit_b_32":
        head = torch.nn.Linear(512, 1)
    else:
        raise ValueError(f"Unsupported clip_model_short: {clip_model_short}")

    state = torch.load(path_to_model, map_location="cpu")
    head.load_state_dict(state)
    head.eval()
    return head


def _ensure_aesthetic_models_loaded() -> None:
    global _CLIP_MODEL, _CLIP_PREPROCESS, _AESTHETIC_HEAD, _AESTHETIC_DEVICE
    if _CLIP_MODEL is not None and _AESTHETIC_HEAD is not None:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP ViT-L/14 on {device} for aesthetic scoring...")
    clip_model, preprocess = clip.load("ViT-L/14", device=device)  # type: ignore
    aesthetic_head = _get_laion_aesthetic_head("vit_l_14").to(device)

    _CLIP_MODEL = clip_model
    _CLIP_PREPROCESS = preprocess
    _AESTHETIC_HEAD = aesthetic_head
    _AESTHETIC_DEVICE = device


def compute_aesthetic_score(frame_bgr: np.ndarray) -> float:
    """
    Compute an aesthetic score (roughly 0–10) using CLIP embeddings
    and the LAION aesthetic predictor.
    """
    _ensure_aesthetic_models_loaded()
    assert _CLIP_MODEL is not None and _CLIP_PREPROCESS is not None
    assert _AESTHETIC_HEAD is not None and _AESTHETIC_DEVICE is not None

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    with torch.no_grad():
        image_input = _CLIP_PREPROCESS(pil_img).unsqueeze(0).to(_AESTHETIC_DEVICE)
        image_features = _CLIP_MODEL.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        score = _AESTHETIC_HEAD(image_features).squeeze().cpu().item()

    return float(score)


# ============================================================
# Technical Quality Assessment
# ============================================================


def compute_technical_quality(frame_bgr: np.ndarray) -> dict:
    """Compute technical metrics and an overall technical quality score."""
    h, w = frame_bgr.shape[:2]
    total_pixels = int(h * w)

    # Sharpness (variance of Laplacian)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    sharpness_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Prepare RGB float image
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb_f = rgb.astype(np.float32) / 255.0

    # Noise estimation (per channel, mean)
    try:
        sigma = restoration.estimate_sigma(rgb_f, channel_axis=-1)
        noise_sigma = float(np.mean(sigma))
    except Exception:
        noise_sigma = 0.0

    # Exposure histogram & clipping
    gray_f = gray.astype(np.float32) / 255.0
    try:
        hist, _ = exposure.histogram(gray_f, nbins=256)
        total = float(hist.sum()) or 1.0
        clip_pct_shadows = float(hist[0:2].sum() / total)
        clip_pct_highlights = float(hist[-2:].sum() / total)
    except Exception:
        clip_pct_shadows = 0.0
        clip_pct_highlights = 0.0
    mean_luma = float(gray_f.mean()) if gray_f.size else 0.0

    # Color cast using LAB mean a/b magnitude
    try:
        lab = color.rgb2lab(rgb_f)
        mean_a = float(np.mean(lab[..., 1]))
        mean_b = float(np.mean(lab[..., 2]))
        color_cast_score = float(np.sqrt(mean_a ** 2 + mean_b ** 2))
    except Exception:
        color_cast_score = 0.0

    # Component scores (normalized 0-1)
    min_sharpness_var = 100.0
    max_noise_sigma = 0.08
    max_clip_pct = 0.01
    max_color_cast_score = 10.0

    sharp_score = float(min(1.0, sharpness_var / min_sharpness_var))
    noise_score = float(1.0 - min(1.0, noise_sigma / max_noise_sigma))
    clip_score_sh = float(1.0 - min(1.0, clip_pct_shadows / max_clip_pct))
    clip_score_hi = float(1.0 - min(1.0, clip_pct_highlights / max_clip_pct))

    if mean_luma < 0.2:
        exp_score = mean_luma / 0.2
    elif mean_luma > 0.8:
        exp_score = (1.0 - mean_luma) / 0.2
    else:
        exp_score = 1.0
    exp_score = float(np.clip(exp_score, 0.0, 1.0))

    color_score = float(1.0 - min(1.0, color_cast_score / max_color_cast_score))

    tech_score = 100.0 * float(np.mean([
        sharp_score,
        noise_score,
        clip_score_sh,
        clip_score_hi,
        exp_score,
        color_score,
    ]))

    return {
        "sharpness_var": sharpness_var,
        "noise_sigma": noise_sigma,
        "clip_pct_shadows": clip_pct_shadows,
        "clip_pct_highlights": clip_pct_highlights,
        "mean_luma": mean_luma,
        "color_cast_score": color_cast_score,
        "resolution_w": int(w),
        "resolution_h": int(h),
        "total_pixels": total_pixels,
        "tech_score": tech_score,
    }


# ============================================================
# Video frame extraction
# ============================================================

def extract_frames(
    video_path: str,
    frame_stride_sec: float,
    max_seconds: float | None = None,
    start_sec: float = 0.0,
) -> Tuple[Iterable[Tuple[int, float, np.ndarray]], int, float]:
    """
    Stream frames sampled by stride: yields (frame_idx, timestamp_sec, frame_bgr).
    Returns (iterator, total_frame_count, fps).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    orientation_prop = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
    rotation_degrees = 0
    if orientation_prop is not None:
        try:
            rotation_value = cap.get(orientation_prop)
            if rotation_value in (90, 180, 270):
                rotation_degrees = int(rotation_value)
        except Exception:
            rotation_degrees = 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride_frames = max(1, int(round(frame_stride_sec * fps)))
    start_sec = max(0.0, float(start_sec))
    start_frame = int(min(total_frames, max(0, round(start_sec * fps))))

    def _frame_iter() -> Iterable[Tuple[int, float, np.ndarray]]:
        def _apply_rotation(frame: np.ndarray) -> np.ndarray:
            if rotation_degrees == 90:
                return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            if rotation_degrees == 180:
                return cv2.rotate(frame, cv2.ROTATE_180)
            if rotation_degrees == 270:
                return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return frame

        frame_idx = start_frame
        try:
            while frame_idx < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                timestamp_sec = frame_idx / fps
                if max_seconds is not None and timestamp_sec >= float(max_seconds):
                    break
                yield frame_idx, timestamp_sec, _apply_rotation(frame).copy()
                frame_idx += stride_frames
        finally:
            cap.release()

    return _frame_iter(), total_frames, float(fps)


def get_video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0:
        fps = 25.0
    return float(total_frames) / max(fps, 1e-6)


# ============================================================
# DuckDB utilities
# ============================================================

def init_db(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS frames (
            video_path      TEXT,
            frame_index     INTEGER,
            timestamp_sec   DOUBLE,
            aesthetic       DOUBLE,
            nima_score      DOUBLE,
            piq_score       DOUBLE,
            overall_rating  DOUBLE,
            tech_score      DOUBLE,
            sharpness_var   DOUBLE,
            noise_sigma     DOUBLE,
            clip_pct_shadows DOUBLE,
            clip_pct_highlights DOUBLE,
            mean_luma       DOUBLE,
            color_cast_score DOUBLE,
            num_faces       INTEGER,
            has_face        BOOLEAN,
            face_quality_score DOUBLE,
            main_face_rel_area DOUBLE,
            composition_score DOUBLE,
            subject_x_norm DOUBLE,
            subject_y_norm DOUBLE,
            embed_dim       INTEGER,
            embedding       BLOB
        );
        """
    )
    for col, dtype in [
        ("tech_score", "DOUBLE"),
        ("sharpness_var", "DOUBLE"),
        ("noise_sigma", "DOUBLE"),
        ("clip_pct_shadows", "DOUBLE"),
        ("clip_pct_highlights", "DOUBLE"),
        ("mean_luma", "DOUBLE"),
        ("color_cast_score", "DOUBLE"),
        ("num_faces", "INTEGER"),
        ("has_face", "BOOLEAN"),
        ("face_quality_score", "DOUBLE"),
        ("main_face_rel_area", "DOUBLE"),
        ("composition_score", "DOUBLE"),
        ("subject_x_norm", "DOUBLE"),
        ("subject_y_norm", "DOUBLE"),
        ("nima_score", "DOUBLE"),
        ("piq_score", "DOUBLE"),
        ("overall_rating", "DOUBLE"),
    ]:
        try:
            conn.execute(f"ALTER TABLE frames ADD COLUMN {col} {dtype};")
        except Exception:
            pass


def insert_frame_row(
    conn: duckdb.DuckDBPyConnection,
    video_path: str,
    frame_index: int,
    timestamp_sec: float,
    aesthetic: float,
    nima_score: Optional[float],
    piq_score: Optional[float],
    overall_rating: Optional[float],
    tech_score: float,
    sharpness_var: float,
    noise_sigma: float,
    clip_pct_shadows: float,
    clip_pct_highlights: float,
    mean_luma: float,
    color_cast_score: float,
    num_faces: int,
    has_face: bool,
    face_quality_score: float,
    main_face_rel_area: float,
    composition_score: float,
    subject_x_norm: float,
    subject_y_norm: float,
    embedding: np.ndarray,
) -> None:
    emb_bytes = embedding.tobytes()
    emb_dim = int(embedding.shape[0])

    conn.execute(
        """
        INSERT INTO frames (
            video_path, frame_index, timestamp_sec,
            aesthetic, nima_score, piq_score, overall_rating,
            tech_score, sharpness_var, noise_sigma,
            clip_pct_shadows, clip_pct_highlights, mean_luma,
            color_cast_score, num_faces, has_face,
            face_quality_score, main_face_rel_area, composition_score,
            subject_x_norm, subject_y_norm,
            embed_dim, embedding
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            video_path,
            frame_index,
            float(timestamp_sec),
            float(aesthetic),
            float(nima_score) if nima_score is not None else None,
            float(piq_score) if piq_score is not None else None,
            float(overall_rating) if overall_rating is not None else None,
            float(tech_score),
            float(sharpness_var),
            float(noise_sigma),
            float(clip_pct_shadows),
            float(clip_pct_highlights),
            float(mean_luma),
            float(color_cast_score),
            int(num_faces),
            bool(has_face),
            float(face_quality_score),
            float(main_face_rel_area),
            float(composition_score),
            float(subject_x_norm),
            float(subject_y_norm),
            emb_dim,
            emb_bytes,
        ],
    )


def canonical_video_key(path: str) -> str:
    try:
        return os.path.realpath(os.path.abspath(path))
    except Exception:
        try:
            return os.path.abspath(path)
        except Exception:
            return path


def video_path_aliases(path: str) -> List[str]:
    aliases = {path}
    try:
        aliases.add(os.path.abspath(path))
    except Exception:
        pass
    try:
        aliases.add(os.path.realpath(path))
    except Exception:
        pass
    return list(aliases)


def clear_previous_video(conn: duckdb.DuckDBPyConnection, video_paths: Iterable[str]) -> None:
    unique_paths = list(dict.fromkeys(video_paths))
    for alias in unique_paths:
        conn.execute("DELETE FROM frames WHERE video_path = ?", [alias])


# ============================================================
# Main pipeline
# ============================================================

def process_video(
    video_path: str,
    db_path: str,
    frame_stride_sec: float = 1.0,
    max_frames: int | None = None,
    output_dir: str = "outputs",
    max_seconds: float | None = None,
    min_aesthetic: float = 0.0,
    dedup_threshold: float = 0.15,
    final_dedup_threshold: float | None = None,
    progress_only: bool = False,
    enable_face_checks: bool = True,
    adaptive_stride: bool = False,
    adaptive_stride_threshold: float = 3.0,
    face_skin_threshold: float = 0.05,
    max_face_tilt: float = MAX_FACE_TILT_DEGREES,
    max_workers: int = 4,
    black_frame_threshold: float = 0.05,
    black_frame_ratio: float = 0.05,
    min_tech_score: float = MIN_TECH_SCORE,
    require_face: bool = False,
    allow_faceless_scenes: bool = False,
    faceless_min_aesthetic: float = 5.0,
    allow_faceless_scenic_only: bool = False,
    faceless_max_skin_ratio: float = 0.05,
    faceless_motion_threshold: float = 2.0,
    faceless_confidence_threshold: float = 0.3,
    faceless_people_ratio: float = 0.05,
    min_face_area_ratio: float = 0.0,
    min_face_y_center: float = 0.0,
    face_sharpness_threshold: float = MIN_FACE_SHARPNESS,
    skip_overlay_cards: bool = False,
    overlay_coverage: float = 0.3,
    overlay_sat_threshold: float = 0.5,
    overlay_sat_ratio: float = 0.6,
    overlay_unique_threshold: float = 0.2,
    overlay_hue_dominance: float = 0.7,
    overlay_color_ratio: float = 0.4,
    use_aesthetic: bool = True,
    start_sec: float = 0.0,
    skip_to_final_stage: bool = False,
    prefer_smiles: bool = True,
    summary_log_path: Optional[str] = None,
    global_state: Optional[dict] = None,
) -> None:
    overall_start = time.time()
    start_sec = max(0.0, float(start_sec))
    first_chunk = start_sec <= 1e-6
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if use_aesthetic and (pyiqa is None or piq is None):
        raise RuntimeError(
            "pyiqa and piq are required for aesthetic scoring in v11. "
            "Install them or run with --disable-aesthetic."
        )
    output_root = os.path.abspath(output_dir)
    video_db_key = canonical_video_key(video_path)
    video_key_aliases = video_path_aliases(video_path)

    if not progress_only:
        chunk_label = (
            f"[chunk {start_sec:.1f}s - {max_seconds if max_seconds is not None else 'end'}s]"
        )
        print(f"Opening DuckDB at {db_path} {chunk_label}")
    conn = duckdb.connect(database=db_path)
    init_db(conn)

    # Prepare image output directory as <output>/<video_name>/
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_out_dir = os.path.join(output_root, f"__work_{video_name}")
    if first_chunk and os.path.exists(video_out_dir):
        shutil.rmtree(video_out_dir)
    os.makedirs(video_out_dir, exist_ok=True)

    final_index_path = os.path.join(video_out_dir, "_final_embeddings.npy")
    if first_chunk:
        try:
            os.remove(final_index_path)
        except Exception:
            pass

    filtered_dir = os.path.join(video_out_dir, "filtered")
    shutil.rmtree(filtered_dir, ignore_errors=True)
    os.makedirs(filtered_dir, exist_ok=True)
    rejected_root = os.path.join(output_root, "rejected")
    os.makedirs(rejected_root, exist_ok=True)

    global_registry = global_state
    if global_registry is not None:
        global_registry.setdefault("records", [])
        if "mat" not in global_registry:
            global_registry["mat"] = None
    tmp_candidate_dir = os.path.join(
        video_out_dir, f"_candidates_tmp_{int(round(start_sec * 1000.0))}"
    )
    shutil.rmtree(tmp_candidate_dir, ignore_errors=True)
    os.makedirs(tmp_candidate_dir, exist_ok=True)

    if first_chunk:
        clear_previous_video(conn, video_key_aliases)

    max_seconds = float(max_seconds) if max_seconds is not None else None

    candidates: List[
        Tuple[int, float, float, np.ndarray, Optional[str], Optional[np.ndarray], dict, dict, dict, dict]
    ] = []

    total_frames = 1
    progress_log_interval = 1
    start_time = time.time()
    existing_final_mat: Optional[np.ndarray] = None
    if os.path.exists(final_index_path):
        try:
            existing_final_mat = np.load(final_index_path)
        except Exception:
            existing_final_mat = None

    if skip_to_final_stage:
        frame_iter = []
        est_frame_cap = 0
        video_fps = 25.0
    else:
        if not progress_only:
            print(
                f"Extracting frames from {video_path} "
                f"(stride {frame_stride_sec}s, chunk_start={start_sec}, max_seconds={max_seconds})..."
            )
        frame_iter, total_frames_est, video_fps = extract_frames(
            video_path, frame_stride_sec, max_seconds, start_sec=start_sec
        )
        stride_frames = max(1, int(round(frame_stride_sec * video_fps)))
        total_duration_sec = total_frames_est / max(video_fps, 1e-6)
        chunk_end = max_seconds if max_seconds is not None else total_duration_sec
        chunk_span = max(0.0, chunk_end - start_sec)
        est_frame_cap = int((chunk_span * max(video_fps, 1e-6)) // stride_frames) + 1
        if not progress_only:
            print(f"Streaming frames (est. {est_frame_cap} frames within limit)")

        total_frames = max(1, est_frame_cap)
        progress_log_interval = max(1, total_frames // 100)

    def record_summary(status: str, frame_idx: int, ts: float, score_info: Optional[dict], path: Optional[str], reason: Optional[str] = None) -> None:
        if not summary_log_path:
            return
        clip_fmt = _format_score(score_info["clip"]) if score_info and "clip" in score_info else "NA"
        nima_fmt = _format_score(score_info["nima"]) if score_info and "nima" in score_info else "NA"
        piq_fmt = _format_score(score_info["piq"]) if score_info and "piq" in score_info else "NA"
        overall_fmt = _format_score(score_info["overall"]) if score_info and "overall" in score_info else "NA"
        rel_path = os.path.relpath(path, output_root) if path else "NA"
        line = (
            f"{video_name}\tstatus={status}\tframe={frame_idx}\tts={ts:.2f}\t"
            f"clip={clip_fmt}\tnima={nima_fmt}\tpiq={piq_fmt}\toverall={overall_fmt}\tfile={rel_path}"
        )
        if reason:
            line += f"\treason={reason}"
        line += "\n"
        try:
            with open(summary_log_path, "a", encoding="utf-8") as log_f:
                log_f.write(line)
        except Exception:
            pass

    def _build_filename(prefix: str, idx: int, ts: float, score_info: dict, suffix: int | None = None) -> str:
        clip_fmt = _format_score(score_info.get("clip"))
        nima_fmt = _format_score(score_info.get("nima"))
        piq_fmt = _format_score(score_info.get("piq"))
        overall_fmt = _format_score(score_info.get("overall"))
        extra = f"_{suffix}" if suffix is not None else ""
        return (
            f"{prefix}{idx}{extra}_t{ts:.2f}_"
            f"clip-value-{clip_fmt}_nima-value-{nima_fmt}_"
            f"piq-value-{piq_fmt}_overall-rating-{overall_fmt}.jpg"
        )

    def persist_rejected_candidate(candidate, reason: str) -> Optional[str]:
        frame_idx, ts, aest, _emb, tmp_path, stored_image, _tech, _face, _comp, score_info = candidate
        name_idx = frame_idx
        dest_name = _build_filename(f"reject_{reason}_", name_idx, ts, score_info)
        dest_path = os.path.join(rejected_root, dest_name)
        suffix = 1
        while os.path.exists(dest_path):
            dest_name = _build_filename(f"reject_{reason}_", name_idx, ts, score_info, suffix)
            dest_path = os.path.join(rejected_root, dest_name)
            suffix += 1
        moved = False
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.replace(tmp_path, dest_path)
                moved = True
            except Exception:
                moved = False
        if stored_image is not None:
            try:
                cv2.imwrite(dest_path if not moved else dest_path, stored_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            except Exception:
                pass
        record_summary("rejected", name_idx, ts, score_info, dest_path, reason=reason)
        return dest_path

    def persist_rejected_raw(frame_idx: int, ts: float, frame_img: np.ndarray, reason: str, score_info: Optional[dict] = None) -> Optional[str]:
        info = score_info or {"clip": None, "nima": None, "piq": None, "overall": None}
        dest_name = _build_filename(f"reject_{reason}_", frame_idx, ts, info)
        dest_path = os.path.join(rejected_root, dest_name)
        suffix = 1
        while os.path.exists(dest_path):
            dest_name = _build_filename(f"reject_{reason}_", frame_idx, ts, info, suffix)
            dest_path = os.path.join(rejected_root, dest_name)
            suffix += 1
        try:
            cv2.imwrite(dest_path, frame_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        except Exception:
            dest_path = None
        record_summary("rejected", frame_idx, ts, info, dest_path, reason=reason)
        return dest_path

    def save_filtered_frame(
        frame_idx: int,
        ts: float,
        frame_img: np.ndarray,
        reason: str,
        details: Optional[dict] = None,
        score_info: Optional[dict] = None,
    ) -> None:
        slug = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in reason.lower())
        detail_suffix = ""
        if details:
            detail_parts = []
            for key, value in sorted(details.items()):
                if isinstance(value, (np.generic,)):
                    value = float(value)
                detail_key = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(key))
                detail_val = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))
                detail_parts.append(f"{detail_key}-{detail_val}")
            if detail_parts:
                detail_suffix = "__" + "__".join(detail_parts)
        fname_base = f"{frame_idx:09d}_t{ts:.2f}_{slug}{detail_suffix}"
        image_path = os.path.join(filtered_dir, f"{fname_base}.jpg")
        try:
            cv2.imwrite(image_path, frame_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        except Exception:
            image_path = ""
        persist_rejected_raw(frame_idx, ts, frame_img, reason, score_info)

    stride_sec = max(frame_stride_sec, 1e-6)

    def skin_probability(frame: np.ndarray) -> float:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 48, 80), (20, 255, 255))
        return float(mask.mean() / 255.0)

    def should_run_face(frame: np.ndarray, precomputed_skin: Optional[float] = None) -> bool:
        if not enable_face_checks or FaceAnalysis is None:
            return False
        ratio = precomputed_skin if precomputed_skin is not None else skin_probability(frame)
        return ratio >= face_skin_threshold

    def looks_like_overlay_card(
        frame: np.ndarray,
        coverage_ratio: float,
        sat_threshold: float,
        sat_ratio_threshold: float,
        unique_threshold: float,
        hue_dominance: float,
        color_ratio_threshold: float,
    ) -> bool:
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0 or coverage_ratio <= 0.0:
            return False
        sections: List[np.ndarray] = []
        seg_w = int(max(1, min(w, w * coverage_ratio)))
        seg_h = int(max(1, min(h, h * 0.20)))
        try:
            sections.append(frame[:, :seg_w])
            sections.append(frame[:, w - seg_w :])
            sections.append(frame[:seg_h, :])
            sections.append(frame[h - seg_h :, :])
        except Exception:
            return False
        for seg in sections:
            if seg.size == 0:
                continue
            try:
                small = cv2.resize(seg, (64, 64))
                hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            except Exception:
                continue
            sat = hsv[..., 1] / 255.0
            high_sat_ratio = float(np.mean(sat >= sat_threshold))
            if high_sat_ratio < sat_ratio_threshold:
                continue
            # approximate uniform color by counting unique colors on downsampled region
            reshaped = small.reshape(-1, 3)
            try:
                unique = np.unique(reshaped, axis=0).shape[0]
            except Exception:
                continue
            unique_ratio = float(unique) / float(max(1, reshaped.shape[0]))
            hue = hsv[..., 0]
            try:
                hist, _ = np.histogram(hue, bins=12, range=(0, 180))
                dom_ratio = float(hist.max()) / float(max(1, hist.sum()))
            except Exception:
                dom_ratio = 0.0
            h_channel = hsv[..., 0]
            sat_mask = sat >= max(0.3, sat_threshold * 0.8)
            blue_mask = sat_mask & ((h_channel >= 80) & (h_channel <= 140))
            yellow_mask = sat_mask & ((h_channel >= 15) & (h_channel <= 55))
            color_ratio = float(np.mean(blue_mask | yellow_mask))
            if (
                unique_ratio <= unique_threshold
                or dom_ratio >= hue_dominance
                or color_ratio >= color_ratio_threshold
            ):
                return True
        return False

    def process_frame(
        frame_idx: int,
        ts: float,
        frame_bgr: np.ndarray,
        frame_skin_prob: Optional[float] = None,
        frame_motion_score: Optional[float] = None,
    ) -> Optional[tuple]:
        original_frame = frame_bgr
        proc_frame = frame_bgr
        skin_prob = frame_skin_prob if frame_skin_prob is not None else skin_probability(proc_frame)
        tech = compute_technical_quality(proc_frame)
        long_edge = max(tech["resolution_w"], tech["resolution_h"])
        if long_edge < MIN_RES_LONG_EDGE or tech["total_pixels"] < MIN_TOTAL_PIXELS:
            persist_rejected_raw(frame_idx, ts, original_frame, "low_resolution")
            return None

        if tech["tech_score"] < min_tech_score:
            persist_rejected_raw(frame_idx, ts, original_frame, "low_tech_score")
            return None

        if is_black_slate(
            cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0,
            tech["mean_luma"],
            black_frame_threshold,
            black_frame_ratio,
        ):
            persist_rejected_raw(frame_idx, ts, original_frame, "black_frame")
            return None

        run_face = should_run_face(proc_frame, skin_prob)
        if run_face:
            face = compute_face_features(proc_frame)
        else:
            face = {
                "num_faces": 0,
                "has_face": False,
                "main_face_bbox": None,
                "face_sharpness_var": 0.0,
                "face_quality_score": 0.0,
                "main_face_rel_area": 0.0,
            }

        faceless_candidate = False
        face_conf = float(face.get("main_face_conf", 0.0))
        face_has_detection = bool(face.get("has_face", False))
        motion_val = frame_motion_score if frame_motion_score is not None else None
        if require_face:
            if face_has_detection and face_conf < faceless_confidence_threshold:
                face_has_detection = False
            if not face_has_detection:
                secondary = detect_secondary_face(proc_frame)
                if secondary is not None:
                    x_min, y_min, x_max, y_max = secondary["bbox"]
                    face.update(
                        {
                            "has_face": True,
                            "main_face_bbox": secondary["bbox"],
                            "main_face_rel_area": float(secondary["rel_area"]),
                            "face_sharpness_var": max(
                                float(face.get("face_sharpness_var", 0.0)),
                                float(secondary.get("sharpness", 0.0)),
                            ),
                            "main_face_conf": max(face_conf, faceless_confidence_threshold),
                        }
                    )
                    face_has_detection = True
                elif allow_faceless_scenes:
                    faceless_candidate = True
                else:
                    return None
        else:
            face_has_detection = face.get("has_face", False)

        if faceless_candidate:
            scenic_ok = True
            if allow_faceless_scenic_only:
                scenic_ok = bool(
                    (skin_prob <= faceless_max_skin_ratio)
                    or (motion_val is not None and motion_val <= faceless_motion_threshold)
                )
                if skin_prob > faceless_max_skin_ratio and (
                    motion_val is None or motion_val > faceless_motion_threshold
                ):
                    scenic_ok = False
                if scenic_ok:
                    people_ratio = estimate_person_ratio(proc_frame)
                    if people_ratio > faceless_people_ratio:
                        scenic_ok = False
                if not scenic_ok:
                    persist_rejected_raw(frame_idx, ts, original_frame, "faceless_not_scenic")
                    return None
        elif not face_has_detection and require_face:
            persist_rejected_raw(frame_idx, ts, original_frame, "no_face")
            return None

        if face.get("has_face", False):
            face_sharpness = face.get("face_sharpness_var", 0.0)
            if face_sharpness < face_sharpness_threshold:
                save_filtered_frame(
                    frame_idx,
                    ts,
                    original_frame,
                    "face_too_blurry",
                    {
                        "face_sharpness_var": face_sharpness,
                        "threshold": face_sharpness_threshold,
                    },
                )
                return None
            tilt = abs(float(face.get("face_tilt_deg", 0.0)))
            if tilt > max_face_tilt:
                save_filtered_frame(
                    frame_idx,
                    ts,
                    original_frame,
                    "face_tilted",
                    {
                        "face_tilt_deg": tilt,
                        "max_face_tilt": max_face_tilt,
                    },
                )
                return None
            face_area = max(0.0, float(face.get("main_face_rel_area", 0.0)))
            if face_area < max(0.0, float(min_face_area_ratio)):
                save_filtered_frame(
                    frame_idx,
                    ts,
                    original_frame,
                    "face_too_small",
                    {
                        "face_area": face_area,
                        "threshold": min_face_area_ratio,
                    },
                )
                return None
            bbox = face.get("main_face_bbox")
            if bbox is not None:
                x_min, y_min, x_max, y_max = bbox
                h, _w = proc_frame.shape[:2]
                center_y = ((y_min + y_max) * 0.5) / max(h, 1)
            else:
                center_y = 0.5
            if center_y < min_face_y_center:
                save_filtered_frame(
                    frame_idx,
                    ts,
                    original_frame,
                    "face_too_low",
                    {
                        "center_y": center_y,
                        "threshold": min_face_y_center,
                    },
                )
                return None

        if face.get("has_face", False):
            face_sharpness = face.get("face_sharpness_var", 0.0)
            if face_sharpness < face_sharpness_threshold:
                persist_rejected_raw(frame_idx, ts, original_frame, "face_too_blurry")
                return None

        has_smile = False
        if face.get("has_face") and face.get("main_face_bbox"):
            x_min, y_min, x_max, y_max = face["main_face_bbox"]
            h, w = proc_frame.shape[:2]
            x_min = max(0, min(w, x_min))
            y_min = max(0, min(h, y_min))
            x_max = max(0, min(w, x_max))
            y_max = max(0, min(h, y_max))
            if x_max > x_min and y_max > y_min:
                face_patch = proc_frame[y_min:y_max, x_min:x_max]
                has_smile = detect_smile(face_patch)
        face["has_smile"] = has_smile

        if skip_overlay_cards and looks_like_overlay_card(
            proc_frame,
            coverage_ratio=overlay_coverage,
            sat_threshold=overlay_sat_threshold,
            sat_ratio_threshold=overlay_sat_ratio,
            unique_threshold=overlay_unique_threshold,
            hue_dominance=overlay_hue_dominance,
            color_ratio_threshold=overlay_color_ratio,
        ):
            persist_rejected_raw(frame_idx, ts, original_frame, "overlay_card")
            return None

        if tech["sharpness_var"] < MIN_SHARPNESS_VAR and run_face:
            face_sharpness = face.get("face_sharpness_var", 0.0)
            if not (face.get("has_face", False) and face_sharpness >= (0.5 * MIN_SHARPNESS_VAR)):
                persist_rejected_raw(frame_idx, ts, original_frame, "global_blur")
                return None

        comp = compute_composition_features(proc_frame, face_info=face)
        nima_score: Optional[float] = None
        piq_score: Optional[float] = None
        overall_rating: Optional[float] = None
        if use_aesthetic:
            aest = compute_aesthetic_score(proc_frame)
            threshold = faceless_min_aesthetic if faceless_candidate else float(min_aesthetic)
            if aest < float(threshold):
                return None
            nima_score = compute_nima_score(proc_frame)
            piq_score = compute_piq_score(proc_frame)
            overall_rating = compute_overall_rating(aest, nima_score, piq_score)
        else:
            aest = 0.0
            if faceless_candidate:
                persist_rejected_raw(frame_idx, ts, original_frame, "faceless_without_aesthetic")
                return None
        emb = compute_embedding(proc_frame)
        score_info = {
            "clip": float(aest),
            "nima": nima_score,
            "piq": piq_score,
            "overall": overall_rating,
        }
        if use_aesthetic and not passes_selection_thresholds(score_info):
            persist_rejected_raw(frame_idx, ts, original_frame, "score", score_info)
            return None

        tmp_fname = f"{frame_idx:09d}_t{ts:.2f}.jpg"
        tmp_path = os.path.join(tmp_candidate_dir, tmp_fname)
        stored_image: Optional[np.ndarray] = None
        try:
            ok = cv2.imwrite(tmp_path, original_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not ok:
                tmp_path = ""
        except Exception:
            tmp_path = ""
        if not tmp_path:
            stored_image = original_frame.copy()

        return (frame_idx, ts, float(aest), emb, tmp_path, stored_image, tech, face, comp, score_info)

    workers = max(1, max_workers)
    executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 and not skip_to_final_stage else None
    futures = []
    prev_frame = None

    if not skip_to_final_stage:
        for i, (frame_idx, ts, frame_bgr) in enumerate(frame_iter, start=1):
            if not progress_only:
                print(f"[{i}/{total_frames}] frame_index={frame_idx}, t={ts:.2f}s")
            elif i == 1 or (i % progress_log_interval == 0) or i == total_frames:
                pct = (float(i) / float(total_frames)) * 100.0
                elapsed = time.time() - start_time
                total_elapsed = time.time() - overall_start
                remaining = (elapsed / max(i, 1)) * (total_frames - i)
                try:
                    print(
                        f"\r[prog] frames {i}/{total_frames} ({pct:.1f}%) "
                        f"elapsed={elapsed:.1f}s total={total_elapsed:.1f}s "
                        f"left={max(0.0, remaining):.1f}s",
                        end="" if i < total_frames else "\n",
                        flush=True,
                    )
                except Exception:
                    pass

            if max_seconds is not None and ts >= max_seconds:
                break

            motion_score = None
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, frame_bgr)
                motion_score = float(diff.mean())
                if adaptive_stride and motion_score < adaptive_stride_threshold:
                    prev_frame = frame_bgr
                    continue
            prev_frame = frame_bgr

            frame_skin_prob = skin_probability(frame_bgr)
            frame_payload = frame_bgr.copy() if executor else frame_bgr
            if executor:
                futures.append(
                    executor.submit(
                        process_frame,
                        frame_idx,
                        ts,
                        frame_payload,
                        frame_skin_prob,
                        motion_score,
                    )
                )
            else:
                result = process_frame(frame_idx, ts, frame_payload, frame_skin_prob, motion_score)
                if result:
                    candidates.append(result)
                    if max_frames is not None and len(candidates) >= max_frames:
                        break

    filter_stage_start = time.time()
    if executor:
        total_jobs = len(futures)
        completed_jobs = 0
        log_every = max(1, total_jobs // 100)
        for future in as_completed(futures):
            result = future.result()
            if result:
                candidates.append(result)
                if max_frames is not None and len(candidates) >= max_frames:
                    break
            completed_jobs += 1
            if progress_only and (completed_jobs == total_jobs or completed_jobs % log_every == 0):
                pct = (completed_jobs / max(1, total_jobs)) * 100.0
                stage_elapsed = time.time() - filter_stage_start
                total_elapsed = time.time() - overall_start
                try:
                    print(
                        f"\r[prog] filtering {completed_jobs}/{total_jobs} ({pct:.1f}%) "
                        f"elapsed={stage_elapsed:.1f}s total={total_elapsed:.1f}s",
                        end="" if completed_jobs < total_jobs else "\n",
                        flush=True,
                    )
                except Exception:
                    pass
        executor.shutdown(wait=True)

    if hasattr(frame_iter, "close") and not skip_to_final_stage:
        try:
            frame_iter.close()  # type: ignore[attr-defined]
        except Exception:
            pass

    # Sort by aesthetic descending so highest quality considered first
    candidates.sort(key=lambda x: x[2], reverse=True)

    kept: List[
        Tuple[int, float, float, np.ndarray, Optional[str], Optional[np.ndarray], dict, dict, dict, dict]
    ] = []
    kept_embs: List[np.ndarray] = []
    effective_threshold = max(0.0, float(dedup_threshold))
    # Vectorized store of kept embeddings for fast similarity checks
    kept_mat: Optional[np.ndarray] = None

    def _quality_score(aesthetic_value: float, score_info: Optional[dict]) -> float:
        if score_info and isinstance(score_info.get("overall"), (int, float)):
            try:
                return float(score_info["overall"])
            except Exception:
                pass
        return float(aesthetic_value)

    dedup_stage_start = time.time()
    total_candidates = max(1, len(candidates))
    dedup_log_every = max(1, total_candidates // 100)
    for idx, cand in enumerate(candidates, start=1):
        frame_idx, ts, aest, emb, tmp_path, stored_image, tech, face, comp, score_info = cand
        # If dedup disabled or nothing kept yet, accept immediately
        if effective_threshold <= 0.0 or kept_mat is None or kept_mat.size == 0:
            kept.append(cand)
            if kept_mat is None:
                kept_mat = emb.reshape(1, -1).astype(np.float32, copy=False)
            else:
                kept_mat = np.vstack([kept_mat, emb])
            continue

        # Vectorized cosine similarity (embeddings are L2-normalized)
        try:
            sims = kept_mat.dot(emb)  # shape (K,)
            if sims.size:
                best_idx = int(np.argmax(sims))
                max_sim = float(sims[best_idx])
            else:
                best_idx = -1
                max_sim = -1.0
        except Exception:
            sims = [float(np.dot(k, emb)) for k in kept_mat]  # type: ignore[arg-type]
            if sims:
                max_sim = max(sims)
                best_idx = sims.index(max_sim)
            else:
                max_sim = -1.0
                best_idx = -1
        min_dist = 1.0 - max_sim
        if min_dist < effective_threshold and best_idx >= 0:
            existing_entry = kept[best_idx]
            existing_face = existing_entry[7]
            existing_smile = bool(existing_face.get("has_smile"))
            candidate_smile = bool(face.get("has_smile"))
            replace_existing = False
            candidate_quality = _quality_score(aest, score_info)
            existing_quality = _quality_score(existing_entry[2], existing_entry[9])
            if prefer_smiles and candidate_smile and not existing_smile:
                replace_existing = True
            elif prefer_smiles and existing_smile and not candidate_smile:
                replace_existing = False
            elif candidate_quality > existing_quality:
                replace_existing = True

            if replace_existing and best_idx >= 0:
                persist_rejected_candidate(kept[best_idx], "replaced")
                kept[best_idx] = cand
                kept_mat[best_idx] = emb
            else:
                persist_rejected_candidate(cand, "dedup")
            continue

        kept.append(cand)
        kept_mat = np.vstack([kept_mat, emb])

        if progress_only and (idx == total_candidates or idx % dedup_log_every == 0):
            pct = (idx / max(1, total_candidates)) * 100.0
            stage_elapsed = time.time() - dedup_stage_start
            total_elapsed = time.time() - overall_start
            try:
                print(
                    f"\r[prog] dedup {idx}/{total_candidates} ({pct:.1f}%) "
                    f"elapsed={stage_elapsed:.1f}s total={total_elapsed:.1f}s",
                    end="" if idx < total_candidates else "\n",
                    flush=True,
                )
            except Exception:
                pass

    final_removed: List[
        Tuple[int, float, float, np.ndarray, Optional[str], Optional[np.ndarray], dict, dict, dict, dict]
    ] = []
    new_final_embeddings: List[np.ndarray] = []
    resolved_final_threshold = final_dedup_threshold if final_dedup_threshold is not None else dedup_threshold
    resolved_final_threshold = max(0.0, float(resolved_final_threshold))
    global_final_mat: Optional[np.ndarray] = None
    if existing_final_mat is not None and existing_final_mat.size > 0:
        try:
            global_final_mat = existing_final_mat.astype(np.float32, copy=False)
        except Exception:
            global_final_mat = existing_final_mat

    def global_duplicate_decision(embedding: np.ndarray, score_info: dict, aest_value: float, face_info: dict) -> tuple[str, Optional[int]]:
        if global_registry is None or resolved_final_threshold <= 0.0:
            return ("keep", None)
        global_mat = global_registry.get("mat")
        records = global_registry.get("records", [])
        if global_mat is None or global_mat.size == 0 or not records:
            return ("keep", None)
        try:
            sims = global_mat.dot(embedding)
        except Exception:
            sims = []
        if sims is None or len(sims) == 0:
            return ("keep", None)
        best_idx = int(np.argmax(sims))
        max_sim = float(sims[best_idx])
        min_dist = 1.0 - max_sim
        if min_dist >= resolved_final_threshold or best_idx >= len(records):
            return ("keep", None)
        existing_record = records[best_idx]
        existing_quality = existing_record.get(
            "quality",
            _quality_score(existing_record.get("aesthetic", 0.0), existing_record.get("score_info")),
        )
        candidate_quality = _quality_score(aest_value, score_info)
        existing_smile = bool(existing_record.get("has_smile"))
        candidate_smile = bool(face_info.get("has_smile"))
        if prefer_smiles:
            if candidate_smile and not existing_smile:
                return ("replace", best_idx)
            if existing_smile and not candidate_smile:
                return ("skip", best_idx)
        if candidate_quality > existing_quality:
            return ("replace", best_idx)
        return ("skip", best_idx)

    def move_existing_selection_to_rejected(record: dict, tag: str) -> Optional[str]:
        path = record.get("path")
        if not path or not os.path.exists(path):
            return None
        base = os.path.basename(path)
        dest_name = f"{tag}_{base}"
        dest_path = os.path.join(rejected_root, dest_name)
        suffix = 1
        while os.path.exists(dest_path):
            dest_path = os.path.join(rejected_root, f"{tag}_{suffix}_{base}")
            suffix += 1
        try:
            os.replace(path, dest_path)
        except Exception:
            return path
        return dest_path

    def register_global_selection(
        dest_path: str,
        embedding: np.ndarray,
        frame_idx: int,
        ts: float,
        aest_value: float,
        score_info: dict,
        face_info: dict,
        replace_idx: Optional[int] = None,
    ) -> None:
        if global_registry is None:
            return
        record = {
            "path": dest_path,
            "frame_idx": frame_idx,
            "ts": ts,
            "video": video_name,
            "score_info": dict(score_info) if score_info is not None else {},
            "quality": _quality_score(aest_value, score_info),
            "aesthetic": float(aest_value),
            "has_smile": bool(face_info.get("has_smile")),
        }
        if replace_idx is None:
            mat = global_registry.get("mat")
            if mat is None:
                global_registry["mat"] = embedding.reshape(1, -1).astype(np.float32, copy=False)
            else:
                global_registry["mat"] = np.vstack([mat, embedding])
            global_registry.setdefault("records", []).append(record)
        else:
            if global_registry.get("mat") is not None:
                global_registry["mat"][replace_idx] = embedding
            else:
                global_registry["mat"] = embedding.reshape(1, -1).astype(np.float32, copy=False)
            global_registry["records"][replace_idx] = record

    if resolved_final_threshold > 0.0 and kept:
        final_kept_list: List[
            Tuple[int, float, float, np.ndarray, Optional[str], Optional[np.ndarray], dict, dict, dict, dict]
        ] = []
        chunk_final_mat: Optional[np.ndarray] = None
        for cand in kept:
            frame_idx, ts, aest, emb, tmp_path, stored_image, tech, face, comp, score_info = cand
            duplicate = False
            if global_final_mat is not None and global_final_mat.size > 0:
                try:
                    sims_prev = global_final_mat.dot(emb)
                    if sims_prev.size:
                        prev_dist = 1.0 - float(np.max(sims_prev))
                        if prev_dist < resolved_final_threshold:
                            duplicate = True
                except Exception:
                    duplicate = False
            if duplicate:
                final_removed.append(cand)
                continue
            replaced_chunk_entry = False
            chunk_best_idx = -1
            if chunk_final_mat is not None and chunk_final_mat.size > 0:
                try:
                    sims_chunk = chunk_final_mat.dot(emb)
                    if sims_chunk.size:
                        chunk_best_idx = int(np.argmax(sims_chunk))
                        chunk_dist = 1.0 - float(sims_chunk[chunk_best_idx])
                        if chunk_dist < resolved_final_threshold and chunk_best_idx >= 0:
                            existing_entry = final_kept_list[chunk_best_idx]
                            existing_face = existing_entry[7]
                            existing_smile = bool(existing_face.get("has_smile"))
                            candidate_smile = bool(face.get("has_smile"))
                            existing_quality = _quality_score(existing_entry[2], existing_entry[9])
                            candidate_quality = _quality_score(aest, score_info)
                            if prefer_smiles and candidate_smile and not existing_smile:
                                replaced_chunk_entry = True
                            elif prefer_smiles and existing_smile and not candidate_smile:
                                replaced_chunk_entry = False
                            elif candidate_quality > existing_quality:
                                replaced_chunk_entry = True
                            if not replaced_chunk_entry:
                                final_removed.append(cand)
                                continue
                except Exception:
                    pass
            if replaced_chunk_entry and chunk_best_idx >= 0:
                final_removed.append(final_kept_list[chunk_best_idx])
                final_kept_list[chunk_best_idx] = cand
                chunk_final_mat[chunk_best_idx] = emb
                new_final_embeddings.append(emb)
            else:
                final_kept_list.append(cand)
                new_final_embeddings.append(emb)
                if chunk_final_mat is None:
                    chunk_final_mat = emb.reshape(1, -1).astype(np.float32, copy=False)
                else:
                    chunk_final_mat = np.vstack([chunk_final_mat, emb])
        kept = final_kept_list
    else:
        for cand in kept:
            new_final_embeddings.append(cand[3])

    # Persist kept frames in timestamp order for readability
    kept.sort(key=lambda x: x[1])

    total_kept = len(kept)
    progress_step = max(1, total_kept // 25)

    # Batch DB writes in a single transaction for speed
    try:
        conn.execute("BEGIN TRANSACTION;")
    except Exception:
        pass

    write_stage_start = time.time()
    for rank, (frame_idx, ts, aest, emb, tmp_path, stored_image, tech, face, comp, score_info) in enumerate(kept, start=1):
        nima_score = score_info.get("nima")
        piq_score = score_info.get("piq")
        overall_rating = score_info.get("overall")
        insert_frame_row(
            conn=conn,
            video_path=video_db_key,
            frame_index=frame_idx,
            timestamp_sec=ts,
            aesthetic=aest,
            nima_score=nima_score,
            piq_score=piq_score,
            overall_rating=overall_rating,
            tech_score=tech["tech_score"],
            sharpness_var=tech["sharpness_var"],
            noise_sigma=tech["noise_sigma"],
            clip_pct_shadows=tech["clip_pct_shadows"],
            clip_pct_highlights=tech["clip_pct_highlights"],
            mean_luma=tech["mean_luma"],
            color_cast_score=tech["color_cast_score"],
            num_faces=face.get("num_faces", 0),
            has_face=face.get("has_face", False),
            face_quality_score=face.get("face_quality_score", 0.0),
            main_face_rel_area=face.get("main_face_rel_area", 0.0),
            composition_score=comp.get("composition_score", 0.0),
            subject_x_norm=comp.get("subject_x_norm", 0.5),
            subject_y_norm=comp.get("subject_y_norm", 0.5),
            embedding=emb,
        )

        current_candidate = (frame_idx, ts, aest, emb, tmp_path, stored_image, tech, face, comp, score_info)
        action, dup_idx = global_duplicate_decision(emb, score_info, aest, face)
        replace_idx = None
        if action == "skip":
            if dup_idx is not None:
                reject_path = persist_rejected_candidate(current_candidate, "global_dedup")
                dup_path = None
                if global_registry and dup_idx < len(global_registry.get("records", [])):
                    dup_path = global_registry["records"][dup_idx].get("path")
                record_summary(
                    "rejected",
                    frame_idx,
                    ts,
                    score_info,
                    reject_path,
                    reason=f"global_dup_of:{dup_path}",
                )
                continue
        elif action == "replace" and dup_idx is not None and global_registry:
            replace_idx = dup_idx
            existing_record = global_registry.get("records", [])[dup_idx]
            moved_path = move_existing_selection_to_rejected(
                existing_record,
                f"global_replaced_{frame_idx:09d}",
            )
            record_summary(
                "global_removed",
                existing_record.get("frame_idx", -1),
                existing_record.get("ts", 0.0),
                existing_record.get("score_info"),
                moved_path,
                reason=f"replaced_by:{video_name}:{frame_idx}",
            )

        clip_fmt = _format_score(score_info.get("clip"))
        nima_fmt = _format_score(nima_score)
        piq_fmt = _format_score(piq_score)
        overall_fmt = _format_score(overall_rating)
        base_name = (
            f"out{rank}_t{ts:.2f}_"
            f"clip-value-{clip_fmt}_"
            f"nima-value-{nima_fmt}_"
            f"piq-value-{piq_fmt}_"
            f"overall-rating-{overall_fmt}.jpg"
        )
        dest_name = base_name
        dest_path = os.path.join(output_root, dest_name)
        suffix = 1
        while os.path.exists(dest_path):
            dest_name = (
                f"out{rank}_{suffix}_t{ts:.2f}_"
                f"clip-value-{clip_fmt}_"
                f"nima-value-{nima_fmt}_"
                f"piq-value-{piq_fmt}_"
                f"overall-rating-{overall_fmt}.jpg"
            )
            dest_path = os.path.join(output_root, dest_name)
            suffix += 1
        moved = False
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.replace(tmp_path, dest_path)
                moved = True
            except Exception:
                moved = False
        if not moved:
            if stored_image is not None:
                try:
                    cv2.imwrite(dest_path, stored_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                except Exception:
                    pass
            elif tmp_path and os.path.exists(tmp_path):
                try:
                    img = cv2.imread(tmp_path)
                    if img is not None:
                        cv2.imwrite(dest_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                except Exception:
                    pass
        record_summary("selected", frame_idx, ts, score_info, dest_path)
        register_global_selection(dest_path, emb, frame_idx, ts, aest, score_info, face, replace_idx=replace_idx)

        if progress_only and (rank == 1 or rank == total_kept or rank % progress_step == 0):
            pct = (rank / max(1, total_kept)) * 100.0
            stage_elapsed = time.time() - write_stage_start
            total_elapsed = time.time() - overall_start
            try:
                print(
                    f"\r[prog] writing kept frames {rank}/{total_kept} ({pct:.1f}%) "
                    f"elapsed={stage_elapsed:.1f}s total={total_elapsed:.1f}s",
                    end="" if rank < total_kept else "\n",
                    flush=True,
                )
            except Exception:
                pass

    try:
        conn.execute("COMMIT;")
    except Exception:
        pass

    # Clean up temporary files for duplicates discarded in final stage
    for dup in final_removed:
        persist_rejected_candidate(dup, "final_dedup")

    if progress_only:
        try:
            print()
        except Exception:
            pass

    shutil.rmtree(tmp_candidate_dir, ignore_errors=True)
    conn.close()

    if new_final_embeddings:
        try:
            new_final_mat_arr = np.vstack(new_final_embeddings).astype(np.float32, copy=False)
            if existing_final_mat is not None and existing_final_mat.size > 0:
                combined = np.vstack([existing_final_mat, new_final_mat_arr])
            else:
                combined = new_final_mat_arr
            np.save(final_index_path, combined)
        except Exception:
            pass
    elif existing_final_mat is not None and os.path.exists(final_index_path):
        # keep existing embeddings file as-is
        pass

    total_time = time.time() - overall_start
    print(
        f"Done. kept={len(kept)} (from {len(candidates)} candidates after filtering, "
        f"final_removed={len(final_removed)}). Total time: {total_time:.1f}s"
    )


def collect_video_inputs(input_path: str) -> List[str]:
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        collected: List[str] = []
        for root, _, files in os.walk(input_path):
            for fname in files:
                if fname.lower().endswith(VIDEO_EXTENSIONS):
                    collected.append(os.path.join(root, fname))
        collected.sort()
        return collected
    raise FileNotFoundError(f"Input path not found: {input_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract keyframes, compute embeddings and CLIP+LAION aesthetic scores, store in DuckDB."
    )
    parser.add_argument("video", help="Path to input video file or directory of videos")
    parser.add_argument(
        "--db",
        default="frames.duckdb",
        help="Path to DuckDB database file (default: frames.duckdb)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output root folder; images are saved under <output>/<video_name>/",
    )
    parser.add_argument(
        "--stride-sec",
        type=float,
        default=1.0,
        help="Frame stride in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on number of frames to process",
    )
    parser.add_argument(
        "--max-seconds",
        "--max-duration",
        dest="max_seconds",
        type=float,
        default=None,
        help="Optional cap on processed seconds per video",
    )
    parser.add_argument(
        "--chunk-duration-secs",
        type=float,
        default=300.0,
        help="Process videos in chunks of this many seconds (set <=0 to process in a single pass)",
    )
    parser.add_argument(
        "--min-aesthetic",
        type=float,
        default=0.0,
        help="Discard frames with aesthetic score below this threshold (default: 0.0)",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.15,
        help="Cosine distance threshold to treat two frames as duplicates (default: 0.15)",
    )
    parser.add_argument(
        "--final-dedup-threshold",
        type=float,
        default=None,
        help="Threshold for final-stage deduplication before saving images (default: inherits --dedup-threshold)",
    )
    parser.add_argument(
        "--progress-only",
        action="store_true",
        help="Suppress per-frame logs and show a compact progress meter",
    )
    parser.add_argument(
        "--min-tech-score",
        type=float,
        default=MIN_TECH_SCORE,
        help=f"Discard frames with technical score below this threshold (default: {MIN_TECH_SCORE})",
    )
    parser.add_argument("--disable-face", action="store_true", help="Skip InsightFace checks entirely")
    parser.add_argument("--adaptive-stride", action="store_true", help="Skip stride frames with low motion")
    parser.add_argument(
        "--adaptive-stride-threshold",
        type=float,
        default=3.0,
        help="Mean absolute difference threshold for adaptive stride skipping (default: 3.0)",
    )
    parser.add_argument(
        "--face-skin-threshold",
        type=float,
        default=0.05,
        help="Minimum skin probability to run face detection when enabled",
    )
    parser.add_argument(
        "--max-face-tilt",
        type=float,
        default=MAX_FACE_TILT_DEGREES,
        help=f"Discard frames when main face tilt exceeds this angle in degrees (default: {MAX_FACE_TILT_DEGREES})",
    )
    parser.add_argument(
        "--face-sharpness-threshold",
        type=float,
        default=MIN_FACE_SHARPNESS,
        help=f"Minimum Laplacian variance for the main face before accepting a frame (default: {MIN_FACE_SHARPNESS})",
    )
    parser.add_argument(
        "--min-face-area-ratio",
        type=float,
        default=0.0,
        help="Minimum relative frame area (0-1) that the detected main face must cover before accepting the frame",
    )
    parser.add_argument(
        "--min-face-y-center",
        type=float,
        default=0.0,
        help="Require the detected main face vertical center (0=top,1=bottom) to be at least this value",
    )
    parser.add_argument(
        "--black-frame-threshold",
        type=float,
        default=0.05,
        help="Minimum ratio of mid-tone pixels to accept (filters black slates)",
    )
    parser.add_argument(
        "--black-frame-mean",
        type=float,
        default=0.05,
        help="Skip frames whose mean luma is below this when paired with low mid-tone ratio",
    )
    parser.add_argument(
        "--require-face",
        action="store_true",
        help="Discard frames that do not contain a detected face",
    )
    parser.add_argument(
        "--allow-faceless-scenes",
        action="store_true",
        help="When --require-face is set, still keep frames without faces if they exceed --faceless-min-aesthetic",
    )
    parser.add_argument(
        "--allow-faceless-scenic-only",
        action="store_true",
        help="Only allow faceless scenes when they look scenic (low skin coverage or low motion)",
    )
    parser.add_argument(
        "--faceless-min-aesthetic",
        type=float,
        default=5.0,
        help="Minimum aesthetic score required to keep a frame without faces when --allow-faceless-scenes is enabled (default: 5.0)",
    )
    parser.add_argument(
        "--faceless-max-skin-ratio",
        type=float,
        default=0.05,
        help="Maximum skin-coverage ratio allowed when accepting faceless frames",
    )
    parser.add_argument(
        "--faceless-motion-threshold",
        type=float,
        default=2.0,
        help="Maximum mean inter-frame difference to treat a faceless frame as low motion",
    )
    parser.add_argument(
        "--faceless-max-confidence",
        type=float,
        default=0.3,
        help="Only fall back to faceless handling when InsightFace confidence is below this threshold",
    )
    parser.add_argument(
        "--faceless-max-person-ratio",
        type=float,
        default=0.05,
        help="Maximum silhouette ratio from the pedestrian detector when accepting faceless frames",
    )
    parser.add_argument(
        "--skip-overlay-cards",
        action="store_true",
        help="Discard frames dominated by saturated overlay cards (news tickers, slides)",
    )
    parser.add_argument(
        "--overlay-coverage",
        type=float,
        default=0.3,
        help="Fraction of width/height to inspect for overlay cards when enabled (default: 0.3)",
    )
    parser.add_argument(
        "--overlay-sat-threshold",
        type=float,
        default=0.5,
        help="Pixel saturation cutoff for overlay detection (default: 0.5)",
    )
    parser.add_argument(
        "--overlay-sat-ratio",
        type=float,
        default=0.6,
        help="Minimum ratio of saturated pixels required for overlay detection (default: 0.6)",
    )
    parser.add_argument(
        "--overlay-unique-threshold",
        type=float,
        default=0.2,
        help="Maximum color diversity for overlay detection (default: 0.2)",
    )
    parser.add_argument(
        "--overlay-hue-dominance",
        type=float,
        default=0.7,
        help="Minimum dominant hue ratio for overlay detection (default: 0.7)",
    )
    parser.add_argument(
        "--overlay-color-ratio",
        type=float,
        default=0.4,
        help="Minimum ratio of vivid blue/yellow pixels to treat as an overlay card (default: 0.4)",
    )
    parser.add_argument(
        "--skip-to-final-stage",
        action="store_true",
        help="Skip new frame processing and only run final deduplication/output steps",
    )
    parser.add_argument(
        "--no-prefer-smiles",
        action="store_true",
        help="Disable preference for smiling faces when resolving duplicates",
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of threads for preprocessing")
    parser.add_argument(
        "--disable-aesthetic",
        action="store_true",
        help="Skip CLIP aesthetic scoring (min-aesthetic ignored)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    video_inputs = collect_video_inputs(args.video)
    if not video_inputs:
        raise FileNotFoundError(f"No video files found in directory: {args.video}")

    total_videos = len(video_inputs)
    chunk_duration = float(getattr(args, "chunk_duration_secs", 300.0))

    output_root = os.path.abspath(args.output)
    args.output = output_root
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)
    summary_log = os.path.join(output_root, "extraction_summary.txt")
    with open(summary_log, "w", encoding="utf-8") as log_f:
        log_f.write("video\tstatus\tframe\tts\tclip\tnima\tpiq\toverall\tfile\treason\n")

    global_state = {"mat": None, "records": []}

    for idx, video_path in enumerate(video_inputs, start=1):
        video_msg = f"video {idx}/{total_videos}: {video_path}"
        try:
            if args.progress_only:
                print(f"[prog] {video_msg}")
            else:
                print(f"[info] {video_msg}")
        except Exception:
            pass

        duration_sec = get_video_duration(video_path)
        max_seconds = getattr(args, "max_seconds", None)
        chunk_plan: List[tuple[float, float | None]] = []
        if chunk_duration <= 0:
            chunk_plan.append((0.0, float(max_seconds) if max_seconds is not None else None))
        else:
            target_end = duration_sec
            if max_seconds is not None:
                target_end = min(target_end, float(max_seconds))
            if target_end <= 0:
                chunk_plan.append((0.0, float(max_seconds) if max_seconds is not None else None))
            else:
                start_sec = 0.0
                while start_sec < target_end - 1e-6:
                    chunk_end = min(target_end, start_sec + chunk_duration)
                    chunk_plan.append((start_sec, chunk_end))
                    if chunk_end >= target_end:
                        break
                    start_sec = chunk_end
                if not chunk_plan:
                    chunk_plan.append((0.0, float(max_seconds) if max_seconds is not None else None))

        prefix = f"[{idx}/{total_videos}] " if not args.progress_only and total_videos > 1 else ""

        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_plan, start=1):
            if not args.progress_only:
                chunk_desc = (
                    f"{prefix}Processing {video_path} chunk {chunk_idx}/{len(chunk_plan)} "
                    f"[{chunk_start:.1f}s - {chunk_end if chunk_end is not None else duration_sec:.1f}s]"
                )
                print(chunk_desc)

            process_video(
                video_path=video_path,
                db_path=args.db,
                frame_stride_sec=args.stride_sec,
                max_frames=args.max_frames,
                output_dir=args.output,
                max_seconds=chunk_end,
                min_aesthetic=getattr(args, "min_aesthetic", 0.0),
                dedup_threshold=getattr(args, "dedup_threshold", 0.15),
                final_dedup_threshold=getattr(args, "final_dedup_threshold", None),
                progress_only=getattr(args, "progress_only", False),
                enable_face_checks=not getattr(args, "disable_face", False),
                adaptive_stride=getattr(args, "adaptive_stride", False),
                adaptive_stride_threshold=getattr(args, "adaptive_stride_threshold", 3.0),
                face_skin_threshold=getattr(args, "face_skin_threshold", 0.05),
                max_face_tilt=getattr(args, "max_face_tilt", MAX_FACE_TILT_DEGREES),
                face_sharpness_threshold=getattr(args, "face_sharpness_threshold", MIN_FACE_SHARPNESS),
                min_face_area_ratio=getattr(args, "min_face_area_ratio", 0.0),
                min_face_y_center=getattr(args, "min_face_y_center", 0.0),
                max_workers=getattr(args, "workers", 4),
                min_tech_score=getattr(args, "min_tech_score", MIN_TECH_SCORE),
                require_face=getattr(args, "require_face", False),
                allow_faceless_scenes=getattr(args, "allow_faceless_scenes", False),
                faceless_min_aesthetic=getattr(args, "faceless_min_aesthetic", 5.0),
                allow_faceless_scenic_only=getattr(args, "allow_faceless_scenic_only", False),
                faceless_max_skin_ratio=getattr(args, "faceless_max_skin_ratio", 0.05),
                faceless_motion_threshold=getattr(args, "faceless_motion_threshold", 2.0),
                faceless_confidence_threshold=getattr(args, "faceless_max_confidence", 0.3),
                faceless_people_ratio=getattr(args, "faceless_max_person_ratio", 0.05),
                skip_overlay_cards=getattr(args, "skip_overlay_cards", False),
                overlay_coverage=getattr(args, "overlay_coverage", 0.3),
                overlay_sat_threshold=getattr(args, "overlay_sat_threshold", 0.5),
                overlay_sat_ratio=getattr(args, "overlay_sat_ratio", 0.6),
                overlay_unique_threshold=getattr(args, "overlay_unique_threshold", 0.2),
                overlay_hue_dominance=getattr(args, "overlay_hue_dominance", 0.7),
                overlay_color_ratio=getattr(args, "overlay_color_ratio", 0.4),
                use_aesthetic=not getattr(args, "disable_aesthetic", False),
                start_sec=chunk_start,
                skip_to_final_stage=getattr(args, "skip_to_final_stage", False),
                prefer_smiles=not getattr(args, "no_prefer_smiles", False),
                summary_log_path=summary_log,
                global_state=global_state,
            )

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        work_dir = os.path.join(output_root, f"__work_{video_name}")
        shutil.rmtree(work_dir, ignore_errors=True)

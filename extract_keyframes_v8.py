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

_FACE_ANALYZER: Optional[FaceAnalysis] = None

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
    if face_region.size > 0:
        try:
            face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            face_sharpness_var = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())
        except Exception:
            face_sharpness_var = 0.0
    else:
        face_sharpness_var = 0.0

    return {
        "num_faces": num_faces,
        "has_face": True,
        "main_face_conf": main_conf,
        "main_face_bbox": (x_min, y_min, x_max, y_max),
        "main_face_rel_area": main_face_rel_area,
        "face_size_score": size_score,
        "face_quality_score": face_quality_score,
        "face_sharpness_var": face_sharpness_var,
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

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride_frames = max(1, int(round(frame_stride_sec * fps)))
    start_sec = max(0.0, float(start_sec))
    start_frame = int(min(total_frames, max(0, round(start_sec * fps))))

    def _frame_iter() -> Iterable[Tuple[int, float, np.ndarray]]:
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
                yield frame_idx, timestamp_sec, frame.copy()
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
            aesthetic, tech_score, sharpness_var, noise_sigma,
            clip_pct_shadows, clip_pct_highlights, mean_luma,
            color_cast_score, num_faces, has_face,
            face_quality_score, main_face_rel_area, composition_score,
            subject_x_norm, subject_y_norm,
            embed_dim, embedding
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            video_path,
            frame_index,
            float(timestamp_sec),
            float(aesthetic),
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


def clear_previous_video(conn: duckdb.DuckDBPyConnection, video_path: str) -> None:
    conn.execute("DELETE FROM frames WHERE video_path = ?", [video_path])


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
    face_skin_threshold: float = 0.05,
    max_workers: int = 4,
    black_frame_threshold: float = 0.05,
    black_frame_ratio: float = 0.05,
    min_tech_score: float = MIN_TECH_SCORE,
    require_face: bool = False,
    skip_overlay_cards: bool = False,
    overlay_coverage: float = 0.3,
    overlay_sat_threshold: float = 0.5,
    overlay_sat_ratio: float = 0.6,
    overlay_unique_threshold: float = 0.2,
    overlay_hue_dominance: float = 0.7,
    overlay_color_ratio: float = 0.4,
    use_aesthetic: bool = True,
    start_sec: float = 0.0,
    resize_long_edge: int = 0,
    skip_to_final_stage: bool = False,
) -> None:
    overall_start = time.time()
    start_sec = max(0.0, float(start_sec))
    first_chunk = start_sec <= 1e-6
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not progress_only:
        chunk_label = (
            f"[chunk {start_sec:.1f}s - {max_seconds if max_seconds is not None else 'end'}s]"
        )
        print(f"Opening DuckDB at {db_path} {chunk_label}")
    conn = duckdb.connect(database=db_path)
    init_db(conn)

    # Prepare image output directory as <output>/<video_name>/
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_out_dir = os.path.join(output_dir, video_name)
    if first_chunk and os.path.exists(video_out_dir):
        shutil.rmtree(video_out_dir)
    os.makedirs(video_out_dir, exist_ok=True)

    final_index_path = os.path.join(video_out_dir, "_final_embeddings.npy")
    final_removed_dir = os.path.join(video_out_dir, "finally_removed")
    if first_chunk:
        shutil.rmtree(final_removed_dir, ignore_errors=True)
        try:
            os.remove(final_index_path)
        except Exception:
            pass
    os.makedirs(final_removed_dir, exist_ok=True)

    filtered_dir = os.path.join(video_out_dir, "filtered")
    os.makedirs(filtered_dir, exist_ok=True)

    tmp_candidate_dir = os.path.join(
        video_out_dir, f"_candidates_tmp_{int(round(start_sec * 1000.0))}"
    )
    os.makedirs(tmp_candidate_dir, exist_ok=True)

    if first_chunk:
        clear_previous_video(conn, video_path)

    max_seconds = float(max_seconds) if max_seconds is not None else None

    candidates: List[
        Tuple[int, float, float, np.ndarray, Optional[str], Optional[np.ndarray], dict, dict, dict]
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

    def save_filtered_frame(frame_img: np.ndarray, reason: str, details: Optional[dict] = None) -> None:
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

    stride_sec = max(frame_stride_sec, 1e-6)

    def skin_probability(frame: np.ndarray) -> float:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 48, 80), (20, 255, 255))
        return float(mask.mean() / 255.0)

    def should_run_face(frame: np.ndarray) -> bool:
        if not enable_face_checks or FaceAnalysis is None:
            return False
        return skin_probability(frame) >= face_skin_threshold

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

    def process_frame(frame_idx: int, ts: float, frame_bgr: np.ndarray) -> Optional[tuple]:
        original_frame = frame_bgr
        proc_frame = frame_bgr
        quality_scale = 1.0
        if resize_long_edge > 0:
            h0, w0 = original_frame.shape[:2]
            long_edge_src = max(h0, w0)
            if long_edge_src > resize_long_edge:
                scale = float(resize_long_edge) / float(long_edge_src)
                new_w = max(1, int(round(w0 * scale)))
                new_h = max(1, int(round(h0 * scale)))
                try:
                    proc_frame = cv2.resize(original_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                except Exception:
                    proc_frame = original_frame
                quality_scale = scale
            else:
                quality_scale = 1.0
        tech = compute_technical_quality(proc_frame)
        long_edge = max(tech["resolution_w"], tech["resolution_h"])
        min_res_edge = MIN_RES_LONG_EDGE * quality_scale
        min_total_pixels = MIN_TOTAL_PIXELS * (quality_scale ** 2)
        min_res_edge = max(1.0, min_res_edge)
        min_total_pixels = max(10.0, min_total_pixels)
        if long_edge < min_res_edge or tech["total_pixels"] < min_total_pixels:
            return None

        if tech["tech_score"] < min_tech_score:
            return None

        if is_black_slate(
            cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0,
            tech["mean_luma"],
            black_frame_threshold,
            black_frame_ratio,
        ):
            return None

        run_face = should_run_face(proc_frame)
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

        if require_face and not face.get("has_face", False):
            return None

        if skip_overlay_cards and looks_like_overlay_card(
            proc_frame,
            coverage_ratio=overlay_coverage,
            sat_threshold=overlay_sat_threshold,
            sat_ratio_threshold=overlay_sat_ratio,
            unique_threshold=overlay_unique_threshold,
            hue_dominance=overlay_hue_dominance,
            color_ratio_threshold=overlay_color_ratio,
        ):
            return None

        if tech["sharpness_var"] < MIN_SHARPNESS_VAR and run_face:
            face_sharpness = face.get("face_sharpness_var", 0.0)
            if not (face.get("has_face", False) and face_sharpness >= (0.5 * MIN_SHARPNESS_VAR)):
                return None

        comp = compute_composition_features(proc_frame, face_info=face)
        if use_aesthetic:
            aest = compute_aesthetic_score(proc_frame)
            if aest < float(min_aesthetic):
                return None
        else:
            aest = 0.0
        emb = compute_embedding(proc_frame)

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

        return (frame_idx, ts, float(aest), emb, tmp_path, stored_image, tech, face, comp)

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

            if adaptive_stride and prev_frame is not None:
                diff = cv2.absdiff(prev_frame, frame_bgr)
                if diff.mean() < 3.0:
                    prev_frame = frame_bgr
                    continue
            prev_frame = frame_bgr

            if executor:
                futures.append(executor.submit(process_frame, frame_idx, ts, frame_bgr.copy()))
            else:
                result = process_frame(frame_idx, ts, frame_bgr)
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
        Tuple[int, float, float, np.ndarray, Optional[str], Optional[np.ndarray], dict, dict, dict]
    ] = []
    kept_embs: List[np.ndarray] = []
    effective_threshold = max(0.0, float(dedup_threshold))
    # Vectorized store of kept embeddings for fast similarity checks
    kept_mat: Optional[np.ndarray] = None

    dedup_stage_start = time.time()
    total_candidates = max(1, len(candidates))
    dedup_log_every = max(1, total_candidates // 100)
    for idx, cand in enumerate(candidates, start=1):
        frame_idx, ts, aest, emb, tmp_path, stored_image, tech, face, comp = cand
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
            max_sim = float(np.max(sims)) if sims.size else -1.0
        except Exception:
            # Fallback to scalar loop in rare failure cases
            sims = [float(np.dot(k, emb)) for k in kept_mat]  # type: ignore[arg-type]
            max_sim = max(sims) if sims else -1.0
        min_dist = 1.0 - max_sim
        if min_dist < effective_threshold:
            # duplicate detected, skip (higher-aesthetic already kept)
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
        Tuple[int, float, float, np.ndarray, Optional[str], Optional[np.ndarray], dict, dict, dict]
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

    if resolved_final_threshold > 0.0 and kept:
        final_kept_list: List[
            Tuple[int, float, float, np.ndarray, Optional[str], Optional[np.ndarray], dict, dict, dict]
        ] = []
        chunk_final_mat: Optional[np.ndarray] = None
        for cand in kept:
            emb = cand[3]
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
            if chunk_final_mat is not None and chunk_final_mat.size > 0:
                try:
                    sims_chunk = chunk_final_mat.dot(emb)
                    if sims_chunk.size:
                        chunk_dist = 1.0 - float(np.max(sims_chunk))
                        if chunk_dist < resolved_final_threshold:
                            final_removed.append(cand)
                            continue
                except Exception:
                    pass
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
    for rank, (frame_idx, ts, aest, emb, tmp_path, stored_image, tech, face, comp) in enumerate(kept, start=1):
        insert_frame_row(
            conn=conn,
            video_path=video_path,
            frame_index=frame_idx,
            timestamp_sec=ts,
            aesthetic=aest,
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

        # Move tmp JPEG into final location when available, avoiding re-encode
        fname = f"{frame_idx:09d}_t{ts:.2f}_aest{aest:.2f}.jpg"
        dest_path = os.path.join(video_out_dir, fname)
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

    # Move duplicates discarded in final stage
    for dup in final_removed:
        frame_idx, ts, aest, emb, tmp_path, stored_image, tech, face, comp = dup
        dup_name = f"{frame_idx:09d}_t{ts:.2f}_dup.jpg"
        dest_path = os.path.join(final_removed_dir, dup_name)
        moved = False
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.replace(tmp_path, dest_path)
                moved = True
            except Exception:
                moved = False
        if not moved:
            src_img = stored_image
            if src_img is None and tmp_path and os.path.exists(tmp_path):
                try:
                    src_img = cv2.imread(tmp_path)
                except Exception:
                    src_img = None
            if src_img is not None:
                try:
                    cv2.imwrite(dest_path, src_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                except Exception:
                    pass

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
        "--face-skin-threshold",
        type=float,
        default=0.05,
        help="Minimum skin probability to run face detection when enabled",
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
        "--resize-long-edge",
        type=int,
        default=0,
        help="If >0, resize frames so their longest edge equals this value before processing",
    )
    parser.add_argument(
        "--skip-to-final-stage",
        action="store_true",
        help="Skip new frame processing and only run final deduplication/output steps",
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

    for idx, video_path in enumerate(video_inputs, start=1):
        if args.progress_only and total_videos > 1:
            try:
                print(f"[prog] video {idx}/{total_videos}: {video_path}")
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

        cumulative_time = 0.0
        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_plan, start=1):
            if not args.progress_only:
                chunk_desc = (
                    f"{prefix}Processing {video_path} chunk {chunk_idx}/{len(chunk_plan)} "
                    f"[{chunk_start:.1f}s - {chunk_end if chunk_end is not None else duration_sec:.1f}s]"
                )
                print(chunk_desc)

            chunk_start_time = time.time()
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
                face_skin_threshold=getattr(args, "face_skin_threshold", 0.05),
                max_workers=getattr(args, "workers", 4),
                min_tech_score=getattr(args, "min_tech_score", MIN_TECH_SCORE),
                require_face=getattr(args, "require_face", False),
                skip_overlay_cards=getattr(args, "skip_overlay_cards", False),
                overlay_coverage=getattr(args, "overlay_coverage", 0.3),
                overlay_sat_threshold=getattr(args, "overlay_sat_threshold", 0.5),
                overlay_sat_ratio=getattr(args, "overlay_sat_ratio", 0.6),
                overlay_unique_threshold=getattr(args, "overlay_unique_threshold", 0.2),
                overlay_hue_dominance=getattr(args, "overlay_hue_dominance", 0.7),
                overlay_color_ratio=getattr(args, "overlay_color_ratio", 0.4),
                use_aesthetic=not getattr(args, "disable_aesthetic", False),
                start_sec=chunk_start,
                resize_long_edge=getattr(args, "resize_long_edge", 0),
                skip_to_final_stage=getattr(args, "skip_to_final_stage", False),
            )
            chunk_elapsed = time.time() - chunk_start_time
            cumulative_time += chunk_elapsed
            if args.progress_only:
                try:
                    print(
                        f"[prog] chunk {chunk_idx}/{len(chunk_plan)} done "
                        f"(chunk={chunk_elapsed:.1f}s total={cumulative_time:.1f}s)"
                    )
                except Exception:
                    pass
            else:
                print(
                    f"{prefix}Finished chunk {chunk_idx}/{len(chunk_plan)} "
                    f"(chunk {chunk_elapsed:.1f}s, cumulative {cumulative_time:.1f}s)"
                )

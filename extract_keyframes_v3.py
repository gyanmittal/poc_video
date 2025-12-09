#!/usr/bin/env python3
import argparse
import json
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

_FACE_ANALYZER: Optional[FaceAnalysis] = None


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
    frame_stride_sec: float = 1.0,
    max_frames: int | None = None,
    max_seconds: float | None = None,
) -> List[Tuple[int, float, np.ndarray]]:
    """
    Extract frames from a video every `frame_stride_sec` seconds.
    Returns list of (frame_idx, timestamp_sec, frame_bgr).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0  # fallback
    frame_stride = int(round(frame_stride_sec * fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames: List[Tuple[int, float, np.ndarray]] = []

    frame_idx = 0
    out_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # compute timestamp for this frame
        timestamp_sec = frame_idx / fps
        # enforce max_seconds cap, exclusive
        if max_seconds is not None and timestamp_sec >= float(max_seconds):
            break

        if frame_idx % frame_stride == 0:
            frames.append((frame_idx, timestamp_sec, frame.copy()))
            out_count += 1
            if max_frames is not None and out_count >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return frames


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
    progress_only: bool = False,
) -> None:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not progress_only:
        print(f"Opening DuckDB at {db_path}")
    conn = duckdb.connect(database=db_path)
    init_db(conn)

    # Prepare image output directory as <output>/<video_name>/
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_out_dir = os.path.join(output_dir, video_name)
    if os.path.exists(video_out_dir):
        shutil.rmtree(video_out_dir)
    os.makedirs(video_out_dir, exist_ok=True)

    filtered_dir = os.path.join(video_out_dir, "filtered")
    os.makedirs(filtered_dir, exist_ok=True)

    clear_previous_video(conn, video_path)

    if not progress_only:
        print(f"Extracting frames from {video_path} (stride {frame_stride_sec}s, max_seconds={max_seconds})...")
    frames = extract_frames(video_path, frame_stride_sec, max_frames, max_seconds)
    if not progress_only:
        print(f"Got {len(frames)} frames to process")

    candidates: List[Tuple[int, float, float, np.ndarray, np.ndarray, dict, dict]] = []

    total_frames = len(frames)
    progress_log_interval = max(1, total_frames // 100) if total_frames > 0 else 1

    def save_filtered_frame(frame_img: np.ndarray, reason: str, details: Optional[dict] = None) -> None:
        slug = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in reason.lower())
        fname_base = f"{frame_idx:09d}_t{ts:.2f}_{slug}"
        image_path = os.path.join(filtered_dir, f"{fname_base}.jpg")
        try:
            cv2.imwrite(image_path, frame_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        except Exception:
            image_path = ""

        detail_dict: dict = {}
        if details:
            for key, value in details.items():
                if isinstance(value, (np.generic,)):
                    detail_dict[key] = float(value)
                else:
                    detail_dict[key] = value

        meta = {
            "frame_index": int(frame_idx),
            "timestamp_sec": float(ts),
            "reason": reason,
            "image_path": image_path,
            "details": detail_dict,
        }
        try:
            with open(os.path.join(filtered_dir, f"{fname_base}.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass

    for i, (frame_idx, ts, frame_bgr) in enumerate(frames, start=1):
        if not progress_only:
            print(f"[{i}/{len(frames)}] frame_index={frame_idx}, t={ts:.2f}s")
        elif i == 1 or (i % progress_log_interval == 0) or i == total_frames:
            denom = max(1, total_frames)
            pct = (float(i) / float(denom)) * 100.0
            try:
                print(
                    f"\r[prog] frames {i}/{denom} ({pct:.1f}%)",
                    end="",
                    flush=True,
                )
            except Exception:
                pass

        tech = compute_technical_quality(frame_bgr)

        long_edge = max(tech["resolution_w"], tech["resolution_h"])
        if long_edge < MIN_RES_LONG_EDGE or tech["total_pixels"] < MIN_TOTAL_PIXELS:
            save_filtered_frame(
                frame_bgr,
                "low_resolution",
                {
                    "resolution_w": tech["resolution_w"],
                    "resolution_h": tech["resolution_h"],
                    "total_pixels": tech["total_pixels"],
                    "min_res_long_edge": MIN_RES_LONG_EDGE,
                    "min_total_pixels": MIN_TOTAL_PIXELS,
                },
            )
            if not progress_only:
                print("  -> skipped: resolution too low")
            continue

        if tech["sharpness_var"] < MIN_SHARPNESS_VAR:
            face = compute_face_features(frame_bgr)
            face_sharpness = face.get("face_sharpness_var", 0.0)
            face_ok = face.get("has_face", False) and face_sharpness >= (0.5 * MIN_SHARPNESS_VAR)
            if not face_ok:
                save_filtered_frame(
                    frame_bgr,
                    "too_blurry",
                    {
                        "frame_sharpness_var": tech["sharpness_var"],
                        "face_sharpness_var": face_sharpness,
                        "min_sharpness_var": MIN_SHARPNESS_VAR,
                        "has_face": face.get("has_face", False),
                    },
                )
                if not progress_only:
                    print("  -> skipped: too blurry")
                continue
        else:
            face = compute_face_features(frame_bgr)

        frame_h, frame_w = frame_bgr.shape[:2]

        if face.get("has_face", False) and face.get("main_face_bbox"):
            x_min, y_min, x_max, y_max = face["main_face_bbox"]
            face_w = max(0, x_max - x_min)
            face_h = max(0, y_max - y_min)
            min_dim = min(face_w, face_h)
            shortest_edge = max(1, min(frame_w, frame_h))
            min_dim_ratio = float(min_dim) / float(shortest_edge)
            if min_dim_ratio < MIN_FACE_MIN_DIM_RATIO:
                save_filtered_frame(
                    frame_bgr,
                    "face_too_small",
                    {
                        "min_dim_ratio": min_dim_ratio,
                        "threshold": MIN_FACE_MIN_DIM_RATIO,
                        "face_width": face_w,
                        "face_height": face_h,
                    },
                )
                if not progress_only:
                    print(
                        "  -> skipped: face too small "
                        f"(min_dim_ratio={min_dim_ratio:.3f})"
                    )
                continue

            aspect = face_w / float(max(1, face_h)) if face_h > 0 else 0.0
            if aspect < FACE_ASPECT_MIN or aspect > FACE_ASPECT_MAX:
                save_filtered_frame(
                    frame_bgr,
                    "face_aspect_weird",
                    {
                        "aspect_ratio": aspect,
                        "min_allowed": FACE_ASPECT_MIN,
                        "max_allowed": FACE_ASPECT_MAX,
                        "face_width": face_w,
                        "face_height": face_h,
                    },
                )
                if not progress_only:
                    print(
                        "  -> skipped: face aspect ratio out of range "
                        f"({aspect:.2f})"
                    )
                continue

        if tech["noise_sigma"] > MAX_NOISE_SIGMA:
            save_filtered_frame(
                frame_bgr,
                "too_noisy",
                {
                    "noise_sigma": tech["noise_sigma"],
                    "max_noise_sigma": MAX_NOISE_SIGMA,
                },
            )
            if not progress_only:
                print("  -> skipped: too noisy")
            continue

        if (
            tech["clip_pct_shadows"] > MAX_CLIP_PCT
            or tech["clip_pct_highlights"] > MAX_CLIP_PCT
        ):
            save_filtered_frame(
                frame_bgr,
                "heavy_clipping",
                {
                    "clip_pct_shadows": tech["clip_pct_shadows"],
                    "clip_pct_highlights": tech["clip_pct_highlights"],
                    "max_clip_pct": MAX_CLIP_PCT,
                },
            )
            if not progress_only:
                print("  -> skipped: heavy clipping")
            continue

        if tech["tech_score"] < MIN_TECH_SCORE:
            save_filtered_frame(
                frame_bgr,
                "low_tech_score",
                {
                    "tech_score": tech["tech_score"],
                    "threshold": MIN_TECH_SCORE,
                },
            )
            if not progress_only:
                print(f"  -> skipped: tech_score={tech['tech_score']:.1f} < {MIN_TECH_SCORE}")
            continue

        comp = compute_composition_features(frame_bgr, face_info=face)

        # Aesthetic score
        aest = compute_aesthetic_score(frame_bgr)

        if aest < float(min_aesthetic):
            save_filtered_frame(
                frame_bgr,
                "low_aesthetic",
                {
                    "aesthetic": float(aest),
                    "min_aesthetic": float(min_aesthetic),
                    "num_faces": face.get("num_faces", 0),
                    "has_face": face.get("has_face", False),
                },
            )
            continue

        # Embedding
        emb = compute_embedding(frame_bgr)

        candidates.append((frame_idx, ts, float(aest), emb, frame_bgr.copy(), tech, face, comp))

    # Sort by aesthetic descending so highest quality considered first
    candidates.sort(key=lambda x: x[2], reverse=True)

    kept: List[Tuple[int, float, float, np.ndarray, np.ndarray, dict, dict]] = []
    kept_embs: List[np.ndarray] = []
    effective_threshold = max(0.0, float(dedup_threshold))

    for cand in candidates:
        frame_idx, ts, aest, emb, frame_bgr, tech, face, comp = cand
        if effective_threshold <= 0.0 or len(kept_embs) == 0:
            kept.append(cand)
            kept_embs.append(emb)
            continue

        # cosine distance to existing kept embeddings
        dists = [1.0 - float(np.dot(k_emb, emb)) for k_emb in kept_embs]
        min_dist = min(dists) if dists else 1.0
        if min_dist < effective_threshold:
            # duplicate detected, skip (keep higher-aesthetic already stored)
            continue

        kept.append(cand)
        kept_embs.append(emb)

    # Persist kept frames in timestamp order for readability
    kept.sort(key=lambda x: x[1])

    for rank, (frame_idx, ts, aest, emb, frame_bgr, tech, face, comp) in enumerate(kept, start=1):
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

        try:
            fname = f"{frame_idx:09d}_t{ts:.2f}_aest{aest:.2f}.jpg"
            cv2.imwrite(os.path.join(video_out_dir, fname), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        except Exception:
            pass

    if progress_only:
        try:
            print()
        except Exception:
            pass

    conn.close()
    print(f"Done. kept={len(kept)} (from {len(candidates)} candidates after filtering)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract keyframes, compute embeddings and CLIP+LAION aesthetic scores, store in DuckDB."
    )
    parser.add_argument("video", help="Path to input video file")
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
        "--progress-only",
        action="store_true",
        help="Suppress per-frame logs and show a compact progress meter",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_video(
        video_path=args.video,
        db_path=args.db,
        frame_stride_sec=args.stride_sec,
        max_frames=args.max_frames,
        output_dir=args.output,
        max_seconds=getattr(args, "max_seconds", None),
        min_aesthetic=getattr(args, "min_aesthetic", 0.0),
        dedup_threshold=getattr(args, "dedup_threshold", 0.15),
        progress_only=getattr(args, "progress_only", False),
    )

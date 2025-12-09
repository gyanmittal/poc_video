#!/usr/bin/env python3
import argparse
import os
import shutil
from typing import List, Tuple

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
            color_cast_score, embed_dim, embedding
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
) -> None:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    print(f"Opening DuckDB at {db_path}")
    conn = duckdb.connect(database=db_path)
    init_db(conn)

    # Prepare image output directory as <output>/<video_name>/
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_out_dir = os.path.join(output_dir, video_name)
    if os.path.exists(video_out_dir):
        shutil.rmtree(video_out_dir)
    os.makedirs(video_out_dir, exist_ok=True)

    clear_previous_video(conn, video_path)

    print(f"Extracting frames from {video_path} (stride {frame_stride_sec}s, max_seconds={max_seconds})...")
    frames = extract_frames(video_path, frame_stride_sec, max_frames, max_seconds)
    print(f"Got {len(frames)} frames to process")

    candidates: List[Tuple[int, float, float, np.ndarray, np.ndarray, dict]] = []

    for i, (frame_idx, ts, frame_bgr) in enumerate(frames, start=1):
        print(f"[{i}/{len(frames)}] frame_index={frame_idx}, t={ts:.2f}s")

        tech = compute_technical_quality(frame_bgr)

        long_edge = max(tech["resolution_w"], tech["resolution_h"])
        if long_edge < MIN_RES_LONG_EDGE or tech["total_pixels"] < MIN_TOTAL_PIXELS:
            print("  -> skipped: resolution too low")
            continue

        if tech["sharpness_var"] < MIN_SHARPNESS_VAR:
            print("  -> skipped: too blurry")
            continue

        if tech["noise_sigma"] > MAX_NOISE_SIGMA:
            print("  -> skipped: too noisy")
            continue

        if (
            tech["clip_pct_shadows"] > MAX_CLIP_PCT
            or tech["clip_pct_highlights"] > MAX_CLIP_PCT
        ):
            print("  -> skipped: heavy clipping")
            continue

        if tech["tech_score"] < MIN_TECH_SCORE:
            print(f"  -> skipped: tech_score={tech['tech_score']:.1f} < {MIN_TECH_SCORE}")
            continue

        # Aesthetic score
        aest = compute_aesthetic_score(frame_bgr)

        if aest < float(min_aesthetic):
            continue

        # Embedding
        emb = compute_embedding(frame_bgr)

        candidates.append((frame_idx, ts, float(aest), emb, frame_bgr.copy(), tech))

    # Sort by aesthetic descending so highest quality considered first
    candidates.sort(key=lambda x: x[2], reverse=True)

    kept: List[Tuple[int, float, float, np.ndarray, np.ndarray, dict]] = []
    kept_embs: List[np.ndarray] = []
    effective_threshold = max(0.0, float(dedup_threshold))

    for cand in candidates:
        frame_idx, ts, aest, emb, frame_bgr, tech = cand
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

    for rank, (frame_idx, ts, aest, emb, frame_bgr, tech) in enumerate(kept, start=1):
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
            embedding=emb,
        )

        try:
            fname = f"{frame_idx:09d}_t{ts:.2f}_aest{aest:.2f}.jpg"
            cv2.imwrite(os.path.join(video_out_dir, fname), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
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
    )

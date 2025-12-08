"""
Extract distinct, high-quality keyframes from a video with minimal RAM usage.

Pipeline overview:
- Stream frames with OpenCV (no full video or feature matrix in RAM).
- For each processed frame:
  - Compute an ImageNet-feature embedding using a TorchVision backbone (default: ResNet-18).
  - L2-normalize the embedding (cosine similarity ready).
  - Compute a quality score (variance of Laplacian as a sharpness proxy; optional contrast term).
- Similarity grouping (threshold-based, time-ordered):
  - Maintain an "anchor" embedding for the last selected representative.
  - A frame starts a new group if its cosine distance to the anchor exceeds a threshold.
  - Within a group, keep the single best-quality frame (overwrite if a better frame appears).
- Storage (DuckDB, minimal RAM):
  - Persist rows incrementally as we stream.
  - Tables: videos, frames, embeddings, frame_groups.
  - Embeddings saved as BLOB (float32 bytes) per frame; not loaded back during processing.
  - Schema designed to be extendable (new columns/tables can be added later).
- Output: save selected frames to disk and emit a CSV summary.

Installation:
    pip install opencv-python torch torchvision numpy duckdb

Optional (if you want to experiment with other metrics or clustering):
    pip install scikit-image scikit-learn faiss-cpu

Usage:
    python extract_keyframes.py \
        --corpus poc_video/videos/your_video.mp4 \
        --output ./output_folder \
        --db ./metadata.duckdb \
        --distance-threshold 0.25 \
        --frame-skip 1

Outputs:
- For an input video named <video_name>, images are saved in <output>/<video_name>/ (JPEG files).
- A CSV summary in <output>/<video_name>/summary.csv.
- DuckDB at --db with metadata for videos, frames, embeddings, groups.

Notes:
- This script uses only open-source libraries and runs locally.
- If pre-trained weights can't be downloaded, it falls back to randomly initialized weights.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from typing import Generator, Iterable, Optional, Tuple, List

import cv2
import duckdb  # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
import timm



# -------------------------------
# Data classes and configuration
# -------------------------------

@dataclass
class ProcessingConfig:
    """Configuration for processing and selection logic."""
    distance_threshold: float = 0.25  # cosine distance threshold to start a new group
    frame_skip: int = 1  # process every Nth frame
    resize_shorter: int = 256  # resize shorter side before model input
    model_input_size: int = 224  # model crop size
    quality_alpha: float = 1.0  # weight for laplacian variance (sharpness)
    quality_beta: float = 0.0   # optional additional contrast weight
    device: Optional[str] = None  # 'cuda', 'mps', or 'cpu'
    pretrained: bool = True  # try to load pretrained weights
    save_jpeg_quality: int = 95  # JPEG save quality
    max_frames: Optional[int] = None  # cap processed frames for debugging
    backbone: str = "resnet50"  # timm model name, e.g., 'resnet50', 'efficientnet_b0'
    dedup_threshold: float = 0.07  # cosine distance threshold to suppress near-duplicate selected frames
    face_filter: bool = False  # require at least one detected face per frame
    min_sharpness: float = 50.0  # discard frames with Laplacian variance below this
    video_face_filter: bool = False  # skip entire video if no prominent faces appear
    min_face_area: float = 0.015  # minimum face area ratio (bbox area / frame area) to be considered prominent
    min_face_frames: int = 1  # require at least this many frames with a prominent face to process video
    debug: bool = False  # debug mode: throttle logs to 10s intervals and completion
    # Person prominence filter (OpenCV HOG-based)
    person_filter: bool = True  # require at least one prominent person per frame
    min_person_area: float = 0.08  # minimum person bbox area ratio to be considered prominent
    # Text prominence filter
    text_filter: bool = False  # skip frames with prominent text overlays
    text_area_threshold: float = 0.15  # total text-like area ratio threshold (higher = less sensitive)
    text_bottom_ratio: float = 0.2  # bottom region proportion emphasized for subtitles (lower = less sensitive)
    text_method: str = "fast"  # 'fast' (edge-based) or 'mser'
    # Subtitle tolerance (do not treat short bottom subtitles as prominent)
    subtitle_tolerant: bool = True
    subtitle_max_rel_height: float = 0.12  # relative to full image height


# -------------------------------
# DuckDB storage helper
# -------------------------------

class DuckDBStorage:
    """Minimal-RAM, streaming-friendly persistence using DuckDB.

    Tables:
    - videos(video_id TEXT PRIMARY KEY, path TEXT UNIQUE, width, height, fps, frame_count, duration_sec, created_at)
    - frames(frame_id TEXT PRIMARY KEY, video_id TEXT, frame_index INT, timestamp_sec DOUBLE,
             quality DOUBLE, group_id TEXT, is_representative BOOLEAN)
    - embeddings(frame_id TEXT PRIMARY KEY, dim INT, embedding BLOB)
    - frame_groups(group_id TEXT PRIMARY KEY, video_id TEXT, start_index INT, end_index INT,
                  selected_frame_id TEXT, anchor_frame_id TEXT)
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.con = duckdb.connect(self.db_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                path TEXT UNIQUE,
                width INTEGER,
                height INTEGER,
                fps DOUBLE,
                frame_count BIGINT,
                duration_sec DOUBLE,
                created_at TIMESTAMP DEFAULT current_timestamp
            );
            """
        )
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS frames (
                frame_id TEXT PRIMARY KEY,
                video_id TEXT,
                frame_index INTEGER,
                timestamp_sec DOUBLE,
                quality DOUBLE,
                group_id TEXT,
                is_representative BOOLEAN DEFAULT FALSE
            );
            """
        )
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                frame_id TEXT PRIMARY KEY,
                dim INTEGER,
                embedding BLOB
            );
            """
        )
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS frame_groups (
                group_id TEXT PRIMARY KEY,
                video_id TEXT,
                start_index INTEGER,
                end_index INTEGER,
                selected_frame_id TEXT,
                anchor_frame_id TEXT
            );
            """
        )

    def get_or_create_video(self, path: str, width: int, height: int, fps: float,
                             frame_count: Optional[int], duration_sec: Optional[float]) -> str:
        cur = self.con.execute("SELECT video_id FROM videos WHERE path = ?", [path])
        row = cur.fetchone()
        if row:
            return row[0]
        video_id = str(uuid.uuid4())
        self.con.execute(
            """
            INSERT INTO videos (video_id, path, width, height, fps, frame_count, duration_sec)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [video_id, path, width, height, float(fps), frame_count, duration_sec],
        )
        return video_id

    def insert_frame(self, video_id: str, frame_index: int, timestamp_sec: float,
                     quality: float, group_id: str, is_representative: bool = False,
                     run_id: Optional[str] = None) -> str:
        frame_id = f"{video_id}:{run_id}:{frame_index}" if run_id else f"{video_id}:{frame_index}"
        self.con.execute(
            """
            INSERT INTO frames (frame_id, video_id, frame_index, timestamp_sec, quality, group_id, is_representative)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [frame_id, video_id, int(frame_index), float(timestamp_sec), float(quality), group_id, is_representative],
        )
        return frame_id

    def insert_embedding(self, frame_id: str, emb: np.ndarray) -> None:
        assert emb.dtype == np.float32 and emb.ndim == 1
        self.con.execute(
            """
            INSERT INTO embeddings (frame_id, dim, embedding) VALUES (?, ?, ?)
            """,
            [frame_id, int(emb.shape[0]), emb.tobytes()],
        )

    def upsert_group(self, group_id: str, video_id: str, start_index: int,
                     end_index: Optional[int] = None,
                     selected_frame_id: Optional[str] = None,
                     anchor_frame_id: Optional[str] = None) -> None:
        # Check if exists
        row = self.con.execute("SELECT 1 FROM frame_groups WHERE group_id = ?", [group_id]).fetchone()
        if not row:
            self.con.execute(
                """
                INSERT INTO frame_groups (group_id, video_id, start_index, end_index, selected_frame_id, anchor_frame_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [group_id, video_id, int(start_index), end_index, selected_frame_id, anchor_frame_id],
            )
        else:
            self.con.execute(
                """
                UPDATE frame_groups SET
                    start_index = COALESCE(?, start_index),
                    end_index = COALESCE(?, end_index),
                    selected_frame_id = COALESCE(?, selected_frame_id),
                    anchor_frame_id = COALESCE(?, anchor_frame_id)
                WHERE group_id = ?
                """,
                [start_index, end_index, selected_frame_id, anchor_frame_id, group_id],
            )

    def mark_representative(self, frame_id: str) -> None:
        self.con.execute(
            "UPDATE frames SET is_representative = TRUE WHERE frame_id = ?",
            [frame_id],
        )

    def set_representative(self, frame_id: str, is_rep: bool) -> None:
        self.con.execute(
            "UPDATE frames SET is_representative = ? WHERE frame_id = ?",
            [is_rep, frame_id],
        )

    def selected_frames_summary(self, video_id: str) -> np.ndarray:
        query = (
            """
            SELECT f.frame_index, f.timestamp_sec, f.quality, f.frame_id
            FROM frames f
            WHERE f.video_id = ? AND f.is_representative = TRUE
            ORDER BY f.frame_index
            """
        )
        return self.con.execute(query, [video_id]).fetchall()

    def close(self) -> None:
        try:
            self.con.close()
        except Exception:
            pass


# -------------------------------
# Model and feature extraction
# -------------------------------

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model(device: Optional[str] = None, pretrained: bool = True, backbone: str = "resnet50"):
    """Load a timm backbone and prepare for embedding extraction.

    Returns (model, device_str, embedding_dim, mean, std).
    Falls back to random weights if pretrained weights can't be loaded.
    """
    if device is None:
        if torch.cuda.is_available():
            dev = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
    else:
        dev = device

    # Load timm backbone (remove classification head by setting num_classes=0 and keep global pooling)
    try:
        model = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool='avg')
    except Exception:
        model = timm.create_model(backbone, pretrained=False, num_classes=0, global_pool='avg')
    model.eval()
    model.to(dev)
    # Resolve normalization config
    try:
        data_cfg = timm.data.resolve_model_data_config(model)
    except Exception:
        data_cfg = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    mean = np.array(data_cfg.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std = np.array(data_cfg.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
    embedding_dim = getattr(model, "num_features", None)
    if embedding_dim is None:
        # Fallback: run a dummy forward pass to infer dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=dev)
            embedding_dim = int(model(dummy).shape[1])
    return model, dev, embedding_dim, mean, std


def _resize_and_normalize_bgr_to_tensor(
    frame_bgr: np.ndarray,
    resize_shorter: int,
    crop_size: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> torch.Tensor:
    """Convert BGR uint8 image to normalized CHW float tensor for ResNet input.

    - Resize so that shorter side == resize_shorter, keep aspect ratio.
    - Center-crop to crop_size x crop_size.
    - Normalize with ImageNet mean/std.
    """
    h, w = frame_bgr.shape[:2]
    # Resize shorter side
    if min(h, w) != resize_shorter:
        if h < w:
            new_h = resize_shorter
            new_w = int(round(w * (resize_shorter / h)))
        else:
            new_w = resize_shorter
            new_h = int(round(h * (resize_shorter / w)))
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = frame_bgr

    # Center crop
    hh, ww = resized.shape[:2]
    top = max(0, (hh - crop_size) // 2)
    left = max(0, (ww - crop_size) // 2)
    crop = resized[top:top + crop_size, left:left + crop_size]
    if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
        crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)

    # BGR -> RGB, to float32 [0,1]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # Normalize
    rgb = (rgb - mean) / std
    # HWC -> CHW
    chw = np.transpose(rgb, (0, 1, 2))  # already HWC, will fix next line
    chw = rgb.transpose(2, 0, 1)
    tensor = torch.from_numpy(chw)
    return tensor


def compute_embedding(
    model: torch.nn.Module,
    frame_bgr: np.ndarray,
    device: str,
    resize_shorter: int,
    crop_size: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Compute a L2-normalized embedding for the given frame using the provided model.

    Returns a 1-D float32 numpy array.
    """
    with torch.no_grad():
        x = _resize_and_normalize_bgr_to_tensor(frame_bgr, resize_shorter, crop_size, mean, std).unsqueeze(0).to(device)
        feats = model(x)  # [1, D]
        feats = F.normalize(feats, p=2, dim=1)
        emb = feats.squeeze(0).detach().to("cpu").numpy().astype(np.float32)
    return emb


# -------------------------------
# Quality scoring
# -------------------------------

def compute_quality(frame_bgr: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> float:
    """Compute a scalar quality score.

    - Sharpness via variance of Laplacian (higher is sharper).
    - Optional contrast term: std-dev of grayscale intensities.
    quality = alpha * lap_var + beta * gray_std
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(lap.var())
    gray_std = float(gray.std())
    return float(alpha * lap_var + beta * gray_std)


# -------------------------------
# Video stream extraction
# -------------------------------

def extract_frames(video_path: str, frame_skip: int = 1,
                   max_frames: Optional[int] = None) -> Generator[Tuple[int, float, np.ndarray], None, None]:
    """Yield frames from a video in a streaming manner.

    Yields (frame_index, timestamp_sec, frame_bgr) for every `frame_skip`-th frame.
    Timestamp is based on CAP_PROP_POS_MSEC if available, else fps fallback.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_idx = -1
    yielded = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue
        # Prefer timestamp from POS_MSEC
        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_msec and pos_msec > 0:
            ts = pos_msec / 1000.0
        else:
            ts = frame_idx / fps if fps > 0 else 0.0
        yield frame_idx, ts, frame
        yielded += 1
        if max_frames is not None and yielded >= max_frames:
            break

    cap.release()


# -------------------------------
# Selection and orchestration
# -------------------------------

def is_text_prominent(frame_bgr: np.ndarray, area_threshold: float = 0.08, bottom_ratio: float = 0.4, method: str = "fast", subtitle_tolerant: bool = False, subtitle_h_ratio: float = 0.12) -> bool:
    """Heuristic text prominence detector using MSER regions.

    - Downsamples to max side 512 for speed.
    - Uses MSER to find character-like regions; unions their bounding boxes into a mask.
    - Computes overall area ratio and (optionally) bottom-region ratio (subtitle emphasis).
    Returns True if text area ratio >= thresholds.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]
    max_side = max(H, W)
    target = 320 if method == "fast" else 512
    scale = float(target) / max_side if max_side > target else 1.0
    if scale < 1.0:
        gray_s = cv2.resize(gray, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    else:
        gray_s = gray

    if method == "fast":
        # Build a binary text mask from adaptive threshold (bright and dark text)
        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bw1 = cv2.adaptiveThreshold(gray_s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
        bw2 = cv2.adaptiveThreshold(gray_s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5)
        bw1 = cv2.morphologyEx(bw1, cv2.MORPH_OPEN, k3, iterations=1)
        bw2 = cv2.morphologyEx(bw2, cv2.MORPH_OPEN, k3, iterations=1)
        mask = cv2.max(bw1, bw2)
        kclose = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kclose, iterations=1)
        Hs, Ws = mask.shape[:2]
        # Keep text-like components (moderate aspect ratio, reasonable density, limited area)
        bin_ = (mask > 0).astype(np.uint8)
        num_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(bin_, connectivity=8)
        txt = np.zeros_like(bin_)
        img_area = float(Hs * Ws)
        for i in range(1, num_lbl):
            x, y, w, h, a = stats[i]
            if a / img_area > 0.25:
                continue
            if w < 6 or h < 6:
                continue
            ar = w / float(h)
            box_area = max(1, w * h)
            density = a / float(box_area)
            # Accept components with moderate aspect ratios and reasonable fill density
            if 0.2 <= ar <= 5.0 and 0.15 <= density <= 0.85:
                txt[lbl == i] = 255
        ratio_all = float((txt > 0).mean())
        # bottom band
        if bottom_ratio > 0:
            bh = int(Hs * bottom_ratio)
            bottom = txt[-bh:, :]
            ratio_bottom = float((bottom > 0).mean())
        else:
            ratio_bottom = 0.0
        # central band (to catch big overlay headlines)
        cbh = int(Hs * 0.30)
        y0 = max(0, Hs // 2 - cbh // 2)
        central = txt[y0:y0 + cbh, :]
        ratio_central = float((central > 0).mean()) if cbh > 0 else 0.0
        # central connectivity: fuse characters into lines/words to catch big overlays
        kwide = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, Ws // 30), 1))
        mask_conn = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kwide, iterations=1)
        central_conn = mask_conn[y0:y0 + cbh, :]
        ratio_central_conn = float((central_conn > 0).mean()) if cbh > 0 else 0.0
        row_cov = ((central_conn > 0).sum(axis=1) / float(Ws)) if cbh > 0 else np.array([0.0])
        max_cov = float(row_cov.max()) if cbh > 0 else 0.0
        # large line-like component heuristic
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(txt, connectivity=8)
        large_line = False
        img_area = float(Hs * Ws)
        for i in range(1, num_labels):
            x, y, w, h, a = stats[i]
            if a / img_area >= 0.005 and w >= 3 * h:
                large_line = True
                break
        # subtitle tolerance: if bottom text is short (single-line) and central is clean, do not flag
        bottom_max_h_rel = 0.0
        if bottom_ratio > 0 and bh > 0:
            num_b, lbl_b, st_b, _ = cv2.connectedComponentsWithStats((bottom > 0).astype(np.uint8), connectivity=8)
            for i in range(1, num_b):
                _, _, _, h, _ = st_b[i]
                bottom_max_h_rel = max(bottom_max_h_rel, float(h) / float(Hs))
        center_thr = max(0.5 * area_threshold, 0.04)
        central_flag = (ratio_central >= center_thr) or (ratio_central_conn >= max(0.6 * center_thr, 0.035)) or (max_cov >= 0.35)
        if subtitle_tolerant and not central_flag and bottom_max_h_rel <= subtitle_h_ratio:
            return False
        return central_flag or large_line or (ratio_bottom >= area_threshold and bottom_max_h_rel > subtitle_h_ratio)

    # method == 'mser'
    try:
        mser = cv2.MSER_create(_delta=5, _min_area=30, _max_area=max(200, int(0.01 * gray_s.size)))
    except Exception:
        # Fallback to the fast threshold-only method
        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bw1 = cv2.adaptiveThreshold(gray_s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
        bw2 = cv2.adaptiveThreshold(gray_s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5)
        bw1 = cv2.morphologyEx(bw1, cv2.MORPH_OPEN, k3, iterations=1)
        bw2 = cv2.morphologyEx(bw2, cv2.MORPH_OPEN, k3, iterations=1)
        mask = cv2.max(bw1, bw2)
        kclose = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kclose, iterations=1)
        Hs, Ws = mask.shape[:2]
        bin_ = (mask > 0).astype(np.uint8)
        num_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(bin_, connectivity=8)
        txt = np.zeros_like(bin_)
        img_area = float(Hs * Ws)
        for i in range(1, num_lbl):
            x, y, w, h, a = stats[i]
            if a / img_area > 0.25:
                continue
            if w < 6 or h < 6:
                continue
            ar = w / float(h)
            box_area = max(1, w * h)
            density = a / float(box_area)
            if 0.2 <= ar <= 5.0 and 0.15 <= density <= 0.85:
                txt[lbl == i] = 255
        ratio_all = float((txt > 0).mean())
        bh = int(Hs * bottom_ratio) if bottom_ratio > 0 else 0
        ratio_bottom = float((txt[-bh:, :] > 0).mean()) if bh > 0 else 0.0
        cbh = int(Hs * 0.30)
        y0 = max(0, Hs // 2 - cbh // 2)
        central = txt[y0:y0 + cbh, :]
        ratio_central = float((central > 0).mean()) if cbh > 0 else 0.0
        kwide = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, Ws // 30), 1))
        mask_conn = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kwide, iterations=1)
        central_conn = mask_conn[y0:y0 + cbh, :]
        ratio_central_conn = float((central_conn > 0).mean()) if cbh > 0 else 0.0
        row_cov = ((central_conn > 0).sum(axis=1) / float(Ws)) if cbh > 0 else np.array([0.0])
        max_cov = float(row_cov.max()) if cbh > 0 else 0.0
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(txt, connectivity=8)
        large_line = False
        img_area = float(Hs * Ws)
        for i in range(1, num_labels):
            x, y, w, h, a = stats[i]
            if a / img_area >= 0.005 and w >= 3 * h:
                large_line = True
                break
        bottom_max_h_rel = 0.0
        if bottom_ratio > 0 and bh > 0:
            num_b, lbl_b, st_b, _ = cv2.connectedComponentsWithStats((mask[-bh:, :] > 0).astype(np.uint8), connectivity=8)
            for i in range(1, num_b):
                _, _, _, h, _ = st_b[i]
                bottom_max_h_rel = max(bottom_max_h_rel, float(h) / float(Hs))
        center_thr = max(0.5 * area_threshold, 0.04)
        central_flag = (ratio_central >= center_thr) or (ratio_central_conn >= max(0.6 * center_thr, 0.035)) or (max_cov >= 0.35)
        if subtitle_tolerant and not central_flag and bottom_max_h_rel <= subtitle_h_ratio:
            return False
        return central_flag or large_line or (ratio_bottom >= area_threshold and bottom_max_h_rel > subtitle_h_ratio)

    regions, _ = mser.detectRegions(gray_s)
    mask = np.zeros_like(gray_s, dtype=np.uint8)
    img_area = gray_s.shape[0] * gray_s.shape[1]
    max_region_area = 0.2 * img_area
    drawn = 0
    for pts in regions:
        x, y, w, h = cv2.boundingRect(pts)
        if w < 8 or h < 8:
            continue
        ar = w / float(h)
        if ar > 10.0 or ar < 0.1:
            continue
        a = w * h
        if a < 50 or a > max_region_area:
            continue
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        drawn += 1
        if drawn > 5000:
            break
    Hs, Ws = mask.shape[:2]
    ratio_all = (mask.sum() / 255.0) / float(img_area)
    ratio_bottom = 0.0
    bh = int(Hs * bottom_ratio) if bottom_ratio and bottom_ratio > 0 else 0
    if bh > 0:
        bottom = mask[-bh:, :]
        ratio_bottom = (bottom.sum() / 255.0) / float(bottom.size)

    # central band ratio and large-line heuristic on MSER mask
    cbh = int(Hs * 0.30)
    y0 = max(0, Hs // 2 - cbh // 2)
    central = mask[y0:y0 + cbh, :]
    ratio_central = (central.sum() / 255.0) / float(central.size) if cbh > 0 else 0.0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    large_line = False
    for i in range(1, num_labels):
        x, y, w, h, a = stats[i]
        if a / float(Hs * Ws) >= 0.005 and w >= 3 * h:
            large_line = True
            break
    bottom_max_h_rel = 0.0
    if bh > 0:
        num_b, lbl_b, st_b, _ = cv2.connectedComponentsWithStats((bottom > 0).astype(np.uint8), connectivity=8)
        for i in range(1, num_b):
            _, _, _, h, _ = st_b[i]
            bottom_max_h_rel = max(bottom_max_h_rel, float(h) / float(Hs))
    # connectivity in central band for MSER rectangles
    kwide = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, Ws // 30), 1))
    mask_conn = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kwide, iterations=1)
    central_conn = mask_conn[y0:y0 + cbh, :]
    ratio_central_conn = (central_conn.sum() / 255.0) / float(central_conn.size) if cbh > 0 else 0.0
    row_cov = ((central_conn > 0).sum(axis=1) / float(Ws)) if cbh > 0 else np.array([0.0])
    max_cov = float(row_cov.max()) if cbh > 0 else 0.0
    center_thr = max(0.5 * area_threshold, 0.04)
    central_flag = (ratio_central >= center_thr) or (ratio_central_conn >= max(0.6 * center_thr, 0.035)) or (max_cov >= 0.35)
    if subtitle_tolerant and not central_flag and bottom_max_h_rel <= subtitle_h_ratio:
        return False
    return central_flag or large_line or (ratio_bottom >= area_threshold and bottom_max_h_rel > subtitle_h_ratio) or (ratio_all >= area_threshold)


def load_person_detector(device: str):
    """Load a person detector.

    Tries torchvision Faster R-CNN (COCO) first; falls back to HOG people detector if download/instantiate fails.
    Returns a dict descriptor with keys: {'type': 'torchvision', 'model': model} or {'type': 'hog', 'hog': hog} or None.
    """
    # Try torchvision detection
    try:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        det_model = fasterrcnn_resnet50_fpn(weights=weights)
        det_model.eval().to(device)
        return {"type": "torchvision", "model": det_model}
    except Exception:
        pass
    # Fallback to HOG
    try:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        return {"type": "hog", "hog": hog}
    except Exception:
        return None


def person_area_ratio(detector: Optional[dict], frame_bgr: np.ndarray, device: str, score_thresh: float = 0.5) -> float:
    """Return max area ratio for detected persons in the frame.

    - TorchVision: use label==1 (COCO 'person') with score >= score_thresh
    - HOG: use detectMultiScale; take max area
    """
    if detector is None:
        return 0.0
    H, W = frame_bgr.shape[:2]
    if detector.get("type") == "torchvision":
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            out = detector["model"]([t.to(device)])[0]
        labels = out["labels"].detach().to("cpu").numpy()
        scores = out["scores"].detach().to("cpu").numpy()
        boxes = out["boxes"].detach().to("cpu").numpy()
        mask = (labels == 1) & (scores >= score_thresh)
        if not np.any(mask):
            return 0.0
        max_ratio = 0.0
        for (x1, y1, x2, y2) in boxes[mask]:
            w = max(0.0, float(x2 - x1))
            h = max(0.0, float(y2 - y1))
            max_ratio = max(max_ratio, (w * h) / float(W * H + 1e-9))
        return float(max_ratio)
    if detector.get("type") == "hog":
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        rects, _ = detector["hog"].detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
        if len(rects) == 0:
            return 0.0
        max_ratio = 0.0
        for (x, y, w, h) in rects:
            max_ratio = max(max_ratio, (float(w) * float(h)) / float(W * H + 1e-9))
        return float(max_ratio)
    return 0.0

def find_videos(input_path: str) -> List[str]:
    """Return a sorted list of video file paths.

    - If input_path is a directory, recursively find common video extensions.
    - If input_path is a file, return it as a single-element list.
    """
    exts = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpg", ".mpeg", ".m4v")
    if os.path.isdir(input_path):
        out: List[str] = []
        for root, _, files in os.walk(input_path):
            for fn in files:
                if fn.lower().endswith(exts):
                    out.append(os.path.join(root, fn))
        out.sort()
        return out
    return [input_path]

@dataclass
class GroupState:
    group_id: str
    start_index: int
    end_index: int
    best_quality: float
    best_frame_index: int
    best_frame_id: Optional[str]
    best_embedding: Optional[np.ndarray]
    best_saved_path: Optional[str]
    best_timestamp: float = 0.0


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance for L2-normalized vectors = 1 - dot(a, b)."""
    return float(1.0 - float(np.dot(a, b)))


def process_video(
    video_path: str,
    output_dir: str,
    db_path: str,
    cfg: ProcessingConfig,
    video_idx: int = 1,
    total_videos: int = 1,
) -> Tuple[str, list]:
    """Run the full pipeline for a single video.

    Returns (video_id, selected_frames_list) where selected_frames_list contains dicts with
    frame_index, timestamp, quality, frame_id.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_out_dir = os.path.join(output_dir, video_name)
    # Defer directory creation until after video-level filters pass

    # Read basic video metadata
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = (frame_count / fps) if (frame_count > 0 and fps > 0) else None
    cap.release()

    # Optional: video-level face prominence precheck BEFORE any DB writes or output creation
    if cfg.video_face_filter:
        try:
            cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
            face_cascade_pre = cv2.CascadeClassifier(cascade_path) if os.path.exists(cascade_path) else None
        except Exception:
            face_cascade_pre = None
        if face_cascade_pre is not None:
            hits = 0
            # Sample frames quickly to decide if any prominent face appears
            sample_skip = max(cfg.frame_skip, 10)
            for _, _, frm in extract_frames(video_path, frame_skip=sample_skip, max_frames=200):
                gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
                faces = face_cascade_pre.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                if len(faces) > 0:
                    areas = [(w * h) / float(width * height + 1e-9) for (x, y, w, h) in faces]
                    if max(areas) >= cfg.min_face_area:
                        hits += 1
                        if hits >= cfg.min_face_frames:
                            break
            if hits < cfg.min_face_frames:
                # Skip this video entirely
                return "SKIPPED", []

    # Create/clean per-video output dir now that precheck passed
    if os.path.exists(video_out_dir):
        shutil.rmtree(video_out_dir)
    os.makedirs(video_out_dir, exist_ok=True)
    # Images will be saved directly into video_out_dir (no 'selected' subfolder)
    skipped_dir = os.path.join(video_out_dir, "skipped")
    os.makedirs(skipped_dir, exist_ok=True)
    if not cfg.debug:
        print(f"[info] Streaming video '{video_name}': fps={fps:.2f}, frames={frame_count}, duration={duration_sec if duration_sec else 0:.1f}s (per-second best enabled)", flush=True)

    # Init storage and video record
    storage = DuckDBStorage(db_path)
    video_id = storage.get_or_create_video(video_path, width, height, fps, frame_count if frame_count > 0 else None, duration_sec)
    # Unique run identifier to avoid primary key collisions on reruns
    run_id = uuid.uuid4().hex

    # Load model for embeddings
    model, device, _, mean, std = load_model(device=cfg.device, pretrained=cfg.pretrained, backbone=cfg.backbone)

    # State for threshold grouping
    anchor_embedding: Optional[np.ndarray] = None
    anchor_frame_id: Optional[str] = None

    group_idx = 0
    group: Optional[GroupState] = None
    selected_summaries: list = []
    selected_entries: list = []
    # Face detector (Haar) if enabled (frame-level subject filter)
    face_cascade = None
    if cfg.face_filter:
        try:
            cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
            if os.path.exists(cascade_path):
                face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            face_cascade = None
    # Person detector (HOG) if enabled
    hog = None
    if cfg.person_filter:
        try:
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        except Exception:
            hog = None

    def start_new_group(start_frame_index: int) -> GroupState:
        nonlocal group_idx
        gid = f"{video_id}:{run_id}:g{group_idx:06d}"
        storage.upsert_group(group_id=gid, video_id=video_id, start_index=start_frame_index)
        group_idx += 1
        return GroupState(
            group_id=gid,
            start_index=start_frame_index,
            end_index=start_frame_index,
            best_quality=-1.0,
            best_frame_index=start_frame_index,
            best_frame_id=None,
            best_embedding=None,
            best_saved_path=None,
        )

    def finalize_group(g: GroupState) -> None:
        nonlocal anchor_embedding, anchor_frame_id
        # Set end_index, save group; Mark representative frame
        storage.upsert_group(group_id=g.group_id, video_id=video_id, start_index=g.start_index,
                             end_index=g.end_index, selected_frame_id=g.best_frame_id,
                             anchor_frame_id=g.best_frame_id)
        if g.best_frame_id:
            storage.mark_representative(g.best_frame_id)
            anchor_frame_id = g.best_frame_id
        # Update anchor embedding for next group's thresholding
        if g.best_embedding is not None:
            anchor_embedding = g.best_embedding.copy()

    # Track best frame per second (quality-only); compute embedding only for that one
    current_sec: Optional[int] = None
    sec_best_idx: Optional[int] = None
    sec_best_ts: float = 0.0
    sec_best_img: Optional[np.ndarray] = None
    sec_best_q: float = -1.0
    committed_count: int = 0
    last_debug_print_sec: int = -10
    blur_skipped: int = 0
    text_skipped: int = 0
    # Skipped-image uniqueness memory (per video)
    skip_hashes: list[int] = []  # aHash64
    skip_vecs: list[np.ndarray] = []  # 32x32 grayscale vector (L2-normalized)

    def _ahash64(img: np.ndarray) -> int:
        try:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception:
            return 0
        small = cv2.resize(g, (8, 8), interpolation=cv2.INTER_AREA)
        m = float(small.mean())
        bits = (small > m).astype(np.uint8).flatten()
        h = 0
        for i in range(64):
            if int(bits[i]):
                h |= (1 << i)
        return int(h)

    def _vec32(img: np.ndarray) -> np.ndarray:
        try:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception:
            return np.zeros((1024,), dtype=np.float32)
        small = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        v = small.reshape(-1)
        v -= float(v.mean())
        std = float(v.std()) + 1e-6
        v /= std
        n = float(np.linalg.norm(v)) + 1e-6
        return (v / n).astype(np.float32)

    def _skip_should_save(img: np.ndarray, thr_hash: int = 6, thr_cos: float = 0.10) -> bool:
        # Hash check
        h = _ahash64(img)
        for prev in skip_hashes:
            if (h ^ prev).bit_count() <= thr_hash:
                return False
        # Cosine check on low-dim vector
        v = _vec32(img)
        if len(skip_vecs) > 0:
            dots = np.dot(np.vstack(skip_vecs), v)
            dists = 1.0 - dots
            if float(dists.min()) <= thr_cos:
                return False
        # Record
        skip_hashes.append(h)
        skip_vecs.append(v)
        return True

    def save_skipped(img: np.ndarray, idx: int, ts: float, reason: str) -> None:
        try:
            if not _skip_should_save(img):
                return
            fname = f"{idx:09d}_{reason}.jpg"
            cv2.imwrite(os.path.join(skipped_dir, fname), img, [int(cv2.IMWRITE_JPEG_QUALITY), cfg.save_jpeg_quality])
        except Exception:
            pass

    def commit_sec_best(end: bool = False) -> None:
        nonlocal group, anchor_embedding, anchor_frame_id
        nonlocal sec_best_idx, sec_best_ts, sec_best_img, sec_best_q
        nonlocal committed_count, last_debug_print_sec, text_skipped
        if sec_best_idx is None or sec_best_img is None:
            return

        # Text filter at commit stage (best-per-second only)
        if False and cfg.text_filter:
            if is_text_prominent(
                sec_best_img,
                cfg.text_area_threshold,
                cfg.text_bottom_ratio,
                method=getattr(cfg, 'text_method', 'fast'),
                subtitle_tolerant=getattr(cfg, 'subtitle_tolerant', True),
                subtitle_h_ratio=getattr(cfg, 'subtitle_max_rel_height', 0.12),
            ):
                save_skipped(sec_best_img, sec_best_idx, sec_best_ts, "text")
                text_skipped += 1
                if cfg.debug:
                    cur_s = int(sec_best_ts)
                    if end or (cur_s - last_debug_print_sec >= 10):
                        print(f"[debug] [{video_idx}/{total_videos}] {video_name}: t={cur_s}s, skipped due to text", flush=True)
                        last_debug_print_sec = cur_s
                return

        # Compute embedding for the best frame in the previous second
        emb = compute_embedding(model, sec_best_img, device, cfg.resize_shorter, cfg.model_input_size, mean, std)
        q = sec_best_q

        # Decide if this frame starts a new group
        start_new = False
        if anchor_embedding is None:
            start_new = True
        else:
            dist = cosine_distance(anchor_embedding, emb)
            start_new = dist > cfg.distance_threshold

        if start_new:
            # finalize previous group's representative and record it
            if group is not None:
                old = group
                finalize_group(old)
                if old.best_frame_id is not None and old.best_embedding is not None:
                    selected_entries.append({
                        "frame_index": old.best_frame_index,
                        "timestamp": old.best_timestamp,
                        "quality": old.best_quality,
                        "frame_id": old.best_frame_id,
                        "embedding": old.best_embedding,
                        "image_path": old.best_saved_path,
                    })
            # start a new group anchored at this frame index
            group = start_new_group(sec_best_idx)

        assert group is not None  # for type checker
        group.end_index = sec_best_idx

        # Persist frame + embedding immediately (minimal RAM)
        frame_id = storage.insert_frame(video_id=video_id, frame_index=sec_best_idx,
                                        timestamp_sec=sec_best_ts, quality=q,
                                        group_id=group.group_id, is_representative=False,
                                        run_id=run_id)
        storage.insert_embedding(frame_id=frame_id, emb=emb)

        # Update best-of-group representative if quality improved
        if q > group.best_quality:
            group.best_quality = q
            group.best_frame_index = sec_best_idx
            group.best_frame_id = frame_id
            group.best_embedding = emb
            group.best_timestamp = sec_best_ts
            # Save/overwrite the currently-best frame image for this group
            save_path = os.path.join(video_out_dir, f"{group.group_id.replace(':','_')}.jpg")
            cv2.imwrite(save_path, sec_best_img, [int(cv2.IMWRITE_JPEG_QUALITY), cfg.save_jpeg_quality])
            group.best_saved_path = save_path
        committed_count += 1
        if cfg.debug:
            cur_s = int(sec_best_ts)
            if end or (cur_s - last_debug_print_sec >= 10):
                print(f"[debug] [{video_idx}/{total_videos}] {video_name}: t={cur_s}s, committed={committed_count}", flush=True)
                last_debug_print_sec = cur_s

    # Iterate frames streaming; keep only the best-quality frame per second
    for frame_index, timestamp_sec, frame_bgr in extract_frames(video_path, frame_skip=cfg.frame_skip, max_frames=cfg.max_frames):
        # Determine current second bucket
        s_bucket = int(timestamp_sec) if timestamp_sec is not None else (int(frame_index / fps) if fps and fps > 0 else frame_index)
        if current_sec is None:
            current_sec = s_bucket
        elif s_bucket != current_sec:
            # Commit the best frame from the previous second
            commit_sec_best(end=False)
            # Reset best-of-second tracking
            sec_best_idx = None
            sec_best_ts = 0.0
            sec_best_img = None
            sec_best_q = -1.0
            current_sec = s_bucket

        # Optional filters: person, face, and blur before considering as candidate (text checked at commit stage)
        if cfg.person_filter and hog is not None:
            H, W = frame_bgr.shape[:2]
            gray_p = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            rects, _ = hog.detectMultiScale(gray_p, winStride=(8, 8), padding=(8, 8), scale=1.05)
            if len(rects) == 0:
                save_skipped(frame_bgr, frame_index, float(timestamp_sec), "no_person")
                continue
            max_ratio = 0.0
            for (x, y, w, h) in rects:
                max_ratio = max(max_ratio, (float(w) * float(h)) / float(W * H + 1e-9))
            if max_ratio < cfg.min_person_area:
                save_skipped(frame_bgr, frame_index, float(timestamp_sec), "person_small")
                continue
        if cfg.face_filter and face_cascade is not None:
            gray_face = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            if len(faces) == 0:
                save_skipped(frame_bgr, frame_index, float(timestamp_sec), "no_face")
                continue
            areas = [(w * h) / float(frame_bgr.shape[0] * frame_bgr.shape[1] + 1e-9) for (x, y, w, h) in faces]
            if max(areas) < cfg.min_face_area:
                save_skipped(frame_bgr, frame_index, float(timestamp_sec), "face_small")
                continue
        # Text filter applied once per second during commit

        if cfg.min_sharpness and cfg.min_sharpness > 0:
            gray_sharp = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            lap = cv2.Laplacian(gray_sharp, cv2.CV_64F)
            lv = float(lap.var())
            if lv < cfg.min_sharpness:
                save_skipped(frame_bgr, frame_index, float(timestamp_sec), "blur")
                blur_skipped += 1
                continue

        # Compute quality (cheap) and keep only the best within this second
        q = compute_quality(frame_bgr, alpha=cfg.quality_alpha, beta=cfg.quality_beta)
        if q > sec_best_q:
            sec_best_idx = frame_index
            sec_best_ts = float(timestamp_sec)
            # copy to avoid later in-place modifications by cv2
            sec_best_img = frame_bgr.copy()
            sec_best_q = q

    # End of stream: commit the last second and finalize last group
    commit_sec_best(end=True)
    if group is not None:
        finalize_group(group)
        if group.best_frame_id is not None and group.best_embedding is not None:
            selected_entries.append({
                "frame_index": group.best_frame_index,
                "timestamp": group.best_timestamp,
                "quality": group.best_quality,
                "frame_id": group.best_frame_id,
                "embedding": group.best_embedding,
                "image_path": group.best_saved_path,
            })

    # Post-process: suppress near-duplicate selected frames (vectorized)
    def deduplicate_selected(entries: list) -> list:
        total = len(entries)
        kept: list = []
        kept_embs: Optional[np.ndarray] = None  # shape (k, D)
        for idx, s in enumerate(entries):
            emb = s["embedding"]
            if kept_embs is None or kept_embs.shape[0] == 0:
                kept.append(s)
                kept_embs = emb[None, :]
                storage.set_representative(s["frame_id"], True)
            else:
                # compute distances to all kept in one shot
                dots = kept_embs @ emb
                dists = 1.0 - dots
                j = int(np.argmin(dists))
                if dists[j] < cfg.dedup_threshold:
                    k = kept[j]
                    if s["quality"] > k["quality"]:
                        storage.set_representative(k["frame_id"], False)
                        if k.get("image_path") and os.path.exists(k["image_path"]):
                            try:
                                _im = cv2.imread(k["image_path"]) if os.path.exists(k["image_path"]) else None
                                if _im is not None and _skip_should_save(_im):
                                    dst = os.path.join(skipped_dir, f"duplicate_{os.path.basename(k['image_path'])}")
                                    shutil.move(k["image_path"], dst)
                                else:
                                    os.remove(k["image_path"])  # drop duplicate skipped image
                            except Exception:
                                pass
                        kept[j] = s
                        kept_embs[j, :] = emb
                        storage.set_representative(s["frame_id"], True)
                    else:
                        storage.set_representative(s["frame_id"], False)
                        if s.get("image_path") and os.path.exists(s["image_path"]):
                            try:
                                _im = cv2.imread(s["image_path"]) if os.path.exists(s["image_path"]) else None
                                if _im is not None and _skip_should_save(_im):
                                    dst = os.path.join(skipped_dir, f"duplicate_{os.path.basename(s['image_path'])}")
                                    shutil.move(s["image_path"], dst)
                                else:
                                    os.remove(s["image_path"])  # drop duplicate skipped image
                            except Exception:
                                pass
                else:
                    kept.append(s)
                    kept_embs = np.vstack([kept_embs, emb]) if kept_embs is not None else emb[None, :]
                    storage.set_representative(s["frame_id"], True)
        if cfg.debug:
            print(f"[debug] [{video_idx}/{total_videos}] {video_name}: dedup kept {len(kept)} / {total}", flush=True)
        return kept

    final_entries = deduplicate_selected(selected_entries) if cfg.dedup_threshold and cfg.dedup_threshold > 0 else selected_entries

    # Final blur prune on saved outputs
    removed_blurry = 0
    pruned_entries: list = []
    for r in final_entries:
        imgp = r.get("image_path")
        keep = True
        if imgp and os.path.exists(imgp):
            img = cv2.imread(imgp)
            if img is not None:
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                lv = float(cv2.Laplacian(g, cv2.CV_64F).var())
                if cfg.min_sharpness and lv < cfg.min_sharpness:
                    storage.set_representative(r["frame_id"], False)
                    try:
                        if _skip_should_save(img):
                            dst = os.path.join(skipped_dir, f"blur_{os.path.basename(imgp)}")
                            shutil.move(imgp, dst)
                        else:
                            os.remove(imgp)
                    except Exception:
                        pass
                    keep = False
                    removed_blurry += 1
        if keep:
            pruned_entries.append(r)
    if removed_blurry > 0:
        print(f"[prune] Removed {removed_blurry} blurred images below threshold {cfg.min_sharpness}", flush=True)
    final_entries = pruned_entries
    # Final stats
    print(f"[stats] [{video_idx}/{total_videos}] {video_name}: skipped_blur={blur_skipped}, pruned_blur={removed_blurry}, skipped_text={text_skipped}, final={len(final_entries)}", flush=True)

    # Emit CSV summary for convenience
    csv_path = os.path.join(video_out_dir, "summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("frame_index,timestamp,quality,frame_id\n")
        for r in final_entries:
            f.write(f"{r['frame_index']},{r['timestamp']:.6f},{r['quality']:.6f},{r['frame_id']}\n")
    if cfg.debug:
        print(f"[debug] [{video_idx}/{total_videos}] {video_name}: wrote summary {csv_path}", flush=True)
    # Graceful shutdown of DB connection
    try:
        storage.close()
    except Exception:
        pass

    return video_id, final_entries


# -------------------------------
# CLI
# -------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract distinct high-quality keyframes with minimal RAM using PyTorch + DuckDB")
    p.add_argument("--input", "--corpus", dest="input", required=True, help="Path to input video file or directory (recursive); alias: --corpus")
    p.add_argument("--output", required=True, help="Output directory for selected frames and summary")
    p.add_argument("--db", required=True, help="Path to DuckDB database file")

    p.add_argument("--distance-threshold", type=float, default=0.25, help="Cosine distance threshold to start a new group (default 0.25)")
    p.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame (default 1)")
    p.add_argument("--resize-shorter", type=int, default=256, help="Resize shorter image side before center-crop (default 256)")
    p.add_argument("--model-input-size", type=int, default=224, help="Model input crop size (default 224)")
    p.add_argument("--quality-alpha", type=float, default=1.0, help="Weight for Laplacian variance sharpness score (default 1.0)")
    p.add_argument("--quality-beta", type=float, default=0.0, help="Optional weight for grayscale std contrast term (default 0.0)")
    p.add_argument("--device", type=str, default=None, help="Force device: 'cuda', 'mps', or 'cpu' (default: auto)")
    p.add_argument("--no-pretrained", action="store_true", help="Disable pretrained weights (fallback to random)")
    p.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality for saving selected frames (default 95)")
    p.add_argument("--max-frames", type=int, default=None, help="Optional cap on number of processed frames (debug)")
    p.add_argument("--backbone", type=str, default="resnet50", help="timm model name (e.g., 'resnet50', 'efficientnet_b0')")
    p.add_argument("--dedup-threshold", type=float, default=0.07, help="Cosine distance threshold to treat two selected frames as duplicates (default 0.07)")
    # Filters
    try:
        boolean_action = argparse.BooleanOptionalAction  # Python 3.9+
    except Exception:
        boolean_action = None
    if boolean_action is not None:
        p.add_argument("--face-filter", action=boolean_action, default=False, help="Require at least one detected face per frame (default: disabled)")
        p.add_argument("--video-face-filter", action=boolean_action, default=False, help="Skip entire video if no prominent face appears (default: disabled)")
        p.add_argument("--person-filter", action=boolean_action, default=True, help="Require at least one prominent person per frame (OpenCV HOG) (default: enabled)")
        p.add_argument("--text-filter", action=boolean_action, default=False, help="Skip frames with prominent text overlays (default: disabled)")
        p.add_argument("--subtitle-tolerant", action=boolean_action, default=True, help="Do not treat short bottom subtitles as prominent text (default: enabled)")
    else:
        p.add_argument("--face-filter", action="store_true", help="Require at least one detected face per frame")
        p.add_argument("--video-face-filter", action="store_true", help="Skip entire video if no prominent face appears")
        p.add_argument("--person-filter", action="store_true", help="Require at least one prominent person per frame (OpenCV HOG)")
        p.add_argument("--text-filter", action="store_true", help="Skip frames with prominent text overlays")
        p.add_argument("--subtitle-tolerant", action="store_true", help="Do not treat short bottom subtitles as prominent text")
    p.add_argument("--min-sharpness", type=float, default=50.0, help="Discard frames with Laplacian variance below this (default 50.0)")
    p.add_argument("--min-face-area", type=float, default=0.015, help="Minimum face area ratio (bbox area / frame area) to be considered prominent (default 0.015)")
    p.add_argument("--min-person-area", type=float, default=0.08, help="Minimum person bbox area ratio to be considered prominent (default 0.08)")
    p.add_argument("--text-method", type=str, default="fast", choices=["fast", "mser"], help="Text detection method: 'fast' (edge) or 'mser' (slower)")
    p.add_argument("--text-area-threshold", type=float, default=0.08, help="Text area ratio threshold to treat overlays as prominent (default 0.08)")
    p.add_argument("--text-bottom-ratio", type=float, default=0.4, help="Bottom region proportion emphasized for subtitles (default 0.4)")
    p.add_argument("--subtitle-max-rel-height", type=float, default=0.12, help="Max relative height of bottom text to be tolerated as subtitles (default 0.12)")
    p.add_argument("--min-face-frames", type=int, default=1, help="Require at least this many frames with a prominent face to process video (default 1)")
    p.add_argument("--debug", action="store_true", help="Enable debug logging at 10s intervals and on completion")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = ProcessingConfig(
        distance_threshold=args.distance_threshold,
        frame_skip=args.frame_skip,
        resize_shorter=args.resize_shorter,
        model_input_size=args.model_input_size,
        quality_alpha=args.quality_alpha,
        quality_beta=args.quality_beta,
        device=args.device,
        pretrained=not args.no_pretrained,
        save_jpeg_quality=args.jpeg_quality,
        max_frames=args.max_frames,
        backbone=args.backbone,
        dedup_threshold=args.dedup_threshold,
        face_filter=getattr(args, 'face_filter', True),
        min_sharpness=args.min_sharpness,
        video_face_filter=getattr(args, 'video_face_filter', True),
        min_face_area=args.min_face_area,
        min_face_frames=args.min_face_frames,
        debug=getattr(args, 'debug', False),
        person_filter=getattr(args, 'person_filter', True),
        min_person_area=args.min_person_area,
        text_filter=getattr(args, 'text_filter', False),
        text_area_threshold=args.text_area_threshold,
        text_bottom_ratio=args.text_bottom_ratio,
        text_method=getattr(args, 'text_method', 'fast'),
        subtitle_tolerant=getattr(args, 'subtitle_tolerant', True),
        subtitle_max_rel_height=args.subtitle_max_rel_height,
    )

    inputs = find_videos(args.input)
    if not inputs:
        raise RuntimeError(f"No videos found under: {args.input}")

    print(f"# Processing {len(inputs)} video(s)")
    grand_total = 0
    total_time = 0.0
    for i, vp in enumerate(inputs, start=1):
        t0 = time.time()
        video_id, selected = process_video(vp, args.output, args.db, cfg, video_idx=i, total_videos=len(inputs))
        dt = time.time() - t0
        total_time += dt
        grand_total += len(selected)

        print("\n# Results (per video)")
        print(f"[{i}/{len(inputs)}] Input: {vp}")
        print(f"Video ID: {video_id}")
        if video_id == "SKIPPED":
            print("Skipped: no prominent faces detected (per video filter)")
        else:
            print(f"Selected {len(selected)} frames in {dt:.2f}s")
            for r in selected:
                print(f"- frame_index={r['frame_index']} ts={r['timestamp']:.3f}s quality={r['quality']:.2f} frame_id={r['frame_id']}")
            video_name = os.path.splitext(os.path.basename(vp))[0]
            video_out_dir = os.path.join(args.output, video_name)
            print(f"Outputs: images_dir={video_out_dir} summary={os.path.join(video_out_dir, 'summary.csv')}")

    print(f"\n# Aggregate")
    print(f"Videos processed: {len(inputs)} | Total selected: {grand_total} | Total time: {total_time:.2f}s")
    print(f"DuckDB: {args.db}")


if __name__ == "__main__":
    main()

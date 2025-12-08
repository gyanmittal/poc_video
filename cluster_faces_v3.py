from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import torch
import timm
import duckdb
from PIL import Image
import imagehash


# -------------------------------
# Config
# -------------------------------

@dataclass
class FaceClusterConfig:
    corpus: str
    output: str

    device: Optional[str] = None
    pretrained: bool = True

    # Detection
    detector: str = "haar"  # 'haar' or 'mtcnn'
    min_face_rel_size: float = 0.06

    # Face quality
    face_min_area_ratio: float = 0.02
    face_min_sharpness: float = 60.0

    # Sampling
    sample_fps: float = 2.0

    # Embedding
    embedder: str = "timm"  # 'timm' or 'facenet'
    backbone: str = "resnet50"
    input_size: int = 224

    # Clustering
    cluster_scope: str = "per_video"  # 'global' or 'per_video'
    similarity_threshold: float = 0.30
    min_cluster_size: int = 3

    # Output control
    save_representative_only: bool = False
    clean_on_rerun: bool = True

    # Clips
    make_clips: bool = True
    clip_pre_sec: float = 0.0
    clip_post_sec: float = 0.0
    clip_min_dur_sec: float = 0.5
    clip_merge_gap_sec: float = 0.5
    ffmpeg_reencode: bool = True

    # Performance
    fp16: bool = False
    embed_batch_size: int = 16
    defer_crop_save: bool = False
    detect_every_k: int = 5
    track_between_detects: bool = True
    tracker_type: str = "KCF"
    max_tracks: int = 8
    adaptive_sampling: bool = False
    adapt_boost_fps: float = 4.0
    adapt_boost_secs: float = 2.0

    # Debug/progress
    debug: bool = False
    progress_only: bool = False
    max_video_secs: float = 0.0

    # DB dedup
    db_enable: bool = False
    db_path: Optional[str] = None
    dedup_method: str = "both"
    dedup_hash_thresh: int = 6
    dedup_emb_thresh: float = 0.995
    dedup_window_sec: float = 5.0

    # V3 annotations
    annotate_clips: bool = False
    annotate_thickness: int = 3
    annotate_color: str = "0,255,0"  # BGR string
    body_expand_w: float = 1.8
    body_expand_h: float = 3.0


# -------------------------------
# IO helpers
# -------------------------------

def list_videos(path_or_dir: str) -> List[str]:
    exts = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpg", ".mpeg", ".m4v")
    if os.path.isdir(path_or_dir):
        out: List[str] = []
        for r, _, files in os.walk(path_or_dir):
            for f in files:
                if f.lower().endswith(exts):
                    out.append(os.path.join(r, f))
        out.sort()
        return out
    return [path_or_dir]


# -------------------------------
# Detection
# -------------------------------

def load_haar_detector() -> Optional[cv2.CascadeClassifier]:
    try:
        cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        if not os.path.exists(cascade_path):
            cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_alt2.xml')
        if os.path.exists(cascade_path):
            return cv2.CascadeClassifier(cascade_path)
    except Exception:
        pass
    return None


def detect_faces_haar(img_bgr: np.ndarray, detector: Optional[cv2.CascadeClassifier], min_rel: float = 0.06) -> List[Tuple[int, int, int, int]]:
    if detector is None:
        return []
    H, W = img_bgr.shape[:2]
    minsz = int(min(H, W) * float(min_rel))
    minsz = max(24, minsz)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(minsz, minsz))
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


# Optional MTCNN

def load_mtcnn(device: Optional[str] = None):
    try:
        from facenet_pytorch import MTCNN  # type: ignore
        if device == "mps":
            dev = "cpu"
        elif device is not None:
            dev = device
        else:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        return MTCNN(keep_all=True, device=dev)
    except Exception:
        return None


def detect_faces_mtcnn(img_bgr: np.ndarray, mtcnn, min_rel: float = 0.06) -> List[Tuple[int, int, int, int]]:
    if mtcnn is None:
        return []
    H, W = img_bgr.shape[:2]
    minsz = int(min(H, W) * float(min_rel))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    boxes, probs = mtcnn.detect(img_rgb)
    out: List[Tuple[int, int, int, int]] = []
    if boxes is None or probs is None:
        return out
    for (bx, by, ex, ey), p in zip(boxes, probs):
        if p is None or p < 0.90:
            continue
        x0, y0, x1, y1 = int(max(0, bx)), int(max(0, by)), int(min(W, ex)), int(min(H, ey))
        w, h = max(0, x1 - x0), max(0, y1 - y0)
        if w < minsz or h < minsz:
            continue
        out.append((x0, y0, w, h))
    return out


def detect_faces(img_bgr: np.ndarray, cfg: FaceClusterConfig, haar, mtcnn) -> List[Tuple[int, int, int, int]]:
    if cfg.detector == 'mtcnn' and mtcnn is not None:
        return detect_faces_mtcnn(img_bgr, mtcnn, min_rel=cfg.min_face_rel_size)
    return detect_faces_haar(img_bgr, haar, min_rel=cfg.min_face_rel_size)


def crop_with_margin(img: np.ndarray, x: int, y: int, w: int, h: int, margin: float = 0.25) -> np.ndarray:
    H, W = img.shape[:2]
    cx, cy = x + w / 2.0, y + h / 2.0
    size = max(w, h)
    size = int(size * (1.0 + margin))
    x0 = max(0, int(cx - size / 2))
    y0 = max(0, int(cy - size / 2))
    x1 = min(W, x0 + size)
    y1 = min(H, y0 + size)
    return img[y0:y1, x0:x1]


def dprint(cfg: FaceClusterConfig, msg: str) -> None:
    if getattr(cfg, 'debug', False) and not getattr(cfg, 'progress_only', False):
        try:
            print(msg)
        except Exception:
            pass


def create_tracker(tracker_type: str):
    t = tracker_type.upper()
    try:
        if t == 'CSRT':
            if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
                return cv2.legacy.TrackerCSRT_create()
            if hasattr(cv2, 'TrackerCSRT_create'):
                return cv2.TrackerCSRT_create()
        if t == 'MOSSE':
            if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerMOSSE_create'):
                return cv2.legacy.TrackerMOSSE_create()
            if hasattr(cv2, 'TrackerMOSSE_create'):
                return cv2.TrackerMOSSE_create()
        if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create'):
            return cv2.legacy.TrackerKCF_create()
        if hasattr(cv2, 'TrackerKCF_create'):
            return cv2.TrackerKCF_create()
    except Exception:
        return None
    return None


def save_crop_from_video(video_path: str, frame_idx: int, bbox: Tuple[int, int, int, int], dst: str) -> bool:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
    except Exception:
        pass
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    x, y, w, h = bbox
    crop = crop_with_margin(frame, x, y, w, h, margin=0.25)
    try:
        cv2.imwrite(dst, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return True
    except Exception:
        return False


# -------------------------------
# Embedding
# -------------------------------

class FaceEmbedder:
    def __init__(self, device: Optional[str] = None, embedder: str = "timm", backbone: str = "resnet50", input_size: int = 224, pretrained: bool = True, use_fp16: bool = False):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.type = None
        self.model = None
        self.size = input_size
        self.mean = None
        self.std = None
        self._facenet_norm = False
        self.use_fp16 = bool(use_fp16)

        if embedder == "facenet":
            try:
                from facenet_pytorch import InceptionResnetV1  # type: ignore
                self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
                self.type = "facenet"
                self.size = 160
                self._facenet_norm = True
            except Exception:
                pass
        if self.model is None:
            self.model = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool='avg')
            self.model.eval().to(self.device)
            self.type = "timm"
            data_cfg = timm.data.resolve_model_data_config(self.model)
            self.size = data_cfg['input_size'][1]
            self.mean = np.array(data_cfg['mean'], dtype=np.float32)
            self.std = np.array(data_cfg['std'], dtype=np.float32)

    def embed_bgr(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        if img_bgr is None or img_bgr.size == 0:
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if self.type == "facenet":
            im = cv2.resize(img_rgb, (self.size, self.size), interpolation=cv2.INTER_AREA)
            t = torch.from_numpy(im).float() / 255.0
            t = (t - 0.5) / 0.5
            t = t.permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.model(t)
            v = emb.detach().to('cpu').numpy().astype(np.float32)[0]
            n = np.linalg.norm(v) + 1e-9
            return (v / n).astype(np.float32)
        else:
            im = cv2.resize(img_rgb, (self.size, self.size), interpolation=cv2.INTER_AREA)
            t = torch.from_numpy(im).float() / 255.0
            t = t.permute(2, 0, 1).unsqueeze(0)
            if self.mean is not None and self.std is not None:
                mean = torch.tensor(self.mean[:, None, None])
                std = torch.tensor(self.std[:, None, None])
                t = (t - mean) / std
            t = t.to(self.device)
            with torch.no_grad():
                emb = self.model(t)
            v = emb.detach().to('cpu').numpy().astype(np.float32)[0]
            n = np.linalg.norm(v) + 1e-9
            return (v / n).astype(np.float32)

    def embed_batch_bgr(self, imgs_bgr: List[np.ndarray]) -> np.ndarray:
        if len(imgs_bgr) == 0:
            return np.zeros((0, 1), dtype=np.float32)
        imgs_rgb = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs_bgr]
        if self.type == "facenet":
            arr = [cv2.resize(im, (self.size, self.size), interpolation=cv2.INTER_AREA) for im in imgs_rgb]
            t = torch.from_numpy(np.stack(arr)).float() / 255.0
            t = (t - 0.5) / 0.5
            t = t.permute(0, 3, 1, 2).to(self.device)
            with torch.no_grad():
                emb = self.model(t)
            v = emb.detach().to('cpu').numpy().astype(np.float32)
            n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
            return (v / n).astype(np.float32)
        else:
            arr = [cv2.resize(im, (self.size, self.size), interpolation=cv2.INTER_AREA) for im in imgs_rgb]
            t = torch.from_numpy(np.stack(arr)).float() / 255.0
            t = t.permute(0, 3, 1, 2)
            if self.mean is not None and self.std is not None:
                mean = torch.tensor(self.mean[None, :, None, None])
                std = torch.tensor(self.std[None, :, None, None])
                t = (t - mean) / std
            t = t.to(self.device)
            use_amp = self.use_fp16 and (self.device.type == 'cuda')
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    with torch.no_grad():
                        emb = self.model(t)
            else:
                with torch.no_grad():
                    emb = self.model(t)
            v = emb.detach().to('cpu').numpy().astype(np.float32)
            n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
            return (v / n).astype(np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# Dedup helpers (phash + DuckDB)

def compute_phash_bgr(img_bgr: np.ndarray) -> str:
    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        img_rgb = img_bgr
    im = Image.fromarray(img_rgb)
    try:
        h = imagehash.phash(im)
    except Exception:
        h = imagehash.average_hash(im)
    return str(h)


class FaceCacheDB:
    def __init__(self, path: str):
        self.con = duckdb.connect(path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS faces(
                face_id TEXT PRIMARY KEY,
                video_id TEXT,
                frame_idx INTEGER,
                t REAL,
                x INTEGER,
                y INTEGER,
                w INTEGER,
                h INTEGER,
                sharp REAL,
                area REAL,
                face_hash TEXT,
                face_emb BLOB,
                img_path TEXT
            );
            """
        )
        self.con.execute("ALTER TABLE faces ADD COLUMN IF NOT EXISTS run_id TEXT;")

    def recent_faces(self, video_id: str, t: float, window_sec: float, run_id: Optional[str]):
        try:
            if not run_id:
                return []
            t0 = float(t) - float(window_sec)
            t1 = float(t) + float(window_sec)
            cur = self.con.execute(
                "SELECT face_hash, face_emb FROM faces WHERE video_id = ? AND run_id = ? AND t BETWEEN ? AND ?",
                [video_id, run_id, t0, t1],
            )
            rows = cur.fetchall()
            return rows if rows is not None else []
        except Exception:
            return []

    def insert_face(self, video_id: str, frame_idx: int, t: float,
                    x: int, y: int, w: int, h: int,
                    face_hash: str, face_emb_bytes: bytes, img_path: str, run_id: Optional[str]) -> None:
        try:
            face_id = f"{video_id}:{frame_idx}:{int(round(t*1000))}:{x}:{y}:{w}:{h}"
            self.con.execute(
                """
                INSERT INTO faces(face_id, video_id, frame_idx, t, x, y, w, h, sharp, area, face_hash, face_emb, img_path, run_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?)
                """,
                [face_id, video_id, int(frame_idx), float(t), int(x), int(y), int(w), int(h), face_hash, face_emb_bytes, img_path, run_id],
            )
        except Exception:
            pass


# -------------------------------
# Processing
# -------------------------------

@dataclass
class FaceEntry:
    video: str
    frame_idx: int
    timestamp: float
    bbox: Tuple[int, int, int, int]
    face_path: str
    emb: np.ndarray


def parse_color_bgr(color_str: str) -> Tuple[int, int, int]:
    try:
        parts = [int(x.strip()) for x in color_str.split(',')]
        if len(parts) == 3:
            b, g, r = parts[0], parts[1], parts[2]
            return (b, g, r)
    except Exception:
        pass
    return (0, 255, 0)


def expand_to_body_bbox(face_bbox: Tuple[int, int, int, int], W: int, H: int, cfg: FaceClusterConfig) -> Tuple[int, int, int, int]:
    x, y, w, h = face_bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    new_w = int(max(1, w * float(cfg.body_expand_w)))
    new_h = int(max(1, h * float(cfg.body_expand_h)))
    # bias downward to include torso
    y0 = int(max(0, y - 0.3 * h))
    y1 = y0 + new_h
    if y1 > H:
        y1 = H
        y0 = max(0, H - new_h)
    x0 = int(max(0, int(cx - new_w / 2)))
    x1 = x0 + new_w
    if x1 > W:
        x1 = W
        x0 = max(0, W - new_w)
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def annotate_segment_opencv(video_src: str,
                            start_sec: float,
                            end_sec: float,
                            cluster_frame_boxes: Dict[int, Tuple[int, int, int, int]],
                            cfg: FaceClusterConfig,
                            out_path: str) -> bool:
    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        return False
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0 or W <= 0 or H <= 0:
        cap.release()
        return False
    start_f = max(0, int(round(start_sec * fps)))
    end_f = max(start_f, int(round(end_sec * fps)))

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    except Exception:
        pass

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    tmp_out = out_path + ".tmp.mp4"
    writer = cv2.VideoWriter(tmp_out, fourcc, fps, (W, H))
    if not writer.isOpened():
        cap.release()
        return False

    color = parse_color_bgr(cfg.annotate_color)
    thickness = max(1, int(cfg.annotate_thickness))

    # Prepare tracker
    tracker = None
    cur_idx = start_f
    # Precompute sorted detection frames for quick nearest search
    det_frames = sorted(cluster_frame_boxes.keys())
    det_ptr = 0
    last_bb_body: Optional[Tuple[int, int, int, int]] = None
    linger_frames = max(1, int(0.5 * fps))
    linger_left = 0

    iou_thr_nd = 0.10
    iou_thr_prev = 0.30
    max_nd_window = max(1, int(1.5 * fps))
    max_track_without_det_frames = max(1, int(2.0 * fps))
    last_trk_face = None
    last_present_idx = start_f - 1

    def nearest_detection(frame_idx: int, max_window: int = int(2.0 * fps)) -> Optional[Tuple[int, Tuple[int, int, int, int]]]:
        if not det_frames:
            return None
        # Binary search for closest index
        lo, hi = 0, len(det_frames) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if det_frames[mid] < frame_idx:
                lo = mid + 1
            else:
                hi = mid
        best_f = det_frames[lo]
        # Check neighbor
        if lo > 0 and abs(det_frames[lo - 1] - frame_idx) < abs(best_f - frame_idx):
            best_f = det_frames[lo - 1]
        if abs(best_f - frame_idx) <= max_window:
            return best_f, cluster_frame_boxes[best_f]
        return None

    # Initialize from nearest detection at start
    init = nearest_detection(start_f)
    if init is not None:
        tracker = create_tracker(cfg.tracker_type)
        if tracker is not None:
            fidx0, bb0 = init
            try:
                ok_seek = True
                if fidx0 != start_f:
                    try:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx0)
                        cur_idx = fidx0
                    except Exception:
                        ok_seek = False
                ok_read, frame = cap.read()
                if not ok_read:
                    ok_seek = False
                if ok_seek and frame is not None:
                    tracker.init(frame, (float(bb0[0]), float(bb0[1]), float(bb0[2]), float(bb0[3])))
            except Exception:
                tracker = None
    # Seek back to start for output
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        cur_idx = start_f
    except Exception:
        pass

    while cur_idx < end_f:

        ok, frame = cap.read()

        if not ok or frame is None:

            break

        det_box = cluster_frame_boxes.get(cur_idx)

        # Reinit tracker on detection at this exact frame

        if det_box is not None:

            tr2 = create_tracker(cfg.tracker_type)

            if tr2 is not None:

                try:

                    tr2.init(frame, (float(det_box[0]), float(det_box[1]), float(det_box[2]), float(det_box[3])))

                    tracker = tr2

                except Exception:

                    pass

        present_now = False

        to_draw = None

        bb_trk = None

        if tracker is not None:

            try:

                ok2, bb = tracker.update(frame)

            except Exception:

                ok2, bb = False, None

            if ok2 and bb is not None:

                bb_trk = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))

        if det_box is not None:

            bx, by, bw, bh = expand_to_body_bbox(det_box, W, H, cfg)

            to_draw = (bx, by, bw, bh)

            present_now = True

            last_trk_face = det_box

            last_present_idx = cur_idx

        elif bb_trk is not None:

            # Confirm tracker via nearest detection IoU within ~1s window

            nd = nearest_detection(cur_idx, max_window=max_nd_window)

            if nd is not None:

                _, bbN = nd

                ax, ay, aw, ah = bb_trk

                bx, by, bw, bh = bbN

                ax2, ay2 = ax + aw, ay + ah

                bx2, by2 = bx + bw, by + bh

                ix1, iy1 = max(ax, bx), max(ay, by)

                ix2, iy2 = min(ax2, bx2), min(ay2, by2)

                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)

                inter = iw * ih

                union = aw*ah + bw*bh - inter

                iou = (inter / max(1e-6, union)) if union > 0 else 0.0

                if iou >= iou_thr_nd:

                    bx2, by2, bw2, bh2 = expand_to_body_bbox(bb_trk, W, H, cfg)

                    to_draw = (bx2, by2, bw2, bh2)

                    present_now = True

                    last_trk_face = bb_trk

                    last_present_idx = cur_idx

        if to_draw is None and bb_trk is not None and last_trk_face is not None:
            ax, ay, aw, ah = bb_trk
            bx, by, bw, bh = last_trk_face
            ax2, ay2 = ax + aw, ay + ah
            bx2, by2 = bx + bw, by + bh
            ix1, iy1 = max(ax, bx), max(ay, by)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            union = aw*ah + bw*bh - inter
            iou_prev = (inter / max(1e-6, union)) if union > 0 else 0.0
            if iou_prev >= iou_thr_prev and (cur_idx - last_present_idx) <= max_track_without_det_frames:
                bx2, by2, bw2, bh2 = expand_to_body_bbox(bb_trk, W, H, cfg)
                to_draw = (bx2, by2, bw2, bh2)
                present_now = True
                last_trk_face = bb_trk
                last_present_idx = cur_idx

        if to_draw is None and last_bb_body is not None and linger_left > 0:

            # minimal smoothing; will not reset linger if not present_now

            bx, by, bw, bh = last_bb_body

            to_draw = (bx, by, bw, bh)

            linger_left = max(0, linger_left - 1)

        if to_draw is not None:

            cv2.rectangle(frame, (to_draw[0], to_draw[1]), (to_draw[0] + to_draw[2], to_draw[1] + to_draw[3]), color, thickness)

            last_bb_body = to_draw

            if present_now:

                linger_left = linger_frames

        writer.write(frame)

        cur_idx += 1


    writer.release()
    cap.release()
    try:
        os.replace(tmp_out, out_path)
    except Exception:
        try:
            shutil.move(tmp_out, out_path)
        except Exception:
            return False
    return True


def extract_faces_from_video(video_path: str, out_root: str, cfg: FaceClusterConfig, haar, mtcnn, embedder: FaceEmbedder, db: Optional[FaceCacheDB] = None, run_id: Optional[str] = None) -> List[FaceEntry]:
    entries: List[FaceEntry] = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return entries
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    progress_total_frames = frame_count
    if cfg.max_video_secs and cfg.max_video_secs > 0 and fps > 0:
        limit_frames = int(round(cfg.max_video_secs * fps))
        progress_total_frames = min(frame_count, limit_frames)
    step_base = 1
    if fps > 0 and cfg.sample_fps > 0:
        step_base = max(1, int(round(fps / cfg.sample_fps)))
    step_boost = max(1, int(round(fps / max(1e-6, cfg.adapt_boost_fps)))) if cfg.adaptive_sampling and fps > 0 else step_base

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.join(out_root, video_name)
    if cfg.clean_on_rerun and os.path.exists(video_dir):
        try:
            shutil.rmtree(video_dir)
        except Exception:
            pass
    faces_dir = os.path.join(out_root, video_name, "faces")
    os.makedirs(faces_dir, exist_ok=True)

    dprint(cfg, f"[init] {video_name} fps={fps:.2f} frames={frame_count} sample_fps={cfg.sample_fps} step_base={step_base} adaptive={cfg.adaptive_sampling} step_boost={step_boost} detect_every_k={cfg.detect_every_k} tracking={cfg.track_between_detects} tracker={cfg.tracker_type} batch={cfg.embed_batch_size} defer={cfg.defer_crop_save}")

    trackers = []
    detect_ctr = 0
    next_sample_idx = 0
    idx = -1
    adapt_left = 0
    sample_ctr = 0
    face_seen_ctr = 0
    kept_ctr = 0
    next_log_frame = 1000

    start_ts = time.time()
    while True:
        ok = cap.grab()
        if not ok:
            break
        idx += 1
        if idx < next_sample_idx:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        ts = (float(idx) / float(fps)) if fps > 0 else 0.0
        if cfg.max_video_secs and cfg.max_video_secs > 0 and ts >= cfg.max_video_secs:
            break
        current_step = step_boost if (cfg.adaptive_sampling and adapt_left > 0) else step_base
        next_sample_idx = idx + current_step
        sample_ctr += 1

        curr_faces: List[Tuple[int, int, int, int]] = []
        if (cfg.detect_every_k <= 1) or (detect_ctr % max(1, cfg.detect_every_k) == 0) or (not cfg.track_between_detects) or (len(trackers) == 0):
            curr_faces = detect_faces(frame, cfg, haar, mtcnn)
            if cfg.track_between_detects and len(curr_faces) > 0:
                trackers = []
                for (x, y, w, h) in curr_faces[: max(1, cfg.max_tracks)]:
                    tr = create_tracker(cfg.tracker_type)
                    if tr is not None:
                        try:
                            tr.init(frame, (int(x), int(y), int(w), int(h)))
                            trackers.append(tr)
                        except Exception:
                            pass
            dprint(cfg, f"[detect] idx={idx} t={ts:.2f}s faces_raw={len(curr_faces)} trackers={len(trackers)}")
        else:
            upd: List[Tuple[int, int, int, int]] = []
            new_trackers = []
            for tr in trackers:
                try:
                    ok2, bb = tr.update(frame)
                except Exception:
                    ok2, bb = False, None
                if ok2 and bb is not None:
                    x, y, w, h = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
                    upd.append((x, y, w, h))
                    new_trackers.append(tr)
            trackers = new_trackers
            curr_faces = upd
            dprint(cfg, f"[track] idx={idx} t={ts:.2f}s tracks={len(trackers)} boxes={len(curr_faces)}")

        H, W = frame.shape[:2]
        crops: List[np.ndarray] = []
        metas: List[Tuple[int, int, int, int]] = []
        for (x, y, w, h) in curr_faces:
            area_ratio = (float(w) * float(h)) / float(W * H + 1e-9)
            if area_ratio < cfg.face_min_area_ratio:
                continue
            crop = crop_with_margin(frame, x, y, w, h, margin=0.25)
            if cfg.face_min_sharpness and cfg.face_min_sharpness > 0:
                _g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                _lv = float(cv2.Laplacian(_g, cv2.CV_64F).var())
                if _lv < cfg.face_min_sharpness:
                    continue
            crops.append(crop)
            metas.append((x, y, w, h))
        face_seen_ctr += len(curr_faces)
        kept_ctr += len(metas)

        embs_all: np.ndarray = np.zeros((0, 1), dtype=np.float32)
        if len(crops) > 0:
            if cfg.embed_batch_size > 0 and len(crops) > 1:
                out_list = []
                for s in range(0, len(crops), cfg.embed_batch_size):
                    batch = crops[s: s + cfg.embed_batch_size]
                    out_list.append(embedder.embed_batch_bgr(batch))
                embs_all = np.vstack(out_list) if len(out_list) > 0 else np.zeros((0, 1), dtype=np.float32)
            else:
                embs_all = embedder.embed_batch_bgr(crops)

        if embs_all.shape[0] == len(metas):
            for j, (x, y, w, h) in enumerate(metas):
                v = embs_all[j]
                if v is None or not np.isfinite(v).all():
                    continue
                # DB-backed dedup
                is_dup = False
                face_hash = ""
                if cfg.db_enable and db is not None:
                    try:
                        face_hash = compute_phash_bgr(crops[j])
                        rows = db.recent_faces(video_name, float(ts), float(cfg.dedup_window_sec), run_id)
                        for rh, remb in rows:
                            if cfg.dedup_method in ("hash", "both") and rh is not None:
                                try:
                                    if (imagehash.hex_to_hash(rh) - imagehash.hex_to_hash(face_hash)) <= int(cfg.dedup_hash_thresh):
                                        is_dup = True
                                        break
                                except Exception:
                                    pass
                            if cfg.dedup_method in ("embedding", "both") and remb is not None:
                                try:
                                    e2 = np.frombuffer(remb, dtype=np.float32)
                                    if e2.size == v.size and float(np.dot(v, e2)) >= float(cfg.dedup_emb_thresh):
                                        is_dup = True
                                        break
                                except Exception:
                                    pass
                    except Exception:
                        pass
                if is_dup:
                    continue
                if cfg.defer_crop_save:
                    fp = ""
                else:
                    fn = f"{idx:09d}_t{ts:.2f}_x{x}_y{y}_w{w}_h{h}.jpg"
                    fp = os.path.join(faces_dir, fn)
                    try:
                        cv2.imwrite(fp, crops[j], [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    except Exception:
                        fp = ""
                entries.append(FaceEntry(video=video_path, frame_idx=idx, timestamp=float(ts), bbox=(x, y, w, h), face_path=fp, emb=v))
                if cfg.db_enable and db is not None:
                    try:
                        db.insert_face(video_name, idx, float(ts), x, y, w, h, face_hash if face_hash else compute_phash_bgr(crops[j]), v.tobytes(), fp, run_id)
                    except Exception:
                        pass

        if (idx + 1) >= next_log_frame:
            denom = float(max(1, progress_total_frames))
            pct = (float(idx + 1) / denom) * 100.0
            elapsed = max(0.0, time.time() - start_ts)
            prog = (float(idx + 1) / denom) if denom > 0 else 0.0
            eta = (elapsed / prog - elapsed) if prog > 0 else 0.0
            if cfg.progress_only:
                try:
                    print(f"\r[prog] {video_name} {idx+1}/{int(denom)} ({pct:.1f}%) elapsed={int(elapsed)}s eta={int(eta)}s", end="", flush=True)
                except Exception:
                    pass
            elif cfg.debug:
                dprint(cfg, f"[prog] {video_name} frames {idx+1}/{int(denom)} ({pct:.1f}%) elapsed={elapsed:.1f}s eta={eta:.1f}s samples={sample_ctr} total_faces={len(entries)}")
            next_log_frame += 1000

        if cfg.adaptive_sampling:
            if len(metas) > 0:
                adapt_left = max(adapt_left, int(round(cfg.adapt_boost_secs * fps)))
            else:
                adapt_left = max(0, adapt_left - current_step)

        detect_ctr += 1

    cap.release()
    if cfg.progress_only:
        try:
            print()
        except Exception:
            pass
    dprint(cfg, f"[video done] {video_name} entries={len(entries)} samples={sample_ctr} faces_seen={face_seen_ctr} kept={kept_ctr}")
    return entries


# -------------------------------
# Clustering (greedy centroid)
# -------------------------------

@dataclass
class Cluster:
    id: int
    centroid: np.ndarray
    count: int
    members: List[int]


def greedy_cluster(embs: np.ndarray, thr_sim: float) -> List[int]:
    N, D = embs.shape
    labels = [-1] * N
    clusters: List[Cluster] = []
    next_id = 0

    for i in range(N):
        v = embs[i]
        if len(clusters) == 0:
            clusters.append(Cluster(id=next_id, centroid=v.copy(), count=1, members=[i]))
            labels[i] = next_id
            next_id += 1
            continue
        sims = np.array([float(np.dot(v, c.centroid)) for c in clusters], dtype=np.float32)
        j = int(np.argmax(sims))
        if sims[j] >= thr_sim:
            c = clusters[j]
            new_c = (c.centroid * c.count + v)
            new_c = new_c / max(1e-9, np.linalg.norm(new_c))
            c.centroid = new_c.astype(np.float32)
            c.count += 1
            c.members.append(i)
            labels[i] = c.id
        else:
            clusters.append(Cluster(id=next_id, centroid=v.copy(), count=1, members=[i]))
            labels[i] = next_id
            next_id += 1
    return labels


# -------------------------------
# Clip helpers
# -------------------------------

def get_video_duration(path: str) -> float:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0.0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    dur = (frames / fps) if fps > 0 else 0.0
    cap.release()
    return float(dur)


def build_segments(times: List[float], pre: float, post: float, merge_gap: float, max_dur: float, min_dur: float) -> List[Tuple[float, float]]:
    if not times:
        return []
    ivals: List[Tuple[float, float]] = []
    for t in sorted(times):
        s = max(0.0, float(t) - float(pre))
        e = float(t) + float(post)
        if max_dur > 0:
            e = min(float(max_dur), e)
        # Enforce minimum duration around the hit time
        if (e - s) < float(min_dur):
            half = float(min_dur) / 2.0
            s = max(0.0, float(t) - half)
            e = s + float(min_dur)
            if max_dur > 0 and e > float(max_dur):
                e = float(max_dur)
                s = max(0.0, e - float(min_dur))
        ivals.append((s, e))
    ivals.sort()
    merged: List[Tuple[float, float]] = []
    for s, e in ivals:
        if not merged:
            merged.append((s, e))
        else:
            ls, le = merged[-1]
            if s <= le + float(merge_gap):
                if e > le:
                    merged[-1] = (ls, e)
            else:
                merged.append((s, e))
    return [(s, e) for (s, e) in merged if e > s + 0.05]


def save_segment_ffmpeg(src: str, start: float, end: float, dst: str, reencode: bool) -> bool:
    dur = max(0.0, float(end) - float(start))
    if dur <= 0.05:
        return False
    if reencode:
        cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-ss', f'{start:.3f}', '-t', f'{dur:.3f}', '-i', src,
               '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-c:a', 'aac', '-movflags', '+faststart', dst]
    else:
        cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-ss', f'{start:.3f}', '-t', f'{dur:.3f}', '-i', src,
               '-c', 'copy', dst]
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception:
        return False


def concat_segments_ffmpeg(files: List[str], dst: str, reencode: bool) -> bool:
    if not files:
        return False
    if len(files) == 1:
        try:
            shutil.move(files[0], dst)
            return True
        except Exception:
            try:
                shutil.copyfile(files[0], dst)
                return True
            except Exception:
                return False
    list_path = dst + '.txt'
    try:
        base_dir = os.path.dirname(os.path.abspath(dst))
        with open(list_path, 'w', encoding='utf-8') as f:
            for pth in files:
                rel = os.path.relpath(os.path.abspath(pth), base_dir)
                f.write(f"file '{rel}'\n")
        if reencode:
            cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', list_path,
                   '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-c:a', 'aac', '-movflags', '+faststart', dst]
        else:
            cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', list_path,
                   '-c', 'copy', dst]
        subprocess.run(cmd, check=True)
        return True
    except Exception:
        return False
    finally:
        try:
            os.remove(list_path)
        except Exception:
            pass


# -------------------------------
# Main
# -------------------------------

def main():
    p = argparse.ArgumentParser(description="Extract prominent faces from videos and cluster similar faces (v3 with annotated clips)")
    p.add_argument('--corpus', required=True, help='Video file or directory')
    p.add_argument('--output', required=True, help='Output directory')

    p.add_argument('--device', default=None)
    p.add_argument('--no-pretrained', action='store_true')

    p.add_argument('--detector', default='haar', choices=['haar', 'mtcnn'])
    p.add_argument('--min-face-rel-size', type=float, default=0.06)
    p.add_argument('--face-min-area-ratio', type=float, default=0.02)
    p.add_argument('--face-min-sharpness', type=float, default=60.0)

    p.add_argument('--sample-fps', type=float, default=2.0)
    p.add_argument('--max-video-secs', type=float, default=0.0)

    p.add_argument('--embedder', default='timm', choices=['timm', 'facenet'])
    p.add_argument('--backbone', default='resnet50')
    p.add_argument('--input-size', type=int, default=224)

    p.add_argument('--cluster-scope', default='per_video', choices=['global', 'per_video'])
    p.add_argument('--similarity-threshold', type=float, default=0.30)
    p.add_argument('--min-cluster-size', type=int, default=3)
    p.add_argument('--representative-only', action='store_true')
    p.add_argument('--no-clean', action='store_true')

    p.add_argument('--no-clips', action='store_true')
    p.add_argument('--clip-pre-sec', type=float, default=0.0)
    p.add_argument('--clip-post-sec', type=float, default=0.0)
    p.add_argument('--clip-min-dur', type=float, default=0.5)
    p.add_argument('--clip-merge-gap', type=float, default=0.5)
    p.add_argument('--no-ffmpeg-reencode', action='store_true')

    # v2 performance flags
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--embed-batch-size', type=int, default=16)
    p.add_argument('--defer-crop-save', action='store_true')
    p.add_argument('--detect-every-k', type=int, default=5)
    p.add_argument('--no-track-between-detects', action='store_true')
    p.add_argument('--tracker', default='KCF', choices=['KCF','CSRT','MOSSE'])
    p.add_argument('--max-tracks', type=int, default=8)
    p.add_argument('--adaptive-sampling', action='store_true')
    p.add_argument('--adapt-boost-fps', type=float, default=4.0)
    p.add_argument('--adapt-boost-secs', type=float, default=2.0)

    p.add_argument('--debug', action='store_true')
    p.add_argument('--progress-only', action='store_true')

    # DB & dedup
    p.add_argument('--db-enable', action='store_true')
    p.add_argument('--db-path', default='poc_video/face_cache.duckdb')
    p.add_argument('--dedup-method', default='both', choices=['hash','embedding','both'])
    p.add_argument('--dedup-hash-thresh', type=int, default=6)
    p.add_argument('--dedup-emb-thresh', type=float, default=0.995)
    p.add_argument('--dedup-window-sec', type=float, default=5.0)

    # V3 annotate flags
    p.add_argument('--annotate-clips', action='store_true', help='Draw box around person (face+body) in output clips')
    p.add_argument('--annotate-thickness', type=int, default=3)
    p.add_argument('--annotate-color', default='0,255,0', help='BGR color as "B,G,R"')
    p.add_argument('--body-expand-w', type=float, default=1.8)
    p.add_argument('--body-expand-h', type=float, default=3.0)

    args = p.parse_args()

    run_id = str(uuid.uuid4())

    cfg = FaceClusterConfig(
        corpus=args.corpus,
        output=args.output,
        device=args.device,
        pretrained=not args.no_pretrained,
        detector=args.detector,
        min_face_rel_size=args.min_face_rel_size,
        face_min_area_ratio=args.face_min_area_ratio,
        face_min_sharpness=args.face_min_sharpness,
        sample_fps=args.sample_fps,
        max_video_secs=args.max_video_secs,
        embedder=args.embedder,
        backbone=args.backbone,
        input_size=args.input_size,
        cluster_scope=args.cluster_scope,
        similarity_threshold=args.similarity_threshold,
        min_cluster_size=args.min_cluster_size,
        save_representative_only=args.representative_only,
        clean_on_rerun=not args.no_clean,
        make_clips=not args.no_clips,
        clip_pre_sec=args.clip_pre_sec,
        clip_post_sec=args.clip_post_sec,
        clip_min_dur_sec=args.clip_min_dur,
        clip_merge_gap_sec=args.clip_merge_gap,
        ffmpeg_reencode=not args.no_ffmpeg_reencode,
        fp16=args.fp16,
        embed_batch_size=args.embed_batch_size,
        defer_crop_save=args.defer_crop_save,
        detect_every_k=args.detect_every_k,
        track_between_detects=(not args.no_track_between_detects),
        tracker_type=args.tracker,
        max_tracks=args.max_tracks,
        adaptive_sampling=args.adaptive_sampling,
        adapt_boost_fps=args.adapt_boost_fps,
        adapt_boost_secs=args.adapt_boost_secs,
        debug=args.debug,
        progress_only=args.progress_only,
        db_enable=args.db_enable,
        db_path=args.db_path,
        dedup_method=args.dedup_method,
        dedup_hash_thresh=args.dedup_hash_thresh,
        dedup_emb_thresh=args.dedup_emb_thresh,
        dedup_window_sec=args.dedup_window_sec,
        annotate_clips=args.annotate_clips,
        annotate_thickness=args.annotate_thickness,
        annotate_color=args.annotate_color,
        body_expand_w=args.body_expand_w,
        body_expand_h=args.body_expand_h,
    )

    os.makedirs(cfg.output, exist_ok=True)

    # Load detectors/embedders
    haar = load_haar_detector()
    mtcnn = load_mtcnn(cfg.device) if cfg.detector == 'mtcnn' else None
    embedder = FaceEmbedder(device=cfg.device, embedder=cfg.embedder, backbone=cfg.backbone, input_size=cfg.input_size, pretrained=cfg.pretrained, use_fp16=cfg.fp16)

    videos = list_videos(cfg.corpus)
    if cfg.debug:
        dprint(cfg, f"Processing {len(videos)} video(s)")

    db = FaceCacheDB(cfg.db_path) if cfg.db_enable and cfg.db_path else None

    def cluster_and_save(entries: List[FaceEntry], out_subdir: str):
        if len(entries) == 0:
            return
        embs = np.vstack([e.emb for e in entries]).astype(np.float32)
        labels = greedy_cluster(embs, cfg.similarity_threshold)
        clusters: Dict[int, List[int]] = {}
        for idx, lb in enumerate(labels):
            clusters.setdefault(lb, []).append(idx)
        items = sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True)
        dprint(cfg, f"[cluster] {out_subdir} faces={len(entries)} candidate_clusters={len(items)} min_size={cfg.min_cluster_size}")
        base = os.path.join(cfg.output, out_subdir)
        if cfg.clean_on_rerun and os.path.exists(base):
            try:
                shutil.rmtree(base)
            except Exception:
                pass
        os.makedirs(base, exist_ok=True)
        csv_path = os.path.join(base, 'clusters.csv')
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('cluster_id,size,video,frame_idx,timestamp,face_path\n')
            cid = 0
            kept_clusters = 0
            for orig_id, idxs in items:
                if len(idxs) < cfg.min_cluster_size:
                    continue
                kept_clusters += 1
                folder_name = f'face{cid+1}'
                cdir = os.path.join(base, folder_name)
                os.makedirs(cdir, exist_ok=True)
                centroid = np.mean(embs[idxs, :], axis=0)
                centroid = centroid / max(1e-9, np.linalg.norm(centroid))
                sims = (embs[idxs, :] @ centroid)
                j_local = int(np.argmax(sims))
                rep_idx = idxs[j_local]
                rep_entry = entries[rep_idx]
                rep_dst = os.path.join(cdir, 'representative.jpg')
                copied = False
                if rep_entry.face_path and os.path.exists(rep_entry.face_path):
                    try:
                        shutil.copyfile(rep_entry.face_path, rep_dst)
                        copied = True
                    except Exception:
                        copied = False
                if not copied:
                    save_crop_from_video(rep_entry.video, rep_entry.frame_idx, rep_entry.bbox, rep_dst)
                dprint(cfg, f"[cluster face{cid+1}] size={len(idxs)} rep_t={rep_entry.timestamp:.2f}s rep_frame={rep_entry.frame_idx}")
                for k in idxs:
                    e = entries[k]
                    f.write(f"face{cid+1},{len(idxs)},{os.path.basename(e.video)},{e.frame_idx},{e.timestamp:.3f},{e.face_path}\n")
                    if not cfg.save_representative_only:
                        if e.face_path and os.path.exists(e.face_path):
                            dst = os.path.join(cdir, os.path.basename(e.face_path))
                            try:
                                shutil.copyfile(e.face_path, dst)
                            except Exception:
                                pass
                        else:
                            dst = os.path.join(cdir, f"{e.frame_idx:09d}_t{e.timestamp:.2f}.jpg")
                            save_crop_from_video(e.video, e.frame_idx, e.bbox, dst)
                if cfg.make_clips:
                    video_src = entries[idxs[0]].video
                    times = sorted([entries[k].timestamp for k in idxs])
                    duration = get_video_duration(video_src)
                    segs = build_segments(times, cfg.clip_pre_sec, cfg.clip_post_sec, cfg.clip_merge_gap_sec, duration, cfg.clip_min_dur_sec)
                    dprint(cfg, f"[segments face{cid+1}] hits={len(times)} segments={len(segs)} pre={cfg.clip_pre_sec} post={cfg.clip_post_sec} gap={cfg.clip_merge_gap_sec}")
                    seg_paths = []
                    # Build mapping from frame_idx to bbox for this cluster
                    cluster_map: Dict[int, Tuple[int, int, int, int]] = {}
                    for k in idxs:
                        ce = entries[k]
                        cluster_map[int(ce.frame_idx)] = ce.bbox
                    for si, (ss, ee) in enumerate(segs):
                        seg_out = os.path.join(cdir, f'seg_{si:03d}.mp4')
                        ok = False
                        if cfg.annotate_clips:
                            ok = annotate_segment_opencv(video_src, ss, ee, cluster_map, cfg, seg_out)
                        else:
                            ok = save_segment_ffmpeg(video_src, ss, ee, seg_out, reencode=cfg.ffmpeg_reencode)
                        dprint(cfg, f"[cut face{cid+1}] seg={si} {ss:.2f}-{ee:.2f}s -> {os.path.basename(seg_out)} ok={ok}")
                        if ok:
                            seg_paths.append(seg_out)
                    combined_path = os.path.join(cdir, 'combined.mp4')
                    if seg_paths:
                        ok2 = concat_segments_ffmpeg(seg_paths, combined_path, reencode=cfg.ffmpeg_reencode)
                        dprint(cfg, f"[concat face{cid+1}] files={len(seg_paths)} -> combined.mp4 ok={ok2}")
                        if ok2 and not cfg.debug:
                            for sp in seg_paths:
                                try:
                                    os.remove(sp)
                                except Exception:
                                    pass
                cid += 1
            dprint(cfg, f"[cluster] kept_clusters={kept_clusters} written_to={base}")

    all_entries: List[FaceEntry] = []
    if cfg.cluster_scope == 'per_video':
        for i, vp in enumerate(videos, start=1):
            if cfg.debug:
                dprint(cfg, f"[{i}/{len(videos)}] {vp}")
            t0 = time.time()
            ents = extract_faces_from_video(vp, cfg.output, cfg, haar, mtcnn, embedder, db, run_id)
            if cfg.debug:
                dprint(cfg, f"Collected {len(ents)} faces for {os.path.basename(vp)}")
            video_name = os.path.splitext(os.path.basename(vp))[0]
            cluster_and_save(ents, f'clusters_{video_name}')
            elapsed = time.time() - t0
            try:
                print(f"[time] {os.path.basename(vp)} processed in {elapsed:.2f}s")
            except Exception:
                pass
    else:
        for i, vp in enumerate(videos, start=1):
            if cfg.debug:
                dprint(cfg, f"[{i}/{len(videos)}] {vp}")
            t0 = time.time()
            ents = extract_faces_from_video(vp, cfg.output, cfg, haar, mtcnn, embedder, db, run_id)
            elapsed = time.time() - t0
            try:
                print(f"[time] {os.path.basename(vp)} extracted in {elapsed:.2f}s")
            except Exception:
                pass
            all_entries.extend(ents)
        if cfg.debug:
            dprint(cfg, f"Collected {len(all_entries)} faces")
        cluster_and_save(all_entries, 'clusters_global')

    print("Done. Output:", cfg.output)


if __name__ == '__main__':
    main()

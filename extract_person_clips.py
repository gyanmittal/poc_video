from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import timm

try:
    import duckdb  # type: ignore
except Exception:
    duckdb = None  # optional

# -------------------------------
# Config
# -------------------------------

@dataclass
class PersonClipsConfig:
    photos: List[str]
    corpus: str
    output: str
    db_path: Optional[str] = None

    device: Optional[str] = None
    pretrained: bool = True

    # Detection
    detector: str = "haar"  # 'haar' or 'mtcnn' (if installed)
    min_face_rel_size: float = 0.06  # relative to min(H, W)

    # Embedding
    embedder: str = "timm"  # 'timm' or 'facenet' (if installed)
    backbone: str = "resnet50"
    input_size: int = 224

    # Matching
    sample_fps: float = 2.0
    match_threshold: float = 0.55  # cosine similarity threshold [0..1]
    min_consecutive_hits: int = 2  # require N consecutive hits before accepting a timestamp
    center_margin: float = 0.05  # allow centroid similarity a bit lower than match_threshold
    use_query_centroid: bool = True
    require_all_queries: bool = True  # if multiple photos, require match to all
    # Face quality gates
    face_min_area_ratio: float = 0.02  # relative bbox area (w*h)/(W*H)
    face_min_sharpness: float = 60.0   # Laplacian variance threshold on face crop
    # Segment validation
    segment_validate: bool = True
    segment_min_accept_ratio: float = 0.6
    segment_validate_fps: float = 4.0

    # Debug dumping
    dump_matches: bool = False
    dump_match_frames: bool = False
    dump_warmup: bool = True
    dump_limit_per_video: int = 500
    # Super-lenient mode to diagnose pipeline (disables/relaxes gates)
    super_lenient: bool = False

    # Clip assembly
    pre_roll: float = 1.0
    post_roll: float = 1.0
    join_gap: float = 2.0
    max_clips_per_video: Optional[int] = None

    # Output clips
    save_clips: bool = True
    ffmpeg_reencode: bool = False  # if False uses -c copy

    debug: bool = False


# -------------------------------
# Utilities
# -------------------------------

def list_images(path_or_dir: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    if os.path.isdir(path_or_dir):
        out: List[str] = []
        for r, _, files in os.walk(path_or_dir):
            for f in files:
                if f.lower().endswith(exts):
                    out.append(os.path.join(r, f))
        out.sort()
        return out
    return [p for p in path_or_dir.split(",") if p.strip()]


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
# Face detection
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


# Optional MTCNN detector (facenet_pytorch). Loaded lazily if requested.
def load_mtcnn(device: Optional[str] = None):
    try:
        from facenet_pytorch import MTCNN  # type: ignore
        # Workaround: facenet_pytorch MTCNN can crash on MPS with 'Adaptive pool MPS' error.
        # Prefer CUDA if present; otherwise use CPU (not MPS) for MTCNN.
        if device == "mps":
            dev = "cpu"
        elif device is not None:
            dev = device
        else:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        mtcnn = MTCNN(keep_all=True, device=dev)
        return mtcnn
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


def detect_faces(img_bgr: np.ndarray, cfg: PersonClipsConfig, haar: Optional[cv2.CascadeClassifier], mtcnn) -> List[Tuple[int, int, int, int]]:
    if cfg.detector == 'mtcnn' and mtcnn is not None:
        return detect_faces_mtcnn(img_bgr, mtcnn, min_rel=cfg.min_face_rel_size)
    return detect_faces_haar(img_bgr, haar, min_rel=cfg.min_face_rel_size)


# -------------------------------
# Embedding
# -------------------------------

class FaceEmbedder:
    def __init__(self, device: Optional[str] = None, embedder: str = "timm", backbone: str = "resnet50", input_size: int = 224, pretrained: bool = True):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.type = None
        self.model = None
        self.size = input_size
        self.mean = None
        self.std = None
        self._facenet_norm = False

        if embedder == "facenet":
            try:
                from facenet_pytorch import InceptionResnetV1  # type: ignore
                self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
                self.type = "facenet"
                self.size = 160
                self._facenet_norm = True
            except Exception:
                # fallback to timm
                pass
        if self.model is None:
            # timm fallback
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
            t = (t - 0.5) / 0.5  # normalize to [-1,1]
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


# -------------------------------
# Matching and clip assembly
# -------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def gather_query_embeddings(cfg: PersonClipsConfig, embedder: FaceEmbedder, haar: Optional[cv2.CascadeClassifier]) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for p in cfg.photos:
        try:
            img = cv2.imread(p)
            if img is None:
                continue
            faces = detect_faces_haar(img, haar, min_rel=cfg.min_face_rel_size)
            crop = None
            if faces:
                # choose largest
                faces = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)
                x, y, w, h = faces[0]
                crop = crop_with_margin(img, x, y, w, h, margin=0.25)
            else:
                crop = img
            v = embedder.embed_bgr(crop)
            if v is not None:
                out.append(v)
        except Exception:
            continue
    return out


def hits_to_segments(hits: List[Tuple[float, float]], join_gap: float, pre: float, post: float, duration: Optional[float]) -> List[Tuple[float, float, int, float]]:
    if not hits:
        return []
    hits = sorted(hits, key=lambda x: x[0])
    segs: List[Tuple[float, float, int, float]] = []
    s = hits[0][0]
    e = hits[0][0]
    n = 1
    best = hits[0][1]
    for t, sim in hits[1:]:
        if t - e <= join_gap:
            e = t
            n += 1
            best = max(best, sim)
        else:
            segs.append((s - pre, e + post, n, best))
            s, e, n, best = t, t, 1, sim
    segs.append((s - pre, e + post, n, best))
    # clamp and sort
    out: List[Tuple[float, float, int, float]] = []
    for (a, b, c, d) in segs:
        if duration is not None:
            a = max(0.0, a)
            b = min(duration, b)
        if b > a:
            out.append((a, b, c, d))
    return out


def find_ffmpeg() -> Optional[str]:
    from shutil import which
    return which('ffmpeg')


def save_clip_ffmpeg(input_path: str, start: float, end: float, dst_path: str, reencode: bool = False) -> bool:
    os.makedirs(os.path.dirname(dst_path) or '.', exist_ok=True)
    dur = max(0.01, end - start)
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        return False
    if reencode:
        cmd = [ffmpeg, '-y', '-ss', f'{start:.3f}', '-i', input_path, '-t', f'{dur:.3f}', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-c:a', 'aac', '-b:a', '128k', dst_path]
    else:
        cmd = [ffmpeg, '-y', '-ss', f'{start:.3f}', '-i', input_path, '-t', f'{dur:.3f}', '-c', 'copy', dst_path]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


def save_clip_opencv(input_path: str, start: float, end: float, dst_path: str) -> bool:
    os.makedirs(os.path.dirname(dst_path) or '.', exist_ok=True)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(dst_path, fourcc, fps, (width, height))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start) * 1000.0)
    ok = True
    try:
        while True:
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_ms > end * 1000.0:
                break
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
    except Exception:
        ok = False
    finally:
        out.release()
        cap.release()
    return ok


# -------------------------------
# Main processing
# -------------------------------

def process_video(video_path: str, cfg: PersonClipsConfig, query_vecs: List[np.ndarray], q_center: Optional[np.ndarray], haar: Optional[cv2.CascadeClassifier], mtcnn, embedder: FaceEmbedder, out_dir: str) -> List[dict]:
    entries: List[dict] = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return entries
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (frame_count / fps) if (frame_count > 0 and fps > 0) else None

    step = 1
    if fps > 0 and cfg.sample_fps > 0:
        step = max(1, int(round(fps / cfg.sample_fps)))

    hits: List[Tuple[float, float]] = []
    hit_streak: int = 0

    def accept_frame(frame_bgr: np.ndarray) -> Tuple[bool, float, float, float, Optional[Tuple[int, int, int, int]]]:
        faces = detect_faces(frame_bgr, cfg, haar, mtcnn)
        Hf, Wf = frame_bgr.shape[:2]
        best = (False, 1.0, 0.0, 0.0)  # matched, smin, smax, scen
        best_bbox: Optional[Tuple[int, int, int, int]] = None
        for (x, y, w, h) in faces:
            area_ratio = (float(w) * float(h)) / float(Wf * Hf + 1e-9)
            if area_ratio < cfg.face_min_area_ratio:
                continue
            crop = crop_with_margin(frame_bgr, x, y, w, h, margin=0.25)
            if cfg.face_min_sharpness and cfg.face_min_sharpness > 0:
                _g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                _lv = float(cv2.Laplacian(_g, cv2.CV_64F).var())
                if _lv < cfg.face_min_sharpness:
                    continue
            v = embedder.embed_bgr(crop)
            if v is None:
                continue
            sims = [cosine_sim(v, q) for q in query_vecs]
            if len(sims) == 0:
                continue
            smax = float(np.max(sims))
            smin = float(np.min(sims))
            scen = float(np.dot(v, q_center)) if (q_center is not None) else smax
            cond_centroid = (not cfg.use_query_centroid) or (scen >= (cfg.match_threshold - cfg.center_margin))
            cond_perphoto = (smin >= cfg.match_threshold) if cfg.require_all_queries else (smax >= cfg.match_threshold)
            is_hit = cond_perphoto and cond_centroid
            # choose best by smin if requiring-all, else by smax
            key = smin if cfg.require_all_queries else smax
            best_key = best[1] if cfg.require_all_queries else best[2]
            if (is_hit and key > best_key) or (not best[0] and key > best_key):
                best = (is_hit, smin, smax, scen)
                best_bbox = (x, y, w, h)
        return best + (best_bbox,)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if fps > 0 else (frame_idx / fps if fps > 0 else 0.0)

        matched, smin, smax, scen, bbox = accept_frame(frame)
        if matched:
            hit_streak += 1
            if hit_streak >= cfg.min_consecutive_hits:
                hits.append((ts, smax))
                if cfg.debug:
                    print(f"[hit] {os.path.basename(video_path)} t={ts:.2f}s sim_min={smin:.3f} sim_max={smax:.3f} sim_centroid={scen:.3f} streak={hit_streak}")
                # dump accepted match
                if cfg.dump_matches:
                    try:
                        if dump_count < cfg.dump_limit_per_video:
                            if bbox is not None:
                                x, y, w, h = bbox
                                crop = crop_with_margin(frame, x, y, w, h, margin=0.25)
                                fp = os.path.join(debug_dir, f"{frame_idx:09d}_t{ts:.2f}_hit_face_smin{smin:.3f}_smax{smax:.3f}.jpg")
                                cv2.imwrite(fp, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                            if cfg.dump_match_frames:
                                fp2 = os.path.join(debug_dir, f"{frame_idx:09d}_t{ts:.2f}_hit_frame_smin{smin:.3f}_smax{smax:.3f}.jpg")
                                cv2.imwrite(fp2, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                            dump_count += 1
                    except Exception:
                        pass
            else:
                if cfg.debug:
                    print(f"[warmup] {os.path.basename(video_path)} t={ts:.2f}s sim_min={smin:.3f} sim_max={smax:.3f} sim_centroid={scen:.3f} streak={hit_streak}")
                # dump warmup match if enabled
                if cfg.dump_matches and cfg.dump_warmup:
                    try:
                        if dump_count < cfg.dump_limit_per_video and bbox is not None:
                            x, y, w, h = bbox
                            crop = crop_with_margin(frame, x, y, w, h, margin=0.25)
                            fp = os.path.join(debug_dir, f"{frame_idx:09d}_t{ts:.2f}_warm_face_smin{smin:.3f}_smax{smax:.3f}.jpg")
                            cv2.imwrite(fp, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                            if cfg.dump_match_frames:
                                fp2 = os.path.join(debug_dir, f"{frame_idx:09d}_t{ts:.2f}_warm_frame_smin{smin:.3f}_smax{smax:.3f}.jpg")
                                cv2.imwrite(fp2, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                            dump_count += 1
                    except Exception:
                        pass
        else:
            hit_streak = 0
        frame_idx += 1

    cap.release()

    segs = hits_to_segments(hits, cfg.join_gap, cfg.pre_roll, cfg.post_roll, duration)
    if cfg.max_clips_per_video is not None and len(segs) > cfg.max_clips_per_video:
        segs = segs[: cfg.max_clips_per_video]

    clips_dir = os.path.join(out_dir, os.path.splitext(os.path.basename(video_path))[0])
    # Clean previous outputs for this video to avoid stale clips on reruns
    if os.path.exists(clips_dir):
        try:
            shutil.rmtree(clips_dir)
        except Exception:
            pass
    os.makedirs(clips_dir, exist_ok=True)
    debug_dir = os.path.join(clips_dir, "debug_matches")
    if cfg.dump_matches:
        os.makedirs(debug_dir, exist_ok=True)
    dump_count = 0

    # Optional segment validation to prune false positives
    if cfg.segment_validate and len(segs) > 0:
        cap2 = cv2.VideoCapture(video_path)
        if cap2.isOpened() and fps > 0 and cfg.segment_validate_fps > 0:
            step2 = max(1, int(round(fps / cfg.segment_validate_fps)))
            validated = []
            for (a, b, n_hits, best_sim) in segs:
                # seek to start
                try:
                    cap2.set(cv2.CAP_PROP_POS_MSEC, max(0.0, a) * 1000.0)
                except Exception:
                    pass
                pass_cnt = 0
                total_cnt = 0
                while True:
                    pos_ms = cap2.get(cv2.CAP_PROP_POS_MSEC)
                    if pos_ms > b * 1000.0:
                        break
                    ret2, frm2 = cap2.read()
                    if not ret2:
                        break
                    if total_cnt % step2 == 0:
                        m2, _, _, _ = accept_frame(frm2)
                        pass_cnt += 1 if m2 else 0
                    total_cnt += 1
                ratio = (float(pass_cnt) / max(1, float(total_cnt // step2))) if total_cnt > 0 else 0.0
                if ratio >= cfg.segment_min_accept_ratio:
                    validated.append((a, b, n_hits, best_sim))
                elif cfg.debug:
                    print(f"[prune] segment {a:.2f}-{b:.2f}s rejected: accept_ratio={ratio:.2f} < {cfg.segment_min_accept_ratio:.2f}")
            segs = validated
        cap2.release()

    for i, (a, b, n_hits, best_sim) in enumerate(segs, start=1):
        out_path = os.path.join(clips_dir, f"clip_{i:03d}_{a:.2f}-{b:.2f}.mp4")
        saved = False
        if cfg.save_clips:
            saved = save_clip_ffmpeg(video_path, a, b, out_path, reencode=cfg.ffmpeg_reencode)
            if not saved:
                saved = save_clip_opencv(video_path, a, b, out_path)
        entries.append({
            'video': video_path,
            'start': float(a),
            'end': float(b),
            'duration': float(max(0.0, b - a)),
            'hits': int(n_hits),
            'best_sim': float(best_sim),
            'saved_path': out_path if (cfg.save_clips and saved) else None,
        })
    return entries


def maybe_init_db(db_path: Optional[str]):
    if db_path is None or duckdb is None:
        return None
    os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
    con = duckdb.connect(db_path)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS person_clips (
            run_id TEXT,
            query_id TEXT,
            video TEXT,
            start DOUBLE,
            end DOUBLE,
            duration DOUBLE,
            hits INT,
            best_sim DOUBLE,
            saved_path TEXT
        )
        """
    )
    return con


def main():
    p = argparse.ArgumentParser(description="Extract clips of a target person from a corpus using one or more photos")
    p.add_argument('--photos', required=True, help='Path to a photo file or a directory of photos (comma-separated files allowed)')
    p.add_argument('--corpus', required=True, help='Path to a video file or a directory of videos (recursive)')
    p.add_argument('--output', required=True, help='Output directory for clips and CSV summaries')
    p.add_argument('--db', default=None, help='Optional DuckDB path to log results')

    p.add_argument('--device', default=None)
    p.add_argument('--no-pretrained', action='store_true')
    p.add_argument('--detector', default='haar', choices=['haar', 'mtcnn'])
    p.add_argument('--embedder', default='timm', choices=['timm', 'facenet'])
    p.add_argument('--backbone', default='resnet50')
    p.add_argument('--input-size', type=int, default=224)

    p.add_argument('--sample-fps', type=float, default=2.0)
    p.add_argument('--match-threshold', type=float, default=0.55)
    p.add_argument('--min-consecutive-hits', type=int, default=2, help='Require N consecutive matching frames to accept a hit (default 2)')
    p.add_argument('--center-margin', type=float, default=0.05, help='Centroid similarity can be this much below match-threshold (default 0.05)')
    p.add_argument('--face-min-area-ratio', type=float, default=0.02, help='Minimum face bbox area ratio to consider (default 0.02)')
    p.add_argument('--face-min-sharpness', type=float, default=60.0, help='Minimum Laplacian variance on face crop (default 60.0)')
    p.add_argument('--require-all-queries', dest='require_all_queries', action='store_true', help='Require match with all query photos (default: enabled)')
    p.add_argument('--no-require-all-queries', dest='require_all_queries', action='store_false', help='Allow match with any one query photo')
    p.set_defaults(require_all_queries=True)
    p.add_argument('--no-segment-validate', dest='segment_validate', action='store_false', help='Disable segment-level validation (default: enabled)')
    p.add_argument('--segment-min-accept-ratio', type=float, default=0.6, help='Min fraction of validated frames inside a segment (default 0.6)')
    p.add_argument('--segment-validate-fps', type=float, default=4.0, help='Validation sampling FPS within segments (default 4.0)')

    p.add_argument('--pre-roll', type=float, default=1.0)
    p.add_argument('--post-roll', type=float, default=1.0)
    p.add_argument('--join-gap', type=float, default=2.0)
    p.add_argument('--max-clips-per-video', type=int, default=None)

    p.add_argument('--no-save-clips', action='store_true')
    p.add_argument('--ffmpeg-reencode', action='store_true')
    # Debug dumping and super-lenient mode
    p.add_argument('--dump-matches', action='store_true', help='Save matched face crops (and frames if enabled) for debugging')
    p.add_argument('--dump-match-frames', action='store_true', help='Also save full frames, not only face crops')
    p.add_argument('--no-dump-warmup', action='store_true', help='Do not dump warmup (pre-accept) matches')
    p.add_argument('--dump-limit', type=int, default=500, help='Max number of dumped items per video (default 500)')
    p.add_argument('--super-lenient', action='store_true', help='Relax criteria heavily to debug matching pipeline')

    p.add_argument('--debug', action='store_true')

    args = p.parse_args()

    cfg = PersonClipsConfig(
        photos=list_images(args.photos),
        corpus=args.corpus,
        output=args.output,
        db_path=args.db,
        device=args.device,
        pretrained=not args.no_pretrained,
        detector=args.detector,
        embedder=args.embedder,
        backbone=args.backbone,
        input_size=args.input_size,
        sample_fps=args.sample_fps,
        match_threshold=args.match_threshold,
        min_consecutive_hits=args.min_consecutive_hits,
        center_margin=args.center_margin,
        face_min_area_ratio=args.face_min_area_ratio,
        face_min_sharpness=args.face_min_sharpness,
        require_all_queries=args.require_all_queries,
        segment_validate=getattr(args, 'segment_validate', True),
        segment_min_accept_ratio=args.segment_min_accept_ratio,
        segment_validate_fps=args.segment_validate_fps,
        pre_roll=args.pre_roll,
        post_roll=args.post_roll,
        join_gap=args.join_gap,
        max_clips_per_video=args.max_clips_per_video,
        save_clips=(not args.no_save_clips),
        ffmpeg_reencode=args.ffmpeg_reencode,
        debug=args.debug,
        dump_matches=args.dump_matches,
        dump_match_frames=args.dump_match_frames,
        dump_warmup=not args.no_dump_warmup,
        dump_limit_per_video=args.dump_limit,
        super_lenient=args.super_lenient,
    )

    if len(cfg.photos) == 0:
        raise RuntimeError('No photos found')

    os.makedirs(cfg.output, exist_ok=True)
    run_id = uuid.uuid4().hex

    # Apply super-lenient overrides if requested
    if cfg.super_lenient:
        cfg.require_all_queries = False
        cfg.use_query_centroid = False
        cfg.match_threshold = min(cfg.match_threshold, 0.50)
        cfg.min_consecutive_hits = 1
        cfg.center_margin = max(cfg.center_margin, 0.10)
        cfg.face_min_area_ratio = min(cfg.face_min_area_ratio, 0.02)
        cfg.face_min_sharpness = 0.0
        cfg.segment_validate = False

    # Load detectors/embedders
    haar = load_haar_detector() if cfg.detector == 'haar' else load_haar_detector()
    mtcnn = load_mtcnn(cfg.device) if cfg.detector == 'mtcnn' else None
    embedder = FaceEmbedder(device=cfg.device, embedder=cfg.embedder, backbone=cfg.backbone, input_size=cfg.input_size, pretrained=cfg.pretrained)

    # Query embeddings
    query_vecs = gather_query_embeddings(cfg, embedder, haar)
    if len(query_vecs) == 0:
        raise RuntimeError('Failed to compute embeddings from the provided photos')
    q_center = None
    if cfg.use_query_centroid and len(query_vecs) > 0:
        Q = np.vstack(query_vecs).astype(np.float32)
        c = Q.mean(axis=0)
        n = float(np.linalg.norm(c) + 1e-9)
        q_center = (c / n).astype(np.float32)

    # Process videos
    videos = list_videos(cfg.corpus)
    all_entries: List[dict] = []

    # Optional DB
    con = maybe_init_db(cfg.db_path)

    t0 = time.time()
    for i, vp in enumerate(videos, start=1):
        if cfg.debug:
            print(f"[{i}/{len(videos)}] {vp}")
        entries = process_video(vp, cfg, query_vecs, q_center, haar, mtcnn, embedder, cfg.output)
        all_entries.extend(entries)
        # DB logging
        if con is not None:
            for e in entries:
                con.execute(
                    "INSERT INTO person_clips VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [run_id, os.path.basename(cfg.photos[0]), e['video'], e['start'], e['end'], e['duration'], e['hits'], e['best_sim'], e['saved_path']],
                )
    if con is not None:
        try:
            con.close()
        except Exception:
            pass

    # CSV summary
    csv_path = os.path.join(cfg.output, f"clips_{run_id}.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('video,start,end,duration,hits,best_sim,saved_path\n')
        for e in all_entries:
            f.write(f"{e['video']},{e['start']:.3f},{e['end']:.3f},{e['duration']:.3f},{e['hits']},{e['best_sim']:.4f},{e['saved_path'] if e['saved_path'] else ''}\n")
    dt = time.time() - t0
    print(f"Processed {len(videos)} video(s) in {dt:.2f}s; clips: {len(all_entries)}; CSV: {csv_path}")


if __name__ == '__main__':
    main()

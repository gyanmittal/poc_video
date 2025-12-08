from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import torch
import timm

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
    cluster_scope: str = "per_video"  # 'global' or 'per_video' (default per_video)
    similarity_threshold: float = 0.65  # cosine similarity >= thr -> same cluster
    min_cluster_size: int = 3

    # Output control
    save_representative_only: bool = False
    clean_on_rerun: bool = True

    # Clips
    make_clips: bool = True
    clip_pre_sec: float = 0.75
    clip_post_sec: float = 0.75
    clip_merge_gap_sec: float = 0.5
    ffmpeg_reencode: bool = True

    debug: bool = False


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
        # Avoid MPS crash, prefer CUDA then CPU for MTCNN
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


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


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


def extract_faces_from_video(video_path: str, out_root: str, cfg: FaceClusterConfig, haar, mtcnn, embedder: FaceEmbedder) -> List[FaceEntry]:
    entries: List[FaceEntry] = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return entries
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = 1
    if fps > 0 and cfg.sample_fps > 0:
        step = max(1, int(round(fps / cfg.sample_fps)))

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.join(out_root, video_name)
    if cfg.clean_on_rerun and os.path.exists(video_dir):
        try:
            shutil.rmtree(video_dir)
        except Exception:
            pass
    faces_dir = os.path.join(out_root, video_name, "faces")
    os.makedirs(faces_dir, exist_ok=True)

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step != 0:
            idx += 1
            continue
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if fps > 0 else (idx / fps if fps > 0 else 0.0)

        faces = detect_faces(frame, cfg, haar, mtcnn)
        H, W = frame.shape[:2]
        for (x, y, w, h) in faces:
            # prominence gates
            area_ratio = (float(w) * float(h)) / float(W * H + 1e-9)
            if area_ratio < cfg.face_min_area_ratio:
                continue
            crop = crop_with_margin(frame, x, y, w, h, margin=0.25)
            if cfg.face_min_sharpness and cfg.face_min_sharpness > 0:
                _g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                _lv = float(cv2.Laplacian(_g, cv2.CV_64F).var())
                if _lv < cfg.face_min_sharpness:
                    continue
            v = embedder.embed_bgr(crop)
            if v is None:
                continue
            fn = f"{idx:09d}_t{ts:.2f}_x{x}_y{y}_w{w}_h{h}.jpg"
            fp = os.path.join(faces_dir, fn)
            cv2.imwrite(fp, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            entries.append(FaceEntry(video=video_path, frame_idx=idx, timestamp=float(ts), bbox=(x, y, w, h), face_path=fp, emb=v))
        idx += 1

    cap.release()
    return entries


# -------------------------------
# Clustering (greedy centroid)
# -------------------------------

@dataclass
class Cluster:
    id: int
    centroid: np.ndarray
    count: int
    members: List[int]  # indexes into faces list


def greedy_cluster(embs: np.ndarray, thr_sim: float) -> List[int]:
    """Return cluster assignment per embedding using greedy centroid merging.
    Two embs are in same cluster if similarity >= thr_sim to the cluster centroid.
    """
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
            # update centroid (running mean then renorm)
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


def get_video_duration(path: str) -> float:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0.0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    dur = (frames / fps) if fps > 0 else 0.0
    cap.release()
    return float(dur)


def build_segments(times: List[float], pre: float, post: float, merge_gap: float, max_dur: float) -> List[Tuple[float, float]]:
    if not times:
        return []
    ivals: List[Tuple[float, float]] = []
    for t in sorted(times):
        s = max(0.0, float(t) - float(pre))
        e = float(t) + float(post)
        if max_dur > 0:
            e = min(float(max_dur), e)
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
                # Use paths relative to the list file directory to avoid duplicate prefixes
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
    p = argparse.ArgumentParser(description="Extract prominent faces from videos and cluster similar faces")
    p.add_argument('--corpus', required=True, help='Video file or directory')
    p.add_argument('--output', required=True, help='Output directory')

    p.add_argument('--device', default=None)
    p.add_argument('--no-pretrained', action='store_true')

    p.add_argument('--detector', default='haar', choices=['haar', 'mtcnn'])
    p.add_argument('--min-face-rel-size', type=float, default=0.06)
    p.add_argument('--face-min-area-ratio', type=float, default=0.02)
    p.add_argument('--face-min-sharpness', type=float, default=60.0)

    p.add_argument('--sample-fps', type=float, default=2.0)

    p.add_argument('--embedder', default='timm', choices=['timm', 'facenet'])
    p.add_argument('--backbone', default='resnet50')
    p.add_argument('--input-size', type=int, default=224)

    p.add_argument('--cluster-scope', default='per_video', choices=['global', 'per_video'])
    p.add_argument('--similarity-threshold', type=float, default=0.65)
    p.add_argument('--min-cluster-size', type=int, default=3)
    p.add_argument('--representative-only', action='store_true')
    p.add_argument('--no-clean', action='store_true')

    p.add_argument('--no-clips', action='store_true')
    p.add_argument('--clip-pre-sec', type=float, default=0.75)
    p.add_argument('--clip-post-sec', type=float, default=0.75)
    p.add_argument('--clip-merge-gap', type=float, default=0.5)
    p.add_argument('--no-ffmpeg-reencode', action='store_true')

    p.add_argument('--debug', action='store_true')

    args = p.parse_args()

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
        clip_merge_gap_sec=args.clip_merge_gap,
        ffmpeg_reencode=not args.no_ffmpeg_reencode,
        debug=args.debug,
    )

    os.makedirs(cfg.output, exist_ok=True)

    # Load detectors/embedders
    haar = load_haar_detector()
    mtcnn = load_mtcnn(cfg.device) if cfg.detector == 'mtcnn' else None
    embedder = FaceEmbedder(device=cfg.device, embedder=cfg.embedder, backbone=cfg.backbone, input_size=cfg.input_size, pretrained=cfg.pretrained)

    videos = list_videos(cfg.corpus)
    if cfg.debug:
        print(f"Processing {len(videos)} video(s)")

    def cluster_and_save(entries: List[FaceEntry], out_subdir: str):
        if len(entries) == 0:
            return
        embs = np.vstack([e.emb for e in entries]).astype(np.float32)
        labels = greedy_cluster(embs, cfg.similarity_threshold)
        # organize clusters
        clusters: Dict[int, List[int]] = {}
        for idx, lb in enumerate(labels):
            clusters.setdefault(lb, []).append(idx)
        # sort clusters by size desc
        items = sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True)
        base = os.path.join(cfg.output, out_subdir)
        if cfg.clean_on_rerun and os.path.exists(base):
            try:
                shutil.rmtree(base)
            except Exception:
                pass
        os.makedirs(base, exist_ok=True)
        # summary CSV
        csv_path = os.path.join(base, 'clusters.csv')
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('cluster_id,size,video,frame_idx,timestamp,face_path\n')
            cid = 0
            for orig_id, idxs in items:
                if len(idxs) < cfg.min_cluster_size:
                    continue
                folder_name = f'face{cid+1}'
                cdir = os.path.join(base, folder_name)
                os.makedirs(cdir, exist_ok=True)
                # choose representative: highest similarity to cluster centroid (recompute)
                centroid = np.mean(embs[idxs, :], axis=0)
                centroid = centroid / max(1e-9, np.linalg.norm(centroid))
                sims = (embs[idxs, :] @ centroid)
                j_local = int(np.argmax(sims))
                rep_idx = idxs[j_local]
                rep_entry = entries[rep_idx]
                # copy representative
                rep_dst = os.path.join(cdir, f'representative.jpg')
                try:
                    shutil.copyfile(rep_entry.face_path, rep_dst)
                except Exception:
                    pass
                for k in idxs:
                    e = entries[k]
                    f.write(f"face{cid+1},{len(idxs)},{os.path.basename(e.video)},{e.frame_idx},{e.timestamp:.3f},{e.face_path}\n")
                    if not cfg.save_representative_only:
                        # copy or link face
                        dst = os.path.join(cdir, os.path.basename(e.face_path))
                        try:
                            shutil.copyfile(e.face_path, dst)
                        except Exception:
                            pass
                if cfg.make_clips:
                    video_src = entries[idxs[0]].video
                    times = sorted([entries[k].timestamp for k in idxs])
                    duration = get_video_duration(video_src)
                    segs = build_segments(times, cfg.clip_pre_sec, cfg.clip_post_sec, cfg.clip_merge_gap_sec, duration)
                    seg_paths = []
                    for si, (ss, ee) in enumerate(segs):
                        seg_out = os.path.join(cdir, f'seg_{si:03d}.mp4')
                        ok = save_segment_ffmpeg(video_src, ss, ee, seg_out, reencode=cfg.ffmpeg_reencode)
                        if ok:
                            seg_paths.append(seg_out)
                    combined_path = os.path.join(cdir, 'combined.mp4')
                    if seg_paths:
                        ok2 = concat_segments_ffmpeg(seg_paths, combined_path, reencode=cfg.ffmpeg_reencode)
                        if ok2 and not cfg.debug:
                            for sp in seg_paths:
                                try:
                                    os.remove(sp)
                                except Exception:
                                    pass
                cid += 1

    all_entries: List[FaceEntry] = []
    if cfg.cluster_scope == 'per_video':
        for i, vp in enumerate(videos, start=1):
            if cfg.debug:
                print(f"[{i}/{len(videos)}] {vp}")
            ents = extract_faces_from_video(vp, cfg.output, cfg, haar, mtcnn, embedder)
            if cfg.debug:
                print(f"Collected {len(ents)} faces for {os.path.basename(vp)}")
            video_name = os.path.splitext(os.path.basename(vp))[0]
            cluster_and_save(ents, f'clusters_{video_name}')
    else:
        for i, vp in enumerate(videos, start=1):
            if cfg.debug:
                print(f"[{i}/{len(videos)}] {vp}")
            ents = extract_faces_from_video(vp, cfg.output, cfg, haar, mtcnn, embedder)
            all_entries.extend(ents)
        if cfg.debug:
            print(f"Collected {len(all_entries)} faces")

    if cfg.cluster_scope == 'global':
        cluster_and_save(all_entries, 'clusters_global')

    print("Done. Output:", cfg.output)


if __name__ == '__main__':
    main()

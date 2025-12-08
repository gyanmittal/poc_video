from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import time
from typing import Optional, List, Tuple

import cv2
import numpy as np

VIDEO_CANDIDATES = [
    # Blender Foundation – Tears of Steel (CC BY)
    "https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov",
    "https://media.xiph.org/tearsofsteel/tearsofsteel-720p.webm",
]


def mkdirp(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def try_download(url: str, dst: str, timeout: int = 60) -> bool:
    try:
        print(f"[download] {url} -> {dst}")
        with urllib.request.urlopen(url, timeout=timeout) as r, open(dst, "wb") as f:
            f.write(r.read())
        return True
    except Exception as e:
        print(f"[warn] download failed: {e}")
        if os.path.exists(dst):
            try:
                os.remove(dst)
            except Exception:
                pass
        return False


def load_haar() -> Optional[cv2.CascadeClassifier]:
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


def extract_query_photos(video_path: str, out_dir: str, max_photos: int = 3, sample_fps: float = 1.0, min_rel: float = 0.06) -> int:
    mkdirp(out_dir)
    face_cascade = load_haar()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[error] failed to open video: {video_path}")
        return 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    step = 1
    if fps > 0 and sample_fps > 0:
        step = max(1, int(round(fps / sample_fps)))
    saved = 0
    idx = 0
    while saved < max_photos:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step != 0:
            idx += 1
            continue
        faces = detect_faces_haar(frame, face_cascade, min_rel=min_rel)
        if faces:
            faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
            x, y, w, h = faces[0]
            crop = crop_with_margin(frame, x, y, w, h, margin=0.25)
            cv2.imwrite(os.path.join(out_dir, f"query_{saved+1:02d}.jpg"), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            saved += 1
        idx += 1
    cap.release()
    print(f"[info] saved {saved} query photo(s) to {out_dir}")
    return saved


def main():
    ap = argparse.ArgumentParser(description="Prepare sample data (videos + query photos) for extract_person_clips.py")
    ap.add_argument('--root', default='poc_video/sample_data', help='Output root directory')
    ap.add_argument('--max-photos', type=int, default=3)
    ap.add_argument('--sample-fps', type=float, default=1.0)
    args = ap.parse_args()

    root = args.root
    vids = os.path.join(root, 'videos')
    photos = os.path.join(root, 'photos')
    mkdirp(vids)
    mkdirp(photos)

    # Download a small public video with people (Tears of Steel)
    dst = os.path.join(vids, 'tears_of_steel_720p.mov')
    ok = False
    for url in VIDEO_CANDIDATES:
        ok = try_download(url, dst)
        if ok:
            break
    if not ok:
        print("[error] could not download a sample video. Please place any video with people into:")
        print(f"        {vids}")
        print("Then re-run this script to extract query photos.")
        sys.exit(1)

    # Extract a few query photos from the downloaded video
    n = extract_query_photos(dst, photos, max_photos=args.max_photos, sample_fps=args.sample_fps)
    if n == 0:
        print("[warn] no faces were detected in the sample video. You can manually add 1-3 face photos to:")
        print(f"       {photos}")
        print("and proceed with the POC.")

    print("\n[done] Sample data ready")
    print(f"- Photos: {photos}")
    print(f"- Videos: {vids}")
    print("\nNext steps:")
    print("1) Run the POC to extract clips:")
    print("   python3 poc_video/extract_person_clips.py --photos poc_video/sample_data/photos --corpus poc_video/sample_data/videos --output poc_video/person_clips_out --sample-fps 2.0 --match-threshold 0.55 --debug")


if __name__ == '__main__':
    main()

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

import cluster_faces_v3 as v3


def _clamp_bbox(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h


def _smooth_bbox(prev: Tuple[int, int, int, int], curr: Tuple[int, int, int, int], alpha: float = 0.35) -> Tuple[int, int, int, int]:
    px, py, pw, ph = prev
    cx, cy, cw, ch = curr
    sx = int(round(alpha * cx + (1.0 - alpha) * px))
    sy = int(round(alpha * cy + (1.0 - alpha) * py))
    sw = int(round(alpha * cw + (1.0 - alpha) * pw))
    sh = int(round(alpha * ch + (1.0 - alpha) * ph))
    return sx, sy, sw, sh


def annotate_segment_opencv(
    video_src: str,
    start_sec: float,
    end_sec: float,
    cluster_frame_boxes: Dict[int, Tuple[int, int, int, int]],
    cfg: v3.FaceClusterConfig,
    out_path: str,
) -> bool:
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
    video_only_tmp = out_path + ".vtmp.mp4"
    writer = cv2.VideoWriter(video_only_tmp, fourcc, fps, (W, H))
    if not writer.isOpened():
        cap.release()
        return False

    tracker = None
    cur_idx = start_f

    det_frames = sorted(cluster_frame_boxes.keys())
    last_bb_body: Optional[Tuple[int, int, int, int]] = None
    last_smooth_body: Optional[Tuple[int, int, int, int]] = None
    linger_frames = max(1, int(0.5 * fps))
    linger_left = 0

    def nearest_detection(frame_idx: int, max_window: int = int(1.5 * fps)) -> Optional[Tuple[int, Tuple[int, int, int, int]]]:
        if not det_frames:
            return None
        lo, hi = 0, len(det_frames) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if det_frames[mid] < frame_idx:
                lo = mid + 1
            else:
                hi = mid
        best_f = det_frames[lo]
        if lo > 0 and abs(det_frames[lo - 1] - frame_idx) < abs(best_f - frame_idx):
            best_f = det_frames[lo - 1]
        if abs(best_f - frame_idx) <= max_window:
            return best_f, cluster_frame_boxes[best_f]
        return None

    init = nearest_detection(start_f)
    if init is not None:
        # Prefer user-provided tracker; recommend CSRT via CLI for stability
        tracker = v3.create_tracker(cfg.tracker_type)
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
        if det_box is not None:
            tr2 = v3.create_tracker(cfg.tracker_type)
            if tr2 is not None:
                try:
                    tr2.init(frame, (float(det_box[0]), float(det_box[1]), float(det_box[2]), float(det_box[3])))
                    tracker = tr2
                except Exception:
                    pass

        present_now = False
        to_draw: Optional[Tuple[int, int, int, int]] = None
        bb_trk: Optional[Tuple[int, int, int, int]] = None

        if tracker is not None:
            try:
                ok2, bb = tracker.update(frame)
            except Exception:
                ok2, bb = False, None
            if ok2 and bb is not None:
                bb_trk = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))

        if det_box is not None:
            bx, by, bw, bh = v3.expand_to_body_bbox(det_box, W, H, cfg)
            to_draw = (bx, by, bw, bh)
            present_now = True
        elif bb_trk is not None:
            nd = nearest_detection(cur_idx, max_window=int(1.5 * fps))
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
                union = aw * ah + bw * bh - inter
                iou = (inter / max(1e-6, union)) if union > 0 else 0.0
                if iou >= 0.10:
                    bx2, by2, bw2, bh2 = v3.expand_to_body_bbox(bb_trk, W, H, cfg)
                    to_draw = (bx2, by2, bw2, bh2)
                    present_now = True

        if to_draw is None and last_bb_body is not None and linger_left > 0:
            bx, by, bw, bh = last_bb_body
            to_draw = (bx, by, bw, bh)
            linger_left = max(0, linger_left - 1)

        # Smooth bbox with margin to reduce flicker when present
        if to_draw is not None:
            x0, y0, w0, h0 = to_draw
            pad = int(round(max(w0, h0) * 0.20))
            x0, y0, w0, h0 = x0 - pad // 2, y0 - pad // 2, w0 + pad, h0 + pad
            to_draw = _clamp_bbox(x0, y0, w0, h0, W, H)
            if last_smooth_body is not None:
                to_draw = _smooth_bbox(last_smooth_body, to_draw, alpha=0.25)
            # Snap to even pixels to reduce subpixel jitter
            x1, y1, w1, h1 = to_draw
            to_draw = _clamp_bbox((x1 // 2) * 2, (y1 // 2) * 2, (w1 // 2) * 2, (h1 // 2) * 2, W, H)

        if to_draw is not None:
            x, y, w, h = to_draw
            # Black background; paste only the person region (no grey/blur)
            out = np.zeros_like(frame)
            out[y:y + h, x:x + w] = frame[y:y + h, x:x + w]
            last_bb_body = (x, y, w, h)
            last_smooth_body = (x, y, w, h)
            if present_now:
                linger_left = linger_frames
        else:
            out = np.zeros_like(frame)

        writer.write(out)
        cur_idx += 1

    writer.release()
    cap.release()

    # Mux original audio (voice) for this time range into the masked video
    dur = max(0.0, float(end_sec) - float(start_sec))
    if dur <= 0.02:
        # Too short; just move video-only
        try:
            os.replace(video_only_tmp, out_path)
            return True
        except Exception:
            try:
                shutil.move(video_only_tmp, out_path)
                return True
            except Exception:
                return False

    mux_tmp = out_path + ".mux.mp4"
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-ss', f'{start_sec:.3f}', '-t', f'{dur:.3f}',
        '-i', video_src,  # input 0: source (for audio)
        '-i', video_only_tmp,  # input 1: masked video-only
        '-map', '1:v:0', '-map', '0:a:0',
        '-c:v', 'copy', '-c:a', 'aac',
        '-shortest', mux_tmp
    ]
    mux_ok = False
    try:
        subprocess.run(cmd, check=True)
        mux_ok = True
    except Exception:
        mux_ok = False

    try:
        if mux_ok:
            try:
                os.replace(mux_tmp, out_path)
            except Exception:
                shutil.move(mux_tmp, out_path)
            # cleanup video-only tmp
            try:
                os.remove(video_only_tmp)
            except Exception:
                pass
            return True
        else:
            # Fallback to video-only
            try:
                os.replace(video_only_tmp, out_path)
            except Exception:
                shutil.move(video_only_tmp, out_path)
            return True
    except Exception:
        return False


if __name__ == '__main__':
    # Override v3 annotator with v5 (audio + smoothing)
    v3.annotate_segment_opencv = annotate_segment_opencv
    # Ensure DB filename is versioned for v5
    import sys
    has_db = False
    for i, tok in enumerate(list(sys.argv)):
        if tok == '--db-path' and i + 1 < len(sys.argv):
            dbp = sys.argv[i + 1]
            root, ext = os.path.splitext(dbp)
            if not root.endswith('_v5'):
                sys.argv[i + 1] = f"{root}_v5{ext}"
            has_db = True
        elif tok.startswith('--db-path='):
            dbp = tok.split('=', 1)[1]
            root, ext = os.path.splitext(dbp)
            if not root.endswith('_v5'):
                sys.argv[i] = f"--db-path={root}_v5{ext}"
            has_db = True
    if not has_db:
        sys.argv.extend(['--db-path', 'poc_video/face_cache_v5.duckdb'])
    # Do not set max-video-secs here; default = full video, user can pass e.g. --max-video-secs 500
    v3.main()

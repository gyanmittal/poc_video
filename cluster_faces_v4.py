from __future__ import annotations

import os
import shutil
from typing import Dict, Optional, Tuple

import sys
import cv2
import numpy as np

import cluster_faces_v3 as v3


def annotate_segment_opencv(video_src: str,
                            start_sec: float,
                            end_sec: float,
                            cluster_frame_boxes: Dict[int, Tuple[int, int, int, int]],
                            cfg: v3.FaceClusterConfig,
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

    tracker = None
    cur_idx = start_f

    det_frames = sorted(cluster_frame_boxes.keys())
    last_bb_body: Optional[Tuple[int, int, int, int]] = None
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

        if to_draw is not None:
            x, y, w, h = to_draw
            x = max(0, min(x, W - 1))
            y = max(0, min(y, H - 1))
            w = max(1, min(w, W - x))
            h = max(1, min(h, H - y))
            # Blur background, keep the person's region sharp
            k = max(5, int(round(min(W, H) * 0.05)))
            if k % 2 == 0:
                k += 1
            out = cv2.GaussianBlur(frame, (k, k), 0)
            out[y:y + h, x:x + w] = frame[y:y + h, x:x + w]
            last_bb_body = (x, y, w, h)
            if present_now:
                linger_left = linger_frames
        else:
            out = np.zeros_like(frame)

        writer.write(out)
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


if __name__ == '__main__':
    v3.annotate_segment_opencv = annotate_segment_opencv
    has_db = False
    for i, tok in enumerate(list(sys.argv)):
        if tok == '--db-path' and i + 1 < len(sys.argv):
            dbp = sys.argv[i + 1]
            root, ext = os.path.splitext(dbp)
            if not root.endswith('_v4'):
                sys.argv[i + 1] = f"{root}_v4{ext}"
            has_db = True
        elif tok.startswith('--db-path='):
            dbp = tok.split('=', 1)[1]
            root, ext = os.path.splitext(dbp)
            if not root.endswith('_v4'):
                sys.argv[i] = f"--db-path={root}_v4{ext}"
            has_db = True
    if not has_db:
        sys.argv.extend(['--db-path', 'poc_video/face_cache_v4.duckdb'])
    v3.main()

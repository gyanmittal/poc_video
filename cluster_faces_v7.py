from __future__ import annotations

import os
import shutil
import subprocess
from typing import Dict, Optional, Tuple

import cluster_faces_v6 as v6


def annotate_segment_opencv(
    video_src: str,
    start_sec: float,
    end_sec: float,
    cluster_frame_boxes: Dict[int, Tuple[int, int, int, int]],
    cfg: v6.FaceClusterConfig,
    out_path: str,
) -> bool:
    video_only_tmp = out_path + ".vtmp.mp4"
    ok = v6._orig_annotate(video_src, start_sec, end_sec, cluster_frame_boxes, cfg, video_only_tmp)
    if not ok:
        try:
            if os.path.exists(video_only_tmp):
                os.remove(video_only_tmp)
        except Exception:
            pass
        return False

    dur = max(0.0, float(end_sec) - float(start_sec))
    if dur <= 0.02:
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

    def try_mux(filter_expr: Optional[str]) -> bool:
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-ss', f'{start_sec:.3f}', '-t', f'{dur:.3f}',
            '-i', video_src,
            '-i', video_only_tmp,
        ]
        if filter_expr:
            cmd += ['-filter_complex', f'[0:a:0]{filter_expr}[a]']
            cmd += ['-map', '1:v:0', '-map', '[a]']
        else:
            cmd += ['-map', '1:v:0', '-map', '0:a:0']
        cmd += ['-c:v', 'copy', '-c:a', 'aac', '-shortest', mux_tmp]
        try:
            subprocess.run(cmd, check=True)
            return True
        except Exception:
            return False

    # Attempt robust vocal isolation
    filters = [
        'pan=mono|c0=0.5*FL+0.5*FR,highpass=f=200,lowpass=f=3800,afftdn=nf=-25,alimiter=limit=0.95',
        'highpass=f=200,lowpass=f=3800,afftdn=nf=-25,alimiter=limit=0.95',
        'highpass=f=200,lowpass=f=3800,alimiter=limit=0.95',
        None,  # final fallback: keep original audio if filters unavailable
    ]

    mux_ok = False
    for fexpr in filters:
        if try_mux(fexpr):
            mux_ok = True
            break

    try:
        if mux_ok:
            try:
                os.replace(mux_tmp, out_path)
            except Exception:
                shutil.move(mux_tmp, out_path)
            try:
                os.remove(video_only_tmp)
            except Exception:
                pass
            return True
        else:
            try:
                os.replace(video_only_tmp, out_path)
            except Exception:
                shutil.move(video_only_tmp, out_path)
            return True
    except Exception:
        return False


if __name__ == '__main__':
    v6.annotate_segment_opencv = annotate_segment_opencv
    v6.main()

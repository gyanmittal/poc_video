from __future__ import annotations

import os
import sys
import shutil
import subprocess
import argparse
from typing import Dict, Optional, Tuple, List

import cv2

import cluster_faces_v6 as v6

# v8 options (set in __main__)
_DIARIZE: bool = False
_HF_TOKEN: Optional[str] = None
_USE_BANDPASS: bool = True
_MIN_SPK_SEG_DUR: float = 0.25
_PRES_PAD: float = 0.15


def _merge_intervals(intervals: List[Tuple[float, float]], min_gap: float = 0.15) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted([(float(max(0.0, s)), float(max(0.0, e))) for s, e in intervals if e > s], key=lambda x: x[0])
    out: List[Tuple[float, float]] = []
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s <= ce + float(min_gap):
            ce = max(ce, e)
        else:
            out.append((cs, ce))
            cs, ce = s, e
    out.append((cs, ce))
    return out


def _overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    return max(0.0, e - s)


def _sum_overlap(a_list: List[Tuple[float, float]], b_list: List[Tuple[float, float]]) -> float:
    tot = 0.0
    if not a_list or not b_list:
        return 0.0
    i, j = 0, 0
    a_list = sorted(a_list)
    b_list = sorted(b_list)
    while i < len(a_list) and j < len(b_list):
        a = a_list[i]
        b = b_list[j]
        tot += _overlap(a, b)
        if a[1] < b[1]:
            i += 1
        else:
            j += 1
    return tot


def _presence_intervals_from_boxes(cluster_frame_boxes: Dict[int, Tuple[int, int, int, int]], start_sec: float, end_sec: float, fps: float) -> List[Tuple[float, float]]:
    try:
        start_f = int(round(float(start_sec) * float(fps)))
        end_f = int(round(float(end_sec) * float(fps)))
        dur = max(0.0, float(end_sec) - float(start_sec))
        ts = []
        for f in sorted(cluster_frame_boxes.keys()):
            if f < start_f or f >= end_f:
                continue
            t = (float(f) / float(fps)) - float(start_sec)
            ts.append(max(0.0, min(dur, t)))
        if not ts:
            return []
        # Merge close timestamps into intervals, pad edges
        thr = max(0.20, 3.0 / max(1.0, float(fps)))
        ivals: List[Tuple[float, float]] = []
        s = max(0.0, ts[0] - _PRES_PAD)
        e = min(dur, ts[0] + _PRES_PAD)
        for t in ts[1:]:
            if t - e <= thr:
                e = min(dur, max(e, t + _PRES_PAD))
            else:
                ivals.append((s, e))
                s = max(0.0, t - _PRES_PAD)
                e = min(dur, t + _PRES_PAD)
        ivals.append((s, e))
        return _merge_intervals(ivals, min_gap=thr)
    except Exception:
        return []


def _diarize_intervals(audio_wav_path: str, token: Optional[str]) -> Dict[str, List[Tuple[float, float]]]:
    out: Dict[str, List[Tuple[float, float]]] = {}
    try:
        try:
            from pyannote.audio import Pipeline  # type: ignore
        except Exception:
            return out
        # Prefer 3.1; fallback to legacy
        pipeline_id = "pyannote/speaker-diarization-3.1"
        try:
            pipeline = Pipeline.from_pretrained(pipeline_id, use_auth_token=token) if token else Pipeline.from_pretrained(pipeline_id)
        except Exception:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=token) if token else Pipeline.from_pretrained("pyannote/speaker-diarization")
        diar = pipeline({"audio": audio_wav_path})
        for turn, _, spk in diar.itertracks(yield_label=True):
            s = float(getattr(turn, 'start', 0.0) or 0.0)
            e = float(getattr(turn, 'end', 0.0) or 0.0)
            if e > s:
                out.setdefault(str(spk), []).append((s, e))
        # Merge and drop tiny segments
        for k, segs in list(out.items()):
            merged = _merge_intervals(segs, min_gap=0.05)
            out[k] = [(s, e) for (s, e) in merged if (e - s) >= _MIN_SPK_SEG_DUR]
        return out
    except Exception:
        return {}


def annotate_segment_opencv(
    video_src: str,
    start_sec: float,
    end_sec: float,
    cluster_frame_boxes: Dict[int, Tuple[int, int, int, int]],
    cfg: v6.FaceClusterConfig,
    out_path: str,
) -> bool:
    # Render video-only first using v6's original annotator (no audio)
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

    # Build presence intervals from frame boxes (for fallback or to choose speaker)
    fps = 0.0
    try:
        cap = cv2.VideoCapture(video_src)
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
    except Exception:
        fps = 0.0
    pres_intervals: List[Tuple[float, float]] = []
    if fps > 0:
        pres_intervals = _presence_intervals_from_boxes(cluster_frame_boxes, float(start_sec), float(end_sec), float(fps))

    # Try diarization to find target speaker
    spk_intervals: List[Tuple[float, float]] = []
    if _DIARIZE:
        clip_wav = out_path + ".clip.wav"
        try:
            # mono 16k wav for diarization
            cmd_wav = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-ss', f'{start_sec:.3f}', '-t', f'{dur:.3f}',
                '-i', video_src, '-vn', '-ac', '1', '-ar', '16000', clip_wav
            ]
            subprocess.run(cmd_wav, check=True)
            by_spk = _diarize_intervals(clip_wav, _HF_TOKEN)
            if by_spk:
                # pick speaker with max overlap with presence intervals; fallback to longest speech
                best_label = None
                best_score = -1.0
                for spk, segs in by_spk.items():
                    score = _sum_overlap(segs, pres_intervals) if pres_intervals else sum(e - s for s, e in segs)
                    if score > best_score:
                        best_score = score
                        best_label = spk
                if best_label is not None:
                    spk_intervals = by_spk.get(best_label, [])
        except Exception:
            spk_intervals = []
        finally:
            try:
                if os.path.exists(clip_wav):
                    os.remove(clip_wav)
            except Exception:
                pass

    # Construct filter: voice bandpass + gating intervals
    def _build_gating_expr(ivals: List[Tuple[float, float]]) -> Optional[str]:
        if not ivals:
            return None
        parts = [f"between(t,{max(0.0,s):.3f},{max(0.0,e):.3f})" for s, e in ivals if e > s]
        if not parts:
            return None
        return "+".join(parts)

    gating_ivals = spk_intervals if spk_intervals else pres_intervals
    gating_expr = _build_gating_expr(gating_ivals)

    filter_chain_parts: List[str] = []
    if _USE_BANDPASS:
        filter_chain_parts.append('highpass=f=200,lowpass=f=3800,afftdn=nf=-25,alimiter=limit=0.95')
    if gating_expr:
        filter_chain_parts.append(f"volume=if(gt({gating_expr},0),1,0)")
    # ensure sane format
    filter_chain_parts.append('aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo')

    if filter_chain_parts:
        filter_complex = f"[0:a:0]{','.join(filter_chain_parts)}[a]"
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-ss', f'{start_sec:.3f}', '-t', f'{dur:.3f}',
            '-i', video_src,
            '-i', video_only_tmp,
            '-filter_complex', filter_complex,
            '-map', '1:v:0', '-map', '[a]',
            '-c:v', 'copy', '-c:a', 'aac', '-shortest', mux_tmp
        ]
    else:
        # fallback: keep original audio for this range
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-ss', f'{start_sec:.3f}', '-t', f'{dur:.3f}',
            '-i', video_src,
            '-i', video_only_tmp,
            '-map', '1:v:0', '-map', '0:a:0',
            '-c:v', 'copy', '-c:a', 'aac', '-shortest', mux_tmp
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
    # Parse v8-specific flags and pass remaining through to v6
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--diarize', action='store_true', help='Enable speaker diarization to keep only target speaker voice')
    p.add_argument('--hf-token', default=None, help='HuggingFace access token for pyannote models')
    p.add_argument('--no-bandpass', action='store_true', help='Disable bandpass/noise-reduction filter')
    p.add_argument('--min-spk-dur', type=float, default=0.25, help='Minimum diarized speaker segment duration (seconds)')
    p.add_argument('--presence-pad', type=float, default=0.15, help='Padding added around face presence timestamps (seconds)')
    args, rest = p.parse_known_args()

    _DIARIZE = bool(args.diarize)
    _HF_TOKEN = args.hf_token or os.environ.get('HF_TOKEN')
    _USE_BANDPASS = not bool(args.no_bandpass)
    _MIN_SPK_SEG_DUR = float(args.min_spk_dur)
    _PRES_PAD = float(args.presence_pad)

    # Override v6 annotator then delegate to v6 main
    v6.annotate_segment_opencv = annotate_segment_opencv
    sys.argv = [sys.argv[0]] + rest
    v6.main()

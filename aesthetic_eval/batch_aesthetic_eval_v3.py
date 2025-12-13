#!/usr/bin/env python3
from __future__ import annotations

import os

# Allow PyTorch and optional IQA dependencies to coexist when they each ship a
# bundled OpenMP runtime. Without this on macOS, libomp aborts on duplicate init.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS", "0")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("BATCH_AESTHETIC_FORCE_CPU", "1")

import argparse
import atexit
import multiprocessing as mp
import queue
import shutil
import subprocess
import tempfile
import time
from typing import Dict, List, Optional, cast

from urllib.request import urlretrieve

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import clip  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    clip = None

try:
    import pyiqa  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pyiqa = None

try:
    import piq  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    piq = None

_BRISQUE_STANDALONE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "batch_aesthetic_eval_v1.py"
)
_BRISQUE_AVAILABLE = os.path.exists(_BRISQUE_STANDALONE)

LAION_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "laion_aesthetic")
CLIP_MODEL_SHORT = "vit_l_14"

_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_AESTHETIC_HEAD = None
_AESTHETIC_DEVICE = None

_PIQ_METRIC = None
_NIMA_REQUEST_QUEUE: Optional[mp.Queue] = None
_NIMA_RESPONSE_QUEUE: Optional[mp.Queue] = None
_NIMA_WORKER: Optional[mp.Process] = None
_NIMA_TASK_COUNTER = 0

# Score thresholds for selecting high-quality images.
CLIP_THRESHOLD = 5.0
NIMA_THRESHOLD = 4.0
PIQ_THRESHOLD = 0.5
BRISQUE_THRESHOLD = 0.35


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    mins, sec = divmod(int(seconds + 0.5), 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs:d}h {mins:02d}m {sec:02d}s"
    if mins > 0:
        return f"{mins:d}m {sec:02d}s"
    return f"{sec:d}s"


def _device() -> str:
    if os.environ.get("BATCH_AESTHETIC_FORCE_CPU") == "1":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _ensure_clip_aesthetic_models() -> None:
    global _CLIP_MODEL, _CLIP_PREPROCESS, _AESTHETIC_HEAD, _AESTHETIC_DEVICE
    if _CLIP_MODEL is not None and _AESTHETIC_HEAD is not None:
        return
    if clip is None:
        raise RuntimeError("clip package is required for CLIP+LAION scoring")
    device = _device()
    model, preprocess = clip.load("ViT-L/14", device=device)  # type: ignore
    head = _load_laion_head(device=device)
    _CLIP_MODEL = model
    _CLIP_PREPROCESS = preprocess
    _AESTHETIC_HEAD = head
    _AESTHETIC_DEVICE = device


def _load_laion_head(device: str) -> torch.nn.Module:
    os.makedirs(LAION_CACHE_DIR, exist_ok=True)
    head_path = os.path.join(LAION_CACHE_DIR, f"sa_0_4_{CLIP_MODEL_SHORT}_linear.pth")
    if not os.path.exists(head_path):
        print(f"Downloading LAION aesthetic head to {head_path} ...")
        url = (
            "https://github.com/LAION-AI/aesthetic-predictor/raw/main/"
            f"sa_0_4_{CLIP_MODEL_SHORT}_linear.pth"
        )
        urlretrieve(url, head_path)
    dim = 768 if CLIP_MODEL_SHORT == "vit_l_14" else 512
    head = torch.nn.Linear(dim, 1)
    state = torch.load(head_path, map_location="cpu")
    head.load_state_dict(state)
    head.eval().to(device)
    return head


def score_clip_aesthetic(image: Image.Image) -> Optional[float]:
    try:
        _ensure_clip_aesthetic_models()
    except Exception:
        return None
    assert _CLIP_MODEL is not None and _CLIP_PREPROCESS is not None
    assert _AESTHETIC_HEAD is not None and _AESTHETIC_DEVICE is not None
    with torch.no_grad():
        clip_input = _CLIP_PREPROCESS(image).unsqueeze(0).to(_AESTHETIC_DEVICE)
        image_features = _CLIP_MODEL.encode_image(clip_input)  # type: ignore[attr-defined]
        image_features = F.normalize(image_features, dim=-1)
        score = _AESTHETIC_HEAD(image_features)
        return float(score.squeeze(0).cpu().item())


def _stop_nima_worker() -> None:
    global _NIMA_REQUEST_QUEUE, _NIMA_RESPONSE_QUEUE, _NIMA_WORKER
    if _NIMA_REQUEST_QUEUE is not None:
        try:
            _NIMA_REQUEST_QUEUE.put_nowait(None)
        except Exception:
            pass
    worker = _NIMA_WORKER
    if worker is not None:
        worker.join(timeout=5)
    _NIMA_REQUEST_QUEUE = None
    _NIMA_RESPONSE_QUEUE = None
    _NIMA_WORKER = None


def _ensure_nima_worker() -> bool:
    global _NIMA_REQUEST_QUEUE, _NIMA_RESPONSE_QUEUE, _NIMA_WORKER
    if pyiqa is None:
        return False
    if _NIMA_WORKER is not None and _NIMA_WORKER.is_alive():
        return True

    _stop_nima_worker()
    ctx = mp.get_context("spawn")
    req_queue: mp.Queue = ctx.Queue()
    resp_queue: mp.Queue = ctx.Queue()
    worker = ctx.Process(
        target=_nima_worker_main,
        args=(req_queue, resp_queue, _device()),
        daemon=True,
    )
    worker.start()
    try:
        status, _ = resp_queue.get(timeout=120)
    except queue.Empty:
        _stop_nima_worker()
        return False
    if status != "READY":
        _stop_nima_worker()
        return False

    _NIMA_REQUEST_QUEUE = req_queue
    _NIMA_RESPONSE_QUEUE = resp_queue
    _NIMA_WORKER = worker
    atexit.register(_stop_nima_worker)
    return True


def _nima_worker_main(
    req_queue: mp.Queue, resp_queue: mp.Queue, device: str
) -> None:
    if pyiqa is None:
        resp_queue.put(("INIT_ERROR", "pyiqa missing"))
        return
    try:
        metric = pyiqa.create_metric("nima", device=device)
    except Exception as exc:  # pragma: no cover - initialization errors
        resp_queue.put(("INIT_ERROR", str(exc)))
        return

    resp_queue.put(("READY", None))

    while True:
        item = req_queue.get()
        if item is None:
            break
        task_id, array = item
        try:
            tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                score = metric(tensor.to(device))
            value: Optional[float] = float(score.detach().cpu().item())
        except Exception:
            value = None
        resp_queue.put((task_id, value))

    resp_queue.put(("STOPPED", None))


def score_nima_pyiqa(image: Image.Image) -> Optional[float]:
    global _NIMA_TASK_COUNTER
    if pyiqa is None:
        return None
    if os.environ.get("BATCH_AESTHETIC_DISABLE_NIMA") == "1":
        return None
    if not _ensure_nima_worker():
        return None
    assert _NIMA_REQUEST_QUEUE is not None
    assert _NIMA_RESPONSE_QUEUE is not None

    array = np.asarray(image).astype(np.float32) / 255.0
    _NIMA_TASK_COUNTER += 1
    task_id = _NIMA_TASK_COUNTER

    try:
        _NIMA_REQUEST_QUEUE.put((task_id, array), timeout=5)
    except Exception:
        _stop_nima_worker()
        return None

    try:
        while True:
            resp_id, value = _NIMA_RESPONSE_QUEUE.get(timeout=120)
            if resp_id == task_id:
                return value
            if resp_id in {"STOPPED", "INIT_ERROR"}:
                return None
    except queue.Empty:
        return None


def _compute_brisque_scores(
    folder: str, image_paths: List[str]
) -> Dict[str, Optional[float]]:
    if os.environ.get("BATCH_AESTHETIC_DISABLE_BRISQUE") == "1":
        return {path: None for path in image_paths}
    if not _BRISQUE_AVAILABLE:
        return {path: None for path in image_paths}

    tmpdir = tempfile.mkdtemp(prefix="aesthetic_brisque_")
    dataset_name = os.path.basename(os.path.normpath(folder))
    out_path = os.path.join(tmpdir, f"{dataset_name}_brisque_scores.md")
    cmd = ["python3", _BRISQUE_STANDALONE, folder, "--output", tmpdir]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not os.path.exists(out_path):
            return {path: None for path in image_paths}
        with open(out_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    scores: Dict[str, Optional[float]] = {path: None for path in image_paths}
    for line in lines:
        if not line.startswith("|") or "Image" in line:
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) != 2:
            continue
        path, value = parts
        if path in scores:
            if value.upper() == "N/A":
                scores[path] = None
            else:
                try:
                    scores[path] = float(value)
                except ValueError:
                    scores[path] = None
    return scores


def score_piq(image: Image.Image) -> Optional[float]:
    global _PIQ_METRIC
    if piq is None:
        return None
    if _PIQ_METRIC is None:
        try:
            _PIQ_METRIC = piq.CLIPIQA().to(_device())
        except Exception:
            _PIQ_METRIC = None
            return None
    tensor = torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(_device())
    try:
        score = _PIQ_METRIC(tensor)
        return float(score.detach().cpu().item())
    except Exception:
        return None


def score_brisque(image_path: str, cache: Dict[str, Optional[float]]) -> Optional[float]:
    return cache.get(image_path)


def collect_images(folder: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    images: List[str] = []
    for root, _, files in os.walk(folder):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in exts:
                images.append(os.path.join(root, fname))
    images.sort()
    return images


def format_markdown(results: List[Dict[str, Optional[float]]], approaches: List[str]) -> str:
    headers = ["Image"] + approaches
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for entry in results:
        row = [entry["image"]]
        for name in approaches:
            value = entry.get(name)
            row.append(f"{value:.3f}" if isinstance(value, (int, float)) else "N/A")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute aesthetic scores for a folder of images.")
    parser.add_argument("folder", help="Folder containing images to score")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional directory where the Markdown table will be written (file name auto-generated)",
    )
    args = parser.parse_args()

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Input folder not found: {folder}")

    image_paths = collect_images(folder)
    if not image_paths:
        print("No images found.")
        return

    approaches = [
        "CLIP_LAION",
        "NIMA_pyiqa",
        "piq_CLIPIQA",
        "BRISQUE",
    ]

    brisque_scores = _compute_brisque_scores(folder, image_paths)

    total_images = len(image_paths)
    start_time = time.time()
    results: List[Dict[str, Optional[float]]] = []
    processed = 0
    selection_dir: Optional[str] = None
    rejected_dir: Optional[str] = None
    mapping_lines: Optional[List[str]] = None
    mapping_count = 0
    rejected_count = 0
    dataset_name: Optional[str] = None
    output_dir: Optional[str] = None

    def _passes_copy_filters(entry: Dict[str, Optional[float]]) -> bool:
        clip_score = entry.get("CLIP_LAION")
        nima_score = entry.get("NIMA_pyiqa")
        piq_score = entry.get("piq_CLIPIQA")
        brisque_score = entry.get("BRISQUE")
        if not all(isinstance(x, (int, float)) for x in (clip_score, nima_score, piq_score, brisque_score)):
            return False
        return (
            cast(float, clip_score) >= CLIP_THRESHOLD
            and cast(float, nima_score) >= 5.0
            and cast(float, piq_score) >= 0.4
            and cast(float, brisque_score) <= 40.0
        )

    def _fmt_score(value: Optional[float]) -> str:
        return "NA" if not isinstance(value, (int, float)) else f"{value:.2f}"

    if args.output:
        output_dir = os.path.abspath(args.output)
        os.makedirs(output_dir, exist_ok=True)
        dataset_name = os.path.basename(os.path.normpath(folder))
        selection_dir = os.path.join(output_dir, f"{dataset_name}_selected")
        if os.path.exists(selection_dir):
            shutil.rmtree(selection_dir)
        os.makedirs(selection_dir, exist_ok=True)
        rejected_dir = os.path.join(output_dir, f"{dataset_name}_rejected")
        if os.path.exists(rejected_dir):
            shutil.rmtree(rejected_dir)
        os.makedirs(rejected_dir, exist_ok=True)
        mapping_lines = [
            "| Original | Copied | CLIP_LAION | NIMA_pyiqa | piq_CLIPIQA | BRISQUE |",
            "|---|---|---|---|---|---|",
        ]

    for path in image_paths:
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            continue
        entry: Dict[str, Optional[float]] = {"image": path}
        entry["CLIP_LAION"] = score_clip_aesthetic(image)
        entry["NIMA_pyiqa"] = score_nima_pyiqa(image)
        entry["piq_CLIPIQA"] = score_piq(image)
        entry["BRISQUE"] = score_brisque(path, brisque_scores)
        results.append(entry)
        if selection_dir and rejected_dir and mapping_lines:
            _, ext = os.path.splitext(os.path.basename(path))
            if _passes_copy_filters(entry):
                mapping_count += 1
                new_name = (
                    f"out{mapping_count}_"
                    f"CLIP_LAION-{entry['CLIP_LAION']:.2f}_"
                    f"NIMA_pyiqa-{entry['NIMA_pyiqa']:.2f}_"
                    f"piq_CLIPIQA-{entry['piq_CLIPIQA']:.2f}_"
                    f"BRISQUE-{entry['BRISQUE']:.2f}{ext}"
                )
                dst = os.path.join(selection_dir, new_name)
                copied_path = dst
                try:
                    shutil.copy2(path, dst)
                except Exception:
                    print(f"Failed to copy {path} to {dst}")
                    copied_path = "COPY_FAILED"
                mapping_lines.append(
                    "| {orig} | {copied} | {clip:.3f} | {nima:.3f} | {piq:.3f} | {brisque:.3f} |".format(
                        orig=path,
                        copied=copied_path,
                        clip=entry["CLIP_LAION"],
                        nima=entry["NIMA_pyiqa"],
                        piq=entry["piq_CLIPIQA"],
                        brisque=entry["BRISQUE"],
                    )
                )
            else:
                rejected_count += 1
                reject_name = (
                    f"reject{rejected_count}_"
                    f"CLIP_LAION-{_fmt_score(entry['CLIP_LAION'])}_"
                    f"NIMA_pyiqa-{_fmt_score(entry['NIMA_pyiqa'])}_"
                    f"piq_CLIPIQA-{_fmt_score(entry['piq_CLIPIQA'])}_"
                    f"BRISQUE-{_fmt_score(entry['BRISQUE'])}{ext}"
                )
                reject_dst = os.path.join(rejected_dir, reject_name)
                try:
                    shutil.copy2(path, reject_dst)
                except Exception:
                    print(f"Failed to copy rejected {path} to {reject_dst}")
        processed += 1
        elapsed = time.time() - start_time
        avg = elapsed / processed if processed else 0.0
        expected_total = avg * total_images if processed else 0.0
        percent = (processed / total_images) * 100 if total_images else 100.0
        progress_msg = (
            f"[{processed}/{total_images}] "
            f"{percent:5.1f}% "
            f"elapsed {_format_duration(elapsed)} / {_format_duration(expected_total)}"
        )
        print(progress_msg)

    table = format_markdown(results, approaches)

    if output_dir and dataset_name:
        out_path = os.path.join(output_dir, f"{dataset_name}_aesthetic_scores.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(table + "\n")
        print(f"Saved results to {out_path}")
        if mapping_lines and mapping_count > 0:
            mapping_path = os.path.join(
                output_dir, f"{dataset_name}_aesthetic_scores_mapping.md"
            )
            with open(mapping_path, "w", encoding="utf-8") as f:
                f.write("\n".join(mapping_lines) + "\n")
            print(f"Saved image mapping to {mapping_path}")
    else:
        print(table)


if __name__ == "__main__":
    main()
"""
Batch image aesthetic evaluation script.

Given a folder of images, this script computes scores using the following approaches
whenever their dependencies are installed locally:
    * CLIP + LAION aesthetic head
    * NIMA (via pyiqa)
    * pyiqa metrics (NIMA/other IQA models)
    * piq (perceptual image-quality metrics)
    * BRISQUE (via brisque/libsvm)

Results are printed as a Markdown table with the absolute path of each image so you
can open them directly from terminals/editors that understand file links.

Installation prerequisites:
1. Install Python dependencies:
   pip install -r poc_video/aesthetic_eval/requirements.txt

2. Run the evaluator (CPU-only by default to avoid MPS crashes):
   python3 poc_video/aesthetic_eval/batch_aesthetic_eval.py \
       <input_folder> \
       --output <output_folder>

   NIMA scoring runs inside a helper subprocess to sidestep macOS
   OpenMP/libomp crashes. Set `BATCH_AESTHETIC_DISABLE_NIMA=1` if you need
   to disable that metric entirely. BRISQUE scores are delegated to the
   standalone `batch_aesthetic_eval_v1.py` script via subprocess so libsvm's
   OpenMP runtime loads in a separate interpreter. Disable it with
   `BATCH_AESTHETIC_DISABLE_BRISQUE=1` if necessary.
"""

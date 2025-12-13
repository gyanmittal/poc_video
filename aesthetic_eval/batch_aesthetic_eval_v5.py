#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
import multiprocessing as mp
import shutil
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from urllib.request import urlretrieve

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Ensure PyTorch/libsvm OpenMP runtimes can coexist on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS", "0")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("BATCH_AESTHETIC_FORCE_CPU", "1")

try:
    import clip  # type: ignore
except ImportError:  # pragma: no cover
    clip = None

try:
    import pyiqa  # type: ignore
except ImportError:  # pragma: no cover
    pyiqa = None

try:
    import piq  # type: ignore
except ImportError:  # pragma: no cover
    piq = None

LAION_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "laion_aesthetic")
CLIP_MODEL_SHORT = "vit_l_14"

CLIP_THRESHOLD = 5.0
NIMA_THRESHOLD = 5.0
PIQ_THRESHOLD = 0.65
MIN_OVERALL_RATING = 9.0
MIN_NIMA_SCORE = 5.0
MIN_PIQ_SCORE = 0.8

WORKER_BATCH_SIZE = int(os.environ.get("BATCH_AESTHETIC_BATCH", "8"))
WORKER_PROCESSES = int(
    os.environ.get(
        "BATCH_AESTHETIC_WORKERS", str(max(1, min(8, mp.cpu_count() // 2 or 1)))
    )
)
DEFAULT_MAX_IMAGE_SIZE = int(os.environ.get("BATCH_AESTHETIC_MAX_SIDE", "1024"))
MAX_IMAGE_SIZE = DEFAULT_MAX_IMAGE_SIZE

_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_AESTHETIC_HEAD = None
_AESTHETIC_DEVICE: Optional[str] = None

_NIMA_METRIC = None
_NIMA_DEVICE: Optional[str] = None
_PIQ_METRIC = None
_PIQ_DEVICE: Optional[str] = None


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


def _ensure_clip_aesthetic_models(device_override: Optional[str] = None) -> None:
    global _CLIP_MODEL, _CLIP_PREPROCESS, _AESTHETIC_HEAD, _AESTHETIC_DEVICE
    if _CLIP_MODEL is not None and _AESTHETIC_HEAD is not None:
        return
    if clip is None:
        raise RuntimeError("clip package is required for CLIP+LAION scoring")
    device = device_override or _device()
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


def _get_nima_metric(device: str):
    global _NIMA_METRIC, _NIMA_DEVICE
    if pyiqa is None:
        return None
    if _NIMA_METRIC is None or _NIMA_DEVICE != device:
        _NIMA_METRIC = pyiqa.create_metric("nima", device=device)
        _NIMA_DEVICE = device
    return _NIMA_METRIC


def _get_piq_metric(device: str):
    global _PIQ_METRIC, _PIQ_DEVICE
    if piq is None:
        return None
    if _PIQ_METRIC is None or _PIQ_DEVICE != device:
        _PIQ_METRIC = piq.CLIPIQA().to(device)
        _PIQ_DEVICE = device
    return _PIQ_METRIC


def _fmt_score(value: Optional[float]) -> str:
    return "NA" if not isinstance(value, (int, float)) else f"{value:.2f}"


def _compute_overall_rating(entry: Dict[str, Optional[float]]) -> Optional[float]:
    clip_score = entry.get("CLIP_LAION")
    nima_score = entry.get("NIMA_pyiqa")
    piq_score = entry.get("piq_CLIPIQA")
    if not all(isinstance(x, (int, float)) for x in (clip_score, nima_score, piq_score)):
        return None
    clip_val = float(clip_score)  # type: ignore[arg-type]
    nima_val = float(nima_score)  # type: ignore[arg-type]
    piq_val = float(piq_score)  # type: ignore[arg-type]
    return (clip_val + nima_val) * piq_val


def _passes_copy_filters(entry: Dict[str, Optional[float]]) -> bool:
    clip_score = entry.get("CLIP_LAION")
    nima_score = entry.get("NIMA_pyiqa")
    piq_score = entry.get("piq_CLIPIQA")
    if not isinstance(clip_score, (int, float)) or clip_score < CLIP_THRESHOLD:
        return False
    if not isinstance(nima_score, (int, float)) or nima_score < MIN_NIMA_SCORE:
        return False
    if not isinstance(piq_score, (int, float)) or piq_score < MIN_PIQ_SCORE:
        return False
    if (
        isinstance(clip_score, (int, float))
        and isinstance(nima_score, (int, float))
        and isinstance(piq_score, (int, float))
        and (clip_score + nima_score) * piq_score <= 5.5
    ):
        return False
    if isinstance(clip_score, (int, float)) and clip_score >= 6.0:
        return True
    if isinstance(nima_score, (int, float)) and nima_score >= 5.5:
        return True
    if isinstance(clip_score, (int, float)) and isinstance(nima_score, (int, float)):
        if (clip_score + nima_score) >= 10.5:
            return True
    if isinstance(piq_score, (int, float)) and piq_score >= 0.85:
        return True
    if (
        isinstance(clip_score, (int, float))
        and isinstance(nima_score, (int, float))
        and isinstance(piq_score, (int, float))
        and (clip_score + nima_score) * piq_score >= 8.0
    ):
        return True
    numeric_fields = (clip_score, nima_score, piq_score)
    if not all(isinstance(x, (int, float)) for x in numeric_fields):
        return False
    clip_val = float(clip_score)  # type: ignore[arg-type]
    nima_val = float(nima_score)  # type: ignore[arg-type]
    piq_val = float(piq_score)  # type: ignore[arg-type]
    return (
        clip_val >= CLIP_THRESHOLD
        and nima_val >= NIMA_THRESHOLD
        and piq_val >= max(PIQ_THRESHOLD, MIN_PIQ_SCORE)
    )


def _chunked(seq: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]


def _downscale_image(image: Image.Image) -> Image.Image:
    if MAX_IMAGE_SIZE <= 0:
        return image
    width, height = image.size
    if max(width, height) <= MAX_IMAGE_SIZE:
        return image
    scale = MAX_IMAGE_SIZE / float(max(width, height))
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.BICUBIC)


def _process_batch(paths: Sequence[str]) -> List[Dict[str, Optional[float]]]:
    torch.set_num_threads(1)
    device = _device()
    _ensure_clip_aesthetic_models(device_override=device)
    nima_metric = _get_nima_metric(device)
    piq_metric = _get_piq_metric(device)

    valid_paths: List[str] = []
    clip_tensors: List[torch.Tensor] = []
    image_tensors: List[torch.Tensor] = []

    for path in paths:
        try:
            with Image.open(path) as img:
                image = _downscale_image(img.convert("RGB"))
        except Exception:
            continue
        valid_paths.append(path)
        clip_tensors.append(_CLIP_PREPROCESS(image))  # type: ignore[arg-type]
        tensor = torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1)
        image_tensors.append(tensor)

    if not valid_paths:
        return []

    results: List[Dict[str, Optional[float]]] = []
    clip_batch = torch.stack(clip_tensors, dim=0).to(_AESTHETIC_DEVICE)
    with torch.no_grad():
        clip_features = _CLIP_MODEL.encode_image(clip_batch)  # type: ignore[attr-defined]
        clip_features = F.normalize(clip_features, dim=-1)
        clip_scores = (
            _AESTHETIC_HEAD(clip_features).squeeze(1).detach().cpu().tolist()  # type: ignore[call-arg]
        )

    max_h = max(tensor.shape[1] for tensor in image_tensors)
    max_w = max(tensor.shape[2] for tensor in image_tensors)
    padded_tensors: List[torch.Tensor] = []
    for tensor in image_tensors:
        _, h, w = tensor.shape
        pad_h = max_h - h
        pad_w = max_w - w
        if pad_h or pad_w:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h))
        padded_tensors.append(tensor)
    tensor_batch = torch.stack(padded_tensors, dim=0)
    nima_scores: List[Optional[float]] = [None] * len(valid_paths)
    if nima_metric is not None:
        with torch.no_grad():
            scores = nima_metric(tensor_batch.to(device))
            nima_scores = scores.detach().cpu().view(-1).tolist()

    piq_scores: List[Optional[float]] = [None] * len(valid_paths)
    if piq_metric is not None:
        with torch.no_grad():
            scores = piq_metric(tensor_batch.to(device))
            piq_scores = scores.detach().cpu().view(-1).tolist()

    for idx, path in enumerate(valid_paths):
        entry: Dict[str, Optional[float]] = {
            "image": path,
            "CLIP_LAION": float(clip_scores[idx]),
            "NIMA_pyiqa": float(nima_scores[idx]) if nima_scores[idx] is not None else None,
            "piq_CLIPIQA": float(piq_scores[idx]) if piq_scores[idx] is not None else None,
        }
        entry["OVERALL_RATING"] = _compute_overall_rating(entry)
        results.append(entry)
    return results


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


def _prepare_output_dirs(output_dir: str, dataset_name: str) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    selection_dir = os.path.join(output_dir, f"{dataset_name}_selected")
    rejected_dir = os.path.join(output_dir, f"{dataset_name}_rejected")
    for d in (selection_dir, rejected_dir):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    return selection_dir, rejected_dir


def _run_scoring_pass(
    image_paths: Sequence[str],
    folder: str,
    dataset_name: str,
    output_dir: Optional[str],
    table_suffix: str = "",
) -> List[Dict[str, Optional[float]]]:
    include_nima = pyiqa is not None and os.environ.get("BATCH_AESTHETIC_DISABLE_NIMA") != "1"
    include_piq = piq is not None

    approaches = ["CLIP_LAION"]
    if include_nima:
        approaches.append("NIMA_pyiqa")
    if include_piq:
        approaches.append("piq_CLIPIQA")
    approaches.append("OVERALL_RATING")

    selection_dir: Optional[str] = None
    rejected_dir: Optional[str] = None
    mapping_lines: Optional[List[str]] = None
    mapping_count = 0
    rejected_count = 0
    if output_dir:
        selection_dir, rejected_dir = _prepare_output_dirs(output_dir, dataset_name)
        mapping_lines = [
            "| Original | Copied | CLIP_LAION | NIMA_pyiqa | piq_CLIPIQA | OVERALL_RATING |",
            "|---|---|---|---|---|---|",
        ]

    total_images = len(image_paths)
    results: List[Dict[str, Optional[float]]] = []
    start_time = time.time()
    processed = 0

    ctx = mp.get_context("spawn")
    batches = list(_chunked(list(image_paths), WORKER_BATCH_SIZE))
    with ctx.Pool(processes=WORKER_PROCESSES) as pool:
        for batch_results in pool.imap_unordered(_process_batch, batches):
            for entry in batch_results:
                processed += 1
                overall_rating = entry.get("OVERALL_RATING")
                clip_score = entry.get("CLIP_LAION")
                nima_score = entry.get("NIMA_pyiqa")
                piq_score = entry.get("piq_CLIPIQA")
                passes_rating = (
                    isinstance(overall_rating, (int, float))
                    and overall_rating >= MIN_OVERALL_RATING
                )
                clip_ok = isinstance(clip_score, (int, float)) and clip_score >= CLIP_THRESHOLD
                nima_ok = isinstance(nima_score, (int, float)) and nima_score >= MIN_NIMA_SCORE
                piq_ok = isinstance(piq_score, (int, float)) and piq_score >= MIN_PIQ_SCORE
                if selection_dir and rejected_dir and mapping_lines:
                    should_select = bool(
                        clip_ok and nima_ok and piq_ok and passes_rating and _passes_copy_filters(entry)
                    )
                    dest_dir = selection_dir if should_select else rejected_dir
                    if dest_dir is selection_dir:
                        mapping_count += 1
                        name_prefix = f"out{mapping_count}_"
                    else:
                        rejected_count += 1
                        name_prefix = f"reject{rejected_count}_"
                    _, ext = os.path.splitext(os.path.basename(entry["image"]))
                    filename = (
                        f"{name_prefix}"
                        f"CLIP_LAION-{_fmt_score(entry['CLIP_LAION'])}_"
                        f"NIMA_pyiqa-{_fmt_score(entry['NIMA_pyiqa'])}_"
                        f"piq_CLIPIQA-{_fmt_score(entry['piq_CLIPIQA'])}_"
                        f"OVERALL_RATING-{_fmt_score(entry['OVERALL_RATING'])}{ext}"
                    )
                    dst = os.path.join(dest_dir, filename)
                    try:
                        shutil.copy2(entry["image"], dst)
                    except Exception:
                        print(f"Failed to copy {entry['image']} to {dst}")
                    if dest_dir is selection_dir:
                        mapping_lines.append(
                            "| {orig} | {copied} | {clip} | {nima} | {piq} | {overall} |".format(
                                orig=entry["image"],
                                copied=dst,
                                clip=_fmt_score(entry["CLIP_LAION"]),
                                nima=_fmt_score(entry["NIMA_pyiqa"]),
                                piq=_fmt_score(entry["piq_CLIPIQA"]),
                                overall=_fmt_score(entry["OVERALL_RATING"]),
                            )
                        )

                if not passes_rating or not clip_ok or not nima_ok or not piq_ok:
                    continue
                results.append(entry)

            elapsed = time.time() - start_time
            avg = elapsed / processed if processed else 0.0
            expected_total = avg * total_images if processed else 0.0
            percent = (processed / total_images) * 100 if total_images else 100.0
            progress_msg = (
                f"[{processed}/{total_images}] "
                f"{percent:5.1f}% elapsed {_format_duration(elapsed)} "
                f"/ {_format_duration(expected_total)}"
            )
            print(progress_msg)

    table = format_markdown(results, approaches)
    if output_dir:
        suffix = f"_{table_suffix}" if table_suffix else ""
        out_path = os.path.join(output_dir, f"{dataset_name}{suffix}_aesthetic_scores.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(table + "\n")
        print(f"Saved results to {out_path}")
        if mapping_lines and mapping_count > 0:
            mapping_path = os.path.join(
                output_dir, f"{dataset_name}_aesthetic_scores_mapping.md"
            )
            with open(mapping_path, "w", encoding="utf-8") as f:
                f.write("\n".join(mapping_lines) + "\n")
            print(f"Saved selected-image mapping to {mapping_path}")
    else:
        print(table)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute aesthetic scores for a folder of images.")
    parser.add_argument("folder", help="Folder containing images to score")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional directory where the Markdown table will be written (file name auto-generated)",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=DEFAULT_MAX_IMAGE_SIZE,
        help="Resize images so the long edge is at most this many pixels (default env BATCH_AESTHETIC_MAX_SIDE or 1024)",
    )
    args = parser.parse_args()

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Input folder not found: {folder}")

    global MAX_IMAGE_SIZE
    MAX_IMAGE_SIZE = max(0, args.max_image_size)

    image_paths = collect_images(folder)
    if not image_paths:
        print("No images found.")
        return

    output_dir = os.path.abspath(args.output) if args.output else None
    dataset_name = os.path.basename(os.path.normpath(folder))

    _run_scoring_pass(
        image_paths,
        folder,
        dataset_name,
        output_dir,
    )


if __name__ == "__main__":
    main()

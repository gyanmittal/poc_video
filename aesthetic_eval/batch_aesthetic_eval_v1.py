#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
from typing import Dict, List, Optional

# Keep the same OpenMP safeguards so this script mirrors the environment
# the main evaluator uses.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2

try:
    from brisque import BRISQUE  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    BRISQUE = None


_BRISQUE_MODEL: Optional[BRISQUE] = None  # type: ignore[misc]


def collect_images(folder: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    images: List[str] = []
    for root, _, files in os.walk(folder):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in exts:
                images.append(os.path.join(root, fname))
    images.sort()
    return images


def score_brisque(image_path: str) -> Optional[float]:
    global _BRISQUE_MODEL
    if BRISQUE is None:
        return None
    if _BRISQUE_MODEL is None:
        try:
            _BRISQUE_MODEL = BRISQUE()
        except Exception:
            _BRISQUE_MODEL = None
            return None
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        return float(_BRISQUE_MODEL.score(rgb))
    except Exception:
        return None


def format_markdown(results: List[Dict[str, Optional[float]]]) -> str:
    headers = ["Image", "BRISQUE"]
    lines = ["| Image | BRISQUE |", "|---|---|"]
    for entry in results:
        value = entry.get("BRISQUE")
        formatted = f"{value:.3f}" if isinstance(value, (int, float)) else "N/A"
        lines.append(f"| {entry['image']} | {formatted} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute only BRISQUE aesthetic scores for debugging purposes."
    )
    parser.add_argument("folder", help="Folder containing images to score")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional directory where the Markdown table will be saved",
    )
    args = parser.parse_args()

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Input folder not found: {folder}")

    image_paths = collect_images(folder)
    if not image_paths:
        print("No images found.")
        return

    results: List[Dict[str, Optional[float]]] = []
    for path in image_paths:
        entry: Dict[str, Optional[float]] = {"image": path}
        entry["BRISQUE"] = score_brisque(path)
        results.append(entry)

    table = format_markdown(results)

    if args.output:
        output_dir = os.path.abspath(args.output)
        os.makedirs(output_dir, exist_ok=True)
        dataset_name = os.path.basename(os.path.normpath(folder))
        out_path = os.path.join(output_dir, f"{dataset_name}_brisque_scores.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(table + "\n")
        print(f"Saved results to {out_path}")
    else:
        print(table)


if __name__ == "__main__":
    main()

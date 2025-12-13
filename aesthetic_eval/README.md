Batch Aesthetic Evaluators
==========================

This folder keeps every iteration of our batch image-aesthetic scorers so we can
choose the right trade-off between speed, dependencies, and post-processing for
any project. All variants expect the dependencies listed in
`requirements.txt`. Install them once with:

```bash
pip install -r poc_video/aesthetic_eval/requirements.txt
```

At a minimum each script accepts a folder of images and (optionally) an
`--output` directory where reports or copied files are written. Pass absolute
paths or paths relative to the repo root when launching them, e.g.:

```bash
python3 poc_video/aesthetic_eval/batch_aesthetic_eval_v5.py <image_folder> --output <output_dir>
```

Overview by Script
------------------

| Script | Scoring stack | Throughput model | Copy/filter behaviour | When to use |
| --- | --- | --- | --- | --- |
| `batch_aesthetic_eval.py` | CLIP+LAION aesthetic head, NIMA (pyiqa), PIQ CLIPIQA | Sequential, single process with a background worker just for NIMA | No automatic selection/rejection, emits a Markdown table only | Simple baselines, debugging metric outputs without extra files |
| `batch_aesthetic_eval_v1.py` | BRISQUE only (OpenCV + brisque/libsvm) | Sequential | No copying; prints or saves a Markdown table | Stand-alone BRISQUE scoring or as the helper launched by later versions |
| `batch_aesthetic_eval_v2.py` | Adds BRISQUE by delegating to `v1`, keeps CLIP/NIMA/PIQ | Sequential main loop + helper worker for NIMA + subprocess for BRISQUE | No file copying; optional BRISQUE via `BATCH_AESTHETIC_DISABLE_BRISQUE=1` | Full-metric reports when you still only need Markdown output |
| `batch_aesthetic_eval_v3.py` | Same metrics as `v2` | Sequential | Creates `<dataset>_selected` and `<dataset>_rejected` folders, renames copies with their scores, writes a Markdown mapping; thresholds hard-coded (CLIP ≥5, NIMA ≥5, PIQ ≥0.4, BRISQUE ≤40) | When you need curated folders plus the BRISQUE gate but can live with single-threaded speed |
| `batch_aesthetic_eval_v4.py` | CLIP/NIMA/PIQ + optional BRISQUE | Multiprocessing pool that batches images, configurable long-edge resizing (`--max-image-size`) and env vars (`BATCH_AESTHETIC_BATCH`, `BATCH_AESTHETIC_WORKERS`) | Keeps the selected/rejected output from `v3`, adds detailed progress logging, supports `--disable-brisque` | High-volume runs that still require BRISQUE and per-image copies |
| `batch_aesthetic_eval_v5.py` | CLIP/NIMA/PIQ plus derived `OVERALL_RATING = (CLIP + NIMA) * PIQ`; BRISQUE removed | Same multiprocessing pipeline as `v4` | Copies both accepted and rejected files every run (subfolders are recreated each time). Only entries with CLIP ≥5, NIMA ≥5, PIQ ≥0.8, and `OVERALL_RATING ≥ 9.0` appear in the Markdown/mapping. Filenames append the overall rating. | Final curation step where BRISQUE is unnecessary but a single scalar ranking (and aggressive filtering) is |

Key Differences
---------------

### `batch_aesthetic_eval.py`
* Minimal interface: prints a Markdown table (`dataset_aesthetic_scores.md` if you
  pass `--output`) and nothing else.
* NIMA runs in a helper subprocess for stability on macOS, but the rest of the
  logic is single-threaded.
* Good for quickly validating that CLIP/NIMA/PIQ are installed and functioning.

### `batch_aesthetic_eval_v1.py`
* Dedicated BRISQUE scorer used both standalone and as the subprocess consumer
  for later versions.
* Keeps dependencies isolated so libsvm’s OpenMP runtime does not conflict with
  PyTorch.

### `batch_aesthetic_eval_v2.py`
* Extends the original script with BRISQUE aggregation by launching `v1`.
* Still processes images one-by-one, so total runtime is governed by Python I/O.
* Honors `BATCH_AESTHETIC_DISABLE_NIMA` / `BATCH_AESTHETIC_DISABLE_BRISQUE` env
  toggles if you want to skip expensive metrics.

### `batch_aesthetic_eval_v3.py`
* Adds duration tracking and simple progress updates per image.
* Introduces selection/rejection directories with renamed files embedding their
  metric values plus a Markdown mapping table of copied assets.
* Uses fixed thresholds (CLIP, NIMA, PIQ, and BRISQUE) to decide which files get
  promoted.

### `batch_aesthetic_eval_v4.py`
* Rewrites scoring to use multiprocessing pools that ingest batches of images,
  drastically reducing runtime on multi-core machines.
* Normalises images to a configurable long-edge (`--max-image-size`) to keep
  memory usage predictable.
* Adds `--disable-brisque`, continues to produce the same selection/rejection
  folders and mapping as `v3`, and logs ETA-style progress messages.

### `batch_aesthetic_eval_v5.py`
* Cloned from `v4` but removes BRISQUE entirely and instead computes an
  `OVERALL_RATING` scalar `(CLIP_LAION + NIMA_pyiqa) * piq_CLIPIQA`.
* Filters aggressively: only images satisfying CLIP ≥5, NIMA ≥5, PIQ ≥0.8, and
  `OVERALL_RATING ≥ 9` are written to the Markdown summary or marked as
  “selected”. Everything else is copied into the rejected folder with scores in
  the filename for later inspection.
* The output subfolders specific to the dataset are fully rebuilt on each run so
  no stale assets are left behind.

Reference Material
------------------

* `requirements.txt` – Python dependencies for every variant.
* `sample_input/`, `sample_output/` – Small fixtures that show the expected
  folder structure and Markdown output layout.

Pick the version that matches your needs (BRISQUE vs. derived scoring, serial vs.
parallel, table-only vs. dataset curation) and run it from the repo root as
shown above.

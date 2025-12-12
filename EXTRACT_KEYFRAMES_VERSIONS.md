# Extract Keyframe Pipeline Versions

This document tracks the seven extractor scripts that live in `poc_video/`. Each release was created to try a different balance of accuracy vs. throughput. The notes below explain what each script is optimized for, list the notable filtering behavior, and summarize the CLI/parameter differences so you can pick the right one quickly.

## Example Command (v10)

```bash
python3 poc_video/extract_keyframes_v10.py "${INPUT_PATH}" \
  --db "${DB_PATH}" \
  --output "${OUTPUT_DIR}" \
  --stride-sec 3 \
  --min-tech-score 60.0 \
  --min-aesthetic 5.0 \
  --dedup-threshold 0.2 \
  --final-dedup-threshold 0.2 \
  --resize-long-edge 360 \
  --adaptive-stride \
  --adaptive-stride-threshold 3.0 \
  --progress-only \
  --require-face \
  --skip-overlay-cards \
  --chunk-duration-secs 300
```

Set `INPUT_PATH`, `DB_PATH`, and `OUTPUT_DIR` to the source video (file or directory), DuckDB database file, and destination directory respectively before running.

## Version Highlights

### `extract_keyframes.py` (original)
- Processes an entire corpus (file or directory), clustering embeddings to ensure broad scene coverage.
- Rich filtering stack: scene/person/face gating, text overlay rejection, indoor/outdoor heuristics, optional two-pass dedup.
- Heavyweight configuration via `ProcessingConfig`; dozens of CLI switches (distance thresholds, frame skip, scenic heuristics, subtitle tolerance, etc.).
- Best when you need fine-grained control and can tolerate slower, multi-pass processing.

### `extract_keyframes_v2.py`
- First slimmed-down rewrite: single video input, sequential processing, no face/composition analysis.
- Computes technical metrics + aesthetic score for every frame, then deduplicates once at the end.
- CLI exposes only the minimal knobs (`--db`, `--output`, stride, max seconds/frames, `--min-aesthetic`, `--dedup-threshold`).
- Useful for quick experiments where InsightFace and composition aren’t required.

### `extract_keyframes_v3.py`
- Adds InsightFace-based size/aspect checks, composition scoring, and “best frame per stride window” selection to reduce inference cost.
- Still processes every decoded frame but only stores embeddings for the top frame in each `stride-sec` bucket.
- CLI matches v2 plus `--progress-only` and `--min-tech-score` (configurable post-filter introduced recently).
- Good when face framing matters and you want deterministic per-second sampling.

### `extract_keyframes_v4.py`
- Refines the window-buffering logic (explicit `flush_window_best`) and keeps per-frame reasoning similar to v3.
- No batching or concurrency yet; interface identical to v3 (minus `--min-tech-score` until added manually).
- Works well as a stable baseline when you want v3 logic without later batching tweaks.

### `extract_keyframes_v5.py`
- Introduces CLIP aesthetic batching per window to avoid scoring every frame individually.
- Tracks replacement statistics when the fallback pass upgrades the chosen frame.
- CLI remains the lean set (same as v4) even though the internals are more complex.
- Best for speeding up aesthetic scoring when GPU memory is tight.

### `extract_keyframes_v6.py`
- Changes the sampling strategy: seeks ahead by `stride-sec` rather than reading every frame, which greatly reduces total frames evaluated on long clips.
- Writes candidate JPEGs to disk before embedding to conserve RAM; otherwise keeps the v5 filtering stack.
- CLI still mirrors v4/v5, so “stride” means “actual frame skip” here unlike v2–v5.
- Choose this when throughput is more important than within-second scoring fidelity.

### `extract_keyframes_v7.py`
- End-to-end overhaul geared toward multi-video workflows: accepts directories, runs preprocessing across worker threads, adds adaptive stride (skip low-motion windows), black-frame detection, skin gating before face analysis, and configurable face checks.
- Restores full-frame iteration (needed for adaptive stride) but parallelizes filtering; only writes embeddings for frames that pass the stricter gating.
- CLI extends the base options with `--min-tech-score`, `--disable-face`, `--adaptive-stride`, `--face-skin-threshold`, black-frame thresholds, and `--workers`.
- Ideal when you need the richest set of heuristics plus concurrency.

### `extract_keyframes_v8.py`
- Builds on v7 with chunked processing (`--chunk-duration-secs`) so long videos are processed in smaller bursts; each chunk reuses the same DB/output paths but reports per-chunk timing plus grand totals.
- Adds on-disk candidate caching, a second “final dedup” pass, and a “finally_removed” folder for relocated duplicates, along with `--final-dedup-threshold` and `--skip-to-final-stage`.
- Introduces optional smile preference, `--resize-long-edge` downscaling, and overlay-card suppression flags (`--skip-overlay-cards` plus the hue/saturation knobs).
- Provides `--require-face` and `--disable-aesthetic` so you can enforce people-only results or skip CLIP entirely for speed.

### `extract_keyframes_v9.py`
- Keeps all v8 functionality but adds adaptive stride as a first-class CLI flag, smarter logging, and cleaner separation between candidate filtering and the final dedup stage.
- Smile-aware dedup is now the default behavior, and the chunk pipeline exposes `--adaptive-stride-threshold`, enabling motion-sensitive frame skipping even during chunked runs.
- Adds `--resize-long-edge` scaling defaults (often 90 px for fast experimentation) and refined DB housekeeping so reruns overwrite older chunks safely.

### `extract_keyframes_v10.py`
- Extends v9 with aggressive face-quality enforcement (sharpness, tilt, minimum face area, vertical center), canonicalized DB keys so reruns truly overwrite prior metadata, and a secondary Haar detector plus pedestrian heuristic for “faceless scenic” fallback logic.
- Adds scenic gating flags (`--allow-faceless-scenic-only`, `--faceless-max-skin-ratio`, `--faceless-motion-threshold`, `--faceless-max-confidence`, `--faceless-max-person-ratio`) and allows canonical frame moves into the root output folder after each multi-video run.
- Provides new CLI switches for chunk-level image resizing, smile dedup toggles, final dedup thresholds, and strict overlay rejection. This is the production-grade workflow currently maintained.

## Comparison Table

| Version | Input Mode | Frame Sampling | Face/Composition Checks | Aesthetic Strategy | Notable CLI Flags / Capabilities |
|---------|------------|----------------|-------------------------|--------------------|-------------------------------|
| `extract_keyframes.py` | File or recursive directory, corpus aware | Full scan (configurable `--frame-skip`) | Multi-stage (faces, persons, scene, text) with indoor/outdoor logic | Always, weighting via `--quality-gamma` | `--distance-threshold`, `--frame-skip`, `--face-filter`, `--two-pass`, `--text-filter`, etc. |
| `v2` | Single file | Full scan, sequential | None | Per-frame, dedup at end | Core flags only (`--stride-sec`, `--min-aesthetic`, `--dedup-threshold`) |
| `v3` | Single file | Full scan, keep best per stride bucket | InsightFace + composition metrics | Per-frame, best frame per bucket stored | Core flags + `--progress-only`, `--min-tech-score` |
| `v4` | Single file | Full scan, window buffering | Same as v3 | Per-frame | Same as v3 (minus `--min-tech-score` unless patched) |
| `v5` | Single file | Full scan, window buffering | Same as v4/v3 | Batched CLIP per window | Same as v4 |
| `v6` | Single file | Stride-based seeking (skips intermediate frames) | Same as v5 | Per-frame on sampled windows | Same as v4/v5 |
| `v7` | File or directory | Full scan with optional adaptive stride + threading | InsightFace optional, skin gating, black-frame filters, composition | Per-frame, threaded | Core flags + `--min-tech-score`, `--disable-face`, `--adaptive-stride`, `--face-skin-threshold`, `--black-frame-*`, `--workers` |
| `v8` | File or directory | Full scan per chunk (configurable chunk duration) | All v7 filters + smile preference + overlay suppression | Per-frame, chunk totals logged | Adds `--chunk-duration-secs`, `--final-dedup-threshold`, `--skip-to-final-stage`, `--require-face`, `--skip-overlay-cards`, overlay tuning knobs |
| `v9` | File or directory | Same as v8 plus adaptive stride refinements | Same as v8 | Per-frame, chunked | Extends v8 with `--adaptive-stride` defaults, `--resize-long-edge`, `--prefer-smiles` toggle |
| `v10` | File or directory | Chunked full scan with optional adaptive stride + scenic gating | InsightFace + secondary Haar + strict face quality + faceless heuristics | Per-frame, chunked, scenic fallback | Adds face-quality knobs (`--face-sharpness-threshold`, `--max-face-tilt`, `--min-face-area-ratio`, `--min-face-y-center`), scenic flags (`--allow-faceless-scenic-only`, etc.), canonical DB overwrites, and root-level frame consolidation |

## Parameter Comparison

“Default” denotes the value wired into the script. “—” means the flag is not exposed in that version (some older scripts use a different knob, noted inline).

| Parameter | `extract_keyframes.py` | `v2` | `v3` | `v4` | `v5` | `v6` | `v7` | `v8` | `v9` | `v10` |
|-----------|-----------------------|------|------|------|------|------|------|------|------|-------|
| Input argument | `--input/--corpus` (required) | positional `video` | positional `video` | positional `video` | positional `video` | positional `video` | file/dir | file/dir | file/dir | file/dir |
| `--db` | required | default `frames.duckdb` | same | same | same | same | same | same | same | same |
| `--output` | required | required | required | required | required | required | required | required | required | required |
| `--stride-sec` | — (`--frame-skip` instead) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 (seek) | 1.0 | 1.0 | 1.0 | 1.0 |
| `--frame-skip` | default 1 | — | — | — | — | — | — | — | — | — |
| `--chunk-duration-secs` | — | — | — | — | — | — | — | default 300.0 | default 300.0 | default 300.0 |
| `--max-frames` / `--max-seconds` | default `None` | `None` | `None` | `None` | `None` | `None` | `None` | `None` | `None` | `None` |
| `--min-aesthetic` | default 25.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `--min-tech-score` | via `--min-sharpness` | — | 70.0 | — | — | — | 70.0 | 70.0 | 70.0 | 70.0 |
| `--dedup-threshold` | 0.07 | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 |
| `--final-dedup-threshold` | — | — | — | — | — | — | — | inherits `--dedup` by default | same | same (plus scenic dedup logging) |
| `--progress-only` | False | — | False | False | False | False | False | False | False | False |
| `--disable-face` | use `--face-filter` | — | — | — | — | — | flag | flag | flag | flag |
| `--require-face` | — | — | — | — | — | — | optional | optional | optional | optional (with faceless heuristics) |
| `--adaptive-stride` | — | — | — | — | — | — | flag | flag | flag (tuned) | flag (with scenic motion feed) |
| `--face-skin-threshold` | — | — | — | — | — | — | 0.05 | 0.05 | 0.05 | 0.05 (also reused for scenic gating) |
| `--face-sharpness-threshold` | — | — | — | — | — | — | — | optional | optional | default tightened (170+) |
| `--max-face-tilt` / `--min-face-area-ratio` / `--min-face-y-center` | — | — | — | — | — | — | — | — | — | available |
| `--skip-overlay-cards` + hue/sat knobs | — | — | — | — | — | — | — | available | available | available |
| `--resize-long-edge` | — | — | — | — | — | — | — | available | available | available |
| `--skip-to-final-stage` / `--prefer-smiles` | — | — | — | — | — | — | — | available | available | available |
| `--allow-faceless-scenes` | — | — | — | — | — | — | — | available | available | optional scenic gating (`--allow-faceless-scenic-only`) |
| `--faceless-*` heuristics | — | — | — | — | — | — | — | — | — | available (`--faceless-max-skin-ratio`, `--faceless-motion-threshold`, `--faceless-max-confidence`, `--faceless-max-person-ratio`) |
| `--black-frame-*` | — | — | — | — | — | — | available | available | available | available |
| `--workers` | — | — | — | — | — | — | default 4 | default 4 | default 4 | default 4 |

> **Tip:** Versions ≥ v8 share the same CLI parser, so scripts can be swapped without changing automation. Older versions require custom wrappers if you need the extended flags.

## Keeping Parameters Consistent

- The scripts share common core arguments (`video/input`, `--db`, `--output`, `--stride-sec`, `--max-frames`, `--max-seconds`, `--min-aesthetic`, `--dedup-threshold`, `--progress-only`, `--min-tech-score`). Consider factoring these into a shared helper module so new versions automatically support the same CLI.
- Newer options (e.g., adaptive stride, worker pools, face disabling) only exist in v7 today; using keyword-only parameters in `process_video` lets older versions accept those arguments harmlessly, which makes swapping scripts easier.
- Documenting these differences (this file) helps decide whether to favor accuracy (v3/v4/v5) or throughput (v6) or extensive filtering/concurrency (v7).

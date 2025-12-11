# Extract Keyframe Pipeline Versions

This document tracks the seven extractor scripts that live in `poc_video/`. Each release was created to try a different balance of accuracy vs. throughput. The notes below explain what each script is optimized for, list the notable filtering behavior, and summarize the CLI/parameter differences so you can pick the right one quickly.

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

## Comparison Table

| Version | Input Mode | Frame Sampling | Face/Composition Checks | Aesthetic Strategy | Notable CLI Flags |
|---------|------------|----------------|-------------------------|--------------------|-------------------|
| `extract_keyframes.py` | File or recursive directory, corpus aware | Full scan (configurable frame_skip) | Multiple (faces, persons, scene, text) with indoor/outdoor logic | Always, weight controlled via `--quality-gamma` | `--distance-threshold`, `--frame-skip`, `--face-filter`, `--two-pass`, `--text-filter`, etc. |
| `extract_keyframes_v2.py` | Single file | Full scan, sequential | None | Per-frame scoring, dedup at end | Core flags only (`--stride-sec`, `--min-aesthetic`, `--dedup-threshold`) |
| `extract_keyframes_v3.py` | Single file | Full scan, keep best per stride bucket | InsightFace gating + composition metrics | Per-frame, but only best frame per bucket stored | Core flags + `--progress-only`, `--min-tech-score` |
| `extract_keyframes_v4.py` | Single file | Full scan, window buffering | Same as v3 | Per-frame | Same as v3 (minus `--min-tech-score` unless backported) |
| `extract_keyframes_v5.py` | Single file | Full scan, window buffering | Same as v4/v3 | Batched CLIP scoring per window | Same as v4 |
| `extract_keyframes_v6.py` | Single file | Stride-based seeking (skips intermediate frames) | Same as v5 | Per-frame on sampled windows | Same as v4/v5 |
| `extract_keyframes_v7.py` | File or directory | Full scan with optional adaptive stride + threading | Optional InsightFace, skin gating, black-frame filters, composition | Per-frame, parallel preprocessing | Core flags + `--min-tech-score`, `--disable-face`, `--adaptive-stride`, `--face-skin-threshold`, `--black-frame-*`, `--workers` |

## Parameter Comparison

“Default” denotes the value wired into the script. “—” means the flag is not exposed in that version (some older scripts use a different knob, noted inline).

| Parameter | `extract_keyframes.py` | `v2` | `v3` | `v4` | `v5` | `v6` | `v7` |
|-----------|-----------------------|------|------|------|------|------|------|
| Input argument | `--input/--corpus` (required) | positional `video` (required) | positional `video` | positional `video` | positional `video` | positional `video` | positional `video` or directory |
| `--db` | required | default `frames.duckdb` | default `frames.duckdb` | default `frames.duckdb` | default `frames.duckdb` | default `frames.duckdb` | default `frames.duckdb` |
| `--output` | required | required | required | required | required | required | required |
| `--stride-sec` | — (uses `--frame-skip`, default `1`) | default `1.0` | default `1.0` | default `1.0` | default `1.0` | default `1.0` (seek-based) | default `1.0` |
| `--frame-skip` | default `1` | — | — | — | — | — | — |
| `--distance-threshold` | default `0.25` | — | — | — | — | — | — |
| `--max-frames` | default `None` | default `None` | default `None` | default `None` | default `None` | default `None` | default `None` |
| `--max-seconds` | default `None` | default `None` | default `None` | default `None` | default `None` | default `None` | default `None` |
| `--min-aesthetic` | default `25.0` | default `0.0` | default `0.0` | default `0.0` | default `0.0` | default `0.0` | default `0.0` |
| `--min-tech-score` | — (use `--min-sharpness`, etc.) | — | default `70.0` | — | — | — | default `70.0` |
| `--dedup-threshold` | default `0.07` | default `0.15` | default `0.15` | default `0.15` | default `0.15` | default `0.15` | default `0.15` |
| `--progress-only` | default `False` | — | default `False` | default `False` | default `False` | default `False` | default `False` |
| `--disable-face` | — (see `--face-filter`/`--person-filter`) | — | — | — | — | — | flag (default face checks enabled) |
| `--adaptive-stride` | — | — | — | — | — | — | flag (default disabled) |
| `--face-skin-threshold` | — | — | — | — | — | — | default `0.05` |
| `--black-frame-threshold` | — | — | — | — | — | — | default `0.05` |
| `--black-frame-mean` | — | — | — | — | — | — | default `0.05` |
| `--workers` | — | — | — | — | — | — | default `4` |

## Keeping Parameters Consistent

- The scripts share common core arguments (`video/input`, `--db`, `--output`, `--stride-sec`, `--max-frames`, `--max-seconds`, `--min-aesthetic`, `--dedup-threshold`, `--progress-only`, `--min-tech-score`). Consider factoring these into a shared helper module so new versions automatically support the same CLI.
- Newer options (e.g., adaptive stride, worker pools, face disabling) only exist in v7 today; using keyword-only parameters in `process_video` lets older versions accept those arguments harmlessly, which makes swapping scripts easier.
- Documenting these differences (this file) helps decide whether to favor accuracy (v3/v4/v5) or throughput (v6) or extensive filtering/concurrency (v7).

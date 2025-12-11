# Cluster Face Pipelines

The `cluster_faces*.py` scripts collect face crops from videos, embed them, cluster similar people, and optionally produce annotated clips. Each revision experiments with new heuristics or clip styles. Use this guide to pick the right clustering tool quickly.

## Version Highlights

### `cluster_faces.py`
- Baseline pipeline: enumerates files under `--corpus`, samples frames at `--sample-fps`, detects faces via Haar or MTCNN, embeds them with timm/FaceNet, and clusters greedily using a `0.65` similarity threshold.
- Optional clip export trims `clip_pre/post` seconds around each cluster while preserving audio via FFmpeg; no dedup cache or tracking optimizations.

### `cluster_faces_v2.py`
- Adds performance controls: frame-to-frame tracking (skip detections for `--detect-every-k`), adaptive sampling, FP16 embedding, and batched inference.
- Introduces DuckDB-backed deduplication (`--db-enable`, hash/embedding thresholds) so repeated faces across videos are cached and skipped.
- Clip padding defaults drop to zero and `clip_min_dur` ensures segments are meaningful.

### `cluster_faces_v3.py`
- Builds on v2 by adding annotated clip export: bounding boxes/bodies are drawn with configurable color, thickness, and body expansion. Clips can optionally highlight the whole person.
- Same dedup/tracking stack as v2; still invokes FFmpeg for clips.

### `cluster_faces_v4.py`
- Wrapper around v3 that swaps in a custom annotator to blur the background and keep the subject sharp, then forwards to v3’s CLI.
- Automatically rewrites `--db-path` to versioned files (e.g., `face_cache_v4.duckdb`) so caches don’t conflict with other runs.

### `cluster_faces_v5.py`
- Another v3 wrapper that smooths bounding boxes, suppresses bystanders via fresh detections, lingers on the subject briefly, and muxes the annotated video with filtered audio in FFmpeg.
- Like v4, the CLI comes from v3 but DB paths are versioned (`face_cache_v5.duckdb`).

### `cluster_faces_v6.py`
- Full rewrite: inlines the config/dataclass, handles dedup, DuckDB, and imagehash caching inside the script, and keeps annotations/audio hooks exposed.
- Maintains the v3 CLI surface (including annotation flags) but tightens clustering defaults (`similarity_threshold=0.30`) and adds better logging/progress support.

### `cluster_faces_v7.py`
- Wraps v6 to upgrade the clip mux stage: the override annotator applies a voice-band filter and multiple FFmpeg attempts to keep intelligible speech when muxing annotated video back with source audio.
- CLI is identical to v6 since it simply overrides the annotator before calling `v6.main()`.

### `cluster_faces_v8.py`
- Extends v6 with optional speaker diarization (pyannote) to isolate the subject’s audio, plus gating based on detected presence intervals; supports Hugging Face tokens via CLI.
- Adds wrapper-only flags (`--diarize`, `--hf-token`, `--no-bandpass`, `--min-spk-dur`, `--presence-pad`) and otherwise reuses v6’s argument set.

## Comparison Table

| Version | Base Pipeline | Tracking & Dedup | Clip & Annotation Style | Audio Handling | Notable Flags/Changes |
|---------|---------------|------------------|-------------------------|----------------|-----------------------|
| `cluster_faces.py` | Simple Haar/MTCNN + timm/facenet embedding, greedy clustering (`thr=0.65`) | None (per-run only), no DB cache | Optional clips with fixed `0.75s` pre/post padding; no overlays | Direct FFmpeg trims with optional re-encode | `--detector`, `--sample-fps`, `--clip-pre/post`, `--similarity-threshold` |
| `cluster_faces_v2.py` | Same detectors/embedders but with FP16 + batch embedding | Tracking skips (`--detect-every-k`, `--tracker`), DuckDB cache, hash/embedding dedup | Clips default to zero padding + `clip_min_dur`; still plain video | Source audio copied as-is | Adds `--max-video-secs`, `--fp16`, `--embed-batch-size`, `--db-*`, `--adaptive-sampling` |
| `cluster_faces_v3.py` | v2 core | Same as v2 | Annotated clips (face/body box, optional expansion) | Same audio passthrough | Adds `--annotate-clips`, `--annotate-color`, `--body-expand-*` |
| `cluster_faces_v4.py` | v3 core via import | Same as v3 | Custom annotator blurs background, spotlights subject | Same passthrough | No new flags (inherits v3) but auto-versioned DB path |
| `cluster_faces_v5.py` | v3 core via import | Same as v3 | Smooth, margin-expanded boxes; suppresses other faces; lingers after detection | Annotated video muxed with original audio through FFmpeg | Inherits v3 flags; wrapper rewrites DB path to `_v5` |
| `cluster_faces_v6.py` | Standalone refactor with integrated DuckDB cache/imagehash | Built-in dedup, progress logging, same trackers as v2/v3 | Same annotation hooks as v3 (default off) | Plain passthrough until overridden | Same CLI surface as v3 but defaults (e.g., `similarity=0.30`) baked in |
| `cluster_faces_v7.py` | v6 core via import | Same as v6 | Same overlays as v6 (depends on `--annotate-clips`) | Voice-band filtering and multi-pass mux | No new flags; wrapper just overrides annotator |
| `cluster_faces_v8.py` | v6 core via import | Same as v6 | Same overlays as v6 | Optional diarization + gating, HF token support, band-pass toggle | Adds wrapper flags `--diarize`, `--hf-token`, `--no-bandpass`, `--min-spk-dur`, `--presence-pad` |

## Parameter Comparison

“Default” is the built-in value; “—” indicates the option does not exist in that version. v4/v5 forward all v3 flags, while v7 inherits v6 flags (only v8 adds extra wrapper options).

| Parameter | `cluster_faces.py` | `v2` | `v3` | `v4` | `v5` | `v6` | `v7` | `v8` |
|-----------|--------------------|------|------|------|------|------|------|------|
| Input flag | Required `--corpus`/`--output` | Same | Same | Same (via v3) | Same (via v3) | Same (via v6) | Same (via v6) | Same (via v6) |
| `--detector` | choices `haar/mtcnn`, default `haar` | Same | Same | Same | Same | Same | Same | Same |
| `--sample-fps` | Default `2.0` | Default `2.0` | Default `2.0` | Default `2.0` | Default `2.0` | Default `2.0` | Default `2.0` | Default `2.0` |
| `--max-video-secs` | — | Default `0.0` (disabled unless set) | Default `0.0` | Same as v3 | Same as v3 | Default `0.0` | Default `0.0` | Default `0.0` |
| `--similarity-threshold` | Default `0.65` | Default `0.30` | Default `0.30` | Same as v3 | Same as v3 | Default `0.30` | Default `0.30` | Default `0.30` |
| `--min-cluster-size` | Default `3` | Default `3` | Default `3` | Same | Same | Default `3` | Default `3` | Default `3` |
| Clip padding (`--clip-pre/post-sec`) | Default `0.75` | Default `0.0` | Default `0.0` | Same | Same | Default `0.0` | Default `0.0` | Default `0.0` |
| `--clip-min-dur` | — | Default `0.5s` | Default `0.5s` | Same | Same | Default `0.5s` | Default `0.5s` | Default `0.5s` |
| `--fp16` / `--embed-batch-size` | — | Available (`False`, `16`) | Available | Same | Same | Available | Available | Available |
| `--detect-every-k` | — | Default `5` | Default `5` | Same | Same | Default `5` | Default `5` | Default `5` |
| `--tracker` / `--max-tracks` | — | Default `KCF`, `8` | Same | Same | Same | Same | Same | Same |
| `--adaptive-sampling` (`--adapt-boost-*`) | — | Available (defaults `False`, `4.0fps/2.0s`) | Same | Same | Same | Same | Same | Same |
| `--db-enable` / `--db-path` | — | Available (default disabled / `poc_video/face_cache.duckdb`) | Available (same defaults) | Same but wrapper rewrites to `_v4` by default | Same but `_v5` DB default | Available (`poc_video/face_cache.duckdb`) | Available | Available |
| Dedup controls (`--dedup-method`, hash/emb thresholds) | — | Available (`both`, `6`, `0.995`, `5s`) | Available | Same | Same | Available | Available | Available |
| Annotation flags (`--annotate-clips`, `--annotate-color`, `--body-expand-*`) | — | — | Available (default off, green, 1.8/3.0) | Same (wrapper relies on them) | Same | Available | Available | Available |
| Wrapper audio flags | — | — | — | — | — | — | — | `--diarize` (False), `--hf-token` (env fallback), `--no-bandpass` (default bandpass on), `--min-spk-dur` `0.25`, `--presence-pad` `0.15` |

When migrating between scripts, treat v4/v5 as “v3 plus a custom annotator” and v7/v8 as “v6 plus enhanced audio.” That lets you keep a consistent CLI surface while opting into the clip style that matches your project.

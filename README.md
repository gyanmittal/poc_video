# Keyframe Extraction POC (PyTorch + OpenCV + DuckDB)

Extract distinct, high‑quality frames from videos with minimal RAM usage. Frames are streamed, embedded with a CNN, grouped by cosine distance, and the sharpest frame per group is selected and saved.

## Getting Started

- Install requirements (installs CLIP, NIMA/PIQIQA, InsightFace, etc.)
  ```
  pip3 install -r poc_video/requirements.txt
  ```
- Install ffmpeg (macOS)
  ```
  brew install ffmpeg
  ```
- Quick run (v2, DB dedup, progress-only)
  ```
  python3 poc_video/cluster_faces_v2.py     --corpus poc_video/sample_data/videos     --output poc_video/face_clusters_out_v2     --detector mtcnn --embedder facenet     --cluster-scope per_video     --db-enable --db-path poc_video/face_cache.duckdb     --progress-only
  ```
- Annotated clips (v3)
  ```
  python3 poc_video/cluster_faces_v3.py     --corpus poc_video/sample_data/videos     --output poc_video/face_clusters_out_v3     --detector mtcnn --embedder facenet     --cluster-scope per_video     --db-enable --db-path poc_video/face_cache.duckdb     --progress-only --annotate-clips
  ```

### Extract Keyframes v11 (CLIP + NIMA + PIQ)

`extract_keyframes_v11.py` is the latest pipeline that evaluates every candidate frame with CLIP (LAION head), NIMA (via `pyiqa`), and PIQ CLIPIQA (`piq`). It keeps detailed logs (`extraction_summary.txt`) and saves all rejected frames under `poc_video/out/rejected/` for inspection.

Example command:
```bash
python3 poc_video/extract_keyframes_v11.py poc_video/sample_data/videos/ \
  --db poc_video/video_meta_v11.duckdb \
  --output poc_video/out \
  --stride-sec 3 \
  --min-tech-score 60 \
  --min-aesthetic 5 \
  --dedup-threshold 0.15 \
  --final-dedup-threshold 0.15 \
  --adaptive-stride \
  --adaptive-stride-threshold 3.0 \
  --require-face \
  --skip-overlay-cards \
  --chunk-duration-secs 300 \
  --progress-only
```
After the run:
- Selected frames live directly under `poc_video/out/` with filenames like `out1_t12.34_clip-value-6.1_...`.
- Rejected frames are written to `poc_video/out/rejected/` with their rejection reason.
- `poc_video/out/extraction_summary.txt` lists every processed frame, its scores, status, file path, and reason (if rejected).



## Device selection (cuda / mps / cpu)

All scripts support choosing the compute device via `--device`:

- Values: `cuda`, `mps`, or `cpu`. Default: auto (CUDA → MPS → CPU).
- MTCNN note (macOS/MPS): MTCNN runs on CPU when `--device mps` to avoid a known MPS pooling issue.
- Verify your environment:
  - CUDA: `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`
  - MPS (Apple Silicon): `python -c "import torch; print(torch.backends.mps.is_available())"`

Examples:
```
# Force CUDA
python3 poc_video/cluster_faces_v2.py --device cuda ...
# Force MPS
python3 poc_video/cluster_faces_v3.py --device mps --annotate-clips ...
# Force CPU
python3 poc_video/extract_person_clips.py --device cpu ...
```

## Quick Copy/Paste (macOS fresh env)

```bash
# 1) Create & activate venv
python3 -m venv .venv
source .venv/bin/activate

# 2) Install deps
pip3 install -r poc_video/requirements.txt

# 3) Install ffmpeg (macOS, via Homebrew)
brew install ffmpeg

# 4) Run v3 with annotated clips (recommended defaults)
python3 poc_video/cluster_faces_v3.py   --corpus poc_video/sample_data/videos   --output poc_video/face_clusters_out_v3   --detector mtcnn --embedder facenet   --cluster-scope per_video   --db-enable --db-path poc_video/face_cache.duckdb   --progress-only --annotate-clips
```

## Features
- Streaming frame processing (no full video/feature matrices in memory).
- Embeddings with `timm` backbones (default `resnet50`, configurable via `--backbone`).
- Cosine distance thresholding to form groups of similar frames.
- Post-process near-duplicate suppression across selected frames via `--dedup-threshold`.
- Quality score via variance of Laplacian (sharpness) + optional contrast term.
- Incremental persistence to DuckDB: videos, frames, embeddings (BLOB), frame_groups.
- Saves representative frames and CSV summary.
- Optional filters:
  - Exclude frames with no detectable faces (`--face-filter`).
  - Filter out blurry or motion‑blurred frames via Laplacian variance threshold (`--min-sharpness`).

## Requirements
- Python 3.9+
- Install minimal deps:
```
pip3 install -r poc_video/requirements.txt
```
Or directly (subset shown; see `requirements.txt` for the authoritative list):
```
pip3 install opencv-python torch torchvision numpy duckdb timm facenet-pytorch Pillow ImageHash \
    scikit-image pyiqa piq insightface onnxruntime ftfy regex \
    git+https://github.com/openai/CLIP.git
```

### System dependencies (macOS)
- ffmpeg (required for clip cutting/concat):
```
brew install ffmpeg
ffmpeg -version
```

## Project Python files overview


## Feature Matrix

| Script                      | Streaming | Detection | Embedding | Clustering | DB Dedup | Clips | Annotated Clips | Tracking | Progress-only |
|-----------------------------|:---------:|:---------:|:---------:|:----------:|:--------:|:-----:|:----------------:|:--------:|:-------------:|
| extract_keyframes.py        |    ✓      |    —      |    ✓      |     —      |    —     |  —    |        —         |    —     |       —       |
| extract_person_clips.py     |    ✓      |    ✓      |    ✓      |     —      |    —     |  ✓    |        —         |    —     |       —       |
| cluster_faces.py            |    ✓      |    ✓      |    ✓      |     ✓      |    —     |  ✓    |        —         |    —     |       —       |
| cluster_faces_v2.py         |    ✓      |    ✓      |    ✓      |     ✓      |    ✓     |  ✓    |        —         |    ✓     |       ✓       |
| cluster_faces_v3.py         |    ✓      |    ✓      |    ✓      |     ✓      |    ✓     |  ✓    |        ✓         |    ✓     |       ✓       |
| cluster_faces_v4.py         |    ✓      |    ✓      |    ✓      |     ✓      |    ✓     |  ✓    |        ✓         |    ✓     |       ✓       |
| cluster_faces_v5.py         |    ✓      |    ✓      |    ✓      |     ✓      |    ✓     |  ✓    |        ✓         |    ✓     |       ✓       |
| cluster_faces_v6.py         |    ✓      |    ✓      |    ✓      |     ✓      |    ✓     |  ✓    |        ✓         |    ✓     |       ✓       |
| cluster_faces_v7.py         |    ✓      |    ✓      |    ✓      |     ✓      |    ✓     |  ✓    |        ✓         |    ✓     |       ✓       |
| cluster_faces_v8.py         |    ✓      |    ✓      |    ✓      |     ✓      |    ✓     |  ✓    |        ✓         |    ✓     |       ✓       |
| prepare_person_clips_sample.py |  —     |    —      |    —      |     —      |    —     |  —    |        —         |    —     |       —       |

Notes:
- Streaming: processes frames without loading whole videos or matrices into RAM.
- DB Dedup: v2/v3 use DuckDB + pHash/embedding to skip near-duplicate face crops.
- Tracking: OpenCV KCF/CSRT/MOSSE to bridge frames between detections (v2/v3).
Below is a detailed overview of each script with core features and key technologies used.

- **extract_keyframes.py**
  - Features
    - Streams videos, extracts CNN embeddings per sampled frame, groups by cosine distance, and selects a sharp representative per group.
    - Persists to DuckDB: videos, frames, embeddings (BLOB), frame_groups for resumability.
    - Quality scoring via Laplacian variance; optional contrast term; CSV summary and per‑video outputs.
  - Technologies
    - PyTorch + timm backbones (e.g., resnet50) for embeddings.
    - OpenCV for video I/O and image ops.
    - DuckDB for structured local storage; NumPy for vector math.

- **extract_person_clips.py**
  - Features
    - Finds clips of a target person using one or more reference photos.
    - Face detection (Haar or MTCNN), face embeddings (timm or facenet), robust matching gates, and segment assembly.
    - Optional segment validation; tunable quality thresholds and consecutive‑hit logic.
  - Technologies
    - facenet-pytorch (InceptionResnetV1) or timm for embeddings.
    - OpenCV + MTCNN/Haar for detection; ffmpeg for cutting/concat.

- **cluster_faces.py**
  - Features
    - Extracts face crops from videos, embeds them, and performs greedy centroid clustering per video or globally.
    - Saves representative crop and optionally all member crops per cluster.
    - Builds clip segments around hit timestamps and concatenates them to combined.mp4.
  - Technologies
    - OpenCV for detection (Haar/MTCNN via facenet-pytorch), I/O.
    - PyTorch + timm/facenet embeddings; NumPy similarity; ffmpeg for segments.

- **cluster_faces_v2.py**
  - Features
    - Performance-focused upgrade of cluster_faces.py.
    - DB-backed dedup cache: skip saving near-duplicate face crops by perceptual hash (pHash) and/or embedding similarity within a time window.
    - Progress-only mode; batch embedding; optional tracking between detects; adaptive sampling; per-video timing.
    - Tuned defaults: similarity-threshold=0.30, clip-pre/post=0.0, clip-min-dur=0.5.
  - Technologies
    - DuckDB for persistent cache (face rows with img_path, pHash, embedding BLOB).
    - Pillow + ImageHash for pHash; NumPy cosine similarity; OpenCV trackers (KCF/CSRT/MOSSE).

- **cluster_faces_v3.py**
  - Features
    - Builds on v2 and adds annotated clips: draws a box around the person (expanded face→body) across each segment.
    - Tracks boxes across frames and re‑initializes from detections when available.
    - Inherits v2’s DB dedup, sampling, and clip controls.
  - Technologies
    - OpenCV VideoCapture/Writer and rectangle drawing; trackers for propagation.
    - Same embedding/dedup stack as v2 (timm/facenet, DuckDB, ImageHash, NumPy).

- **prepare_person_clips_sample.py**
  - Features
    - Creates a small sample dataset structure (videos/photos) to quickly try person‑clip extraction.
    - Helpful for demos and sanity checks without custom media.
  - Technologies
    - Standard Python + OpenCV file I/O utilities.


# Person Clips Extraction (extract_person_clips.py)

Extract relevant clips of a target person from a video corpus using one or more reference photos. Uses face detection (Haar/MTCNN), face embeddings (timm or facenet), robust matching gates, and clip assembly with ffmpeg.

## Requirements (additions)
- For MTCNN/facenet:
  - `pip install facenet-pytorch`
- ffmpeg must be available in PATH for best results.

## Quickstart (with sample data)
Use the helper to create a sample video and reference photos:
```
python3 poc_video/prepare_person_clips_sample.py --output poc_video/sample_data
```
Run clips extraction:
```
python3 poc_video/extract_person_clips.py \
  --photos poc_video/sample_data/photos \
  --corpus poc_video/sample_data/videos \
  --output poc_video/person_clips_out \
  --embedder facenet --detector mtcnn \
  --sample-fps 3.0 \
  --match-threshold 0.70 \
  --min-consecutive-hits 3 \
  --center-margin 0.02 \
  --face-min-area-ratio 0.04 \
  --face-min-sharpness 80 \
  --segment-min-accept-ratio 0.80 \
  --segment-validate-fps 6.0 \
  --ffmpeg-reencode \
  --debug
```

## Key Options
- `--photos`: file or directory of reference photos.
- `--corpus`: file or directory of videos (recursive).
- `--embedder`: `timm` or `facenet` (facenet recommended for identity).
- `--detector`: `haar` or `mtcnn`.
- `--match-threshold`: cosine similarity threshold. Higher = stricter.
- `--min-consecutive-hits`: require N consecutive matches before accepting a hit.
- `--face-min-area-ratio`, `--face-min-sharpness`: quality gates for face crops.
- `--segment-validate`: re-check segments and prune false positives.
- `--pre-roll`, `--post-roll`, `--join-gap`: control clip assembly.
- `--ffmpeg-reencode`: re-encode clips (more robust playback).
- Debugging: `--dump-matches`, `--dump-match-frames`, `--super-lenient`.

## Outputs
- `poc_video/person_clips_out/<video_name>/clip_###_start-end.mp4`
- Optional debugging crops/frames when enabled.
- CSV summary `clips_<run_id>.csv` at the output root.

## Tips
- Multiple photos: by default requires match to all (`--require-all-queries`).
- Too few clips? Lower `--match-threshold` or relax gates. Too many false positives? Increase quality gates and use segment validation.


# Face Clustering and Combined Clips (cluster_faces.py)

Extract prominent face crops from videos, cluster similar faces, and save a combined clip per face cluster.

## Quickstart
```
python3 poc_video/cluster_faces.py \
  --corpus poc_video/sample_data/videos \
  --output poc_video/face_clusters_out \
  --detector mtcnn \
  --embedder facenet \
  --cluster-scope per_video \
  --similarity-threshold 0.30 \
  --clip-pre-sec 0.0 --clip-post-sec 0.0 \
  --clip-merge-gap 0.5 \
  --debug
```

## Key Options
- `--cluster-scope`: `per_video` (default) or `global`.
- `--similarity-threshold`: cosine sim to cluster centroid; lower to merge more clusters.
- `--min-cluster-size`: ignore clusters smaller than this size.
- `--embedder`: `timm` or `facenet` (facenet recommended).
- `--detector`: `haar` or `mtcnn`.
- Clip generation (per face cluster):
  - `--no-clips`: disable clips.
  - `--clip-pre-sec`, `--clip-post-sec`: padding around each appearance.
  - `--clip-merge-gap`: merge nearby windows before cutting.
  - `--no-ffmpeg-reencode`: faster concat/cut but may be less robust.
- Cleanup:
  - default behavior cleans only the processed video’s outputs on rerun (`faces/` and `clusters_<video_name>/`).
  - disable with `--no-clean`.
- `--sample-fps`: sampling FPS for detection/embedding.
- `--face-min-area-ratio`, `--face-min-sharpness`: quality gates for faces.

## Outputs
- Per video (per_video scope):
  - `.../<video_name>/faces/` all extracted crops
  - `.../clusters_<video_name>/face1/`, `face2/`, ...
    - `representative.jpg`
    - `combined.mp4` (concatenated segments where this face appears)
    - member crops (unless `--representative-only`)
  - `clusters.csv` summary in the clusters folder

## Tuning cluster merge/split
- Merge more clusters: lower `--similarity-threshold` (e.g., 0.70 → 0.65 → 0.60).
- Reduce fragmentation: raise `--face-min-area-ratio` and `--face-min-sharpness` to drop tiny/blurred faces.
- Merge across videos: use `--cluster-scope global`.

## Troubleshooting
- ffmpeg needed for segment cutting/concat.
- MTCNN on Apple Silicon: the script runs MTCNN on CPU when `device=mps` to avoid known pooling errors.

---

# Face Clustering v2 (cluster_faces_v2.py)

Performance-optimized face clustering with DB-backed dedup, progress-only mode, and tuned defaults.

## Quickstart (recommended defaults)
```
python3 poc_video/cluster_faces_v2.py \
  --corpus poc_video/sample_data/videos \
  --output poc_video/face_clusters_out_v2 \
  --detector mtcnn \
  --embedder facenet \
  --cluster-scope per_video \
  --db-enable --db-path poc_video/face_cache.duckdb \
  --progress-only
```

For fast tuning on long videos, process only the first 5 minutes:
```
  --max-video-secs 300
```

## Defaults (changed vs v1)
- `--similarity-threshold`: 0.30
- `--clip-pre-sec`: 0.0
- `--clip-post-sec`: 0.0
- `--clip-merge-gap`: 0.5
- `--clip-min-dur`: 0.5 (ensures non-zero segments even with 0/0 pre/post)

## Progress and debug
- `--progress-only`: prints one updating line every 1000 frames, suppressing verbose logs.
- `--debug`: verbose logs for detection/track/segments.

## DB-backed dedup (optional)
- Enable with `--db-enable` and set path via `--db-path poc_video/face_cache.duckdb`.
- Dedup method: `--dedup-method both|hash|embedding` (default both).
- Thresholds:
  - `--dedup-hash-thresh 6` (pHash Hamming distance)
  - `--dedup-emb-thresh 0.995` (cosine similarity)
  - `--dedup-window-sec 5.0` (compare within ±5s per video)
- Images are kept on disk; DB stores `img_path`, embedding blob, and pHash.

## Outputs
- Same structure as v1 (per-video clusters with `face1/face2/...`, `representative.jpg`, `combined.mp4`).

---

# Face Clustering v3 (cluster_faces_v3.py) — Annotated Clips

v3 builds on v2 and can draw a box around the person (face+body) throughout each clip segment.

## Quickstart (annotated clips)
```
python3 poc_video/cluster_faces_v3.py \
  --corpus poc_video/sample_data/videos \
  --output poc_video/face_clusters_out_v3 \
  --detector mtcnn \
  --embedder facenet \
  --cluster-scope per_video \
  --db-enable --db-path poc_video/face_cache.duckdb \
  --progress-only \
  --annotate-clips
```

## Annotation controls
- `--annotate-clips`: enable drawing.
- `--annotate-thickness`: line thickness (default 3).
- `--annotate-color`: BGR color string, e.g. `"0,255,0"`.
- Body box expansion around face:
  - `--body-expand-w` (default 1.8)
  - `--body-expand-h` (default 3.0)

## How it works
- Uses face detections from the pipeline as anchors.
- Tracks between detections (KCF by default) to keep the box stable.
- Draws an expanded body box per frame across the entire segment; then concatenates segments per cluster to `combined.mp4`.
- Re-encode is recommended for robust concat (default behavior).

## Notes
- v3 shares all v2 options (sampling, DB dedup, progress/debug, clip controls).
- For faster experiments, use `--max-video-secs 300` and `--sample-fps`.
## Quickstart
Use the included sample video (if present): `poc_video/videos/onepiece_demo.mp4`
```
python3 poc_video/extract_keyframes.py \
  --input poc_video/videos/onepiece_demo.mp4 \
  --output poc_video/out \
  --db poc_video/video_meta.duckdb \
  --backbone resnet50 \
  --distance-threshold 0.35 \
  --dedup-threshold 0.10 \
  --frame-skip 1
```
Process an entire folder recursively:
```
python3 poc_video/extract_keyframes.py \
  --input poc_video/videos/ \
  --output poc_video/out \
  --db poc_video/video_meta.duckdb \
  --backbone resnet50 \
  --distance-threshold 0.35 \
  --dedup-threshold 0.10 \
  --frame-skip 1
```
Offline or behind strict firewall? Avoid weight downloads:
```
python3 poc_video/extract_keyframes.py \
  --input poc_video/videos/onepiece_demo.mp4 \
  --output poc_video/out \
  --db poc_video/video_meta.duckdb \
  --backbone resnet50 \
  --no-pretrained
```

## CLI Options (common)
- `--input`: path to video file or directory (recursive).
- `--output`: output directory root for per-video subfolders and `summary.csv`.
- `--db`: DuckDB file path (created if missing).
- `--backbone`: `timm` model name (e.g., `resnet50`, `tf_efficientnet_b0_ns`).
- `--distance-threshold`: cosine distance to start a new group (default 0.25). Larger → fewer, more distinct groups.
- `--dedup-threshold`: cosine distance threshold to treat two selected frames as duplicates (default 0.07).
- `--face-filter`: require at least one detected subject (face or person) per frame (defaults to enabled).
- `--min-sharpness`: discard frames with Laplacian variance below this threshold (default 50.0).
- `--video-face-filter`: skip an entire video if it contains no prominent faces (defaults to enabled).
- `--min-face-area`: minimum face bbox area ratio relative to frame area to consider a face prominent (default 0.03).
- `--min-face-frames`: require at least this many frames to have a prominent face for the video to be processed (default 1).
- `--frame-skip`: process every Nth frame (speed vs. recall).
- `--device`: `cuda`, `mps`, or `cpu` (auto by default).
- `--no-pretrained`: disable pretrained weights (offline mode).
- Other flags: `--resize-shorter`, `--model-input-size`, `--quality-alpha`, `--quality-beta`, `--jpeg-quality`, `--max-frames`.

## Outputs
- For each input video named `<video_name>`:
  - Images saved directly in `out/<video_name>/` (no `selected/` subfolder).
  - `out/<video_name>/summary.csv`: `frame_index,timestamp,quality,frame_id`.
- DuckDB at `--db` with tables: `videos`, `frames`, `embeddings`, `frame_groups`.
  - The per-video output folder is cleaned (deleted and recreated) before each run for that video.

## How it works
- **Embedding model & similarity**: `timm` backbone with `num_classes=0` and `global_pool='avg'` → D‑dim features (e.g., 2048 for resnet50). Preprocessing uses the model’s mean/std. L2‑normalize embeddings; similarity via cosine distance.
- **Quality scoring**: Variance of Laplacian (sharpness). Optional contrast via grayscale std. `score = alpha*lap_var + beta*gray_std`.
- **Grouping/selection**: Maintain an anchor (last group representative). A new group starts if distance(anchor, current) > threshold. Within a group keep the frame with highest quality.
- **Near-duplicate suppression**: After selection, compare representatives pairwise; if cosine distance < `--dedup-threshold`, keep the higher-quality one and drop the other.
- **Storage design**: Rows written per frame to DuckDB. Embeddings stored as float32 BLOBs. Schema is extendable (add columns/tables for new metadata or embedding types) without loading large matrices into RAM.

## Tips & Tuning
- Increase `--distance-threshold` (e.g., 0.3–0.4) to reduce number of groups.
- Increase `--frame-skip` (e.g., 2–5) to speed up and reduce I/O.
- Use `--no-pretrained` to run fully offline.
- For long/fast videos, consider higher threshold and frame skip.

## Filtering
- **Subject filter** (`--face-filter`): keeps frames that have a prominent subject (either a face or a person).
  - Faces: OpenCV Haar cascade (`haarcascade_frontalface_default.xml`).
  - Persons: TorchVision Faster R-CNN (COCO) if available; falls back to OpenCV HOG people detector.
  - Enabled by default; disable with `--no-face-filter` (on Python versions supporting boolean optional args).
  - If the TorchVision model weights cannot be downloaded, the HOG fallback is used automatically.
- **Blur filter** (`--min-sharpness`): frames with variance of Laplacian below the threshold are discarded before embedding.
  - Default: `50.0`. Raise it to be stricter (e.g., 80–120) if many blurry frames remain.
- **Video-level subject filter** (`--video-face-filter`): quick precheck before any writes; if across sampled frames no prominent subject (face area ≥ `--min-face-area` OR person area ≥ `--min-person-area`) appears in at least `--min-face-frames` frames, the video is skipped and no outputs are written.
  - Example to disable (process all videos regardless of subject): add `--no-video-face-filter`.
  - Example to tune prominence: `--min-face-area 0.02 --min-person-area 0.10 --min-face-frames 3`.

### Subject prominence tuning
- Face prominence: `--min-face-area` (default 0.03). Lower to be less strict (e.g., 0.01–0.02) if faces are small.
- Person prominence: `--min-person-area` (default 0.12). Lower to keep full-body subjects that occupy less of the frame (e.g., 0.08–0.10).

## Troubleshooting
- Pretrained weights download blocked: use `--no-pretrained`.
- Pip issues behind proxy/firewall:
  - Set proxy or CA: `HTTPS_PROXY`, `HTTP_PROXY`, `REQUESTS_CA_BUNDLE`.
  - Retry with `--retries 5 --timeout 60` or use a trusted PyPI mirror.

## Folder Structure
```
poc_video/
├─ extract_keyframes.py
├─ requirements.txt
├─ README.md
└─ videos/
   └─ onepiece_demo.mp4  (optional sample)
```

## Extending
- Add new embedding backbones or multi‑head embeddings; store type/shape in a new table or columns.
- Add more quality metrics (e.g., SSIM, exposure); store per‑frame metrics in `frames` or a new `frame_metrics` table.
- Swap grouping logic for clustering (e.g., incremental) while keeping the same storage interfaces.

## DuckDB CLI installation (macOS)
- Install the DuckDB command-line tool:
```
brew install duckdb
```
- Verify:
```
duckdb --version
```

If you don’t want to install the CLI, you can always query the DB via Python (shown below).

## Get row counts from the database
Assuming your DB is at `poc_video/video_meta.duckdb`.

### Using DuckDB CLI
```
duckdb poc_video/video_meta.duckdb -c "
SELECT 'videos' AS table, COUNT(*) AS rows FROM videos
UNION ALL SELECT 'frames', COUNT(*) FROM frames
UNION ALL SELECT 'embeddings', COUNT(*) FROM embeddings
UNION ALL SELECT 'frame_groups', COUNT(*) FROM frame_groups
UNION ALL SELECT 'selected_frames (is_representative=1)', COUNT(*) FROM frames WHERE is_representative = TRUE;
"
```

### Using Python (one-liner)
```
python3 -c "import duckdb; con=duckdb.connect('poc_video/video_meta.duckdb'); \
print(con.sql(\"SELECT 'videos' t, COUNT(*) c FROM videos \
UNION ALL SELECT 'frames', COUNT(*) FROM frames \
UNION ALL SELECT 'embeddings', COUNT(*) FROM embeddings \
UNION ALL SELECT 'frame_groups', COUNT(*) FROM frame_groups \
UNION ALL SELECT 'selected_frames (is_representative=1)', COUNT(*) FROM frames WHERE is_representative=TRUE\").fetchall())"
```

### Using Python (readable)
```
python3 - <<'PY'
import duckdb
con = duckdb.connect('poc_video/video_meta.duckdb')
for t in ['videos','frames','embeddings','frame_groups']:
    n = con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
    print(f'{t}: {n}')
n_sel = con.execute('SELECT COUNT(*) FROM frames WHERE is_representative = TRUE').fetchone()[0]
print(f'selected_frames: {n_sel}')
PY
```


## Examples (v2/v3 visuals)

- Create a quick GIF from a cluster combined clip (requires ffmpeg):
```
ffmpeg -i poc_video/face_clusters_out_v3/clusters_<video>/face1/combined.mp4   -vf "fps=12,scale=-1:480:flags=lanczos" -loop 0 face1_preview.gif
```
- Grab a single preview frame (PNG):
```
ffmpeg -i poc_video/face_clusters_out_v3/clusters_<video>/face1/combined.mp4   -frames:v 1 face1_preview.png
```
- Run v3 with annotation enabled (recommended defaults):
```
python3 poc_video/cluster_faces_v3.py   --corpus poc_video/sample_data/videos   --output poc_video/face_clusters_out_v3   --detector mtcnn --embedder facenet   --cluster-scope per_video   --db-enable --db-path poc_video/face_cache.duckdb   --progress-only --annotate-clips
```
- Run v2 (DB-dedup, progress-only, tuned defaults):
```
python3 poc_video/cluster_faces_v2.py   --corpus poc_video/sample_data/videos   --output poc_video/face_clusters_out_v2   --detector mtcnn --embedder facenet   --cluster-scope per_video   --db-enable --db-path poc_video/face_cache.duckdb   --progress-only
```

---

# Face Clustering v4 (cluster_faces_v4.py) — Person-only with blurred background

## Quickstart
```
python3 poc_video/cluster_faces_v4.py \
  --corpus poc_video/sample_data/videos \
  --output poc_video/face_clusters_out_v4 \
  --detector mtcnn --embedder facenet \
  --cluster-scope per_video \
  --db-enable --db-path poc_video/face_cache.duckdb \
  --progress-only --annotate-clips
```

## Notes
- Person region remains sharp; background is blurred. If absent, frame is black.
- DB path is versioned: if you pass `--db-path poc_video/face_cache.duckdb`, v4 writes to `poc_video/face_cache_v4.duckdb`.

---

# Face Clustering v5 (cluster_faces_v5.py) — Keep scene, grey out others, add audio

## Quickstart
```
python3 poc_video/cluster_faces_v5.py \
  --corpus poc_video/sample_data/videos \
  --output poc_video/face_clusters_out_v5 \
  --detector mtcnn --embedder facenet \
  --cluster-scope per_video \
  --db-enable --db-path poc_video/face_cache.duckdb \
  --progress-only --annotate-clips \
  --similarity-threshold 0.45 --min-cluster-size 4 \
  --face-min-sharpness 90 --face-min-area-ratio 0.04 \
  --tracker CSRT --detect-every-k 3 --sample-fps 3.0 \
  --annotate-thickness 3 --annotate-color 0,255,0 \
  --max-video-secs 2000
```

## Notes
- Keeps full scene, greys out other persons, draws a green box on the target.
- Muxes original audio into the clip.
- DB path is versioned: v5 appends `_v5` (defaults to `poc_video/face_cache_v5.duckdb`).
- To process all frames, omit `--sample-fps`.

---

# Face Clustering v6 (cluster_faces_v6.py) — v3 + audio

## Quickstart
```
python3 poc_video/cluster_faces_v6.py \
  --corpus poc_video/sample_data/videos \
  --output poc_video/face_clusters_out_v6 \
  --detector mtcnn --embedder facenet \
  --cluster-scope per_video \
  --db-enable --db-path poc_video/face_cache.duckdb \
  --progress-only --annotate-clips
```

## Notes
- Same annotations as v3, but audio is muxed into each clip.
- Uses unversioned DB path by default (`poc_video/face_cache.duckdb`).

---

# Face Clustering v7 (cluster_faces_v7.py) — Vocals-only audio (music reduced)

## Quickstart
```
python3 poc_video/cluster_faces_v7.py \
  --corpus poc_video/sample_data/videos \
  --output poc_video/face_clusters_out_v7 \
  --detector mtcnn --embedder facenet \
  --cluster-scope per_video \
  --db-enable --db-path poc_video/face_cache.duckdb \
  --progress-only --annotate-clips
```

## Notes
- Applies FFmpeg voice-focused filters (band-pass 200–3800 Hz, light denoise, limiter) to attenuate music and keep speech.
- No extra Python deps vs v6; requires FFmpeg installed.

---

# Face Clustering v8 (cluster_faces_v8.py) — Optional diarization for target-only voice

## Quickstart (presence-gated vocals)
```
python3 poc_video/cluster_faces_v8.py \
  --corpus poc_video/sample_data/videos \
  --output poc_video/face_clusters_out_v8 \
  --detector mtcnn --embedder facenet \
  --cluster-scope per_video \
  --db-enable --db-path poc_video/face_cache.duckdb \
  --progress-only --annotate-clips
```

## With diarization (target-only voice)
```
# Pass token explicitly
python3 poc_video/cluster_faces_v8.py --diarize --hf-token YOUR_HF_TOKEN \
  --corpus poc_video/sample_data/videos --output poc_video/face_clusters_out_v8 \
  --detector mtcnn --embedder facenet --cluster-scope per_video \
  --db-enable --db-path poc_video/face_cache.duckdb --progress-only --annotate-clips

# Or via env var
export HF_TOKEN=YOUR_HF_TOKEN
python3 poc_video/cluster_faces_v8.py --diarize \
  --corpus poc_video/sample_data/videos --output poc_video/face_clusters_out_v8 \
  --detector mtcnn --embedder facenet --cluster-scope per_video \
  --db-enable --db-path poc_video/face_cache.duckdb --progress-only --annotate-clips
```

## Notes
- If diarization is unavailable, v8 falls back to presence-gated vocals (band-pass + gating).
- Optional deps for diarization: `pip3 install -r poc_video/requirements-v8-optional.txt` (includes pyannote.audio).

---

# Versioned DB filenames
- v3/v6 use unversioned DB path by default: `poc_video/face_cache.duckdb`.
- v4 appends `_v4` if missing; default `poc_video/face_cache_v4.duckdb`.
- v5 appends `_v5` if missing; default `poc_video/face_cache_v5.duckdb`.

# Processing all frames
- To process every frame at native FPS, omit `--sample-fps`.

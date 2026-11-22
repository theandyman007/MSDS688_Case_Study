# MSDS688_Case_Study
A case study for Regis University Class MSDS688: Artificial Intelligence


## Frame Extraction using auto_extraction.py
`scripts/auto_extraction.py` pulls the sharpest frame from evenly spaced time
windows of each video. It is a wrapper around `extract_sharp_frames` in
`scripts/extract_frames.py`.

### Setup

1. Install Python 3 and the dependency:

        pip install opencv-python

2. Put your videos in `source/videos/`. The script creates this folder on its
first run if it doesn't exist. Supported formats: `.mp4`, `.mov`, `.avi`,
`.mkv`, `.m4v`.

The `source/` folder is in `.gitignore`, so videos and frames are not
committed to the repo.

### Usage

Run from the project root:

python scripts/auto_extraction.py                   # every video in source/videos

python scripts/auto_extraction.py --skip-existing   # resume after an interruption

python scripts/auto_extraction.py my_video.mp4      # a single video

python scripts/auto_extraction.py --n 100 --max-side 1280


| Option | Default | Description |
|---|---|---|
| `video` | all videos | Single video: a filename in `source/videos/` or a full path |
| `--n` | 200 | Number of windows, so the number of frames saved per video |
| `--max-side` | 1920 | Resize so the longest side is this many pixels (`0` = full resolution) |
| `--skip-existing` | off | Skip videos whose output folder already has `frames_log.csv` |

### Output

Each video gets its own folder, `source/frames/<video_name>/`, containing:

- `frame_0000.jpg`, `frame_0001.jpg`, ...: the sharpest frame from each window
- `frames_log.csv`: one row per saved frame (`file`, `window`, `source_frame`,
  `sharpness`)

### How it works

Each video is split into `--n` equal windows. Every frame is scored for
sharpness (variance of the Laplacian), and only the highest-scoring frame in
each window is saved. This avoids blurry frames and near-duplicates.

If a video can't be opened or reports 0 frames, the script prints `FAILED`,
continues with the remaining videos, and lists the failures at the end.




## Stage 2A: Photogrammetry Mesh using photogrammetry.py
`scripts/photogrammetry.py` turns the Stage 1 COLMAP points into a polygon mesh
using open3d.

### Setup

    pip install open3d

Run `scripts/run_sfm.py` first. It writes the `data/colmap/<scene>/` folder this
script reads.

### Usage

Run from the project root:

python scripts/photogrammetry.py data/colmap/<scene>                  # sparse points

python scripts/photogrammetry.py data/colmap/<scene> --source dense   # dense MVS (CUDA only)

python scripts/photogrammetry.py data/colmap/<scene> --depth 10 --view

| Option | Default | Description |
|---|---|---|
| `colmap_dir` | required | Stage 1 output folder |
| `--out` | `data/mesh/<scene>` | Output folder |
| `--source` | `sparse` | `sparse` uses `sparse_points.ply`; `dense` runs pycolmap MVS (needs CUDA) |
| `--voxel` | 0 | Voxel downsample size (`0` = off) |
| `--nb-neighbors` / `--std-ratio` | 20 / 2.0 | Statistical outlier removal settings |
| `--depth` | 9 | Poisson octree depth (higher = more detail, slower) |
| `--density-quantile` | 0.05 | Removes this fraction of the least-supported vertices |
| `--no-crop` | off | Keep the mesh outside the point cloud's bounding box |
| `--view` | off | Open a 3D viewer when finished |

### Output

`data/mesh/<scene>/` contains `cleaned_points.ply`, `mesh.ply`, and `mesh.obj`.

### How it works

Outlier points are removed, surface normals are estimated and oriented
consistently, and Screened Poisson Surface Reconstruction builds a watertight
mesh. Poisson also creates a "balloon" surface far from the real data, so the
lowest-density vertices are removed and the mesh is cropped to the point
cloud's bounding box.

Things to know

- Check your Python version for open3d. As far as I know, open3d's prebuilt packages only go up to Python 3.12. Your .venv is 3.13 and .venv-mac is 3.14, so pip install open3d may fail on both. If it does, make a 3.12 virtual environment for this stage.
- Expect a rough mesh from sparse points. The sparse cloud has only a few thousand points, so the mesh will be blobby. --source dense gives much better results but needs an NVIDIA GPU with a CUDA build of pycolmap, so it won't work on a Mac.
- The pycolmap dense-stereo calls (patch_match_stereo and stereo_fusion) haven't been tested. I wrote them from my knowledge of pycolmap's API. If they error on your version, check the signatures in the docs.
- Website to look at final splats.ply:
    https://superspl.at/editor

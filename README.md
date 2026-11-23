# MSDS688_Case_Study
  A case study for Regis University Class MSDS688: Artificial Intelligence

  ## Pipeline Overview

  | Stage | Script | Input | Output |
  |---|---|---|---|
  | 0: Frame extraction | `scripts/auto_extraction.py` | `source/videos/` | `source/frames/<video_name>/` |
  | 1: Structure-from-Motion | `scripts/run_sfm.py` | `source/frames/<scene>/` | `data/colmap/<scene>/` |
  | 2A: Photogrammetry mesh | `scripts/photogrammetry.py` | `data/colmap/<scene>/` | `data/mesh/<scene>/` |
  | 2B: Gaussian splatting | `scripts/gaussian_splat.py` | `data/colmap/<scene>/` | `data/splats/<scene>/` |

  Stages 2A and 2B are separate branches. Both read the Stage 1 output.

  ## Installation

  Install every dependency from the project root:

  ```
  pip install -r requirements.txt
  ```

  | Package | Needed for |
  |---|---|
  | `opencv-python`, `numpy` | Stage 0 |
  | `pycolmap` (4.2.1 or newer) | Stages 1, 2A, 2B |
  | `open3d` | Stage 2A |
  | `torch`, `gsplat`, `lpips`, `Pillow` | Stage 2B |

  Notes:
  - **open3d** prebuilt packages may only go up to Python 3.12. If `pip install`
    fails on open3d, make a Python 3.12 virtual environment for Stage 2A, or
    remove `open3d` from `requirements.txt` if you don't need that stage.
  - **Stage 2B needs an NVIDIA GPU with CUDA.** Install a CUDA build of PyTorch
    first (https://pytorch.org/get-started/locally/), then install the rest.
  - Don't install `opencv-python` and `opencv-python-headless` in the same
    environment. Either one works, but having both can cause conflicts.


  ## Stage 0: Frame Extraction using auto_extraction.py
  `scripts/auto_extraction.py` pulls the sharpest frame from evenly spaced time
  windows of each video. It is a wrapper around `extract_sharp_frames` in
  `scripts/extract_frames.py`.

  ### Setup

  1. Install the dependencies (see [Installation](#installation)).

  2. Put your videos in `source/videos/`. The script creates this folder on its
  first run if it doesn't exist. Supported formats: `.mp4`, `.mov`, `.avi`,
  `.mkv`, `.m4v`.

  The `source/` folder is in `.gitignore`, so videos and frames are not
  committed to the repo.

  ### Usage

  Run from the project root:

  ```
  python scripts/auto_extraction.py                   # every video in source/videos
  python scripts/auto_extraction.py --skip-existing   # resume after an interruption
  python scripts/auto_extraction.py my_video.mp4      # a single video
  python scripts/auto_extraction.py --n 100 --max-side 1280
  ```

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


  ## Stage 1: Structure-from-Motion using run_sfm.py
  `scripts/run_sfm.py` uses pycolmap to estimate a camera pose for each frame and
  build a sparse 3D point cloud of the scene. Both Stage 2 branches use its output.

  ### Usage

  Run from the project root:

  ```
  python scripts/run_sfm.py source/frames/<scene>                          # sequential matching
  python scripts/run_sfm.py source/frames/<scene> --matcher exhaustive     # slower, more robust
  python scripts/run_sfm.py source/frames/<scene> --threads 2 --max-image-size 1200   # less RAM
  ```

  | Option | Default | Description |
  |---|---|---|
  | `images` | required | Folder of frames (only `.jpg` files are used) |
  | `--out` | `data/colmap/<scene>` | Output folder |
  | `--matcher` | `sequential` | `sequential` (video frames in order) or `exhaustive` (every pair) |
  | `--overlap` | 20 | Number of neighbouring frames to match in sequential mode |
  | `--camera-model` | `OPENCV` | COLMAP camera model (one shared camera for all frames) |
  | `--threads` | 4 | Feature extraction threads (more = faster but more RAM) |
  | `--max-image-size` | 1600 | Downscale the long side to this for SIFT (`-1` = full size) |

  ### Output

  `data/colmap/<scene>/` contains:

  - `database.db`: COLMAP features and matches
  - `sparse/`: every model COLMAP reconstructed
  - `sparse_best/`: the largest model (`cameras`, `images`, `points3D` `.bin`)
  - `sparse_points.ply`: the sparse point cloud (used by Stage 2A)
  - `undistorted/`: undistorted images and pinhole cameras (used by Stage 2B)

  ### How it works

  SIFT keypoints are extracted from each frame, matched between frames, and fed
  to incremental Structure-from-Motion with bundle adjustment. If COLMAP splits
  the scene into several models, the one with the most registered images is kept.
  Each run deletes the previous database and sparse models and starts clean.

  Things to know

  - CPU SIFT uses a lot of RAM per thread. If the process is killed for running
    out of memory, lower `--threads` or `--max-image-size`.
  - If few images register, the frames may be too blurry or lack parallax.
    Try `--matcher exhaustive` or capture more overlapping footage.


  ## Stage 2A: Photogrammetry Mesh using photogrammetry.py
  `scripts/photogrammetry.py` turns the Stage 1 COLMAP points into a polygon mesh
  using open3d.

  ### Setup

  Install open3d (see the Python version note in [Installation](#installation)).

  Run `scripts/run_sfm.py` first. It writes the `data/colmap/<scene>/` folder this
  script reads.

  ### Usage

  Run from the project root:

  ```
  python scripts/photogrammetry.py data/colmap/<scene>                  # sparse points
  python scripts/photogrammetry.py data/colmap/<scene> --source dense   # dense MVS (CUDA only)
  python scripts/photogrammetry.py data/colmap/<scene> --depth 10 --view
  ```

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

  - Expect a rough mesh from sparse points. The sparse cloud has only a few
    thousand points, so the mesh will be blobby. `--source dense` gives much
    better results but needs an NVIDIA GPU with a CUDA build of pycolmap, so it
    won't work on a Mac.
  - The pycolmap dense-stereo calls (`patch_match_stereo` and `stereo_fusion`)
    haven't been tested. If they error on your version, check the signatures in
    the pycolmap docs.


  ## Stage 2B: Gaussian Splatting using gaussian_splat.py
  `scripts/gaussian_splat.py` seeds 3D Gaussians from the Stage 1 points and
  camera poses, then trains them with gsplat to produce a photorealistic splat
  scene.

  ### Setup

  Needs an NVIDIA GPU with CUDA and a CUDA build of PyTorch (see
  [Installation](#installation)). Run `scripts/run_sfm.py` first.
  ### Usage

  Run from the project root:

  ```
  python scripts/gaussian_splat.py data/colmap/<scene>                    # 30k steps, half resolution
  python scripts/gaussian_splat.py data/colmap/<scene> --resume           # continue after a crash / Ctrl+C
  python scripts/gaussian_splat.py data/colmap/<scene> --steps 7000       # quick preview
  python scripts/gaussian_splat.py data/colmap/<scene> --data-factor 1    # full resolution (more VRAM)
  python scripts/gaussian_splat.py data/colmap/<scene> --test-every 0     # train on every frame
  python scripts/gaussian_splat.py data/colmap/<scene> --export-only      # write splats.ply from a checkpoint
  ```

  | Option | Default | Description |
  |---|---|---|
  | `colmap_dir` | required | Stage 1 output folder |
  | `--out` | `data/splats/<scene>` | Output folder |
  | `--steps` | 30000 | Training steps |
  | `--data-factor` | 2 | Downscale images by this factor (`1` = full size, more VRAM and slower) |
  | `--sh-degree` | 3 | Spherical-harmonic degree for view-dependent colour |
  | `--ssim-lambda` | 0.2 | Weight of the D-SSIM term in the loss |
  | `--grow-grad2d` | 0.0008 | Densification threshold (higher = fewer Gaussians, less VRAM) |
  | `--test-every` | 8 | Hold out every Nth frame for evaluation (`0` = train on all) |
  | `--white-bg` | off | White background (helps with open sky) |
  | `--ckpt-every` | 5000 | Save a checkpoint every N steps (`0` = off) |
  | `--resume` | off | Continue from `<out>/checkpoint.pt` |
  | `--export-only` | off | No training: write `splats.ply` and metrics from `<out>/checkpoint.pt` |
  | `--vram-soft` | 6.5 | GB: stop densifying above this (training continues) |
  | `--vram-hard` | 7.5 | GB: save a checkpoint and stop above this |
  | `--aniso-weight` | 0.0 | Needle-splat penalty weight (`0` = off; try `0.01`) |
  | `--aniso-max` | 5.0 | Allowed longest/middle axis ratio before the penalty applies |
  | `--exposure-comp` | off | Learn a per-image colour correction (handles exposure drift) |

  ### Output

  `data/splats/<scene>/` contains:

  - `splats.ply`: the trained Gaussians in the standard 3DGS layout
  - `checkpoint.pt`: the latest training checkpoint
  - `renders/`: side-by-side renders of the held-out test frames
  - `renders/`: side-by-side renders of the held-out test frames
  - `metrics.txt`: evaluation scores on the held-out frames
  - `exposure.csv`: per-image colour corrections (only with `--exposure-comp`)
  
  ### How it works

  Each sparse point becomes a Gaussian, sized by the distance to its nearest
  neighbours. The Gaussians are optimized against the training frames with an
  L1 + D-SSIM loss. During training, gsplat's default strategy clones, splits,
  and prunes Gaussians to add detail where it's needed.

  Things to know

  - Press Ctrl+C once to finish the current step, save a checkpoint, and stop.
    Press it twice to quit immediately. Rerun with `--resume` to continue.
  - If training crashes (for example, out of memory), the last periodic
    checkpoint is still valid, so `--resume` picks up from it.
  - If you run out of VRAM, raise `--data-factor`, raise `--grow-grad2d`, or
    lower `--vram-soft` / `--vram-hard` to match your GPU.
  - To view the final `splats.ply`, open it in https://superspl.at/editor
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

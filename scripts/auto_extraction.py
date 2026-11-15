"""Shortcut for extract_frames.py.

Usage (from project root):
python scripts/auto_extraction.py                  # every video in source/videos
python scripts/auto_extraction.py --skip-existing  # resume after an interruption
python scripts/auto_extraction.py my_video.mp4     # still works for a single video

"""
import argparse
import sys
from pathlib import Path


# Let this file run directly as `python scripts/auto_extraction.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_frames import extract_sharp_frames

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "source" / "videos"     # where your videos live (adjust)
FRAMES_DIR = ROOT / "source" / "frames"    # output root
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

p = argparse.ArgumentParser(description="Extract sharpest frames from videos")
p.add_argument("video", nargs="?", help="one video (name in source/videos, or full path). Omit to process ALL videos.")
p.add_argument("--n", type=int, default=200)
p.add_argument("--max-side", type=int, default=1920)
p.add_argument("--skip-existing", action="store_true", help="skip videos that already have a frames folder")
args = p.parse_args()

if args.video:
    video = Path(args.video)
    videos = [video if video.exists() else VIDEO_DIR / args.video]
else:
    videos = sorted(f for f in VIDEO_DIR.iterdir() if f.suffix.lower() in VIDEO_EXTS)

if not videos:
    sys.exit(f"No videos found in {VIDEO_DIR}")

failed = []
for i, video in enumerate(videos, 1):
    out = FRAMES_DIR / video.stem
    if args.skip_existing and (out / "frames_log.csv").exists():
        print(f"[{i}/{len(videos)}] Skipping {video.name} (already done)")
        continue
    print(f"[{i}/{len(videos)}] {video.name}")
    try:
        extract_sharp_frames(video, out, args.n, args.max_side or None)
    except RuntimeError as e:   # unreadable or corrupt video: log it and keep going
        print(f"  FAILED: {e}")
        failed.append(video.name)

if failed:
    print(f"\n{len(failed)} failed: {', '.join(failed)}")


import cv2
import argparse
from pathlib import Path
import csv





#Returns the "sharpness" of a frame, which is the variance of the Laplacian of the image.
def sharpness(frame, work_width = 960):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale = work_width / gray.shape[1]
    small = cv2.resize(gray, None, fx = scale, fy = scale, interpolation = cv2.INTER_AREA)

    return cv2.Laplacian(small, cv2.CV_64F).var()





def resize_max_side(frame, max_side):
    if max_side is None:
        return frame

    h, w = frame.shape[:2]
    scale = max_side / max(h, w)

    if scale >= 1.0:
        return frame

    new_size = (round(w * scale), round(h * scale))

    return cv2.resize(frame, new_size, interpolation = cv2.INTER_AREA)





def extract_sharp_frames(video_path, output_dir, n_windows = 200, max_side = 1920, jpg_quality = 95):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents = True, exist_ok = True)

    cap = cv2.VideoCapture(str(video_path))

    #Ensure that the video is a valid video file
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #Ensure that the video is actually a video and not corrupted
    if total <= 0:
        raise RuntimeError(f"Video reports 0 frames.")

    #If total windows in video is less than the n_windows, we still use the video
    n_windows = min(n_windows, total)

    #One row per saved frame, for the CSV log
    records = []
    current_window = 0
    best_score, best_idx, best_frame = -1.0, -1, None

    def save_best(window):
        name = f"frame_{window:04d}.jpg"
        out = resize_max_side(best_frame, max_side)
        cv2.imwrite(str(output_dir / name), out, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
        records.append({'file': name,
                       'window': window,
                       'source_frame': best_idx,
                       'sharpness': round(best_score, 2)
                       })



    idx = 0
    while True:
        ok, frame = cap.read()

        if not ok:
            break

        window = min(idx * n_windows // total, n_windows -1)

        #Entered a new window -> write out the winner of the previous one
        if window != current_window:
            if best_frame is not None:
                save_best(current_window)

            current_window = window
            best_score, best_idx, best_frame = -1.0, -1, None

        score = sharpness(frame)
        if score > best_score:
            best_score, best_idx, best_frame = score, idx, frame

        idx += 1

    #The loop ends without "entering a new window", so save the last one manually
    if best_frame is not None:
        save_best(current_window)
    cap.release()

    with open(output_dir / "frames_log.csv", "w", newline = '') as f:
        writer = csv.DictWriter(f, fieldnames = records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print(f"Read {idx} frames (header said {total}); saved {len(records)} to {output_dir}")





if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 0: extract sharpest frame per time window")
    p.add_argument("video")
    p.add_argument("--out", default="data/frames")
    p.add_argument("--n", type=int, default=200, help="number of frames / windows")
    p.add_argument("--max-side", type=int, default=1920, help="resize long side (0 = full res)")
    args = p.parse_args()

    extract_sharp_frames(args.video, args.out, args.n, args.max_side or None)

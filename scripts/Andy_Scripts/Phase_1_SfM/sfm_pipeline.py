"""Stage 1: Structure-from-Motion with pycolmap (pinned 4.2.1).

Input : folder of Stage 0 frames (frame_0000.jpg ...), consecutive in time.
Output: <out_dir>/
          database.db        COLMAP database
          sparse/            raw mapper output (one numbered subfolder per model)
          sparse_best/       largest model, COLMAP binary: cameras/images/points3D
          sparse_points.ply  sparse cloud for Stage 2A
          undistorted/       undistorted images + pinhole model for Stage 2B (gsplat)
          summary.txt        registration summary

NOTE: keyword names marked UNVERIFIED are not in the pycolmap docs page.
Confirm them with the signature printout (notebook cell 3) before trusting them.
"""
import shutil
from pathlib import Path

import pycolmap

import csv, json, time
import numpy as np


def _cam_center(im):
    try:
        return np.asarray(im.projection_center())
    except AttributeError:
        T = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        M = T.matrix()
        return -M[:, :3].T @ M[:, 3]


def export_metrics(rec, maps, names, out_dir, timings, settings):
    """Write metrics.json and camera_path.csv for later visualization/comparison."""
    out_dir = Path(out_dir)
    reg = sorted(rec.images.values(), key=lambda im: im.name)
    reg_names = [im.name for im in reg]

    with open(out_dir / "camera_path.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["name", "x", "y", "z"])
        for im in reg:
            w.writerow([im.name, *_cam_center(im).tolist()])

    metrics = {
        "pycolmap": pycolmap.__version__,
        "has_cuda": bool(getattr(pycolmap, "has_cuda", False)),
        "settings": settings,
        "timings_s": timings,
        "n_models": len(maps),
        "registered_per_model": {str(k): m.num_reg_images() for k, m in maps.items()},
        "registered": rec.num_reg_images(),
        "total": len(names),
        "unregistered": sorted(set(names) - set(reg_names)),
        "points3D": rec.num_points3D(),
        "observations": rec.compute_num_observations(),
        "mean_reproj_px": rec.compute_mean_reprojection_error(),
        "cameras": [{"id": cid, "text": str(c)} for cid, c in rec.cameras.items()],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

def prepare_frames(src_dir, dst_dir, n_frames=None):
    """Copy the first n_frames *consecutive* frames (all if None) to dst_dir.

    Consecutive, not strided, so neighbouring frames keep their overlap.
    Returns the list of image names used.
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    frames = sorted(src_dir.glob("*.jpg"))   # skips frames_log.csv
    if not frames:
        raise RuntimeError(f"No .jpg files in {src_dir}")
    if n_frames is not None:
        frames = frames[:n_frames]

    shutil.rmtree(dst_dir, ignore_errors=True)
    dst_dir.mkdir(parents=True)
    for f in frames:
        shutil.copy(f, dst_dir / f.name)
    return [f.name for f in frames]


def extract(database_path, image_dir, camera_model="OPENCV"):
    """Step 1: SIFT features, one shared camera for all frames."""
    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = camera_model                 # UNVERIFIED field name
    pycolmap.extract_features(
        database_path,
        image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,                # UNVERIFIED kwarg + enum
        reader_options=reader_options,                         # UNVERIFIED kwarg
    )


def match(database_path, overlap=20):
    """Step 2: sequential matching (frames come from one continuous walk)."""
    pairing = pycolmap.SequentialPairingOptions()              # UNVERIFIED class name
    pairing.overlap = overlap                                  # UNVERIFIED field name
    pycolmap.match_sequential(database_path, pairing_options=pairing)  # UNVERIFIED kwarg


def map_incremental(database_path, image_dir, sparse_dir):
    """Step 3: incremental mapping with bundle adjustment. Returns {id: Reconstruction}."""
    sparse_dir = Path(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    maps = pycolmap.incremental_mapping(database_path, image_dir, sparse_dir)
    if not maps:
        raise RuntimeError("No model was reconstructed (check parallax, blur, overlap).")
    return maps


def pick_best(maps):
    """Largest model by registered image count."""
    best_id = max(maps, key=lambda k: maps[k].num_reg_images())
    return best_id, maps[best_id]


def summarize(maps, best_id, rec, image_names):
    """Registration summary string (uses only documented Reconstruction methods)."""
    reg_names = {im.name for im in rec.images.values()}
    missing = sorted(set(image_names) - reg_names)
    lines = [
        f"models found:            {len(maps)}  "
        f"(registered per model: { {k: m.num_reg_images() for k, m in maps.items()} })",
        f"best model id:           {best_id}",
        f"registered / total:      {rec.num_reg_images()} / {len(image_names)}",
        f"sparse 3D points:        {rec.num_points3D()}",
        f"observations:            {rec.compute_num_observations()}",
        f"mean reprojection error: {rec.compute_mean_reprojection_error():.4f} px",
        f"unregistered frames ({len(missing)}): {missing}",
    ]
    return "\n".join(lines)


def save_outputs(rec, image_dir, out_dir, undistort=True):
    """Write COLMAP binary model, sparse PLY, and (optionally) undistorted images."""
    out_dir = Path(out_dir)
    best_dir = out_dir / "sparse_best"
    shutil.rmtree(best_dir, ignore_errors=True)
    best_dir.mkdir(parents=True)
    rec.write(best_dir)                                        # cameras/images/points3D .bin
    rec.export_PLY(str(out_dir / "sparse_points.ply"))
    if undistort:
        und = out_dir / "undistorted"
        shutil.rmtree(und, ignore_errors=True)
        # arg order (output, sparse input, images) inferred from the README example
        pycolmap.undistort_images(und, best_dir, image_dir)


# def run_sfm(frames_src, work_dir, n_frames=None, camera_model="OPENCV",
#             overlap=20, undistort=True):
#     """Full Stage 1. Returns (reconstruction, summary string)."""
#     work_dir = Path(work_dir)
#     shutil.rmtree(work_dir, ignore_errors=True)   # COLMAP appends to old databases
#     work_dir.mkdir(parents=True)

#     image_dir = work_dir / "images"
#     database_path = work_dir / "database.db"

#     names = prepare_frames(frames_src, image_dir, n_frames)
#     print(f"Using {len(names)} frames")

#     extract(database_path, image_dir, camera_model)
#     match(database_path, overlap)
#     maps = map_incremental(database_path, image_dir, work_dir / "sparse")

#     best_id, rec = pick_best(maps)
#     text = summarize(maps, best_id, rec, names)
#     print(text)
#     (work_dir / "summary.txt").write_text(text + "\n\n" + rec.summary())

#     save_outputs(rec, image_dir, work_dir, undistort)
#     return rec, text
def run_sfm(frames_src, work_dir, n_frames=None, camera_model="OPENCV",
            overlap=20, undistort=True):
    """Full Stage 1. Returns (reconstruction, summary string)."""
    work_dir = Path(work_dir)
    shutil.rmtree(work_dir, ignore_errors=True)   # COLMAP appends to old databases
    work_dir.mkdir(parents=True)

    image_dir = work_dir / "images"
    database_path = work_dir / "database.db"
    timings = {}

    names = prepare_frames(frames_src, image_dir, n_frames)
    print(f"Using {len(names)} frames")

    t = time.time(); extract(database_path, image_dir, camera_model)
    timings["extract"] = time.time() - t
    t = time.time(); match(database_path, overlap)
    timings["match"] = time.time() - t
    t = time.time(); maps = map_incremental(database_path, image_dir, work_dir / "sparse")
    timings["mapping"] = time.time() - t

    best_id, rec = pick_best(maps)
    text = summarize(maps, best_id, rec, names)
    print(text)
    (work_dir / "summary.txt").write_text(text + "\n\n" + rec.summary())

    t = time.time(); save_outputs(rec, image_dir, work_dir, undistort)
    timings["save_undistort"] = time.time() - t
    timings = {k: round(v, 1) for k, v in timings.items()}
    print("timings (s):", timings)

    export_metrics(rec, maps, names, work_dir, timings,
                   {"n_frames": n_frames, "camera_model": camera_model, "overlap": overlap})
    return rec, text
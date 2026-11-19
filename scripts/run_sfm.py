import argparse
import shutil
from pathlib import Path
import pycolmap




def run_sfm(image_dir, out_dir, matcher = 'sequential', overlap = 20, camera_model = 'OPENCV'):
    image_dir = Path(image_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents = True, exist_ok = True)

    database_path = out_dir / "database.db"
    sparse_dir = out_dir / 'sparse'

    # COLMAP appends to an existing database, so start clean ever run
    if database_path.exists():
        database_path.unlink()

    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)

    sparse_dir.mkdir()

    # Only feed the .jpgs (skip frames_log.csv)
    image_names = sorted(p.name for p in image_dir.glob('*.jpg'))
    if not image_names:
        raise RuntimeError(f"No .jpg files in {image_dir}")

    print(f"Found {len(image_names)} images")

    # 1) SIFT keypoints - one shared camera (same phone/lens, AE/AF locked)
    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = camera_model
    pycolmap.extract_features(
        database_path,
        image_dir,
        image_names = image_names,
        camera_mode = pycolmap.CameraMode.SINGLE,
        reader_options = reader_options
    )

    # 2) Matching
    if matcher == 'sequential':
        seq_options = pycolmap.SequentialPairingOptions()
        seq_options.overlap = overlap
        pycolmap.match_sequential(database_path, pairing_options
  = seq_options)
    else:
        pycolmap.match_exhaustive(database_path)

    # 3) Incremental SfM + bundle adjustment
    maps = pycolmap.incremental_mapping(database_path, image_dir, sparse_dir)
    if not maps:
        raise RuntimeError('COLMAP could not register any images (check parallax / blur)')

    # Keep the largest model if COLMAP split the scene
    best_id = max(maps, key = lambda k: maps[k].num_reg_images())
    rec = maps[best_id]

    print(f"{len(maps)} model(s) found; using model {best_id}")
    print(rec.summary())
    print(f"Registered {rec.num_reg_images()} / {len(image_names)} images")

    # 4) Save outputs
    best_dir = out_dir / 'sparse_best'
    best_dir.mkdir(exist_ok = True)
    rec.write(best_dir)                                 # cameras/images/points3D .bin
    rec.export_PLY(str(out_dir / 'sparse_points.ply'))  # for Stage 2A

    # Undistorted images + pinhole cameras (images/ + sparse/ layout gsplat expects)
    pycolmap.undistort_images(out_dir / 'undistorted', best_dir, image_dir)

    return rec




if __name__ == "__main__":
    p = argparse.ArgumentParser(description = 'Stage 1: Structure-from-Motion with pycolmap')
    p.add_argument('images', help = 'folder of frames')
    p.add_argument('--out', default = None, help = 'default: data/colmap/<scene name>')
    p.add_argument('--matcher', choices = ['sequential', 'exhaustive'], default = 'sequential')
    p.add_argument('--overlap', type = int, default = 20)
    p.add_argument('--camera-model', default = 'OPENCV')
    args = p.parse_args()

    out = args.out or f"data/colmap/{Path(args.images).name}"
    run_sfm(args.images, out, args.matcher, args.overlap, args.camera_model)


"""Read-only look at Stage 1 output. Usage: python inspect_stage1.py IMG_9175"""
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pycolmap

ROOT = Path(__file__).resolve().parents[3]          # Case_Study
scene = sys.argv[1]
run = ROOT / "data" / "colmap" / scene / "run_full_colab"

pcd = o3d.io.read_point_cloud(str(run / "sparse_points.ply"))
rec = pycolmap.Reconstruction(str(run / "sparse_best"))
centers = np.array([im.projection_center() for im in rec.images.values()])

print("open3d", o3d.__version__, "| pycolmap", pycolmap.__version__)
print("registered frames:", rec.num_reg_images())
print("points:", len(pcd.points), "| has colors:", pcd.has_colors())

cams = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centers))
cams.paint_uniform_color([1.0, 0.0, 0.0])           # camera positions in red
o3d.visualization.draw_geometries([pcd, cams], window_name=scene)
"""
Stage 2A: Photogrammetry branch.

COLMAP points -> clean outliers -> estimate normals -> Screened Poisson -> .ply / .obj mesh

Usage:
python scripts/photogrammetry.py data/colmap/<scene>                    # sparse points (works everywhere)
python scripts/photogrammetry.py data/colmap/<scene> --source dense     # dense MVS (needs CUDA pycolmap)
python scripts/photogrammetry.py data/colmap/<scene> --view             # open a viewer when done
"""

import argparse
from pathlib import Path
import numpy as np
import open3d as o3d

MIN_POINTS = 100


def load_sparse(colmap_dir):
    ply = colmap_dir / 'sparse_points.ply'  # written by run_sfm.py
    if not ply.exists():
        raise RuntimeError(f"{ply} not found - run scripts/run_sfm.py first")

    return o3d.io.read_point_cloud(str(ply))




def load_dense(colmap_dir):
    """
    Multi-View Stereo on the undistorted images.  Needs pycolmap built with CUDA.
    """
    import pycolmap

    workspace = colmap_dir / 'undistorted'
    dense_ply = colmap_dir / 'dense_points.ply'

    # MVS is slow, so reuse a previous result if one exists
    if dense_ply.exists():
        print(f"Reusing {dense_ply}")
        return o3d.io.read_point_cloud(str(dense_ply))

    if not workspace.exists():
        raise RuntimeError(f"{workspace} not found - run scripts/run_sfm.py first")

    try:
        pycolmap.patch_match_stereo(workspace)          # per-image depth maps
        pycolmap.stereo_fusion(dense_ply, workspace)    # fuse depth maps -> one cloud
    except Exception as e:
        raise RuntimeError(f"Dense MVS failed ({e}). This needs a CUDA build of pycolmap; "
                           f"use --source sparse instead.") from e

    return o3d.io.read_point_cloud(str(dense_ply))



def clean_points(pcd, voxel = 0.0, nb_neighbors = 20, std_ratio = 2.0):
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)

    before = len(pcd.points)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors = nb_neighbors, std_ratio = std_ratio)

    print(f"Outlier removal: kept {len(pcd.points)} / {before} points "
          f"({before - len(pcd.points)} removed)")
    return pcd



def estimate_normals(pcd, radius = None):
    # COLMAP's exported normals are zeros, so always recompute
    # Default radius scales with the cloud: ~2X the mean nearest-neighbour spacing
    if radius is None:
        radius = 2.0 * float(np.mean(pcd.compute_nearest_neighbor_distance()))

    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius = radius, max_nn = 30))

    # Poisson needs normals that agree on which side is "outside"
    pcd.orient_normals_consistent_tangent_plane(k = 15)

    return pcd



def poisson_mesh(pcd, depth = 9, density_quantile = 0.05, crop = True):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth = depth)

    # Poisson closes the surface everywhere, inflating a "balloon" far from real data
    # Those faces have low density (few supporting points), so cut them off.
    densities = np.asarray(densities)
    if density_quantile > 0:
        mesh.remove_vertices_by_mask(densities < np.quantile(densities, density_quantile))

    if crop:
        mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    return mesh



def run_photogrammetry(colmap_dir, out_dir, source = 'sparse', voxel = 0.0, nb_neighbors = 20,
                       std_ratio = 2.0, depth = 9, density_quantile = 0.05, crop = True):
    colmap_dir = Path(colmap_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents = True, exist_ok = True)

    # 1) Load points
    # Note: a sparse SfM cloud is thin (a few thousand points), so its mesh will be blobby.
    pcd = load_dense(colmap_dir) if source == 'dense' else load_sparse(colmap_dir)

    print(f"Loaded {len(pcd.points)} {source} points")
    if len(pcd.points) < MIN_POINTS:
        raise RuntimeError(f"Only {len(pcd.points)} points - too few to mesh (check stage 1)")

    # 2) Clean statistical outliers
    pcd = clean_points(pcd, voxel, nb_neighbors, std_ratio)

    # 3) Surface normals
    pcd = estimate_normals(pcd)

    # 4) Screened Poisson + trim
    mesh = poisson_mesh(pcd, depth, density_quantile, crop)
    if len(mesh.triangles) == 0:
        raise RuntimeError('Poisson produced an empty mesh - try a lower --density-quantile or --depth')

    # 5) Save outputs
    o3d.io.write_point_cloud(str(out_dir / 'cleaned_points.ply'), pcd)
    o3d.io.write_triangle_mesh(str(out_dir / 'mesh.ply'), mesh)
    o3d.io.write_triangle_mesh(str(out_dir / 'mesh.obj'), mesh)

    print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles -> {out_dir}")

    return mesh



if __name__ == "__main__":
    p = argparse.ArgumentParser(description = 'Stage 2A: Photogrammetry mesh with open3d')
    p.add_argument('colmap_dir', help = 'Stage 1 output folder, e.g. data/colmap/<scene>')

    p.add_argument('--out', default = None, help = 'default: data/mesh/<scene name>')

    p.add_argument('--source', choices = ['sparse', 'dense'], default = 'sparse')

    p.add_argument('--voxel', type = float, default = 0.0, help = 'voxel downsample size (0 = off)')

    p.add_argument('--nb-neighbors', type = int, default = 20)

    p.add_argument('--std-ratio', type = float, default = 2.0)

    p.add_argument('--depth', type = int, default = 9, help = 'Poisson octree depth (higher = more detail)')

    p.add_argument('--density-quantile', type = float, default = 0.05,
                   help = 'trim this fraction of lowest-density vertices (0 = off)')

    p.add_argument('--no-crop', action = 'store_true', help = "don't crop mesh to the point cloud's bounding box")

    p.add_argument('--view', action = 'store_true', help = 'open a viewer when done')

    args = p.parse_args()

    out = args.out or f"data/mesh/{Path(args.colmap_dir).name}"
    mesh = run_photogrammetry(args.colmap_dir, out, args.source, args.voxel, args.nb_neighbors,
                              args.std_ratio, args.depth, args.density_quantile, not args.no_crop)

    if args.view:
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face = True)
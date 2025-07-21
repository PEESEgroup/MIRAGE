import trimesh
import pyrender
import utils
import open3d as o3d
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import os
import bspline_utils
import geomdl
from geomdl.visualization import VisVTK
from bspline_utils import get_mesh_from_surfaces, get_approx_mesh
from numpy import dot
from numpy.linalg import norm

def get_leaf_obj_from_ctrpts(plant_id, ctrpts_folder, voxels_folder):
    files = [f for f in os.listdir(ctrpts_folder) if f.startswith(plant_id) and os.path.isfile(os.path.join(ctrpts_folder, f))]
    files_dict = {f.split('.')[0].split('Leaf')[1]: f for f in files}
    voxel_metric = np.load(voxels_folder + f'/{plant_id}.npy')
    surf_list = []
    # Load the voxel metric and find unique leaf IDs
    for n in range(1, len(files)+1):
        leaf_approx_id = n
        filename = files_dict[str(leaf_approx_id)]
        leaf1_voxel =  np.argwhere(voxel_metric == leaf_approx_id) 
        voffset = np.array([16, 16, 16]) - np.mean(leaf1_voxel, axis=0).astype(int)
        ctrlpts = np.loadtxt(os.path.join(ctrpts_folder, filename))
        ctrlpts = (ctrlpts - voffset)/64
        approx_mesh, surf = bspline_utils.get_approx_mesh(ctrlpts, degree_u=2, degree_v=3, size=20)
        surf_list.append(surf)
    #     o3d_mesh = o3d.geometry.TriangleMesh()
    # # Set vertices and faces
    #     o3d_mesh.vertices = o3d.utility.Vector3dVector(approx_mesh.vertices)
    #     o3d_mesh.triangles = o3d.utility.Vector3iVector(approx_mesh.faces)
    #     leaf_meshes.append(o3d_mesh)
    # combined_mesh = o3d.geometry.TriangleMesh()
    # for m in leaf_meshes:
    #     combined_mesh += m
    # o3d.io.write_triangle_mesh("all_leaf_meshes.obj", combined_mesh)
    return surf_list

def leaf_base_v(leaf_surf):
    leaf_end = np.array(leaf_surf.evaluate_single((0.5, 1.0)))  # Assuming the leaf tip is at (0.5, 1.0)
    leaf_start = np.array(leaf_surf.evaluate_single((0.5, 0.0)))  # Assuming the leaf base is at (0.5, 0.0)
    if np.linalg.norm(leaf_end[:2] - [0.5,0.5]) < np.linalg.norm(leaf_start[:2] - [0.5,0.5]):
        return 1.0
    else:
        return 0.0

def get_leaf_direction(leaf_surf):
    from augment import compute_3d_curvature, find_peaks_in_curvature, pca_endpoint_angle
    v_fixed = 0.5  # Fixed v value for evaluation
    # Evaluate the surface at fixed v and varying u
    if leaf_base_v(leaf_surf) == 0.0:
        us = np.linspace(0, 1, 100)
    else:
        us = np.linspace(1, 0, 100)
    points = np.array([leaf_surf.evaluate_single((v_fixed, u)) for u in us])
    curvature = compute_3d_curvature(points)
    peak_indices, properties = find_peaks_in_curvature(curvature, 2, height=0.05, distance=20)
    print(f"Found peaks at indices: {peak_indices}, heights: {properties['peak_heights']}")
    _, dir_start, _ , _, _  = pca_endpoint_angle(points, peak_indices)
    return dir_start

# leaf_mesh = get_leaf_obj_from_ctrpts('0100', r"C:\Users\Jacob\Desktop\Skeleton\ctrlpts_prediction\ctrlpts", r"C:\Users\Jacob\Desktop\Skeleton\maize_voxel")
# # Visualize the leaf mesh
# o3d.visualization.draw_geometries([leaf_mesh])

def match_leaf_at_u(leaf, stem, u, stem_centerline_curve, resolution=40):
    uv_coords = np.linspace(0, 1, resolution)
    stem_surface_point_at_v = np.array([stem.evaluate_single((uv_coords[u] , v)) for v in uv_coords])           
    stem_center_point = np.array(stem_centerline_curve.evaluate_single(uv_coords[u])) 
    stem_dirs = stem_surface_point_at_v - stem_center_point 
    leaf_dir = get_leaf_direction(leaf)
    dir_similarities = [dot(stem_dir, leaf_dir) / (norm(stem_dir) * norm(leaf_dir)) for stem_dir in stem_dirs]
    largest_idx = np.argmax(dir_similarities) #v coord
    return largest_idx

def attach_leaf_to_stem(stem_surf, stem_centerline_curve, leaf_surf):

    resolution = 40  # Number of points in each direction
    uv_coords = np.linspace(0, 1, resolution)
    stem_points = np.zeros((resolution, resolution, 3))

    for i, u in enumerate(uv_coords):
        for j, v in enumerate(uv_coords):
            pt = stem_surf.evaluate_single((u, v))
            stem_points[i, j] = pt

    new_surf_list = []
    leaf_locations = []
    for leaf in leaf_surf:
        # leaf = leaf_surf[2]
        stem_points_reshaped = stem_points.reshape(-1, 3)
        leaf_tip_np = np.array(leaf.evaluate_single((0.5, leaf_base_v(leaf))))  # Assuming leaf_base_u is the base u value for the leaf tip
        distances = np.linalg.norm(stem_points_reshaped - leaf_tip_np, axis=1)
        closest_idx = np.argmin(distances)
        closest_point = stem_points_reshaped[closest_idx]
        closest_idx_2d = np.unravel_index(closest_idx, stem_points.shape[:2])
        largest_idx = match_leaf_at_u(leaf, stem_surf, closest_idx_2d[0], stem_centerline_curve)

        leaf_locations.append((closest_idx_2d[0],largest_idx))

        closest_point = stem_points[closest_idx_2d[0],largest_idx, :]

        move_dir = leaf_tip_np - closest_point
        leaf_ctrlpts = np.array(leaf.ctrlpts)
        leaf_ctrlpts -= move_dir
        # print(f"Largest similarity index: {largest_idx}, value: {dir_similarities[largest_idx]}")
        approx_mesh, surf = get_approx_mesh(leaf_ctrlpts, degree_u=2, degree_v=3, size=20)
        new_surf_list.append(surf)
        # plant_mesh = get_mesh_from_surfaces(leaf_surf + [stem_surf])

    return new_surf_list, leaf_locations

from stem_reconstruction import get_stem_centerline_direction_at_u

def get_leaf_stem_angle(leaf_surf, leaf_location, stem_centerline_curve):
    leaf_dir = get_leaf_direction(leaf_surf)
    stem_dir = get_stem_centerline_direction_at_u(stem_centerline_curve, leaf_location[0])
    angle = np.arccos(np.clip(dot(leaf_dir, stem_dir) / (norm(leaf_dir) * norm(stem_dir)), -1.0, 1.0))
    return angle
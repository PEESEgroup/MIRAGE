import random
from stem_reconstruction import get_stem_obj_from_voxel
from leaf_reconstruction import get_leaf_obj_from_ctrpts, get_leaf_direction
from bspline_utils import get_mesh_from_surfaces
import open3d as o3d
from augment import adjust_leaf_loc_on_stem, shrink_leaf_by_mask, scale_leaf_by_mask, adjust_leaf_curvature, adjust_leaf_stem_angle
import os
import numpy as np
import trimesh

def sample_leaf_uv(leaf_number):
    # Sample attachment points for each leaf
    attachment_points = []
    for _ in range(leaf_number):
        # Randomly choose side: left ([0, 0.2]) or right ([0.5, 0.7])
        if random.random() < 0.5:
            u_range = (0.0, 0.2)
        else:
            u_range = (0.5, 0.7)
        u = random.uniform(*u_range)

        # Sample v with minimum separation of 0.2 from other leaves on the same side
        side_points = [pt for pt in attachment_points if (pt[0] >= 0.0 and pt[0] <= 0.2) == (u_range == (0.0, 0.2))]
        max_attempts = 100
        for attempt in range(max_attempts):
            v = random.uniform(0.0, 1.0)
            if all(abs(v - pt[1]) >= 0.2 for pt in side_points):
                break
        else:
            # If unable to find a valid v after max_attempts, just pick one (may overlap)
            v = random.uniform(0.0, 1.0)
        attachment_points.append((u, v))

    return attachment_points

plant_id_list = [f'{i:04d}' for i in range(1, 521)]  # Example plant IDs from 0001 to 0520
voxels_path = r'C:\Users\Jacob\Desktop\Skeleton\maize_voxel/'
ctrpts_folder = r"C:\Users\Jacob\Desktop\Skeleton\ctrlpts_prediction\ctrlpts"

# Prepare to collect all leaf surfaces
all_leafs = []
for filename in os.listdir(ctrpts_folder):
    if filename.endswith('.npy'):
        plant_id = filename.split('.')[0]  # Remove the .npy extension
        leaf_surf_list = get_leaf_obj_from_ctrpts(plant_id, ctrpts_folder, voxels_path)
        all_leafs.extend(leaf_surf_list)

N = 1000  # Number of leaf surfaces to sample
for n in range(N):  # Augmentation sample number
    plant_id = random.choice(plant_id_list)
    stem_surf, stem_centerline_curve = get_stem_obj_from_voxel(voxels_path + plant_id + '.npy')
    leaf_number = random.randint(3, 12)  # Randomly choose number of leaves
    leaf_surf_list = random.sample(all_leafs, leaf_number)  # Randomly sample leaf surfaces
    attachment_points = sample_leaf_uv(leaf_number)

    for leaf_n in range(leaf_number):
        # Sample attachment points for the leaf
        uv = attachment_points[leaf_n]
        leaf_surf = leaf_surf_list[leaf_n]
        # Adjust leaf location on stem
        leaf_surf = adjust_leaf_loc_on_stem(leaf_surf, stem_surf, uv)
        # Randomly choose augmentation to apply
        augment_number = random.randint(1, 3)  # Randomly choose one of the four augmentations
        augment_funcs = [shrink_leaf_by_mask, scale_leaf_by_mask, adjust_leaf_curvature, adjust_leaf_stem_angle]
        random.shuffle(augment_funcs)
        for an in range(augment_number):
            aug_func = augment_funcs[an]
            if aug_func == shrink_leaf_by_mask:
                leaf_surf = aug_func(leaf_surf, [0, 2], np.random.uniform(0.25, 2))
            elif aug_func == scale_leaf_by_mask:
                leaf_surf = aug_func(leaf_surf, [0, 2], np.random.uniform(0.25, 2))
            elif aug_func == adjust_leaf_curvature:
                leaf_surf = aug_func(leaf_surf, given_angle = np.random.uniform(30/180, 1)*np.pi)
            elif aug_func == adjust_leaf_stem_angle:
                leaf_surf = aug_func(leaf_surf, uv,stem_centerline_curve, np.random.uniform(0, 150/180)*np.pi)
        leaf_surf_list[leaf_n] = leaf_surf


    plant_mesh = get_mesh_from_surfaces(leaf_surf_list +  [stem_surf])
    # Combine all trimesh objects into a single scene and export as a file
    scene = trimesh.Scene(plant_mesh)
    scene.export(f'id_{n}.obj')
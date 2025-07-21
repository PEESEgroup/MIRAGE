import open3d as o3d
import numpy as np
import networkx as nx
import os
from sklearn.neighbors import KDTree, BallTree
from vis import show_graphs, show_subgraphs, show_points, show_pcd_cluster
import pcd_utils
import utils
import fielgrown
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import json
import bisect

def find_key_point_in_stem(stem_point, root_idx):
    stem_pcd = o3d.geometry.PointCloud()
    stem_pcd.points = o3d.utility.Vector3dVector(stem_point)
    g = pcd_utils.knn_graph(stem_pcd, 5)
    pcd_utils.connect_graph(g, stem_pcd, root_idx)
    cluster_values, cluster_sets, cluster_centers,_ = pcd_utils.distance_to_root_clusters(g, root_idx, stem_pcd, 0.05, min_size=0)

    # cluster_label = []
    # for i, s in enumerate(cluster_sets):
    #     if np.sum(binary_labels[list(s)] == 0) > len(s) * 0.2:
    #         cluster_label.append(0)
    #     else:
    #         cluster_label.append(1)
    stem_centriods = np.vstack(cluster_centers)
    return stem_centriods


def get_leaf_label_by_color(labels, points, threshold=100):
    color_labels = np.unique(labels, axis=0)
    color_labels = color_labels[~np.all(color_labels == [0, 0, 0], axis=1)]
    color_label_counts = {tuple(color): np.sum(np.all(labels == color, axis=1)) for color in color_labels}
    filtered_color_labels = [color for color, count in color_label_counts.items() if count > threshold]
    points_labels = {color: list(np.where(np.all(labels == color, axis=1))[0]) for color in filtered_color_labels}
    return points_labels

def relabel_match_control_point(points_labels, points, dat_file):
    control_points = utils.read_leaf_data(dat_file)
    label_list = []
    new_leaf_label_dict = {}
    leaf_diff_num = len(points_labels) - len(control_points)
    print(f"Leaf diff num: {leaf_diff_num}")
    for key, pts_idx in points_labels.items():
        label, distance = fielgrown.get_leaf_label(points[pts_idx], control_points)
        if distance > 0.05 and leaf_diff_num > 0:
            continue
        label_no = int(label.replace('Leaf', '')) 
        label_list.append(label)
        new_leaf_label_dict[label_no] = pts_idx
    assert len(label_list) == len(set(label_list))
    return new_leaf_label_dict

def get_branch_side_tip_of_leaf(leaf_points, root_point, top_point):
    stem_dir = top_point - root_point
    # Calculate the vector from root_point to each point in highest_points
    vectors_to_points = leaf_points - root_point
    # Normalize the stem direction vector
    stem_dir_normalized = stem_dir / np.linalg.norm(stem_dir)
    # Project the vectors onto the stem direction
    projections = np.dot(vectors_to_points, stem_dir_normalized)
    # Calculate the closest points on the line to each point in highest_points
    closest_points_on_line = root_point + np.outer(projections, stem_dir_normalized)
    # Calculate the distances between highest_points and the closest points on the line
    distances_to_centroid = np.linalg.norm(leaf_points - closest_points_on_line, axis=1)
    # Sort the distances to find the closest points
    closest_indices = np.argsort(distances_to_centroid)[:50]

    # Ensure the lower stem points are reserached
    closest_points_to_centroid = leaf_points[closest_indices]
    leaf_branch_ref = np.mean(closest_points_to_centroid, axis=0)
    return leaf_branch_ref

def find_conject_points_in_leaf_stem(stem_points, root_point, top_point, leaf_point):
    hpc = o3d.geometry.PointCloud()
    hpc.points = o3d.utility.Vector3dVector(leaf_point)
    cl, ind = hpc.remove_statistical_outlier(nb_neighbors=20, std_ratio=10.0)
    leaf_point = np.asarray(hpc.select_by_index(ind).points)

    leaf_branch_ref = get_branch_side_tip_of_leaf(leaf_point, root_point, top_point)
    lower_stem_points = stem_points[stem_points[:, 2] < leaf_branch_ref[2]]

    stem_branch_points, leaf_branch_point = utils.find_closest_pair(lower_stem_points, leaf_point, 5)
    # height_point_color = labels[np.argmax(stem_centroids[:, 2])]
    stem_conject_point = np.mean(stem_branch_points.reshape(-1, 3), axis=0)
    leaf_conject_point = np.mean(leaf_branch_point.reshape(-1, 3), axis=0)
    return stem_conject_point, leaf_conject_point

def find_above_points_in_the_dir_to_top(current, centriods, top_point, threshold=0.2):
    
    above_points = centriods[centriods[:, 2] > current[2]]
    # Calculate the direction vector from the branch point to the top point
    direction_vector = top_point - current
    direction_vector /= np.linalg.norm(direction_vector)  # Normalize the direction vector

    # Calculate the dot product of each point with the direction vector
    cs = cosine_similarity(direction_vector.reshape(1, -1), above_points - current.reshape(1, -1))[0]
    # Find points that are above the branch point in the direction of the top point
    above_points = above_points[cs > threshold]
    # print(f"Above points count: {len(above_points)}")
    if len(above_points) == 0:
        return above_points
    distances = np.linalg.norm(above_points - current, ord=2)
    closest_indices = np.argsort(distances)
    above_points = above_points[closest_indices]
    return above_points

def current_to_top_vector(current, centriods, top_point):
    above_points = find_above_points_in_the_dir_to_top(current, centriods, top_point)
    if len(above_points) == 0:
        return top_point - current
    else:
        dirs = above_points - current
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8  # Normalize each direction vector
    mean_vector = np.mean(dirs, axis=0)
    direction_vector = mean_vector / np.linalg.norm(mean_vector)  # Normalize the mean vector
    return direction_vector

def find_stempath_from_root_to_top(points, stem_centroids, top_point, stop_point, root_point):
    # step by step
    kdtree = KDTree(points, metric='euclidean')
    stem_ids = []
    is_seg = np.zeros(len(points), dtype=bool)
    pre_dir = np.array([0.0, 0.0, 0.0])
    iter_num = 0
    s_r = 0.015
    seed_pt = np.array(root_point, dtype=float)
    seed_pt_list = []
    while True:
        if seed_pt[2] > stop_point[2]:
            break
        adj = kdtree.query_radius(seed_pt.reshape(1, -1), r=s_r)[0]
        nn_adj = adj[~is_seg[adj]]
        above_dir = current_to_top_vector(seed_pt, stem_centroids, top_point)
        if len(nn_adj) == 0:
            dir_vec = above_dir
            seed_pt = seed_pt + s_r * dir_vec
            # seed_pt_list.append(seed_pt.copy())
            continue
        nn_pts = points[nn_adj]
        dirs = nn_pts - seed_pt
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs = dirs / (norms + 1e-8)  # avoid divide-by-zero

        med_x = np.median(dirs[:, 0])
        med_y = np.median(dirs[:, 1])
        med_z = np.median(dirs[:, 2])
        dir_vec = np.array([med_x, med_y, med_z])

        if np.linalg.norm(dir_vec) != 0:
            dir_vec = dir_vec / np.linalg.norm(dir_vec)

        # Smooth the direction with previous direction
        dir_vec = 0.1 * dir_vec + 0.6 * pre_dir + 0.3 * above_dir
        if np.linalg.norm(dir_vec) != 0:
            dir_vec = dir_vec / np.linalg.norm(dir_vec)

        pre_dir = dir_vec.copy()

        # Move seed point along the direction
        seed_pt = seed_pt + s_r * dir_vec

        # Update stem points
        stem_ids.extend(nn_adj.tolist())
        is_seg[nn_adj] = True
        seed_pt_list.append(seed_pt.copy())
        iter_num += 1
        if iter_num > 200:
            break
    return stem_ids, seed_pt_list

def create_skeleton_graph(stem_key_points, points, colors, root_point):
    skel_graph = nx.Graph()

    # # Add nodes to the graph
    # for i, point in enumerate(stem_key_points):
    #     skel_graph.add_node(tuple(point), label=0)

    # # Add edges to connect nodes sequentially
    # for i in range(len(stem_key_points) - 1):
    #     skel_graph.add_edge(tuple(stem_key_points[i]), tuple(stem_key_points[i+1]))

    # iterate leaves
    leaf_labels = np.unique(colors[:,0])
    connected_stem_points = set()
    for leaf_label in leaf_labels:
        if leaf_label == 0:
            continue
        leaf_points = points[colors[:, 0] == leaf_label]
        leaf_kdtree = KDTree(leaf_points)
        dists, indexes = leaf_kdtree.query(stem_key_points, k=10)
        min_idx = np.argmin(np.mean(dists, axis=1))
        leaf_branch_point = np.mean(leaf_points[indexes[min_idx]],axis=0)

        # leaf_points = pcd_utils.down_sample(leaf_points, 2000)
        leaf_points = np.vstack([leaf_points, leaf_branch_point])
        leaf_pcd = o3d.geometry.PointCloud()
        leaf_pcd.points = o3d.utility.Vector3dVector(leaf_points)

        g = pcd_utils.knn_graph(leaf_pcd, 5)
        pcd_utils.connect_graph(g, leaf_pcd, len(leaf_points)-1)
        # cluster_values, cluster_sets, cluster_centers, dist_bin_dict = \
        #     pcd_utils.distance_to_root_clusters(g, len(leaf_points)-1, leaf_pcd, 0.02,)
        # leaf_branch_key_point = np.mean(stem_bracnh_points, axis=0)
        # skel_graph.add_node(tuple(leaf_branch_point), label=leaf_label)

        shortest_path = pcd_utils.shortest_path_root_to_tip(g, len(leaf_points)-1, 0.01)
        
        # Find the stem point on the tangle direction of leaf path
        stem_branch_points = pcd_utils.find_closest_point_to_tangent_line(shortest_path, stem_key_points, k=3)
        connected_stem_points.add(tuple(stem_branch_points))
        leaf_key_points = pcd_utils.detect_key_points_by_rdp(shortest_path, 0.005)
        
        for i, c in enumerate(leaf_key_points):
            skel_graph.add_node(tuple(c), label=leaf_label)
            if i == 0:
                skel_graph.add_edge(tuple(c), tuple(stem_branch_points)) 
            else:
                skel_graph.add_edge(tuple(c), tuple(leaf_key_points[i-1]))
    # Sort connected_stem_points by their order in stem_key_points
    sorted_connected_stem_points = sorted(
        connected_stem_points, 
        key=lambda point: np.argmin(np.linalg.norm(stem_key_points - point, axis=1))
    )
    
    # Add edges between sorted connected stem points and leaf key points
    skel_graph.add_node(tuple(root_point), label=0)

    for i, c in enumerate(sorted_connected_stem_points):
        skel_graph.add_node(tuple(c), label=0)
        if i == 0:
            skel_graph.add_edge(tuple(c), tuple(root_point)) 
        else:
            skel_graph.add_edge(tuple(c), tuple(sorted_connected_stem_points[i-1]))
    if tuple(stem_key_points[-1]) not in sorted_connected_stem_points:
        skel_graph.add_node(tuple(stem_key_points[-1]), label=0)
        skel_graph.add_edge(tuple(stem_key_points[-1]), tuple(sorted_connected_stem_points[-1]))
    return skel_graph

def get_root_points(points, stem_idx = None):
    """
    Get the root node of the first stem subgraph.
    :param subgraphs: List of subgraphs.
    :param stem_graphs_idx: Indices of stem graphs.
    :return: Coordinates of the root node.
    """
    if stem_idx is not None:
        candidate_pts = points[stem_idx,:]
    else:
        candidate_pts = points
    # Select 10% of points with the minimal z values
    num_points = max(1, int(0.5 * len(candidate_pts)))
    centroid = np.mean(candidate_pts[np.argsort(candidate_pts[:, 2])[:num_points],:], axis=0)
    # Find the 10 points with the minimal z values
    min_z = np.min(candidate_pts[:, 2])
    min_z_indices = np.argwhere(candidate_pts[:, 2] == min_z).ravel()
    min_z_points = candidate_pts[min_z_indices]
    root_idx = np.argmin(np.linalg.norm(min_z_points[:,:2] - centroid[:2], axis=1))
    root_point = min_z_points[root_idx]
    root_index = stem_idx[min_z_indices[root_idx]]
    return root_point, root_index

def calculate_stem_radius(stem_key_points, stem_points):
    radius_list = []
    for s in stem_key_points:
        z_min, z_max = s[2] - 0.04, s[2] + 0.04
        stem_points_in_range = stem_points[(stem_points[:, 2] >= z_min) & (stem_points[:, 2] <= z_max)]
        if len(stem_points_in_range) > 0:
            z_value = np.unique(stem_points_in_range[:, 2])            
            raidus = []
            for z in z_value:
                z_stem_points = stem_points_in_range[stem_points_in_range[:, 2] == z]
                radius_xy = np.max(z_stem_points[:, :2], axis=0) - np.min(z_stem_points[:, :2], axis=0)
                raidus.append(np.mean(radius_xy))       
            # print(f"Radius: {raidus}")     
            radius_list.append(np.mean(raidus))
        else:
            radius_list.append(radius_list[-1])
    return radius_list

def get_stem_centriod_raidus(voxel_file, outlier_remove=False):
    voxel = np.load(voxel_file)
    indx = np.argwhere(voxel >= 0)
    points = (indx ) / 64
    if outlier_remove:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        points = points[ind]
        indx = indx[ind]
    labels = voxel[indx[:, 0], indx[:, 1], indx[:, 2]]
    binary_labels = np.where(labels == 0, 0, 1)
    stem_idx = np.where(binary_labels == 0)[0]

    # o3d.visualization.draw_geometries([clean_pcd])
    root_point, root_index = get_root_points(points, stem_idx)
    top_index_stem = np.argmax(points[stem_idx][:, 2])
    top_point = points[stem_idx][top_index_stem]

    # 2. Find key point on stem
    root_index_stem = np.where(root_index == stem_idx)[0][0]
    stem_points = points[stem_idx]
    stem_centroids = find_key_point_in_stem(stem_points, root_index_stem)
    stem_centroids = np.vstack((root_point,stem_centroids, top_point))

    smoothed_stem_key_points = pcd_utils.smooth_path_savgoal(stem_centroids, window_size=13)
    # Clip the z-axis of smoothed_stem_key_points
    z_min, z_max = root_point[2], top_point[2]  # Define the range for clipping
    smoothed_stem_key_points[:, 2] = np.clip(smoothed_stem_key_points[:, 2], z_min, z_max)
    radius = calculate_stem_radius(smoothed_stem_key_points, stem_points)
    return smoothed_stem_key_points, radius


def main():
    voxel_path = r"C:\Users\Jacob\Desktop\Skeleton\maize_voxel"
    voxe_files = [os.path.join(voxel_path, f) for f in os.listdir(voxel_path) if f.endswith('.npy')]
    for i, voxl_path in enumerate(voxe_files):
        # if i != 216:
        #     continue
        print(f"Processing {i+1}/{len(voxe_files)}: {os.path.basename(voxl_path)}")
        smoothed_stem_key_points, radius = get_stem_centriod_raidus(voxl_path)
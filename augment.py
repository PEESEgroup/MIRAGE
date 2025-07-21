import os
from geomdl import fitting
from geomdl import BSpline, utilities
from geomdl.visualization import VisMPL
from scipy.linalg import lstsq
from geomdl.helpers import basis_function, find_span_linear
from geomdl import tessellate
from geomdl import NURBS
from scipy.spatial.distance import directed_hausdorff
import numpy as np
import trimesh
import open3d as o3d
import numpy as np

from scipy.signal import find_peaks
import bspline_utils
from leaf_reconstruction import get_leaf_direction, leaf_base_v, match_leaf_at_u

def compute_tangent_planes_at_control_points(control_points):
    """
    Computes tangent planes at each interior control point using finite differences.

    Args:
        control_points (np.ndarray): shape (M, N, 3)

    Returns:
        List of planes, each as dict:
            {
                'i': row index,
                'j': col index,
                'point': np.array([x, y, z]),
                'normal': unit normal vector,
                'd': plane offset (for ax + by + cz + d = 0)
            }
    """
    M, N, _ = control_points.shape
    planes = {}

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            pt = control_points[i, j]
            t_u = control_points[i + 1, j] - control_points[i - 1, j]
            t_v = control_points[i, j + 1] - control_points[i, j - 1]
            normal = np.cross(t_u, t_v)
            norm = np.linalg.norm(normal)
            if norm == 0:
                continue  # skip degenerate cases
            normal /= norm
            d = -np.dot(normal, pt)

            planes[(i, j)] = {
                'point': pt,
                'normal': normal,
                'd': d
            }

    return planes


def compute_vertical_plane_through_two_points(p1, p2, pca_normal):
    """
    Compute a vertical plane (orthogonal to PCA plane) that passes through p1 and p2.
    
    Args:
        p1, p2: (3,) numpy arrays (points in 3D)
        pca_normal: (3,) unit vector normal to the PCA-fitted plane
    
    Returns:
        normal: unit normal vector of the vertical plane
        d: offset in ax + by + cz + d = 0
        point: one point on the plane (p1)
    """
    v = p2 - p1
    n_vertical = np.cross(pca_normal, v)
    norm = np.linalg.norm(n_vertical)
    if norm == 0:
        raise ValueError("Points are colinear with PCA normal — can't define vertical plane")
    n_vertical /= norm
    d = -np.dot(n_vertical, p1)
    return n_vertical, d, p1

def fit_plane_to_points_by_pca(control_points):
    M, N, _ = control_points.shape
    points = control_points.reshape(-1, 3)
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[-1]  # normal to the best-fitting plane (least variance)
    # Step 4: Compute d in plane equation ax + by + cz + d = 0
    d = -np.dot(normal, centroid)

    return normal, d, centroid

def normalize(v):
    return v / np.linalg.norm(v)

def project_to_plane(p, normal, d):
    # Project point p onto plane defined by normal and d
    n = normalize(normal)
    distance = np.dot(n, p) + d
    return p - distance * n

def smooth_flatten_to_plane(control_points, anchor_mask, plane_normal, plane_d, alpha=0.5):
    """
    Smoothly flattens a B-spline surface to an arbitrary plane.
    
    Args:
        control_points (np.ndarray): shape (M, N, 3)
        plane_normal (np.ndarray): shape (3,)
        plane_d (float): plane offset in equation a·x + b·y + c·z + d = 0
        alpha (float or np.ndarray): 0~1 scalar or array of shape (M, N)
                                     blending weight; 1 = full flattening
    
    Returns:
        np.ndarray: new control points (M, N, 3)
    """
    M, N, _ = control_points.shape
    deformed_cp = np.copy(control_points)

    for i in range(M):
        for j in range(N):
            if not anchor_mask[i, j]:
                p = control_points[i, j]
                proj = project_to_plane(p, plane_normal, plane_d)
                w = alpha if isinstance(alpha, float) else alpha[i, j]
                deformed_cp[i, j] = (1 - w) * p + w * proj

    return deformed_cp


def create_plane(center, normal, size=0.2, color=[1, 0, 0]):

    normal = normal / np.linalg.norm(normal)

    if abs(normal[2]) < 0.9:
        tangent = np.cross(normal, [0, 0, 1])
    else:
        tangent = np.cross(normal, [0, 1, 0])
    tangent = tangent / np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)

    corners = []
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            corner = center + size * dx * tangent + size * dy * bitangent
            corners.append(corner)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(corners)
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [1, 3, 2]])
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh

def compute_3d_curvature(points):
    d1 = np.gradient(points, axis=0)
    d2 = np.gradient(d1, axis=0)
    cross = np.cross(d1, d2)
    num = np.linalg.norm(cross, axis=1)
    denom = np.linalg.norm(d1, axis=1)**3 + 1e-8
    return num / denom

def find_peaks_in_curvature(curvature, topk=2, height=0.01, distance=10, edge_ratio=0.1, dominance_ratio=0.3, single_peak_num=20):
    peaks, properties = find_peaks(curvature, height=height, distance=distance)
    n = len(curvature)
    edge_limit = int(edge_ratio * n)
    keep_mask = (peaks >= edge_limit) & (peaks < n - edge_limit)
    filtered_peaks = peaks[keep_mask]
    filtered_properties = {k: np.array(v)[keep_mask] for k, v in properties.items()}
    if len(filtered_peaks) == 0:
        return [20, n-20] , properties

    if len(filtered_peaks) == 1:
        peak_indices = [max(10,filtered_peaks[0] - single_peak_num), min(filtered_peaks[0] + single_peak_num, n-10)]
        return peak_indices, properties

    heights = filtered_properties['peak_heights']
    sorted_indices = np.argsort(-heights)[:topk]
    ratio = heights[sorted_indices[1]] / heights[sorted_indices[0]]

    if ratio < dominance_ratio:
        peak = filtered_peaks[[sorted_indices[0]]][0]
        peak_indices =  [max(10, peak - single_peak_num), min(peak + single_peak_num, n-10)]
        return peak_indices, properties
    else:
        return filtered_peaks[sorted(sorted_indices)], properties

def pca_endpoint_angle(points, arch_ids):
    from sklearn.decomposition import PCA

    pca_start = PCA(n_components=1)
    start_end = max(0, arch_ids[0]-100)
    pca_start.fit(points[start_end:arch_ids[0]])
    d_start = points[(start_end+ arch_ids[0]) // 2]
    center_vector = points[arch_ids[0]] - points[0]
    dir_start = pca_start.components_[0]
    if np.dot(dir_start, center_vector) < 0:
        dir_start = -dir_start 

    pca_end = PCA(n_components=1)
    end_end = min(arch_ids[1]+100, len(points)-1)
    pca_end.fit(points[arch_ids[1]:end_end])
    dir_end = pca_end.components_[0]
    d_end = points[(arch_ids[1] + end_end) // 2]
    center_vector =  points[arch_ids[1]] -points[-1]
    if np.dot(dir_end, center_vector) < 0:
        dir_end = -dir_end 

    dot = np.clip(np.dot(dir_start, dir_end), -1.0, 1.0)
    angle_rad = np.arccos(dot)
    angle_deg = np.degrees(angle_rad)

    return angle_deg, dir_start, d_start, dir_end, d_end

from scipy.spatial.transform import Rotation as R
from bspline_utils import get_approx_mesh
from stem_reconstruction import get_stem_centerline_direction_at_u
def adjust_leaf_loc_on_stem(leaf, stem_surf, stem_centerline_curve, new_uv):
    leaf_ctrlpts = np.array(leaf.ctrlpts)
    matched_v = match_leaf_at_u(leaf, stem_surf, new_uv[0], stem_centerline_curve)
    leaf_tip_np = np.array(leaf.evaluate_single((0.5, leaf_base_v(leaf)))) 

    new_stem_points = np.array(stem_surf.evaluate_single((new_uv[0]/39, matched_v/39)))
    move_dir = new_stem_points - leaf_tip_np
    leaf_ctrlpts += move_dir

    rotation_degree = (new_uv[1] - matched_v) / 40 * 2 * np.pi
    # Compute tangent vector of stem_centerline_curve at new_uv[0]
    delta = 1e-4
    centerline_pts = np.array(stem_centerline_curve.evaluate_single(new_uv[0]/39))
    centerline_dir = get_stem_centerline_direction_at_u(stem_centerline_curve, new_uv[0], resolution=40)

    # Translate ctrlpts so base_point is at origin
    ctrlpts_centered = leaf_ctrlpts - centerline_pts

    # Normalize centerline_dir
    axis = centerline_dir / np.linalg.norm(centerline_dir)
    rot = R.from_rotvec(axis * rotation_degree)
    rotated_ctrlpts = rot.apply(ctrlpts_centered)

    # Translate back
    r_leaf_ctrlpts = rotated_ctrlpts + centerline_pts
    _, surf = get_approx_mesh(r_leaf_ctrlpts, 2,3, 20)
    return surf

def shrink_leaf_by_mask(leaf, mask_indices, scale_factor=0.5): # at U
    anchor_mask = np.zeros((3, 8), dtype=bool)
    anchor_mask[:, mask_indices[0]:mask_indices[1]] = True   # anchor left column
    leaf_ctrlpts = np.array(leaf.ctrlpts)
    leaf_ctrlpts = leaf_ctrlpts.reshape(3, 8, 3)  # Ensure shape is (3, 8, 3)
    if leaf_base_v(leaf) == 1.0:
        leaf_ctrlpts = np.flip(leaf_ctrlpts, axis=1)
    deformed_cp = np.copy(leaf_ctrlpts)

    for i in range(leaf_ctrlpts.shape[0]):
        for j in range(leaf_ctrlpts.shape[1]):
            if not anchor_mask[i, j]:
                dis = leaf_ctrlpts[i, j] - leaf_ctrlpts[1, j]
                deformed_cp[i, j] = leaf_ctrlpts[1,j] + dis * scale_factor

    _, surf = get_approx_mesh(deformed_cp, 2,3, 20)
    return surf

def scale_leaf_by_mask(leaf, mask_indices, scale_factor=1.5, mask_for_start = True, keep_z = False): # at V
    anchor_mask = np.zeros((3, 8), dtype=bool)
    anchor_mask[:, mask_indices[0]:mask_indices[1]] = True   # anchor left column
    leaf_ctrlpts = np.array(leaf.ctrlpts)
    leaf_ctrlpts = leaf_ctrlpts.reshape(3, 8, 3)  # Ensure shape is (3, 8, 3)
    if leaf_base_v(leaf) == 1.0:
        leaf_ctrlpts = np.flip(leaf_ctrlpts, axis=1)
    deformed_cp = np.copy(leaf_ctrlpts)
    if mask_for_start:
        for i in range(leaf_ctrlpts.shape[0]):
            for j in range(leaf_ctrlpts.shape[1]):
                if not anchor_mask[i, j]:
                    if not keep_z:
                        dis = leaf_ctrlpts[i, j] - leaf_ctrlpts[i, mask_indices[1]]
                        deformed_cp[i, j] = leaf_ctrlpts[i , mask_indices[1]] + dis * scale_factor
                    else:
                        dis = leaf_ctrlpts[i, j][:2] - leaf_ctrlpts[i, mask_indices[1]][:2]
                        deformed_cp[i, j][:2] = leaf_ctrlpts[i , mask_indices[1]][:2] + dis * scale_factor
                        deformed_cp[i, j][2] = leaf_ctrlpts[i, j][2]
    else:
        for i in range(leaf_ctrlpts.shape[0]):
            for j in range(leaf_ctrlpts.shape[1]):
                if not anchor_mask[i, j]:
                    dis = leaf_ctrlpts[i, j] - leaf_ctrlpts[i, mask_indices[0]]
                    deformed_cp[i, j] = leaf_ctrlpts[i , mask_indices[0]] + dis * scale_factor

    _, surf = get_approx_mesh(deformed_cp, 2,3, 20)
    return surf

def compute_rotation(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    axis = np.cross(v1, v2)
    if np.linalg.norm(axis) < 1e-8:
        angle = 0.0 if np.dot(v1, v2) > 0 else np.pi
        axis = np.array([1, 0, 0]) 
    else:
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return  angle, axis

def rotate_with_mask(control_points, anchor_mask, rotation, center):
    M, N, _ = control_points.shape
    center = control_points[1,center,:]
    deformed_cp = np.copy(control_points)

    for i in range(M):
        for j in range(N):
            if not anchor_mask[i, j]:
                deformed_cp[i, j] = rotation.apply(control_points[i, j]- center) + center
    return deformed_cp

def get_anchor_mask(peak_indices, given_mask = '0'):
    if isinstance(given_mask, str):
        col_num = round(peak_indices[int(given_mask)]*8/100) 
    else:        
        col_num = given_mask
    print(f"Anchor column number: {col_num}")
    anchor_mask = np.zeros((3, 8), dtype=bool)
    anchor_mask[:, :col_num] = True   # anchor left column
    return anchor_mask, col_num

def adjust_leaf_curvature(leaf, single_peak_num = 10, given_mask='0', curvature_factor=1.0, given_angle=None):
    leaf_ctrlpts = np.array(leaf.ctrlpts)
    leaf_ctrlpts = leaf_ctrlpts.reshape(3, 8, 3)
    n_points = 100
    if leaf_base_v(leaf) == 1.0:
        leaf_ctrlpts = np.flip(leaf_ctrlpts, axis=1)
        us = np.linspace(1, 0, n_points)
    else:
        us = np.linspace(0, 1, n_points)

    points = np.array([leaf.evaluate_single((0.5, u)) for u in us])
    curvature = compute_3d_curvature(points)
    peak_indices, properties = find_peaks_in_curvature(curvature, 2, height=0.05, distance=20, single_peak_num=single_peak_num)
    angle_deg, dir_start, p0 , dir_end, p1  = pca_endpoint_angle(points, peak_indices)

    angle, axis = compute_rotation(dir_start, dir_end)
    if given_angle is not None:
        angle = given_angle
    else:
        angle = np.pi - angle
        angle *= curvature_factor  # Adjust curvature by factor

    rotation = R.from_rotvec(axis * angle)  # Rotate by angle around axis
    anchor_mask, col_num = get_anchor_mask(peak_indices, given_mask)
    roated_points = rotate_with_mask(leaf_ctrlpts, anchor_mask, rotation, col_num)
    _, surf = get_approx_mesh(roated_points, 2,3, 20)
    return surf

from numpy import dot
from numpy.linalg import norm

def adjust_leaf_stem_angle(leaf_surf, leaf_location, stem_centerline_curve, angle):
    ctrlpts = np.array(leaf_surf.ctrlpts)
    leaf_dir = get_leaf_direction(leaf_surf)
    stem_dir = get_stem_centerline_direction_at_u(stem_centerline_curve, leaf_location[0])
    # angle = np.arccos(np.clip(dot(leaf_dir, stem_dir) / (norm(leaf_dir) * norm(stem_dir)), -1.0, 1.0))
    ori_angle, axis = compute_rotation(leaf_dir, stem_dir)
    print(f"Leaf direction: {leaf_dir}, Stem direction: {stem_dir}, Angle (radians): {ori_angle}")
    rotation = R.from_rotvec((ori_angle- angle) * axis)  # Rotate by angle around axis
    leaf_tip = np.array(leaf_surf.evaluate_single((0.5, leaf_base_v(leaf_surf))))  # Assuming leaf_base_v is the base v value for the leaf tip
    ctrlpts = rotation.apply(ctrlpts- leaf_tip) + leaf_tip
    _, surf = get_approx_mesh(ctrlpts, 2,3, 20)
    return surf
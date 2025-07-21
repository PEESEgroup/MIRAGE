import numpy as np
from geomdl import BSpline, NURBS
from geomdl.fitting import approximate_curve
from geomdl.visualization import VisVTK
from geomdl import utilities
import open3d as o3d
from preprocess_voxel import get_stem_centriod_raidus
import vis
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from geomdl import exchange
from geomdl import tessellate
import trimesh
import os
def moving_average(x, w=3):
    return np.convolve(x, np.ones(w)/w, mode='valid')

# -----------------------------
# 3. Generate cross-section
# -----------------------------
def generate_circle_points(radius=1.0, resolution=8):
    angles = np.linspace(0, 2 * np.pi, resolution, endpoint=True)
    return np.array([[radius * np.cos(a), radius * np.sin(a), 0.0] for a in angles])

# -----------------------------
# 4. Create local frames and sweep
# -----------------------------
def frenet_frame(tangent):
    # Gram-Schmidt to get normal and binormal
    tangent = tangent / np.linalg.norm(tangent)
    arbitrary = np.array([0, 0, 1]) if abs(tangent[2]) < 0.9 else np.array([1, 0, 0])
    normal = np.cross(tangent, arbitrary)
    normal = normal / np.linalg.norm(normal)
    binormal = np.cross(tangent, normal)
    binormal = binormal / np.linalg.norm(binormal)
    return tangent, normal, binormal

def create_stem(height, radius):
    centerline_pts = np.array([[np.random.uniform(0.495 ,0.505), np.random.uniform(0.495 ,0.505), z] for z in np.linspace(0, height, num=20)])
    radius = np.array([np.random.uniform(radius * 0.7 ,radius *1.3) for _ in range(len(centerline_pts))])
    radius = moving_average(radius, w=13)

    # Create an interpolation function for the radius
    radius_interp = interp1d(np.arange(len(radius))/ (len(radius)-1), radius, kind='cubic')

    # -----------------------------
    centerline_pts = list(centerline_pts)
    # Approximate as a NURBS curve
    curve = approximate_curve(centerline_pts, degree=3, ctrlpts_size=8)

    # -----------------------------
    # 2. Parameters
    # -----------------------------

    circle_resolution = 20
    sweep_samples = 20

    sweep_points = []
    for i in range(sweep_samples):
        u = i / (sweep_samples - 1)
        pt = np.array(curve.evaluate_single(u))
        deriv = np.array(curve.derivatives(u, order=1)[1])
        tangent, normal, binormal = frenet_frame(deriv)

        R = np.column_stack([normal, binormal, tangent])
        circle_radius = radius_interp(u)  # Get the radius at this point
        circle_radius *= 0.75
        circle = generate_circle_points(radius=circle_radius, resolution=circle_resolution)
        circle_transformed = pt + (R @ circle.T).T
        sweep_points.append(circle_transformed.tolist())
    # -----------------------------
    # 5. Create surface
    # -----------------------------
    surf = BSpline.Surface()
    surf.degree_u = 3  # around cross-section
    surf.degree_v = 3  # along sweep

    # Flatten control points
    ctrlpts2d = sweep_points  # shape: [sweep_samples][circle_resolution][3]

    # geomdl expects 1D list of control points
    flattened_ctrlpts = [pt for row in ctrlpts2d for pt in row]
    surf.set_ctrlpts(flattened_ctrlpts, circle_resolution, sweep_samples)
    surf.knotvector_u = utilities.generate_knot_vector(surf.degree_u, circle_resolution)
    surf.knotvector_v = utilities.generate_knot_vector(surf.degree_v, sweep_samples)
    # Auto-generate knot vectors
    surf.delta = 0.025
    size = 40
    surf.evaluate()
    # exchange.export_obj(surf, 'stem_obj' + f"{os.path.basename(voxels_path).split('.')[0]}.obj")
    return surf, curve


def get_stem_obj_from_voxel(voxels_path, clean_outiers=False):

    centerline_pts, radius = get_stem_centriod_raidus(voxels_path, clean_outiers)
    print(f"Centerline points: {centerline_pts}")
    # Smooth and interpolate the radius array

    radius = moving_average(radius, w=11)

    # Create an interpolation function for the radius
    radius_interp = interp1d(np.arange(len(radius))/ (len(radius)-1), radius, kind='cubic')

    # -----------------------------
    centerline_pts = list(centerline_pts)
    # Approximate as a NURBS curve
    curve = approximate_curve(centerline_pts, degree=3, ctrlpts_size=8)

    # -----------------------------
    # 2. Parameters
    # -----------------------------

    circle_resolution = 20
    sweep_samples = 20

    sweep_points = []
    for i in range(sweep_samples):
        u = i / (sweep_samples - 1)
        pt = np.array(curve.evaluate_single(u))
        deriv = np.array(curve.derivatives(u, order=1)[1])
        tangent, normal, binormal = frenet_frame(deriv)

        R = np.column_stack([normal, binormal, tangent])
        circle_radius = radius_interp(u)  # Get the radius at this point
        circle_radius *= 0.75
        circle = generate_circle_points(radius=circle_radius, resolution=circle_resolution)
        circle_transformed = pt + (R @ circle.T).T
        sweep_points.append(circle_transformed.tolist())
    # -----------------------------
    # 5. Create surface
    # -----------------------------
    surf = BSpline.Surface()
    surf.degree_u = 3  # around cross-section
    surf.degree_v = 3  # along sweep

    # Flatten control points
    ctrlpts2d = sweep_points  # shape: [sweep_samples][circle_resolution][3]

    # geomdl expects 1D list of control points
    flattened_ctrlpts = [pt for row in ctrlpts2d for pt in row]
    surf.set_ctrlpts(flattened_ctrlpts, circle_resolution, sweep_samples)
    surf.knotvector_u = utilities.generate_knot_vector(surf.degree_u, circle_resolution)
    surf.knotvector_v = utilities.generate_knot_vector(surf.degree_v, sweep_samples)
    # Auto-generate knot vectors
    surf.delta = 0.025
    size = 40
    surf.evaluate()
    # exchange.export_obj(surf, 'stem_obj' + f"{os.path.basename(voxels_path).split('.')[0]}.obj")
    return surf, curve
    # renderer_vertices = np.array(surf.evalpts)
    # evalverices = list(surf.evalpts)
    # tessellator = tessellate.TrimTessellate()
    # tessellator.tessellate(evalverices, size_u=size, size_v=size)
    # renderer_vertices, renderer_face = tessellator.vertices, tessellator.faces
    # renderer_vertices = np.array([r.data for r in renderer_vertices])
    # renderer_face = np.array([r.data for r in renderer_face])

    # # 4. Export as STL using trimesh
    # mesh = trimesh.Trimesh(vertices=renderer_vertices, faces=renderer_face)
    # mesh.show()
    # mesh.export('stem_001.stl')
    # -----------------------------
    # 6. Visualize
    # -----------------------------

    # exchange.export_obj(surf, "stem_001.obj")

def get_stem_centerline_direction_at_u(curve, u, resolution=40):
    resolution = resolution -1
    centerline_pts =  np.array(curve.evaluate_single(u/resolution))
    if u == resolution:
        centerline_dir = centerline_pts - np.array(curve.evaluate_single(u/resolution - 1/resolution))
    else:
        centerline_dir = np.array(curve.evaluate_single(u/resolution + 1/resolution)) - centerline_pts
    return centerline_dir
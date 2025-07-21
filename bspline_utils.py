import os
from geomdl import fitting
from geomdl import BSpline, utilities
from geomdl.visualization import VisMPL
from scipy.linalg import lstsq
from geomdl.helpers import basis_function, find_span_linear
from geomdl import tessellate
from geomdl import NURBS
import numpy as np
import trimesh
import open3d as o3d
import matplotlib.pyplot as plt

def get_approx_mesh(control_points, degree_u, degree_v, size):
    if len(control_points.shape) == 2:
        control_points = control_points.reshape(3, 8, 3)
    surf = NURBS.Surface()
    surf.degree_u = degree_u
    surf.degree_v = degree_v
    surf.ctrlpts2d = [[control_points[j,i,:].tolist() + [1] for i in range(8)] for j in range(3)]
    surf.knotvector_u = utilities.generate_knot_vector(degree_u, control_points.shape[0])
    surf.knotvector_v = utilities.generate_knot_vector(degree_v, control_points.shape[1])
    # surf.sample_size = 20
    surf.delta = 0.05
    surf.evaluate()
    renderer_vertices = np.array(surf.evalpts)
    evalverices = list(surf.evalpts)
    tsl = tessellate.TrimTessellate()
    tsl.tessellate(evalverices, size_u=size, size_v=size)
    renderer_vertices, renderer_face = tsl.vertices, tsl.faces
    renderer_vertices = np.array([r.data for r in renderer_vertices])
    renderer_face = np.array([r.data for r in renderer_face])
    renderer_mesh = trimesh.Trimesh(vertices=renderer_vertices, faces=renderer_face)
    # renderer_mesh.visual.uv =  np.stack([np.arange(400) % 20 / (20 - 1), np.floor(np.arange(400) / 20) / (20 - 1)], axis=1)
    return renderer_mesh, surf

def get_mesh_from_surfaces(surf_list):
    mesh_list = []
    for surf in surf_list:
        surf.evaluate()
        # print(f"Evaluating surface with delta {surf.delta}")
        size_u = int(1 / surf.delta[0])
        size_v = int(1 / surf.delta[1])
        renderer_vertices = np.array(surf.evalpts)
        evalverices = list(surf.evalpts)
        tsl = tessellate.TrimTessellate()
        tsl.tessellate(evalverices, size_u=size_u, size_v=size_v)
        renderer_vertices, renderer_face = tsl.vertices, tsl.faces
        renderer_vertices = np.array([r.data for r in renderer_vertices])
        renderer_face = np.array([r.data for r in renderer_face])
        renderer_mesh = trimesh.Trimesh(vertices=renderer_vertices, faces=renderer_face)
        # o3d_mesh = o3d.geometry.TriangleMesh()
        # # Set vertices and faces
        # o3d_mesh.vertices = o3d.utility.Vector3dVector(renderer_mesh.vertices)
        # o3d_mesh.triangles = o3d.utility.Vector3iVector(renderer_mesh.faces)
        mesh_list.append(renderer_mesh)

    return mesh_list


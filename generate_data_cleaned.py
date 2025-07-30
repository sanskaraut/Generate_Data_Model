import open3d as o3d
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import exposure
import random
import json
import os
import networkx as nx
from scipy.spatial import cKDTree
import cv2
import logging

# === SETUP LOGGER ===
logging.basicConfig(
    filename="generate.log",
    filemode="a",
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)

# === LOAD BASE BONE MODEL ===
model_path = "/home/user119/Generate_Data_Model/models/original/forarmbone.ply"
mesh = o3d.io.read_triangle_mesh(model_path)
if not mesh.has_triangles():
    logging.error(f"❌ Failed to load mesh from: {model_path}")
    raise ValueError(f"❌ Failed to load mesh from: {model_path}")
logging.info("✅ Loaded original bone mesh.")

# === PREPROCESSING ===
mesh.remove_duplicated_vertices()
mesh.remove_degenerate_triangles()
mesh.remove_duplicated_triangles()
mesh.remove_non_manifold_edges()
mesh.remove_unreferenced_vertices()
bbox_check = mesh.get_max_bound() - mesh.get_min_bound()
if np.any(bbox_check == 0):
    logging.error("❌ Invalid mesh dimensions (zero bounding box axis)")
    raise ValueError("❌ Invalid mesh dimensions (zero bounding box axis)")
mesh.scale(1 / np.max(bbox_check), center=mesh.get_center())
mesh.translate(-mesh.get_center())
mesh.compute_vertex_normals()

# === BOUNDING BOX & SPLIT ===
bbox = mesh.get_axis_aligned_bounding_box()
size = bbox.get_max_bound() - bbox.get_min_bound()
plane_height, plane_depth = size[1], size[2]
if plane_height <= 0 or plane_depth <= 0:
    logging.error("❌ Invalid bounding box size: height or depth is zero")
    raise ValueError("❌ Invalid bounding box size: height or depth is zero")

vertical_plane = o3d.geometry.TriangleMesh.create_box(width=0.001, height=plane_height, depth=plane_depth)
vertical_plane.translate((-0.0005, -plane_height / 2, -plane_depth / 2))
center_x = (bbox.get_min_bound()[0] + bbox.get_max_bound()[0]) / 2
vertical_plane.translate((center_x, 0, 0))
angle_from_x = 94.11222884471846
tilt_angle = angle_from_x - 90
angle_rad = np.deg2rad(-tilt_angle)
R = vertical_plane.get_rotation_matrix_from_axis_angle([0, 0, angle_rad])
vertical_plane.rotate(R, center=vertical_plane.get_center())
vertical_plane.translate((-0.015, 0, 0))

plane_normal = R @ np.array([1.0, 0.0, 0.0])
plane_normal /= np.linalg.norm(plane_normal)
plane_center = vertical_plane.get_center()
points = np.asarray(mesh.vertices)
signed_distances = np.dot(points - plane_center, plane_normal)
mask_above = signed_distances > 0
mask_below = signed_distances <= 0
mesh_ulna = mesh.select_by_index(np.where(mask_above)[0].tolist())
mesh_radius = mesh.select_by_index(np.where(mask_below)[0].tolist())
meshs = [mesh_ulna, mesh_radius]
logging.info("✅ Split into ulna and radius.")

# === UTILITY FUNCTIONS ===

def create_angle_mesh(mesh, angles, split_ratio):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    min_y, max_y = vertices[:, 1].min(), vertices[:, 1].max()
    mid_y = min_y + (max_y - min_y) * split_ratio
    top_mask = vertices[:, 1] >= mid_y
    bottom_mask = ~top_mask

    def rotate_part(mask, angle_deg, center_y):
        indices = np.where(mask)[0]
        sub_vertices = np.copy(vertices[indices])
        index_map = -np.ones(len(vertices), dtype=int)
        index_map[indices] = np.arange(len(indices))
        tri_mask = np.all(mask[triangles], axis=1)
        sub_triangles = triangles[tri_mask]
        mapped_triangles = index_map[sub_triangles]
        angle_rad = np.radians(angle_deg)
        R = mesh.get_rotation_matrix_from_axis_angle([0, 0, angle_rad])
        center = [sub_vertices[:, 0].mean(), center_y, sub_vertices[:, 2].mean()]
        rotated = (R @ (sub_vertices - center).T).T + center
        sub_mesh = o3d.geometry.TriangleMesh()
        sub_mesh.vertices = o3d.utility.Vector3dVector(rotated)
        sub_mesh.triangles = o3d.utility.Vector3iVector(mapped_triangles)
        sub_mesh.compute_vertex_normals()
        return sub_mesh

    top_mesh = rotate_part(top_mask, angles[0], mid_y)
    bottom_mesh = rotate_part(bottom_mask, -angles[1], mid_y)
    return top_mesh + bottom_mesh

def get_deformed_part(mesh, split_ratio):
    bbox = mesh.get_axis_aligned_bounding_box()
    min_bound = bbox.get_min_bound()
    max_bound = bbox.get_max_bound()
    size = max_bound - min_bound
    y_min = min_bound[1]
    y_size = size[1]
    split_y = y_min + split_ratio * y_size
    lower_y = split_y - 0.15 * y_size
    upper_y = split_y + 0.15 * y_size
    crop_min = np.array([min_bound[0], lower_y, min_bound[2]])
    crop_max = np.array([max_bound[0], upper_y, max_bound[2]])
    crop_bbox = o3d.geometry.AxisAlignedBoundingBox(crop_min, crop_max)
    return mesh.crop(crop_bbox)

def create_xy_projection(points, resolution=512):
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    norm_points = (points - min_vals) / (max_vals - min_vals + 1e-8)
    norm_points *= resolution - 1
    xy_proj = np.zeros((resolution, resolution), dtype=np.float32)
    for x, y, _ in norm_points:
        i, j = int(y), int(x)
        if 0 <= i < resolution and 0 <= j < resolution:
            xy_proj[i, j] += 1
    return xy_proj

def simulate_xray(mesh, resolution=512):
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(number_of_points=100000)
    points = np.asarray(pcd.points)
    if len(points) == 0:
        raise ValueError("No points were sampled from the mesh.")
    xy_proj = create_xy_projection(points, resolution=resolution)
    xy_proj = gaussian_filter(xy_proj, sigma=1.5)
    xy_proj -= xy_proj.min()
    if xy_proj.max() > 0:
        xy_proj /= xy_proj.max()
    xy_proj = exposure.equalize_adapthist(xy_proj, clip_limit=0.03)
    xy_proj = 1.0 - xy_proj
    return (xy_proj * 255).astype(np.uint8)

def Generate_xray(mesh):
    return simulate_xray(mesh)

def get_segments(meshs):
    index = random.randint(0, len(meshs) - 1)
    mesh = meshs[index]
    top_angle = random.randint(0, 60)
    bottom_angle = random.randint(0, 60)
    split_ratio = random.uniform(0.3, 0.7)
    mesh = create_angle_mesh(mesh, [top_angle, bottom_angle], split_ratio)
    mesh = get_deformed_part(mesh, split_ratio)
    x_ray_image = Generate_xray(mesh)
    bone = "ulna" if index == 0 else "radius"
    return mesh, bone, top_angle, bottom_angle, split_ratio, x_ray_image

def create_surface_between_lines(coords1, coords2):
    assert len(coords1) == len(coords2), "Line point counts must match"
    points = np.vstack((coords1, coords2))
    triangles = []
    n = len(coords1)
    for i in range(n - 1):
        triangles.append([i, i + 1, n + i])
        triangles.append([i + 1, n + i + 1, n + i])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    mesh.compute_vertex_normals()
    return mesh

def get_center_closest_lines_joined(mesh):
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    edge_count = {}
    for tri in triangles:
        for edge in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            edge = tuple(sorted(edge))
            edge_count[edge] = edge_count.get(edge, 0) + 1
    boundary_edges = [e for e, count in edge_count.items() if count == 1]
    G = nx.Graph()
    G.add_edges_from(boundary_edges)
    components = list(nx.connected_components(G))
    grouped_edges = []
    for comp in components:
        comp_edges = []
        comp_vertices = list(comp)
        for i in range(len(comp_vertices)):
            for j in range(i + 1, len(comp_vertices)):
                edge = tuple(sorted((comp_vertices[i], comp_vertices[j])))
                if edge in boundary_edges:
                    comp_edges.append(edge)
        grouped_edges.append(comp_edges)
    center = np.mean(vertices, axis=0)
    group_centers = []
    for group in grouped_edges:
        pts = np.array([vertices[i] for edge in group for i in edge])
        group_center = np.mean(pts, axis=0)
        dist = np.linalg.norm(group_center - center)
        group_centers.append((dist, group))
    group_centers.sort()
    closest_group1 = group_centers[0][1]
    closest_group2 = group_centers[1][1]
    pts1_idx = np.unique(np.array(closest_group1).flatten())
    pts2_idx = np.unique(np.array(closest_group2).flatten())
    coords1_all = vertices[pts1_idx]
    coords2_all = vertices[pts2_idx]
    tree = cKDTree(coords2_all)
    used_pts2 = set()
    connecting_lines = []
    matched_coords1 = []
    matched_coords2 = []
    for i, pt in enumerate(coords1_all):
        dists, indices = tree.query(pt, k=len(coords2_all))
        for j in indices:
            if j not in used_pts2:
                used_pts2.add(j)
                connecting_lines.append([len(matched_coords1), len(matched_coords1) + len(pts1_idx)])
                matched_coords1.append(pt)
                matched_coords2.append(coords2_all[j])
                break
    matched_coords1 = np.array(matched_coords1)
    matched_coords2 = np.array(matched_coords2)
    surface = create_surface_between_lines(matched_coords1, matched_coords2)
    return o3d.geometry.LineSet(), surface

# === MAIN GENERATION LOOP ===
n = 1
num_of_samples = 3000
logging.info(f"🚀 Starting generation of {num_of_samples} bone samples...")
for _ in range(num_of_samples):
    try:
        mesh, bone, top_angle, bottom_angle, split_ratio, x_ray_image = get_segments(meshs)

        o3d.io.write_triangle_mesh(f"models/deformed/bone{n}.ply", mesh)
        cv2.imwrite(f"x-ray images/image{n}.png", x_ray_image)

        _, surface_mesh = get_center_closest_lines_joined(mesh)
        mesh1 = mesh + surface_mesh
        # o3d.io.write_triangle_mesh(f"models/normal/bone{n}.ply", mesh1)

        with open(f"jsons/bone{n}.json", "w") as f:
            json.dump({
                "bone": bone,
                "location": split_ratio,
                "top_angle": top_angle,
                "bottom_angle": bottom_angle
            }, f, indent=4)

        logging.info(f"✅ Sample {n} generated: {bone}, angles=({top_angle},{bottom_angle}), location={split_ratio:.2f}")
        n += 1

    except Exception as e:
        logging.error(f"❌ Failed to generate sample {n}: {e}")

logging.info("✅ All sample generation completed.")
# === POST-PROCESSING ===
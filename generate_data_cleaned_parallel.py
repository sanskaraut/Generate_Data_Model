import open3d as o3d
import numpy as np
import cv2
import os
import json
import random
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count, Value, Lock
from pathlib import Path

# Output directories
DEFORMED_DIR = "models/deformed_parallel"
NORMAL_DIR = "models/normal_parallel"
XRAY_DIR = "x-ray images_parallel"
JSON_DIR = "jsons_parallel"
LOG_FILE = "current_bone_generation_status_parallel.txt"

os.makedirs(DEFORMED_DIR, exist_ok=True)
os.makedirs(NORMAL_DIR, exist_ok=True)
os.makedirs(XRAY_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

# === LOAD & PREPROCESS BASE MESH ===
base_mesh_path = "models/original/forarmbone.ply"
base_mesh = o3d.io.read_triangle_mesh(base_mesh_path)
if not base_mesh.has_triangles():
    raise ValueError("Failed to load base mesh.")

base_mesh.remove_duplicated_vertices()
base_mesh.remove_degenerate_triangles()
base_mesh.remove_duplicated_triangles()
base_mesh.remove_non_manifold_edges()
base_mesh.remove_unreferenced_vertices()
bbox_size = base_mesh.get_max_bound() - base_mesh.get_min_bound()
base_mesh.scale(1 / np.max(bbox_size), center=base_mesh.get_center())
base_mesh.translate(-base_mesh.get_center())
base_mesh.compute_vertex_normals()

# === SPLIT MESH INTO ULNA AND RADIUS ===
bbox = base_mesh.get_axis_aligned_bounding_box()
size = bbox.get_max_bound() - bbox.get_min_bound()
plane_height, plane_depth = size[1], size[2]

plane = o3d.geometry.TriangleMesh.create_box(width=0.001, height=plane_height, depth=plane_depth)
plane.translate((-0.0005, -plane_height / 2, -plane_depth / 2))
center_x = (bbox.get_min_bound()[0] + bbox.get_max_bound()[0]) / 2
plane.translate((center_x, 0, 0))

angle_rad = np.deg2rad(-4.1122)
R = plane.get_rotation_matrix_from_axis_angle([0, 0, angle_rad])
plane.rotate(R, center=plane.get_center())
plane.translate((-0.015, 0, 0))

plane_normal = R @ np.array([1.0, 0.0, 0.0])
plane_normal /= np.linalg.norm(plane_normal)
plane_center = plane.get_center()
points = np.asarray(base_mesh.vertices)
signed_distances = np.dot(points - plane_center, plane_normal)
mask_above = signed_distances > 0
mask_below = signed_distances <= 0
mesh_ulna = base_mesh.select_by_index(np.where(mask_above)[0].tolist())
mesh_radius = base_mesh.select_by_index(np.where(mask_below)[0].tolist())
meshes = [mesh_ulna, mesh_radius]

# === Shared Counter
counter = Value('i', 1)
lock = Lock()
TOTAL_SAMPLES = 3000

# === HELPERS ===
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
    xy_proj = cv2.GaussianBlur(xy_proj, (3, 3), 1.5)
    xy_proj -= xy_proj.min()
    if xy_proj.max() > 0:
        xy_proj /= xy_proj.max()
    xy_proj = 1.0 - xy_proj
    return (xy_proj * 255).astype(np.uint8)

def simulate_xray(mesh):
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(100000)
    points = np.asarray(pcd.points)
    if len(points) == 0:
        raise ValueError("No points sampled from mesh.")
    return create_xy_projection(points)

def create_angle_mesh(mesh, top_angle, bottom_angle, split_ratio):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    min_y, max_y = vertices[:, 1].min(), vertices[:, 1].max()
    mid_y = min_y + (max_y - min_y) * split_ratio
    top_mask = vertices[:, 1] >= mid_y
    bottom_mask = ~top_mask

    def rotate(mask, angle_deg):
        idx = np.where(mask)[0]
        sub_vertices = np.copy(vertices[idx])
        index_map = -np.ones(len(vertices), dtype=int)
        index_map[idx] = np.arange(len(idx))
        tri_mask = np.all(mask[triangles], axis=1)
        sub_triangles = triangles[tri_mask]
        mapped_triangles = index_map[sub_triangles]
        angle_rad = np.radians(angle_deg)
        R = mesh.get_rotation_matrix_from_axis_angle([0, 0, angle_rad])
        center = [sub_vertices[:, 0].mean(), mid_y, sub_vertices[:, 2].mean()]
        rotated = (R @ (sub_vertices - center).T).T + center
        part = o3d.geometry.TriangleMesh()
        part.vertices = o3d.utility.Vector3dVector(rotated)
        part.triangles = o3d.utility.Vector3iVector(mapped_triangles)
        part.compute_vertex_normals()
        return part

    return rotate(top_mask, top_angle) + rotate(bottom_mask, -bottom_angle)

def crop_deformed_part(mesh, split_ratio):
    bbox = mesh.get_axis_aligned_bounding_box()
    min_bound, max_bound = bbox.get_min_bound(), bbox.get_max_bound()
    split_y = min_bound[1] + split_ratio * (max_bound[1] - min_bound[1])
    lower_y = split_y - 0.15 * (max_bound[1] - min_bound[1])
    upper_y = split_y + 0.15 * (max_bound[1] - min_bound[1])
    crop_box = o3d.geometry.AxisAlignedBoundingBox(
        [min_bound[0], lower_y, min_bound[2]],
        [max_bound[0], upper_y, max_bound[2]]
    )
    return mesh.crop(crop_box)

# === WORKER ===
def generate_sample(i):
    start_time = time.time()
    idx = random.randint(0, 1)
    mesh = meshes[idx]
    bone = "ulna" if idx == 0 else "radius"
    top_angle = random.randint(0, 60)
    bottom_angle = random.randint(0, 60)
    split_ratio = random.uniform(0.3, 0.7)

    try:
        mesh = create_angle_mesh(mesh, top_angle, bottom_angle, split_ratio)
        mesh = crop_deformed_part(mesh, split_ratio)
        xray = simulate_xray(mesh)

        o3d.io.write_triangle_mesh(f"{DEFORMED_DIR}/bone{i}.ply", mesh)
        cv2.imwrite(f"{XRAY_DIR}/image{i}.png", xray)
        o3d.io.write_triangle_mesh(f"{NORMAL_DIR}/bone{i}.ply", mesh)

        metadata = {
            "bone": bone,
            "location": split_ratio,
            "top_angle": top_angle,
            "bottom_angle": bottom_angle
        }
        with open(f"{JSON_DIR}/bone{i}.json", "w") as f:
            json.dump(metadata, f, indent=4)

        duration = time.time() - start_time
        with open(LOG_FILE, "a") as log:
            log.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] ✅ Sample {i} done in {duration:.2f}s\n")

    except Exception as e:
        with open(LOG_FILE, "a") as log:
            log.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] ❌ Error in sample {i}: {str(e)}\n")


# === MAIN PARALLEL RUN ===
if __name__ == "__main__":
    TOTAL_SAMPLES = 3000
    workers = min(cpu_count(), 16)
    print(f"🚀 Generating {TOTAL_SAMPLES} samples using {workers} processes")
    with Pool(processes=workers) as pool:
        pool.map(generate_sample, range(1, TOTAL_SAMPLES + 1))  # sequentially pass 1 to 3000

import os
import trimesh
import numpy as np
import h5py
from tqdm import tqdm

# === CONFIGURATION ===
script_dir = os.getcwd()
root_dir = os.path.join(script_dir, 'PU1K_raw_meshes', 'ShapeNetCore.v2.subsample')
output_h5_path = os.path.join(script_dir, 'data','custom', 'pu1k_uniform256_uniform1024_from_meshes.h5')
input_size = 256
gt_size = 1024

# === SAMPLING FUNCTION ===
def uniform_sample(mesh, n):
    """
    Uniformly sample `n` points from a mesh surface.
    """
    points, _ = trimesh.sample.sample_surface(mesh, n)
    return points

# === DATA PROCESSING ===
input_point_clouds = []
gt_point_clouds = []

# Traverse all subdirectories and .off files
mesh_files = []
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.off'):
            mesh_files.append(os.path.join(subdir, file))

print(f"Found {len(mesh_files)} mesh files.")

# Process each mesh
for mesh_path in tqdm(mesh_files, desc="Processing meshes"):
    try:
        mesh = trimesh.load(mesh_path, force='mesh')
        input_pc = uniform_sample(mesh, input_size)
        gt_pc = uniform_sample(mesh, gt_size)
        
        input_point_clouds.append(input_pc.astype(np.float32))
        gt_point_clouds.append(gt_pc.astype(np.float32))
    except Exception as e:
        print(f"[Error] {mesh_path}: {e}")

# Convert lists to arrays
input_array = np.array(input_point_clouds, dtype=np.float32)  # shape (N, 256, 3)
gt_array = np.array(gt_point_clouds, dtype=np.float32)        # shape (N, 1024, 3)

# === SAVE TO HDF5 ===
with h5py.File(output_h5_path, 'w') as f:
    f.create_dataset('poisson_256', data=input_array)     # Compatible with PU-GCN
    f.create_dataset('poisson_1024', data=gt_array)

print(f"✅ Saved {len(input_array)} samples to {output_h5_path}")

import os
import trimesh
import numpy as np
import h5py
from tqdm import tqdm

# === CONFIGURATION ===
script_dir = os.getcwd()

root_dir = os.path.join(script_dir, 'data', 'PU1K_raw_meshes', 'ShapeNetCore.v2.subsample')
output_h5_path = os.path.join(script_dir, 'data', 'custom', 'pu1k_fps256_uniform1024_from_meshes.h5')
input_size = 256
gt_size = 1024
oversample_factor = 10  # for initial dense sampling

# === SAMPLING FUNCTIONS ===
def uniform_sample(mesh, n):
    points, _ = trimesh.sample.sample_surface(mesh, n)
    return points

def farthest_point_sampling(points, n_samples):
    N = points.shape[0]
    sampled_indices = np.zeros(n_samples, dtype=int)
    distances = np.full(N, np.inf)
    
    sampled_indices[0] = np.random.randint(N)
    selected = points[sampled_indices[0]]
    
    for i in range(1, n_samples):
        dist = np.linalg.norm(points - selected, axis=1)
        distances = np.minimum(distances, dist)
        sampled_indices[i] = np.argmax(distances)
        selected = points[sampled_indices[i]]
    
    return points[sampled_indices]

# === DATA PROCESSING ===
input_point_clouds = []
gt_point_clouds = []

mesh_files = []
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.off'):
            mesh_files.append(os.path.join(subdir, file))

print(f"Found {len(mesh_files)} mesh files.")

for mesh_path in tqdm(mesh_files, desc="Processing meshes"):
    try:
        mesh = trimesh.load(mesh_path, force='mesh')
        
        # Step 1: Sample a dense cloud for FPS to act on
        dense_points, _ = trimesh.sample.sample_surface(mesh, input_size * oversample_factor)
        
        # Step 2: FPS for input
        input_pc = farthest_point_sampling(dense_points, input_size)
        
        # Step 3: Uniform sampling for GT
        gt_pc = uniform_sample(mesh, gt_size)
        
        input_point_clouds.append(input_pc.astype(np.float32))
        gt_point_clouds.append(gt_pc.astype(np.float32))
    except Exception as e:
        print(f"[Error] {mesh_path}: {e}")

input_array = np.array(input_point_clouds, dtype=np.float32)
gt_array = np.array(gt_point_clouds, dtype=np.float32)

# === SAVE TO HDF5 ===
with h5py.File(output_h5_path, 'w') as f:
    f.create_dataset('poisson_256', data=input_array)     # Name stays the same for compatibility
    f.create_dataset('poisson_1024', data=gt_array)

print(f"✅ Saved {len(input_array)} samples to {output_h5_path}")

import os
import trimesh
import numpy as np
import h5py
from tqdm import tqdm

# === CONFIGURATION ===
script_dir = os.getcwd()
root_dir = os.path.join(script_dir, 'data', 'PU1K_raw_meshes', 'ShapeNetCore.v2.subsample')
output_h5_path = os.path.join(script_dir, 'data', 'custom', 'pu1k_pugcnstyle_256from1024_poisson.h5')
input_size = 256
gt_size = 1024

# === DATA BUFFERS ===
input_point_clouds = []
gt_point_clouds = []

# === GET ALL MESH FILES ===
mesh_files = []
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.off'):
            mesh_files.append(os.path.join(subdir, file))

print(f"Found {len(mesh_files)} mesh files.")

# === PROCESS EACH MESH ===
for mesh_path in tqdm(mesh_files, desc="Generating PU-GCN-style samples"):
    try:
        mesh = trimesh.load(mesh_path, force='mesh')

        # Step 1: Poisson-disk sample 1024 points for GT
        gt_pc, _ = trimesh.sample.sample_surface_even(mesh, gt_size)

        # Step 2: Randomly choose 256 points as input (subset of GT)
        indices = np.random.choice(gt_pc.shape[0], input_size, replace=False)
        input_pc = gt_pc[indices]

        # Save results
        input_point_clouds.append(input_pc.astype(np.float32))
        gt_point_clouds.append(gt_pc.astype(np.float32))

    except Exception as e:
        print(f"[Error] {mesh_path}: {e}")

# === SAVE TO HDF5 ===
input_array = np.array(input_point_clouds, dtype=np.float32)
gt_array = np.array(gt_point_clouds, dtype=np.float32)

with h5py.File(output_h5_path, 'w') as f:
    f.create_dataset('poisson_256', data=input_array)     # PU-GCN compatibility
    f.create_dataset('poisson_1024', data=gt_array)

print(f"✅ Saved {len(input_array)} samples to {output_h5_path}")


import os
import trimesh
import numpy as np
import h5py
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
import time

# === CONFIGURATION ===
script_dir = os.getcwd()
root_dir = os.path.join(script_dir, 'PU1K_raw_meshes', 'ShapeNetCore.v2.subsample')
patches_per_shape = 50  # As mentioned in the paper
input_size = 256
gt_size = 1024
dense_sample_size = 200000  # Dense sampling for patch extraction

# === SAMPLING FUNCTIONS ===
def uniform_sample(mesh, n):
    """Uniformly sample points from a mesh surface."""
    points, _ = trimesh.sample.sample_surface(mesh, n)
    return points

def poisson_disk_sample(mesh, n):
    """Sample points using Poisson disk sampling for more uniform distribution."""
    try:
        points, _ = trimesh.sample.sample_surface_even(mesh, n)
        return points
    except Exception as e:
        # Fallback to uniform sampling if Poisson fails
        print(f"Poisson sampling failed, falling back to uniform: {e}")
        return uniform_sample(mesh, n)

def farthest_point_sampling(points, npoint):
    """Farthest point sampling for more uniform coverage."""
    N, D = points.shape
    centroids = np.zeros(npoint, dtype=np.int32)
    distance = np.ones(N) * 1e10
    
    # Start with a random point
    farthest = np.random.randint(0, N)
    
    for i in range(npoint):
        centroids[i] = farthest
        centroid = points[farthest, :].reshape(1, D)
        dist = np.sum((points - centroid) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
        
    return centroids

def extract_patches_with_radius(points, patch_size, num_patches, radius_percent=0.05):
    """Extract patches based on radius."""
    num_points = points.shape[0]
    
    # Calculate bounding box and appropriate radius
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    diagonal = np.linalg.norm(bbox_max - bbox_min)
    radius = diagonal * radius_percent
    
    # Use farthest point sampling for seed selection to get better coverage
    seed_indices = farthest_point_sampling(points, num_patches)
    
    patches = []
    patch_centers = []
    
    for idx in seed_indices:
        seed_point = points[idx]
        nn = NearestNeighbors(radius=radius, algorithm='kd_tree').fit(points)
        indices = nn.radius_neighbors([seed_point], radius=radius, return_distance=False)[0]
        
        # If too many points in radius, randomly select patch_size
        if len(indices) > patch_size:
            indices = np.random.choice(indices, patch_size, replace=False)
        # If too few points, use nearest neighbors to reach patch_size
        elif len(indices) < patch_size:
            nn = NearestNeighbors(n_neighbors=patch_size, algorithm='kd_tree').fit(points)
            _, neighbor_indices = nn.kneighbors(seed_point.reshape(1, -1))
            indices = neighbor_indices[0]
        
        patch = points[indices]
        center = np.mean(patch, axis=0, keepdims=True)
        
        patches.append(patch)
        patch_centers.append(center)
    
    # Normalize patches to be centered at origin
    normalized_patches = [patches[i] - patch_centers[i] for i in range(len(patches))]
    
    # Convert to numpy array with shape (num_patches, patch_size, 3)
    return np.array(normalized_patches)

# === EXPERIMENT CONFIGURATIONS ===
experiments = [
    {
        'name': 'patch_uniform_to_uniform',
        'description': 'Patch-based with uniform GT sampling and uniform input subsampling',
        'gt_sampling': 'uniform',
        'input_sampling': 'uniform',
        'patch_based': True
    },
    # {
    #     'name': 'patch_uniform_to_fps',
    #     'description': 'Patch-based with uniform GT sampling and FPS input subsampling',
    #     'gt_sampling': 'uniform',
    #     'input_sampling': 'fps',
    #     'patch_based': True
    # },
    # {
    #     'name': 'patch_poisson_to_uniform',
    #     'description': 'Patch-based with Poisson GT sampling and uniform input subsampling',
    #     'gt_sampling': 'poisson',
    #     'input_sampling': 'uniform',
    #     'patch_based': True
    # },
    # {
    #     'name': 'patch_poisson_to_fps',
    #     'description': 'Patch-based with Poisson GT sampling and FPS input subsampling',
    #     'gt_sampling': 'poisson',
    #     'input_sampling': 'fps',
    #     'patch_based': True
    # },
    # {
    #     'name': 'whole_uniform_to_uniform',
    #     'description': 'Whole shape with uniform GT sampling and uniform input subsampling',
    #     'gt_sampling': 'uniform',
    #     'input_sampling': 'uniform',
    #     'patch_based': False
    # },
    # {
    #     'name': 'whole_uniform_to_fps',
    #     'description': 'Whole shape with uniform GT sampling and FPS input subsampling',
    #     'gt_sampling': 'uniform',
    #     'input_sampling': 'fps',
    #     'patch_based': False
    # },
    # {
    #     'name': 'whole_poisson_to_uniform',
    #     'description': 'Whole shape with Poisson GT sampling and uniform input subsampling',
    #     'gt_sampling': 'poisson',
    #     'input_sampling': 'uniform',
    #     'patch_based': False
    # },
    # {
    #     'name': 'whole_poisson_to_fps',
    #     'description': 'Whole shape with Poisson GT sampling and FPS input subsampling',
    #     'gt_sampling': 'poisson',
    #     'input_sampling': 'fps',
    #     'patch_based': False
    # }
]

# Traverse all subdirectories and .off files
test_mesh_dir = os.path.join(script_dir, 'data', 'PU1K_raw_meshes', 'test', 'original_meshes')


# === Load test mesh names to exclude ===
test_mesh_basenames = set()
for file in os.listdir(test_mesh_dir):
    if file.endswith('.off'):
        test_mesh_basenames.add(os.path.splitext(file)[0])

# === Collect mesh files to process ===
mesh_files = []
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.off'):
            mesh_basename = os.path.splitext(file)[0]
            if mesh_basename not in test_mesh_basenames:
                mesh_files.append(os.path.join(subdir, file))

print(f"Found {len(mesh_files)} mesh files (excluding test set).")

# Run each experiment
for experiment in experiments:
    print(f"Running experiment: {experiment['name']}")
    print(f"Description: {experiment['description']}")
    
    start_time = time.time()
    
    # Reset data collections
    input_patches = []
    gt_patches = []
    
    # Process each mesh according to experiment configuration
    for mesh_path in tqdm(mesh_files, desc=f"Processing {experiment['name']}"):
        try:
            mesh = trimesh.load(mesh_path, force='mesh')
            
            if experiment['patch_based']:
                # Patch-based approach
                dense_pc = uniform_sample(mesh, dense_sample_size)
                
                # Extract patches from dense point cloud
                if experiment['gt_sampling'] == 'uniform':
                    # For uniform GT sampling, use the radius extraction method
                    gt_pc_patches = extract_patches_with_radius(dense_pc, gt_size, patches_per_shape)
                elif experiment['gt_sampling'] == 'poisson':
                    # For Poisson GT sampling, use a simplified approach
                    # Extract uniform patches first
                    uniform_patches = extract_patches_with_radius(dense_pc, gt_size*2, patches_per_shape)
                    gt_pc_patches = []
                    
                    for patch in uniform_patches:
                        # We'll approximate Poisson disk by using farthest point sampling
                        # which creates more uniform distributions without requiring mesh conversion
                        indices = farthest_point_sampling(patch, gt_size)
                        poisson_patch = patch[indices]
                        gt_pc_patches.append(poisson_patch)
                    
                    gt_pc_patches = np.array(gt_pc_patches)
                
                # For each ground truth patch, create corresponding input by subsampling
                for i, gt_patch in enumerate(gt_pc_patches):
                    if experiment['input_sampling'] == 'uniform':
                        # Random subset
                        indices = np.random.choice(gt_size, input_size, replace=False)
                        input_patch = gt_patch[indices]
                    elif experiment['input_sampling'] == 'fps':
                        # FPS subset
                        indices = farthest_point_sampling(gt_patch, input_size)
                        input_patch = gt_patch[indices]
                    
                    # Ensure proper shape (256, 3) and (1024, 3)
                    if input_patch.shape != (input_size, 3):
                        print(f"Warning: input_patch shape is {input_patch.shape}, expected ({input_size}, 3)")
                        continue
                        
                    if gt_patch.shape != (gt_size, 3):
                        print(f"Warning: gt_patch shape is {gt_patch.shape}, expected ({gt_size}, 3)")
                        continue
                    
                    input_patches.append(input_patch.astype(np.float32))
                    gt_patches.append(gt_patch.astype(np.float32))
            
            else:
                # Whole-shape approach (no patches)
                if experiment['gt_sampling'] == 'uniform':
                    gt_pc = uniform_sample(mesh, gt_size)
                elif experiment['gt_sampling'] == 'poisson':
                    gt_pc = poisson_disk_sample(mesh, gt_size)
                
                # Create input cloud based on the sampling method
                if experiment['input_sampling'] == 'uniform':
                    # Random subset of GT
                    indices = np.random.choice(gt_size, input_size, replace=False)
                    input_pc = gt_pc[indices]
                elif experiment['input_sampling'] == 'fps':
                    # For whole shape FPS, we'll sample a denser point cloud and apply FPS
                    dense_pc = uniform_sample(mesh, input_size*10)
                    fps_indices = farthest_point_sampling(dense_pc, input_size)
                    input_pc = dense_pc[fps_indices]
                
                # Check shapes
                if input_pc.shape != (input_size, 3):
                    print(f"Warning: input_pc shape is {input_pc.shape}, expected ({input_size}, 3)")
                    continue
                    
                if gt_pc.shape != (gt_size, 3):
                    print(f"Warning: gt_pc shape is {gt_pc.shape}, expected ({gt_size}, 3)")
                    continue
                
                input_patches.append(input_pc.astype(np.float32))
                gt_patches.append(gt_pc.astype(np.float32))
                
        except Exception as e:
            print(f"[Error] {mesh_path}: {e}")
    
    # Convert lists to arrays
    input_array = np.array(input_patches, dtype=np.float32)
    gt_array = np.array(gt_patches, dtype=np.float32)
    
    # Verify shapes before saving
    print(f"Final input array shape: {input_array.shape}")
    print(f"Final GT array shape: {gt_array.shape}")
    
    # Check for NaN values
    if np.isnan(input_array).any() or np.isnan(gt_array).any():
        print("Warning: NaN values detected in the arrays!")
    
    # Save to HDF5
    output_h5_path = os.path.join(script_dir, 'data', 'custom', f'pu1k_{experiment["name"]}.h5')
    with h5py.File(output_h5_path, 'w') as f:
        f.create_dataset('poisson_256', data=input_array)  # Keep names for compatibility
        f.create_dataset('poisson_1024', data=gt_array)
    
    elapsed_time = time.time() - start_time
    print(f"✅ Saved {len(input_array)} samples to {output_h5_path}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print("------------------------------------")
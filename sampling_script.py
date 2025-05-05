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

def random_sample(points, n):
    """Randomly sample n points from points."""
    indices = np.random.choice(points.shape[0], n, replace=False)
    return points[indices]

def extract_patches_with_radius(points, patch_size, num_patches, radius_percent=0.05):
    """Extract patches based on radius."""
    # Same implementation as before
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
    
    # Find points within radius for each seed point
    tree = KDTree(points)
    
    for idx in seed_indices:
        seed_point = points[idx]
        indices = tree.query_ball_point(seed_point, radius)
        
        # If too many points in radius, randomly select patch_size
        if len(indices) > patch_size:
            indices = np.random.choice(indices, patch_size, replace=False)
        # If too few points, use nearest neighbors to reach patch_size
        elif len(indices) < patch_size:
            nn = NearestNeighbors(n_neighbors=patch_size, algorithm='kd_tree').fit(points)
            _, neighbor_indices = nn.kneighbors(points[idx].reshape(1, -1))
            indices = neighbor_indices[0]
        
        patch = points[indices]
        center = np.mean(patch, axis=0, keepdims=True)
        
        patches.append(patch)
        patch_centers.append(center)
    
    # Normalize patches to be centered at origin
    normalized_patches = [patches[i] - patch_centers[i] for i in range(len(patches))]
    
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
    {
        'name': 'patch_uniform_to_fps',
        'description': 'Patch-based with uniform GT sampling and FPS input subsampling',
        'gt_sampling': 'uniform',
        'input_sampling': 'fps',
        'patch_based': True
    },
    {
        'name': 'patch_poisson_to_uniform',
        'description': 'Patch-based with Poisson GT sampling and uniform input subsampling',
        'gt_sampling': 'poisson',
        'input_sampling': 'uniform',
        'patch_based': True
    },
    {
        'name': 'patch_poisson_to_fps',
        'description': 'Patch-based with Poisson GT sampling and FPS input subsampling',
        'gt_sampling': 'poisson',
        'input_sampling': 'fps',
        'patch_based': True
    },
    {
        'name': 'whole_uniform_to_uniform',
        'description': 'Whole shape with uniform GT sampling and uniform input subsampling',
        'gt_sampling': 'uniform',
        'input_sampling': 'uniform',
        'patch_based': False
    },
    {
        'name': 'whole_uniform_to_fps',
        'description': 'Whole shape with uniform GT sampling and FPS input subsampling',
        'gt_sampling': 'uniform',
        'input_sampling': 'fps',
        'patch_based': False
    },
    {
        'name': 'whole_poisson_to_uniform',
        'description': 'Whole shape with Poisson GT sampling and uniform input subsampling',
        'gt_sampling': 'poisson',
        'input_sampling': 'uniform',
        'patch_based': False
    },
    {
        'name': 'whole_poisson_to_fps',
        'description': 'Whole shape with Poisson GT sampling and FPS input subsampling',
        'gt_sampling': 'poisson',
        'input_sampling': 'fps',
        'patch_based': False
    }
]

# Traverse all subdirectories and .off files
mesh_files = []
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.off'):
            mesh_files.append(os.path.join(subdir, file))

print(f"Found {len(mesh_files)} mesh files.")

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
                    # For Poisson GT sampling, extract patches and then resample each with Poisson
                    temp_patches = extract_patches_with_radius(dense_pc, gt_size*2, patches_per_shape)
                    gt_pc_patches = []
                    for patch in temp_patches:
                        # Create a temporary mesh from patch points to do Poisson sampling
                        # This is an approximation - in practice you might need a more sophisticated approach
                        temp_cloud = trimesh.PointCloud(patch)
                        try:
                            # Try to convert to mesh for Poisson sampling
                            temp_mesh = trimesh.convex.convex_hull(temp_cloud)
                            pts = poisson_disk_sample(temp_mesh, gt_size)
                            gt_pc_patches.append(pts)
                        except:
                            # Fallback to random sampling if mesh creation fails
                            indices = np.random.choice(patch.shape[0], gt_size, replace=False)
                            gt_pc_patches.append(patch[indices])
                    gt_pc_patches = np.array(gt_pc_patches)
                
                # For each ground truth patch, create corresponding input by subsampling
                for gt_patch in gt_pc_patches:
                    if experiment['input_sampling'] == 'uniform':
                        # Random subset
                        indices = np.random.choice(gt_size, input_size, replace=False)
                        input_patch = gt_patch[indices]
                    elif experiment['input_sampling'] == 'fps':
                        # FPS subset
                        indices = farthest_point_sampling(gt_patch, input_size)
                        input_patch = gt_patch[indices]
                    
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
                    # Use FPS on ground truth
                    dense_pc = uniform_sample(mesh, input_size*10)  # Oversample
                    indices = farthest_point_sampling(dense_pc, input_size)
                    input_pc = dense_pc[indices]
                
                input_patches.append(input_pc.astype(np.float32))
                gt_patches.append(gt_pc.astype(np.float32))
                
        except Exception as e:
            print(f"[Error] {mesh_path}: {e}")
    
    # Convert lists to arrays
    input_array = np.array(input_patches, dtype=np.float32)
    gt_array = np.array(gt_patches, dtype=np.float32)
    
    # Save to HDF5
    output_h5_path = os.path.join(script_dir, 'data', 'custom', f'pu1k_{experiment["name"]}.h5')
    with h5py.File(output_h5_path, 'w') as f:
        f.create_dataset('poisson_256', data=input_array)  # Keep names for compatibility
        f.create_dataset('poisson_1024', data=gt_array)
    
    elapsed_time = time.time() - start_time
    print(f"✅ Saved {len(input_array)} samples to {output_h5_path}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print("------------------------------------")


# import os
# import trimesh
# import numpy as np
# import h5py
# from tqdm import tqdm
# from sklearn.neighbors import NearestNeighbors
# from sklearn.neighbors import KDTree

# # === CONFIGURATION ===
# script_dir = os.getcwd()
# root_dir = os.path.join(script_dir, 'PU1K_raw_meshes', 'ShapeNetCore.v2.subsample')
# output_h5_path = os.path.join(script_dir, 'data', 'custom', 'pu1k_uniform256_uniform1024_patches.h5')
# patches_per_shape = 50  # As mentioned in the paper
# input_size = 256
# gt_size = 1024

# # === SAMPLING FUNCTIONS ===
# def uniform_sample(mesh, n):
#     """
#     Uniformly sample `n` points from a mesh surface.
#     """
#     points, _ = trimesh.sample.sample_surface(mesh, n)
#     return points

# dense_sample_size = 200000  # Much larger to ensure good coverage for patches

# def extract_patches_with_radius(points, patch_size, num_patches, radius_percent=0.05):
#     """
#     Extract patches from point cloud by randomly selecting seed points
#     and gathering points within a certain radius.
    
#     Args:
#         points: (N, 3) point cloud
#         patch_size: number of points per patch
#         num_patches: number of patches to extract
#         radius_percent: radius as a percentage of the object's bounding box diagonal
    
#     Returns:
#         patches: (num_patches, patch_size, 3)
#     """
#     num_points = points.shape[0]
    
#     # Calculate bounding box and appropriate radius
#     bbox_min = np.min(points, axis=0)
#     bbox_max = np.max(points, axis=0)
#     diagonal = np.linalg.norm(bbox_max - bbox_min)
#     radius = diagonal * radius_percent
    
#     # Use farthest point sampling for seed selection to get better coverage
#     seed_indices = farthest_point_sampling(points, num_patches)
    
#     patches = []
#     patch_centers = []
    
#     # Find points within radius for each seed point
#     tree = KDTree(points)
    
#     for idx in seed_indices:
#         seed_point = points[idx]
#         indices = tree.query_ball_point(seed_point, radius)
        
#         # If too many points in radius, randomly select patch_size
#         if len(indices) > patch_size:
#             indices = np.random.choice(indices, patch_size, replace=False)
#         # If too few points, use nearest neighbors to reach patch_size
#         elif len(indices) < patch_size:
#             nn = NearestNeighbors(n_neighbors=patch_size, algorithm='kd_tree').fit(points)
#             _, neighbor_indices = nn.kneighbors(points[idx].reshape(1, -1))
#             indices = neighbor_indices[0]
        
#         patch = points[indices]
#         center = np.mean(patch, axis=0, keepdims=True)
        
#         patches.append(patch)
#         patch_centers.append(center)
    
#     # Normalize patches to be centered at origin
#     normalized_patches = [patches[i] - patch_centers[i] for i in range(len(patches))]
    
#     return np.array(normalized_patches)

# def farthest_point_sampling(points, npoint):
#     """
#     Farthest point sampling algorithm for more uniform seed selection.
#     """
#     N, D = points.shape
#     centroids = np.zeros(npoint, dtype=np.int32)
#     distance = np.ones(N) * 1e10
    
#     # Start with a random point
#     farthest = np.random.randint(0, N)
    
#     for i in range(npoint):
#         centroids[i] = farthest
#         centroid = points[farthest, :].reshape(1, D)
#         dist = np.sum((points - centroid) ** 2, axis=1)
#         mask = dist < distance
#         distance[mask] = dist[mask]
#         farthest = np.argmax(distance)
        
#     return centroids

# # === DATA PROCESSING ===
# input_patches = []
# gt_patches = []

# # Traverse all subdirectories and .off files
# mesh_files = []
# for subdir, _, files in os.walk(root_dir):
#     for file in files:
#         if file.endswith('.off'):
#             mesh_files.append(os.path.join(subdir, file))

# print(f"Found {len(mesh_files)} mesh files.")

# # Then in your main processing loop:
# for mesh_path in tqdm(mesh_files, desc="Processing meshes"):
#     try:
#         mesh = trimesh.load(mesh_path, force='mesh')
        
#         # 1) Sample dense point cloud with 200000 points
#         dense_pc = uniform_sample(mesh, dense_sample_size)
        
#         # 2) Use FPS to extract 50 patches per sample, where each patch is of size gt_size within radius_percent of the object's bounding box diagonal
#         gt_pc_patches = extract_patches_with_radius(dense_pc, gt_size, patches_per_shape)
        
#         # 3) For each ground truth patch, create corresponding input by subsampling with FPS
#         for gt_patch in gt_pc_patches:
#             # Use farthest point sampling to create more uniform input
#             indices = farthest_point_sampling(gt_patch, input_size)
#             input_patch = gt_patch[indices]
            
#             input_patches.append(input_patch.astype(np.float32))
#             gt_patches.append(gt_patch.astype(np.float32))
#     except Exception as e:
#         print(f"[Error] {mesh_path}: {e}")

# # Convert lists to arrays
# input_array = np.array(input_patches, dtype=np.float32)  # shape (N*patches_per_shape, 256, 3)
# gt_array = np.array(gt_patches, dtype=np.float32)        # shape (N*patches_per_shape, 1024, 3)

# # === SAVE TO HDF5 ===
# with h5py.File(output_h5_path, 'w') as f:
#     f.create_dataset('uniform_256', data=input_array)
#     f.create_dataset('uniform_1024', data=gt_array)

# print(f"✅ Saved {len(input_array)} samples to {output_h5_path}")

# # # SCRIPT 2
# # output_h5_path = os.path.join(script_dir, 'data', 'custom', 'pu1k_fps256_uniform1024_from_meshes.h5')
# # input_size = 256
# # gt_size = 1024
# # oversample_factor = 10  # for initial dense sampling

# # # === SAMPLING FUNCTIONS ===
# # def uniform_sample(mesh, n):
# #     points, _ = trimesh.sample.sample_surface(mesh, n)
# #     return points

# # def farthest_point_sampling(points, n_samples):
# #     N = points.shape[0]
# #     sampled_indices = np.zeros(n_samples, dtype=int)
# #     distances = np.full(N, np.inf)
    
# #     sampled_indices[0] = np.random.randint(N)
# #     selected = points[sampled_indices[0]]
    
# #     for i in range(1, n_samples):
# #         dist = np.linalg.norm(points - selected, axis=1)
# #         distances = np.minimum(distances, dist)
# #         sampled_indices[i] = np.argmax(distances)
# #         selected = points[sampled_indices[i]]
    
# #     return points[sampled_indices]

# # # === DATA PROCESSING ===
# # input_point_clouds = []
# # gt_point_clouds = []

# # mesh_files = []
# # for subdir, _, files in os.walk(root_dir):
# #     for file in files:
# #         if file.endswith('.off'):
# #             mesh_files.append(os.path.join(subdir, file))

# # print(f"Found {len(mesh_files)} mesh files.")

# # for mesh_path in tqdm(mesh_files, desc="Processing meshes"):
# #     try:
# #         mesh = trimesh.load(mesh_path, force='mesh')
        
# #         # Step 1: Sample a dense cloud for FPS to act on
# #         dense_points, _ = trimesh.sample.sample_surface(mesh, input_size * oversample_factor)
        
# #         # Step 2: FPS for input
# #         input_pc = farthest_point_sampling(dense_points, input_size)
        
# #         # Step 3: Uniform sampling for GT
# #         gt_pc = uniform_sample(mesh, gt_size)
        
# #         input_point_clouds.append(input_pc.astype(np.float32))
# #         gt_point_clouds.append(gt_pc.astype(np.float32))
# #     except Exception as e:
# #         print(f"[Error] {mesh_path}: {e}")

# # input_array = np.array(input_point_clouds, dtype=np.float32)
# # gt_array = np.array(gt_point_clouds, dtype=np.float32)

# # # === SAVE TO HDF5 ===
# # with h5py.File(output_h5_path, 'w') as f:
# #     f.create_dataset('poisson_256', data=input_array)     # Name stays the same for compatibility
# #     f.create_dataset('poisson_1024', data=gt_array)

# # print(f"✅ Saved {len(input_array)} samples to {output_h5_path}")



# # # SCRIPT 3
# # output_h5_path = os.path.join(script_dir, 'data', 'custom', 'pu1k_pugcnstyle_256from1024_poisson.h5')
# # input_size = 256
# # gt_size = 1024

# # # === DATA BUFFERS ===
# # input_point_clouds = []
# # gt_point_clouds = []

# # # === GET ALL MESH FILES ===
# # mesh_files = []
# # for subdir, _, files in os.walk(root_dir):
# #     for file in files:
# #         if file.endswith('.off'):
# #             mesh_files.append(os.path.join(subdir, file))

# # print(f"Found {len(mesh_files)} mesh files.")

# # # === PROCESS EACH MESH ===
# # for mesh_path in tqdm(mesh_files, desc="Generating PU-GCN-style samples"):
# #     try:
# #         mesh = trimesh.load(mesh_path, force='mesh')

# #         # Step 1: Poisson-disk sample 1024 points for GT
# #         gt_pc, _ = trimesh.sample.sample_surface_even(mesh, gt_size)

# #         # Step 2: Randomly choose 256 points as input (subset of GT)
# #         indices = np.random.choice(gt_pc.shape[0], input_size, replace=False)
# #         input_pc = gt_pc[indices]

# #         # Save results
# #         input_point_clouds.append(input_pc.astype(np.float32))
# #         gt_point_clouds.append(gt_pc.astype(np.float32))

# #     except Exception as e:
# #         print(f"[Error] {mesh_path}: {e}")

# # # === SAVE TO HDF5 ===
# # input_array = np.array(input_point_clouds, dtype=np.float32)
# # gt_array = np.array(gt_point_clouds, dtype=np.float32)

# # with h5py.File(output_h5_path, 'w') as f:
# #     f.create_dataset('poisson_256', data=input_array)     # PU-GCN compatibility
# #     f.create_dataset('poisson_1024', data=gt_array)

# # print(f"✅ Saved {len(input_array)} samples to {output_h5_path}")


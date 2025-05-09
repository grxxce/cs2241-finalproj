import os
import trimesh
import numpy as np
import h5py
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

# === CONFIGURATION ===
script_dir = os.getcwd()
root_dir = os.path.join(script_dir, 'PU1K_raw_meshes', 'ShapeNetCore.v2.subsample')
test_mesh_dir = os.path.join(script_dir, 'PU1K_raw_meshes', 'test', 'original_meshes')
output_dir = os.path.join(script_dir, 'data', 'custom')
os.makedirs(output_dir, exist_ok=True)

patches_per_shape = 50
input_size = 256
gt_size = 1024
dense_sample_size = 200000
max_neighbors = 2048

sampling_combinations = [
    ("uniform", "uniform"),
    ("uniform", "fps"),
    ("poisson", "uniform"),
    ("poisson", "fps")
]

# === HELPERS ===
def uniform_sample(mesh, n):
    points, _ = trimesh.sample.sample_surface(mesh, n)
    return points

def poisson_disk_sample(mesh, n):
    try:
        points, _ = trimesh.sample.sample_surface_even(mesh, n)
        return points
    except Exception as e:
        print(f"Poisson sampling failed, falling back to uniform: {e}")
        return uniform_sample(mesh, n)

def farthest_point_sampling(points, num_samples):
    N = points.shape[0]
    chosen_indices = np.zeros(num_samples, dtype=int)
    dist = np.ones(N) * 1e10
    chosen_indices[0] = np.random.randint(N)
    for i in range(1, num_samples):
        centroid = points[chosen_indices[i - 1], :].reshape(1, -1)
        dist = np.minimum(dist, np.sum((points - centroid) ** 2, axis=1))
        chosen_indices[i] = np.argmax(dist)
    return chosen_indices

def get_mesh_name(path):
    return os.path.splitext(os.path.basename(path))[0]

# === LOAD TEST MESH NAMES TO EXCLUDE ===
test_mesh_basenames = set()
for file in os.listdir(test_mesh_dir):
    if file.endswith('.off'):
        test_mesh_basenames.add(os.path.splitext(file)[0])

# === COLLECT MESH FILES TO PROCESS ===
mesh_files = []
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.off'):
            basename = os.path.splitext(file)[0]
            if basename not in test_mesh_basenames:
                mesh_files.append(os.path.join(subdir, file))

print(f"Found {len(mesh_files)} mesh files (excluding test set).")

# === MAIN LOOP FOR EACH SAMPLING COMBINATION ===
for gt_method, input_method in sampling_combinations:
    input_patches = []
    gt_patches = []
    print(f"\nProcessing combination: GT={gt_method}, Input={input_method}")

    for mesh_path in tqdm(mesh_files, desc=f"{gt_method}_to_{input_method}"):
        try:
            mesh = trimesh.load(mesh_path, force='mesh')
            dense_points = uniform_sample(mesh, dense_sample_size)
            center_indices = farthest_point_sampling(dense_points, patches_per_shape)
            patch_centers = dense_points[center_indices]
            nbrs = NearestNeighbors(n_neighbors=max_neighbors, algorithm='auto').fit(dense_points)

            for center in patch_centers:
                _, neighbor_idx = nbrs.kneighbors(center.reshape(1, -1), n_neighbors=max_neighbors)
                local_points = dense_points[neighbor_idx[0]]

                if local_points.shape[0] < max(gt_size, input_size):
                    continue

                # GT sampling
                if gt_method == "uniform":
                    gt_patch = local_points[np.random.choice(local_points.shape[0], gt_size, replace=False)]
                elif gt_method == "poisson":
                    gt_patch = local_points[farthest_point_sampling(local_points, gt_size)]
                else:
                    raise ValueError("Unsupported GT sampling method")

                # Input sampling
                if input_method == "uniform":
                    input_patch = local_points[np.random.choice(local_points.shape[0], input_size, replace=False)]
                elif input_method == "fps":
                    input_patch = local_points[farthest_point_sampling(local_points, input_size)]
                else:
                    raise ValueError("Unsupported input sampling method")

                # Ensure patches are valid
                if input_patch.shape != (input_size, 3) or gt_patch.shape != (gt_size, 3):
                    continue

                input_patches.append(input_patch.astype(np.float32))
                gt_patches.append(gt_patch.astype(np.float32))

        except Exception as e:
            print(f"[Error] {mesh_path}: {e}")

    # === SAVE TO HDF5 ===
    input_array = np.array(input_patches, dtype=np.float32)
    gt_array = np.array(gt_patches, dtype=np.float32)

    output_h5_path = os.path.join(output_dir, f'pu1k_indep_{gt_method}_gt_{input_method}_input.h5')
    with h5py.File(output_h5_path, 'w') as f:
        f.create_dataset('poisson_256', data=input_array)
        f.create_dataset('poisson_1024', data=gt_array)

    print(f"✅ Saved {len(input_array)} samples to {output_h5_path}")
import h5py
import numpy as np

# Path to the working dataset file
file_path = '/n/home09/grxxce/implicit-neurons-repo/cs2241-finalproj/data/PU1K/train/pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5'

# Open the file and print the structure
with h5py.File(file_path, 'r') as f:
    # List all groups and datasets
    print("File structure:")
    for key in f.keys():
        print(f"Dataset: {key}, Shape: {f[key].shape}, Type: {f[key].dtype}")
    
    # Check specific shapes
    if 'poisson_256' in f and 'poisson_1024' in f:
        print("\nInput shape:", f['poisson_256'].shape)
        print("GT shape:", f['poisson_1024'].shape)
        
        # Print sample data points
        print("\nSample from input:", f['poisson_256'][0, 0:5, :])
        print("Sample from GT:", f['poisson_1024'][0, 0:5, :])
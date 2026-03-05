import h5py

with h5py.File("SHT_A/train/gt_density_map/IMG_1.h5", 'r') as f:
    print("Keys:", list(f.keys()))
    if 'density' in f:
        print("Density shape:", f['density'].shape)

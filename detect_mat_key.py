import scipy.io
import numpy as np

# Load the .mat file
mat_file = 'ground_truths/GT_IMG_115.mat'
mat_data = scipy.io.loadmat(mat_file)

# List all top-level keys
print("Top-level keys:", mat_data.keys())

# Remove meta keys
valid_keys = [k for k in mat_data.keys() if not k.startswith('__')]
print("Valid keys:", valid_keys)

# Explore each valid key
for key in valid_keys:
    value = mat_data[key]
    print(f"\nKey: '{key}' - Type: {type(value)} - Shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")

    # Try to print structure of arrays
    try:
        if isinstance(value, np.ndarray) and value.size > 0:
            print("First element contents (truncated):", value[0])
    except Exception as e:
        print(f"Could not access first element: {e}")

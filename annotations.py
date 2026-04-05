import os
import h5py
import scipy.io
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Filenames: --->
img_file = 'ground_truths/IMG_215.jpg'
h5_file = 'ground_truths/GAK_IMG_215.h5'
mat_file = 'ground_truths/GT_IMG_215.mat'
plot_file = 'annotations/gak_2.png'

# Check whether the directory exists or not: --->
os.makedirs("annotations", exist_ok = True)

# Load the original image: --->
image = cv2.imread(img_file)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load the ground truth density map from the .h5 file: --->
with h5py.File(h5_file, 'r') as f:
    density_map = np.array(f['density'])

# Load the head annotations from the .mat file: --->
mat_data = scipy.io.loadmat(mat_file)

# Adjust key based on structure: commonly 'image_info' -> [0][0][0][0] = coordinates
# Try to autodetect the key if unsure
if 'image_info' in mat_data:
    annotations = mat_data['image_info'][0, 0][0, 0][0]
else:
    raise KeyError("Expected 'image_info' in .mat file")

# Create an empty annotation image: --->
annot_img = np.copy(image)
for point in annotations:
    x, y = int(point[0]), int(point[1])
    if 0 <= y < annot_img.shape[0] and 0 <= x < annot_img.shape[1]:
        annot_img[y-2 : y+2, x-2 : x+2] = [255, 0, 0]  # red square marker

# Plot images side by side: --->
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(image)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(annot_img)
axes[1].set_title('Head Annotations')
axes[1].axis('off')

axes[2].imshow(density_map, cmap = 'jet')
axes[2].set_title('Density Map (Geometry Adaptive Kernel)')
axes[2].axis('off')

# Save to disk: --->
plt.tight_layout()
plt.savefig(plot_file)
plt.close()

print(f"Plot saved as: {plot_file}.")

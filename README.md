# 👥 Crowd Counting and Crowd Density Estimation using Autoencoder Induced Density Maps

### 🎓 Indian Institute of Technology Guwahati, North-Guwahati, Kamrup (R), Assam - 781039
### 🎓 Cotton University, Panbazar, Guwahati, Kamrup (M), Assam - 781001

---

**👨‍🏫 Supervisor:** Dr. Prithwijit Guha, Associate Professor, Dept. Of EEE, IIT-G
**🧠 Mentor:** Shahbaz Ahmed, PhD Scholar, Dept. Of EEE, IIT-G
**👨‍🎓 Author:** Raan Saurav Bhuyan, Student of MCA 4th Sem, CU

---

## 📑 Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Directory Structure](#dataset-directory-structure)
3. [Model Architecture](#model-architecture)
4. [Loss Function](#loss-function)
5. [Training Strategy](#training-strategy)
6. [Execution Instructions](#execution-instructions)

---

## 🚀 Project Overview
This repository hosts the source code for an MLP-based autoencoder designed for generating crowd density maps from input images and estimating the crowd density and overall count in a given scene. The pipeline utilizes PyTorch and demonstrates an end-to-end flow from data preprocessing (using OpenCV) to density map inference and evaluation.

---

## 📂 Dataset Directory Structure
The project expects the datasets (e.g., ShanghaiTech Part-A and Part-B) to follow a specific hierarchical structure. The density maps (`.h5` files) are loaded from a directory whose name matches the kernel utilized for ground truth generation (e.g., `sigma_15`).

```text
dataset/
├── SHT_A/                                # ShanghaiTech Part-A
│   ├── train/
    │   ├── images/                       # Original JPG images
    │   └── sigma_4/
    │   └── sigma_15/
    │   └── gak/                     # Generated .h5 density maps (GT_DENSITY_MAP_ROOT)
│   └── test/
│       ├── images/
    │   └── sigma_4/
    │   └── sigma_15/
    │   └── gak/
└── SHT_B/                                # ShanghaiTech Part-B (similar structure)
    ├── train/
    │   ├── images/
    │   └── sigma_4/
    │   └── sigma_15/
    │   └── gak/
    └── test/
        ├── images/
    │   └── sigma_4/
    │   └── sigma_15/
    │   └── gak/
```

---

## 🏗️ Model Architecture
The core model (`CrowdMLP`) is a Multi-Layer Perceptron (MLP) autoencoder. It takes a flattened image of dimension $224 \times 224 \times C$ (where $C$ is 1 for Monochrome and 3 for RGB) and compresses it into a bottleneck of 16 features before reconstructing it into a $224 \times 224$ density map.

### 🧮 Parameter Calculation Table (Assuming Monochrome Input $C=1$)

| Layer / Component | Input Shape | Output Shape | Parameters Calculation (Weights + Biases) | Total Parameters |
| :--- | :--- | :--- | :--- | :--- |
| **Encoder** | | | | |
| Linear 1 + ReLU | 50,176 | 1024 | $(50176 \times 1024) + 1024$ | 51,381,248 |
| Linear 2 + ReLU | 1024 | 256 | $(1024 \times 256) + 256$ | 262,400 |
| Linear 3 + ReLU | 256 | 64 | $(256 \times 64) + 64$ | 16,448 |
| Linear 4 + ReLU | 64 | 16 | $(64 \times 16) + 16$ | 1,040 |
| **Decoder** | | | | |
| Linear 5 + ReLU | 16 | 64 | $(16 \times 64) + 64$ | 1,088 |
| Linear 6 + ReLU | 64 | 256 | $(64 \times 256) + 256$ | 16,640 |
| Linear 7 + ReLU | 256 | 1024 | $(256 \times 1024) + 1024$ | 263,168 |
| Linear 8 | 1024 | 50,176 | $(1024 \times 50176) + 50176$ | 51,430,400 |
| **Total** | | | | **~ 103.37 Million** |

---

## 🎯 Loss Function
This pipeline utilizes a custom `CompositeCrowdLoss` objective, which is a combination of two distinct loss metrics:

1. **Mean Squared Error (MSE) / Density Map Loss:** Measures the pixel-wise difference between the predicted density map and the ground truth density map. This ensures the model correctly captures the spatial distribution of the crowd.
2. **Mean Absolute Error (MAE) / Crowd Count Loss:** Computes the absolute difference between the total predicted crowd count (the discrete summation of the predicted density map matrix) and the ground truth count. This ensures the global estimation of the crowd is highly accurate.

**Total Loss** = $MSE(Map_{pred}, Map_{gt}) + MAE(Count_{pred}, Count_{gt})$

---

## 🏋️ Training Strategy
The repository contains two approaches for training the network: a normal training loop (`train.py`) and a special active-learning style strategy (`train_strategy.py`).

### 🔄 Normal Training (`train.py`)
Standard deep learning training. The dataset is shuffled, and the model updates its weights sequentially iterating through batches in each epoch.

### 🧠 Special Training Strategy (`train_strategy.py`)
This approach utilizes a "Hard Batch Mining" paradigm to dynamically focus the network's learning on instances it finds difficult.
**📝 Algorithm:**
1. Divide the training dataset into $K$ fixed batches.
2. Heavily train the model (5 iterations) on the first batch to establish a solid initial baseline weight set.
3. Over $K$ iterations, evaluate the current model state across all remaining, unseen batches.
4. Sort the batches by evaluation loss in descending order to identify the "worst performing batch" (the batch the model struggled with the most).
5. Train the model exclusively on this identified worst batch.
6. Repeat the evaluation and training process, allowing the model to dynamically shift focus to challenging crowd density distributions.

---

## 💻 Execution Instructions

Follow these point-wise instructions to set up and execute the project on a local system.

1. **📥 Clone the Repository:**
   Clone the project to your local workspace.
   ```bash
   git clone <repository_url>
   cd CCDE_1_MLP
   ```

2. **🐍 Set up the Environment:**
   It is highly recommended to use a virtual environment.
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **📦 Install Dependencies:**
   Ensure all required libraries (PyTorch, OpenCV, Matplotlib, etc.) are installed.
   ```bash
   pip install -r requirements.txt
   ```

4. **📁 Prepare the Dataset:**
   Download the ShanghaiTech (Part A/B) dataset. Extract it into a folder named `dataset/` at the root of the project. Ensure the directory structure matches the one outlined in the Dataset Directory Structure section.

5. **⚙️ Configure Constants:**
   Open `const.py` to tune the hyperparameters. You can define `BATCH_SIZE`, `EPOCHS`, `LR`, `DATASET_ROOT`, `KERNEL`, and switch `IMAGE_COLOR` between `"MONO"` and `"RGB"`.

6. **🚀 Execution (Training & Evaluation):**
   To start the pipeline, run the main script. By default, `main.py` is configured to run the normal training loop followed by evaluation.
   ```bash
   python main.py
   ```
   *(Note: To utilize the special training strategy, uncomment `from train_strategy import train_model` and comment out `from train import train_model` in `main.py`).*

7. **📊 Analyze Results:**
   After the script finishes execution, you can analyze the generated artifacts in the respective folders:
   * `checkpoints/`: Contains the saved model `.pth` files.
   * `results/`: Contains the `.csv` comparisons for predicted vs GT crowd counts per image.
   * `plots/`: Contains the loss graphs (MSE, MAE, Total) over epochs/batches.
   * `visualizations/`: Contains the side-by-side inference visualization of the input image, ground truth density map, and predicted density map.

# Import libraries: --->
import torch

# Define Constants: --->
BATCH_SIZE = 8
EPOCHS = 20
LR = 3e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VISUALISATIONS = 5
IMAGE_COLOR = "MONO"                            #! <--------------

GRAY = None
if IMAGE_COLOR == "MONO":
    GRAY = True
elif IMAGE_COLOR == "RGB":
    GRAY = False
else:
    pass

MODEL_ROOT = "checkpoints"
DATASET_ROOT = "dataset/SHT_A"                 #! <--------------
BOTTLENECK = "1024_256_64_16"                  #! <--------------
KERNEL = "SIGMA_15"                                    #! <--------------
GT_DENSITY_MAP_ROOT = KERNEL.lower()
VISUAL_ROOT = "visualizations"
CSV_ROOT = "results"
PLOT_ROOT = "plots"

# File names: --->
MODEL = f"{MODEL_ROOT}/ts_{IMAGE_COLOR.lower()}_{KERNEL.lower()}_{BOTTLENECK.lower()}_1.pth"
CSV_NAME = f"{CSV_ROOT}/ts_test_count_{IMAGE_COLOR.lower()}_{KERNEL.lower()}_{BOTTLENECK.lower()}_1.csv"

# Plot names: --->
TRAIN_TOTAL_LOSS = f"{PLOT_ROOT}/train_total_{IMAGE_COLOR.lower()}_{KERNEL.lower()}_{BOTTLENECK.lower()}_1.png"
TEST_TOTAL_LOSS = f"{PLOT_ROOT}/test_total_{IMAGE_COLOR.lower()}_{KERNEL.lower()}_{BOTTLENECK.lower()}_1.png"
TRAIN_MSE_LOSS = f"{PLOT_ROOT}/train_mse_{IMAGE_COLOR.lower()}_{KERNEL.lower()}_{BOTTLENECK.lower()}_1.png"
TEST_MSE_LOSS = f"{PLOT_ROOT}/test_mse_{IMAGE_COLOR.lower()}_{KERNEL.lower()}_{BOTTLENECK.lower()}_1.png"
TRAIN_MAE_LOSS = f"{PLOT_ROOT}/train_mae_{IMAGE_COLOR.lower()}_{KERNEL.lower()}_{BOTTLENECK.lower()}_1.png"
TEST_MAE_LOSS = f"{PLOT_ROOT}/test_mae_{IMAGE_COLOR.lower()}_{KERNEL.lower()}_{BOTTLENECK.lower()}_1.png"

# config.py
import torch

# Dataset
DATA_DIR    = 'dataset'
NUM_CLASSES = 10
CLASS_NAMES = [
    'pho', 'bun_bo_hue', 'com_tam', 'banh_mi', 'bun_rieu',
    'banh_xeo', 'goi_cuon', 'hu_tieu', 'mi_quang', 'cao_lau'
]

# Training
BATCH_SIZE  = 32
NUM_EPOCHS  = 30
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Models cần train và kích thước ảnh tương ứng
MODELS_CONFIG = {
    'efficientnet_b3': {'img_size': 300, 'lr': 0.001},
    'resnet50':        {'img_size': 224, 'lr': 0.001},
    'inceptionv3':     {'img_size': 299, 'lr': 0.0005},
}

# Paths
MODEL_SAVE_DIR  = 'models'
RESULTS_DIR     = 'results'
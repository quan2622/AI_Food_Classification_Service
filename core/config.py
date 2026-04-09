from pathlib import Path
import os

import torch
from dotenv import load_dotenv

CORE_DIR = Path(__file__).resolve().parent
load_dotenv(CORE_DIR / '.env')

DATA_DIR = CORE_DIR / 'dataset'
MODEL_SAVE_DIR = CORE_DIR / 'models'
RESULTS_DIR = CORE_DIR / 'results'
PRETRAINED_DIR = CORE_DIR / 'pretrained'
RUNTIME_DATA_DIR = CORE_DIR / 'data_runtime'
UPLOADS_DIR = RUNTIME_DATA_DIR / 'uploads'
REVIEWED_DIR = RUNTIME_DATA_DIR / 'reviewed'
DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'postgresql+psycopg://postgres:123456@192.168.30.128:5432/ai_food_db?schema=public',
)


def resolve_split_dir(*candidates):
    if not DATA_DIR.exists():
        return None

    subdirs = {item.name.lower(): item for item in DATA_DIR.iterdir() if item.is_dir()}
    for candidate in candidates:
        match = subdirs.get(candidate.lower())
        if match:
            return match
    return None


def discover_class_names():
    train_dir = resolve_split_dir('train')
    if not train_dir:
        return []
    return sorted(item.name for item in train_dir.iterdir() if item.is_dir())


# Dataset
CLASS_NAMES = discover_class_names()
NUM_CLASSES = len(CLASS_NAMES)

# Training
BATCH_SIZE = 32
NUM_EPOCHS = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEFAULT_MODEL_NAME = 'efficientnet_b3'

# Models cần train và kích thước ảnh tương ứng
MODELS_CONFIG = {
    'efficientnet_b3': {'img_size': 300, 'lr': 0.0002},
    'resnet50': {'img_size': 224, 'lr': 0.0002},
    'inceptionv3': {'img_size': 299, 'lr': 0.0001},
}

PRETRAINED_WEIGHTS = {
    'efficientnet_b3': PRETRAINED_DIR / 'efficientnet_b3_rwightman-b3899882.pth',
    'resnet50': PRETRAINED_DIR / 'resnet50-0676ba61.pth',
    'inceptionv3': PRETRAINED_DIR / 'inception_v3_google-0cc3c7bd.pth',
}

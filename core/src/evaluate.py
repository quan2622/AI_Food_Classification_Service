# src/evaluate.py
import argparse
import os
import sys

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import (
    CLASS_NAMES,
    DEFAULT_MODEL_NAME,
    DEVICE,
    MODEL_SAVE_DIR,
    MODELS_CONFIG,
    NUM_CLASSES,
    RESULTS_DIR,
)
from src.dataset import get_dataloaders
from src.models import get_model

def evaluate_model(model_name):
    cfg = MODELS_CONFIG[model_name]
    _, _, test_dl = get_dataloaders(cfg['img_size'])

    model = get_model(model_name, NUM_CLASSES).to(DEVICE)
    model_path = MODEL_SAVE_DIR / f'best_{model_name}.pth'
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_dl:
            inputs = inputs.to(DEVICE)
            _, preds = torch.max(model(inputs), 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # In report
    print(f"\n=== {model_name.upper()} ===")
    print(classification_report(all_labels, all_preds,
                                target_names=CLASS_NAMES, digits=4))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix — {model_name}')
    plt.ylabel('Thực tế'); plt.xlabel('Dự đoán')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(RESULTS_DIR / f'cm_{model_name}.png', dpi=150)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Danh gia model tren test set.')
    parser.add_argument(
        '--model',
        default=DEFAULT_MODEL_NAME,
        choices=MODELS_CONFIG.keys(),
        help='Model can danh gia.',
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Danh gia toan bo model da train.',
    )
    args = parser.parse_args()

    models_to_eval = MODELS_CONFIG.keys() if args.all else [args.model]
    for model_name in models_to_eval:
        evaluate_model(model_name)

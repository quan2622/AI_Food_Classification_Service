# src/utils.py
import json, os, matplotlib.pyplot as plt
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import RESULTS_DIR

def save_history(model_name, history):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f'{RESULTS_DIR}/history_{model_name}.json', 'w') as f:
        json.dump(history, f)

def plot_curves(model_name, history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, len(history['train_acc']) + 1)
    ax1.plot(ep, history['train_loss'], label='Train')
    ax1.plot(ep, history['val_loss'],   label='Val')
    ax1.set_title(f'{model_name} — Loss')
    ax1.set_xlabel('Epoch'); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(ep, history['train_acc'], label='Train')
    ax2.plot(ep, history['val_acc'],   label='Val')
    ax2.set_title(f'{model_name} — Accuracy')
    ax2.set_xlabel('Epoch'); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(f'{RESULTS_DIR}/curves_{model_name}.png', dpi=150)
    plt.close()
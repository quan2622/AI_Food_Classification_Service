# src/utils.py
import json, os, matplotlib.pyplot as plt
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import RESULTS_DIR

def save_history(model_name, history):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_DIR / f'history_{model_name}.json', 'w') as f:
        json.dump(history, f)


def print_dataset_overview(dataset_info):
    print("\n=== DATASET OVERVIEW ===")
    print(
        f"Train: {dataset_info['train_size']} ảnh | "
        f"Val: {dataset_info['val_size']} ảnh | "
        f"Test: {dataset_info['test_size']} ảnh"
    )
    print(
        f"Train batches: {dataset_info['train_batches']} | "
        f"Val batches: {dataset_info['val_batches']} | "
        f"Test batches: {dataset_info['test_batches']}"
    )
    print("\nCác lớp đang train:")
    for idx, class_name in enumerate(dataset_info['class_names'], start=1):
        train_count = dataset_info['train_counts'][class_name]
        val_count = dataset_info['val_counts'][class_name]
        test_count = dataset_info['test_counts'][class_name]
        print(
            f"{idx:02d}. {class_name:<20} "
            f"train={train_count:<4} | val={val_count:<3} | test={test_count:<3}"
        )


def build_training_summary(model_name, history, dataset_info, training_config):
    best_epoch = max(
        range(len(history['val_acc'])),
        key=lambda idx: history['val_acc'][idx],
    ) + 1
    return {
        'model_name': model_name,
        'best_epoch': best_epoch,
        'best_val_acc': max(history['val_acc']),
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1],
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'dataset': dataset_info,
        'config': training_config,
    }

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
    plt.savefig(RESULTS_DIR / f'curves_{model_name}.png', dpi=150)
    plt.close()


def plot_training_summary(model_name, history, dataset_info, training_summary):
    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.4])

    ax_info = fig.add_subplot(grid[0, 0])
    ax_loss = fig.add_subplot(grid[0, 1])
    ax_acc = fig.add_subplot(grid[1, 0])
    ax_dist = fig.add_subplot(grid[1, 1])

    ep = range(1, len(history['train_acc']) + 1)

    config_text = "\n".join([
        f"Model: {training_summary['model_name']}",
        f"Best epoch: {training_summary['best_epoch']}",
        f"Best val acc: {training_summary['best_val_acc']:.2f}%",
        f"Final train acc: {training_summary['final_train_acc']:.2f}%",
        f"Final val acc: {training_summary['final_val_acc']:.2f}%",
        f"Final train loss: {training_summary['final_train_loss']:.4f}",
        f"Final val loss: {training_summary['final_val_loss']:.4f}",
        f"Batch size: {training_summary['config']['batch_size']}",
        f"Epochs: {training_summary['config']['num_epochs']}",
        f"Learning rate: {training_summary['config']['lr']}",
        f"Image size: {training_summary['config']['img_size']}",
        f"Device: {training_summary['config']['device']}",
        f"Trainable params: {training_summary['config']['trainable_params']:,}",
        f"Total params: {training_summary['config']['total_params']:,}",
    ])
    ax_info.axis('off')
    ax_info.text(
        0.02, 0.98, config_text,
        va='top', ha='left', fontsize=11,
        bbox={'facecolor': '#f4f4f4', 'edgecolor': '#999999', 'boxstyle': 'round,pad=0.6'}
    )
    ax_info.set_title('Tong ket train', fontsize=13)

    ax_loss.plot(ep, history['train_loss'], label='Train loss', linewidth=2)
    ax_loss.plot(ep, history['val_loss'], label='Val loss', linewidth=2)
    ax_loss.set_title('Loss theo epoch')
    ax_loss.set_xlabel('Epoch')
    ax_loss.grid(alpha=0.3)
    ax_loss.legend()

    ax_acc.plot(ep, history['train_acc'], label='Train acc', linewidth=2)
    ax_acc.plot(ep, history['val_acc'], label='Val acc', linewidth=2)
    ax_acc.set_title('Accuracy theo epoch')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy (%)')
    ax_acc.grid(alpha=0.3)
    ax_acc.legend()

    class_names = dataset_info['class_names']
    train_counts = [dataset_info['train_counts'][name] for name in class_names]
    val_counts = [dataset_info['val_counts'][name] for name in class_names]
    test_counts = [dataset_info['test_counts'][name] for name in class_names]
    x_positions = range(len(class_names))
    width = 0.25

    ax_dist.bar([x - width for x in x_positions], train_counts, width=width, label='Train')
    ax_dist.bar(x_positions, val_counts, width=width, label='Val')
    ax_dist.bar([x + width for x in x_positions], test_counts, width=width, label='Test')
    ax_dist.set_title('Phan bo du lieu theo mon an')
    ax_dist.set_xticks(list(x_positions))
    ax_dist.set_xticklabels(class_names, rotation=55, ha='right', fontsize=9)
    ax_dist.grid(alpha=0.2, axis='y')
    ax_dist.legend()

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(RESULTS_DIR / f'summary_{model_name}.png', dpi=180)
    plt.close()

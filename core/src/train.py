# src/train.py
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import (
    BATCH_SIZE,
    DEFAULT_MODEL_NAME,
    DEVICE,
    MODELS_CONFIG,
    MODEL_SAVE_DIR,
    NUM_CLASSES,
    NUM_EPOCHS,
)
from src.dataset import get_dataloaders
from src.models import get_model
from src.utils import (
    build_training_summary,
    plot_curves,
    plot_training_summary,
    print_dataset_overview,
    save_history,
)


def count_parameters(model):
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total_params, trainable_params


def print_cuda_status():
    cuda_available = torch.cuda.is_available()
    print("\n=== GPU STATUS ===")
    print(f"torch.cuda.is_available(): {cuda_available}")
    print(f"Configured device         : {DEVICE}")
    print(f"GPU count                 : {torch.cuda.device_count()}")
    if cuda_available:
        current_gpu = torch.cuda.get_device_name(0)
        print(f"GPU name                  : {current_gpu}")
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU memory                : {total_memory_gb:.2f} GB")
    else:
        print("GPU name                  : CPU fallback")
        print("Canh bao                  : PyTorch hien tai chua nhan CUDA, model se train bang CPU.")


def print_training_header(model_name, cfg, dataset_info, total_params, trainable_params):
    print("\n" + "=" * 80)
    print(f"TRAINING MODEL: {model_name}")
    print("=" * 80)
    print(f"Device           : {DEVICE}")
    print(f"Image size       : {cfg['img_size']}")
    print(f"Batch size       : {BATCH_SIZE}")
    print(f"Epochs           : {NUM_EPOCHS}")
    print(f"Learning rate    : {cfg['lr']}")
    print(f"Total params     : {total_params:,}")
    print(f"Trainable params : {trainable_params:,}")
    print_dataset_overview(dataset_info)
    print("\nBat dau train...\n")

def train_one_model(model_name):
    if NUM_CLASSES == 0:
        raise ValueError('Không tìm thấy class nào trong thư mục Train.')

    cfg      = MODELS_CONFIG[model_name]
    img_size = cfg['img_size']
    lr       = cfg['lr']

    train_dl, val_dl, _, dataset_info = get_dataloaders(img_size, return_info=True)
    print_cuda_status()
    model     = get_model(model_name, NUM_CLASSES).to(DEVICE)
    total_params, trainable_params = count_parameters(model)
    print_training_header(model_name, cfg, dataset_info, total_params, trainable_params)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-3
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    EARLY_STOP_PATIENCE = 7
    history  = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[], 'lr':[], 'epoch_time_sec':[]}
    best_acc = 0.0
    best_epoch = 0
    no_improve_count = 0
    start    = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        # --- Train ---
        model.train()
        t_loss = t_correct = t_total = 0
        for inputs, labels in train_dl:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            if model_name == 'inceptionv3':
                out, aux = model(inputs)
                loss = criterion(out, labels) + 0.4 * criterion(aux, labels)
            else:
                out  = model(inputs)
                loss = criterion(out, labels)
            loss.backward(); optimizer.step()
            t_loss   += loss.item()
            _, preds  = torch.max(out, 1)
            t_correct += (preds == labels).sum().item()
            t_total   += labels.size(0)

        # --- Validation ---
        model.eval()
        v_loss = v_correct = v_total = 0
        with torch.no_grad():
            for inputs, labels in val_dl:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                out   = model(inputs)
                loss  = criterion(out, labels)
                v_loss   += loss.item()
                _, preds  = torch.max(out, 1)
                v_correct += (preds == labels).sum().item()
                v_total   += labels.size(0)

        scheduler.step()
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        t_acc = 100 * t_correct / t_total
        v_acc = 100 * v_correct / v_total
        history['train_loss'].append(t_loss / len(train_dl))
        history['val_loss'].append(v_loss / len(val_dl))
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)
        history['lr'].append(current_lr)
        history['epoch_time_sec'].append(epoch_time)

        if v_acc > best_acc:
            best_acc = v_acc
            best_epoch = epoch + 1
            no_improve_count = 0
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            torch.save(model.state_dict(),
                       f'{MODEL_SAVE_DIR}/best_{model_name}.pth')
        else:
            no_improve_count += 1

        print(
            f"[{model_name}] Epoch {epoch+1:02d}/{NUM_EPOCHS} "
            f"| train_loss={history['train_loss'][-1]:.4f} "
            f"| val_loss={history['val_loss'][-1]:.4f} "
            f"| train_acc={t_acc:.2f}% "
            f"| val_acc={v_acc:.2f}% "
            f"| lr={current_lr:.6f} "
            f"| time={epoch_time/60:.2f}m"
            f"| no_improve={no_improve_count}/{EARLY_STOP_PATIENCE}"
              + (" ★" if v_acc == best_acc else ""))

        if no_improve_count >= EARLY_STOP_PATIENCE:
            print(f"\n[Early Stopping] Val accuracy không cải thiện sau {EARLY_STOP_PATIENCE} epoch. Dừng sớm.")
            break

    elapsed = (time.time() - start) / 60
    training_config = {
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'lr': lr,
        'img_size': img_size,
        'device': str(DEVICE),
        'total_params': total_params,
        'trainable_params': trainable_params,
    }
    training_summary = build_training_summary(model_name, history, dataset_info, training_config)
    save_history(model_name, history)
    plot_curves(model_name, history)
    plot_training_summary(model_name, history, dataset_info, training_summary)

    print("\n" + "=" * 80)
    print(f"KET THUC TRAIN MODEL: {model_name}")
    print("=" * 80)
    print(f"Best validation accuracy : {best_acc:.2f}% (epoch {best_epoch})")
    print(f"Final train accuracy     : {history['train_acc'][-1]:.2f}%")
    print(f"Final validation acc     : {history['val_acc'][-1]:.2f}%")
    print(f"Final train loss         : {history['train_loss'][-1]:.4f}")
    print(f"Final validation loss    : {history['val_loss'][-1]:.4f}")
    print(f"Tong thoi gian train     : {elapsed:.1f} phut")
    print(f"Model tot nhat           : {MODEL_SAVE_DIR / f'best_{model_name}.pth'}")
    print(f"Do thi loss/acc          : results/curves_{model_name}.png")
    print(f"So do tong ket           : results/summary_{model_name}.png")
    print("=" * 80 + "\n")
    return best_acc, elapsed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model phan loai mon an.')
    parser.add_argument(
        '--model',
        default=DEFAULT_MODEL_NAME,
        choices=MODELS_CONFIG.keys(),
        help='Model train truoc. Mac dinh la efficientnet_b3.',
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Train toan bo model trong MODELS_CONFIG.',
    )
    args = parser.parse_args()

    models_to_train = MODELS_CONFIG.keys() if args.all else [args.model]
    for model_name in models_to_train:
        train_one_model(model_name)

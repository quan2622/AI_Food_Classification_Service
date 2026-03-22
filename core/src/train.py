# src/train.py
import torch, time, os, sys
import torch.nn as nn
import torch.optim as optim
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import DEVICE, NUM_EPOCHS, NUM_CLASSES, MODELS_CONFIG, MODEL_SAVE_DIR
from src.dataset import get_dataloaders
from src.models import get_model
from src.utils import save_history, plot_curves

def train_one_model(model_name):
    cfg      = MODELS_CONFIG[model_name]
    img_size = cfg['img_size']
    lr       = cfg['lr']

    train_dl, val_dl, _ = get_dataloaders(img_size)
    model     = get_model(model_name, NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    history  = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
    best_acc = 0.0
    start    = time.time()

    for epoch in range(NUM_EPOCHS):
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
        t_acc = 100 * t_correct / t_total
        v_acc = 100 * v_correct / v_total
        history['train_loss'].append(t_loss / len(train_dl))
        history['val_loss'].append(v_loss / len(val_dl))
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)

        if v_acc > best_acc:
            best_acc = v_acc
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            torch.save(model.state_dict(),
                       f'{MODEL_SAVE_DIR}/best_{model_name}.pth')

        print(f"[{model_name}] Epoch {epoch+1:02d}/{NUM_EPOCHS} "
              f"| Train: {t_acc:.2f}% | Val: {v_acc:.2f}%"
              + (" ★" if v_acc == best_acc else ""))

    elapsed = (time.time() - start) / 60
    save_history(model_name, history)
    plot_curves(model_name, history)
    print(f"\n{model_name} done — Best: {best_acc:.2f}% | {elapsed:.1f} phút\n")
    return best_acc, elapsed


if __name__ == '__main__':
    for model_name in MODELS_CONFIG:
        train_one_model(model_name)
# src/predict.py
from contextlib import nullcontext
from threading import Lock
from typing import Any

import torch
import torchvision.transforms as transforms
from PIL import Image
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import CLASS_NAMES, DEFAULT_MODEL_NAME, DEVICE, MODEL_SAVE_DIR, MODELS_CONFIG, NUM_CLASSES
from src.models import get_model

def build_transform(model_name):
    if model_name not in MODELS_CONFIG:
        raise ValueError(f"Model '{model_name}' không được hỗ trợ.")

    img_size = MODELS_CONFIG[model_name]['img_size']
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])


def load_model_for_inference(model_name=DEFAULT_MODEL_NAME):
    if model_name not in MODELS_CONFIG:
        raise ValueError(f"Model '{model_name}' không được hỗ trợ.")

    model_path = MODEL_SAVE_DIR / f'best_{model_name}.pth'
    if not model_path.exists():
        raise FileNotFoundError(f'Chưa tìm thấy model đã train: {model_path}')

    model = get_model(model_name, NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def load_available_models():
    loaded_models = {}
    for model_name in MODELS_CONFIG:
        model_path = MODEL_SAVE_DIR / f'best_{model_name}.pth'
        if model_path.exists():
            loaded_models[model_name] = load_model_for_inference(model_name)
    return loaded_models


def warmup_model(model, model_name=DEFAULT_MODEL_NAME):
    transform = build_transform(model_name)
    img_size = MODELS_CONFIG[model_name]['img_size']
    dummy_input = torch.zeros((1, 3, img_size, img_size), device=DEVICE)
    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)
    return transform


def predict_with_model(image_path, model, model_name=DEFAULT_MODEL_NAME, model_lock: Any = None):
    transform = build_transform(model_name)
    img = Image.open(image_path).convert('RGB')
    inp = transform(img).unsqueeze(0).to(DEVICE)

    lock_context = model_lock if model_lock is not None else nullcontext()
    with lock_context:
        with torch.no_grad():
            probs = torch.softmax(model(inp), dim=1)[0]

    top3_prob, top3_idx = torch.topk(probs, 3)
    predictions = []
    print(f"\nKết quả dự đoán cho: {image_path}")
    print("-" * 35)
    for i, (p, idx) in enumerate(zip(top3_prob, top3_idx)):
        tag = " ← DỰ ĐOÁN" if i == 0 else ""
        prediction = {
            'rank': i + 1,
            'class_name': CLASS_NAMES[idx.item()],
            'confidence': round(p.item(), 6),
        }
        predictions.append(prediction)
        print(f"  {i+1}. {prediction['class_name']:<20} {p.item()*100:.1f}%{tag}")

    return {
        'image_path': str(image_path),
        'model_name': model_name,
        'top1': predictions[0],
        'predictions': predictions,
    }


def predict(image_path, model_name=DEFAULT_MODEL_NAME):
    model = load_model_for_inference(model_name)
    return predict_with_model(image_path, model, model_name=model_name)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Predict 1 anh bang model da train.')
    parser.add_argument('image_path', help='Duong dan toi anh can du doan.')
    parser.add_argument(
        '--model',
        default=DEFAULT_MODEL_NAME,
        choices=MODELS_CONFIG.keys(),
        help='Model dung de du doan.',
    )
    args = parser.parse_args()

    predict(args.image_path, model_name=args.model)

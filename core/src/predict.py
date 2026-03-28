# src/predict.py
import torch
import torchvision.transforms as transforms
from PIL import Image
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import CLASS_NAMES, DEFAULT_MODEL_NAME, DEVICE, MODEL_SAVE_DIR, MODELS_CONFIG, NUM_CLASSES
from src.models import get_model

def predict(image_path, model_name=DEFAULT_MODEL_NAME):
    if model_name not in MODELS_CONFIG:
        raise ValueError(f"Model '{model_name}' không được hỗ trợ.")

    img_size = MODELS_CONFIG[model_name]['img_size']
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    model = get_model(model_name, NUM_CLASSES).to(DEVICE)
    model_path = MODEL_SAVE_DIR / f'best_{model_name}.pth'
    if not model_path.exists():
        raise FileNotFoundError(f'Chưa tìm thấy model đã train: {model_path}')

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    img = Image.open(image_path).convert('RGB')
    inp = transform(img).unsqueeze(0).to(DEVICE)

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

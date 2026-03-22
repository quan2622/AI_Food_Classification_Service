# src/predict.py
import torch
import torchvision.transforms as transforms
from PIL import Image
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import DEVICE, NUM_CLASSES, CLASS_NAMES, MODEL_SAVE_DIR
from src.models import get_model

def predict(image_path, model_name='efficientnet_b3', img_size=300):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    model = get_model(model_name, NUM_CLASSES).to(DEVICE)
    model.load_state_dict(
        torch.load(f'{MODEL_SAVE_DIR}/best_{model_name}.pth',
                   map_location=DEVICE)
    )
    model.eval()

    img = Image.open(image_path).convert('RGB')
    inp = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(inp), dim=1)[0]

    top3_prob, top3_idx = torch.topk(probs, 3)
    print(f"\nKết quả dự đoán cho: {image_path}")
    print("-" * 35)
    for i, (p, idx) in enumerate(zip(top3_prob, top3_idx)):
        tag = " ← DỰ ĐOÁN" if i == 0 else ""
        print(f"  {i+1}. {CLASS_NAMES[idx.item()]:<15} {p.item()*100:.1f}%{tag}")

if __name__ == '__main__':
    predict('test_food.jpg')
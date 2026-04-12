# src/models.py
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import PRETRAINED_WEIGHTS


def load_local_pretrained_weights(model, model_name):
    weight_path = PRETRAINED_WEIGHTS.get(model_name)
    if weight_path is None:
        return model

    weight_path = Path(weight_path)
    if not weight_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy pretrained weight cho '{model_name}'. "
            f"Hãy đặt file vào: {weight_path}"
        )

    state_dict = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(state_dict)
    return model

def get_model(name, num_classes):
    if name == 'efficientnet_b3':
        m = models.efficientnet_b3(weights=None)
        m = load_local_pretrained_weights(m, name)
        for layer in list(m.features.children())[:6]:
            for p in layer.parameters():
                p.requires_grad = False
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_f, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    elif name == 'resnet50':
        m = models.resnet50(weights=None)
        m = load_local_pretrained_weights(m, name)
        for name_l, param in m.named_parameters():
            if not any(x in name_l for x in ['layer3','layer4','fc']):
                param.requires_grad = False
        in_f = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_f, num_classes))

    elif name == 'inceptionv3':
        m = models.inception_v3(weights=None, aux_logits=True)
        m = load_local_pretrained_weights(m, name)
        for param in m.parameters():
            param.requires_grad = False
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        m.AuxLogits.fc = nn.Linear(768, num_classes)
        for param in m.fc.parameters():
            param.requires_grad = True
        for param in m.AuxLogits.fc.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Model '{name}' không tồn tại!")

    return m

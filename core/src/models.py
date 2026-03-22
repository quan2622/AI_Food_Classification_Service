# src/models.py
import torch.nn as nn
import torchvision.models as models

def get_model(name, num_classes):
    if name == 'efficientnet_b3':
        m = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
        )
        for layer in list(m.features.children())[:6]:
            for p in layer.parameters():
                p.requires_grad = False
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(in_f, 512),
            nn.ReLU(),       nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    elif name == 'resnet50':
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for name_l, param in m.named_parameters():
            if not any(x in name_l for x in ['layer3','layer4','fc']):
                param.requires_grad = False
        in_f = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_f, num_classes))

    elif name == 'inceptionv3':
        m = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1,
            aux_logits=True
        )
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
# src/dataset.py
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import BATCH_SIZE, resolve_split_dir

def get_transforms(img_size):
    train_tf = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return train_tf, val_tf

def summarize_dataset(dataset):
    idx_to_class = {idx: name for name, idx in dataset.class_to_idx.items()}
    class_counts = {idx_to_class[idx]: 0 for idx in idx_to_class}
    for _, label_idx in dataset.samples:
        class_counts[idx_to_class[label_idx]] += 1
    return class_counts


def get_dataloaders(img_size, return_info=False):
    train_tf, val_tf = get_transforms(img_size)
    train_dir = resolve_split_dir('train')
    val_dir = resolve_split_dir('validate', 'val')
    test_dir = resolve_split_dir('test')

    if not train_dir or not val_dir or not test_dir:
        raise FileNotFoundError(
            'Dataset phải có đủ 3 thư mục Train/Validate/Test (không phân biệt hoa thường).'
        )

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)
    test_ds = datasets.ImageFolder(str(test_dir), transform=val_tf)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    if not return_info:
        return train_dl, val_dl, test_dl

    dataset_info = {
        'class_names': train_ds.classes,
        'train_counts': summarize_dataset(train_ds),
        'val_counts': summarize_dataset(val_ds),
        'test_counts': summarize_dataset(test_ds),
        'train_size': len(train_ds),
        'val_size': len(val_ds),
        'test_size': len(test_ds),
        'train_batches': len(train_dl),
        'val_batches': len(val_dl),
        'test_batches': len(test_dl),
    }
    return train_dl, val_dl, test_dl, dataset_info

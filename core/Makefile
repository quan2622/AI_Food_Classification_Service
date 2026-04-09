.PHONY: help setup install train train-all train-quick train-resnet evaluate predict api api-prod split-dataset clean

PYTHON := python3
VENV := .venv
VENV_BIN := $(VENV)/bin
PIP := $(VENV_BIN)/pip
PYTHON_VENV := $(VENV_BIN)/python

# Epochs override: make train EPOCHS=5
EPOCHS ?= 30
MODEL  ?= efficientnet_b3
HOST ?= 0.0.0.0
PORT ?= 8000

help:
	@echo "AI Core Service - Available commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make setup      - Tạo virtual environment"
	@echo "  make install    - Cài thư viện từ requirements.txt"
	@echo ""
	@echo "Development:"
	@echo "  make api        - Chạy FastAPI (reload mode) tại http://$(HOST):$(PORT)"
	@echo "  make api-prod   - Chạy FastAPI (production, không reload)"
	@echo ""
	@echo "Training:"
	@echo "  make train              - Train efficientnet_b3 (30 epoch)"
	@echo "  make train EPOCHS=5     - Train với số epoch tuỳ chọn"
	@echo "  make train-quick        - Train resnet50 5 epoch (test nhanh CPU)"
	@echo "  make train-resnet       - Train resnet50 đầy đủ (nhẹ hơn CPU)"
	@echo "  make train-all          - Train tất cả models"
	@echo ""
	@echo "Evaluation & Inference:"
	@echo "  make evaluate   - Đánh giá model efficientnet_b3"
	@echo "  make predict    - Dự đoán ảnh (IMAGE=path/to/image.jpg)"
	@echo ""
	@echo "Dataset:"
	@echo "  make split-dataset SRC=path/to/raw  - Chia ảnh thô → dataset/train/val/test (70/15/15)"
	@echo ""
	@echo "  make clean      - Xóa virtualenv và cache"
	@echo ""
	@echo "Ví dụ:"
	@echo "  make setup && make install && make api"
	@echo "  make api PORT=8001"
	@echo "  make predict IMAGE=data/test.jpg"

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "========================================="
	@echo "Setup xong! Chạy 'make api' để khởi động."
	@echo "========================================="

install:
	$(PIP) install -r requirements.txt

train:
	$(PYTHON_VENV) src/train.py --model $(MODEL) --epochs $(EPOCHS)

train-quick:
	$(PYTHON_VENV) src/train.py --model resnet50 --epochs 5

train-resnet:
	$(PYTHON_VENV) src/train.py --model resnet50 --epochs $(EPOCHS)

train-all:
	$(PYTHON_VENV) src/train.py --all

evaluate:
	$(PYTHON_VENV) src/evaluate.py --model efficientnet_b3

predict:
ifndef IMAGE
	@echo "Error: IMAGE is not set. Usage: make predict IMAGE=path/to/image.jpg"
	@exit 1
endif
	$(PYTHON_VENV) src/predict.py $(IMAGE) --model efficientnet_b3

api:
	$(VENV_BIN)/uvicorn src.api:app --host $(HOST) --port $(PORT) --reload

api-prod:
	$(VENV_BIN)/uvicorn src.api:app --host $(HOST) --port $(PORT) --workers 2

# ─── Dataset ──────────────────────────────────────────────────────────────────
# Chia thư mục ảnh gốc (SRC) thành train/val/test
# Cách dùng: make split-dataset SRC=path/to/raw_images
# Tỷ lệ mặc định: 70/15/15
split-dataset:
ifndef SRC
	@echo "Lỗi: SRC chưa được đặt."
	@echo "Cách dùng: make split-dataset SRC=path/to/raw_images"
	@exit 1
endif
	$(PYTHON_VENV) -c "\
import os, shutil, random; \
src='$(SRC)'; dst='dataset'; \
splits={'train':0.70,'val':0.15,'test':0.15}; \
classes=[d for d in os.listdir(src) if os.path.isdir(os.path.join(src,d))]; \
[os.makedirs(os.path.join(dst,sp,cl), exist_ok=True) for sp in splits for cl in classes]; \
[random.shuffle(imgs := [f for f in os.listdir(os.path.join(src,cl)) if f.lower().endswith(('.jpg','.jpeg','.png'))]) or \
 [shutil.copy2(os.path.join(src,cl,f), os.path.join(dst, \
  ('train' if i < int(len(imgs)*0.70) else 'val' if i < int(len(imgs)*0.85) else 'test'), cl, f)) \
  for i,f in enumerate(imgs)] \
 for cl in classes]; \
print('Split xong! Cấu trúc dataset/ đã sẵn sàng.')"

clean:
	rm -rf $(VENV)
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Cleaned up virtual environment and cache files"

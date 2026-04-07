.PHONY: help setup install train train-all evaluate predict api clean

PYTHON := python3
VENV := .venv
VENV_BIN := $(VENV)/bin
PIP := $(VENV_BIN)/pip
PYTHON_VENV := $(VENV_BIN)/python

help:
	@echo "Available commands:"
	@echo "  make setup      - Create virtual environment"
	@echo "  make install    - Install all dependencies from requirements.txt"
	@echo "  make train      - Train single model (efficientnet_b3)"
	@echo "  make train-all  - Train all models"
	@echo "  make evaluate   - Evaluate model (efficientnet_b3)"
	@echo "  make predict    - Predict single image (set IMAGE=path/to/image.jpg)"
	@echo "  make api        - Run FastAPI server with uvicorn"
	@echo "  make clean      - Remove virtual environment and cache files"

setup:
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created at $(VENV)"
	@echo "Run 'source $(VENV)/bin/activate' to activate"

install:
	$(PIP) install -r requirements.txt

train:
	$(PYTHON_VENV) src/train.py --model efficientnet_b3

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
	$(VENV_BIN)/uvicorn src.api:app --reload

clean:
	rm -rf $(VENV)
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Cleaned up virtual environment and cache files"

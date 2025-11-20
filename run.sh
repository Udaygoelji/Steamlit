#!/usr/bin/env bash
set -e
python3 -m venv .venv || python -m venv .venv
source .venv/bin/activate || . .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python scripts/prepare_dataset.py
python src/train.py --epochs 5
echo "Training complete. Testing prediction on a validation image..."
python src/predict.py --image_path "$(ls data/animals/val/*/* | head -n 1)"

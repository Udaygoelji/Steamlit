# Animal Species Prediction (CIFAR-10 Subset)

This is a complete, working project for **Animal Species Prediction** using a subset of the CIFAR-10 dataset (animals only: `bird`, `cat`, `deer`, `dog`, `frog`, `horse`).

The project includes:
- Data preparation script that **auto-downloads** CIFAR-10 and exports a tidy folder dataset.
- PyTorch training script (transfer learning on a small model).
- Inference script for single-image predictions.
- Reproducible environment via `requirements.txt`.
- One-click runner scripts (`run.sh` / `run.bat`).

> Note: The dataset is **downloaded automatically** on first run (CIFAR-10, ~170 MB). It will be exported into `data/animals/` with a train/val split. This keeps the zip small and installation easy.

---

## Quickstart (Windows, PowerShell)

```powershell
python -m venv .venv
.\\.venv\\Scripts\\activate
pip install -r requirements.txt
python scripts/prepare_dataset.py
python src/train.py
python src/predict.py --image_path sample_images/dog.jpg
```

## Quickstart (Linux/macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_dataset.py
python src/train.py
python src/predict.py --image_path sample_images/dog.jpg
```

---

## Project Structure

```
animal-species-prediction/
├── data/
│   └── animals/
│       ├── train/
│       └── val/
├── models/
├── sample_images/
├── scripts/
│   └── prepare_dataset.py
├── src/
│   ├── dataset.py
│   ├── train.py
│   └── predict.py
├── requirements.txt
├── run.sh
├── run.bat
└── README.md
```

---

## Classes

We keep these **6 animal classes** from CIFAR-10:

```
bird, cat, deer, dog, frog, horse
```

---

## Tips

- First training run may take a while as it compiles code and caches datasets.
- If you **don’t have a GPU**, it will automatically use CPU.
- You can change hyperparameters (epochs, batch size, lr) via CLI args in `train.py`.

---

## License

- Code: MIT
- Dataset: CIFAR-10 (per its terms). Downloaded via `torchvision`.

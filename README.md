**Project README (clear & concise)**

This repository contains code for predicting building damage grades (three classes) under earthquake and it is partial code of my project 412. The README below is a focused, easy-to-read English version based only on the current `data/`, `models/`, `src/`, and `tools/` folders.

--

1) Quick overview

- Data: `data/train.csv`, `data/test.csv`, `data/submission.csv` (output)
- Models: checkpoints stored in `models/` (e.g. `best_model_fold_0.pth`)
- Code: training/inference/ensemble scripts in `src/`
- Utilities: small helper scripts in `tools/`

2) Quick start (PowerShell)

Change to repo root:
```powershell
cd "..\ Data Mining\Project"
```

Generate a submission (recommended fast path):
```powershell
python src\hgb_focused_solution.py
```

Train the neural network (longer):
```powershell
python src\net_optimized_edition.py
```

Run a quick ensemble using existing NN checkpoints:
```powershell
python src\quick_ensemble.py
```

Run diagnostics (per-fold metrics, confusion matrices):
```powershell
python src\deep_diagnosis.py
```

Check model checkpoints:
```powershell
python tools\check_checkpoint.py
```

3) Key scripts (what each does)

- `src/hgb_focused_solution.py` — Train 5-fold HistGradientBoosting models, average probabilities, optionally boost class-3 probability, write `data/submission.csv`.
- `src/train_fast.py` — Fast end-to-end training example (CV, save checkpoints). Good for quick baseline runs.
- `src/net_optimized_edition.py` — Full neural network training pipeline (mixup, EMA, focal loss options). Use for deeper NN experiments.
- `src/quick_ensemble.py` — Load NN checkpoints (if present) and combine NN + HGB probabilities via simple weighted average to produce a submission.
- `src/deep_diagnosis.py` — Produce per-fold/per-class metrics and confusion matrices to help debugging and model analysis.

4) Tools

- `tools/check_checkpoint.py` — Verify `.pth` files in `models/` can be loaded.
- `tools/compute_f1.py` (if present) — Compute weighted F1 locally (requires true labels).

5) Models

- Checkpoints are typically named `best_model_fold_{i}.pth` (i = 0..4). Neural nets save `state_dict` — load them with the corresponding model class defined in `src/`.

Example load snippet:
```python
import torch
from src.net_optimized_edition import ImprovedNet

model = ImprovedNet(input_dim=...)  # adjust args as defined in file
model.load_state_dict(torch.load('models/best_model_fold_0.pth', map_location='cpu'))
model.eval()
```

6) Environment & dependencies

- Recommended Python: 3.9–3.11
- Minimal packages:
```powershell
pip install pandas numpy scikit-learn torch torchvision matplotlib seaborn
```

If you want, I can generate a `requirements.txt` for pinned versions.

7) Common edits (where to change things)

- HGB hyperparameters: edit `src/hgb_focused_solution.py` (e.g. `learning_rate`, `max_iter`).
- NN hyperparameters: edit `src/net_optimized_edition.py` or `src/train_fast.py` (`lr`, `batch_size`, `epochs`).
- Ensemble / post-processing: edit `src/quick_ensemble.py` or `src/hgb_focused_solution.py` (weights, class boost factors).

8) Recommended workflow

1. Test pipelines on a small subset of data to validate preprocessing.
2. Run HGB baseline: `python src\hgb_focused_solution.py`.
3. Optionally train NN: `python src\net_optimized_edition.py` and save checkpoints to `models/`.
4. Run ensemble: `python src\quick_ensemble.py`.
5. Diagnose and iterate: `python src\deep_diagnosis.py`.



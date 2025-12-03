import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OptimizedNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.bn_input = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 3)

    def forward(self, x):
        x = self.bn_input(x)
        x = F.gelu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.gelu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.gelu(self.bn3(self.fc3(x)))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.gelu(self.fc4(x))
        x = F.dropout(x, p=0.2, training=self.training)
        return self.fc_out(x)


# Load data
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

cat_cols = train_df.select_dtypes(include=["object"]).columns.tolist()
for col in cat_cols:
    train_df[col] = train_df[col].astype("category").cat.codes
    test_df[col] = test_df[col].astype("category").cat.codes

X = train_df.drop(["building_id", "damage_grade"], axis=1).values.astype(np.float32)
y = (train_df["damage_grade"].values - 1).astype(int)

scaler = StandardScaler()
X = scaler.fit_transform(X)

print("=" * 60)
print("VALIDATION F1 SCORES ACROSS FOLDS")
print("=" * 60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_f1_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_val = X[val_idx]
    y_val = y[val_idx]

    model_path = f"best_model_fold_{fold}.pth"
    try:
        model = OptimizedNet(X.shape[1]).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        X_val_t = torch.from_numpy(X_val).to(device)
        with torch.no_grad():
            logits = model(X_val_t)
            val_preds = torch.argmax(logits, dim=1).cpu().numpy()

        f1_weighted = f1_score(y_val, val_preds, average="weighted", zero_division=0)
        f1_macro = f1_score(y_val, val_preds, average="macro", zero_division=0)
        fold_f1_scores.append((f1_weighted, f1_macro))

        print(f"Fold {fold+1}:")
        print(f"  F1 (weighted): {f1_weighted:.4f}")
        print(f"  F1 (macro):    {f1_macro:.4f}")

    except FileNotFoundError:
        print(f"Fold {fold+1}: Model not found")
    except Exception as e:
        print(f"Fold {fold+1}: Error - {str(e)[:100]}")

if fold_f1_scores:
    avg_f1_w = np.mean([x[0] for x in fold_f1_scores])
    avg_f1_m = np.mean([x[1] for x in fold_f1_scores])
    print(f"\n{'='*60}")
    print(f"AVERAGE VALIDATION F1:")
    print(f"  Weighted: {avg_f1_w:.4f}")
    print(f"  Macro:    {avg_f1_m:.4f}")
    print(f"{'='*60}")

print(f"\nSubmission file status:")
import os

if os.path.exists("data/submission.csv"):
    sub = pd.read_csv("data/submission.csv")
    print(f"  Rows: {len(sub)}")
    print(f"  Columns: {list(sub.columns)}")
    print(f"  Class distribution:")
    for cls in [1, 2, 3]:
        count = (sub["damage_grade"] == cls).sum()
        pct = 100 * count / len(sub)
        print(f"    Class {cls}: {count:4d} ({pct:5.1f}%)")
else:
    print("  File not found!")

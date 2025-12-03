"""
AGGRESSIVE FIX: Focus on improving Class 3 detection
Strategy: Temperature scaling + optimized thresholds + HGB-heavy ensemble
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import HistGradientBoostingClassifier

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


def temperature_scale(logits, temperature=1.0):
    """Apply temperature scaling to logits"""
    return F.softmax(logits / temperature, dim=1)


# Load data
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
test_ids = test_df["building_id"].values

cat_cols = train_df.select_dtypes(include=["object"]).columns.tolist()
for col in cat_cols:
    train_df[col] = train_df[col].astype("category").cat.codes
    test_df[col] = test_df[col].astype("category").cat.codes

X = train_df.drop(["building_id", "damage_grade"], axis=1).values.astype(np.float32)
y = (train_df["damage_grade"].values - 1).astype(int)
X_test = test_df.drop(["building_id"], axis=1).values.astype(np.float32)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

print("=" * 70)
print("AGGRESSIVE FIX: Class 3 Detection + HGB-Heavy Ensemble")
print("=" * 70)

# Find optimal temperature and ensemble weights through validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_config = None
best_f1 = 0

print("\n[1] Tuning temperature and ensemble weights...")

temperatures = [0.8, 1.0, 1.2, 1.5, 2.0]
hgb_weights = [0.6, 0.7, 0.8]

for temp in temperatures:
    for hgb_w in hgb_weights:
        nn_w = 1.0 - hgb_w
        fold_f1s = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # Load NN
            model = OptimizedNet(X.shape[1]).to(device)
            try:
                model.load_state_dict(
                    torch.load(f"best_model_fold_{fold}.pth", map_location=device)
                )
                model.eval()

                X_val_t = torch.from_numpy(X_val).to(device)
                with torch.no_grad():
                    nn_logits = model(X_val_t)
                    # Apply temperature scaling
                    nn_probs = (
                        temperature_scale(nn_logits, temperature=temp).cpu().numpy()
                    )

            except:
                continue

            # HGB
            hgb = HistGradientBoostingClassifier(
                max_iter=350, learning_rate=0.09, random_state=42
            )
            hgb.fit(X_tr, y_tr)
            hgb_probs = hgb.predict_proba(X_val)

            # Ensemble
            ensemble_probs = nn_w * nn_probs + hgb_w * hgb_probs
            ensemble_preds = np.argmax(ensemble_probs, axis=1)
            f1 = f1_score(y_val, ensemble_preds, average="weighted", zero_division=0)
            fold_f1s.append(f1)

        if fold_f1s:
            avg_f1 = np.mean(fold_f1s)
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_config = {"temp": temp, "hgb_w": hgb_w, "nn_w": nn_w}

print(
    f"Best config found: T={best_config['temp']}, NN={best_config['nn_w']:.1f}, HGB={best_config['hgb_w']:.1f}"
)
print(f"Best F1: {best_f1:.4f}")

# Generate final submission with best config
print(f"\n[2] Generating final submission...")

all_test_nn = []
all_test_hgb = []

for fold in range(5):
    X_tr = X[
        list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y))[
            fold
        ][0]
    ]
    y_tr = y[
        list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y))[
            fold
        ][0]
    ]

    # NN
    model = OptimizedNet(X.shape[1]).to(device)
    try:
        model.load_state_dict(
            torch.load(f"best_model_fold_{fold}.pth", map_location=device)
        )
        model.eval()

        X_test_t = torch.from_numpy(X_test).to(device)
        with torch.no_grad():
            test_logits = model(X_test_t)
            test_nn = (
                temperature_scale(test_logits, temperature=best_config["temp"])
                .cpu()
                .numpy()
            )
        all_test_nn.append(test_nn)
    except:
        pass

    # HGB
    hgb = HistGradientBoostingClassifier(
        max_iter=350, learning_rate=0.09, random_state=42
    )
    hgb.fit(X_tr, y_tr)
    test_hgb = hgb.predict_proba(X_test)
    all_test_hgb.append(test_hgb)

avg_nn = np.mean(all_test_nn, axis=0)
avg_hgb = np.mean(all_test_hgb, axis=0)

# Apply class-specific boosting for Class 3
final_probs = best_config["nn_w"] * avg_nn + best_config["hgb_w"] * avg_hgb

# Boost Class 3 probability specifically
final_probs[:, 2] *= 1.8  # Aggressive boost for Class 3
final_probs = final_probs / final_probs.sum(axis=1, keepdims=True)

final_preds = np.argmax(final_probs, axis=1) + 1

# Save
submission = pd.DataFrame({"building_id": test_ids, "damage_grade": final_preds})
submission.to_csv("data/submission.csv", index=False)

print(f"\n{'-'*70}")
print("FINAL SUBMISSION")
print(f"{'-'*70}")
print(f"Configuration:")
print(f"  Temperature: {best_config['temp']}")
print(f"  NN weight: {best_config['nn_w']:.1f}, HGB weight: {best_config['hgb_w']:.1f}")
print(f"  Class 3 boost: 1.8x")
print(f"  Validation F1: {best_f1:.4f}")

dist = np.bincount(final_preds, minlength=4)[1:4]
print(f"\nClass distribution:")
for cls in range(3):
    print(f"  Class {cls+1}: {dist[cls]:4d} ({100*dist[cls]/1000:5.1f}%)")

print(f"\nFile: data/submission.csv")

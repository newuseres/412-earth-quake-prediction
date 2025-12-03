"""
Simple but effective: Focus on HGB + Class 3 boost
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
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
print("OPTIMIZED SOLUTION: HGB-Heavy + Class 3 Boost")
print("=" * 70)

# Collect predictions
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_hgb_preds = []

print("\n[1] Collecting HGB predictions from all folds...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr = X[train_idx]
    y_tr = y[train_idx]

    # Train HGB with aggressive settings
    hgb = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.08,
        max_leaf_nodes=50,  # Increase capacity
        l2_regularization=0.01,  # Reduce regularization
        random_state=42,
        max_bins=300,
    )
    hgb.fit(X_tr, y_tr)
    test_probs = hgb.predict_proba(X_test)
    test_hgb_preds.append(test_probs)
    print(f"  Fold {fold+1}: OK")

# Average HGB predictions
avg_hgb = np.mean(test_hgb_preds, axis=0)

print(f"\n[2] Applying Class 3 boost and threshold optimization...")

# Strategy: Use HGB as main model, but boost Class 3 aggressively
boosted_probs = avg_hgb.copy()

# Apply class-specific boosts based on diagnosis
# Class 1: Already ~60% accuracy, apply moderate boost
# Class 2: Good (~65% accuracy), minimal boost
# Class 3: Poor (~20-30% accuracy), AGGRESSIVE boost
boosted_probs[:, 0] *= 0.9  # Class 1: slightly reduce
boosted_probs[:, 1] *= 1.0  # Class 2: keep
boosted_probs[:, 2] *= 2.5  # Class 3: aggressive boost

# Renormalize
boosted_probs = boosted_probs / boosted_probs.sum(axis=1, keepdims=True)

# Generate predictions
final_preds = np.argmax(boosted_probs, axis=1) + 1

# Save submission
submission = pd.DataFrame({"building_id": test_ids, "damage_grade": final_preds})
submission.to_csv("data/submission.csv", index=False)

print(f"\n{'-'*70}")
print("SUBMISSION GENERATED")
print(f"{'-'*70}")

dist = np.bincount(final_preds, minlength=4)[1:4]
print(f"Final class distribution:")
for cls in range(3):
    print(f"  Class {cls+1}: {dist[cls]:4d} ({100*dist[cls]/1000:5.1f}%)")

print(f"\nStrategy:")
print(f"  Model: HGB (HistGradientBoostingClassifier)")
print(f"  Ensemble: 5-fold averaging")
print(f"  Class boosts: C1=0.9x, C2=1.0x, C3=2.5x")
print(f"  Focus: Maximize Class 3 detection (critical weakness)")

print(f"\nFile: data/submission.csv (ready for submission)")

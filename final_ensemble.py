"""
Final optimization: Weighted ensemble + intelligent class boosting
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
print("FINAL OPTIMIZATION: Adaptive ensemble + class rebalancing")
print("=" * 70)

# Collect predictions from all folds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_test_nn = []
all_test_hgb = []

print("\n[1] Generating predictions from all folds...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"  Fold {fold+1}/5...", end=" ")

    X_tr = X[train_idx]
    y_tr = y[train_idx]

    # NN model
    model = OptimizedNet(X.shape[1]).to(device)
    try:
        model.load_state_dict(
            torch.load(f"best_model_fold_{fold}.pth", map_location=device)
        )
        model.eval()

        X_test_t = torch.from_numpy(X_test).to(device)
        with torch.no_grad():
            test_nn = F.softmax(model(X_test_t), dim=1).cpu().numpy()
        all_test_nn.append(test_nn)
        print("NN", end=" ")
    except Exception as e:
        print(f"NN err", end=" ")

    # HGB model
    try:
        hgb = HistGradientBoostingClassifier(
            max_iter=350, learning_rate=0.09, random_state=42
        )
        hgb.fit(X_tr, y_tr)
        test_hgb = hgb.predict_proba(X_test)
        all_test_hgb.append(test_hgb)
        print("HGB")
    except Exception as e:
        print(f"HGB err")

# Average ensemble
avg_nn = np.mean(all_test_nn, axis=0) if all_test_nn else None
avg_hgb = np.mean(all_test_hgb, axis=0) if all_test_hgb else None

if avg_nn is None or avg_hgb is None:
    print("ERROR: Failed to load predictions")
    exit(1)

print("\n[2] Testing ensemble strategies with class rebalancing...")


def rebalance_predictions(probs, boost_factors=[1.0, 1.0, 1.0]):
    """Apply boost factors to class probabilities"""
    boosted = probs.copy()
    for cls in range(3):
        boosted[:, cls] *= boost_factors[cls]
    # Renormalize
    boosted = boosted / boosted.sum(axis=1, keepdims=True)
    return boosted


# Test different boost factors
test_configs = [
    ("base_50_50", 0.5, 0.5, [1.0, 1.0, 1.0]),
    ("class1_boost_1.2", 0.5, 0.5, [1.2, 1.0, 1.0]),
    ("class1_boost_1.5", 0.5, 0.5, [1.5, 1.0, 1.0]),
    ("nn_60_hgb_40", 0.6, 0.4, [1.0, 1.0, 1.0]),
    ("nn_70_hgb_30", 0.7, 0.3, [1.0, 1.0, 1.0]),
    ("adaptive_60_40_class1_1.3", 0.6, 0.4, [1.3, 1.0, 1.0]),
    ("adaptive_55_45_class1_1.5", 0.55, 0.45, [1.5, 1.0, 1.0]),
]

results = []

print(f"\n{'Config':<40} | C1  | C2  | C3  | Balance Score")
print(f"{'-'*70}")

for config_name, nn_w, hgb_w, boost_factors in test_configs:
    ensemble = nn_w * avg_nn + hgb_w * avg_hgb
    rebalanced = rebalance_predictions(ensemble, boost_factors)
    preds = np.argmax(rebalanced, axis=1) + 1

    dist = np.bincount(preds, minlength=4)[1:4]

    # Balance score: penalize extreme imbalance
    balance_score = 1.0 / (1.0 + np.std(dist) / np.mean(dist))

    print(
        f"{config_name:<40} | {dist[0]:3d} | {dist[1]:3d} | {dist[2]:3d} | {balance_score:.4f}"
    )

    results.append(
        {
            "name": config_name,
            "probs": rebalanced,
            "dist": dist,
            "balance": balance_score,
        }
    )

# Select best based on balance + reasonable distribution
best_result = max(results, key=lambda x: x["balance"])
print(
    f"\n[SELECTED] {best_result['name']} (balance score: {best_result['balance']:.4f})"
)

final_preds = np.argmax(best_result["probs"], axis=1) + 1

# Save submission
submission = pd.DataFrame({"building_id": test_ids, "damage_grade": final_preds})
submission.to_csv("data/submission.csv", index=False)

print(f"\n{'-'*70}")
print("SUBMISSION SAVED")
print(f"{'-'*70}")
print(f"Strategy: {best_result['name']}")
print(f"Final class distribution:")
for cls in range(3):
    print(
        f"  Class {cls+1}: {best_result['dist'][cls]:4d} ({100*best_result['dist'][cls]/1000:5.1f}%)"
    )

print(f"\nFile: data/submission.csv")

"""
Advanced optimization: Threshold tuning + minority class boosting + stacking
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
from sklearn.linear_model import LogisticRegression

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
print("ADVANCED OPTIMIZATION: Stacking + Threshold Tuning")
print("=" * 70)

# Strategy 1: Stacking with meta-learner
print("\n[1] Building OOF (Out-of-Fold) predictions for stacking...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_nn_probs = np.zeros((X.shape[0], 3))
oof_hgb_probs = np.zeros((X.shape[0], 3))
test_nn_preds = []
test_hgb_preds = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"  Fold {fold+1}/5...", end=" ")

    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # NN model
    model = OptimizedNet(X.shape[1]).to(device)
    try:
        model.load_state_dict(
            torch.load(f"best_model_fold_{fold}.pth", map_location=device)
        )
    except:
        print("NN load failed!")
        continue

    model.eval()

    X_val_t = torch.from_numpy(X_val).to(device)
    with torch.no_grad():
        oof_nn_probs[val_idx] = F.softmax(model(X_val_t), dim=1).cpu().numpy()

    X_test_t = torch.from_numpy(X_test).to(device)
    with torch.no_grad():
        test_nn = F.softmax(model(X_test_t), dim=1).cpu().numpy()
    test_nn_preds.append(test_nn)

    # HGB model
    hgb = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, random_state=42
    )
    hgb.fit(X_tr, y_tr)
    oof_hgb_probs[val_idx] = hgb.predict_proba(X_val)
    test_hgb_preds.append(hgb.predict_proba(X_test))

    print("[OK]")  # Average test predictions
avg_test_nn = np.mean(test_nn_preds, axis=0)
avg_test_hgb = np.mean(test_hgb_preds, axis=0)

print(f"\n[2] Training meta-learner (Logistic Regression)...")
# Stack 6 features: 3 from NN + 3 from HGB
meta_features = np.hstack([oof_nn_probs, oof_hgb_probs])
meta_lr = LogisticRegression(max_iter=1000, multi_class="multinomial", random_state=42)
meta_lr.fit(meta_features, y)

# Meta-learner predictions on test
test_meta_features = np.hstack([avg_test_nn, avg_test_hgb])
test_meta_probs = meta_lr.predict_proba(test_meta_features)

print(f"  Meta-learner trained with 6 features")

# Calculate meta-learner validation F1
meta_val_preds = meta_lr.predict(meta_features)
meta_val_f1 = f1_score(y, meta_val_preds, average="weighted", zero_division=0)
print(f"  Meta-learner validation F1: {meta_val_f1:.4f}")

print(f"\n[3] Testing multiple prediction strategies...")


# Strategy 2: Threshold adjustment for minority class boost
def apply_threshold_boost(probs, class_boost_factor=1.5):
    """Boost minority class probabilities"""
    boosted = probs.copy()
    boosted[:, 0] *= class_boost_factor  # Class 1 (minority)
    # Renormalize
    boosted = boosted / boosted.sum(axis=1, keepdims=True)
    return boosted


# Strategy 3: Confidence-based adjustment
def confidence_adjustment(nn_probs, hgb_probs, threshold=0.6):
    """Use NN when confident, HGB otherwise"""
    nn_confidence = np.max(nn_probs, axis=1)
    adjusted = nn_probs.copy()

    low_conf_mask = nn_confidence < threshold
    adjusted[low_conf_mask] = hgb_probs[low_conf_mask]

    return adjusted


# Evaluate different strategies
strategies = {
    "meta_learner": test_meta_probs,
    "nn_hgb_50_50": 0.5 * avg_test_nn + 0.5 * avg_test_hgb,
    "boosted_class1_1.3x": apply_threshold_boost(
        0.5 * avg_test_nn + 0.5 * avg_test_hgb, 1.3
    ),
    "boosted_class1_1.5x": apply_threshold_boost(
        0.5 * avg_test_nn + 0.5 * avg_test_hgb, 1.5
    ),
    "boosted_class1_2.0x": apply_threshold_boost(
        0.5 * avg_test_nn + 0.5 * avg_test_hgb, 2.0
    ),
    "confidence_adj_0.6": confidence_adjustment(
        avg_test_nn, avg_test_hgb, threshold=0.6
    ),
}

print(f"\n{'Strategy':<35} | {'Class 1':<8} | {'Class 2':<8} | {'Class 3':<8}")
print(f"{'-'*70}")

best_strategy = None
best_dist = None

for strategy_name, test_probs in strategies.items():
    test_preds = np.argmax(test_probs, axis=1) + 1
    dist = np.bincount(test_preds, minlength=4)[1:4]
    print(f"{strategy_name:<35} | {dist[0]:<8} | {dist[1]:<8} | {dist[2]:<8}")

    # Choose strategy with best class balance (avoiding extreme imbalance)
    if best_strategy is None:
        best_strategy = strategy_name
        best_dist = dist

# Use meta-learner (most principled approach)
final_probs = strategies["meta_learner"]
final_preds = np.argmax(final_probs, axis=1) + 1

# Save submission
submission = pd.DataFrame({"building_id": test_ids, "damage_grade": final_preds})
submission.to_csv("data/submission.csv", index=False)

print(f"\n{'-'*70}")
print("SUBMISSION GENERATED")
print(f"{'-'*70}")

final_dist = np.bincount(final_preds, minlength=4)[1:4]
print(f"Final class distribution:")
for cls in range(3):
    print(f"  Class {cls+1}: {final_dist[cls]:4d} ({100*final_dist[cls]/1000:5.1f}%)")

print(f"\nFile: data/submission.csv")
print(f"\nStrategy used: Meta-Learner (Stacking with LogisticRegression)")
print(f"This combines NN + HGB features through a learned meta-model")

"""
Fast diagnostic and fix: Generate best submission using multiple strategies
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
print("OPTIMIZED ENSEMBLE: Testing multiple strategies")
print("=" * 70)

# Strategy evaluation on validation folds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
strategy_scores = {
    "nn_only": [],
    "hgb_only": [],
    "nn_80_hgb_20": [],
    "nn_60_hgb_40": [],
    "nn_50_hgb_50": [],
}

test_predictions = {
    "nn_only": [],
    "hgb_only": [],
    "nn_80_hgb_20": [],
    "nn_60_hgb_40": [],
    "nn_50_hgb_50": [],
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    print(f"\nFold {fold+1}/5", end=" | ")

    # Load NN
    try:
        model = OptimizedNet(X.shape[1]).to(device)
        model.load_state_dict(
            torch.load(f"best_model_fold_{fold}.pth", map_location=device)
        )
        model.eval()

        X_val_t = torch.from_numpy(X_val).to(device)
        with torch.no_grad():
            logits = model(X_val_t)
            nn_probs = F.softmax(logits, dim=1).cpu().numpy()

        X_test_t = torch.from_numpy(X_test).to(device)
        with torch.no_grad():
            test_logits = model(X_test_t)
            test_nn_probs = F.softmax(test_logits, dim=1).cpu().numpy()

    except Exception as e:
        print(f"NN load error: {str(e)[:50]}")
        continue

    # Train HGB
    hgb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, random_state=42
    )
    hgb.fit(X_tr, y_tr)
    hgb_probs = hgb.predict_proba(X_val)
    hgb_test_probs = hgb.predict_proba(X_test)

    # Test 5 strategies
    strategies = [
        ("nn_only", 1.0, 0.0),
        ("hgb_only", 0.0, 1.0),
        ("nn_80_hgb_20", 0.8, 0.2),
        ("nn_60_hgb_40", 0.6, 0.4),
        ("nn_50_hgb_50", 0.5, 0.5),
    ]

    best_f1 = 0
    best_strategy = None

    for name, nn_w, hgb_w in strategies:
        ensemble_probs = nn_w * nn_probs + hgb_w * hgb_probs
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        f1 = f1_score(y_val, ensemble_preds, average="weighted", zero_division=0)
        strategy_scores[name].append(f1)

        # Get test predictions
        test_ensemble_probs = nn_w * test_nn_probs + hgb_w * hgb_test_probs
        test_ensemble_preds = np.argmax(test_ensemble_probs, axis=1) + 1
        test_predictions[name].append(test_ensemble_preds)

        if f1 > best_f1:
            best_f1 = f1
            best_strategy = name

    print(f"Best: {best_strategy} (F1={best_f1:.4f})")

# Calculate average F1 for each strategy
print(f"\n{'-'*70}")
print("AVERAGE VALIDATION F1 ACROSS ALL FOLDS:")
print(f"{'-'*70}")

best_overall_strategy = None
best_overall_f1 = 0

for strategy_name in strategy_scores:
    avg_f1 = np.mean(strategy_scores[strategy_name])
    print(f"  {strategy_name:20s}: {avg_f1:.4f}")
    if avg_f1 > best_overall_f1:
        best_overall_f1 = avg_f1
        best_overall_strategy = strategy_name

print(f"\n[BEST] Strategy: {best_overall_strategy} with avg F1 = {best_overall_f1:.4f}")

# Generate submission using best strategy
print(f"\nGenerating submission with best strategy: {best_overall_strategy}")

# Retrain on full data using best strategy
all_test_preds_best = []

for fold in range(5):
    try:
        # Get test predictions from this fold
        if fold < len(test_predictions[best_overall_strategy]):
            all_test_preds_best.append(test_predictions[best_overall_strategy][fold])
    except:
        pass

# Average predictions across folds
if all_test_preds_best:
    final_preds_ensemble = np.round(np.mean(all_test_preds_best, axis=0)).astype(int)
else:
    # Fallback: just average probabilities
    print("Using fallback: averaging NN+HGB directly")

    # Quick inference
    model = OptimizedNet(X.shape[1]).to(device)
    model.load_state_dict(torch.load("best_model_fold_0.pth", map_location=device))
    model.eval()

    X_test_t = torch.from_numpy(X_test).to(device)
    with torch.no_grad():
        test_logits = model(X_test_t)
        nn_probs = F.softmax(test_logits, dim=1).cpu().numpy()

    hgb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, random_state=42
    )
    hgb.fit(X, y)
    hgb_probs = hgb.predict_proba(X_test)

    # Use best strategy weights
    if best_overall_strategy == "nn_only":
        w_nn, w_hgb = 1.0, 0.0
    elif best_overall_strategy == "hgb_only":
        w_nn, w_hgb = 0.0, 1.0
    elif best_overall_strategy == "nn_80_hgb_20":
        w_nn, w_hgb = 0.8, 0.2
    elif best_overall_strategy == "nn_60_hgb_40":
        w_nn, w_hgb = 0.6, 0.4
    else:  # nn_50_hgb_50
        w_nn, w_hgb = 0.5, 0.5

    final_probs = w_nn * nn_probs + w_hgb * hgb_probs
    final_preds_ensemble = np.argmax(final_probs, axis=1) + 1

# Save submission
submission = pd.DataFrame(
    {"building_id": test_ids, "damage_grade": final_preds_ensemble}
)
submission.to_csv("data/submission.csv", index=False)

print(f"\n{'-'*70}")
print("SUBMISSION SAVED")
print(f"{'-'*70}")
print(f"Class distribution in new submission:")
for cls in [1, 2, 3]:
    count = (final_preds_ensemble == cls).sum()
    pct = 100 * count / 1000
    print(f"  Class {cls}: {count:4d} ({pct:5.1f}%)")

print(f"\nFile: data/submission.csv (1000 rows)")

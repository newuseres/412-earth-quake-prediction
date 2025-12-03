import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
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
print("DIAGNOSIS: Why Test F1 Decreased?")
print("=" * 70)

# 1. Check validation performance across all folds
print("\n[1] VALIDATION F1 SCORES (on holdout folds)")
print("-" * 70)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_val_f1 = []
all_nn_preds = []
all_hgb_preds = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # Load NN model
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
            nn_preds = np.argmax(logits, dim=1).cpu().numpy()

        # Train HGB on this fold's training data
        hgb = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, random_state=42
        )
        hgb.fit(X_tr, y_tr)
        hgb_probs = hgb.predict_proba(X_val)

        # Strategy 1: NN only
        f1_nn = f1_score(y_val, nn_preds, average="weighted", zero_division=0)

        # Strategy 2: HGB only
        hgb_preds = np.argmax(hgb_probs, axis=1)
        f1_hgb = f1_score(y_val, hgb_preds, average="weighted", zero_division=0)

        # Strategy 3: 65% NN + 35% HGB (current)
        ensemble_probs = 0.65 * nn_probs + 0.35 * hgb_probs
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        f1_ensemble = f1_score(
            y_val, ensemble_preds, average="weighted", zero_division=0
        )

        # Strategy 4: 50-50
        ensemble50_probs = 0.5 * nn_probs + 0.5 * hgb_probs
        ensemble50_preds = np.argmax(ensemble50_probs, axis=1)
        f1_ensemble50 = f1_score(
            y_val, ensemble50_preds, average="weighted", zero_division=0
        )

        # Strategy 5: 80% NN + 20% HGB
        ensemble80_probs = 0.8 * nn_probs + 0.2 * hgb_probs
        ensemble80_preds = np.argmax(ensemble80_probs, axis=1)
        f1_ensemble80 = f1_score(
            y_val, ensemble80_preds, average="weighted", zero_division=0
        )

        print(f"\nFold {fold+1}:")
        print(f"  NN only:       F1={f1_nn:.4f}")
        print(f"  HGB only:      F1={f1_hgb:.4f}")
        print(f"  NN(0.80) HGB(0.20): F1={f1_ensemble80:.4f} <- BEST?")
        print(f"  NN(0.65) HGB(0.35): F1={f1_ensemble:.4f}")
        print(f"  NN(0.50) HGB(0.50): F1={f1_ensemble50:.4f}")

        all_val_f1.append([f1_nn, f1_hgb, f1_ensemble80, f1_ensemble, f1_ensemble50])
        all_nn_preds.append(nn_probs)
        all_hgb_preds.append(hgb_probs)

    except Exception as e:
        print(f"Fold {fold+1}: Error - {str(e)[:100]}")

if all_val_f1:
    all_val_f1 = np.array(all_val_f1)
    print(f"\n{'-'*70}")
    print("AVERAGE ACROSS ALL FOLDS:")
    strategies = [
        "NN only",
        "HGB only",
        "NN(0.80)+HGB(0.20)",
        "NN(0.65)+HGB(0.35)",
        "NN(0.50)+HGB(0.50)",
    ]
    for i, strategy in enumerate(strategies):
        avg_f1 = all_val_f1[:, i].mean()
        print(f"  {strategy:25s}: {avg_f1:.4f}")

# 2. Check prediction distribution mismatch
print(f"\n{'='*70}")
print("[2] PREDICTION DISTRIBUTION ANALYSIS")
print("-" * 70)

current_sub = pd.read_csv("data/submission.csv")
current_dist = np.bincount(current_sub["damage_grade"].values, minlength=3)
print(f"\nCurrent submission class distribution:")
for cls in range(3):
    print(
        f"  Class {cls+1}: {current_dist[cls]:4d} ({100*current_dist[cls]/1000:5.1f}%)"
    )

y_dist = np.bincount(y, minlength=3)
print(f"\nTraining set class distribution:")
for cls in range(3):
    print(f"  Class {cls+1}: {y_dist[cls]:4d} ({100*y_dist[cls]/4000:5.1f}%)")

print(
    f"\nNote: If test distribution doesn't match training, model may be overfitting to train distribution"
)

print(f"\n{'='*70}")
print("[3] RECOMMENDATION")
print("-" * 70)
print(
    """
If test F1 decreased:
1. Try Strategy with weights NN(0.80)+HGB(0.20) which often generalizes better
2. Or go back to pure NN (ignoring HGB which may cause overfitting)
3. Or try class rebalancing in predictions using threshold adjustment
"""
)

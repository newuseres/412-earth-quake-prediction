"""
Deep diagnosis: Analyze model confidence, calibration, and potential issues
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report
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
print("DEEP DIAGNOSIS: Model Performance Analysis")
print("=" * 70)

# Analyze on validation set
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_metrics = []
model_confidences = []
hgb_confidences = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n[Fold {fold+1}] Analysis:")
    print("-" * 70)

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
            nn_probs = F.softmax(nn_logits, dim=1).cpu().numpy()

        nn_preds = np.argmax(nn_probs, axis=1)
        nn_confidence = np.max(nn_probs, axis=1)
        nn_f1 = f1_score(y_val, nn_preds, average="weighted", zero_division=0)

        print(f"NN Model:")
        print(f"  F1 Score (weighted): {nn_f1:.4f}")
        print(f"  Avg Confidence: {nn_confidence.mean():.4f}")
        print(
            f"  Min/Max Confidence: {nn_confidence.min():.4f} / {nn_confidence.max():.4f}"
        )
        print(f"  Accuracy: {(nn_preds == y_val).sum() / len(y_val):.4f}")

        # Analyze per-class performance
        for cls in range(3):
            mask = y_val == cls
            if mask.sum() > 0:
                cls_acc = (nn_preds[mask] == y_val[mask]).sum() / mask.sum()
                print(
                    f"    Class {cls+1} Accuracy: {cls_acc:.4f} ({mask.sum()} samples)"
                )

        model_confidences.append(nn_confidence)

    except Exception as e:
        print(f"NN load failed: {e}")

    # Load HGB
    hgb = HistGradientBoostingClassifier(
        max_iter=350, learning_rate=0.09, random_state=42
    )
    hgb.fit(X_tr, y_tr)
    hgb_probs = hgb.predict_proba(X_val)
    hgb_preds = np.argmax(hgb_probs, axis=1)
    hgb_confidence = np.max(hgb_probs, axis=1)
    hgb_f1 = f1_score(y_val, hgb_preds, average="weighted", zero_division=0)

    print(f"\nHGB Model:")
    print(f"  F1 Score (weighted): {hgb_f1:.4f}")
    print(f"  Avg Confidence: {hgb_confidence.mean():.4f}")
    print(f"  Accuracy: {(hgb_preds == y_val).sum() / len(y_val):.4f}")

    # Test ensemble
    ensemble_probs = 0.5 * nn_probs + 0.5 * hgb_probs
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    ensemble_f1 = f1_score(y_val, ensemble_preds, average="weighted", zero_division=0)

    print(f"\nEnsemble (50-50):")
    print(f"  F1 Score (weighted): {ensemble_f1:.4f}")
    print(
        f"  Improvement: {(ensemble_f1 - max(nn_f1, hgb_f1)) / max(nn_f1, hgb_f1) * 100:+.2f}%"
    )

    # Confusion analysis
    print(f"\nConfusion Matrix (Ensemble):")
    cm = confusion_matrix(y_val, ensemble_preds)
    for i in range(3):
        correct = cm[i, i]
        total = cm[i].sum()
        print(f"  Class {i+1}: {correct}/{total} correct ({100*correct/total:.1f}%)")

    all_metrics.append(
        {"fold": fold + 1, "nn_f1": nn_f1, "hgb_f1": hgb_f1, "ensemble_f1": ensemble_f1}
    )

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")

metrics_df = pd.DataFrame(all_metrics)
print(f"\nAverage F1 scores across folds:")
print(
    f"  NN only:       {metrics_df['nn_f1'].mean():.4f} (±{metrics_df['nn_f1'].std():.4f})"
)
print(
    f"  HGB only:      {metrics_df['hgb_f1'].mean():.4f} (±{metrics_df['hgb_f1'].std():.4f})"
)
print(
    f"  Ensemble 50-50:{metrics_df['ensemble_f1'].mean():.4f} (±{metrics_df['ensemble_f1'].std():.4f})"
)

print(f"\n[DIAGNOSIS]")
print(
    f"""
Key findings:
1. NN likely overfitting to training data
2. HGB more stable but lower ceiling
3. Ensemble should help, but may need better weighting

Root causes of low test performance:
- Class imbalance not fully addressed
- Threshold may be misaligned with test distribution
- Possible feature shift between train/test
- Confidence calibration issues
"""
)

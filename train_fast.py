import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


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


# Load data once
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

print(f"X={X.shape}, y_dist={np.bincount(y)}")

set_seed(42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_test_preds = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold+1}/5 {'='*30}")

    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # Create model
    model = OptimizedNet(X.shape[1]).to(device)

    # Class weights
    cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    cw[1] *= 2.0
    cw_t = torch.tensor(cw, dtype=torch.float32).to(device)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=cw_t)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2
    )

    best_f1 = 0
    patience = 20
    no_improve = 0

    # Convert to tensors once
    X_tr_t = torch.from_numpy(X_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).long().to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).long().to(device)
    X_test_t = torch.from_numpy(X_test).to(device)

    for epoch in range(100):
        model.train()
        # Mini-batch training
        for i in range(0, len(X_tr), 32):
            end = min(i + 32, len(X_tr))
            xb = X_tr_t[i:end]
            yb = y_tr_t[i:end]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validate every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t)
                val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()

            f1 = f1_score(y_val, val_preds, average="weighted", zero_division=0)
            print(f"  E{epoch+1:3d}: F1={f1:.4f}", end="")

            if f1 > best_f1 + 1e-4:
                best_f1 = f1
                torch.save(model.state_dict(), f"best_model_fold_{fold}.pth")
                no_improve = 0
                print(" [SAVE]")
            else:
                no_improve += 1
                print()
                if no_improve >= patience:
                    print(f"  Early stop")
                    break

    # Test predictions
    model.load_state_dict(torch.load(f"best_model_fold_{fold}.pth"))
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t)
        test_probs = F.softmax(test_logits, dim=1).cpu().numpy()
    all_test_preds.append(test_probs)
    print(f"Fold {fold+1} -> Best F1={best_f1:.4f}")

# Final ensemble
avg_probs = np.mean(all_test_preds, axis=0)
final_preds = np.argmax(avg_probs, axis=1) + 1

submission = pd.DataFrame({"building_id": test_ids, "damage_grade": final_preds})
submission.to_csv("data/submission.csv", index=False)

print(f"\n{'='*50}")
print("Submission saved!")
for cls in [1, 2, 3]:
    count = (final_preds == cls).sum()
    print(f"  Class {cls}: {count:4d} ({100*count/1000:5.1f}%)")

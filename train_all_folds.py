import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
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


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=3.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


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


def mixup(xb, yb, alpha=0.3):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(xb.shape[0])
    return lam * xb + (1 - lam) * xb[idx], yb, yb[idx], lam


def train_all_folds():
    set_seed(42)

    # Load data
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    test_ids = test_df["building_id"].values

    # Preprocess
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

    print(f"Data shape: X={X.shape}, y_dist={np.bincount(y)}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_test_preds = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold+1}/5 {'='*40}")

        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Create model
        model = OptimizedNet(X.shape[1]).to(device)

        # Class weights (boost minority)
        cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
        cw[1] *= 2.0
        cw = torch.tensor(cw, dtype=torch.float32).to(device)

        # Weighted sampler
        class_counts = np.bincount(y_tr)
        sample_weights = 1.0 / np.power(class_counts + 1e-8, 0.6)
        sampler = WeightedRandomSampler(sample_weights[y_tr], len(y_tr) * 2, True)

        train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)
        val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
        criterion = FocalLoss(alpha=cw, gamma=3.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=15, T_mult=2, eta_min=1e-6
        )

        best_f1 = 0
        patience = 20
        no_improve = 0

        for epoch in range(100):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                xb_mix, ya, yb_idx, lam = mixup(xb, yb, alpha=0.4)

                optimizer.zero_grad()
                logits = model(xb_mix)
                loss = lam * criterion(logits, ya) + (1 - lam) * criterion(
                    logits, yb_idx
                )
                loss.backward()
                optimizer.step()

            scheduler.step()

            if (epoch + 1) % 10 == 0:
                model.eval()
                val_preds = []
                with torch.no_grad():
                    for xb, _ in val_loader:
                        xb = xb.to(device)
                        val_preds.extend(torch.argmax(model(xb), dim=1).cpu().numpy())

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
                        print(f"  Early stop (epoch {epoch+1})")
                        break

        # Test predictions
        model.load_state_dict(torch.load(f"best_model_fold_{fold}.pth"))
        model.eval()
        test_ds = TensorDataset(torch.from_numpy(X_test))
        test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
        test_probs = []
        with torch.no_grad():
            for (xb,) in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                test_probs.append(F.softmax(logits, dim=1).cpu().numpy())
        all_test_preds.append(np.vstack(test_probs))
        print(f"Fold {fold+1} complete. Best F1={best_f1:.4f}")

    # Ensemble predictions
    avg_probs = np.mean(all_test_preds, axis=0)
    final_preds = np.argmax(avg_probs, axis=1) + 1

    submission = pd.DataFrame({"building_id": test_ids, "damage_grade": final_preds})
    submission.to_csv("data/submission.csv", index=False)

    print(f"\n{'='*50}")
    print(f"Submission saved!")
    print(f"Class distribution:")
    for cls in [1, 2, 3]:
        count = (final_preds == cls).sum()
        print(f"  Class {cls}: {count:4d} ({100*count/1000:5.1f}%)")
    print(f"{'='*50}")


if __name__ == "__main__":
    train_all_folds()

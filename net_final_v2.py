import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import HistGradientBoostingClassifier


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FocalLoss(nn.Module):
    """Focal Loss targeting hard negatives."""

    def __init__(self, alpha=None, gamma=3.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        return focal_loss.sum()


class FinalNet(nn.Module):
    """Optimized network for F1."""

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
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.4)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.bn_input(x)
        x = F.gelu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.gelu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.gelu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = F.gelu(self.fc4(x))
        x = self.dropout4(x)
        return self.fc_out(x)


def mixup(x, y, alpha=0.3):
    """Mixup augmentation."""
    batch_size = x.shape[0]
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(batch_size)
    x_mix = lam * x + (1 - lam) * x[idx]
    return x_mix, y, y[idx], lam


def train_final():
    set_seed(42)

    # Load data
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    test_ids = test_df["building_id"].values

    # Preprocess
    cat_cols = train_df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        combined = pd.concat([train_df[col], test_df[col]], axis=0).astype("category")
        train_df[col] = pd.Categorical(
            train_df[col], categories=combined.cat.categories
        ).codes
        test_df[col] = pd.Categorical(
            test_df[col], categories=combined.cat.categories
        ).codes

    X = train_df.drop(["building_id", "damage_grade"], axis=1).values.astype(np.float32)
    y = (train_df["damage_grade"].values - 1).astype(int)
    X_test = test_df.drop(["building_id"], axis=1).values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    # 5-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_nn_preds = []
    all_hgb_preds = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*50}\nFold {fold+1}/5\n{'='*50}")

        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Heavy class weighting - boost minority
        cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
        cw[1] *= 2.0  # Extra boost for minority class
        class_weights = torch.tensor(cw, dtype=torch.float32).to(device)

        # Aggressive weighted sampler
        class_counts = np.bincount(y_tr)
        class_weights_np = 1.0 / np.power(class_counts + 1e-12, 0.6)
        sample_weights = class_weights_np[y_tr]
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(y_tr) * 2, replacement=True
        )

        train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)
        val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

        model = FinalNet(X.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
        criterion = FocalLoss(alpha=class_weights, gamma=3.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )

        best_f1 = 0
        patience = 25
        no_improve = 0

        for epoch in range(120):
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

            # Validation every 10 epochs
            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_preds = []
                    for xb, _ in val_loader:
                        xb = xb.to(device)
                        logits = model(xb)
                        val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

                f1 = f1_score(y_val, val_preds, average="weighted", zero_division=0)
                print(f"Epoch {epoch+1:3d}: Val F1={f1:.4f}")

                if f1 > best_f1 + 1e-4:
                    best_f1 = f1
                    torch.save(model.state_dict(), f"best_model_fold_{fold}.pth")
                    no_improve = 0
                    print(f"  -> Saved (F1={f1:.4f})")
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

        # Load best and evaluate
        model.load_state_dict(torch.load(f"best_model_fold_{fold}.pth"))
        model.eval()

        val_preds = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

        print(
            f"\nFold {fold+1} validation F1 (weighted): {f1_score(y_val, val_preds, average='weighted', zero_division=0):.4f}"
        )
        print(classification_report(y_val, val_preds, digits=4))

        # Predictions on test
        test_ds = TensorDataset(torch.from_numpy(X_test))
        test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
        nn_test_logits = []
        with torch.no_grad():
            for (xb,) in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                nn_test_logits.append(F.softmax(logits, dim=1).cpu().numpy())
        all_nn_preds.append(np.vstack(nn_test_logits))

        # Train HGB on this fold
        hgb = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, l2_regularization=0.1, random_state=42
        )
        hgb.fit(X_tr, y_tr)
        hgb_test_probs = hgb.predict_proba(X_test)
        all_hgb_preds.append(hgb_test_probs)

    # Ensemble predictions
    nn_avg = np.mean(all_nn_preds, axis=0)
    hgb_avg = np.mean(all_hgb_preds, axis=0)

    # Weighted average (favor NN which uses Focal Loss)
    final_probs = 0.65 * nn_avg + 0.35 * hgb_avg
    final_preds = np.argmax(final_probs, axis=1) + 1

    # Save
    pd.DataFrame({"building_id": test_ids, "damage_grade": final_preds}).to_csv(
        "data/submission.csv", index=False
    )
    print("\n" + "=" * 50)
    print(f"Submission saved. Class distribution: {np.bincount(final_preds)}")
    print("=" * 50)


if __name__ == "__main__":
    train_final()

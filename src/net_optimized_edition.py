import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
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
    """Focal Loss for handling class imbalance (reducing easy samples)."""

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
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
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class ImprovedNet(nn.Module):
    """Deeper and wider architecture with GELU activation."""

    def __init__(self, input_dim):
        super().__init__()
        self.bn_input = nn.BatchNorm1d(input_dim)

        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        self.layer3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        self.layer4 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.head = nn.Linear(64, 3)

    def forward(self, x):
        x = self.bn_input(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head(x)
        return x


def mixup_batch(x, y, alpha=0.2):
    """Apply Mixup data augmentation."""
    batch_size = x.shape[0]
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(batch_size)
    x_mixed = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return x_mixed, y_a, y_b, lam


def mixup_loss(criterion, logits, y_a, y_b, lam):
    """Compute Mixup loss."""
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


class EMA:
    """Exponential Moving Average for model weights."""

    def __init__(self, model, decay=0.99):
        self.model = model
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    self.decay * self.shadow[name].data + (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name].data


def create_sampler(y_train):
    class_count = np.bincount(y_train)
    class_weights = 1.0 / (class_count + 1e-12)
    sample_weights = class_weights[y_train]
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )


def train_and_evaluate():
    set_seed(42)

    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    test_building_id = test_df["building_id"].copy()

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
    y = (train_df["damage_grade"].values.astype(int) - 1).astype(int)
    X_test = test_df.drop(["building_id"], axis=1).values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_f1_scores = []
    test_logits_ensemble = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n=== Fold {fold+1}/{n_splits} ===")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_dataset = TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train)
        )
        val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

        sampler = create_sampler(y_train)
        train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        model = ImprovedNet(X.shape[1]).to(device)

        cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weights = torch.tensor(cw, dtype=torch.float32).to(device)

        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-2, weight_decay=1e-4, betas=(0.9, 0.999)
        )

        epochs = 120
        iters_per_epoch = max(1, len(train_loader))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-5
        )

        ema = EMA(model, decay=0.999)

        best_val_f1 = 0.0
        best_state = None
        patience = 25
        no_improve = 0

        for epoch in range(epochs):
            model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)

                # Apply Mixup
                xb_mix, y_a, y_b, lam = mixup_batch(xb, yb, alpha=0.3)

                optimizer.zero_grad()
                logits = model(xb_mix)
                loss = mixup_loss(criterion, logits, y_a, y_b, lam)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                ema.update()

                train_losses.append(loss.item())

            scheduler.step()

            # Validation
            model.eval()
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    logits = model(xb)
                    _, preds = torch.max(logits, 1)
                    val_preds.extend(preds.cpu().numpy())
                    val_targets.extend(yb.cpu().numpy())

            val_f1 = f1_score(
                val_targets, val_preds, average="weighted", zero_division=0
            )
            val_acc = accuracy_score(val_targets, val_preds)

            if (epoch + 1) % 10 == 0 or no_improve < 5:
                print(
                    f"Fold {fold+1} Epoch {epoch+1}/{epochs} - train_loss {np.mean(train_losses):.4f} val_f1 {val_f1:.4f} val_acc {val_acc:.4f}"
                )

            if val_f1 > best_val_f1 + 1e-4:
                best_val_f1 = val_f1
                best_state = model.state_dict()
                torch.save(best_state, f"best_model_fold_{fold}.pth")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best and evaluate
        model.load_state_dict(torch.load(f"best_model_fold_{fold}.pth"))
        model.eval()
        val_loader_full = DataLoader(val_dataset, batch_size=256, shuffle=False)
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for xb, yb in val_loader_full:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                _, preds = torch.max(logits, 1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(yb.cpu().numpy())

        f1 = f1_score(val_targets, val_preds, average="weighted", zero_division=0)
        fold_f1_scores.append(f1)
        print(f"Fold {fold+1} validation F1 (weighted): {f1:.4f}\n")

        # Predict test logits
        test_ds = TensorDataset(torch.from_numpy(X_test))
        test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
        logits_list = []
        with torch.no_grad():
            for (xb,) in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                logits_list.append(logits.cpu().numpy())
        logits_all = np.vstack(logits_list)
        test_logits_ensemble.append(logits_all)

    mean_f1 = np.mean(fold_f1_scores)
    print(f"\nMean CV F1 (weighted): {mean_f1:.4f}")

    # Ensemble: average logits across folds
    avg_logits = np.mean(np.stack(test_logits_ensemble, axis=0), axis=0)
    preds = np.argmax(avg_logits, axis=1) + 1

    submission = pd.DataFrame({"building_id": test_building_id, "damage_grade": preds})
    os.makedirs("data", exist_ok=True)
    submission.to_csv("data/submission.csv", index=False)
    print("Saved optimized ensemble submission to data/submission.csv")


if __name__ == "__main__":
    train_and_evaluate()

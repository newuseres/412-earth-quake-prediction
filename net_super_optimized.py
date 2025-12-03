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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FocalLoss(nn.Module):
    """Aggressive Focal Loss with strong focus on hard samples."""

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


class SuperNet(nn.Module):
    """Deeper architecture optimized for F1 score."""

    def __init__(self, input_dim):
        super().__init__()
        self.bn_input = nn.BatchNorm1d(input_dim)

        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 768),
            nn.BatchNorm1d(768),
            nn.GELU(),
            nn.Dropout(0.5),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),
        )

        self.layer3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        self.layer4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.layer5 = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.head = nn.Linear(64, 3)

    def forward(self, x):
        x = self.bn_input(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.head(x)
        return x


def mixup_batch(x, y, alpha=0.3):
    """Mixup augmentation."""
    batch_size = x.shape[0]
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(batch_size)
    x_mixed = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return x_mixed, y_a, y_b, lam


def mixup_loss(criterion, logits, y_a, y_b, lam):
    """Mixup loss computation."""
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


def create_aggressive_sampler(y_train):
    """Aggressive sampling: heavily oversample minority class."""
    class_count = np.bincount(y_train)
    # Make minority classes much more frequent
    class_weights = 1.0 / np.power(class_count + 1e-12, 0.5)
    sample_weights = class_weights[y_train]
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights) * 2, replacement=True
    )


def train_and_evaluate_super():
    """Enhanced training focused on maximizing F1 score."""
    set_seed(42)

    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    test_building_id = test_df["building_id"].copy()

    # Categorical encoding
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
    test_nn_logits = []
    oof_nn_probs = np.zeros((X.shape[0], 3))
    oof_hgb_probs = np.zeros((X.shape[0], 3))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n=== Fold {fold+1}/{n_splits} ===")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(f"Class distribution in train: {np.bincount(y_train)}")
        print(f"Class distribution in val: {np.bincount(y_val)}")

        train_dataset = TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train)
        )
        val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

        # Aggressive weighted sampling
        sampler = create_aggressive_sampler(y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

        model = SuperNet(X.shape[1]).to(device)

        # Aggressive class weights: boost minority classes heavily
        cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        # Further boost class 1 (minority)
        cw[1] *= 1.5
        class_weights = torch.tensor(cw, dtype=torch.float32).to(device)

        # Aggressive Focal Loss (gamma=3.0)
        criterion = FocalLoss(alpha=class_weights, gamma=3.0)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)

        epochs = 150
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=25, T_mult=2, eta_min=1e-6
        )

        best_val_f1 = 0.0
        best_state = None
        patience = 30
        no_improve = 0

        for epoch in range(epochs):
            model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)

                # Apply Mixup
                xb_mix, y_a, y_b, lam = mixup_batch(xb, yb, alpha=0.4)

                optimizer.zero_grad()
                logits = model(xb_mix)
                loss = mixup_loss(criterion, logits, y_a, y_b, lam)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_losses.append(loss.item())

            scheduler.step()

            # Validation
            model.eval()
            val_preds = []
            val_targets = []
            val_logits = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    logits = model(xb)
                    val_logits.append(logits.cpu().numpy())
                    _, preds = torch.max(logits, 1)
                    val_preds.extend(preds.cpu().numpy())
                    val_targets.extend(yb.cpu().numpy())

            val_f1_macro = f1_score(
                val_targets, val_preds, average="macro", zero_division=0
            )
            val_f1_weighted = f1_score(
                val_targets, val_preds, average="weighted", zero_division=0
            )

            if (epoch + 1) % 20 == 0 or no_improve < 5:
                print(
                    f"Epoch {epoch+1}/{epochs} - train_loss {np.mean(train_losses):.4f} val_f1_macro {val_f1_macro:.4f} val_f1_weighted {val_f1_weighted:.4f}"
                )

            if val_f1_macro > best_val_f1 + 1e-4:
                best_val_f1 = val_f1_macro
                best_state = model.state_dict()
                torch.save(best_state, f"best_model_fold_{fold}.pth")
                no_improve = 0
                print(f"  -> Saved checkpoint (F1 macro: {val_f1_macro:.4f})")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best and evaluate
        model.load_state_dict(torch.load(f"best_model_fold_{fold}.pth"))
        model.eval()

        val_loader_full = DataLoader(val_dataset, batch_size=512, shuffle=False)
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

        f1_macro = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        f1_weighted = f1_score(
            val_targets, val_preds, average="weighted", zero_division=0
        )
        fold_f1_scores.append(f1_macro)

        print(
            f"\nFold {fold+1} Final - F1 macro: {f1_macro:.4f}, F1 weighted: {f1_weighted:.4f}"
        )
        print(f"Classification report (val):")
        print(classification_report(val_targets, val_preds))

        # NN test predictions
        test_ds = TensorDataset(torch.from_numpy(X_test))
        test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
        test_logits = []
        with torch.no_grad():
            for (xb,) in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                test_logits.append(logits.cpu().numpy())
        test_logits_all = np.vstack(test_logits)
        test_nn_logits.append(test_logits_all)

        # OOF NN probs for stacking
        val_probs = torch.softmax(
            torch.from_numpy(
                np.vstack(
                    [
                        logits
                        for logits in [
                            model(torch.from_numpy(X_val[i : i + 512]).to(device)).cpu()
                            for i in range(0, len(X_val), 512)
                        ]
                    ]
                )
            ),
            dim=1,
        ).numpy()
        oof_nn_probs[val_idx] = val_probs

        # HGB on this fold's train set
        hgb = HistGradientBoostingClassifier(
            random_state=42,
            max_iter=500,
            learning_rate=0.1,
            max_leaf_nodes=31,
            l2_regularization=1e-1,
        )
        hgb.fit(X_train, y_train)
        oof_hgb_probs[val_idx] = hgb.predict_proba(X_val)

    # Ensemble with stacking
    mean_f1 = np.mean(fold_f1_scores)
    print(f"\n{'='*60}")
    print(f"Mean CV F1 (macro): {mean_f1:.4f}")
    print(f"{'='*60}")

    # Average NN logits
    avg_nn_logits = np.mean(np.stack(test_nn_logits, axis=0), axis=0)

    # Train HGB on full data for final predictions
    hgb_final = HistGradientBoostingClassifier(random_state=42, max_iter=500)
    hgb_final.fit(X, y)
    hgb_test_probs = hgb_final.predict_proba(X_test)

    # Convert logits to probs
    def softmax(a):
        e = np.exp(a - np.max(a, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    nn_test_probs = softmax(avg_nn_logits)

    # Strategy: Weighted ensemble favoring NN (which uses aggressive Focal Loss)
    final_probs = 0.7 * nn_test_probs + 0.3 * hgb_test_probs
    final_preds = np.argmax(final_probs, axis=1) + 1

    submission = pd.DataFrame(
        {"building_id": test_building_id, "damage_grade": final_preds}
    )
    os.makedirs("data", exist_ok=True)
    submission.to_csv("data/submission.csv", index=False)

    print(f"Prediction distribution in submission:")
    print(f"  Class 1: {(final_preds == 1).sum()}")
    print(f"  Class 2: {(final_preds == 2).sum()}")
    print(f"  Class 3: {(final_preds == 3).sum()}")
    print(f"\nSaved super-optimized submission to data/submission.csv")


if __name__ == "__main__":
    train_and_evaluate_super()

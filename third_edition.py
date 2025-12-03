import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import HistGradientBoostingClassifier


# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TabularNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        return self.net(x)


class MyNetOld(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    def forward(self, x):
        return self.layers(x)


class CheckpointNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 3),
        )

    def forward(self, x):
        return self.layers(x)


def try_load_state(model, state):
    try:
        model.load_state_dict(state)
        return True
    except Exception:
        pass

    model_state = model.state_dict()
    assigned = set()
    new_state = {}
    for mkey, mt in model_state.items():
        new_state[mkey] = mt

    for skey, sval in state.items():
        for mkey, mt in model_state.items():
            if mkey in assigned:
                continue
            try:
                if mt.shape == sval.shape:
                    new_state[mkey] = sval
                    assigned.add(mkey)
                    break
            except Exception:
                continue

    try:
        model.load_state_dict(new_state)
        return True
    except Exception:
        return False


def load_model_from_state(state, input_dim):
    model = CheckpointNet(input_dim)
    if try_load_state(model, state):
        return model
    model = TabularNet(input_dim)
    model.layers = model.net
    if try_load_state(model, state):
        return model
    model = MyNetOld(input_dim)
    model.net = model.layers
    if try_load_state(model, state):
        return model
    raise RuntimeError("Cannot map checkpoint to known model architecture")


def create_sampler(y_train):
    # sample weights inversely proportional to class frequency
    classes = np.unique(y_train)
    class_count = np.bincount(y_train)
    class_weights = 1.0 / (class_count + 1e-12)
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
    return sampler


def train_and_evaluate():
    set_seed(42)

    # load data
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    test_building_id = test_df["building_id"].copy()

    # consistent categorical encoding: combine then transform
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

    fold_accuracies = []
    test_logits_ensemble = []

    # If all fold checkpoints exist, skip training and load logits from them
    checkpoints_exist = all(
        os.path.exists(f"best_model_fold_{i}.pth") for i in range(n_splits)
    )
    if checkpoints_exist:
        print(
            "Found existing best_model_fold_*.pth files — skipping NN training and loading checkpoints."
        )
        test_ds = TensorDataset(torch.from_numpy(X_test))
        test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
        for fold in range(n_splits):
            state = torch.load(f"best_model_fold_{fold}.pth", map_location=device)
            model = load_model_from_state(state, X.shape[1])
            model.to(device)
            model.eval()
            logits_list = []
            with torch.no_grad():
                for (xb,) in test_loader:
                    xb = xb.to(device)
                    logits = model(xb)
                    logits_list.append(logits.cpu().numpy())
            logits_all = np.vstack(logits_list)
            test_logits_ensemble.append(logits_all)

        if len(test_logits_ensemble) == 0:
            raise RuntimeError("Loaded checkpoints but failed to produce logits.")

        # We cannot compute fold_accuracies here because we skipped per-fold validation runs
        mean_acc = None
        print("Loaded NN checkpoints and produced test logits.")
    else:
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n=== Fold {fold+1}/{n_splits} ===")
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            train_dataset = TensorDataset(
                torch.from_numpy(X_train), torch.from_numpy(y_train)
            )
            val_dataset = TensorDataset(
                torch.from_numpy(X_val), torch.from_numpy(y_val)
            )

            sampler = create_sampler(y_train)
            train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler)
            val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

            model = TabularNet(X.shape[1]).to(device)

            # compute balanced class weights for loss
            cw = compute_class_weight(
                class_weight="balanced", classes=np.unique(y_train), y=y_train
            )
            class_weights = torch.tensor(cw, dtype=torch.float32).to(device)

            # Use label_smoothing + class weights
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=1e-3, weight_decay=1e-5
            )

            # OneCycleLR requires total_steps; approximate with epochs*iters
            epochs = 80
            iters_per_epoch = max(1, len(train_loader))
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=1e-2, steps_per_epoch=iters_per_epoch, epochs=epochs
            )

            best_val_loss = float("inf")
            best_state = None
            patience = 20
            no_improve = 0

            for epoch in range(epochs):
                model.train()
                train_losses = []
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    optimizer.zero_grad()
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    train_losses.append(loss.item())

                # validation
                model.eval()
                val_loss = 0.0
                val_preds = []
                val_targets = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        logits = model(xb)
                        loss = criterion(logits, yb)
                        val_loss += loss.item()
                        _, preds = torch.max(logits, 1)
                        val_preds.extend(preds.cpu().numpy())
                        val_targets.extend(yb.cpu().numpy())

                val_loss /= len(val_loader)
                val_acc = accuracy_score(val_targets, val_preds)
                print(
                    f"Fold {fold+1} Epoch {epoch+1}/{epochs} - train_loss {np.mean(train_losses):.4f} val_loss {val_loss:.4f} val_acc {val_acc:.4f}"
                )

                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    best_state = model.state_dict()
                    torch.save(best_state, f"best_model_fold_{fold}.pth")
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

            # load best and evaluate full val
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

            acc = accuracy_score(val_targets, val_preds)
            fold_accuracies.append(acc)
            print(f"Fold {fold+1} validation accuracy: {acc:.4f}")

            # predict test logits for ensemble
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

    if len(fold_accuracies) > 0:
        mean_acc = np.mean(fold_accuracies)
        print(f"\nMean CV accuracy: {mean_acc:.4f}")
    else:
        print("No CV fold accuracies (we loaded existing checkpoints).")

    # ensemble: average logits across folds
    avg_logits = np.mean(np.stack(test_logits_ensemble, axis=0), axis=0)

    # convert logits to probabilities (softmax)
    def softmax(a):
        e = np.exp(a - np.max(a, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    nn_probs = softmax(avg_logits)

    # Train a simple tree model on full data and get its probabilities
    print(
        "Training HistGradientBoostingClassifier on full training set for ensembling..."
    )
    hgb = HistGradientBoostingClassifier(random_state=42)
    try:
        hgb.fit(X, y)
        tree_probs = hgb.predict_proba(X_test)
    except Exception as e:
        print("Warning: failed to train HGB, falling back to NN-only submission.", e)
        tree_probs = None

    if tree_probs is not None:
        alpha = 0.6
        final_probs = alpha * nn_probs + (1 - alpha) * tree_probs
        final_preds = np.argmax(final_probs, axis=1) + 1
        print(f"Ensembling NN + HGB with weight alpha={alpha}")
    else:
        final_preds = np.argmax(nn_probs, axis=1) + 1

    submission = pd.DataFrame(
        {"building_id": test_building_id, "damage_grade": final_preds}
    )
    os.makedirs("data", exist_ok=True)
    submission.to_csv("data/submission.csv", index=False)
    print("Saved ensemble submission to data/submission.csv")


if __name__ == "__main__":
    train_and_evaluate()

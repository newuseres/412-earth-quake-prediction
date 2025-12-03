import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


class ImprovedNet(nn.Module):
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
    model = ImprovedNet(input_dim)
    if try_load_state(model, state):
        return model
    raise RuntimeError("Cannot load model from state")


def diagnose():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    print("=" * 60)
    print("DIAGNOSTIC: Optimized NN Performance on Validation Folds")
    print("=" * 60)

    all_val_preds = []
    all_val_targets = []
    test_nn_probs = np.zeros((X_test.shape[0], 3))
    test_hgb_probs = np.zeros((X_test.shape[0], 3))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Load NN checkpoint
        ckpt = f"best_model_fold_{fold}.pth"
        if not os.path.exists(ckpt):
            print(f"Checkpoint {ckpt} not found")
            continue

        state = torch.load(ckpt, map_location=device)
        model = load_model_from_state(state, X.shape[1])
        model.to(device)
        model.eval()

        # NN validation predictions
        val_ds = TensorDataset(torch.from_numpy(X_val))
        val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
        nn_probs_list = []
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                nn_probs_list.append(probs)
        nn_val_probs = np.vstack(nn_probs_list)
        nn_val_preds = np.argmax(nn_val_probs, axis=1)

        # HGB on this fold
        hgb = HistGradientBoostingClassifier(random_state=42, max_iter=500)
        hgb.fit(X_train, y_train)
        hgb_val_probs = hgb.predict_proba(X_val)
        hgb_val_preds = hgb.predict(X_val)

        # Test predictions
        test_ds = TensorDataset(torch.from_numpy(X_test))
        test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
        nn_test_probs_list = []
        with torch.no_grad():
            for (xb,) in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                nn_test_probs_list.append(probs)
        nn_test_probs = np.vstack(nn_test_probs_list)
        test_nn_probs += nn_test_probs / n_splits
        test_hgb_probs += hgb.predict_proba(X_test) / n_splits

        # Fold metrics
        nn_f1 = f1_score(y_val, nn_val_preds, average="weighted", zero_division=0)
        hgb_f1 = f1_score(y_val, hgb_val_preds, average="weighted", zero_division=0)

        print(f"\nFold {fold+1}:")
        print(f"  NN F1 (weighted):  {nn_f1:.4f}")
        print(f"  HGB F1 (weighted): {hgb_f1:.4f}")
        print(
            f"  NN pred dist: {np.bincount(nn_val_preds, minlength=3)} / {len(y_val)}"
        )
        print(
            f"  HGB pred dist: {np.bincount(hgb_val_preds, minlength=3)} / {len(y_val)}"
        )

        all_val_preds.extend(nn_val_preds)
        all_val_targets.extend(y_val)

    # Overall NN F1 on validation
    overall_nn_f1 = f1_score(
        all_val_targets, all_val_preds, average="weighted", zero_division=0
    )
    print(f"\n{'='*60}")
    print(f"Overall NN F1 (weighted) on validation folds: {overall_nn_f1:.4f}")
    print(f"Class distribution in validation targets: {np.bincount(all_val_targets)}")
    print(f"Class distribution in NN predictions: {np.bincount(all_val_preds)}")
    print(f"{'='*60}")

    # Strategy 1: Use only NN (argmax)
    nn_only_preds = np.argmax(test_nn_probs, axis=1) + 1
    sub1 = pd.DataFrame(
        {"building_id": test_building_id, "damage_grade": nn_only_preds}
    )
    sub1.to_csv("data/submission_nn_only.csv", index=False)
    print("\nSaved NN-only submission to submission_nn_only.csv")

    # Strategy 2: NN + HGB weighted average (50-50)
    avg_probs = 0.5 * test_nn_probs + 0.5 * test_hgb_probs
    avg_preds = np.argmax(avg_probs, axis=1) + 1
    sub2 = pd.DataFrame({"building_id": test_building_id, "damage_grade": avg_preds})
    sub2.to_csv("data/submission_nn_hgb_avg.csv", index=False)
    print("Saved NN+HGB (50-50) submission to submission_nn_hgb_avg.csv")

    # Strategy 3: Stacking meta-model on OOF
    print("\nTraining stacking meta-model...")
    n = X.shape[0]
    oof_nn = np.zeros((n, 3))
    oof_hgb = np.zeros((n, 3))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        ckpt = f"best_model_fold_{fold}.pth"
        state = torch.load(ckpt, map_location=device)
        model = load_model_from_state(state, X.shape[1])
        model.to(device)
        model.eval()

        val_ds = TensorDataset(torch.from_numpy(X_val))
        val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
        nn_probs_list = []
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                nn_probs_list.append(probs)
        oof_nn[val_idx] = np.vstack(nn_probs_list)

        hgb = HistGradientBoostingClassifier(random_state=42, max_iter=500)
        hgb.fit(X_train, y_train)
        oof_hgb[val_idx] = hgb.predict_proba(X_val)

    # Meta-model: LR on concatenated OOF probs
    X_meta = np.hstack([oof_nn, oof_hgb])
    meta_clf = LogisticRegression(
        multi_class="multinomial", solver="lbfgs", max_iter=2000, C=0.1
    )
    meta_clf.fit(X_meta, y)

    test_meta = np.hstack([test_nn_probs, test_hgb_probs])
    meta_preds = meta_clf.predict(test_meta) + 1
    sub3 = pd.DataFrame({"building_id": test_building_id, "damage_grade": meta_preds})
    sub3.to_csv("data/submission_stacking.csv", index=False)
    print("Saved stacking submission to submission_stacking.csv")

    # Strategy 4: Threshold-based adjustment (boost minority class)
    print("\nApplying threshold-based adjustment for minority classes...")
    adj_probs = test_nn_probs.copy()
    # Boost class 1 (minority) predictions
    adj_probs[:, 1] *= 1.3
    # Normalize
    adj_probs /= adj_probs.sum(axis=1, keepdims=True)
    adj_preds = np.argmax(adj_probs, axis=1) + 1
    sub4 = pd.DataFrame({"building_id": test_building_id, "damage_grade": adj_preds})
    sub4.to_csv("data/submission_adjusted.csv", index=False)
    print("Saved threshold-adjusted submission to submission_adjusted.csv")

    # Final: Use the best strategy (we'll default to stacking)
    sub3.to_csv("data/submission.csv", index=False)
    print("\n" + "=" * 60)
    print("FINAL: Saved stacking-based submission to data/submission.csv")
    print("=" * 60)


if __name__ == "__main__":
    diagnose()

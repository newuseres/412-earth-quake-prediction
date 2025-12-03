import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


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


def softmax(a):
    e = np.exp(a - np.max(a, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    test_building_id = test_df["building_id"].copy()

    # categorical encoding consistent with training scripts
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

    n = X.shape[0]
    oof_nn = np.zeros((n, 3), dtype=np.float32)
    oof_hgb = np.zeros((n, 3), dtype=np.float32)
    test_nn_sum = np.zeros((X_test.shape[0], 3), dtype=np.float32)
    test_hgb_sum = np.zeros((X_test.shape[0], 3), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold} stacking...")

        # load NN checkpoint for this fold
        ckpt = f"best_model_fold_{fold}.pth"
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"Checkpoint {ckpt} not found; run third_edition to produce it."
            )
        state = torch.load(ckpt, map_location=device)
        model = load_model_from_state(state, X.shape[1])
        model.to(device)
        model.eval()

        X_val = X[val_idx]
        val_ds = TensorDataset(torch.from_numpy(X_val))
        val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
        probs_list = []
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs_list.append(probs)
        val_probs = np.vstack(probs_list)
        oof_nn[val_idx] = val_probs

        # test probs from this NN
        test_ds = TensorDataset(torch.from_numpy(X_test))
        test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
        tlist = []
        with torch.no_grad():
            for (xb,) in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                tlist.append(torch.softmax(logits, dim=1).cpu().numpy())
        test_fold_probs = np.vstack(tlist)
        test_nn_sum += test_fold_probs

        # train HGB on this fold's train split and get OOF probs
        hgb = HistGradientBoostingClassifier(random_state=42)
        hgb.fit(X[train_idx], y[train_idx])
        oof_hgb[val_idx] = hgb.predict_proba(X_val)
        test_hgb_sum += hgb.predict_proba(X_test)

    # average test probs across folds
    test_nn_avg = test_nn_sum / n_splits
    test_hgb_avg = test_hgb_sum / n_splits

    # meta features: concatenate oof_nn and oof_hgb
    X_meta = np.hstack([oof_nn, oof_hgb])
    meta_clf = LogisticRegression(
        multi_class="multinomial", solver="lbfgs", max_iter=2000
    )
    meta_clf.fit(X_meta, y)

    test_meta = np.hstack([test_nn_avg, test_hgb_avg])
    final_probs = meta_clf.predict_proba(test_meta)
    final_preds = np.argmax(final_probs, axis=1) + 1

    submission = pd.DataFrame(
        {"building_id": test_building_id, "damage_grade": final_preds}
    )
    submission.to_csv("data/submission.csv", index=False)
    print("Saved stacked ensemble submission to data/submission.csv")


if __name__ == "__main__":
    main()

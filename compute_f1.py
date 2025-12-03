import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report

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
    """兼容原始脚本中使用的 MyNet (使用 attribute 名称 `layers`)"""

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
    """兼容现有 best_model_fold_*.pth 中的结构（属性名为 layers，宽层次为 256-128-64-32-16-3）"""

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


def main():
    # load data
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    # encode categoricals same as training script
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

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_trues = []
    all_preds = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Processing fold {fold}...")
        X_val = X[val_idx]
        y_val = y[val_idx]

        model_path = f"best_model_fold_{fold}.pth"
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found, skipping fold.")
            continue

        # 尝试用当前 TabularNet 加载；若 key 不匹配则回退到旧的 MyNetOld
        state = torch.load(model_path, map_location=device)

        # 先使用与 checkpoint 对应的结构（属性名为 layers）加载
        def try_load(model, state):
            # 先尝试严格加载
            try:
                model.load_state_dict(state)
                return True
            except Exception:
                pass

            # 再尝试按形状匹配参数（非严格映射）
            model_state = model.state_dict()
            assigned = set()
            new_state = {}
            for mkey, mt in model_state.items():
                new_state[mkey] = mt

            for skey, sval in state.items():
                # 找一个形状匹配且尚未被赋值的目标键
                for mkey, mt in model_state.items():
                    if mkey in assigned:
                        continue
                    if mt.shape == sval.shape:
                        new_state[mkey] = sval
                        assigned.add(mkey)
                        break

            try:
                model.load_state_dict(new_state)
                return True
            except Exception:
                return False

        # 首先尝试与 checkpoint 对应的结构
        model = CheckpointNet(X.shape[1])
        ok = try_load(model, state)
        if not ok:
            model = TabularNet(X.shape[1])
            model.layers = model.net
            ok = try_load(model, state)
        if not ok:
            model = MyNetOld(X.shape[1])
            model.net = model.layers
            ok = try_load(model, state)

        if not ok:
            raise RuntimeError(
                f"无法把 checkpoint 映射到任何已知模型结构: {model_path}"
            )

        model.to(device)
        model.eval()

        preds = []
        with torch.no_grad():
            xb = torch.from_numpy(X_val).to(device)
            logits = model(xb)
            _, predicted = torch.max(logits, 1)
            preds = predicted.cpu().numpy()

        all_trues.extend(y_val.tolist())
        all_preds.extend(preds.tolist())

    if len(all_trues) == 0:
        print("No predictions were made. Ensure model files exist in the workspace.")
        return

    f1_macro = f1_score(all_trues, all_preds, average="macro")
    f1_weighted = f1_score(all_trues, all_preds, average="weighted")
    print(f"F1 macro: {f1_macro:.4f}")
    print(f"F1 weighted: {f1_weighted:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(all_trues, all_preds, digits=4))


if __name__ == "__main__":
    main()

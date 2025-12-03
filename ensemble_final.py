import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SuperNet(nn.Module):
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


def load_data():
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    test_ids = test_df["building_id"].values

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
    y = (train_df["damage_grade"].values - 1).astype(int)
    X_test = test_df.drop(["building_id"], axis=1).values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    return X, y, X_test, test_ids, scaler


def load_existing_models(X_test):
    """Load existing best_model_fold_*.pth files if available."""
    models = []
    for fold in range(5):
        model_path = Path(f"best_model_fold_{fold}.pth")
        if model_path.exists():
            model = SuperNet(X_test.shape[1]).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            models.append(model)
            print(f"Loaded best_model_fold_{fold}.pth")
        else:
            print(f"Warning: best_model_fold_{fold}.pth not found")
    return models


def get_nn_predictions(models, X_test):
    """Get averaged NN predictions from ensemble."""
    test_ds = TensorDataset(torch.from_numpy(X_test))
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    all_probs = []

    for model in models:
        probs = []
        model.eval()
        with torch.no_grad():
            for (xb,) in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs.append(F.softmax(logits, dim=1).cpu().numpy())
        all_probs.append(np.vstack(probs))

    if all_probs:
        avg_probs = np.mean(all_probs, axis=0)
        print(f"Using {len(all_probs)} NN models for ensemble")
        return avg_probs
    else:
        print("No models found! Using random predictions.")
        return np.ones((X_test.shape[0], 3)) / 3


def train_hgb_ensemble(X_train, y_train, X_test, num_iterations=3):
    """Train multiple HGB models for ensemble."""
    all_hgb_probs = []

    for i in range(num_iterations):
        hgb = HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.05 + i * 0.02,
            l2_regularization=0.1,
            random_state=42 + i,
            max_leaf_nodes=31,
        )
        hgb.fit(X_train, y_train)
        hgb_probs = hgb.predict_proba(X_test)
        all_hgb_probs.append(hgb_probs)
        print(f"HGB model {i+1} trained")

    return np.mean(all_hgb_probs, axis=0)


def main():
    print("Loading data...")
    X_train, y_train, X_test, test_ids, scaler = load_data()

    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"Y class distribution: {np.bincount(y_train)}")

    print("\nLoading existing NN models...")
    models = load_existing_models(X_test)

    if not models:
        print("ERROR: No pre-trained models found! Please train models first.")
        return

    print("\nGenerating NN predictions...")
    nn_probs = get_nn_predictions(models, X_test)

    print("Training HGB ensemble...")
    hgb_probs = train_hgb_ensemble(X_train, y_train, X_test, num_iterations=3)

    # Weighted ensemble: favor NN (which uses Focal Loss for F1 optimization)
    print("\nCreating final ensemble...")
    final_probs = 0.7 * nn_probs + 0.3 * hgb_probs
    final_preds = np.argmax(final_probs, axis=1) + 1

    # Save submission
    submission = pd.DataFrame({"building_id": test_ids, "damage_grade": final_preds})
    submission.to_csv("data/submission.csv", index=False)

    print(f"\nFinal prediction distribution:")
    for cls in [1, 2, 3]:
        count = (final_preds == cls).sum()
        pct = 100 * count / len(final_preds)
        print(f"  Class {cls}: {count:4d} ({pct:5.1f}%)")

    print(f"\nSubmission saved to data/submission.csv")
    print(f"First 5 rows:")
    print(submission.head())


if __name__ == "__main__":
    main()

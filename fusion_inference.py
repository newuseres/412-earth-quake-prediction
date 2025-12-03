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


class MyNet(nn.Module):
    """Original architecture from net_second_edition.py"""

    def __init__(self, input_dim):
        super(MyNet, self).__init__()
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


def load_data():
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    test_ids = test_df["building_id"].values

    # Categorical encoding (matching original approach)
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

    return X, y, X_test, test_ids


def load_existing_models(X_test):
    """Load existing best_model_fold_*.pth files."""
    models = []
    for fold in range(5):
        model_path = Path(f"best_model_fold_{fold}.pth")
        if model_path.exists():
            model = MyNet(X_test.shape[1]).to(device)
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                models.append(model)
                print(f"✓ Loaded best_model_fold_{fold}.pth")
            except RuntimeError as e:
                print(f"✗ Could not load best_model_fold_{fold}.pth: {str(e)[:100]}")
        else:
            print(f"✗ best_model_fold_{fold}.pth not found")
    return models


def get_nn_predictions(models, X_test):
    """Get averaged NN predictions from ensemble."""
    test_ds = TensorDataset(torch.from_numpy(X_test))
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    all_probs = []

    for i, model in enumerate(models):
        probs = []
        model.eval()
        with torch.no_grad():
            for (xb,) in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs.append(F.softmax(logits, dim=1).cpu().numpy())
        all_probs.append(np.vstack(probs))
        print(f"  Model {i+1}: {np.vstack(probs).shape}")

    if all_probs:
        avg_probs = np.mean(all_probs, axis=0)
        print(f"Using {len(all_probs)} NN models for ensemble")
        return avg_probs
    else:
        print("ERROR: No models found!")
        return None


def train_hgb_ensemble(X_train, y_train, X_test):
    """Train HGB model for ensemble."""
    hgb = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.1,
        l2_regularization=0.1,
        random_state=42,
        max_leaf_nodes=31,
    )
    hgb.fit(X_train, y_train)
    hgb_probs = hgb.predict_proba(X_test)
    print(f"HGB predictions: {hgb_probs.shape}")
    return hgb_probs


def main():
    print("=" * 60)
    print("ENSEMBLE INFERENCE PIPELINE")
    print("=" * 60)

    print("\n1. Loading data...")
    X_train, y_train, X_test, test_ids = load_data()
    print(f"   X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"   Y distribution: {np.bincount(y_train)}")

    print("\n2. Loading NN models...")
    models = load_existing_models(X_test)

    if not models:
        print("\nERROR: No models found. Cannot proceed.")
        return

    print(f"\n3. Generating NN predictions ({len(models)} models)...")
    nn_probs = get_nn_predictions(models, X_test)

    if nn_probs is None:
        print("ERROR: Failed to generate NN predictions")
        return

    print(f"\n4. Training HGB model...")
    hgb_probs = train_hgb_ensemble(X_train, y_train, X_test)

    print(f"\n5. Creating weighted ensemble...")
    print(f"   NN weight: 0.65  |  HGB weight: 0.35")
    final_probs = 0.65 * nn_probs + 0.35 * hgb_probs
    final_preds = np.argmax(final_probs, axis=1) + 1

    # Save submission
    submission = pd.DataFrame({"building_id": test_ids, "damage_grade": final_preds})
    submission.to_csv("data/submission.csv", index=False)

    print(f"\n6. Results:")
    print(f"   {'Class':<10} {'Count':<10} {'Percentage':<10}")
    print(f"   {'-'*30}")
    for cls in [1, 2, 3]:
        count = (final_preds == cls).sum()
        pct = 100 * count / len(final_preds)
        print(f"   {cls:<10} {count:<10} {pct:<10.1f}%")

    print(f"\n✓ Submission saved to data/submission.csv")
    print(f"\nFirst 5 rows:")
    print(submission.head().to_string(index=False))
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

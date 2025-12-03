from matplotlib import scale
from sklearn.discriminant_analysis import StandardScaler
from sklearn.utils import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, recall_score

import pandas as pd
import numpy as np

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
test_building_id = test_df["building_id"].copy()


# deal with categorical features
cat_cols = train_df.select_dtypes(include=["object"]).columns.tolist()
for col in cat_cols:
    train_df[col] = train_df[col].astype("category").cat.codes
    test_df[col] = test_df[col].astype("category").cat.codes

# prepare for features and labels
X = train_df.drop(["building_id", "damage_grade"], axis=1).values
Y = train_df["damage_grade"].values.astype(int) - 1

X_test = test_df.drop(["building_id"], axis=1).values

# Normalize for data -- (ljh key change point)
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
# X_test = scaler.transform(X_test)

# compute for weights for classes -- to deal with the problem of class's unbalance
# class_weights = compute_class_weight("balanced", classes=np.unique(Y), y=Y)
# class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)


# define our model
# class MyNet(nn.Module):
#     def __init__(self, input_dim):
#         super(MyNet, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Dropout(0.2),  # continue to lower dropout bility
#             nn.Linear(32, 16),
#             nn.ReLU(),
#             nn.Linear(
#                 16, 3
#             ),  # output layer will be 3 , because damage_grade is a 3-class classification task
#         )

#     def forward(self, x):
#         return self.layers(x)


# """ My net before
class MyNet(nn.Module):
    def __init__(self, input_dim):
        super(MyNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.4),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(
                16, 3
            ),  # output layer will be 3 , because damage_grade is a 3-class classification task
        )

    def forward(self, x):
        return self.layers(x)


# """


# cross validation
fold_num = 5
epoch_num = 100
kfold = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=42)
accuracies = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X, Y)):
    X_train, X_val = X[train_idx], X[val_idx]
    Y_train, Y_val = Y[train_idx], Y[val_idx]

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.long)
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # initialize model, criterion and optimizer
    model = MyNet(X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    #    criterion = nn.CrossEntropyLoss(weight=class_weights)  # Based on weighted class
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_model = None
    best_val_loss = float("inf")

    # parameters for early stopping
    patience = 20  # when validation loss doesn't improve for 10 epochs, stop training
    no_improve_coutners = 0  # counter
    delta = 0.001  # minimum change in validation loss to qualify as an improvement
    early_stop = False  # sign for early stopping

    # training loop
    for epoch in range(epoch_num):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item()

                # get predicted features
                _, predicted = torch.max(pred, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(yb.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        val_recall = recall_score(val_targets, val_preds, average="macro")
        val_f1 = f1_score(val_targets, val_preds, average="macro")

        if val_loss < best_val_loss - delta:  # validation loss have enough improvement
            best_val_loss = val_loss
            best_model = model.state_dict()
            # save best model
            torch.save(best_model, f"best_model_fold_{fold}.pth")
            no_improve_coutners = 0  # reset counter
        else:
            no_improve_coutners += 1  # increment counter

            if no_improve_coutners >= patience:  # no improvement for 'patience' epochs
                early_stop = True
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        print(
            f"Fold {fold+1}, Epoch {epoch+1}, Val Loss: {val_loss:.4f}, "
            f"Val Accuracy: {val_acc:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}"
        )

    accuracies.append(val_acc)

mean_acc = np.mean(accuracies)
print(f"Mean Accuracy: {mean_acc:.4f}")


# Use best model
best_model = MyNet(X.shape[1]).to(device)
best_model.load_state_dict(torch.load(f"best_model_fold_{fold_num-1}.pth"))
best_model.eval()

# predict on train set
X_full_tensor = torch.tensor(X, dtype=torch.float32).to(device)
with torch.no_grad():
    y_pred_logits = best_model(X_full_tensor)
    _, y_pred_full = torch.max(y_pred_logits, 1)

y_pred_full = y_pred_full.cpu().numpy()

# give a measurement on our NN
print("Accuracy on full training set:", accuracy_score(Y, y_pred_full))
print("\nClassification Report:\n", classification_report(Y, y_pred_full))
recall = recall_score(Y, y_pred_full, average="macro")
f1 = f1_score(Y, y_pred_full, average="macro")
print(f"\nMacro Recall on full training set: {recall:.4f}")
print(f"Macro F1-score on full training set: {f1:.4f}")

# predict on test set
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
with torch.no_grad():
    y_pred_logits = best_model(X_test_tensor)
    _, y_pred_test = torch.max(y_pred_logits, 1)

y_pred_test = y_pred_test.cpu().numpy()


# Save submission
y_pred_test = y_pred_test + 1
submission = pd.DataFrame(
    {"building_id": test_building_id, "damage_grade": y_pred_test}
)
submission.to_csv("data/submission.csv", index=False)
print("\nYes Submission saved to 'data/submission.csv'")

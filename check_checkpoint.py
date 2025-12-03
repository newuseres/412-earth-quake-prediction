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


# Load checkpoint to see actual architecture
checkpoint = torch.load("best_model_fold_0.pth", map_location="cpu")
print("Checkpoint keys:")
for key in sorted(checkpoint.keys())[:10]:
    print(f"  {key}")

# Create model that matches the checkpoint
if "bn_input.weight" in checkpoint:
    print("\n=> Checkpoint is from optimized (wide) architecture")

    class WideNet(nn.Module):
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

    model = WideNet(35)
    model.load_state_dict(checkpoint)
    model.to(device)
    print("Loaded WideNet successfully!")
else:
    print("\n=> Checkpoint is from narrow architecture")


print("\nTest inference...")
X_test = np.random.randn(10, 35).astype(np.float32)
X_test_tensor = torch.from_numpy(X_test).to(device)
with torch.no_grad():
    output = model(X_test_tensor)
    print(f"Output shape: {output.shape}")
    probs = F.softmax(output, dim=1)
    print(f"Probs shape: {probs.shape}")
    print(f"First sample probs: {probs[0]}")

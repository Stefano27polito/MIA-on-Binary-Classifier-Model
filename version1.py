# ============================================================
# VERSION 1 - BASELINE MLP ON ADULT DATASET
# ============================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ------------------------------------------------------------
# 1. LOAD AND PREPROCESS DATASET
# ------------------------------------------------------------

print("Loading dataset...")

df = pd.read_csv("adult.csv")

# Use a reduced dataset for a lightweight baseline
df = df.sample(n=5000, random_state=42)

# Handle missing values
df = df.replace(" ?", np.nan).dropna()

# Encode target label
label_encoder = LabelEncoder()
df["income"] = label_encoder.fit_transform(df["income"])

y = df["income"].values
X = df.drop(columns=["income"])

# Encode categorical features
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

X = X.astype(np.float32)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Dataset loaded.")
print("Train samples:", len(X_train))
print("Test samples:", len(X_test))


# ------------------------------------------------------------
# 2. DATASET DEFINITION
# ------------------------------------------------------------

class AdultDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = AdultDataset(X_train, y_train)
test_dataset  = AdultDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=16)


# ------------------------------------------------------------
# 3. MODEL DEFINITION (SMALL BASELINE)
# ------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.net(x)


model = MLP(input_size=X_train.shape[1])

print(model)


# ------------------------------------------------------------
# 4. TRAINING SETUP
# ------------------------------------------------------------

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20


# ------------------------------------------------------------
# 5. TRAINING LOOP
# ------------------------------------------------------------

print("\nStarting training...\n")

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch:02d} | Loss: {total_loss:.4f}")


# ------------------------------------------------------------
# 6. EVALUATION
# ------------------------------------------------------------

print("\nEvaluating model...")

model.eval()
y_pred = []

with torch.no_grad():
    for xb, _ in test_loader:
        preds = torch.argmax(model(xb), dim=1)
        y_pred.extend(preds.numpy())

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred))


# ------------------------------------------------------------
# 7. SAVE MODEL
# ------------------------------------------------------------

torch.save(model.state_dict(), "adult_mlp_v1.pth")
print("\nModel saved as adult_mlp_v1.pth")

# ============================================================
# TRAIN A NEURAL NETWORK ON ADULT DATASET (PYTORCH)
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

# Remove missing values
df = df.replace(" ?", np.nan).dropna()

# Encode label (income)
label_encoder = LabelEncoder()
df["income"] = label_encoder.fit_transform(df["income"])

y = df["income"].values
X = df.drop(columns=["income"])

# Encode categorical features
X = pd.get_dummies(X)

# Normalize numerical values
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
# 2. PYTORCH DATASET
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

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=128)


# ------------------------------------------------------------
# 3. NEURAL NETWORK MODEL (MLP)
# ------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)


model = MLP(
    input_size=X_train.shape[1],
    hidden_size=50,
    output_size=len(np.unique(y_train))
)

print("Model created.")
print(model)


# ------------------------------------------------------------
# 4. TRAINING SETUP
# ------------------------------------------------------------

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50


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

    if epoch % 5 == 0:
        print(f"Epoch {epoch:03d} | Loss: {total_loss:.4f}")


# ------------------------------------------------------------
# 6. EVALUATION (LABELING)
# ------------------------------------------------------------

print("\nEvaluating model...")

model.eval()
y_pred = []

with torch.no_grad():
    for xb, _ in test_loader:
        outputs = model(xb)
        preds = torch.argmax(outputs, dim=1)
        y_pred.extend(preds.numpy())

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred))


# ------------------------------------------------------------
# 7. SAVE MODEL
# ------------------------------------------------------------

torch.save(model.state_dict(), "adult_mlp_model.pth")
print("\nModel saved as adult_mlp_model.pth")

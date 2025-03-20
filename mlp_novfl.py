import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Combined MLP model
class MLPModel(nn.Module):
    # def __init__(self, input_size, hidden_layers):
    #     super(MLPModel, self).__init__()
    #     layers = []
    #     prev_size = input_size
    #     for hidden_size in hidden_layers:
    #         layers.extend([nn.Linear(prev_size, hidden_size), nn.ReLU()])
    #         prev_size = hidden_size
    #     layers.append(nn.Linear(prev_size, 1))
    #     self.mlp_layers = nn.Sequential(*layers)

    # def forward(self, x):
    #     return self.mlp_layers(x)
    def __init__(self, input_dim):
        super(MLPModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 1)  # Single output for regression
        )

    def forward(self, x):
        return self.layers(x)

# Main function to run the model
def run_regression():
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    X, y = data.data, data.target
    input_dim = X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).view(-1, 1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).view(-1, 1)

    model = MLPModel(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    training_losses = []

    for epoch in range(num_epochs):
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        training_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    with torch.no_grad():
        y_pred_test = model(X_test)
        test_loss = criterion(y_pred_test, y_test)
        print(f'Test Loss: {test_loss.item():.4f}')
        rmse = torch.sqrt(test_loss)
        print(f'Test RMSE: {rmse.item():.4f}')

if __name__ == "__main__":
    run_regression()
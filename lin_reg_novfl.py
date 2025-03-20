import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the Linear Regression model
class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Single output for regression
        
    def forward(self, x):
        return self.linear(x)

# Fetch the data
# def fetch_data_superconductivity():
#     df = pd.read_csv(r'C:\Users\dc\Downloads\VFL\Datasets\super conductivity\train.csv')
#     df = df.applymap(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
#     df = df.astype(float)
#     data = df[df.columns[:-1]]
#     target = df[df.columns[-1]]

#     data = data.to_numpy()
#     target = target.to_numpy()
#     return data, target, data.shape[1]

# Main function to run the model
def run_regression():
    # Get the data
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    X, y = data.data, data.target
    input_dim = X.shape[1]
    # X, y, input_dim = fetch_data_superconductivity()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).view(-1, 1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).view(-1, 1)
    
    # Initialize the model
    model = LinearRegression(input_dim)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Train the model
    num_epochs = 100
    training_losses = []
    
    for epoch in range(num_epochs):
        # Forward pass
        y_pred = model(X_train)
        
        # Compute loss
        loss = criterion(y_pred, y_train)
        training_losses.append(loss.item())
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # Evaluate the model
    with torch.no_grad():
        y_pred_test = model(X_test)
        test_loss = criterion(y_pred_test, y_test)
        print(f'Test Loss: {test_loss.item():.4f}')
        rmse = torch.sqrt(test_loss)  
        print(f'Test RMSE: {rmse.item():.4f}')
    
    # Plot the loss over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_epochs), training_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.show()
    
    # Get the model parameters
    w = model.linear.weight.data.numpy()
    b = model.linear.bias.data.numpy()
    print(f'Weight shape: {w.shape}')
    print(f'Bias: {b[0]:.4f}')
    
    return model, test_loss.item()

if __name__ == "__main__":
    model, test_loss = run_regression()

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.dataset import generate_dataset
from src.dynamics import InvertedPendulum   # ← ADD THIS IMPORT

os.makedirs("models", exist_ok=True)


class DynamicsModel(nn.Module):
    """
    Neural network model for learning residual dynamics.

    Learns:
        residual = true_dynamics − physics_dynamics
    """

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.network(x)


def train():

    # Generate dataset
    X, U, Y = generate_dataset()

    inputs = np.hstack((X, U))

    pendulum = InvertedPendulum()
    dt = pendulum.dt

    targets = []

    for i in range(len(X)):

        state = X[i]
        u = U[i][0]

        # True derivative from data
        true_derivative = (Y[i] - X[i]) / dt

        # Physics model derivative
        physics_derivative = pendulum.physics_derivative(state, u)

        # Residual dynamics
        residual = true_derivative - physics_derivative

        targets.append(residual)

    targets = np.array(targets)

    # Normalize inputs
    input_mean = inputs.mean(axis=0)
    input_std = inputs.std(axis=0) + 1e-8
    inputs = (inputs - input_mean) / input_std

    # Normalize targets
    target_mean = targets.mean(axis=0)
    target_std = targets.std(axis=0) + 1e-8
    targets = (targets - target_mean) / target_std

    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)

    model = DynamicsModel()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 150

    for epoch in range(epochs):

        predictions = model(inputs)
        loss = criterion(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

    # Save trained model
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_mean": input_mean,
        "input_std": input_std,
        "target_mean": target_mean,
        "target_std": target_std
    }

    torch.save(checkpoint, "models/dynamics_model.pth")

    print("Model saved to models/dynamics_model.pth")


if __name__ == "__main__":
    train()
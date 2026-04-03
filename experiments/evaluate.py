import os
import numpy as np
import pandas as pd
import torch

from src.dynamics import InvertedPendulum
from src.dataset import generate_dataset
from src.train_model import DynamicsModel
from experiments.linear_baseline import identify_linear_model


RESULTS_DIR = "results"
MODEL_PATH = "models/dynamics_model.pth"


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rollout_residual(model, pendulum, x0, controls, input_mean, input_std, target_mean, target_std):

    x = x0.copy()
    traj = [x]

    for u in controls:

        nn_input = np.array([x[0], x[1], u])
        nn_input = (nn_input - input_mean) / input_std

        nn_input = torch.tensor(nn_input, dtype=torch.float32)

        with torch.no_grad():
            residual_norm = model(nn_input).numpy()

        residual = residual_norm * target_std + target_mean

        physics = pendulum.physics_derivative(x, u)

        xdot = physics + residual

        x = x + pendulum.dt * xdot

        traj.append(x)

    return np.array(traj)


def rollout_linear(A, B, x0, controls, dt):

    x = x0.copy()
    traj = [x]

    for u in controls:

        dx = A @ x + B.flatten() * u
        x = x + dt * dx

        traj.append(x)

    return np.array(traj)


def run_trial():

    pendulum = InvertedPendulum()

    # generate dataset
    X, U, Y = generate_dataset()

    initial_state = X[0]

    controls = U[:200].flatten()

    # load neural model
    model = DynamicsModel()
    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    input_mean = checkpoint["input_mean"]
    input_std = checkpoint["input_std"]

    target_mean = checkpoint["target_mean"]
    target_std = checkpoint["target_std"]

    model.eval()

    # linear baseline
    A, B = identify_linear_model()

    # true rollout
    true_states = [initial_state]
    x = initial_state.copy()

    for u in controls:

        x = pendulum.step(x, u)
        true_states.append(x)

    true_states = np.array(true_states)

    # model rollouts
    residual_traj = rollout_residual(
    model,
    pendulum,
    initial_state,
    controls,
    input_mean,
    input_std,
    target_mean,
    target_std
                                    )
    linear_traj = rollout_linear(A, B, initial_state, controls, pendulum.dt)

    # compute errors over the full state [theta, theta_dot]

    rmse_linear = rmse(true_states, linear_traj)
    mae_linear = mae(true_states, linear_traj)

    rmse_residual = rmse(true_states, residual_traj)
    mae_residual = mae(true_states, residual_traj)

    return rmse_linear, mae_linear, rmse_residual, mae_residual


def evaluate(num_trials=10):

    results = []

    for _ in range(num_trials):

        rmse_l, mae_l, rmse_r, mae_r = run_trial()

        results.append({
            "rmse_linear": rmse_l,
            "mae_linear": mae_l,
            "rmse_residual": rmse_r,
            "mae_residual": mae_r
        })

    df = pd.DataFrame(results)

    summary = pd.DataFrame({
        "model": ["Linear Model", "Residual Neural Network"],
        "rmse_mean": [df.rmse_linear.mean(), df.rmse_residual.mean()],
        "rmse_std": [df.rmse_linear.std(), df.rmse_residual.std()],
        "mae_mean": [df.mae_linear.mean(), df.mae_residual.mean()],
        "mae_std": [df.mae_linear.std(), df.mae_residual.std()]
    })

    os.makedirs(RESULTS_DIR, exist_ok=True)

    summary.to_csv(f"{RESULTS_DIR}/error_table.csv", index=False)

    print("\nEvaluation Results\n")
    print(summary)


if __name__ == "__main__":

    evaluate(num_trials=10)
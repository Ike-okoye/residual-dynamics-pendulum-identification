import torch
import numpy as np
import matplotlib.pyplot as plt

from src.dynamics import InvertedPendulum
from src.train_model import DynamicsModel
from experiments.linear_baseline import identify_linear_model


def wrap_angle(theta):
    return (theta + np.pi) % (2*np.pi) - np.pi


def rollout_error():

    checkpoint = torch.load("models/dynamics_model.pth", weights_only=False)

    model = DynamicsModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    input_mean = checkpoint["input_mean"]
    input_std = checkpoint["input_std"]
    target_mean = checkpoint["target_mean"]
    target_std = checkpoint["target_std"]

    pendulum = InvertedPendulum()

    A, B = identify_linear_model()

    horizon = 200
    trials = 20

    errors_residual = np.zeros((trials, horizon))
    errors_linear = np.zeros((trials, horizon))

    for t in range(trials):

        true_state = np.array([0.2, 0.0])
        nn_state = true_state.copy()
        lin_state = true_state.copy()

        for k in range(horizon):

            u = np.random.uniform(-5, 5)

            # ---------- TRUE DYNAMICS ----------
            true_next = pendulum.step(true_state, u)
            true_next[0] = wrap_angle(true_next[0])

            # ---------- RESIDUAL MODEL ----------
            nn_input = np.array([nn_state[0], nn_state[1], u])
            nn_input = (nn_input - input_mean) / input_std
            nn_input = torch.tensor(nn_input, dtype=torch.float32)

            with torch.no_grad():
                residual_norm = model(nn_input).numpy()

            residual = residual_norm * target_std + target_mean

            physics = pendulum.physics_derivative(nn_state, u)

            state_dot = physics + residual

            nn_next = nn_state + pendulum.dt * state_dot
            nn_next[0] = wrap_angle(nn_next[0])

            # ---------- LINEAR MODEL ----------
            dx = A @ lin_state + B.flatten() * u
            lin_next = lin_state + pendulum.dt * dx
            lin_next[0] = wrap_angle(lin_next[0])

            # ---------- ERRORS ----------
            errors_residual[t, k] = np.linalg.norm(true_next - nn_next)
            errors_linear[t, k] = np.linalg.norm(true_next - lin_next)

            true_state = true_next
            nn_state = nn_next
            lin_state = lin_next

    rmse_residual = np.sqrt(np.mean(errors_residual**2, axis=0))
    rmse_linear = np.sqrt(np.mean(errors_linear**2, axis=0))

    plt.figure(figsize=(5.5, 3.2))

    plt.plot(rmse_residual, label="Residual physics model")
    plt.plot(rmse_linear, "--", label="Linear model")

    plt.xlabel("Prediction Horizon")
    plt.ylabel("RMSE")
    plt.title("Multi-Step Rollout Prediction (200-step horizon)")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("results/rmse_eval.pdf")
    plt.show()


if __name__ == "__main__":
    rollout_error()
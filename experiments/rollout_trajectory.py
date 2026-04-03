import torch
import numpy as np
import matplotlib.pyplot as plt

from src.dynamics import InvertedPendulum
from src.train_model import DynamicsModel
from experiments.linear_baseline import identify_linear_model


def wrap_angle(theta):
    return (theta + np.pi) % (2*np.pi) - np.pi


def rollout():

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

    true_state = np.array([0.2, 0.0])
    nn_state = true_state.copy()
    lin_state = true_state.copy()

    true_traj = []
    nn_traj = []
    lin_traj = []

    for _ in range(horizon):

        u = np.random.uniform(-5,5)

        # true system
        true_next = pendulum.step(true_state,u)
        true_next[0] = wrap_angle(true_next[0])

        # residual model
        nn_input = np.array([nn_state[0],nn_state[1],u])
        nn_input = (nn_input-input_mean)/input_std
        nn_input = torch.tensor(nn_input,dtype=torch.float32)

        with torch.no_grad():
            residual_norm = model(nn_input).numpy()

        residual = residual_norm*target_std + target_mean

        physics = pendulum.physics_derivative(nn_state,u)

        nn_next = nn_state + pendulum.dt*(physics+residual)
        nn_next[0] = wrap_angle(nn_next[0])

        # linear model
        dx = A @ lin_state + B.flatten()*u
        lin_next = lin_state + pendulum.dt*dx
        lin_next[0] = wrap_angle(lin_next[0])

        true_traj.append(true_next[0])
        nn_traj.append(nn_next[0])
        lin_traj.append(lin_next[0])

        true_state = true_next
        nn_state = nn_next
        lin_state = lin_next

    true_traj = np.array(true_traj)
    nn_traj = np.array(nn_traj)
    lin_traj = np.array(lin_traj)

    plt.figure(figsize=(6,3.5))

    plt.plot(true_traj,label="True dynamics")
    plt.plot(nn_traj,"--",label="Residual physics model")
    plt.plot(lin_traj,":",label="Linear model")

    plt.xlabel("Time step")
    plt.ylabel(r"$\theta$ (rad)")
    plt.title("Multi-Step Rollout Trajectory (200-step horizon)")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("results/rollout_trajectory.pdf")
    plt.show()


if __name__ == "__main__":
    rollout()
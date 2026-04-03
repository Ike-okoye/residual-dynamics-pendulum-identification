import torch
import numpy as np
import matplotlib.pyplot as plt

from src.dynamics import InvertedPendulum
from src.train_model import DynamicsModel
from experiments.linear_baseline import identify_linear_model


plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "lines.linewidth": 2,
})


def wrap_angle(theta):
    return (theta + np.pi) % (2*np.pi) - np.pi


def comparison():

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

    true_state = np.array([0.2, 0.0])
    nn_state = true_state.copy()
    lin_state = true_state.copy()

    true_states = []
    nn_states = []
    lin_states = []

    for _ in range(200):

        u = 0.0

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

        true_states.append(true_next)
        nn_states.append(nn_next)
        lin_states.append(lin_next)

        true_state = true_next
        nn_state = nn_next
        lin_state = lin_next

    true_states = np.array(true_states)
    nn_states = np.array(nn_states)
    lin_states = np.array(lin_states)

    plt.figure(figsize=(5.5, 3.2))

    # Phase portrait: theta vs theta_dot
    plt.plot(true_states[:,0], true_states[:,1], label="True dynamics")
    plt.plot(nn_states[:,0], nn_states[:,1], "--", label="Residual physics model")
    plt.plot(lin_states[:,0], lin_states[:,1], ":", linewidth=3, label="Linear model")

    plt.xlabel(r"$\theta$ (rad)")
    plt.ylabel(r"$\dot{\theta}$ (rad/s)")
    plt.title("Phase Portrait of the Inverted Pendulum")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("results/phase_portrait.pdf")
    plt.show()


if __name__ == "__main__":
    comparison()
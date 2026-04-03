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

        u = np.random.uniform(-5,5)

        true_next = pendulum.step(true_state, u)
        true_next[0] = (true_next[0] + np.pi) % (2*np.pi) - np.pi

        # residual model
        nn_input = np.array([nn_state[0],nn_state[1],u])
        nn_input = (nn_input-input_mean)/input_std
        nn_input = torch.tensor(nn_input,dtype=torch.float32)

        with torch.no_grad():
            residual_norm = model(nn_input).numpy()

        residual = residual_norm*target_std + target_mean

        physics = pendulum.physics_derivative(nn_state,u)

        state_dot = physics + residual

        nn_next = nn_state + pendulum.dt * state_dot
        nn_next[0] = (nn_next[0] + np.pi) % (2*np.pi) - np.pi

        # linear model
        dx = A @ lin_state + B.flatten()*u
        lin_next = lin_state + pendulum.dt * dx

        true_states.append(true_next)
        nn_states.append(nn_next)
        lin_states.append(lin_next)

        true_state = true_next
        nn_state = nn_next
        lin_state = lin_next

    true_states = np.array(true_states)
    nn_states = np.array(nn_states)
    lin_states = np.array(lin_states)

    plt.figure(figsize=(5.5,3.2))

    plt.plot(true_states[:,0], label="True dynamics", linewidth=2)
    plt.plot(nn_states[:,0],"--",label="Residual Physics Model")
    plt.plot(lin_states[:,0],":",label="Linear model")

    plt.xlabel("Time step")
    plt.ylabel(r"$\theta$ (rad)")
    plt.title("Trajectory Comparison of Identified Models")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig("results/model_comparison.pdf")
    plt.show()

if __name__ == "__main__":
    comparison()
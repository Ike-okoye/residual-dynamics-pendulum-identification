import numpy as np
import matplotlib.pyplot as plt

from src.dataset import generate_dataset
from src.dynamics import InvertedPendulum


def identify_linear_model():

    X, U, Y = generate_dataset()

    # learn state increment
    dX = Y - X

    Phi = np.hstack((X, U))

    W = np.linalg.lstsq(Phi, dX, rcond=None)[0]

    A = W[:2, :].T
    B = W[2:, :].T

    return A, B


def rollout_linear(A, B):

    np.random.seed(0)

    pendulum = InvertedPendulum()

    true_state = np.array([0.2, 0.0])
    lin_state = true_state.copy()

    true_states = []
    lin_states = []

    for _ in range(200):

        u = np.random.uniform(-5, 5)

        # true system
        true_next = pendulum.step(true_state, u)

        # linear prediction
        dx = A @ lin_state + B[:,0] * u
        lin_next = lin_state + dx

        true_states.append(true_next)
        lin_states.append(lin_next)

        true_state = true_next
        lin_state = lin_next

    true_states = np.array(true_states)
    lin_states = np.array(lin_states)

    plt.figure(figsize=(8,5))

    plt.plot(true_states[:,0], label="True angle", linewidth=3)
    plt.plot(lin_states[:,0], "--", label="Linear model")

    plt.xlabel("Time Step")
    plt.ylabel("Angle (rad)")
    plt.title("Linear Model Rollout")

    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":

    A, B = identify_linear_model()

    rollout_linear(A, B)
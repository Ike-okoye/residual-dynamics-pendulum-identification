import numpy as np
from src.dynamics import InvertedPendulum


def generate_dataset(num_samples=100000):

    pendulum = InvertedPendulum()

    states = []
    inputs = []
    next_states = []

    for _ in range(num_samples):

        # random state
        state = np.random.uniform([-np.pi, -5], [np.pi, 5])

        # random control input
        u = np.random.uniform(-5, 5)

        next_state = pendulum.step(state, u)

        states.append(state)
        inputs.append(u)
        next_states.append(next_state)

    states = np.array(states)
    inputs = np.array(inputs).reshape(-1, 1)
    next_states = np.array(next_states)

    return states, inputs, next_states
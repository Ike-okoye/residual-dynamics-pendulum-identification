import torch
import numpy as np
import matplotlib.pyplot as plt

from src.dynamics import InvertedPendulum
from src.train_model import DynamicsModel

def rollout_error():

    checkpoint=torch.load("models/dynamics_model.pth",weights_only=False)

    model=DynamicsModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    input_mean=checkpoint["input_mean"]
    input_std=checkpoint["input_std"]
    target_mean=checkpoint["target_mean"]
    target_std=checkpoint["target_std"]

    pendulum=InvertedPendulum()

    horizon=200
    trials=20

    errors=np.zeros((trials,horizon))

    for t in range(trials):

        true_state=np.array([0.2,0.0])
        nn_state=true_state.copy()

        for k in range(horizon):

            u=np.random.uniform(-5,5)

            true_next=pendulum.step(true_state,u)

            nn_input=np.array([nn_state[0],nn_state[1],u])
            nn_input=(nn_input-input_mean)/input_std
            nn_input=torch.tensor(nn_input,dtype=torch.float32)

            with torch.no_grad():
                residual_norm=model(nn_input).numpy()

            residual=residual_norm*target_std + target_mean

            physics=pendulum.physics_derivative(nn_state,u)

            state_dot=physics+residual

            nn_next=nn_state + pendulum.dt * state_dot

            error=np.linalg.norm(true_next-nn_next)

            errors[t,k]=error

            true_state=true_next
            nn_state=nn_next

    rmse=np.sqrt(np.mean(errors**2,axis=0))

    plt.figure(figsize=(5.5,3.2))
    plt.plot(rmse)

    plt.xlabel("Prediction Horizon")
    plt.ylabel("RMSE")
    plt.title("Rollout Error vs Prediction Horizon")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig("results/rmse_curve.pdf")
    plt.show()

if __name__=="__main__":
    rollout_error()
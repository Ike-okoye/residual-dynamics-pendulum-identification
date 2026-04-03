import numpy as np


class InvertedPendulum:
    """
    Nonlinear inverted pendulum dynamics.

    State vector:
        x = [theta, theta_dot]

    Control input:
        u = torque applied at the pivot
    """
    
    def __init__(self, mass=0.2, length=0.5, gravity=9.81, dt=0.02):

        self.m = mass
        self.l = length
        self.g = gravity
        self.dt = dt


    def dynamics(self, state, u):
        """
        Compute state derivatives.

        Returns
        -------
        [theta_dot, theta_ddot]
        """

        theta, theta_dot = state
        theta_ddot = (self.g/self.l) * np.sin(theta) \
        + (1/(self.m*self.l**2))*u \
            - 0.1*theta_dot

        return np.array([theta_dot, theta_ddot])


    def physics_derivative(self, state, u):
        """
        Physics-based model used for residual learning.

        Returns the derivative predicted purely from physics.
        """

        theta, theta_dot = state

        theta_ddot = (self.g / self.l) * np.sin(theta) + (1 / (self.m * self.l**2)) * u

        return np.array([theta_dot, theta_ddot])


    def step(self, state, u):
        """
        Perform one simulation step using Euler integration.
        """

        state_dot = self.dynamics(state, u)
        next_state = state + self.dt * state_dot

        return next_state
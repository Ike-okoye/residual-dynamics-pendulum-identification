# Learning-Based System Identification for Control of Nonlinear Dynamical Systems

## Overview
This project investigates the use of supervised learning for system identification of nonlinear dynamical systems. A neural network is trained to learn the state transition dynamics of a system from simulated data. The learned model is then used for prediction and model-based control experiments.

The repository contains tools for simulating nonlinear systems, generating training datasets, training neural network dynamics models, and evaluating their performance against analytical models.

---

## Motivation
Classical control methods depend on accurate mathematical models of physical systems. In many real-world applications, system dynamics are nonlinear, uncertain, or time-varying, making analytical modelling difficult. Data-driven approaches provide an alternative by learning system dynamics directly from observed data.

Learning-based system identification enables the development of models that can capture nonlinear behaviors and support control design when accurate analytical models are unavailable.

---

## Project Objectives
- Simulate a nonlinear dynamical system.
- Generate datasets for system identification.
- Train a neural network to learn system dynamics.
- Evaluate prediction accuracy of the learned model.
- Investigate the use of the learned model in model-based control.

---

## System Model
Experiments are conducted using a nonlinear inverted pendulum system. The system state is defined as

\[
x = [\theta, \dot{\theta}]
\]

where:
- \( \theta \) represents the pendulum angle
- \( \dot{\theta} \) represents angular velocity

The control input is a torque applied at the pivot.

The system dynamics are nonlinear and provide a suitable benchmark for evaluating learning-based identification methods.

---

## Methodology
The project follows a data-driven system identification workflow:

1. Simulate nonlinear system dynamics.
2. Generate training data using randomized control inputs.
3. Train a neural network to approximate the state transition function.
4. Evaluate model accuracy on unseen trajectories.
5. Compare predictions from the learned model with analytical dynamics.

The learned model represents the state transition function:

\[
x_{k+1} = f_\theta(x_k, u_k)
\]

where \( f_\theta \) is a neural network parameterized by \( \theta \).

---

## Repository Structure

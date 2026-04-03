
# Residual Dynamics Learning for Nonlinear System Identification

This repository implements a **hybrid physics–neural network approach** for identifying the nonlinear dynamics of an inverted pendulum.
A neural network is trained to learn the **residual dynamics** that correct an analytical physics model.

The repository includes scripts to:

* generate simulation data
* train the residual dynamics model
* evaluate prediction accuracy
* reproduce all experimental plots

---

# Repository Structure

```
src/
├── dynamics.py            # Inverted pendulum physics model
├── dataset.py             # Dataset generation
└── train_model.py         # Residual neural network training

experiments/
├── linear_baseline.py     # Linear system identification baseline
├── evaluate.py            # RMSE / MAE evaluation
├── model_comparison.py    # One-step trajectory comparison
├── rollout_trajectory.py  # Multi-step rollout trajectory
├── rollout_error_curve.py # Rollout RMSE vs horizon
└── phase_portrait.py      # Phase portrait comparison

models/                    # Trained neural network weights
results/                   # Generated plots and tables
```

---

# Installation

Install the required Python packages:

```bash
pip install torch numpy matplotlib pandas
```

---

# 1. Train the Residual Dynamics Model

Run the training script:

```bash
python -m src.train_model
```

This generates the trained model:

```
models/dynamics_model.pth
```

---

# 2. Compute Quantitative Evaluation Metrics

Run:

```bash
python -m experiments.evaluate
```

This produces a table comparing the residual model and the linear baseline.

Output:

```
results/error_table.csv
```

---

# 3. Generate Figures

The following scripts reproduce the figures used in the results section.

---

## Trajectory Comparison

```bash
python -m experiments.model_comparison
```

Output:

```
results/model_comparison.pdf
```

Compares the predicted trajectory of each model against the true dynamics.

---

## Multi-Step Rollout Trajectory

```bash
python -m experiments.rollout_trajectory
```

Output:

```
results/rollout_trajectory.pdf
```

Shows long-horizon prediction behavior.

---

## Rollout Error Curve

```bash
python -m experiments.rollout_error_curve
```

Output:

```
results/rmse_curve.pdf
```

Shows prediction error growth over the rollout horizon.

---

## Phase Portrait

```bash
python -m experiments.phase_portrait
```

Output:

```
results/phase_portrait.pdf
```

Compares the state-space dynamics of each model.

---

# Results

Running the scripts above reproduces:

* trajectory comparison
* phase portrait
* rollout trajectory
* rollout error curve
* RMSE / MAE evaluation table

All outputs are saved in the **results/** directory.

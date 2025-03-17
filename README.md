This repository contains an implementation of the deep learning algorithms, applied to a datasets.

# Perceptron.py

## Perceptron.py Description
- The dataset consists of **1000 samples**, each with **10 features**.
- The true decision boundary is defined by the vector **w* = [1, 0, ..., 0]**.
- The perceptron algorithm learns a separating hyperplane by iteratively updating its weights.
- A convergence graph is plotted to visualize the decrease in misclassifications over iterations.

## Features
- **Unit Sphere Sampling**: Randomly generates points on a 10-dimensional unit sphere.
- **Perceptron Learning**: Implements the perceptron update rule to find a linear decision boundary.
- **Convergence Tracking**: Logs the number of misclassified points at each iteration.
- **Visualization**: Plots a graph showing the number of misclassifications over iterations.

## Installation
Ensure you have Python installed along with NumPy and Matplotlib:

# SGDandFGD.py

This repository contains a comparison of **Full Gradient Descent (GD)** and **Stochastic Gradient Descent (SGD)** for optimizing a linear regression model.

## 📝 Description
- **Data**: 100 samples, 10 features (`x ∈ [0,1]^{10}`)
- **True Coefficients**: `[1, 2, ..., 10]`
- **Noise**: `0.1 * N(0,1)`
- **Optimization Methods**:
  - Full Gradient Descent (Batch GD)
  - Stochastic Gradient Descent (SGD)
- **Objective**: Minimize the **Mean Squared Error (MSE)**
- **Convergence Criteria**:
  - `||grad|| / ||w|| < 1e-6` (relative gradient norm)
  - Ensures stopping when weight updates become insignificant

## 📌 Features
- **Full Gradient Descent**:
  - Uses **all** data points for each update.
  - Slower but provides stable convergence.
- **Stochastic Gradient Descent**:
  - Updates weights using **a single random data point** per iteration.
  - Faster but introduces more variance.

## 🔧 Installation
Ensure you have Python installed along with NumPy and Matplotlib:

```sh
pip install numpy matplotlib sklearn



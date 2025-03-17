# Perceptron Algorithm on High-Dimensional Unit Sphere

This repository contains an implementation of the Perceptron learning algorithm, applied to a dataset sampled from a 10-dimensional unit sphere.

## Description
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

```sh
pip install numpy matplotlib

import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

n_points = 1000
n_dim = 10

# w* = [1, 0, ..., 0]
w_star = np.zeros(n_dim)
w_star[0] = 1

def sample_unit_sphere(n, d):
    """Samples n points from a unit sphere in d-dimensional space."""
    X = np.random.randn(n, d)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

# Sample from the unit sphere
X_all = sample_unit_sphere(n_points * 2, n_dim)  # Take extra samples
# Margin: w*^T x = x[0] (since w* = [1,0,...,0])
margins = X_all[:, 0]

# Select samples with margin at least 0.1 (positive or negative)
mask = np.abs(margins) >= 0.1
X_filtered = X_all[mask]

# Ensure we have enough samples
while X_filtered.shape[0] < n_points:
    X_more = sample_unit_sphere(n_points, n_dim)
    margins_more = X_more[:, 0]
    mask_more = np.abs(margins_more) >= 0.1
    X_filtered = np.concatenate([X_filtered, X_more[mask_more]], axis=0)

# Take the first 1000 samples
X = X_filtered[:n_points]
# Labels: y = sign(w*^T x) = sign(x[0])
y = np.sign(X[:, 0])
# If any label is zero (very unlikely), set it to 1
y[y == 0] = 1

# Perceptron algorithm
def perceptron(X, y, max_iter=100):
    """Trains a perceptron classifier and tracks misclassifications per iteration."""
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    mistakes = []
    
    for it in range(max_iter):
        error_count = 0
        for i in range(n_samples):
            # Update weights if the sample is misclassified
            if y[i] * np.dot(w, X[i]) <= 0:
                w = w + y[i] * X[i]
                error_count += 1

        mistakes.append(error_count)
        
        # Stop if there are no misclassifications
        if error_count == 0:
            print(f"Converged in {it+1} iterations.")
            break

    return w, mistakes

w_perceptron, mistakes_history = perceptron(X, y, max_iter=10)  # Reduced max_iter

print("Learned weights:", w_perceptron)
print("True w*:", w_star)

# Plot misclassifications over iterations
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(mistakes_history) + 1), mistakes_history, marker='o')
plt.xlabel("Iteration")
plt.ylabel("Number of Misclassifications")
plt.title("Perceptron Convergence")
plt.show()

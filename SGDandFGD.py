import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def generate_data(n_samples=100, n_features=10):
    np.random.seed(42)  # For reproducibility
    X = np.random.uniform(0, 1, (n_samples, n_features))  # X in range [0,1]
    noise = 0.1 * np.random.normal(0, 1, n_samples)  # Gaussian noise
    y = np.sum([(i + 1) * X[:, i] for i in range(n_features)], axis=0) + noise
    return X, y

def gradient_descent(X, y, lr=0.05, epoch=100):
    m = np.zeros(X.shape[1])  # Initial parameters
    b = 0
    log, mse = [], []
    N = len(X)
    
    for _ in range(epoch):
        f = y - (X.dot(m) + b)
        m -= lr * (-2 * X.T.dot(f) / N)
        b -= lr * (-2 * np.sum(f) / N)
        
        log.append((m.copy(), b))
        mse.append(mean_squared_error(y, X.dot(m) + b))
    
    return m, b, log, mse

def SGD(X, y, lr=0.05, epoch=100, batch_size=1):
    m = np.zeros(X.shape[1])  # Initial parameters
    b = 0
    log, mse = [], []
    
    for _ in range(epoch):
        indexes = np.random.randint(0, len(X), batch_size)
        Xs, ys = X[indexes], y[indexes]
        N = len(Xs)
        
        f = ys - (Xs.dot(m) + b)
        m -= lr * (-2 * Xs.T.dot(f) / N)
        b -= lr * (-2 * np.sum(f) / N)
        
        log.append((m.copy(), b))
        mse.append(mean_squared_error(y, X.dot(m) + b))
    
    return m, b, log, mse

# Generate data
X, y = generate_data()

# Run Gradient Descent
m_gd, b_gd, log_gd, mse_gd = gradient_descent(X, y)

# Run Stochastic Gradient Descent
m_sgd, b_sgd, log_sgd, mse_sgd = SGD(X, y)

# Plot MSE comparison
plt.plot(mse_gd, label='GD MSE')
plt.plot(mse_sgd, label='SGD MSE')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('MSE Progression for GD vs. SGD')
plt.show()

# Print final results
print("Final GD Coefficients:", m_gd, "Intercept:", b_gd)
print("Final SGD Coefficients:", m_sgd, "Intercept:", b_sgd)

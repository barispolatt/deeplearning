import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Rastgelelik için seed ayarlanıyor.
np.random.seed(42)

# 100 adet 10 boyutlu veri üretimi: x ∈ [0,1]^{10}
n_samples, n_features = 100, 10
X = np.random.uniform(0, 1, (n_samples, n_features))

# Gerçek katsayılar: [1, 2, ..., 10]
true_w = np.arange(1, n_features + 1)

# gürültülü çıktı üretimi: y = sum(i*x_i) + 0.1 * N(0,1)
noise = 0.1 * np.random.randn(n_samples)
y = X.dot(true_w) + noise

# Kayıp fonksiyonu (ortalama kare hata)
def compute_loss(w, X, y):
    pred = X.dot(w)
    return np.mean((pred - y)**2)

# Gradient hesaplama (tam veri kümesi için)
def compute_grad(w, X, y):
    pred = X.dot(w)
    grad = 2 * X.T.dot(pred - y) / X.shape[0]
    return grad

# Full Gradient Descent
def gradient_descent(X, y, lr=0.05, epoch=10):
    
    '''
    Gradient Descent for a single feature
    '''
    
    m, b = 0.33, 0.48 # parameters
    log, mse = [], [] # lists to store learning process
    N = len(X) # number of samples
    
    for _ in range(epoch):
                
        f = y - (m*X + b)
    
        # Updating m and b
        m -= lr * (-2 * X.dot(f).sum() / N)
        b -= lr * (-2 * f.sum() / N)
        
        log.append((m, b))
        mse.append(mean_squared_error(y, (m*X + b)))        
    
    return m, b, log, mse

# Stochastic Gradient Descent
def SGD(X, y, lr=0.05, epoch=10, batch_size=1):
        
    '''
    Stochastic Gradient Descent for a single feature
    '''
    
    m, b = 0.33, 0.48 # initial parameters
    log, mse = [], [] # lists to store learning process
    
    for _ in range(epoch):
        
        indexes = np.random.randint(0, len(X), batch_size) # random sample
        
        Xs = np.take(X, indexes)
        ys = np.take(y, indexes)
        N = len(Xs)
        
        f = ys - (m*Xs + b)
        
        # Updating parameters m and b
        m -= lr * (-2 * Xs.dot(f).sum() / N)
        b -= lr * (-2 * f.sum() / N)
        
        log.append((m, b))
        mse.append(mean_squared_error(y, m*X+b))        
    
    return m, b, log, mse

# Full GD ile optimizasyon
w_full, loss_full = full_gradient_descent(X, y, lr=0.05, n_iter=10)
print("Full Gradient Descent sonucu (w):", w_full)
print("Gerçek katsayılar:", true_w)

# SGD ile optimizasyon
w_sgd, loss_sgd = stochastic_gradient_descent(X, y, lr=0.055, n_iter=10)
print("Stochastic Gradient Descent sonucu (w):", w_sgd)
print("Gerçek katsayılar:", true_w)

# Kayıp değerlerinin görselleştirilmesi
plt.figure(figsize=(10,4))
plt.plot(loss_full, label="Full GD")
plt.plot(loss_sgd, label="SGD", alpha=0.7)
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.legend()
plt.title("Gradient Descent vs Stochastic Gradient Descent")
plt.show()

import numpy as np
import pandas as pd

data = {
    "size": [80, 120, 150, 200, 90, 180, 220, 130, 250, 110, 300, 170, 200, 90, 140],
    "location": [1, 2, 3, 3, 1, 2, 3, 2, 3, 2, 3, 2, 3, 1, 2],
    "num_bedrooms": [2, 3, 3, 4, 2, 3, 4, 3, 5, 2, 5, 3, 4, 2, 3],
    "sea_distance": [5, 3, 2, 1, 4, 2, 1, 3, 1, 4, 1, 2, 2, 5, 3],
    "price": [1200, 2500, 4000, 6500, 1800, 4800, 7200, 3000, 8500, 2400, 9500, 4600, 7000, 1500, 3500]
}

df = pd.DataFrame(data)

# --- Normalization ---
X = df[["size", "location", "num_bedrooms", "sea_distance"]]
X = (X - X.mean()) / X.std()
X.insert(0, "ones", 1)
Y = df["price"].values

X = X.values
W = np.random.randn(X.shape[1]) * 0.01

LR = 0.1
iterations = 1000

def gradient(X, Y, W):
    N = len(Y)
    y_pred = np.dot(X, W)
    #print(f"y_pred\n{y_pred}")
    error = y_pred - Y
    #print(f"error\n{error}")
    grad = (1/N) * np.dot(X.T, error)
    #print(f"grad\n{grad}")
    return grad

for i in range(iterations):
    grads = gradient(X, Y, W)
    W = W - LR * grads

    if i % 100 == 0:
        loss = np.mean((np.dot(X, W) - Y)**2)
        print(f"Iter {i}, Loss = {loss:.2f}")

print("Final Weights:", W)


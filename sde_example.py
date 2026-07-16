"""
Ornstein-Uhlenbeck process: a mean-reverting stochastic differential
equation,

    dX = -THETA * X * dt + SIGMA * dW

simulated with the Euler-Maruyama method. Unlike the Wiener process /
random walk above, each step depends on the *current* value (the
mean-reversion term), so it can't be reduced to a plain cumulative sum
-- the loop below is inherent to the recursion, not something to
vectorize away.
"""
import matplotlib.pyplot as plt
import numpy as np

NUM_STEPS = 500
DT = 0.01
THETA = 10   # mean-reversion speed: how fast X is pulled back toward 0
SIGMA = 3    # noise magnitude

if __name__ == "__main__":
    rng = np.random.default_rng()
    dW = rng.normal(0, np.sqrt(DT), NUM_STEPS)   # Wiener process increments

    x = np.zeros(NUM_STEPS)
    for i in range(1, NUM_STEPS):
        x[i] = x[i - 1] - THETA * x[i - 1] * DT + SIGMA * dW[i - 1]

    plt.plot(x)
    plt.title("Ornstein-Uhlenbeck process (mean-reverting SDE)")
    plt.show()

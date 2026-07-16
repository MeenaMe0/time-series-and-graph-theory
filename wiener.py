"""
Wiener process (standard Brownian motion): the cumulative sum of
Gaussian increments with variance DT per step, so the process has
independent, stationary increments and variance growing linearly
with time.
"""
import matplotlib.pyplot as plt
import numpy as np

NUM_STEPS = 500
DT = 0.01
SIGMA = np.sqrt(DT)   # std dev per step, chosen so Var(x[t]) = t * DT

if __name__ == "__main__":
    rng = np.random.default_rng()
    increments = rng.normal(0, SIGMA, size=NUM_STEPS - 1)
    x = np.concatenate(([0.0], np.cumsum(increments)))

    plt.plot(x)
    plt.title("Wiener process")
    plt.show()

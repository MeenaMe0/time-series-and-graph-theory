"""
Random walk: the cumulative sum of an IID +1/-1 binary process (see
binary.py). Each step moves +1 with probability P_PLUS, else -1.
"""
import matplotlib.pyplot as plt
import numpy as np

NUM_STEPS = 500
P_PLUS = 0.8   # P(+1); P(-1) = 1 - P_PLUS

if __name__ == "__main__":
    rng = np.random.default_rng()
    steps = rng.choice([-1, 1], size=NUM_STEPS - 1, p=[1 - P_PLUS, P_PLUS])
    random_walk = np.concatenate(([0], np.cumsum(steps)))

    # Theoretical mean path: a + t*(2p - 1), starting from a = 0
    drift = 2 * P_PLUS - 1

    plt.plot(random_walk)
    plt.plot((0, NUM_STEPS), (0, NUM_STEPS * drift), color="grey", label=f"drift = {drift:.2f}")
    plt.title("Random walk")
    plt.legend()
    plt.show()

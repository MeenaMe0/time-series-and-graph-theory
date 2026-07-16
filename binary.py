"""
Binary process: an IID sequence taking value +1 or -1 at each step,
with P(+1) = P_PLUS. This is the building block for the random walk
in randomwalk.py (a random walk is just the cumulative sum of one).
"""
import matplotlib.pyplot as plt
import numpy as np

NUM_STEPS = 500
P_PLUS = 0.8   # P(+1); P(-1) = 1 - P_PLUS

if __name__ == "__main__":
    rng = np.random.default_rng()
    binary_p = rng.choice([-1, 1], size=NUM_STEPS, p=[1 - P_PLUS, P_PLUS])

    # Theoretical mean: E[X] = (+1)*p + (-1)*(1-p) = 2p - 1
    mean = 2 * P_PLUS - 1

    plt.plot(binary_p)
    plt.plot((0, NUM_STEPS), (mean, mean), color="grey", label=f"mean = {mean:.2f}")
    plt.ylim(-2, 2)
    plt.title("Binary process")
    plt.legend()
    plt.show()

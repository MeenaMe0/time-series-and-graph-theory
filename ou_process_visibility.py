"""
Ornstein-Uhlenbeck process with a nonzero mean-reversion target, with
its natural visibility graph overlaid (see sde_example_visibility.py
for the zero-mean version and the visibility graph background).
"""
import matplotlib.pyplot as plt
import numpy as np

from visibility_graph import natural_visibility_graph, plot_visibility_graph

NUM_STEPS = 500
DT = 0.01
THETA = 10    # mean-reversion speed
SIGMA = 100   # noise magnitude
MEAN = 20     # mean-reversion target
VG_WINDOW = 60

if __name__ == "__main__":
    rng = np.random.default_rng()
    dW = rng.normal(0, np.sqrt(DT), NUM_STEPS)

    x = np.zeros(NUM_STEPS)
    for i in range(1, NUM_STEPS):
        x[i] = x[i - 1] + THETA * (MEAN - x[i - 1]) * DT + SIGMA * dW[i - 1]

    fig, ax = plt.subplots()
    ax.plot(x, color="black")
    ax.plot((0, NUM_STEPS), (MEAN, MEAN), color="red", label=f"mean = {MEAN}")
    ax.set_title("Ornstein-Uhlenbeck process (nonzero mean)")
    ax.legend()

    t = np.arange(VG_WINDOW, dtype=float)
    G = natural_visibility_graph(t, x[:VG_WINDOW])
    plot_visibility_graph(t, x[:VG_WINDOW], G,
                           f"OU process -- natural visibility graph (first {VG_WINDOW} points)")

    plt.show()

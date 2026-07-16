"""
Ornstein-Uhlenbeck process (see sde_example.py) with its natural
visibility graph overlaid, per Section 2.2 of the accompanying report.

The visibility graph is computed on a shorter window (VG_WINDOW points)
of the full simulation rather than all NUM_STEPS: natural_visibility_graph
handles 500 points in a fraction of a second, but a graph with hundreds
of overlapping edges is unreadable once plotted. Raise VG_WINDOW freely
if you want more of the series.
"""
import matplotlib.pyplot as plt
import numpy as np

from visibility_graph import natural_visibility_graph, plot_visibility_graph

NUM_STEPS = 500
DT = 0.01
THETA = 10   # mean-reversion speed
SIGMA = 3    # noise magnitude
VG_WINDOW = 60

if __name__ == "__main__":
    rng = np.random.default_rng()
    dW = rng.normal(0, np.sqrt(DT), NUM_STEPS)

    x = np.zeros(NUM_STEPS)
    for i in range(1, NUM_STEPS):
        x[i] = x[i - 1] - THETA * x[i - 1] * DT + SIGMA * dW[i - 1]

    fig, ax = plt.subplots()
    ax.plot(x, color="black")
    ax.set_title("Ornstein-Uhlenbeck process")

    t = np.arange(VG_WINDOW, dtype=float)
    G = natural_visibility_graph(t, x[:VG_WINDOW])
    plot_visibility_graph(t, x[:VG_WINDOW], G,
                           f"OU process -- natural visibility graph (first {VG_WINDOW} points)")

    plt.show()

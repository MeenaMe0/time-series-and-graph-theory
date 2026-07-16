"""
Generate one AR(p) time series, build its natural visibility graph, and
plot the graph's degree distribution -- exploring the report's Section
1.3 claim that a visibility graph's degree distribution (mean, std) and
the underlying series' Hurst exponent scale with series length n.

See ar_visibility_common.py for the shared AR-series/visibility-graph
helpers (this used to duplicate that logic with Adjacent_AR_test.py).
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

from ar_visibility_common import (
    DEFAULT_AR_COEFFS,
    degree_sequence,
    draw_circular,
    freq_dist,
    generate_ar_series,
    visibility_graph_with_weights,
)

NUM_STEPS = 100
AR_COEFFS = DEFAULT_AR_COEFFS   # AR(4); see ar_visibility_common.py

if __name__ == "__main__":
    ar_model = generate_ar_series(AR_COEFFS, NUM_STEPS)

    fig, ax = plt.subplots()
    ax.plot(ar_model, color="black", linewidth=0.5)
    ax.set_title(f"AR({len(AR_COEFFS)}) time series")

    G = visibility_graph_with_weights(ar_model)
    degrees = degree_sequence(G)
    degree_freq = freq_dist(degrees)

    fig, ax = plt.subplots()
    ax.plot(degree_freq)
    ax.set_title(f"AR({len(AR_COEFFS)}) visibility graph -- degree distribution")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Number of nodes")
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))

    fig, ax = plt.subplots()
    draw_circular(G, node_color=degrees)
    ax.set_title(f"AR({len(AR_COEFFS)}) visibility graph, n={NUM_STEPS}")

    plt.show()

"""
Explore how a single AR(1) realization's natural visibility graph
changes across a handful of sub-window lengths -- the report's Section
1.3 discusses how visibility-graph degree-distribution properties scale
with series length n.

The original version of this script looped over every (start, end)
sub-window pair of a 30-point series (435 pairs) and saved two PNGs per
pair to disk (~870 files per run). That's almost certainly not what you
want by default, so this version instead shows a handful of
representative window lengths interactively -- extend WINDOW_LENGTHS
below if you want a finer sweep.
"""
import matplotlib.pyplot as plt

from ar_visibility_common import (
    DEFAULT_AR_COEFFS,
    degree_sequence,
    draw_circular,
    generate_ar_series,
    visibility_graph_with_weights,
)

NUM_STEPS = 30
AR_COEFFS = DEFAULT_AR_COEFFS[:1]           # AR(1), matching the original porder = 1
WINDOW_LENGTHS = [5, 10, 15, 20, NUM_STEPS]  # sub-window lengths to compare

if __name__ == "__main__":
    ar_model = generate_ar_series(AR_COEFFS, NUM_STEPS)

    fig, ax = plt.subplots()
    ax.plot(ar_model, color="black", linewidth=0.5)
    ax.set_title(f"AR({len(AR_COEFFS)}) time series")

    for window in WINDOW_LENGTHS:
        sub_series = ar_model[-window:]   # most recent `window` points

        fig, ax = plt.subplots()
        ax.plot(sub_series, color="black", linewidth=0.5)
        ax.set_title(f"AR({len(AR_COEFFS)}), last {window} points")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")

        G = visibility_graph_with_weights(sub_series)
        fig, ax = plt.subplots()
        draw_circular(G, node_color=degree_sequence(G))
        ax.set_title(f"Visibility graph, last {window} points")

    plt.show()

"""
Visibility graph construction from a time series.

Two constructions, both from the accompanying report ("Novel Prediction
Method for Bitcoin Price Using EMD and Visibility Graph"), Section 2.2:

- Natural Visibility Graph (NVG): nodes i and j are connected iff every
  intermediate point (t_k, y_k), i < k < j, lies strictly below the
  straight line connecting (t_i, y_i) and (t_j, y_j):

      y_k < y_j + (y_i - y_j) * (t_j - t_k) / (t_j - t_i)   for all i < k < j

- Horizontal Visibility Graph (HVG): the same idea but with a horizontal
  line instead, giving a stricter condition -- the report notes HVG is
  always a subgraph of the NVG of the same series:

      y_k < min(y_i, y_j)                                    for all i < k < j

Both graphs are always connected (consecutive points i, i+1 are always
mutually visible -- the "for all k" condition is vacuous when j = i+1)
and undirected, and are invariant under scaling/shifting of the series
(report Sec. 2.2).

Both are built here with the standard O(n^2) sweep, matching the
complexity the report's Analysis section (3.4) assumes: for each start
point i, scan j = i+1, i+2, ... to the right while maintaining a single
running value (max slope for NVG, max height for HVG) instead of
re-examining every intermediate point from scratch for every (i, j)
pair.
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def natural_visibility_graph(t, y):
    """Build the NVG of series (t, y). See module docstring for the definition.

    j is visible from i iff the slope of the line (i, j) is strictly
    greater than the slope of every line (i, k) for i < k < j -- an
    algebraic rearrangement of the "below the connecting line" condition
    above, and what makes the O(n) scan per start point possible.
    """
    n = len(y)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        max_slope = -np.inf
        for j in range(i + 1, n):
            slope = (y[j] - y[i]) / (t[j] - t[i])
            if slope > max_slope:
                G.add_edge(i, j)
                max_slope = slope
    return G


def horizontal_visibility_graph(t, y):
    """Build the HVG of series (t, y). See module docstring for the definition.

    j is visible from i iff no point strictly between them is as tall as
    the shorter of the two endpoints -- tracked here as a running max
    height, with an early exit once that running max reaches y[i] (no
    later j can be visible from i after that point).
    """
    n = len(y)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        running_max = -np.inf
        for j in range(i + 1, n):
            if running_max < min(y[i], y[j]):
                G.add_edge(i, j)
            running_max = max(running_max, y[j])
            if running_max >= y[i]:
                break
    return G


def plot_visibility_graph(t, y, G, title):
    """Draw the series as bars (as in the report's explanatory figure)
    with a line for every visibility edge connecting the bar tops."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(t, y, width=0.6, color="#cccccc", zorder=1)
    for i, j in G.edges():
        ax.plot([t[i], t[j]], [y[i], y[j]], color="#1f77b4", lw=1, zorder=2)
    ax.plot(t, y, "o", color="#d62728", zorder=3, markersize=4)
    ax.set_title(f"{title}  (|V|={G.number_of_nodes()}, |E|={G.number_of_edges()})")
    return fig, ax


if __name__ == "__main__":
    # Small example series so the graph stays readable when plotted -- a
    # short random walk, same construction as randomwalk.py.
    NUM_POINTS = 25
    P_PLUS = 0.8

    rng = np.random.default_rng()
    steps = rng.choice([-1, 1], size=NUM_POINTS - 1, p=[1 - P_PLUS, P_PLUS])
    y = np.concatenate(([0], np.cumsum(steps))).astype(float)
    t = np.arange(NUM_POINTS, dtype=float)

    nvg = natural_visibility_graph(t, y)
    hvg = horizontal_visibility_graph(t, y)

    print(f"NVG: {nvg.number_of_nodes()} nodes, {nvg.number_of_edges()} edges, "
          f"mean degree {2 * nvg.number_of_edges() / nvg.number_of_nodes():.2f}")
    print(f"HVG: {hvg.number_of_nodes()} nodes, {hvg.number_of_edges()} edges, "
          f"mean degree {2 * hvg.number_of_edges() / hvg.number_of_nodes():.2f}")

    # HVG must always be a subgraph of the NVG of the same series (report Sec. 2.2)
    nvg_edges = set(frozenset(e) for e in nvg.edges())
    hvg_edges = set(frozenset(e) for e in hvg.edges())
    assert hvg_edges <= nvg_edges, "HVG should always be a subgraph of the NVG"

    plot_visibility_graph(t, y, nvg, "Natural visibility graph")
    plot_visibility_graph(t, y, hvg, "Horizontal visibility graph")
    plt.show()

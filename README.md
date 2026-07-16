# Time Series & Graph Theory

Small standalone scripts simulating stochastic processes and building
visibility graphs from time series, written while studying time series
analysis.

## What's here

| Script | Topic | Notes |
|---|---|---|
| `binary.py` | Binary process | IID +1/-1 sequence, P(+1) = 0.8 |
| `randomwalk.py` | Random walk | Cumulative sum of a binary process |
| `wiener.py` | Wiener process (Brownian motion) | Cumulative sum of Gaussian increments |
| `sde_example.py` | Ornstein-Uhlenbeck process | Mean-reverting SDE, simulated with Euler-Maruyama |
| `visibility_graph.py` | Natural & horizontal visibility graphs | Converts a time series into a graph (Lacasa et al.); see module docstring for the definitions |

Each script is self-contained -- run it directly and it plots one
realization of that process (or, for `visibility_graph.py`, a random
walk and its two visibility graphs).

## Usage

```bash
pip install numpy matplotlib networkx
python3 binary.py                # or randomwalk.py / wiener.py / sde_example.py
python3 visibility_graph.py
```

## Status

Covers a handful of stochastic-process building blocks plus the
visibility-graph constructions from the accompanying report ("Novel
Prediction Method for Bitcoin Price Using EMD and Visibility Graph",
Section 2.2). Not yet implemented: the AR/MA/ARMA/ARIMA models, the
report's EMD decomposition + weighted-random-walk forecasting pipeline,
and its degree-distribution/Hurst-exponent experiments (Section 1.3).

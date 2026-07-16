# Time Series & Graph Theory

Small standalone scripts simulating and plotting discrete- and
continuous-time stochastic processes, written while studying time
series analysis.

## What's here

| Script | Process | Notes |
|---|---|---|
| `binary.py` | Binary process | IID +1/-1 sequence, P(+1) = 0.8 |
| `randomwalk.py` | Random walk | Cumulative sum of a binary process |
| `wiener.py` | Wiener process (Brownian motion) | Cumulative sum of Gaussian increments |
| `sde_example.py` | Ornstein-Uhlenbeck process | Mean-reverting SDE, simulated with Euler-Maruyama |

Each script is self-contained -- run it directly and it plots one
realization of that process.

## Usage

```bash
pip install numpy matplotlib
python3 binary.py        # or randomwalk.py / wiener.py / sde_example.py
```

## Status

Currently covers a handful of stochastic-process building blocks. The
AR/MA/ARMA/ARIMA models and the graph-theory (visibility graph) side
implied by the repo name haven't been added yet.

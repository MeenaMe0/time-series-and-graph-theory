# Time Series & Graph Theory

Small standalone scripts simulating stochastic processes and building
visibility graphs from time series, written while studying time series
analysis and while researching the accompanying report ("Novel
Prediction Method for Bitcoin Price Using EMD and Visibility Graph").

## What's here

### Stochastic process simulations

| Script | Process | Notes |
|---|---|---|
| `binary.py` | Binary process | IID +1/-1 sequence, P(+1) = 0.8 |
| `randomwalk.py` | Random walk | Cumulative sum of a binary process |
| `wiener.py` | Wiener process (Brownian motion) | Cumulative sum of Gaussian increments |
| `sde_example.py` | Ornstein-Uhlenbeck process | Mean-reverting SDE, simulated with Euler-Maruyama |

### Visibility graphs

| Script | Topic | Notes |
|---|---|---|
| `visibility_graph.py` | Natural & horizontal visibility graphs | Core, reusable construction (Lacasa et al.); see module docstring for the definitions (report Sec. 2.2) |
| `sde_example_visibility.py` | OU process + its visibility graph | |
| `ou_process_visibility.py` | OU process, nonzero mean-reversion target, + its visibility graph | |
| `cir_process_visibility.py` | CIR process + its visibility graph | Square-root diffusion, stays non-negative |
| `ar_visibility_common.py` | Shared AR-series / weighted-visibility-graph helpers | Used by the two scripts below |
| `useful_randomthing_about_graph_AR_length.py` | AR(4) series -> visibility graph degree distribution | Explores report Sec. 1.3 |
| `Adjacent_AR_test.py` | AR(1) series -> visibility graph across several sub-window lengths | Explores report Sec. 1.3 |

Each script is self-contained -- run it directly and it plots one
realization of that process/graph. The AR scripts currently generate
their series from a placeholder coefficient array (`DEFAULT_AR_COEFFS`
in `ar_visibility_common.py`) rather than a real fitted model -- swap in
your own coefficients once you have them.

## Usage

```bash
pip install -r requirements.txt
python3 binary.py                # or randomwalk.py / wiener.py / sde_example.py
python3 visibility_graph.py
python3 sde_example_visibility.py   # or ou_process_visibility.py / cir_process_visibility.py
python3 useful_randomthing_about_graph_AR_length.py
python3 Adjacent_AR_test.py
```

## Status

Covers a handful of stochastic-process building blocks plus the
visibility-graph constructions and AR degree-distribution experiments
from the accompanying report (Sections 1.3 and 2.2). Not yet
implemented: the AR/MA/ARMA/ARIMA forecasting models themselves, and
the report's full EMD decomposition + weighted-random-walk Bitcoin
prediction pipeline (Sections 3-4).

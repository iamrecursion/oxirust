"""phop vs PySR head-to-head on phop-bench's feynman_like_suite (M6).

The phop side is the Rust `recovery` bin:

    cargo run -p phop-bench --release --bin recovery

This script is the PySR side: it replicates `synth_1d` EXACTLY (inclusive linspace, N=48, no
noise) and the same 14 laws + ranges, runs PySR (Julia/SymbolicRegression.jl backend) on each
with a modest fixed budget, and records R^2 + wall-clock + the recovered equation. Output: a
per-case line plus JSON (after the ===JSON=== marker) for the comparison table.

    python3 crates/phop-bench/scripts/pysr_headtohead.py

Observed (this repo, 2026-06-24; PySR 1.5.10, identical data): PySR recovered 14/14 (rich
operator basis exp/log/sqrt/sin/square/cube/div) at a median ~12 s/eq; phop recovered 5/14
(identity, exp, e^x - ln x, 1/x, sqrt(x) -- exactly) at a median ~0.75 s/eq, i.e. ~16x faster.
The gap is reachability (single-operator EML at depth 3 vs PySR's broad operator set), not EML
representability -- see TODO.md "Deeper recovery".
"""
import json, time, math
import numpy as np
from pysr import PySRRegressor

N = 48

def linspace_incl(lo, hi, n):
    step = (hi - lo) / (n - 1)
    return np.array([lo + step * i for i in range(n)])

# (name, f, lo, hi) — mirrors crates/phop-bench/src/lib.rs feynman_like_suite()
CASES = [
    ("identity",       lambda x: x,                          0.5, 5.0),
    ("exp",            lambda x: np.exp(x),                  0.0, 2.0),
    ("exp_2x",         lambda x: np.exp(2.0 * x),            0.0, 2.0),
    ("exp_minus_x",    lambda x: np.exp(-x),                 0.0, 3.0),
    ("exp_minus_ln",   lambda x: np.exp(x) - np.log(x),      0.5, 3.0),
    ("ln",             lambda x: np.log(x),                  0.5, 5.0),
    ("square",         lambda x: x * x,                      0.5, 5.0),
    ("cube",           lambda x: x * x * x,                  0.5, 3.0),
    ("reciprocal",     lambda x: 1.0 / x,                    0.5, 5.0),
    ("sqrt",           lambda x: np.sqrt(x),                 0.5, 5.0),
    ("pow_1_5",        lambda x: np.power(x, 1.5),           0.5, 5.0),
    ("linear",         lambda x: 2.0 * x + 1.0,              0.5, 5.0),
    ("sin",            lambda x: np.sin(x),                  0.0, math.pi),
    ("exp_neg_square", lambda x: np.exp(-(x * x)),           0.0, 2.5),
]

def r2(y, p):
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")

def make_model():
    return PySRRegressor(
        niterations=40,
        population_size=30,
        maxsize=20,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log", "sqrt", "sin", "square", "cube"],
        deterministic=True,
        parallelism="serial",
        random_state=0,
        progress=False,
        verbosity=0,
        temp_equation_file=True,
    )

# Warm up Julia/JIT once (not counted in per-case timings).
_x = linspace_incl(0.5, 5.0, N).reshape(-1, 1)
make_model().fit(_x, _x.ravel())

results = {}
for name, f, lo, hi in CASES:
    xs = linspace_incl(lo, hi, N)
    X = xs.reshape(-1, 1)
    y = f(xs)
    m = make_model()
    t0 = time.time()
    m.fit(X, y)
    dt = time.time() - t0
    pred = m.predict(X)
    score = r2(y, pred)
    best = m.get_best()
    results[name] = {
        "r2": score,
        "time_s": dt,
        "recovered": bool(score >= 0.9999),
        "eq": str(best["equation"]),
    }
    print(f"{name:16s} r2={score:.6f} t={dt:6.2f}s  {best['equation']}", flush=True)

print("===JSON===")
print(json.dumps(results))

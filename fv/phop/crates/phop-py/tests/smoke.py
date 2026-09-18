"""Smoke test for the phop Python extension.

Run after `maturin develop` (or against an installed wheel):

    maturin develop -m crates/phop-py/Cargo.toml
    python crates/phop-py/tests/smoke.py
"""

import numpy as np

import phop


def main() -> None:
    x = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float64)
    y = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float64)

    disco = phop.Discoverer(
        population=64,
        max_depth=3,
        max_epochs=50,
        seed=0,
        top_k=5,
        lambda_complexity=1e-3,
        lambda_sparsity=1e-3,
        lambda_parsimony=1e-3,
    )
    result = disco.fit(x, y)

    assert len(result) >= 0
    latexes = result.top_latex(5)
    assert isinstance(latexes, list)

    sympies = result.top_sympy(5)
    assert isinstance(sympies, list)

    best = result.best()
    if best is not None:
        assert isinstance(best.latex, str)
        assert isinstance(best.pretty, str)
        assert isinstance(best.rust_code, str)
        assert isinstance(best.numpy_code, str)
        assert isinstance(best.sympy_code, str)
        assert isinstance(best.mse, float)
        assert isinstance(best.complexity, int)
        print("best:", best.pretty, "mse=", best.mse, "complexity=", best.complexity)

    print("top latex:", latexes)
    print("top sympy:", sympies)
    print("OK")


if __name__ == "__main__":
    main()

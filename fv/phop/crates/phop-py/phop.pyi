"""Type stubs for the `phop` extension module.

These describe the Python-visible surface of the Rust `phop` extension built by
PyO3/maturin. They exist purely for editor completion and static type checkers;
the runtime behaviour is defined in the compiled module.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

class Solution:
    """A single discovered closed-form expression and its quality metrics.

    Instances are produced by :class:`DiscoveryResult` accessors and are
    read-only.
    """

    @property
    def latex(self) -> str:
        """LaTeX rendering of the expression."""
        ...

    @property
    def pretty(self) -> str:
        """Human-readable infix rendering of the expression."""
        ...

    @property
    def rust_code(self) -> str:
        """Rust source code that evaluates the expression."""
        ...

    @property
    def numpy_code(self) -> str:
        """NumPy/Python source code that evaluates the expression."""
        ...

    @property
    def sympy_code(self) -> str:
        """SymPy source code that constructs the expression."""
        ...

    @property
    def mse(self) -> float:
        """Mean squared error of the expression on the training data."""
        ...

    @property
    def complexity(self) -> int:
        """Structural complexity (node count) of the expression."""
        ...

    def __repr__(self) -> str: ...

class DiscoveryResult:
    """The Pareto front returned by :meth:`Discoverer.fit`."""

    def top_latex(self, k: int) -> list[str]:
        """Return the LaTeX strings of the top-``k`` Pareto-optimal solutions."""
        ...

    def top_sympy(self, k: int) -> list[str]:
        """Return the SymPy source strings of the top-``k`` Pareto-optimal solutions."""
        ...

    def top(self, k: int) -> list[Solution]:
        """Return the top-``k`` Pareto-optimal solutions as :class:`Solution` objects."""
        ...

    def best(self) -> Solution | None:
        """Return the single best solution, or ``None`` if the front is empty."""
        ...

    def __len__(self) -> int:
        """Number of solutions in the Pareto front."""
        ...

class Discoverer:
    """Differentiable symbolic discovery driver."""

    def __init__(
        self,
        population: int = ...,
        max_depth: int = ...,
        max_epochs: int = ...,
        learning_rate: float = ...,
        seed: int = ...,
        top_k: int = ...,
        lambda_complexity: float = ...,
        lambda_sparsity: float = ...,
        lambda_parsimony: float = ...,
    ) -> None: ...
    def fit(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> DiscoveryResult:
        """Run discovery on ``x`` (2D, rows x n_vars) and ``y`` (1D) float64 arrays."""
        ...

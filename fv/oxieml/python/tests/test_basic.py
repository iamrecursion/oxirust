"""Basic Python binding tests."""
import math
import numpy as np
import pytest
from oxieml import SymRegConfig, SymRegEngine, DiscoveredFormula


def test_symregconfig_quick():
    config = SymRegConfig.quick()
    assert config.depth_limit >= 1
    # adam_steps should be positive
    assert config.adam_steps >= 1


def test_symregconfig_setters():
    config = SymRegConfig.quick()
    config.depth_limit = 3
    assert config.depth_limit == 3
    config.max_formulas = 10
    assert config.max_formulas == 10
    config.adam_steps = 500
    assert config.adam_steps == 500


def test_symregengine_discover_linear():
    """Discover y = 2*x from synthetic data."""
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 50).reshape(-1, 1)
    y = 2.0 * x.ravel()

    config = SymRegConfig.quick()
    config.max_formulas = 5
    config.seed = 42

    engine = SymRegEngine(config)
    formulas = engine.discover(x, y)

    assert len(formulas) > 0
    # Best formula should have low MSE
    best = min(formulas, key=lambda f: f.mse)
    assert best.mse < 1.0  # very loose bound


def test_discovered_formula_attributes():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 20).reshape(-1, 1)
    y = x.ravel()

    config = SymRegConfig.quick()
    config.max_formulas = 3
    engine = SymRegEngine(config)
    formulas = engine.discover(x, y)

    for f in formulas:
        assert isinstance(f.pretty, str)
        assert isinstance(f.mse, float)
        assert isinstance(f.complexity, int)
        assert isinstance(f.score, float)


def test_engine_repr():
    engine = SymRegEngine(SymRegConfig.quick())
    assert "SymRegEngine" in repr(engine)


def test_formula_eval():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, 30).reshape(-1, 1)
    y = x.ravel()

    config = SymRegConfig.quick()
    engine = SymRegEngine(config)
    formulas = engine.discover(x, y)

    if formulas:
        f = formulas[0]
        result = f.eval([0.5])
        assert isinstance(result, float)

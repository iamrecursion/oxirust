"""Pytest configuration and fixtures for scirs2 tests."""

import pytest
import numpy as np


@pytest.fixture
def rng():
    """Provide a seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_2d_data():
    """Provide sample 2D data for clustering tests."""
    return np.array([
        [1.0, 1.0], [1.2, 0.8], [0.8, 1.2],
        [5.0, 5.0], [5.2, 4.8], [4.8, 5.2]
    ])


@pytest.fixture
def sample_timeseries():
    """Provide sample time series data."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])


@pytest.fixture
def sample_matrix():
    """Provide sample matrix for linear algebra tests."""
    return np.array([[4.0, 2.0], [2.0, 3.0]])


@pytest.fixture
def sample_vector():
    """Provide sample vector for linear algebra tests."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])

"""Tests for scirs2 spatial module."""

import pytest
import numpy as np
import scirs2


class TestDistanceFunctions:
    """Test individual distance functions."""

    def test_euclidean_basic(self):
        """Test basic Euclidean distance."""
        u = np.array([0.0, 0.0])
        v = np.array([3.0, 4.0])

        result = scirs2.euclidean_py(u, v)
        assert abs(result - 5.0) < 1e-10

    def test_euclidean_zero_distance(self):
        """Test Euclidean distance between identical points."""
        u = np.array([1.0, 2.0, 3.0])
        result = scirs2.euclidean_py(u, u)
        assert abs(result) < 1e-10

    def test_euclidean_3d(self):
        """Test Euclidean distance in 3D."""
        u = np.array([0.0, 0.0, 0.0])
        v = np.array([1.0, 2.0, 2.0])

        result = scirs2.euclidean_py(u, v)
        assert abs(result - 3.0) < 1e-10

    def test_cityblock_basic(self):
        """Test Manhattan distance."""
        u = np.array([0.0, 0.0])
        v = np.array([3.0, 4.0])

        result = scirs2.cityblock_py(u, v)
        assert abs(result - 7.0) < 1e-10

    def test_cityblock_3d(self):
        """Test Manhattan distance in 3D."""
        u = np.array([1.0, 2.0, 3.0])
        v = np.array([4.0, 6.0, 8.0])

        result = scirs2.cityblock_py(u, v)
        assert abs(result - 12.0) < 1e-10

    def test_chebyshev_basic(self):
        """Test Chebyshev distance."""
        u = np.array([0.0, 0.0])
        v = np.array([3.0, 4.0])

        result = scirs2.chebyshev_py(u, v)
        assert abs(result - 4.0) < 1e-10

    def test_chebyshev_3d(self):
        """Test Chebyshev distance in 3D."""
        u = np.array([0.0, 0.0, 0.0])
        v = np.array([1.0, 5.0, 3.0])

        result = scirs2.chebyshev_py(u, v)
        assert abs(result - 5.0) < 1e-10

    def test_minkowski_p1(self):
        """Test Minkowski with p=1 (Manhattan)."""
        u = np.array([0.0, 0.0])
        v = np.array([3.0, 4.0])

        result = scirs2.minkowski_py(u, v, 1.0)
        cityblock = scirs2.cityblock_py(u, v)
        assert abs(result - cityblock) < 1e-10

    def test_minkowski_p2(self):
        """Test Minkowski with p=2 (Euclidean)."""
        u = np.array([0.0, 0.0])
        v = np.array([3.0, 4.0])

        result = scirs2.minkowski_py(u, v, 2.0)
        euclidean = scirs2.euclidean_py(u, v)
        assert abs(result - euclidean) < 1e-10

    def test_minkowski_p3(self):
        """Test Minkowski with p=3."""
        u = np.array([0.0, 0.0])
        v = np.array([1.0, 1.0])

        # (|1|^3 + |1|^3)^(1/3) = 2^(1/3)
        result = scirs2.minkowski_py(u, v, 3.0)
        expected = 2.0 ** (1/3)
        assert abs(result - expected) < 1e-10

    def test_cosine_orthogonal(self):
        """Test cosine distance for orthogonal vectors."""
        u = np.array([1.0, 0.0])
        v = np.array([0.0, 1.0])

        result = scirs2.cosine_py(u, v)
        assert abs(result - 1.0) < 1e-10  # cos(90°) = 0, distance = 1

    def test_cosine_same_direction(self):
        """Test cosine distance for same direction vectors."""
        u = np.array([1.0, 2.0, 3.0])
        v = np.array([2.0, 4.0, 6.0])

        result = scirs2.cosine_py(u, v)
        assert abs(result) < 1e-10  # cos(0°) = 1, distance = 0

    def test_cosine_opposite(self):
        """Test cosine distance for opposite vectors."""
        u = np.array([1.0, 0.0])
        v = np.array([-1.0, 0.0])

        result = scirs2.cosine_py(u, v)
        assert abs(result - 2.0) < 1e-10  # cos(180°) = -1, distance = 2


class TestPdist:
    """Test pairwise distance computation."""

    def test_pdist_basic(self):
        """Test basic pairwise distances."""
        # 3 points in 2D
        x = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])

        result = scirs2.pdist_py(x, metric="euclidean")

        # Should be 3 distances: (0,1), (0,2), (1,2)
        assert len(result) == 3
        assert abs(result[0] - 1.0) < 1e-10  # dist(0, 1)
        assert abs(result[1] - 1.0) < 1e-10  # dist(0, 2)
        assert abs(result[2] - np.sqrt(2)) < 1e-10  # dist(1, 2)

    def test_pdist_cityblock(self):
        """Test pairwise Manhattan distances."""
        x = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0]
        ])

        result = scirs2.pdist_py(x, metric="cityblock")

        assert len(result) == 3
        assert abs(result[0] - 2.0) < 1e-10  # dist(0, 1) = 1+1
        assert abs(result[1] - 4.0) < 1e-10  # dist(0, 2) = 2+2
        assert abs(result[2] - 2.0) < 1e-10  # dist(1, 2) = 1+1

    def test_pdist_chebyshev(self):
        """Test pairwise Chebyshev distances."""
        x = np.array([
            [0.0, 0.0],
            [1.0, 2.0],
            [3.0, 1.0]
        ])

        result = scirs2.pdist_py(x, metric="chebyshev")

        assert len(result) == 3
        assert abs(result[0] - 2.0) < 1e-10  # max(1, 2)
        assert abs(result[1] - 3.0) < 1e-10  # max(3, 1)
        assert abs(result[2] - 2.0) < 1e-10  # max(2, 1)

    def test_pdist_single_point(self):
        """Test with two points only."""
        x = np.array([
            [0.0, 0.0],
            [3.0, 4.0]
        ])

        result = scirs2.pdist_py(x)
        assert len(result) == 1
        assert abs(result[0] - 5.0) < 1e-10


class TestCdist:
    """Test cross-distance computation."""

    def test_cdist_basic(self):
        """Test basic cross-distances."""
        xa = np.array([
            [0.0, 0.0],
            [1.0, 0.0]
        ])
        xb = np.array([
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.0]
        ])

        result = scirs2.cdist_py(xa, xb)

        # Result should be 2x3
        assert result.shape == (2, 3)

        # Verify some distances
        assert abs(result[0, 0] - 1.0) < 1e-10  # (0,0) to (0,1)
        assert abs(result[0, 1] - np.sqrt(2)) < 1e-10  # (0,0) to (1,1)
        assert abs(result[1, 2] - 1.0) < 1e-10  # (1,0) to (2,0)

    def test_cdist_cityblock(self):
        """Test cross-distances with Manhattan metric."""
        xa = np.array([[0.0, 0.0]])
        xb = np.array([
            [1.0, 2.0],
            [3.0, 4.0]
        ])

        result = scirs2.cdist_py(xa, xb, metric="cityblock")

        assert result.shape == (1, 2)
        assert abs(result[0, 0] - 3.0) < 1e-10
        assert abs(result[0, 1] - 7.0) < 1e-10

    def test_cdist_square(self):
        """Test square distance matrix."""
        x = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])

        result = scirs2.cdist_py(x, x)

        # Diagonal should be 0
        assert abs(result[0, 0]) < 1e-10
        assert abs(result[1, 1]) < 1e-10
        assert abs(result[2, 2]) < 1e-10

        # Should be symmetric
        assert abs(result[0, 1] - result[1, 0]) < 1e-10
        assert abs(result[0, 2] - result[2, 0]) < 1e-10


class TestSquareform:
    """Test squareform conversion."""

    def test_squareform_basic(self):
        """Test basic squareform conversion."""
        # Condensed form for 4 points: 6 distances
        condensed = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        result = scirs2.squareform_py(condensed)

        # Should be 4x4
        assert result.shape == (4, 4)

        # Diagonal should be 0
        for i in range(4):
            assert abs(result[i, i]) < 1e-10

        # Should be symmetric
        for i in range(4):
            for j in range(4):
                assert abs(result[i, j] - result[j, i]) < 1e-10

    def test_squareform_values(self):
        """Test squareform values are correct."""
        # For n=3, we have 3 distances
        condensed = np.array([1.0, 2.0, 3.0])
        result = scirs2.squareform_py(condensed)

        # Expected:
        # [[0, 1, 2],
        #  [1, 0, 3],
        #  [2, 3, 0]]
        assert abs(result[0, 1] - 1.0) < 1e-10
        assert abs(result[0, 2] - 2.0) < 1e-10
        assert abs(result[1, 2] - 3.0) < 1e-10

    def test_squareform_two_points(self):
        """Test squareform for 2 points."""
        condensed = np.array([5.0])
        result = scirs2.squareform_py(condensed)

        assert result.shape == (2, 2)
        assert abs(result[0, 1] - 5.0) < 1e-10
        assert abs(result[1, 0] - 5.0) < 1e-10


class TestEdgeCases:
    """Test edge cases."""

    def test_high_dimensional(self):
        """Test with high-dimensional points."""
        u = np.ones(100)
        v = np.zeros(100)

        result = scirs2.euclidean_py(u, v)
        assert abs(result - 10.0) < 1e-10  # sqrt(100)

    def test_pdist_many_points(self):
        """Test pdist with many points."""
        n = 50
        x = np.random.randn(n, 3)

        result = scirs2.pdist_py(x)

        # n*(n-1)/2 distances
        expected_len = n * (n - 1) // 2
        assert len(result) == expected_len

        # All distances should be non-negative
        assert all(d >= 0 for d in result)

    def test_cdist_different_sizes(self):
        """Test cdist with different sized inputs."""
        xa = np.random.randn(10, 5)
        xb = np.random.randn(20, 5)

        result = scirs2.cdist_py(xa, xb)

        assert result.shape == (10, 20)

    def test_unit_vectors(self):
        """Test cosine distance for unit vectors."""
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])

        result = scirs2.cosine_py(u, v)
        assert abs(result - 1.0) < 1e-10


class TestKDTree:
    """Test KD-Tree spatial data structure."""

    def test_kdtree_basic(self):
        """Test basic KDTree construction and query."""
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ])

        tree = scirs2.KDTree(points)

        # Query for nearest neighbor to (0.1, 0.1)
        result = tree.query(np.array([0.1, 0.1]), k=1)
        assert "indices" in result
        assert "distances" in result
        assert len(result["indices"]) == 1
        # Should find (0, 0) as nearest
        assert result["indices"][0] == 0

    def test_kdtree_knn(self):
        """Test k-nearest neighbors."""
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5]
        ])

        tree = scirs2.KDTree(points)

        # Query for 3 nearest neighbors to (0.5, 0.5)
        result = tree.query(np.array([0.5, 0.5]), k=3)
        assert len(result["indices"]) == 3
        # (0.5, 0.5) is closest to itself (index 4)
        assert 4 in result["indices"]

    def test_kdtree_radius_query(self):
        """Test radius-based query."""
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ])

        tree = scirs2.KDTree(points)

        # Query for all points within radius 1.5 of origin
        result = tree.query_radius(np.array([0.0, 0.0]), r=1.5)
        assert "indices" in result
        # Should find (0, 0), (1, 0), (0, 1)
        assert len(result["indices"]) >= 3

    def test_kdtree_3d(self):
        """Test KDTree in 3D."""
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

        tree = scirs2.KDTree(points)
        result = tree.query(np.array([0.1, 0.1, 0.1]), k=1)
        assert len(result["indices"]) == 1
        # Origin should be closest
        assert result["indices"][0] == 0

    def test_kdtree_many_points(self):
        """Test KDTree with many random points."""
        np.random.seed(42)
        points = np.random.randn(100, 3)

        tree = scirs2.KDTree(points)

        # Query should work
        result = tree.query(np.array([0.0, 0.0, 0.0]), k=5)
        assert len(result["indices"]) == 5
        assert len(result["distances"]) == 5

        # Distances should be sorted (ascending)
        for i in range(len(result["distances"]) - 1):
            assert result["distances"][i] <= result["distances"][i + 1]


class TestConvexHull:
    """Test convex hull computation."""

    def test_convex_hull_triangle(self):
        """Test convex hull of a triangle."""
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0]
        ])

        result = scirs2.convex_hull_py(points)

        assert "vertices" in result
        assert "simplices" in result
        assert "area" in result

        # All 3 points should be vertices
        assert len(result["vertices"]) == 3

    def test_convex_hull_square(self):
        """Test convex hull of a square."""
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]
        ])

        result = scirs2.convex_hull_py(points)

        # All 4 corners should be hull vertices
        assert len(result["vertices"]) == 4
        # Area should be positive (exact value depends on implementation)
        assert result["area"] > 0

    def test_convex_hull_with_interior(self):
        """Test convex hull with interior points."""
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5]  # Interior point
        ])

        result = scirs2.convex_hull_py(points)

        # Only 4 corners should be hull vertices
        assert len(result["vertices"]) == 4

    def test_convex_hull_3d(self):
        """Test convex hull in 3D (tetrahedron)."""
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

        result = scirs2.convex_hull_py(points)

        assert "vertices" in result
        assert "volume" in result

        # All 4 points should be vertices
        assert len(result["vertices"]) == 4
        # Volume should be 1/6
        assert abs(result["volume"] - (1.0 / 6.0)) < 0.01

    def test_convex_hull_class(self):
        """Test ConvexHullPy class."""
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0]
        ])

        hull = scirs2.ConvexHullPy(points)

        vertices = hull.vertices()
        simplices = hull.simplices()
        area = hull.area()

        assert len(vertices) == 3
        assert len(simplices) > 0
        assert area > 0

    def test_convex_hull_contains(self):
        """Test point containment check."""
        points = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0]
        ])

        hull = scirs2.ConvexHullPy(points)

        # Interior point should be contained
        assert hull.contains(np.array([1.0, 1.0]))

        # Point on vertex should be contained
        assert hull.contains(np.array([0.0, 0.0]))

    def test_convex_hull_many_points(self):
        """Test convex hull with many random points."""
        np.random.seed(42)
        # Points in a unit circle (exclude endpoint to avoid duplicate)
        theta = np.linspace(0, 2*np.pi, 21)[:-1]  # 20 unique points
        points = np.column_stack([np.cos(theta), np.sin(theta)])

        result = scirs2.convex_hull_py(points)

        # All points should be on hull (since they're on a circle)
        assert len(result["vertices"]) == 20

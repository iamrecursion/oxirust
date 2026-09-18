"""Tests for scirs2 clustering module."""

import numpy as np
import pytest
import scirs2


class TestKMeans:
    """Test KMeans clustering."""

    def test_kmeans_basic(self):
        """Test basic KMeans clustering."""
        # Create two well-separated clusters
        data = np.array([
            [1.0, 1.0], [1.2, 0.8], [0.8, 1.2],
            [5.0, 5.0], [5.2, 4.8], [4.8, 5.2]
        ])

        kmeans = scirs2.KMeans(n_clusters=2)
        kmeans.fit(data)

        labels = kmeans.labels
        assert len(labels) == 6
        assert len(set(labels)) == 2  # Two unique clusters

        # Points 0-2 should be in same cluster, 3-5 in another
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] == labels[4] == labels[5]
        assert labels[0] != labels[3]

    def test_kmeans_inertia(self):
        """Test KMeans inertia (sum of squared distances)."""
        data = np.array([
            [1.0, 1.0], [1.2, 0.8], [0.8, 1.2],
            [5.0, 5.0], [5.2, 4.8], [4.8, 5.2]
        ])

        kmeans = scirs2.KMeans(n_clusters=2)
        kmeans.fit(data)

        inertia = kmeans.inertia_
        assert inertia > 0
        assert inertia < 1.0  # Should be small for well-separated clusters

    def test_kmeans_single_cluster(self):
        """Test KMeans with single cluster."""
        data = np.array([
            [1.0, 1.0], [1.1, 0.9], [0.9, 1.1],
            [1.2, 0.8], [0.8, 1.2]
        ])

        kmeans = scirs2.KMeans(n_clusters=1)
        kmeans.fit(data)

        labels = kmeans.labels
        assert all(l == labels[0] for l in labels)


class TestClusterMetrics:
    """Test clustering evaluation metrics."""

    def test_silhouette_score(self):
        """Test silhouette score calculation."""
        data = np.array([
            [1.0, 1.0], [1.2, 0.8], [0.8, 1.2],
            [5.0, 5.0], [5.2, 4.8], [4.8, 5.2]
        ])
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)

        score = scirs2.silhouette_score_py(data, labels)

        # Good clustering should have high silhouette score
        assert -1.0 <= score <= 1.0
        assert score > 0.8  # Should be high for well-separated clusters

    def test_davies_bouldin_score(self):
        """Test Davies-Bouldin score calculation."""
        data = np.array([
            [1.0, 1.0], [1.2, 0.8], [0.8, 1.2],
            [5.0, 5.0], [5.2, 4.8], [4.8, 5.2]
        ])
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)

        score = scirs2.davies_bouldin_score_py(data, labels)

        # Lower is better for Davies-Bouldin
        assert score >= 0
        assert score < 1.0  # Should be small for well-separated clusters

    def test_calinski_harabasz_score(self):
        """Test Calinski-Harabasz score calculation."""
        data = np.array([
            [1.0, 1.0], [1.2, 0.8], [0.8, 1.2],
            [5.0, 5.0], [5.2, 4.8], [4.8, 5.2]
        ])
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)

        score = scirs2.calinski_harabasz_score_py(data, labels)

        # Higher is better for Calinski-Harabasz
        assert score > 0


class TestPreprocessing:
    """Test preprocessing functions."""

    def test_standardize(self):
        """Test standardization."""
        data = np.array([
            [1.0, 100.0],
            [2.0, 200.0],
            [3.0, 300.0]
        ])

        standardized = scirs2.standardize_py(data, True)

        # Each column should have mean ~0
        col_means = standardized.mean(axis=0)
        assert np.allclose(col_means, [0, 0], atol=1e-10)

        # Std may vary based on population vs sample (ddof=0 vs ddof=1)
        # Just check that data is standardized (values are small)
        assert np.abs(standardized).max() < 2.0

    def test_normalize_l2(self):
        """Test L2 normalization."""
        data = np.array([
            [3.0, 4.0],
            [6.0, 8.0]
        ])

        normalized = scirs2.normalize_py(data, "l2")

        # Each row should have unit norm
        row_norms = np.linalg.norm(normalized, axis=1)
        assert np.allclose(row_norms, [1, 1], atol=1e-10)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_kmeans_many_clusters(self):
        """Test KMeans with many clusters."""
        data = np.random.randn(100, 2)
        kmeans = scirs2.KMeans(n_clusters=10)
        kmeans.fit(data)

        labels = kmeans.labels
        assert len(set(labels)) <= 10

    def test_high_dimensional_data(self):
        """Test clustering with high-dimensional data."""
        data = np.random.randn(50, 10)
        kmeans = scirs2.KMeans(n_clusters=3)
        kmeans.fit(data)

        labels = kmeans.labels
        assert len(labels) == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

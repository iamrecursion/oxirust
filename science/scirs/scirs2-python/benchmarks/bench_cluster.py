"""
Clustering Performance Benchmarks

Compares scirs2 vs scikit-learn for clustering operations.
"""

import numpy as np
from sklearn.cluster import KMeans as SKLearnKMeans
from sklearn.metrics import (
    silhouette_score as sklearn_silhouette,
    davies_bouldin_score as sklearn_davies_bouldin,
    calinski_harabasz_score as sklearn_calinski
)
import scirs2
import time
import pandas as pd
from typing import List, Dict


def main():
    """Run all clustering benchmarks"""
    print("="*80)
    print("CLUSTERING BENCHMARKS")
    print("="*80)
    print()

    all_results = []

    # K-Means clustering benchmarks
    print("Benchmarking: K-Means Clustering")

    configs = [
        (1000, 10, 5),    # 1k samples, 10 features, 5 clusters
        (5000, 10, 5),    # 5k samples, 10 features, 5 clusters
        (10000, 10, 5),   # 10k samples, 10 features, 5 clusters
        (10000, 20, 5),   # 10k samples, 20 features, 5 clusters
        (10000, 50, 10),  # 10k samples, 50 features, 10 clusters
    ]

    for n_samples, n_features, n_clusters in configs:
        print(f"  Testing K-Means: {n_samples} samples, {n_features} features, {n_clusters} clusters...",
              end="", flush=True)

        # Generate test data
        np.random.seed(42)
        X = np.ascontiguousarray(np.random.randn(n_samples, n_features))

        # Benchmark sklearn
        sklearn_times = []
        for _ in range(5):
            start = time.perf_counter()
            sklearn_model = SKLearnKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            sklearn_model.fit(X)
            sklearn_times.append(time.perf_counter() - start)
        sklearn_time = np.median(sklearn_times)

        # Benchmark scirs2
        scirs2_times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = scirs2.kmeans_py(X, n_clusters=n_clusters, max_iter=300, random_state=42)
            scirs2_times.append(time.perf_counter() - start)
        scirs2_time = np.median(scirs2_times)

        speedup = sklearn_time / scirs2_time
        print(f" ✓ (speedup: {speedup:.2f}x)")

        all_results.append({
            'operation': 'kmeans',
            'n_samples': n_samples,
            'n_features': n_features,
            'n_clusters': n_clusters,
            'sklearn_ms': sklearn_time * 1000,
            'scirs2_ms': scirs2_time * 1000,
            'speedup': speedup,
            'winner': 'scirs2' if speedup > 1.0 else 'sklearn'
        })

    # Clustering evaluation metrics
    print("\nBenchmarking: Clustering Metrics")

    for n_samples in [1000, 5000, 10000]:
        n_features = 10
        n_clusters = 5

        print(f"  Testing metrics with {n_samples} samples...", end="", flush=True)

        # Generate test data with known clusters
        np.random.seed(42)
        X = np.ascontiguousarray(np.random.randn(n_samples, n_features))

        # Get clustering result
        kmeans_result = scirs2.kmeans_py(X, n_clusters=n_clusters, random_state=42)
        labels = kmeans_result['labels']

        # Silhouette Score
        sklearn_times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = sklearn_silhouette(X, labels)
            sklearn_times.append(time.perf_counter() - start)
        sklearn_time = np.median(sklearn_times)

        scirs2_times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = scirs2.silhouette_score_py(X, labels)
            scirs2_times.append(time.perf_counter() - start)
        scirs2_time = np.median(scirs2_times)

        speedup = sklearn_time / scirs2_time
        all_results.append({
            'operation': 'silhouette_score',
            'n_samples': n_samples,
            'n_features': n_features,
            'n_clusters': n_clusters,
            'sklearn_ms': sklearn_time * 1000,
            'scirs2_ms': scirs2_time * 1000,
            'speedup': speedup,
            'winner': 'scirs2' if speedup > 1.0 else 'sklearn'
        })

        # Davies-Bouldin Score
        sklearn_times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = sklearn_davies_bouldin(X, labels)
            sklearn_times.append(time.perf_counter() - start)
        sklearn_time = np.median(sklearn_times)

        scirs2_times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = scirs2.davies_bouldin_score_py(X, labels)
            scirs2_times.append(time.perf_counter() - start)
        scirs2_time = np.median(scirs2_times)

        speedup = sklearn_time / scirs2_time
        all_results.append({
            'operation': 'davies_bouldin_score',
            'n_samples': n_samples,
            'n_features': n_features,
            'n_clusters': n_clusters,
            'sklearn_ms': sklearn_time * 1000,
            'scirs2_ms': scirs2_time * 1000,
            'speedup': speedup,
            'winner': 'scirs2' if speedup > 1.0 else 'sklearn'
        })

        # Calinski-Harabasz Score
        sklearn_times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = sklearn_calinski(X, labels)
            sklearn_times.append(time.perf_counter() - start)
        sklearn_time = np.median(sklearn_times)

        scirs2_times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = scirs2.calinski_harabasz_score_py(X, labels)
            scirs2_times.append(time.perf_counter() - start)
        scirs2_time = np.median(scirs2_times)

        speedup = sklearn_time / scirs2_time
        print(f" ✓ (avg speedup: {speedup:.2f}x)")

        all_results.append({
            'operation': 'calinski_harabasz_score',
            'n_samples': n_samples,
            'n_features': n_features,
            'n_clusters': n_clusters,
            'sklearn_ms': sklearn_time * 1000,
            'scirs2_ms': scirs2_time * 1000,
            'speedup': speedup,
            'winner': 'scirs2' if speedup > 1.0 else 'sklearn'
        })

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Print results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    summary = df.groupby('operation').agg({
        'speedup': ['mean', 'min', 'max'],
        'winner': lambda x: (x == 'scirs2').sum() / len(x) * 100
    }).round(2)
    summary.columns = ['avg_speedup', 'min_speedup', 'max_speedup', 'scirs2_win_rate_%']
    print(summary.to_string())

    overall_speedup = df['speedup'].mean()
    scirs2_wins = (df['winner'] == 'scirs2').sum()
    total_tests = len(df)

    print(f"\n**Overall Average Speedup**: {overall_speedup:.2f}x")
    print(f"**SciRS2 Wins**: {scirs2_wins}/{total_tests} ({scirs2_wins/total_tests*100:.1f}%)")

    # Save results
    output_file = '/tmp/bench_cluster_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Create markdown report
    md_file = '/tmp/bench_cluster_report.md'
    with open(md_file, 'w') as f:
        f.write("# Clustering Performance Benchmarks\n\n")
        f.write("## Test Configuration\n\n")
        f.write("- **K-Means**: Various sample/feature/cluster combinations\n")
        f.write("- **Metrics**: Silhouette, Davies-Bouldin, Calinski-Harabasz\n")
        f.write("- **Runs per Test**: 5 for K-Means, 10 for metrics (median time)\n\n")
        f.write("## Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Summary\n\n")
        f.write(summary.to_markdown())
        f.write(f"\n\n**Overall Average Speedup**: {overall_speedup:.2f}x\n")
        f.write(f"**SciRS2 Win Rate**: {scirs2_wins/total_tests*100:.1f}%\n")

    print(f"Markdown report saved to: {md_file}")

    return df


if __name__ == '__main__':
    main()

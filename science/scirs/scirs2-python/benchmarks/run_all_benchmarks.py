"""
Master Benchmark Runner

Runs all benchmark suites and generates comprehensive performance report.
"""

import sys
import time
import subprocess
from pathlib import Path


def run_benchmark(script_name: str) -> bool:
    """Run a single benchmark script"""
    print(f"\n{'='*80}")
    print(f"Running: {script_name}")
    print(f"{'='*80}\n")

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=Path(__file__).parent,
            check=True,
            capture_output=False
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FAILED: {script_name}")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {script_name}")
        print(f"Error: {e}")
        return False


def create_master_report():
    """Create a master performance report combining all benchmarks"""
    import pandas as pd

    print(f"\n{'='*80}")
    print("CREATING MASTER PERFORMANCE REPORT")
    print(f"{'='*80}\n")

    # Collect all benchmark results
    result_files = {
        'Linear Algebra': '/tmp/bench_linalg_results.csv',
        'Statistics': '/tmp/bench_stats_results.csv',
        'FFT': '/tmp/bench_fft_results.csv',
        'Clustering': '/tmp/bench_cluster_results.csv',
    }

    all_dfs = {}
    for name, file_path in result_files.items():
        try:
            df = pd.read_csv(file_path)
            all_dfs[name] = df
            print(f"✓ Loaded {name}: {len(df)} tests")
        except FileNotFoundError:
            print(f"⚠ Missing: {name}")

    if not all_dfs:
        print("No benchmark results found!")
        return

    # Combine all results
    combined = pd.concat([
        df.assign(category=name) for name, df in all_dfs.items()
    ], ignore_index=True)

    # Calculate overall statistics
    print(f"\n{'='*80}")
    print("OVERALL PERFORMANCE SUMMARY")
    print(f"{'='*80}\n")

    total_tests = len(combined)
    overall_speedup = combined['speedup'].mean()

    # Determine winner column name (winner vs winner column)
    winner_col = 'winner' if 'winner' in combined.columns else None

    if winner_col:
        scirs2_wins = (combined[winner_col] == 'scirs2').sum()
        win_rate = scirs2_wins / total_tests * 100
    else:
        # Fallback: calculate from speedup
        scirs2_wins = (combined['speedup'] > 1.0).sum()
        win_rate = scirs2_wins / total_tests * 100

    print(f"Total Benchmarks: {total_tests}")
    print(f"Overall Average Speedup: {overall_speedup:.2f}x")
    print(f"SciRS2 Wins: {scirs2_wins}/{total_tests} ({win_rate:.1f}%)")

    # Category breakdown
    print(f"\n{'='*80}")
    print("CATEGORY BREAKDOWN")
    print(f"{'='*80}\n")

    category_summary = combined.groupby('category').agg({
        'speedup': ['count', 'mean', 'min', 'max']
    }).round(2)
    category_summary.columns = ['tests', 'avg_speedup', 'min_speedup', 'max_speedup']
    print(category_summary.to_string())

    # Performance tiers
    print(f"\n{'='*80}")
    print("PERFORMANCE TIERS")
    print(f"{'='*80}\n")

    faster_2x = (combined['speedup'] >= 2.0).sum()
    faster_1_5x = ((combined['speedup'] >= 1.5) & (combined['speedup'] < 2.0)).sum()
    faster_1x = ((combined['speedup'] >= 1.0) & (combined['speedup'] < 1.5)).sum()
    slower = (combined['speedup'] < 1.0).sum()

    print(f"≥2.0x faster: {faster_2x} tests ({faster_2x/total_tests*100:.1f}%)")
    print(f"1.5-2.0x faster: {faster_1_5x} tests ({faster_1_5x/total_tests*100:.1f}%)")
    print(f"1.0-1.5x faster: {faster_1x} tests ({faster_1x/total_tests*100:.1f}%)")
    print(f"<1.0x (slower): {slower} tests ({slower/total_tests*100:.1f}%)")

    # Create master markdown report
    md_file = '/tmp/BENCHMARK_MASTER_REPORT.md'
    with open(md_file, 'w') as f:
        f.write("# SciRS2-Python Performance Benchmarks\n\n")
        f.write("## Overall Summary\n\n")
        f.write(f"- **Total Benchmarks**: {total_tests}\n")
        f.write(f"- **Overall Average Speedup**: {overall_speedup:.2f}x\n")
        f.write(f"- **SciRS2 Win Rate**: {win_rate:.1f}%\n\n")

        f.write("## Category Performance\n\n")
        f.write(category_summary.to_markdown())
        f.write("\n\n")

        f.write("## Performance Distribution\n\n")
        f.write(f"- **≥2.0x faster**: {faster_2x} tests ({faster_2x/total_tests*100:.1f}%)\n")
        f.write(f"- **1.5-2.0x faster**: {faster_1_5x} tests ({faster_1_5x/total_tests*100:.1f}%)\n")
        f.write(f"- **1.0-1.5x faster**: {faster_1x} tests ({faster_1x/total_tests*100:.1f}%)\n")
        f.write(f"- **<1.0x (slower)**: {slower} tests ({slower/total_tests*100:.1f}%)\n\n")

        f.write("## Detailed Results by Category\n\n")

        for name, df in all_dfs.items():
            f.write(f"### {name}\n\n")
            f.write(df.head(10).to_markdown(index=False))
            f.write(f"\n\n*({len(df)} total tests)*\n\n")

    print(f"\nMaster report saved to: {md_file}")

    # Save combined results
    combined_file = '/tmp/bench_all_combined.csv'
    combined.to_csv(combined_file, index=False)
    print(f"Combined results saved to: {combined_file}")


def main():
    """Run all benchmarks and create master report"""
    start_time = time.time()

    print("="*80)
    print("SCIRS2-PYTHON BENCHMARK SUITE")
    print("="*80)
    print()
    print("This will run comprehensive performance benchmarks comparing:")
    print("  - scirs2 vs SciPy/NumPy (linalg, stats, fft)")
    print("  - scirs2 vs scikit-learn (clustering)")
    print()
    print("Estimated time: 5-15 minutes depending on system")
    print("="*80)

    # List of benchmark scripts
    benchmarks = [
        'bench_stats.py',      # Fast - run first
        'bench_fft.py',        # Fast
        'bench_cluster.py',    # Medium
        'bench_linalg.py',     # Slow - run last
    ]

    # Run each benchmark
    results = {}
    for script in benchmarks:
        success = run_benchmark(script)
        results[script] = success

    # Print summary
    print(f"\n{'='*80}")
    print("BENCHMARK EXECUTION SUMMARY")
    print(f"{'='*80}\n")

    for script, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {script}")

    successful = sum(results.values())
    print(f"\nCompleted: {successful}/{len(benchmarks)} benchmarks")

    # Create master report
    if successful > 0:
        create_master_report()

    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Total Time: {elapsed:.1f} seconds")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

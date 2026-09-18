"""
SciRS2 Pandas Integration Examples

Demonstrates seamless integration with pandas DataFrames and Series.
"""

import pandas as pd
import numpy as np
import scirs2


def timeseries_example():
    """Convert between pandas Series and SciRS2 TimeSeries"""
    print("\n=== TimeSeries Conversion Example ===")

    # Create pandas Series with datetime index
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    values = np.cumsum(np.random.randn(100)) + 100
    series = pd.Series(values, index=dates)

    print(f"Original pandas Series shape: {series.shape}")
    print(f"Date range: {series.index[0]} to {series.index[-1]}")

    # Convert to SciRS2 TimeSeries
    ts = scirs2.pandas_to_timeseries(series)
    print(f"Converted to TimeSeries: {len(ts)} observations")

    # Fit ARIMA model
    print("\nFitting ARIMA(1,1,1) model...")
    arima = scirs2.PyARIMA(1, 1, 1)
    arima.fit(ts)

    # Forecast
    forecast = arima.forecast(10)
    print(f"Forecast next 10 days: {forecast[:5]}...")

    # Convert back to pandas
    forecast_ts = scirs2.PyTimeSeries(forecast, None)
    forecast_series = scirs2.timeseries_to_pandas(forecast_ts)
    print(f"Forecast as pandas Series: {forecast_series.shape}")


def dataframe_statistics_example():
    """Apply SciRS2 functions to DataFrame columns"""
    print("\n=== DataFrame Statistics Example ===")

    # Create sample DataFrame
    df = pd.DataFrame({
        'temperature': np.random.normal(25, 5, 1000),
        'humidity': np.random.normal(60, 10, 1000),
        'pressure': np.random.normal(1013, 20, 1000),
        'wind_speed': np.random.exponential(10, 1000)
    })

    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Apply scirs2 functions to each column
    print("\nCalculating statistics with SciRS2...")

    means = scirs2.apply_to_dataframe(df, scirs2.mean_py)
    print(f"\nMeans:\n{means}")

    stds = scirs2.apply_to_dataframe(df, scirs2.std_py)
    print(f"\nStandard deviations:\n{stds}")

    medians = scirs2.apply_to_dataframe(df, scirs2.median_py)
    print(f"\nMedians:\n{medians}")


def rolling_statistics_example():
    """Rolling window operations with SciRS2"""
    print("\n=== Rolling Statistics Example ===")

    # Create time series
    dates = pd.date_range('2024-01-01', periods=200, freq='H')
    values = np.cumsum(np.random.randn(200)) + 50
    series = pd.Series(values, index=dates)

    print(f"Series shape: {series.shape}")

    # Calculate rolling statistics using SciRS2
    window = 24  # 24-hour window

    print(f"\nCalculating {window}-hour rolling statistics...")

    rolling_mean = scirs2.rolling_apply(series, window, scirs2.mean_py)
    rolling_std = scirs2.rolling_apply(series, window, scirs2.std_py)
    rolling_min = scirs2.rolling_apply(
        series,
        window,
        lambda x: np.min(x.to_numpy())  # Using numpy for min
    )
    rolling_max = scirs2.rolling_apply(
        series,
        window,
        lambda x: np.max(x.to_numpy())  # Using numpy for max
    )

    # Create result DataFrame
    result_df = pd.DataFrame({
        'value': series,
        'rolling_mean': rolling_mean,
        'rolling_std': rolling_std,
        'rolling_min': rolling_min,
        'rolling_max': rolling_max
    })

    print(f"Result DataFrame shape: {result_df.shape}")
    print(f"\nFirst few rows:\n{result_df.head(30)}")


def dataframe_transformation_example():
    """Transform DataFrame data with SciRS2"""
    print("\n=== DataFrame Transformation Example ===")

    # Create sample data
    df = pd.DataFrame({
        'A': np.random.randn(100),
        'B': np.random.randn(100),
        'C': np.random.randn(100)
    })

    print(f"Original DataFrame shape: {df.shape}")

    # Convert to numpy array
    array = scirs2.dataframe_to_array(df)
    print(f"Converted to array: {array.shape}")

    # Apply standardization
    standardized = scirs2.standardize_py(array, with_mean=True)
    print(f"Standardized array: {standardized.shape}")

    # Convert back to DataFrame
    df_standardized = scirs2.array_to_dataframe(
        standardized,
        columns=['A_std', 'B_std', 'C_std']
    )

    print(f"Standardized DataFrame:\n{df_standardized.head()}")
    print(f"\nMeans after standardization:\n{df_standardized.mean()}")
    print(f"Stds after standardization:\n{df_standardized.std()}")


def correlation_analysis_example():
    """Correlation analysis on DataFrame"""
    print("\n=== Correlation Analysis Example ===")

    # Create correlated data
    n = 1000
    x = np.random.randn(n)
    y = 0.8 * x + 0.2 * np.random.randn(n)
    z = -0.5 * x + 0.5 * np.random.randn(n)

    df = pd.DataFrame({'X': x, 'Y': y, 'Z': z})

    print(f"DataFrame shape: {df.shape}")

    # Calculate pairwise correlations using SciRS2
    print("\nPairwise correlations:")

    corr_xy = scirs2.correlation_py(df['X'].values, df['Y'].values)
    corr_xz = scirs2.correlation_py(df['X'].values, df['Z'].values)
    corr_yz = scirs2.correlation_py(df['Y'].values, df['Z'].values)

    print(f"Corr(X, Y) = {corr_xy:.4f}")
    print(f"Corr(X, Z) = {corr_xz:.4f}")
    print(f"Corr(Y, Z) = {corr_yz:.4f}")

    # Compare with pandas
    print("\nCompare with pandas:")
    print(df.corr())


def row_wise_operations_example():
    """Apply operations row-wise"""
    print("\n=== Row-wise Operations Example ===")

    # Create sample data
    df = pd.DataFrame({
        'Q1': np.random.rand(10) * 100,
        'Q2': np.random.rand(10) * 100,
        'Q3': np.random.rand(10) * 100,
        'Q4': np.random.rand(10) * 100
    })

    print("Quarterly sales data:")
    print(df)

    # Calculate row-wise mean using SciRS2
    print("\nCalculating annual averages (row-wise mean)...")
    annual_avg = scirs2.apply_along_axis(df, scirs2.mean_py, axis=1)

    df['Annual_Avg'] = annual_avg
    print(f"\nWith annual averages:\n{df}")


def main():
    """Run all examples"""
    print("SciRS2 Pandas Integration Examples")
    print("=" * 50)

    timeseries_example()
    dataframe_statistics_example()
    rolling_statistics_example()
    dataframe_transformation_example()
    correlation_analysis_example()
    row_wise_operations_example()

    print("\n" + "=" * 50)
    print("All examples completed successfully!")


if __name__ == "__main__":
    main()

# Basic usage examples for scirs2-series R integration
# 
# This script demonstrates how to use the R bindings for scirs2-series
# time series analysis functionality.

# Load the package (assuming it's installed or sourced)
source("../scirs2_series.R")

# Example 1: Create a time series and calculate basic statistics
cat("=== Example 1: Basic Time Series Operations ===\n")

# Generate sample data (monthly sales data with trend and seasonality)
set.seed(42)
n <- 120  # 10 years of monthly data
trend <- seq(100, 200, length.out = n)
seasonal <- 20 * sin(2 * pi * (1:n) / 12)
noise <- rnorm(n, 0, 5)
sales_data <- trend + seasonal + noise

# Create scirs2 time series object
ts_sales <- scirs2.ts(sales_data, frequency = 12, start = 2014)
print(ts_sales)

# Calculate descriptive statistics
stats <- scirs2.stats(ts_sales)
cat("\nDescriptive Statistics:\n")
print(stats)

# Test stationarity
is_stationary <- scirs2.is_stationary(ts_sales)
cat("\nIs stationary:", is_stationary, "\n")

# Apply differencing to make stationary
ts_diff <- scirs2.diff(ts_sales, periods = 1)
cat("\nAfter differencing (lag 1):\n")
print(ts_diff)

is_stationary_diff <- scirs2.is_stationary(ts_diff)
cat("Is differenced series stationary:", is_stationary_diff, "\n")

# Example 2: ARIMA modeling and forecasting
cat("\n=== Example 2: ARIMA Modeling ===\n")

# Fit ARIMA model manually
arima_model <- scirs2.arima(ts_sales, order = c(1, 1, 1), 
                           seasonal = list(order = c(1, 0, 1), period = 12))
cat("ARIMA model fitted successfully\n")

# Generate forecasts
forecasts <- scirs2.forecast(arima_model, n.ahead = 12)
cat("12-month forecasts:\n")
print(round(forecasts, 2))

# Auto ARIMA model selection
cat("\n=== Auto ARIMA Model Selection ===\n")
auto_model <- scirs2.auto_arima(ts_sales, seasonal = TRUE, seasonal.period = 12)
cat("Auto ARIMA model selected and fitted\n")

# Generate forecasts from auto model
auto_forecasts <- scirs2.forecast(auto_model, n.ahead = 12)
cat("Auto ARIMA 12-month forecasts:\n")
print(round(auto_forecasts, 2))

# Example 3: Anomaly detection
cat("\n=== Example 3: Anomaly Detection ===\n")

# Add some artificial anomalies to demonstrate detection
anomalous_data <- sales_data
anomalous_data[50] <- anomalous_data[50] + 100  # Large positive anomaly
anomalous_data[75] <- anomalous_data[75] - 80   # Large negative anomaly
anomalous_data[90] <- anomalous_data[90] + 60   # Medium positive anomaly

ts_anomalous <- scirs2.ts(anomalous_data, frequency = 12, start = 2014)

# Create anomaly detector
detector <- scirs2.anomaly_detector()

# Detect anomalies using IQR method
iqr_anomalies <- scirs2.detect_anomalies_iqr(detector, ts_anomalous, multiplier = 2.5)
cat("IQR anomalies detected at indices:", iqr_anomalies, "\n")

# Detect anomalies using Z-score method
zscore_anomalies <- scirs2.detect_anomalies_zscore(detector, ts_anomalous, threshold = 3.0)
cat("Z-score anomalies detected at indices:", zscore_anomalies, "\n")

# Example 4: STL Decomposition
cat("\n=== Example 4: STL Decomposition ===\n")

# Create STL decomposition object
stl_decomp <- scirs2.stl(period = 12)

# Perform decomposition
# Note: This is a simplified example - the actual implementation
# would need proper handling of the decomposition result structure
tryCatch({
  decomp_result <- scirs2.decompose(stl_decomp, ts_sales)
  cat("STL decomposition completed successfully\n")
  # In a real implementation, you would extract and plot the components
}, error = function(e) {
  cat("STL decomposition example (structure needs proper implementation)\n")
})

# Example 5: Working with different data types
cat("\n=== Example 5: Different Data Patterns ===\n")

# Example with daily data (no clear seasonality)
daily_data <- cumsum(rnorm(365, 0.1, 1))  # Random walk with drift
ts_daily <- scirs2.ts(daily_data, frequency = 1, start = 1)

cat("Daily data statistics:\n")
daily_stats <- scirs2.stats(ts_daily)
print(daily_stats)

# Check stationarity of random walk
daily_stationary <- scirs2.is_stationary(ts_daily)
cat("Daily data is stationary:", daily_stationary, "\n")

# Apply differencing to random walk
ts_daily_diff <- scirs2.diff(ts_daily, periods = 1)
daily_diff_stationary <- scirs2.is_stationary(ts_daily_diff)
cat("Daily differenced data is stationary:", daily_diff_stationary, "\n")

# Example with high-frequency data
cat("\n=== Example 6: High-Frequency Data ===\n")

# Simulate hourly data for a week
hourly_data <- 50 + 10 * sin(2 * pi * (1:168) / 24) + rnorm(168, 0, 2)
ts_hourly <- scirs2.ts(hourly_data, frequency = 24, start = 1)

cat("Hourly data (168 hours = 1 week):\n")
print(ts_hourly)

# Detect anomalies in hourly data
hourly_anomalies <- scirs2.detect_anomalies_iqr(detector, ts_hourly, multiplier = 2.0)
if (length(hourly_anomalies) > 0) {
  cat("Hourly anomalies detected at indices:", hourly_anomalies, "\n")
} else {
  cat("No anomalies detected in hourly data\n")
}

# Version information
cat("\n=== Package Information ===\n")
cat("SciRS2 Series version:", scirs2.version(), "\n")

cat("\n=== Examples completed successfully! ===\n")

# Performance comparison example (optional)
cat("\n=== Performance Comparison (Optional) ===\n")

# Generate larger dataset for performance testing
large_n <- 10000
large_data <- cumsum(rnorm(large_n, 0, 1))
ts_large <- scirs2.ts(large_data, frequency = 1, start = 1)

# Time the statistics calculation
cat("Calculating statistics for", large_n, "data points...\n")
start_time <- Sys.time()
large_stats <- scirs2.stats(ts_large)
end_time <- Sys.time()
cat("Time taken:", as.numeric(end_time - start_time, units = "secs"), "seconds\n")

# Time anomaly detection
cat("Running anomaly detection on", large_n, "data points...\n")
start_time <- Sys.time()
large_anomalies <- scirs2.detect_anomalies_iqr(detector, ts_large, multiplier = 3.0)
end_time <- Sys.time()
cat("Time taken:", as.numeric(end_time - start_time, units = "secs"), "seconds\n")
cat("Anomalies found:", length(large_anomalies), "\n")

cat("\nAll examples completed successfully!\n")
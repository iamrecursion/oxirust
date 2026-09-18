# R package interface for scirs2-series
# 
# This file provides R functions that wrap the Rust FFI implementation,
# providing seamless integration with R's time series ecosystem.

# Load the shared library
if (!is.loaded("scirs_initialize")) {
  # Try different library names based on platform
  lib_name <- if (.Platform$OS.type == "windows") {
    "scirs2_series.dll"
  } else if (Sys.info()["sysname"] == "Darwin") {
    "libscirs2_series.dylib"
  } else {
    "libscirs2_series.so"
  }
  
  # Try to load from various possible locations
  lib_paths <- c(
    file.path(system.file(package = "scirs2.series"), "libs", lib_name),
    file.path("target", "release", lib_name),
    file.path("target", "debug", lib_name),
    lib_name
  )
  
  loaded <- FALSE
  for (path in lib_paths) {
    if (file.exists(path)) {
      tryCatch({
        dyn.load(path)
        loaded <- TRUE
        break
      }, error = function(e) {
        # Continue to next path
      })
    }
  }
  
  if (!loaded) {
    stop("Could not load scirs2-series shared library. Please ensure it is compiled and available.")
  }
}

# Initialize the library
.Call("scirs_initialize")

#' Create a time series object compatible with scirs2-series
#'
#' @param values Numeric vector of time series values
#' @param frequency Frequency of the time series (default: 1)
#' @param start Start time of the series (default: 1)
#' @return A scirs2.ts object
#' @export
scirs2.ts <- function(values, frequency = 1, start = 1) {
  if (!is.numeric(values)) {
    stop("Values must be numeric")
  }
  
  if (length(values) == 0) {
    stop("Values cannot be empty")
  }
  
  # Create the time series using the Rust implementation
  ts_ptr <- .Call("scirs_create_timeseries", 
                  as.double(values), 
                  as.integer(length(values)),
                  as.double(frequency),
                  as.double(start))
  
  if (is.null(ts_ptr)) {
    stop("Failed to create time series")
  }
  
  # Create R object with additional metadata
  result <- list(
    ptr = ts_ptr,
    values = values,
    frequency = frequency,
    start = start,
    length = length(values)
  )
  
  class(result) <- "scirs2.ts"
  
  # Register finalizer to clean up memory
  reg.finalizer(result, function(obj) {
    if (!is.null(obj$ptr)) {
      .Call("scirs_free_timeseries", obj$ptr)
    }
  }, onexit = TRUE)
  
  return(result)
}

#' Print method for scirs2.ts objects
#' @param x A scirs2.ts object
#' @param ... Additional arguments (ignored)
#' @export
print.scirs2.ts <- function(x, ...) {
  cat("SciRS2 Time Series\n")
  cat("Length:", x$length, "\n")
  cat("Frequency:", x$frequency, "\n")
  cat("Start:", x$start, "\n")
  cat("Values:", paste(head(x$values, 10), collapse = ", "))
  if (x$length > 10) {
    cat(", ... (", x$length - 10, " more values)")
  }
  cat("\n")
}

#' Calculate statistics for a time series
#'
#' @param ts A scirs2.ts object
#' @return A list containing descriptive statistics
#' @export
scirs2.stats <- function(ts) {
  if (!inherits(ts, "scirs2.ts")) {
    stop("Object must be of class scirs2.ts")
  }
  
  # Call Rust implementation
  result <- .Call("scirs_calculate_statistics", ts$ptr)
  
  if (is.null(result)) {
    stop("Failed to calculate statistics")
  }
  
  return(result)
}

#' Test if a time series is stationary
#'
#' @param ts A scirs2.ts object
#' @return Logical value indicating if the series is stationary
#' @export
scirs2.is_stationary <- function(ts) {
  if (!inherits(ts, "scirs2.ts")) {
    stop("Object must be of class scirs2.ts")
  }
  
  result <- .Call("scirs_is_stationary", ts$ptr)
  
  if (result < 0) {
    stop("Error testing stationarity")
  }
  
  return(as.logical(result))
}

#' Apply differencing to a time series
#'
#' @param ts A scirs2.ts object
#' @param periods Number of periods to difference (default: 1)
#' @return A new scirs2.ts object with differenced values
#' @export
scirs2.diff <- function(ts, periods = 1) {
  if (!inherits(ts, "scirs2.ts")) {
    stop("Object must be of class scirs2.ts")
  }
  
  if (periods <= 0) {
    stop("Periods must be positive")
  }
  
  # Prepare output buffer
  max_length <- max(0, ts$length - periods)
  output <- numeric(max_length)
  
  result_length <- .Call("scirs_difference_series", 
                         ts$ptr, 
                         as.integer(periods),
                         output,
                         as.integer(max_length))
  
  if (result_length < 0) {
    stop("Error applying differencing")
  }
  
  # Return new time series with differenced values
  differenced_values <- output[1:result_length]
  return(scirs2.ts(differenced_values, ts$frequency, ts$start + periods))
}

#' Fit ARIMA model to time series
#'
#' @param ts A scirs2.ts object
#' @param order A vector c(p, d, q) specifying the ARIMA order
#' @param seasonal A list with order and period for seasonal ARIMA
#' @return A fitted ARIMA model object
#' @export
scirs2.arima <- function(ts, order = c(1, 1, 1), seasonal = list(order = c(0, 0, 0), period = 0)) {
  if (!inherits(ts, "scirs2.ts")) {
    stop("Object must be of class scirs2.ts")
  }
  
  if (length(order) != 3) {
    stop("Order must be a vector of length 3")
  }
  
  p <- as.integer(order[1])
  d <- as.integer(order[2])
  q <- as.integer(order[3])
  
  seasonal_p <- as.integer(seasonal$order[1])
  seasonal_d <- as.integer(seasonal$order[2])
  seasonal_q <- as.integer(seasonal$order[3])
  seasonal_period <- as.integer(seasonal$period)
  
  # Create ARIMA model
  model_ptr <- .Call("scirs_create_arima", p, d, q, seasonal_p, seasonal_d, seasonal_q, seasonal_period)
  
  if (is.null(model_ptr)) {
    stop("Failed to create ARIMA model")
  }
  
  # Fit the model
  fit_result <- .Call("scirs_fit_arima", model_ptr, ts$ptr)
  
  if (fit_result != 0) {
    .Call("scirs_free_arima", model_ptr)
    stop("Failed to fit ARIMA model")
  }
  
  # Create R object
  result <- list(
    ptr = model_ptr,
    order = order,
    seasonal = seasonal,
    fitted = TRUE,
    data = ts
  )
  
  class(result) <- "scirs2.arima"
  
  # Register finalizer
  reg.finalizer(result, function(obj) {
    if (!is.null(obj$ptr)) {
      .Call("scirs_free_arima", obj$ptr)
    }
  }, onexit = TRUE)
  
  return(result)
}

#' Automatic ARIMA model selection
#'
#' @param ts A scirs2.ts object
#' @param max.p Maximum AR order to consider (default: 5)
#' @param max.d Maximum differencing order (default: 2)
#' @param max.q Maximum MA order to consider (default: 5)
#' @param seasonal Logical indicating if seasonal ARIMA should be considered
#' @param max.seasonal.p Maximum seasonal AR order (default: 2)
#' @param max.seasonal.d Maximum seasonal differencing order (default: 1)
#' @param max.seasonal.q Maximum seasonal MA order (default: 2)
#' @param seasonal.period Seasonal period (default: frequency of ts)
#' @return A fitted ARIMA model object
#' @export
scirs2.auto_arima <- function(ts, max.p = 5, max.d = 2, max.q = 5, seasonal = TRUE,
                              max.seasonal.p = 2, max.seasonal.d = 1, max.seasonal.q = 2,
                              seasonal.period = NULL) {
  if (!inherits(ts, "scirs2.ts")) {
    stop("Object must be of class scirs2.ts")
  }
  
  if (is.null(seasonal.period)) {
    seasonal.period <- max(1, ts$frequency)
  }
  
  # Call auto ARIMA
  model_ptr <- .Call("scirs_auto_arima", 
                     ts$ptr,
                     as.integer(max.p),
                     as.integer(max.d),
                     as.integer(max.q),
                     as.integer(seasonal),
                     as.integer(max.seasonal.p),
                     as.integer(max.seasonal.d),
                     as.integer(max.seasonal.q),
                     as.integer(seasonal.period))
  
  if (is.null(model_ptr)) {
    stop("Failed to find optimal ARIMA model")
  }
  
  # Fit the selected model
  fit_result <- .Call("scirs_fit_arima", model_ptr, ts$ptr)
  
  if (fit_result != 0) {
    .Call("scirs_free_arima", model_ptr)
    stop("Failed to fit selected ARIMA model")
  }
  
  # Create R object
  result <- list(
    ptr = model_ptr,
    fitted = TRUE,
    data = ts,
    auto_selected = TRUE
  )
  
  class(result) <- "scirs2.arima"
  
  # Register finalizer
  reg.finalizer(result, function(obj) {
    if (!is.null(obj$ptr)) {
      .Call("scirs_free_arima", obj$ptr)
    }
  }, onexit = TRUE)
  
  return(result)
}

#' Generate forecasts from ARIMA model
#'
#' @param model A fitted scirs2.arima object
#' @param n.ahead Number of periods to forecast
#' @return A vector of forecasted values
#' @export
scirs2.forecast <- function(model, n.ahead = 10) {
  if (!inherits(model, "scirs2.arima")) {
    stop("Model must be of class scirs2.arima")
  }
  
  if (!model$fitted) {
    stop("Model must be fitted before forecasting")
  }
  
  # Prepare output buffer
  output <- numeric(n.ahead)
  
  result_length <- .Call("scirs_forecast_arima", 
                         model$ptr, 
                         as.integer(n.ahead),
                         output,
                         as.integer(n.ahead))
  
  if (result_length < 0) {
    stop("Error generating forecasts")
  }
  
  return(output[1:result_length])
}

#' Create anomaly detector
#'
#' @return An anomaly detector object
#' @export
scirs2.anomaly_detector <- function() {
  detector_ptr <- .Call("scirs_create_anomaly_detector")
  
  if (is.null(detector_ptr)) {
    stop("Failed to create anomaly detector")
  }
  
  result <- list(ptr = detector_ptr)
  class(result) <- "scirs2.anomaly_detector"
  
  # Register finalizer
  reg.finalizer(result, function(obj) {
    if (!is.null(obj$ptr)) {
      .Call("scirs_free_anomaly_detector", obj$ptr)
    }
  }, onexit = TRUE)
  
  return(result)
}

#' Detect anomalies using IQR method
#'
#' @param detector A scirs2.anomaly_detector object
#' @param ts A scirs2.ts object
#' @param multiplier IQR multiplier (default: 2.5)
#' @return A vector of anomaly indices
#' @export
scirs2.detect_anomalies_iqr <- function(detector, ts, multiplier = 2.5) {
  if (!inherits(detector, "scirs2.anomaly_detector")) {
    stop("Detector must be of class scirs2.anomaly_detector")
  }
  
  if (!inherits(ts, "scirs2.ts")) {
    stop("ts must be of class scirs2.ts")
  }
  
  # Prepare output buffer (assume max 10% of points could be anomalies)
  max_anomalies <- max(10, ts$length %/% 10)
  output <- integer(max_anomalies)
  
  result_count <- .Call("scirs_detect_anomalies_iqr",
                        detector$ptr,
                        ts$ptr,
                        as.double(multiplier),
                        output,
                        as.integer(max_anomalies))
  
  if (result_count < 0) {
    stop("Error detecting anomalies")
  }
  
  if (result_count == 0) {
    return(integer(0))
  }
  
  return(output[1:result_count] + 1)  # Convert to 1-based indexing for R
}

#' Detect anomalies using Z-score method
#'
#' @param detector A scirs2.anomaly_detector object
#' @param ts A scirs2.ts object
#' @param threshold Z-score threshold (default: 3.0)
#' @return A vector of anomaly indices
#' @export
scirs2.detect_anomalies_zscore <- function(detector, ts, threshold = 3.0) {
  if (!inherits(detector, "scirs2.anomaly_detector")) {
    stop("Detector must be of class scirs2.anomaly_detector")
  }
  
  if (!inherits(ts, "scirs2.ts")) {
    stop("ts must be of class scirs2.ts")
  }
  
  # Prepare output buffer
  max_anomalies <- max(10, ts$length %/% 10)
  output <- integer(max_anomalies)
  
  result_count <- .Call("scirs_detect_anomalies_zscore",
                        detector$ptr,
                        ts$ptr,
                        as.double(threshold),
                        output,
                        as.integer(max_anomalies))
  
  if (result_count < 0) {
    stop("Error detecting anomalies")
  }
  
  if (result_count == 0) {
    return(integer(0))
  }
  
  return(output[1:result_count] + 1)  # Convert to 1-based indexing for R
}

#' Create STL decomposition object
#'
#' @param period Seasonal period
#' @return An STL decomposition object
#' @export
scirs2.stl <- function(period) {
  if (period <= 0) {
    stop("Period must be positive")
  }
  
  decomp_ptr <- .Call("scirs_create_stl_decomposition", as.integer(period))
  
  if (is.null(decomp_ptr)) {
    stop("Failed to create STL decomposition")
  }
  
  result <- list(
    ptr = decomp_ptr,
    period = period
  )
  
  class(result) <- "scirs2.stl"
  
  # Register finalizer
  reg.finalizer(result, function(obj) {
    if (!is.null(obj$ptr)) {
      .Call("scirs_free_stl_decomposition", obj$ptr)
    }
  }, onexit = TRUE)
  
  return(result)
}

#' Decompose time series using STL
#'
#' @param decomposition A scirs2.stl object
#' @param ts A scirs2.ts object
#' @return A list containing trend, seasonal, and residual components
#' @export
scirs2.decompose <- function(decomposition, ts) {
  if (!inherits(decomposition, "scirs2.stl")) {
    stop("Decomposition must be of class scirs2.stl")
  }
  
  if (!inherits(ts, "scirs2.ts")) {
    stop("ts must be of class scirs2.ts")
  }
  
  # Call decomposition
  result <- .Call("scirs_decompose_stl", decomposition$ptr, ts$ptr)
  
  if (is.null(result)) {
    stop("Error performing STL decomposition")
  }
  
  # The result should contain trend, seasonal, and residual components
  # Note: This is a simplified interface - the actual implementation
  # would need to handle the C structure properly
  
  return(result)
}

#' Get version information
#'
#' @return Version string
#' @export
scirs2.version <- function() {
  return(.Call("scirs_get_version"))
}

# Package cleanup function
.onUnload <- function(libpath) {
  # Cleanup library
  if (is.loaded("scirs_cleanup")) {
    .Call("scirs_cleanup")
  }
  
  # Unload shared library
  if (is.loaded("scirs_initialize")) {
    library.dynam.unload("scirs2.series", libpath)
  }
}

# Print package startup message
.onAttach <- function(libname, pkgname) {
  packageStartupMessage("SciRS2 Time Series Analysis Package")
  packageStartupMessage("Version: ", scirs2.version())
  packageStartupMessage("For help, use: help(package='scirs2.series')")
}
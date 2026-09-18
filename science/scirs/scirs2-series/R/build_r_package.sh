#!/bin/bash
# Build script for scirs2-series R package
#
# This script compiles the Rust library with R integration features
# and prepares the R package for installation.

set -e

echo "🚀 Building SciRS2-Series R Package"
echo "=================================="

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "📁 Project root: $PROJECT_ROOT"
echo "📁 R package dir: $SCRIPT_DIR"

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "❌ Error: Rust/Cargo not found. Please install Rust: https://rustup.rs/"
    exit 1
fi

echo "✅ Rust/Cargo found: $(cargo --version)"

# Check if R is installed
if ! command -v R &> /dev/null; then
    echo "❌ Error: R not found. Please install R: https://www.r-project.org/"
    exit 1
fi

echo "✅ R found: $(R --version | head -n1)"

# Navigate to project root
cd "$PROJECT_ROOT"

echo ""
echo "🔨 Building Rust library with R features..."

# Build the Rust library with R integration features
echo "Building in release mode with R features..."
cargo build --release --features r

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to build Rust library"
    exit 1
fi

echo "✅ Rust library built successfully"

# Determine the library extension based on platform
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    LIB_EXT="so"
    LIB_PREFIX="lib"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    LIB_EXT="dylib"
    LIB_PREFIX="lib"
elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    LIB_EXT="dll"
    LIB_PREFIX=""
else
    echo "⚠️  Warning: Unknown platform $OSTYPE, assuming Linux"
    LIB_EXT="so"
    LIB_PREFIX="lib"
fi

LIB_NAME="${LIB_PREFIX}scirs2_series.${LIB_EXT}"
SOURCE_LIB="target/release/${LIB_NAME}"

echo ""
echo "📦 Preparing R package structure..."

# Create R package directories
mkdir -p "$SCRIPT_DIR/libs"
mkdir -p "$SCRIPT_DIR/man"
mkdir -p "$SCRIPT_DIR/tests"
mkdir -p "$SCRIPT_DIR/vignettes"

# Copy the compiled library to R package libs directory
if [ -f "$SOURCE_LIB" ]; then
    cp "$SOURCE_LIB" "$SCRIPT_DIR/libs/"
    echo "✅ Library copied to R package: $LIB_NAME"
else
    echo "❌ Error: Compiled library not found at $SOURCE_LIB"
    echo "Expected library patterns:"
    echo "  - target/release/libscirs2_series.so (Linux)"
    echo "  - target/release/libscirs2_series.dylib (macOS)"
    echo "  - target/release/scirs2_series.dll (Windows)"
    exit 1
fi

# Create a simple test file
cat > "$SCRIPT_DIR/tests/test_basic.R" << 'EOF'
# Basic tests for scirs2-series R package
library(testthat)

test_that("Package loads correctly", {
  expect_true(file.exists("../scirs2_series.R"))
})

test_that("Version function works", {
  source("../scirs2_series.R")
  version <- scirs2.version()
  expect_type(version, "character")
  expect_true(nchar(version) > 0)
})
EOF

# Create a simple manual page template
cat > "$SCRIPT_DIR/man/scirs2.series-package.Rd" << 'EOF'
\name{scirs2.series-package}
\alias{scirs2.series-package}
\docType{package}
\title{
Time Series Analysis with SciRS2
}
\description{
Comprehensive time series analysis package powered by Rust.
Provides high-performance implementations of ARIMA modeling, 
anomaly detection, time series decomposition, and forecasting.
}
\details{
The DESCRIPTION file:
\packageDESCRIPTION{scirs2.series}
\packageIndices{scirs2.series}

Main functions:
\itemize{
  \item \code{\link{scirs2.ts}} - Create time series objects
  \item \code{\link{scirs2.arima}} - Fit ARIMA models
  \item \code{\link{scirs2.auto_arima}} - Automatic ARIMA model selection
  \item \code{\link{scirs2.forecast}} - Generate forecasts
  \item \code{\link{scirs2.anomaly_detector}} - Create anomaly detector
  \item \code{\link{scirs2.stl}} - STL decomposition
}
}
\author{
SciRS2 Team

Maintainer: SciRS2 Team <info@scirs2.org>
}
\references{
Project homepage: https://github.com/cool-japan/scirs
}
\keyword{package}
\keyword{ts}
\seealso{
\code{\link{ts}}, \code{\link[forecast]{forecast}}, \code{\link[stats]{arima}}
}
\examples{
# Create a time series
data <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
ts_obj <- scirs2.ts(data, frequency = 1)
print(ts_obj)

# Calculate statistics
stats <- scirs2.stats(ts_obj)
print(stats)
}
EOF

echo "✅ R package structure prepared"

echo ""
echo "🧪 Running basic tests..."

# Test if the library can be loaded in R
cd "$SCRIPT_DIR"
R --slave -e "
tryCatch({
    source('scirs2_series.R')
    cat('✅ Package loaded successfully\n')
    version <- scirs2.version()
    cat('✅ Version:', version, '\n')
    
    # Test basic functionality
    test_data <- c(1, 2, 3, 4, 5)
    ts_obj <- scirs2.ts(test_data, frequency = 1)
    cat('✅ Time series created successfully\n')
    
    stats <- scirs2.stats(ts_obj)
    cat('✅ Statistics calculated successfully\n')
    
    cat('🎉 All basic tests passed!\n')
}, error = function(e) {
    cat('❌ Error during testing:\n')
    cat(as.character(e))
    cat('\n')
    quit(status = 1)
})
"

if [ $? -ne 0 ]; then
    echo "❌ Error: Basic tests failed"
    exit 1
fi

echo ""
echo "📖 Creating documentation..."

# Generate basic documentation (requires roxygen2)
R --slave -e "
if (!require('roxygen2', quietly = TRUE)) {
    cat('⚠️  roxygen2 not installed, skipping documentation generation\n')
    cat('   Install with: install.packages(\"roxygen2\")\n')
} else {
    cat('📝 Generating documentation with roxygen2...\n')
    roxygen2::roxygenise('.')
    cat('✅ Documentation generated\n')
}
"

echo ""
echo "🎯 Build Summary"
echo "==============="
echo "✅ Rust library compiled with R features"
echo "✅ Library copied to R package libs directory"
echo "✅ R package structure created"
echo "✅ Basic tests passed"
echo "✅ Documentation prepared"

echo ""
echo "📋 Next Steps:"
echo "1. Install the package in R:"
echo "   R CMD INSTALL ."
echo ""
echo "2. Or load directly in R:"
echo "   source('scirs2_series.R')"
echo ""
echo "3. Run the examples:"
echo "   source('examples/basic_usage.R')"
echo ""
echo "4. For package development:"
echo "   install.packages(c('devtools', 'roxygen2', 'testthat'))"
echo "   devtools::install('.')"

echo ""
echo "🚀 R package build completed successfully!"
echo ""
echo "📍 Package location: $SCRIPT_DIR"
echo "📍 Library location: $SCRIPT_DIR/libs/$LIB_NAME"
echo "📍 Examples: $SCRIPT_DIR/examples/"

# Create a quick install script
cat > "$SCRIPT_DIR/install.R" << 'EOF'
# Quick installation script for scirs2-series R package
#
# Run this script from the R directory to install the package

cat("Installing scirs2-series R package...\n")

# Method 1: Install using R CMD INSTALL (requires command line)
if (Sys.which("R") != "") {
    cat("Installing using R CMD INSTALL...\n")
    system("R CMD INSTALL .")
} else {
    cat("R command not found in PATH\n")
}

# Method 2: Install using devtools (if available)
if (require("devtools", quietly = TRUE)) {
    cat("Installing using devtools...\n")
    devtools::install(".")
} else {
    cat("devtools not available. Install with: install.packages('devtools')\n")
}

# Method 3: Source directly
cat("Alternatively, you can source the package directly:\n")
cat("source('scirs2_series.R')\n")
EOF

echo "💡 Created install.R script for easy installation"

echo ""
echo "🎉 Build process completed successfully!"
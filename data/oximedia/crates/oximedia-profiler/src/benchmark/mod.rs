//! Benchmarking modules.

pub mod compare;
pub mod runner;
pub mod suite;

pub use compare::{BenchmarkComparison, ComparisonResult};
pub use runner::{BenchmarkResult, BenchmarkRunner};
pub use suite::{Benchmark, BenchmarkSuite};

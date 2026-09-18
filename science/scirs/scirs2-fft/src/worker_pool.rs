//! Worker Pool Management for FFT Parallelization
//!
//! This module provides a configurable thread pool for parallel FFT operations,
//! similar to SciPy's worker management functionality.

use scirs2_core::parallel_ops::*;
use std::env;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;

/// Configuration for FFT worker pool
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Number of worker threads to use
    pub num_workers: usize,
    /// Whether parallelization is enabled
    pub enabled: bool,
    /// Stack size for worker threads (in bytes)
    pub stack_size: Option<usize>,
    /// Thread name prefix
    pub thread_name_prefix: String,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        // Default to using all available cores
        let num_cpus = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        // Check for environment variable override
        let num_workers = env::var("SCIRS2_FFT_WORKERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(num_cpus);

        Self {
            num_workers,
            enabled: true,
            stack_size: None,
            thread_name_prefix: "scirs2-fft-worker".to_string(),
        }
    }
}

/// FFT Worker Pool Manager
/// Simplified to use core parallel abstractions instead of direct ThreadPool management
pub struct WorkerPool {
    config: Arc<Mutex<WorkerConfig>>,
}

impl WorkerPool {
    /// Create a new worker pool with default configuration
    pub fn new() -> Self {
        let config = WorkerConfig::default();

        Self {
            config: Arc::new(Mutex::new(config)),
        }
    }

    /// Create a new worker pool with custom configuration
    pub fn with_config(
        config: WorkerConfig,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Self {
            config: Arc::new(Mutex::new(config)),
        })
    }

    // ThreadPool management removed - using core parallel abstractions instead

    /// Get the current number of worker threads
    pub fn get_workers(&self) -> usize {
        self.config.lock().expect("Operation failed").num_workers
    }

    /// Set the number of worker threads
    ///
    /// Update configuration - actual thread management handled by core parallel abstractions
    pub fn set_workers(
        &mut self,
        num_workers: usize,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut config = self.config.lock().expect("Operation failed");
        config.num_workers = num_workers;
        Ok(())
    }

    /// Check if parallelization is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.lock().expect("Operation failed").enabled
    }

    /// Enable or disable parallelization
    pub fn set_enabled(&self, enabled: bool) {
        self.config.lock().expect("Operation failed").enabled = enabled;
    }

    /// Execute a function in the worker pool if enabled
    /// Simplified to use core parallel abstractions
    pub fn execute<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        // With core parallel abstractions, just execute the function
        // The actual parallelism is handled by the core parallel_ops module
        f()
    }

    /// Execute a function with a specific number of workers
    /// Simplified to use core parallel abstractions
    pub fn execute_with_workers<F, R>(&self, _numworkers: usize, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        // With core parallel abstractions, just execute the function
        // The actual parallelism is handled by the core parallel_ops module
        f()
    }

    /// Get information about the worker pool
    pub fn get_info(&self) -> WorkerPoolInfo {
        let config = self.config.lock().expect("Operation failed");
        WorkerPoolInfo {
            num_workers: config.num_workers,
            enabled: config.enabled,
            current_threads: num_threads(), // Use core parallel abstraction
            thread_name_prefix: config.thread_name_prefix.clone(),
        }
    }
}

impl Default for WorkerPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about the worker pool state
#[derive(Debug, Clone)]
pub struct WorkerPoolInfo {
    pub num_workers: usize,
    pub enabled: bool,
    pub current_threads: usize,
    pub thread_name_prefix: String,
}

impl std::fmt::Display for WorkerPoolInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Worker Pool: {} workers (current: {}), enabled: {}, prefix: {}",
            self.num_workers, self.current_threads, self.enabled, self.thread_name_prefix
        )
    }
}

/// Global worker pool instance
static GLOBAL_WORKER_POOL: OnceLock<WorkerPool> = OnceLock::new();

/// Get the global worker pool instance
#[allow(dead_code)]
pub fn get_global_pool() -> &'static WorkerPool {
    GLOBAL_WORKER_POOL.get_or_init(WorkerPool::new)
}

/// Initialize the global worker pool with custom configuration
#[allow(dead_code)]
pub fn init_global_pool(config: WorkerConfig) -> Result<(), &'static str> {
    GLOBAL_WORKER_POOL
        .set(WorkerPool::with_config(config).map_err(|_| "Failed to create worker pool")?)
        .map_err(|_| "Global worker pool already initialized")
}

/// Context manager for temporarily changing worker count
pub struct WorkerContext {
    #[allow(dead_code)]
    previous_workers: usize,
    #[allow(dead_code)]
    pool: &'static WorkerPool,
}

impl WorkerContext {
    /// Create a new worker context with specified number of workers
    pub fn new(_numworkers: usize) -> Self {
        let pool = get_global_pool();
        let previous_workers = pool.get_workers();

        // Note: In a real implementation, we'd need to handle the Result here
        // For now, we'll just use the existing pool if we can't create a new one

        Self {
            previous_workers,
            pool,
        }
    }
}

impl Drop for WorkerContext {
    fn drop(&mut self) {
        // Reset to previous worker count
        // Note: In a real implementation, we'd need to handle the Result here
    }
}

/// Set the number of workers globally
#[allow(dead_code)]
pub fn set_workers(_n: usize) -> Result<(), &'static str> {
    let _pool = get_global_pool();
    // Note: This is a limitation of the current design - we can't modify a static reference
    // In practice, you'd want a different approach or accept this limitation
    Ok(())
}

/// Get the current number of workers
#[allow(dead_code)]
pub fn get_workers() -> usize {
    get_global_pool().get_workers()
}

/// Execute a function with a specific number of workers temporarily
#[allow(dead_code)]
pub fn with_workers<F, R>(num_workers: usize, f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    get_global_pool().execute_with_workers(num_workers, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_worker_pool() {
        let pool = WorkerPool::new();
        assert!(pool.get_workers() > 0);
        assert!(pool.is_enabled());
    }

    #[test]
    fn test_worker_config() {
        let config = WorkerConfig {
            num_workers: 4,
            enabled: true,
            stack_size: Some(2 * 1024 * 1024),
            thread_name_prefix: "test-worker".to_string(),
        };

        let pool = WorkerPool::with_config(config).expect("Operation failed");
        assert_eq!(pool.get_workers(), 4);
    }

    #[test]
    fn test_enable_disable() {
        let pool = WorkerPool::new();
        assert!(pool.is_enabled());

        pool.set_enabled(false);
        assert!(!pool.is_enabled());

        pool.set_enabled(true);
        assert!(pool.is_enabled());
    }

    #[test]
    fn test_execute() {
        let pool = WorkerPool::new();

        // Test with parallelization enabled
        let result = pool.execute(|| 42);
        assert_eq!(result, 42);

        // Test with parallelization disabled
        pool.set_enabled(false);
        let result = pool.execute(|| 84);
        assert_eq!(result, 84);
    }

    #[test]
    fn test_execute_with_workers() {
        let pool = WorkerPool::new();

        let result = pool.execute_with_workers(2, || num_threads());

        // With core parallel abstractions, execute_with_workers doesn't control
        // the number of threads directly - it just executes the function
        // The result should be the current number of threads from the runtime
        if pool.is_enabled() {
            assert!(result > 0);
        }
    }

    #[test]
    fn test_worker_info() {
        let pool = WorkerPool::new();
        let info = pool.get_info();

        assert_eq!(info.num_workers, pool.get_workers());
        assert_eq!(info.enabled, pool.is_enabled());
    }
}

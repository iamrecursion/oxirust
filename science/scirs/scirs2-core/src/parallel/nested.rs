//! Nested parallelism with controlled resource usage
//!
//! This module provides support for nested parallel operations with proper
//! resource management to prevent thread explosion and maintain performance.

use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};
use crate::parallel::scheduler::{SchedulerConfigBuilder, WorkStealingScheduler};
use rayon::iter::ParallelIterator;
use rayon::prelude::*;
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

thread_local! {
    /// Thread-local nesting level
    static NESTING_LEVEL: RefCell<usize> = const { RefCell::new(0) };

    /// Thread-local parent context
    static PARENT_CONTEXT: RefCell<Option<Arc<NestedContext>>> = const { RefCell::new(None) };
}

/// Global resource manager for nested parallelism
static GLOBAL_RESOURCE_MANAGER: std::sync::OnceLock<Arc<ResourceManager>> =
    std::sync::OnceLock::new();

/// Get or create the global resource manager
#[allow(dead_code)]
fn get_resource_manager() -> Arc<ResourceManager> {
    GLOBAL_RESOURCE_MANAGER
        .get_or_init(|| Arc::new(ResourceManager::new()))
        .clone()
}

/// Resource limits for nested parallelism
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    /// Maximum total number of threads across all nesting levels
    pub max_total_threads: usize,
    /// Maximum nesting depth
    pub max_nesting_depth: usize,
    /// Thread limit per nesting level
    pub threads_per_level: Vec<usize>,
    /// Memory limit in bytes
    pub max_memory_bytes: usize,
    /// CPU usage limit (0.0 to 1.0)
    pub max_cpu_usage: f64,
    /// Whether to enable thread pooling
    pub enable_thread_pooling: bool,
    /// Whether to enable work stealing across levels
    pub enable_cross_level_stealing: bool,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        let num_cpus = num_cpus::get();
        Self {
            max_total_threads: num_cpus * 2,
            max_nesting_depth: 3,
            threads_per_level: vec![num_cpus, num_cpus / 2, 1],
            max_memory_bytes: (4u64 * 1024 * 1024 * 1024) as usize, // 4GB
            max_cpu_usage: 0.9,
            enable_thread_pooling: true,
            enable_cross_level_stealing: false,
        }
    }
}

/// Context for nested parallel execution
pub struct NestedContext {
    /// Current nesting level (0 = top level)
    level: usize,
    /// Parent context (if any)
    parent: Option<Arc<NestedContext>>,
    /// Resource limits for this context
    limits: ResourceLimits,
    /// Number of active threads at this level
    active_threads: AtomicUsize,
    /// Scheduler for this level
    scheduler: Option<Arc<Mutex<WorkStealingScheduler>>>,
}

impl NestedContext {
    /// Create a new top-level context
    pub fn new(limits: ResourceLimits) -> Self {
        Self {
            level: 0,
            parent: None,
            limits,
            active_threads: AtomicUsize::new(0),
            scheduler: None,
        }
    }

    /// Create a child context
    pub fn create_child(&self) -> CoreResult<Arc<NestedContext>> {
        if self.level >= self.limits.max_nesting_depth {
            return Err(CoreError::ConfigError(
                ErrorContext::new(format!(
                    "Maximum nesting depth {} exceeded",
                    self.limits.max_nesting_depth
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        let child = NestedContext {
            level: self.level + 1,
            parent: Some(Arc::new(self.clone())),
            limits: self.limits.clone(),
            active_threads: AtomicUsize::new(0),
            scheduler: None,
        };

        Ok(Arc::new(child))
    }

    /// Get the maximum number of threads allowed at this level
    pub fn max_threads_at_level(&self) -> usize {
        if self.level < self.limits.threads_per_level.len() {
            self.limits.threads_per_level[self.level]
        } else {
            1 // Default to single thread for deep nesting
        }
    }

    /// Try to acquire threads for parallel execution
    pub fn try_acquire_threads(&self, requested: usize) -> usize {
        let max_at_level = self.max_threads_at_level();
        let resource_manager = get_resource_manager();

        // Check global limit
        let available_global = resource_manager.try_acquire_threads(requested);

        // Check level limit
        let current = self.active_threads.load(Ordering::Relaxed);
        let available_at_level = max_at_level.saturating_sub(current);

        // Take minimum of all constraints
        let granted = requested.min(available_global).min(available_at_level);

        if granted > 0 {
            self.active_threads.fetch_add(granted, Ordering::Relaxed);
        } else {
            // Return any globally acquired threads if we can't use them
            resource_manager.release_threads(available_global);
        }

        granted
    }

    /// Release acquired threads
    pub fn release_threads(&self, count: usize) {
        self.active_threads.fetch_sub(count, Ordering::Relaxed);
        get_resource_manager().release_threads(count);
    }

    /// Get or create scheduler for this level
    pub fn get_scheduler(&self) -> CoreResult<Arc<Mutex<WorkStealingScheduler>>> {
        if let Some(ref scheduler) = self.scheduler {
            return Ok(scheduler.clone());
        }

        // Create scheduler with appropriate configuration for this level
        let config = SchedulerConfigBuilder::new()
            .workers(self.max_threads_at_level())
            .adaptive(true)
            .enable_stealing_heuristics(true)
            .enable_priorities(true)
            .build();

        let scheduler = WorkStealingScheduler::new(config);
        Ok(Arc::new(Mutex::new(scheduler)))
    }
}

impl Clone for NestedContext {
    fn clone(&self) -> Self {
        Self {
            level: self.level,
            parent: self.parent.clone(),
            limits: self.limits.clone(),
            active_threads: AtomicUsize::new(self.active_threads.load(Ordering::Relaxed)),
            scheduler: self.scheduler.clone(),
        }
    }
}

/// Global resource manager for tracking system-wide resource usage
pub struct ResourceManager {
    /// Total threads in use across all levels
    total_threads: AtomicUsize,
    /// Memory usage tracking
    memory_used: AtomicUsize,
    /// CPU usage tracking
    cpu_usage: RwLock<f64>,
    /// Active contexts by level
    active_contexts: RwLock<Vec<usize>>,
}

impl Default for ResourceManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceManager {
    /// Create a new resource manager
    pub fn new() -> Self {
        let max_levels = 10;
        Self {
            total_threads: AtomicUsize::new(0),
            memory_used: AtomicUsize::new(0),
            cpu_usage: RwLock::new(0.0),
            active_contexts: RwLock::new(vec![0; max_levels]),
        }
    }

    /// Try to acquire threads from the global pool
    pub fn try_acquire_threads(&self, requested: usize) -> usize {
        let mut acquired = 0;

        // Simple atomic loop to acquire threads
        for _ in 0..requested {
            let current = self.total_threads.load(Ordering::Relaxed);
            let max_threads = num_cpus::get() * 2; // Global limit

            if current < max_threads {
                if self
                    .total_threads
                    .compare_exchange(current, current + 1, Ordering::Acquire, Ordering::Relaxed)
                    .is_ok()
                {
                    acquired += 1;
                } else {
                    // Another thread modified the count, retry
                    continue;
                }
            } else {
                break;
            }
        }

        acquired
    }

    /// Release threads back to the global pool
    pub fn release_threads(&self, count: usize) {
        self.total_threads.fetch_sub(count, Ordering::Release);
    }

    /// Update memory usage
    pub fn update_memory_usage(&self, bytes: isize) {
        if bytes > 0 {
            self.memory_used
                .fetch_add(bytes as usize, Ordering::Relaxed);
        } else {
            self.memory_used
                .fetch_sub((-bytes) as usize, Ordering::Relaxed);
        }
    }

    /// Get current resource usage statistics
    pub fn get_usage_stats(&self) -> ResourceUsageStats {
        ResourceUsageStats {
            total_threads: self.total_threads.load(Ordering::Relaxed),
            memory_bytes: self.memory_used.load(Ordering::Relaxed),
            cpu_usage: *self.cpu_usage.read().expect("Operation failed"),
            active_contexts_per_level: self
                .active_contexts
                .read()
                .expect("Operation failed")
                .clone(),
        }
    }
}

/// Resource usage statistics
#[derive(Debug, Clone)]
pub struct ResourceUsageStats {
    /// Total threads in use
    pub total_threads: usize,
    /// Memory usage in bytes
    pub memory_bytes: usize,
    /// CPU usage (0.0 to 1.0)
    pub cpu_usage: f64,
    /// Number of active contexts at each nesting level
    pub active_contexts_per_level: Vec<usize>,
}

/// Nested parallel execution scope
pub struct NestedScope<'a> {
    context: Arc<NestedContext>,
    acquired_threads: usize,
    phantom: std::marker::PhantomData<&'a ()>,
}

impl NestedScope<'_> {
    /// Execute a function in parallel within this scope
    pub fn execute<F, R>(&self, f: F) -> CoreResult<R>
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        // Set thread-local context
        PARENT_CONTEXT.with(|ctx| {
            *ctx.borrow_mut() = Some(self.context.clone());
        });

        // Set nesting level
        NESTING_LEVEL.with(|level| {
            *level.borrow_mut() = self.context.level;
        });

        // Execute the function
        let result = f();

        // Clear thread-local state
        PARENT_CONTEXT.with(|ctx| {
            *ctx.borrow_mut() = None;
        });

        Ok(result)
    }

    /// Execute parallel iterator within this scope
    pub fn par_iter<I, F, R>(&self, items: I, f: F) -> CoreResult<Vec<R>>
    where
        I: IntoParallelIterator,
        I::Item: Send,
        F: Fn(I::Item) -> R + Send + Sync,
        R: Send,
    {
        // Convert to parallel iterator once
        let results: Vec<R> = items.into_par_iter().map(f).collect();

        Ok(results)
    }
}

impl Drop for NestedScope<'_> {
    fn drop(&mut self) {
        // Release acquired threads
        if self.acquired_threads > 0 {
            self.context.release_threads(self.acquired_threads);
        }
    }
}

/// Execute a function with nested parallelism support
#[allow(dead_code)]
pub fn nested_scope<F, R>(f: F) -> CoreResult<R>
where
    F: FnOnce(&NestedScope) -> CoreResult<R>,
{
    nested_scope_with_limits(ResourceLimits::default(), f)
}

/// Execute a function with nested parallelism support and custom limits
#[allow(dead_code)]
pub fn nested_scope_with_limits<F, R>(limits: ResourceLimits, f: F) -> CoreResult<R>
where
    F: FnOnce(&NestedScope) -> CoreResult<R>,
{
    // Check if we're already in a nested context
    let context = match PARENT_CONTEXT
        .with(|ctx| ctx.borrow().as_ref().map(|parent| parent.create_child()))
    {
        Some(child_result) => child_result?,
        None => {
            // No parent context, create a new root context with level 0
            Arc::new(NestedContext::new(limits.clone()))
        }
    };

    // Try to acquire threads
    let requested_threads = context.max_threads_at_level();
    let acquired_threads = context.try_acquire_threads(requested_threads);

    // Create scope
    let scope = NestedScope {
        context: context.clone(),
        acquired_threads,
        phantom: std::marker::PhantomData,
    };

    // Set the nesting level for the current thread
    let old_level = NESTING_LEVEL.with(|level| {
        let old = *level.borrow();
        *level.borrow_mut() = context.level;
        old
    });

    // Set parent context
    let old_context = PARENT_CONTEXT.with(|ctx| ctx.borrow_mut().replace(context));

    // Execute function
    let result = f(&scope);

    // Restore previous nesting level
    NESTING_LEVEL.with(|level| {
        *level.borrow_mut() = old_level;
    });

    // Restore previous context
    PARENT_CONTEXT.with(|ctx| {
        *ctx.borrow_mut() = old_context;
    });

    result
}

/// Get the current nesting level
#[allow(dead_code)]
pub fn current_nesting_level() -> usize {
    NESTING_LEVEL.with(|level| *level.borrow())
}

/// Check if nested parallelism is allowed at the current level
#[allow(dead_code)]
pub fn is_nested_parallelism_allowed() -> bool {
    PARENT_CONTEXT.with(|ctx| {
        if let Some(ref context) = *ctx.borrow() {
            context.level < context.limits.max_nesting_depth
        } else {
            true // Top level always allowed
        }
    })
}

/// Adaptive parallel execution based on nesting level
#[allow(dead_code)]
pub fn adaptive_par_for_each<T, F>(data: Vec<T>, f: F) -> CoreResult<()>
where
    T: Send,
    F: Fn(T) + Send + Sync,
{
    if is_nested_parallelism_allowed() {
        data.into_par_iter().for_each(f);
    } else {
        // Fall back to sequential at deep nesting levels
        data.into_iter().for_each(f);
    }
    Ok(())
}

/// Adaptive parallel map based on nesting level
#[allow(dead_code)]
pub fn adaptive_par_map<T, F, R>(data: Vec<T>, f: F) -> CoreResult<Vec<R>>
where
    T: Send,
    F: Fn(T) -> R + Send + Sync,
    R: Send,
{
    if is_nested_parallelism_allowed() {
        Ok(data.into_par_iter().map(f).collect())
    } else {
        // Fall back to sequential at deep nesting levels
        Ok(data.into_iter().map(f).collect())
    }
}

/// Policy for handling nested parallelism
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NestedPolicy {
    /// Allow nested parallelism with resource limits
    Allow,
    /// Convert to sequential execution at nested levels
    Sequential,
    /// Distribute work to parent level scheduler
    Delegate,
    /// Throw error if nested parallelism is attempted
    Deny,
}

/// Configuration for nested parallel execution
#[derive(Debug, Clone)]
pub struct NestedConfig {
    /// Policy for handling nested parallelism
    pub policy: NestedPolicy,
    /// Resource limits
    pub limits: ResourceLimits,
    /// Whether to track resource usage
    pub track_usage: bool,
    /// Whether to enable adaptive scheduling
    pub adaptive_scheduling: bool,
}

impl Default for NestedConfig {
    fn default() -> Self {
        Self {
            policy: NestedPolicy::Allow,
            limits: ResourceLimits::default(),
            track_usage: true,
            adaptive_scheduling: true,
        }
    }
}

/// Execute with specific nested parallelism policy
#[allow(dead_code)]
pub fn with_nested_policy<F, R>(config: NestedConfig, f: F) -> CoreResult<R>
where
    F: FnOnce() -> CoreResult<R>,
{
    match config.policy {
        NestedPolicy::Allow => nested_scope_with_limits(config.limits, |_scope| f()),
        NestedPolicy::Sequential => {
            // Force sequential execution
            NESTING_LEVEL.with(|level| {
                *level.borrow_mut() = usize::MAX;
            });
            let result = f();
            NESTING_LEVEL.with(|level| {
                *level.borrow_mut() = 0;
            });
            result
        }
        NestedPolicy::Delegate => {
            // Delegate to parent scheduler if available
            // For now, just execute directly as delegation is complex
            f()
        }
        NestedPolicy::Deny => {
            // Check if we're inside any nested scope (even level 0)
            let is_nested = PARENT_CONTEXT.with(|ctx| ctx.borrow().is_some());
            if is_nested {
                Err(CoreError::ConfigError(
                    ErrorContext::new("Nested parallelism not allowed".to_string())
                        .with_location(ErrorLocation::new(file!(), line!())),
                ))
            } else {
                f()
            }
        }
    }
}

/// Get parent level scheduler if available
#[allow(dead_code)]
fn get_parent_scheduler() -> Option<Arc<Mutex<WorkStealingScheduler>>> {
    PARENT_CONTEXT.with(|ctx| {
        ctx.borrow()
            .as_ref()
            .and_then(|context| context.scheduler.clone())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_nested_execution() {
        let result = nested_scope(|scope| {
            let data: Vec<i32> = (0..100).collect();
            scope.par_iter(data, |x| x * 2)
        })
        .expect("Operation failed");

        assert_eq!(result.len(), 100);
        assert_eq!(result[0], 0);
        assert_eq!(result[50], 100);
    }

    #[test]
    fn test_nesting_levels() {
        nested_scope(|outer_scope| {
            assert_eq!(current_nesting_level(), 0);

            outer_scope.execute(|| {
                nested_scope(|inner_scope| {
                    assert_eq!(current_nesting_level(), 1);

                    inner_scope.execute(|| {
                        nested_scope(|_deepest_scope| {
                            assert_eq!(current_nesting_level(), 2);
                            Ok(())
                        })
                        .expect("Operation failed")
                    })
                })
                .expect("Operation failed")
            })
        })
        .expect("Operation failed");
    }

    #[test]
    fn test_resource_limits() {
        let limits = ResourceLimits {
            max_total_threads: 4,
            max_nesting_depth: 2,
            threads_per_level: vec![2, 1],
            ..Default::default()
        };

        let result = nested_scope_with_limits(limits, |scope| {
            let context = &scope.context;
            assert!(context.max_threads_at_level() <= 2);
            Ok(42)
        });

        assert_eq!(result.expect("Operation failed"), 42);
    }

    #[test]
    fn test_sequential_policy() {
        let config = NestedConfig {
            policy: NestedPolicy::Sequential,
            ..Default::default()
        };

        let result = with_nested_policy(config, || {
            // This should run sequentially even if we try parallel
            let data: Vec<i32> = (0..10).collect();
            let sum: i32 = data.into_par_iter().sum();
            Ok(sum)
        });

        assert_eq!(result.expect("Operation failed"), 45);
    }

    #[test]
    fn test_deny_policy() {
        let config = NestedConfig {
            policy: NestedPolicy::Deny,
            ..Default::default()
        };

        // Top level should work
        let result = with_nested_policy(config.clone(), || Ok(1));
        assert!(result.is_ok());

        // Nested should fail - first establish a nested context, then try to use deny policy
        let result = nested_scope(|_scope| {
            // Now we're at nesting level 1
            // This should fail because deny policy forbids nested parallelism
            with_nested_policy(config, || Ok(2))
        });

        assert!(result.is_err());
    }
}

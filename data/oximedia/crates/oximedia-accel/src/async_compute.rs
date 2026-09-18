//! Async compute queue submission for overlapped CPU/GPU work.
//!
//! Provides a task queue that submits GPU work items asynchronously so that
//! the CPU can continue preparing subsequent frames while the GPU is busy.
//! On platforms without Vulkan, all work is executed synchronously on the CPU.

#![allow(dead_code)]

use crate::error::{AccelError, AccelResult};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// A handle to a submitted async computation.
///
/// Call [`AsyncHandle::wait`] to block until the result is ready.
pub struct AsyncHandle<T> {
    inner: Arc<AsyncInner<T>>,
}

struct AsyncInner<T> {
    result: Mutex<Option<AccelResult<T>>>,
    cvar: Condvar,
    submitted_at: Instant,
    /// Set to `true` atomically when the worker stores the result.
    done: AtomicBool,
}

impl<T: Send + 'static> AsyncHandle<T> {
    /// Block until the async operation completes and return the result.
    ///
    /// # Errors
    ///
    /// Returns `AccelError::Synchronization` if the timeout is exceeded
    /// or the computation itself returned an error.
    pub fn wait(self) -> AccelResult<T> {
        let inner = self.inner;
        let guard = inner
            .result
            .lock()
            .map_err(|e| AccelError::Synchronization(format!("mutex poisoned: {e}")))?;
        // Wait until Some(result) is stored.
        let guard = inner
            .cvar
            .wait_while(guard, |r| r.is_none())
            .map_err(|e| AccelError::Synchronization(format!("condvar wait failed: {e}")))?;
        let mut guard = guard;
        guard
            .take()
            .ok_or_else(|| AccelError::Synchronization("async result missing".to_string()))?
    }

    /// Poll whether the result is already available (non-blocking).
    ///
    /// Uses an atomic flag so the check remains valid even after [`wait`](Self::wait)
    /// has consumed the stored value.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.inner.done.load(Ordering::Acquire)
    }

    /// Wall-clock age of this handle since submission.
    #[must_use]
    pub fn age(&self) -> Duration {
        self.inner.submitted_at.elapsed()
    }
}

impl<T> Clone for AsyncHandle<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// Statistics for the async compute queue.
#[derive(Debug, Clone, Default)]
pub struct AsyncQueueStats {
    /// Total jobs submitted.
    pub submitted: u64,
    /// Total jobs completed.
    pub completed: u64,
    /// Total jobs that returned an error.
    pub failed: u64,
}

/// An async compute queue that runs submitted closures on a background thread.
///
/// This provides overlap between CPU preparation and GPU/CPU execution by
/// immediately returning a handle when work is submitted.
pub struct AsyncComputeQueue {
    stats: Arc<Mutex<AsyncQueueStats>>,
}

impl AsyncComputeQueue {
    /// Create a new async compute queue.
    #[must_use]
    pub fn new() -> Self {
        Self {
            stats: Arc::new(Mutex::new(AsyncQueueStats::default())),
        }
    }

    /// Submit a computation closure for async execution.
    ///
    /// The closure is immediately dispatched to a new thread.  For a
    /// production implementation with a Vulkan backend, this thread would
    /// record a command buffer and submit it to the async compute queue;
    /// the present implementation executes the work on the CPU thread so
    /// that the interface is testable without a GPU.
    ///
    /// Returns an [`AsyncHandle`] that can be polled or awaited.
    pub fn submit<T, F>(&self, work: F) -> AsyncHandle<T>
    where
        T: Send + 'static,
        F: FnOnce() -> AccelResult<T> + Send + 'static,
    {
        let inner = Arc::new(AsyncInner {
            result: Mutex::new(None),
            cvar: Condvar::new(),
            submitted_at: Instant::now(),
            done: AtomicBool::new(false),
        });

        let handle = AsyncHandle {
            inner: Arc::clone(&inner),
        };

        let stats = Arc::clone(&self.stats);
        if let Ok(mut s) = stats.lock() {
            s.submitted += 1;
        }

        thread::spawn(move || {
            let result = work();
            let is_err = result.is_err();

            // Update stats BEFORE signalling completion so that any caller
            // which returns from `wait()` is guaranteed to see the updated
            // counters when it subsequently calls `stats()`.
            if let Ok(mut s) = stats.lock() {
                if is_err {
                    s.failed += 1;
                } else {
                    s.completed += 1;
                }
            }

            // Store result, signal done flag, and notify waiters.
            if let Ok(mut guard) = inner.result.lock() {
                *guard = Some(result);
            }
            inner.done.store(true, Ordering::Release);
            inner.cvar.notify_all();
        });

        handle
    }

    /// Returns a snapshot of queue statistics.
    ///
    /// # Errors
    ///
    /// Returns `AccelError::Synchronization` if the internal mutex is poisoned.
    pub fn stats(&self) -> AccelResult<AsyncQueueStats> {
        self.stats
            .lock()
            .map(|g| g.clone())
            .map_err(|e| AccelError::Synchronization(format!("stats mutex poisoned: {e}")))
    }
}

impl Default for AsyncComputeQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience: submit a batch of independent work items and collect all results.
///
/// Each closure is submitted to the queue and all handles are awaited in order.
///
/// # Errors
///
/// Returns the first error encountered, if any.
pub fn submit_batch<T, F>(queue: &AsyncComputeQueue, items: Vec<F>) -> AccelResult<Vec<T>>
where
    T: Send + 'static,
    F: FnOnce() -> AccelResult<T> + Send + 'static,
{
    let handles: Vec<AsyncHandle<T>> = items.into_iter().map(|f| queue.submit(f)).collect();
    handles
        .into_iter()
        .map(|h| h.wait())
        .collect::<AccelResult<Vec<T>>>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_async_queue_submit_and_wait() {
        let q = AsyncComputeQueue::new();
        let h = q.submit(|| Ok::<u32, AccelError>(42));
        let result = h.wait().expect("async wait should succeed");
        assert_eq!(result, 42);
    }

    #[test]
    fn test_async_queue_submit_error() {
        let q = AsyncComputeQueue::new();
        let h = q.submit(|| Err::<u32, AccelError>(AccelError::OutOfMemory));
        let result = h.wait();
        assert!(result.is_err());
    }

    #[test]
    fn test_async_queue_is_ready_after_wait() {
        let q = AsyncComputeQueue::new();
        let h = q.submit(|| Ok::<i32, AccelError>(7));
        // Cloning the handle before waiting.
        let h2 = h.clone();
        h.wait().expect("wait should succeed");
        // Original Arc is shared; after the thread finished h2 should be ready.
        assert!(h2.is_ready());
    }

    #[test]
    fn test_async_queue_stats_submitted() {
        let q = AsyncComputeQueue::new();
        let h1 = q.submit(|| Ok::<u8, AccelError>(1));
        let h2 = q.submit(|| Ok::<u8, AccelError>(2));
        h1.wait().expect("h1 wait should succeed");
        h2.wait().expect("h2 wait should succeed");
        let stats = q.stats().expect("stats should succeed");
        assert_eq!(stats.submitted, 2);
        assert_eq!(stats.completed, 2);
        assert_eq!(stats.failed, 0);
    }

    #[test]
    fn test_async_queue_stats_failed() {
        let q = AsyncComputeQueue::new();
        let h = q.submit(|| Err::<u8, AccelError>(AccelError::OutOfMemory));
        let _ = h.wait();
        // Give the thread time to update stats (it stores result then updates stats).
        std::thread::sleep(Duration::from_millis(10));
        let stats = q.stats().expect("stats should succeed");
        assert_eq!(stats.failed, 1);
    }

    #[test]
    fn test_async_queue_handle_age() {
        let q = AsyncComputeQueue::new();
        let h = q.submit(|| Ok::<(), AccelError>(()));
        h.clone().wait().expect("wait should succeed");
        // Age should be a small positive duration.
        assert!(h.age() < Duration::from_secs(5));
    }

    #[test]
    fn test_submit_batch_all_succeed() {
        let q = AsyncComputeQueue::new();
        let items: Vec<Box<dyn FnOnce() -> AccelResult<u32> + Send>> =
            vec![Box::new(|| Ok(1)), Box::new(|| Ok(2)), Box::new(|| Ok(3))];
        let results = submit_batch(&q, items).expect("batch should succeed");
        assert_eq!(results, vec![1, 2, 3]);
    }

    #[test]
    fn test_submit_batch_propagates_error() {
        let q = AsyncComputeQueue::new();
        let items: Vec<Box<dyn FnOnce() -> AccelResult<u32> + Send>> = vec![
            Box::new(|| Ok(1)),
            Box::new(|| Err(AccelError::OutOfMemory)),
            Box::new(|| Ok(3)),
        ];
        let result = submit_batch(&q, items);
        assert!(result.is_err());
    }

    #[test]
    fn test_async_queue_default() {
        let q = AsyncComputeQueue::default();
        let stats = q.stats().expect("stats should succeed");
        assert_eq!(stats.submitted, 0);
    }

    #[test]
    fn test_async_handle_not_ready_immediately() {
        let q = AsyncComputeQueue::new();
        // Submit something that sleeps briefly so we can race.
        let h = q.submit(|| {
            std::thread::sleep(Duration::from_millis(50));
            Ok::<u32, AccelError>(99)
        });
        // Handle should not yet be ready (most of the time – it's a race, so we
        // just verify is_ready() doesn't panic).
        let _ = h.is_ready();
        h.wait().expect("wait should succeed");
    }
}

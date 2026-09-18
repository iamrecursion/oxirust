//! Chase-Lev work-stealing work queue for multi-threaded media pipelines.
//!
//! This module exposes [`WorkQueue`] — a thread-safe, work-stealing task
//! distributor backed by [`crossbeam_deque`].  Each call to [`WorkQueue::new`]
//! creates one injector (global push point) and `workers` local steal handles.
//!
//! # Design
//!
//! ```text
//!   Producer         Injector          Worker 0 deque
//!   ────────►  push  ──────► steal ───►  pop / steal
//!                                         │
//!                           Worker 1 ─────┘  (steals from Worker 0)
//! ```
//!
//! Tasks pushed via [`WorkQueue::push`] land in the global injector queue.
//! Worker threads call [`WorkQueue::steal`] which first drains the injector,
//! then falls back to stealing from sibling workers.  [`WorkQueue::len`]
//! returns an approximate total count.
//!
//! # Examples
//!
//! ```
//! use oximedia_core::work_queue_ws::WorkQueue;
//! use std::sync::Arc;
//! use std::sync::atomic::{AtomicUsize, Ordering};
//!
//! let wq: WorkQueue<u32> = WorkQueue::new(2);
//! for i in 0..10_u32 {
//!     wq.push(i);
//! }
//! // Any thread can steal.
//! let _item = wq.steal();
//! assert!(wq.len() <= 10);
//! ```

use crossbeam_deque::{Injector, Steal, Stealer, Worker};
use std::sync::{Arc, Mutex};

// ─────────────────────────────────────────────────────────────────────────────
// WorkQueue
// ─────────────────────────────────────────────────────────────────────────────

/// Inner shared state of a [`WorkQueue`].
struct Inner<T> {
    /// The global injection point; any thread may push here.
    injector: Injector<T>,
    /// One stealer handle per logical worker (cloned from the worker deques).
    stealers: Vec<Stealer<T>>,
    /// One worker deque per logical worker (protected behind a mutex so that
    /// steal() can borrow a deque without requiring the caller to own a slot).
    workers: Vec<Mutex<Worker<T>>>,
    /// Approximate item count (incremented on push, decremented on steal).
    len: std::sync::atomic::AtomicIsize,
}

/// A work-stealing work queue for distributing tasks across multiple workers.
///
/// `WorkQueue<T>` is `Clone` — all clones share the same underlying state,
/// so tasks pushed from one clone are visible to all others.
///
/// # Thread safety
///
/// `WorkQueue<T>` is `Send + Sync` when `T: Send`.  Multiple threads may
/// call [`push`](WorkQueue::push) and [`steal`](WorkQueue::steal)
/// concurrently without external synchronisation.
///
/// # Examples
///
/// ```
/// use oximedia_core::work_queue_ws::WorkQueue;
/// use std::thread;
/// use std::sync::Arc;
/// use std::sync::atomic::{AtomicUsize, Ordering};
///
/// let wq = WorkQueue::<u32>::new(4);
/// for i in 0..100_u32 {
///     wq.push(i);
/// }
///
/// let total = Arc::new(AtomicUsize::new(0));
/// let mut handles = Vec::new();
///
/// for _ in 0..4 {
///     let wq2 = wq.clone();
///     let count = Arc::clone(&total);
///     handles.push(thread::spawn(move || {
///         while let Some(_task) = wq2.steal() {
///             count.fetch_add(1, Ordering::Relaxed);
///         }
///     }));
/// }
/// for h in handles { h.join().expect("thread panicked"); }
/// assert_eq!(total.load(Ordering::Relaxed), 100);
/// ```
#[derive(Clone)]
pub struct WorkQueue<T: Send + 'static> {
    inner: Arc<Inner<T>>,
}

impl<T: Send + 'static> WorkQueue<T> {
    /// Creates a new `WorkQueue` with `workers` local deques.
    ///
    /// `workers` controls the number of distinct steal handles.  A value of
    /// `0` is clamped to `1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::work_queue_ws::WorkQueue;
    ///
    /// let wq = WorkQueue::<i32>::new(4);
    /// assert_eq!(wq.len(), 0);
    /// ```
    #[must_use]
    pub fn new(workers: usize) -> Self {
        let num = workers.max(1);
        let injector = Injector::new();
        let mut worker_deques = Vec::with_capacity(num);
        let mut stealers = Vec::with_capacity(num);

        for _ in 0..num {
            let w: Worker<T> = Worker::new_fifo();
            stealers.push(w.stealer());
            worker_deques.push(Mutex::new(w));
        }

        Self {
            inner: Arc::new(Inner {
                injector,
                stealers,
                workers: worker_deques,
                len: std::sync::atomic::AtomicIsize::new(0),
            }),
        }
    }

    /// Pushes a task into the global injection queue.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::work_queue_ws::WorkQueue;
    ///
    /// let wq = WorkQueue::<u32>::new(2);
    /// wq.push(42_u32);
    /// assert_eq!(wq.len(), 1);
    /// ```
    pub fn push(&self, task: T) {
        self.inner.injector.push(task);
        self.inner
            .len
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Attempts to steal a task from any available source.
    ///
    /// The implementation first drains the global injector into a local worker
    /// deque (slot 0), then tries to pop from each worker in round-robin order,
    /// retrying on contention.
    ///
    /// Returns `None` when all queues appear empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::work_queue_ws::WorkQueue;
    ///
    /// let wq = WorkQueue::<u32>::new(2);
    /// wq.push(1_u32);
    /// wq.push(2_u32);
    /// let t1 = wq.steal();
    /// let t2 = wq.steal();
    /// assert!(t1.is_some());
    /// assert!(t2.is_some());
    /// ```
    pub fn steal(&self) -> Option<T> {
        // Try draining the injector into worker 0 first.
        if let Ok(guard) = self.inner.workers[0].lock() {
            loop {
                match self.inner.injector.steal_batch_and_pop(&guard) {
                    Steal::Success(v) => {
                        self.inner
                            .len
                            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                        return Some(v);
                    }
                    Steal::Retry => continue,
                    Steal::Empty => break,
                }
            }
        }

        // Try popping from each worker deque in turn.
        for w_mutex in &self.inner.workers {
            if let Ok(guard) = w_mutex.lock() {
                if let Some(item) = guard.pop() {
                    self.inner
                        .len
                        .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                    return Some(item);
                }
            }
        }

        // Fall back to stealing via stealer handles (cross-thread steal).
        for stealer in &self.inner.stealers {
            loop {
                match stealer.steal() {
                    Steal::Success(v) => {
                        self.inner
                            .len
                            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                        return Some(v);
                    }
                    Steal::Retry => continue,
                    Steal::Empty => break,
                }
            }
        }

        None
    }

    /// Returns the approximate number of tasks currently in the queue.
    ///
    /// This value may be slightly stale due to concurrent operations.  It
    /// saturates at zero rather than going negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::work_queue_ws::WorkQueue;
    ///
    /// let wq = WorkQueue::<u32>::new(2);
    /// wq.push(1_u32);
    /// wq.push(2_u32);
    /// assert_eq!(wq.len(), 2);
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        let v = self.inner.len.load(std::sync::atomic::Ordering::Relaxed);
        v.max(0) as usize
    }

    /// Returns `true` if the queue appears empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    // 1. Basic push and steal.
    #[test]
    fn push_and_steal_basic() {
        let wq = WorkQueue::<u32>::new(1);
        wq.push(10_u32);
        wq.push(20_u32);
        let a = wq.steal();
        let b = wq.steal();
        assert!(a.is_some());
        assert!(b.is_some());
        assert_eq!(wq.len(), 0);
    }

    // 2. Steal from empty returns None.
    #[test]
    fn steal_empty_returns_none() {
        let wq = WorkQueue::<u32>::new(2);
        assert!(wq.steal().is_none());
    }

    // 3. len tracks count.
    #[test]
    fn len_tracks_count() {
        let wq = WorkQueue::<u32>::new(2);
        assert_eq!(wq.len(), 0);
        wq.push(1_u32);
        assert_eq!(wq.len(), 1);
        wq.push(2_u32);
        assert_eq!(wq.len(), 2);
        wq.steal();
        assert_eq!(wq.len(), 1);
    }

    // 4. is_empty.
    #[test]
    fn is_empty_basic() {
        let wq = WorkQueue::<u32>::new(2);
        assert!(wq.is_empty());
        wq.push(1_u32);
        assert!(!wq.is_empty());
    }

    // 5. Clone shares state.
    #[test]
    fn clone_shares_state() {
        let wq = WorkQueue::<u32>::new(2);
        let wq2 = wq.clone();
        wq.push(99_u32);
        let stolen = wq2.steal();
        assert_eq!(stolen, Some(99_u32));
    }

    // 6. Multi-threaded stress test: 4 workers, 10 000 tasks.
    #[test]
    fn threaded_stress_10000_tasks() {
        const TASKS: u32 = 10_000;
        const WORKERS: usize = 4;

        let wq = WorkQueue::<u32>::new(WORKERS);
        for i in 0..TASKS {
            wq.push(i);
        }

        let stolen_count = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::with_capacity(WORKERS);

        for _ in 0..WORKERS {
            let wq_clone = wq.clone();
            let count = Arc::clone(&stolen_count);
            handles.push(thread::spawn(move || {
                let mut local = 0usize;
                // Keep trying until the queue is empty.
                let mut empty_streak = 0usize;
                loop {
                    match wq_clone.steal() {
                        Some(_) => {
                            local += 1;
                            empty_streak = 0;
                        }
                        None => {
                            empty_streak += 1;
                            // After many consecutive misses, assume queue is drained.
                            if empty_streak > 200 {
                                break;
                            }
                            std::hint::spin_loop();
                        }
                    }
                }
                count.fetch_add(local, Ordering::Relaxed);
            }));
        }

        for h in handles {
            h.join().expect("worker thread panicked");
        }

        let total = stolen_count.load(Ordering::Relaxed);
        assert_eq!(
            total, TASKS as usize,
            "expected all {TASKS} tasks to be consumed, got {total}"
        );
    }

    // 7. Push from multiple producers, steal from multiple consumers.
    #[test]
    fn multi_producer_multi_consumer() {
        const PER_PRODUCER: usize = 1_000;
        const PRODUCERS: usize = 4;
        const CONSUMERS: usize = 4;
        const TOTAL: usize = PER_PRODUCER * PRODUCERS;

        let wq = WorkQueue::<usize>::new(CONSUMERS);
        let consumed = Arc::new(AtomicUsize::new(0));

        // Spawn producers.
        let mut handles = Vec::new();
        for p in 0..PRODUCERS {
            let wq_p = wq.clone();
            handles.push(thread::spawn(move || {
                for i in 0..PER_PRODUCER {
                    wq_p.push(p * PER_PRODUCER + i);
                }
            }));
        }
        for h in handles {
            h.join().expect("producer panicked");
        }

        // Spawn consumers.
        let mut handles = Vec::new();
        for _ in 0..CONSUMERS {
            let wq_c = wq.clone();
            let cnt = Arc::clone(&consumed);
            handles.push(thread::spawn(move || {
                let mut miss = 0;
                loop {
                    match wq_c.steal() {
                        Some(_) => {
                            cnt.fetch_add(1, Ordering::Relaxed);
                            miss = 0;
                        }
                        None => {
                            miss += 1;
                            if miss > 500 {
                                break;
                            }
                        }
                    }
                }
            }));
        }
        for h in handles {
            h.join().expect("consumer panicked");
        }

        assert_eq!(consumed.load(Ordering::Relaxed), TOTAL);
    }
}

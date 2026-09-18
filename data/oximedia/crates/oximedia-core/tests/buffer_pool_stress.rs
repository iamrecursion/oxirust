//! Multi-threaded stress tests for `BufferPool`.
//!
//! Verifies that concurrent acquire/release operations across many threads
//! preserve all invariants:
//! - `in_use_count()` returns 0 after all threads complete.
//! - No buffer is lost or double-freed under concurrent access.
//! - Memory-pressure callbacks fire correctly when triggered from multiple threads.

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Barrier,
};
use std::thread;

use oximedia_core::alloc::{BufferPool, PressureConfig};

// ─────────────────────────────────────────────────────────────────────────────
// Basic concurrent acquire/release
// ─────────────────────────────────────────────────────────────────────────────

/// Spawn N threads, each repeatedly acquiring and releasing a buffer.
/// After completion, the pool must show zero in-use buffers.
#[test]
fn test_buffer_pool_concurrent_acquire_release_no_leak() {
    const NUM_THREADS: usize = 8;
    const OPS_PER_THREAD: usize = 500;
    // Large enough pool so most acquires succeed without blocking.
    let pool = Arc::new(BufferPool::new(NUM_THREADS * 2, 64));

    let barrier = Arc::new(Barrier::new(NUM_THREADS));
    let mut handles = Vec::with_capacity(NUM_THREADS);

    for _ in 0..NUM_THREADS {
        let pool_t = Arc::clone(&pool);
        let barrier_t = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            // All threads start simultaneously
            barrier_t.wait();
            for _ in 0..OPS_PER_THREAD {
                let buf = pool_t.acquire_or_alloc();
                // Perform a write to verify the buffer is usable
                {
                    let mut guard = buf.write().expect("write lock should succeed");
                    guard[0] = 0xAB;
                }
                pool_t.release(buf);
            }
        }));
    }

    for h in handles {
        h.join().expect("thread must not panic");
    }

    assert_eq!(
        pool.in_use_count(),
        0,
        "all buffers must be released after threads finish"
    );
}

/// Threads that hold a buffer concurrently with other threads releasing must
/// not interfere: in_use_count tracks checked-out buffers correctly.
#[test]
fn test_buffer_pool_concurrent_in_use_count_accuracy() {
    const NUM_THREADS: usize = 6;
    const POOL_SIZE: usize = NUM_THREADS;
    let pool = Arc::new(BufferPool::new(POOL_SIZE, 128));

    // Phase 1: all threads acquire a buffer simultaneously (pool is large enough)
    let barrier_acquire = Arc::new(Barrier::new(NUM_THREADS));
    let barrier_release = Arc::new(Barrier::new(NUM_THREADS));

    let mut handles = Vec::with_capacity(NUM_THREADS);
    for _ in 0..NUM_THREADS {
        let pool_t = Arc::clone(&pool);
        let ba = Arc::clone(&barrier_acquire);
        let br = Arc::clone(&barrier_release);
        handles.push(thread::spawn(move || {
            // All acquire at the same time
            ba.wait();
            let buf = pool_t.acquire_or_alloc();

            // All release at the same time
            br.wait();
            pool_t.release(buf);
        }));
    }

    for h in handles {
        h.join().expect("thread must not panic");
    }

    assert_eq!(
        pool.in_use_count(),
        0,
        "in_use_count must be 0 after all releases"
    );
    // Buffers released by threads are put back in the pool; the pool may
    // hold up to POOL_SIZE free buffers.
    assert!(
        pool.available() <= POOL_SIZE,
        "available() must not exceed POOL_SIZE"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Stress: acquire_or_alloc under contention with small pool
// ─────────────────────────────────────────────────────────────────────────────

/// Pool smaller than thread count — threads must fall back to allocating fresh
/// buffers.  No deadlock or panic must occur.
#[test]
fn test_buffer_pool_stress_with_small_pool() {
    const NUM_THREADS: usize = 16;
    const POOL_SLOTS: usize = 2; // far fewer than threads
    const OPS_PER_THREAD: usize = 200;
    let pool = Arc::new(BufferPool::new(POOL_SLOTS, 32));

    let mut handles = Vec::with_capacity(NUM_THREADS);
    for _ in 0..NUM_THREADS {
        let pool_t = Arc::clone(&pool);
        handles.push(thread::spawn(move || {
            for _ in 0..OPS_PER_THREAD {
                // acquire_or_alloc never fails
                let buf = pool_t.acquire_or_alloc();
                pool_t.release(buf);
            }
        }));
    }

    for h in handles {
        h.join().expect("thread must not panic");
    }

    assert_eq!(pool.in_use_count(), 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Memory pressure callback under concurrent access
// ─────────────────────────────────────────────────────────────────────────────

/// The pressure callback fires when a burst of concurrent releases pushes the
/// free count above the configured watermark.
///
/// Strategy: main thread acquires NUM_THREADS buffers (all in-use, free=0),
/// then NUM_THREADS threads each release exactly one buffer simultaneously.
/// After all releases free=NUM_THREADS > WATERMARK, so the callback fires ≥ 1.
#[test]
fn test_buffer_pool_pressure_callback_fires_under_concurrency() {
    const NUM_THREADS: usize = 8;
    const WATERMARK: usize = 4;
    const TARGET: usize = 2;

    let counter = Arc::new(AtomicUsize::new(0));
    let counter_cb = Arc::clone(&counter);

    let mut pool = BufferPool::with_capacity(NUM_THREADS * 2, 64);
    pool.set_pressure_config(PressureConfig {
        high_watermark_free: WATERMARK,
        shrink_to_target: TARGET,
    });
    pool.on_pressure(move || {
        counter_cb.fetch_add(1, Ordering::Relaxed);
    });
    let pool = Arc::new(pool);

    // Acquire all NUM_THREADS buffers on the main thread → in_use=8, free=0
    let held: Vec<_> = (0..NUM_THREADS).map(|_| pool.acquire_or_alloc()).collect();
    assert_eq!(pool.in_use_count(), NUM_THREADS);
    assert_eq!(pool.available(), 0);

    // Each thread gets one buffer to release; a barrier ensures simultaneous release
    let barrier = Arc::new(Barrier::new(NUM_THREADS));
    let mut handles = Vec::with_capacity(NUM_THREADS);

    for buf in held {
        let pool_t = Arc::clone(&pool);
        let b = Arc::clone(&barrier);
        // Arc<RwLock<Vec<u8>>> is Send, so moving `buf` to a thread is safe.
        handles.push(thread::spawn(move || {
            b.wait(); // synchronise: all threads release at once
            pool_t.release(buf);
        }));
    }

    for h in handles {
        h.join().expect("thread must not panic");
    }

    // After 8 concurrent releases the free count eventually reaches NUM_THREADS
    // (even serialised by the lock), so the watermark (4) is crossed and the
    // callback must fire at least once.
    assert!(
        counter.load(Ordering::Relaxed) >= 1,
        "pressure callback must fire at least once when {} releases exceed watermark {}",
        NUM_THREADS,
        WATERMARK
    );
    assert_eq!(pool.in_use_count(), 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Shrink_to: in-use buffers survive; free list is trimmed
// ─────────────────────────────────────────────────────────────────────────────

/// Threads holding acquired buffers are not affected by a concurrent shrink_to.
#[test]
fn test_buffer_pool_shrink_does_not_reclaim_in_use_buffers() {
    const HOLDERS: usize = 4;
    const FREE_EXTRA: usize = 6;
    // Capacity large enough for holders + extra free buffers
    let pool = Arc::new(BufferPool::with_capacity(HOLDERS + FREE_EXTRA + 4, 32));

    // Phase 1: Acquire buffers that will be held through the shrink
    let barrier_hold = Arc::new(Barrier::new(HOLDERS + 1)); // +1 for main
    let barrier_release = Arc::new(Barrier::new(HOLDERS + 1));
    let mut holder_handles = Vec::with_capacity(HOLDERS);

    for _ in 0..HOLDERS {
        let pool_t = Arc::clone(&pool);
        let bh = Arc::clone(&barrier_hold);
        let br = Arc::clone(&barrier_release);
        holder_handles.push(thread::spawn(move || {
            let buf = pool_t.acquire_or_alloc();
            bh.wait(); // signal: I have my buffer
            br.wait(); // wait for shrink to complete before releasing
            pool_t.release(buf);
        }));
    }

    // Wait until all holders have their buffers
    barrier_hold.wait();

    // Inject extra free buffers manually and then shrink
    for _ in 0..FREE_EXTRA {
        let buf = pool.acquire_or_alloc();
        pool.release(buf);
    }

    // Shrink to 0 free buffers — must not touch in-use buffers
    pool.shrink_to(0);
    assert_eq!(
        pool.available(),
        0,
        "all free buffers should be dropped by shrink_to(0)"
    );
    assert_eq!(
        pool.in_use_count(),
        HOLDERS,
        "in-use buffers must not be affected by shrink_to"
    );

    // Release all holder threads
    barrier_release.wait();

    for h in holder_handles {
        h.join().expect("thread must not panic");
    }

    assert_eq!(pool.in_use_count(), 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Buffer contents are isolated between threads
// ─────────────────────────────────────────────────────────────────────────────

/// Each thread writes a unique value to its buffer; after release the pool
/// zeroes the buffer.  No thread should see another thread's stale data.
#[test]
fn test_buffer_pool_buffer_isolation_across_threads() {
    const NUM_THREADS: usize = 8;
    let pool = Arc::new(BufferPool::new(NUM_THREADS, 256));

    let errors = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::with_capacity(NUM_THREADS);

    for thread_id in 0u8..NUM_THREADS as u8 {
        let pool_t = Arc::clone(&pool);
        let errors_t = Arc::clone(&errors);
        handles.push(thread::spawn(move || {
            for _ in 0..100 {
                let buf = pool_t.acquire_or_alloc();
                {
                    let mut guard = buf.write().expect("write lock");
                    // Fill with our unique sentinel
                    guard.fill(thread_id);
                }
                pool_t.release(buf);

                // After release, re-acquire (may get a different buffer, zeroed)
                let buf2 = pool_t.acquire_or_alloc();
                {
                    let guard = buf2.read().expect("read lock");
                    // Pool zeroes on release — no data should survive
                    if guard.iter().any(|&b| b == thread_id && thread_id != 0) {
                        // Could be coincidence if value is 0, so only fail for non-zero ids
                        // (The pool zeroes the buffer, so this must be 0)
                        errors_t.fetch_add(1, Ordering::Relaxed);
                    }
                }
                pool_t.release(buf2);
            }
        }));
    }

    for h in handles {
        h.join().expect("thread must not panic");
    }

    assert_eq!(
        errors.load(Ordering::Relaxed),
        0,
        "no thread should see stale data from another thread"
    );
    assert_eq!(pool.in_use_count(), 0);
}

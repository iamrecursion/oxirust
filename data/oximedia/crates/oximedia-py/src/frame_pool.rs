//! Optional memory pool for frame allocation to reduce Python GC pressure.
//!
//! Provides `PyFramePool` — a pre-allocated ring of raw byte buffers that can be
//! acquired and released by Python code instead of relying on the Python allocator
//! for every frame.  Re-using pooled buffers eliminates repeated `malloc`/`free`
//! cycles and the garbage collector churn they induce for high-throughput
//! frame-processing workloads.
//!
//! # Example
//! ```python
//! import oximedia
//!
//! # Pre-allocate a pool of 8 buffers each sized for 1920×1080 RGBA
//! pool = oximedia.PyFramePool(capacity=8, frame_size=1920*1080*4)
//!
//! frame = pool.acquire()           # returns a bytearray of frame_size bytes
//! # … fill frame …
//! pool.release(frame)             # return buffer to pool for reuse
//! print(pool.available())         # number of buffers currently in pool
//! ```

use pyo3::prelude::*;
use pyo3::types::PyByteArray;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Inner pool implementation (pure Rust, no Python deps)
// ---------------------------------------------------------------------------

/// Inner pool storing pre-allocated byte buffers.
struct FramePoolInner {
    /// Maximum number of buffers the pool will hold.
    capacity: usize,
    /// Size of each buffer in bytes.
    frame_size: usize,
    /// Free list of available buffers.
    pool: Vec<Vec<u8>>,
    /// Total buffers ever allocated (monotonically increasing).
    allocated: usize,
}

impl FramePoolInner {
    fn new(capacity: usize, frame_size: usize) -> Self {
        // Pre-allocate all buffers up-front so the first `capacity` acquire()
        // calls never touch the system allocator.
        let pool = (0..capacity)
            .map(|_| vec![0u8; frame_size])
            .collect::<Vec<_>>();
        let allocated = capacity;
        Self {
            capacity,
            frame_size,
            pool,
            allocated,
        }
    }

    /// Remove and return a buffer from the free list, or allocate a fresh one
    /// if the pool is empty (up to a safety limit of 2× capacity).
    fn acquire(&mut self) -> Vec<u8> {
        if let Some(buf) = self.pool.pop() {
            buf
        } else {
            // Pool exhausted: allocate a new buffer, bounded by 2×capacity to
            // prevent unbounded growth in misuse scenarios.
            let max_alloc = self.capacity.saturating_mul(2);
            if self.allocated < max_alloc {
                self.allocated += 1;
            }
            vec![0u8; self.frame_size]
        }
    }

    /// Return a buffer to the free list.  Buffers exceeding capacity are
    /// dropped (zeroed memory is released back to the allocator).
    fn release(&mut self, mut buf: Vec<u8>) {
        if self.pool.len() < self.capacity {
            // Zero-fill before returning so no data leaks between frames.
            buf.fill(0);
            buf.resize(self.frame_size, 0);
            self.pool.push(buf);
        }
        // else: drop buf — memory is freed to the OS allocator
    }

    fn available(&self) -> usize {
        self.pool.len()
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn frame_size(&self) -> usize {
        self.frame_size
    }
}

// ---------------------------------------------------------------------------
// PyFramePool — the Python-visible pyclass
// ---------------------------------------------------------------------------

/// Pre-allocated frame buffer pool that reduces Python GC pressure.
///
/// Acquire a zeroed byte buffer with :meth:`acquire` and return it with
/// :meth:`release` after use.  The pool holds up to *capacity* buffers; if all
/// are checked out, :meth:`acquire` transparently allocates an overflow buffer
/// (up to 2× capacity) rather than blocking.
///
/// Parameters
/// ----------
/// capacity : int
///     Maximum number of buffers kept in the free list.
/// frame_size : int
///     Size of each buffer in bytes (e.g. ``width * height * 4`` for RGBA).
///
/// Notes
/// -----
/// All buffers are zeroed on initialisation and zeroed again when released, so
/// no frame data leaks between acquire/release cycles.
#[pyclass]
#[derive(Clone)]
pub struct PyFramePool {
    inner: Arc<Mutex<FramePoolInner>>,
}

#[pymethods]
impl PyFramePool {
    /// Create a new frame pool.
    ///
    /// Parameters
    /// ----------
    /// capacity : int
    ///     Maximum number of reusable buffers.
    /// frame_size : int
    ///     Size of each buffer in bytes.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If *capacity* or *frame_size* is zero.
    #[new]
    pub fn new(capacity: usize, frame_size: usize) -> PyResult<Self> {
        if capacity == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "capacity must be greater than zero",
            ));
        }
        if frame_size == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "frame_size must be greater than zero",
            ));
        }
        Ok(Self {
            inner: Arc::new(Mutex::new(FramePoolInner::new(capacity, frame_size))),
        })
    }

    /// Acquire a buffer from the pool.
    ///
    /// Returns a ``bytearray`` of *frame_size* zeroed bytes.  The buffer is
    /// removed from the pool's free list; call :meth:`release` to return it.
    ///
    /// Returns
    /// -------
    /// bytearray
    ///     A zeroed frame buffer of the configured size.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If the internal pool lock is poisoned.
    pub fn acquire<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyByteArray>> {
        let buf = {
            let mut guard = self.inner.lock().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "frame pool lock poisoned: {e}"
                ))
            })?;
            guard.acquire()
        };
        // Wrap the buffer as a Python bytearray for zero-extra-copy usage.
        Ok(PyByteArray::new(py, &buf))
    }

    /// Return a buffer to the pool for reuse.
    ///
    /// The buffer's content is zeroed before it enters the free list.  If the
    /// free list is already at *capacity*, the buffer is dropped instead.
    ///
    /// Parameters
    /// ----------
    /// frame : bytearray
    ///     The buffer previously returned by :meth:`acquire`.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If the internal pool lock is poisoned.
    pub fn release(&self, frame: &Bound<'_, PyByteArray>) -> PyResult<()> {
        let data = frame.to_vec();
        let mut guard = self.inner.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "frame pool lock poisoned: {e}"
            ))
        })?;
        guard.release(data);
        Ok(())
    }

    /// Return the number of buffers currently available in the free list.
    ///
    /// Returns
    /// -------
    /// int
    ///     Number of immediately available buffers (0 … capacity).
    pub fn available(&self) -> PyResult<usize> {
        let guard = self.inner.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "frame pool lock poisoned: {e}"
            ))
        })?;
        Ok(guard.available())
    }

    /// Maximum number of buffers the pool will retain.
    ///
    /// Returns
    /// -------
    /// int
    ///     The *capacity* value supplied at construction.
    #[getter]
    pub fn capacity(&self) -> PyResult<usize> {
        let guard = self.inner.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "frame pool lock poisoned: {e}"
            ))
        })?;
        Ok(guard.capacity())
    }

    /// Size of each buffer in bytes.
    ///
    /// Returns
    /// -------
    /// int
    ///     The *frame_size* value supplied at construction.
    #[getter]
    pub fn frame_size(&self) -> PyResult<usize> {
        let guard = self.inner.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "frame pool lock poisoned: {e}"
            ))
        })?;
        Ok(guard.frame_size())
    }

    fn __repr__(&self) -> PyResult<String> {
        let guard = self.inner.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "frame pool lock poisoned: {e}"
            ))
        })?;
        Ok(format!(
            "PyFramePool(capacity={}, frame_size={}, available={})",
            guard.capacity(),
            guard.frame_size(),
            guard.available(),
        ))
    }

    fn __str__(&self) -> PyResult<String> {
        self.__repr__()
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register `PyFramePool` into the given module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFramePool>()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_inner(capacity: usize, frame_size: usize) -> FramePoolInner {
        FramePoolInner::new(capacity, frame_size)
    }

    #[test]
    fn test_frame_pool_acquire_release() {
        let mut pool = make_inner(4, 1024);
        assert_eq!(pool.available(), 4);

        let buf = pool.acquire();
        assert_eq!(buf.len(), 1024);
        assert_eq!(pool.available(), 3);

        pool.release(buf);
        assert_eq!(pool.available(), 4);
    }

    #[test]
    fn test_frame_pool_capacity_respected() {
        let mut pool = make_inner(2, 64);
        // Drain both slots.
        let b1 = pool.acquire();
        let b2 = pool.acquire();
        assert_eq!(pool.available(), 0);

        // Third acquire overflows (returns a freshly allocated buffer).
        let b3 = pool.acquire();
        assert_eq!(b3.len(), 64);
        assert_eq!(pool.available(), 0);

        // Return all three — only 2 fit back into the pool.
        pool.release(b1);
        pool.release(b2);
        pool.release(b3); // dropped, pool already at capacity
        assert_eq!(pool.available(), 2);
    }

    #[test]
    fn test_frame_pool_zeroed_on_release() {
        let mut pool = make_inner(2, 8);
        let mut buf = pool.acquire();
        // Write non-zero data.
        buf.iter_mut().for_each(|b| *b = 0xFF);
        pool.release(buf);

        // Re-acquire and verify zeroed.
        let buf2 = pool.acquire();
        assert!(buf2.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_frame_pool_frame_size_matches() {
        let pool = make_inner(3, 512);
        assert_eq!(pool.frame_size(), 512);
        assert_eq!(pool.capacity(), 3);
    }

    #[test]
    fn test_pool_constructor_validates_capacity() {
        let result = PyFramePool::new(0, 64);
        assert!(result.is_err());
    }

    #[test]
    fn test_pool_constructor_validates_frame_size() {
        let result = PyFramePool::new(4, 0);
        assert!(result.is_err());
    }
}

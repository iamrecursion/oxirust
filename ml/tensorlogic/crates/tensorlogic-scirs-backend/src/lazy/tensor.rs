//! Lazy tensor wrapper with memoization support.

use parking_lot::RwLock;
use std::sync::Arc;

/// A lazily-evaluated tensor that may or may not have been computed yet.
///
/// Wraps a value in an `Arc<RwLock<Option<T>>>` so that the same logical tensor
/// can be shared by multiple graph edges, and the result written exactly once when
/// the computation completes.
pub struct LazyTensor<T: Clone> {
    inner: Arc<RwLock<Option<T>>>,
    shape_hint: Option<Vec<usize>>,
    /// Optional debug / display name for the tensor.
    pub name: Option<String>,
}

impl<T: Clone> LazyTensor<T> {
    /// Create a new lazy tensor that has not yet been computed.
    ///
    /// `shape_hint` is used for memory estimation without requiring the value
    /// to be materialised.
    pub fn pending(shape_hint: Option<Vec<usize>>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(None)),
            shape_hint,
            name: None,
        }
    }

    /// Create a lazy tensor that is already computed (eager).
    pub fn eager(value: T) -> Self {
        Self {
            inner: Arc::new(RwLock::new(Some(value))),
            shape_hint: None,
            name: None,
        }
    }

    /// Returns `true` if the tensor value has been set.
    pub fn is_computed(&self) -> bool {
        self.inner.read().is_some()
    }

    /// Clone the inner value out if it has been computed.
    pub fn get(&self) -> Option<T> {
        self.inner.read().clone()
    }

    /// Store a computed value into this tensor.
    pub fn set(&self, value: T) {
        *self.inner.write() = Some(value);
    }

    /// Extract the value and clear this tensor back to *pending* state.
    ///
    /// Useful for memory-optimal scheduling: after downstream consumers have
    /// read the value it can be released without keeping a second copy.
    pub fn take(&self) -> Option<T> {
        self.inner.write().take()
    }

    /// The shape hint provided at construction time (if any).
    pub fn shape_hint(&self) -> Option<&[usize]> {
        self.shape_hint.as_deref()
    }

    /// Estimated memory footprint in bytes.
    ///
    /// Computed as `∏ shape_hint * 8` (f64 elements).  Returns 0 when no shape
    /// hint is available.
    pub fn memory_estimate_bytes(&self) -> usize {
        match &self.shape_hint {
            Some(shape) => shape.iter().product::<usize>() * 8,
            None => 0,
        }
    }
}

impl<T: Clone> Clone for LazyTensor<T> {
    /// Shallow clone — shares the same inner `Arc`.
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            shape_hint: self.shape_hint.clone(),
            name: self.name.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lazy_tensor_pending_not_computed() {
        let t: LazyTensor<i32> = LazyTensor::pending(Some(vec![3, 4]));
        assert!(!t.is_computed());
        assert!(t.get().is_none());
    }

    #[test]
    fn test_lazy_tensor_set_and_get() {
        let t: LazyTensor<i32> = LazyTensor::pending(None);
        t.set(42);
        assert!(t.is_computed());
        assert_eq!(t.get(), Some(42));
    }

    #[test]
    fn test_lazy_tensor_eager_is_computed() {
        let t = LazyTensor::eager(99_i32);
        assert!(t.is_computed());
        assert_eq!(t.get(), Some(99));
    }

    #[test]
    fn test_lazy_tensor_take_clears() {
        let t = LazyTensor::eager(7_i32);
        let val = t.take();
        assert_eq!(val, Some(7));
        assert!(!t.is_computed());
        assert!(t.get().is_none());
    }

    #[test]
    fn test_lazy_tensor_memory_estimate_with_hint() {
        let t: LazyTensor<i32> = LazyTensor::pending(Some(vec![2, 3, 4]));
        // 2 * 3 * 4 * 8 = 192
        assert_eq!(t.memory_estimate_bytes(), 192);
    }

    #[test]
    fn test_lazy_tensor_memory_estimate_no_hint() {
        let t: LazyTensor<i32> = LazyTensor::pending(None);
        assert_eq!(t.memory_estimate_bytes(), 0);
    }

    #[test]
    fn test_lazy_tensor_clone_shares_inner() {
        let t1: LazyTensor<i32> = LazyTensor::pending(Some(vec![2, 2]));
        let t2 = t1.clone();
        // Writing through t1 is visible via t2
        t1.set(123);
        assert_eq!(t2.get(), Some(123));
        // The two shapes are equal
        assert_eq!(t1.shape_hint(), Some([2_usize, 2_usize].as_ref()));
    }
}

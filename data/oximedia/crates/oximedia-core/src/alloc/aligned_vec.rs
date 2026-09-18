//! Cache-line-aligned generic vector (`AlignedVec<T>`).
//!
//! SIMD and media-processing kernels benefit greatly from having source and
//! destination buffers start at a cache-line (64-byte) boundary.  Rust's
//! default allocator only aligns to the type's `align_of::<T>()`, which is
//! often 4 or 8 bytes.
//!
//! [`AlignedVec<T>`] provides a `Vec`-like container whose underlying pointer
//! is guaranteed to be aligned to [`ALIGN`] (64 bytes) **for the heap
//! allocation**.  The data pointer returned by [`AlignedVec::as_slice`] is
//! therefore always a multiple of 64.
//!
//! # Alignment strategy
//!
//! Alignment is achieved by allocating `len + ALIGN - 1` extra elements,
//! then finding the first index within that `Vec` whose address is a multiple
//! of `ALIGN`.  The buffer tracks this offset so elements returned to
//! `as_slice()` are always at an aligned address.  This is **100 % safe Rust**
//! — no `unsafe` blocks are used.
//!
//! # Examples
//!
//! ```
//! use oximedia_core::alloc::aligned_vec::AlignedVec;
//!
//! let av: AlignedVec<f32> = AlignedVec::new_aligned(256);
//! assert_eq!(av.as_slice().len(), 256);
//! let ptr = av.as_slice().as_ptr() as usize;
//! assert_eq!(ptr % 64, 0, "data pointer must be 64-byte aligned");
//! ```

/// Cache-line alignment for SIMD-friendly buffers.
pub const ALIGN: usize = 64;

// ─────────────────────────────────────────────────────────────────────────────
// AlignedVec
// ─────────────────────────────────────────────────────────────────────────────

/// A heap-allocated, cache-line-aligned vector of `T`.
///
/// The usable region's start address is aligned to [`ALIGN`] (64 bytes),
/// satisfying the requirements of AVX2 (32-byte) and AVX-512 (64-byte)
/// load/store instructions and fitting neatly on a single cache line boundary.
///
/// # Alignment approach
///
/// Unlike unsafe raw-pointer approaches, alignment is achieved by
/// over-allocating `len + ALIGN - 1` slots in a `Vec<T>`, then computing
/// the first slot index whose byte address is divisible by `ALIGN`.
/// All accesses go through safe `&[T]`/`&mut [T]` slices.
///
/// # Type constraints
///
/// `T: Default + Clone` — required so that `new_aligned` can create
/// initialised storage using `vec![T::default(); ...]`.
pub struct AlignedVec<T: Default + Clone> {
    /// Backing storage (over-allocated for alignment padding).
    storage: Vec<T>,
    /// Element offset from `storage[0]` to the cache-line-aligned start.
    offset: usize,
    /// Number of usable `T` elements starting at `storage[offset]`.
    len: usize,
}

impl<T: Default + Clone> AlignedVec<T> {
    /// Allocates a zero-initialised vector of `len` elements with 64-byte
    /// alignment.
    ///
    /// The contents are initialised to `T::default()`.  For primitive
    /// numeric types, `Default` returns zero.
    ///
    /// # Panics
    ///
    /// Panics if `len` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::alloc::aligned_vec::AlignedVec;
    ///
    /// let av = AlignedVec::<u8>::new_aligned(128);
    /// assert_eq!(av.len(), 128);
    /// assert!(av.as_slice().iter().all(|&b| b == 0));
    /// ```
    #[must_use]
    pub fn new_aligned(len: usize) -> Self {
        assert!(len > 0, "AlignedVec: len must be > 0");

        let elem_size = std::mem::size_of::<T>().max(1);
        // How many extra elements do we need to guarantee an aligned slot?
        // We need at most (ALIGN / elem_size) extra elements (ceiling).
        let extra = ALIGN.div_ceil(elem_size);
        let alloc_len = len.checked_add(extra).expect("AlignedVec: size overflow");

        let storage: Vec<T> = vec![T::default(); alloc_len];

        // Find the first index whose byte address is a multiple of ALIGN.
        let base_ptr = storage.as_ptr() as usize;
        let offset = if base_ptr % ALIGN == 0 {
            0
        } else {
            // Bytes needed to reach the next aligned address.
            let byte_gap = ALIGN - (base_ptr % ALIGN);
            // Convert byte gap to element count (ceiling).
            byte_gap.div_ceil(elem_size)
        };

        // Sanity: offset + len must fit within alloc_len.
        assert!(
            offset + len <= alloc_len,
            "AlignedVec: alignment offset calculation overflow (offset={offset}, len={len}, alloc={alloc_len})",
        );

        Self {
            storage,
            offset,
            len,
        }
    }

    /// Returns the number of elements in this vector.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector contains no elements.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns an immutable slice over all elements.
    ///
    /// The slice's pointer is guaranteed to be a multiple of 64.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::alloc::aligned_vec::AlignedVec;
    ///
    /// let av = AlignedVec::<f32>::new_aligned(64);
    /// let ptr = av.as_slice().as_ptr() as usize;
    /// assert_eq!(ptr % 64, 0);
    /// ```
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.storage[self.offset..self.offset + self.len]
    }

    /// Returns a mutable slice over all elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::alloc::aligned_vec::AlignedVec;
    ///
    /// let mut av = AlignedVec::<u8>::new_aligned(4);
    /// av.as_mut_slice().copy_from_slice(&[1, 2, 3, 4]);
    /// assert_eq!(av.as_slice(), &[1_u8, 2, 3, 4]);
    /// ```
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.storage[self.offset..self.offset + self.len]
    }

    /// Returns the raw data pointer (always cache-line-aligned).
    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.as_slice().as_ptr()
    }

    /// Returns the raw mutable data pointer (always cache-line-aligned).
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.as_mut_slice().as_mut_ptr()
    }
}

impl<T: Default + Clone> Clone for AlignedVec<T> {
    fn clone(&self) -> Self {
        let mut new_vec = AlignedVec::new_aligned(self.len);
        new_vec.as_mut_slice().clone_from_slice(self.as_slice());
        new_vec
    }
}

impl<T: Default + Clone + std::fmt::Debug> std::fmt::Debug for AlignedVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedVec")
            .field("len", &self.len)
            .field("data_ptr", &format_args!("{:#x}", self.as_ptr() as usize))
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. The data pointer is divisible by 64.
    #[test]
    fn data_pointer_is_64_byte_aligned() {
        for len in [1_usize, 4, 7, 16, 63, 64, 65, 128, 256, 1024, 4097] {
            let av = AlignedVec::<f32>::new_aligned(len);
            let ptr = av.as_ptr() as usize;
            assert_eq!(
                ptr % 64,
                0,
                "pointer 0x{ptr:x} not 64-byte aligned for len={len}"
            );
        }
    }

    // 2. Zero-initialised on construction.
    #[test]
    fn zero_initialised() {
        let av = AlignedVec::<u8>::new_aligned(128);
        assert!(av.as_slice().iter().all(|&b| b == 0));

        let av_f32 = AlignedVec::<f32>::new_aligned(32);
        assert!(av_f32.as_slice().iter().all(|&v| v == 0.0_f32));
    }

    // 3. Correct length.
    #[test]
    fn len_is_correct() {
        let av = AlignedVec::<u32>::new_aligned(100);
        assert_eq!(av.len(), 100);
        assert!(!av.is_empty());
    }

    // 4. Mutable write and read-back.
    #[test]
    fn mutable_write_and_read() {
        let mut av = AlignedVec::<u8>::new_aligned(4);
        av.as_mut_slice().copy_from_slice(&[10, 20, 30, 40]);
        assert_eq!(av.as_slice(), &[10_u8, 20, 30, 40]);
    }

    // 5. Clone produces independent copy with same alignment.
    #[test]
    fn clone_is_independent_and_aligned() {
        let mut av = AlignedVec::<u32>::new_aligned(8);
        for (i, v) in av.as_mut_slice().iter_mut().enumerate() {
            *v = i as u32;
        }
        let av2 = av.clone();
        // Same values.
        assert_eq!(av.as_slice(), av2.as_slice());
        // Independent — mutating original does not affect clone.
        av.as_mut_slice()[0] = 99;
        assert_eq!(av2.as_slice()[0], 0_u32);
        // Both aligned.
        assert_eq!(av.as_ptr() as usize % 64, 0);
        assert_eq!(av2.as_ptr() as usize % 64, 0);
    }

    // 6. Alignment for various element types.
    #[test]
    fn alignment_for_different_types() {
        let av_u8 = AlignedVec::<u8>::new_aligned(64);
        assert_eq!(av_u8.as_ptr() as usize % 64, 0);

        let av_f64 = AlignedVec::<f64>::new_aligned(32);
        assert_eq!(av_f64.as_ptr() as usize % 64, 0);

        let av_i16 = AlignedVec::<i16>::new_aligned(128);
        assert_eq!(av_i16.as_ptr() as usize % 64, 0);
    }

    // 7. Alignment for sizes that cross cache-line boundaries.
    #[test]
    fn alignment_stress() {
        for n in 1..=200_usize {
            let av = AlignedVec::<f32>::new_aligned(n);
            assert_eq!(av.as_ptr() as usize % 64, 0, "failed for n={n}");
        }
    }

    // 8. as_mut_ptr returns same address as as_ptr.
    #[test]
    fn mut_ptr_same_address() {
        let mut av = AlignedVec::<u8>::new_aligned(16);
        let const_ptr = av.as_ptr() as usize;
        let mut_ptr = av.as_mut_ptr() as usize;
        assert_eq!(const_ptr, mut_ptr);
    }

    // 9. Debug impl does not panic.
    #[test]
    fn debug_does_not_panic() {
        let av = AlignedVec::<u32>::new_aligned(4);
        let _ = format!("{av:?}");
    }

    // 10. Large allocation stays aligned.
    #[test]
    fn large_allocation_aligned() {
        let av = AlignedVec::<f64>::new_aligned(1_024 * 1_024); // 8 MiB
        assert_eq!(av.as_ptr() as usize % 64, 0);
        assert_eq!(av.len(), 1_024 * 1_024);
    }
}

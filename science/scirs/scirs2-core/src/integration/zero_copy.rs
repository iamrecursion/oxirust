//! Zero-Copy Array Views for Cross-Module Data Sharing
//!
//! This module provides efficient zero-copy data sharing mechanisms between
//! SciRS2 modules, enabling high-performance data flow without memory copies.
//!
//! # Key Concepts
//!
//! - **SharedArrayView**: Immutable view into array data
//! - **SharedArrayViewMut**: Mutable view into array data
//! - **ZeroCopyBuffer**: Type-erased buffer for cross-module sharing
//! - **ArrayBridge**: Bridge between different array implementations

use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::slice;
use std::sync::Arc;

use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};

/// Memory alignment requirements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Alignment {
    /// No specific alignment required
    None,
    /// 16-byte alignment (SSE)
    Align16,
    /// 32-byte alignment (AVX)
    Align32,
    /// 64-byte alignment (AVX-512, cache line)
    Align64,
    /// Custom alignment
    Custom(usize),
}

impl Alignment {
    /// Get the byte alignment value
    #[must_use]
    pub const fn bytes(&self) -> usize {
        match self {
            Alignment::None => 1,
            Alignment::Align16 => 16,
            Alignment::Align32 => 32,
            Alignment::Align64 => 64,
            Alignment::Custom(n) => *n,
        }
    }

    /// Check if a pointer is aligned to this alignment
    #[must_use]
    pub fn is_aligned<T>(&self, ptr: *const T) -> bool {
        (ptr as usize) % self.bytes() == 0
    }

    /// Get optimal alignment for SIMD operations based on element type
    #[must_use]
    pub const fn optimal_for_simd<T>() -> Self {
        let size = mem::size_of::<T>();
        if size >= 8 {
            Alignment::Align64
        } else if size >= 4 {
            Alignment::Align32
        } else {
            Alignment::Align16
        }
    }
}

impl Default for Alignment {
    fn default() -> Self {
        Alignment::None
    }
}

/// Memory layout description for array data
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryLayout {
    /// Shape of the array
    pub shape: Vec<usize>,
    /// Strides in bytes
    pub strides: Vec<isize>,
    /// Total number of elements
    pub len: usize,
    /// Size of each element in bytes
    pub element_size: usize,
    /// Memory alignment
    pub alignment: Alignment,
    /// Whether data is contiguous in memory
    pub is_contiguous: bool,
    /// Whether data is in C (row-major) order
    pub is_c_contiguous: bool,
    /// Whether data is in Fortran (column-major) order
    pub is_f_contiguous: bool,
}

impl MemoryLayout {
    /// Create a new memory layout for a contiguous array
    #[must_use]
    pub fn contiguous<T>(shape: &[usize]) -> Self {
        let element_size = mem::size_of::<T>();
        let len: usize = shape.iter().product();
        let strides = Self::compute_c_strides(shape, element_size);

        Self {
            shape: shape.to_vec(),
            strides,
            len,
            element_size,
            alignment: Alignment::optimal_for_simd::<T>(),
            is_contiguous: true,
            is_c_contiguous: true,
            is_f_contiguous: shape.len() <= 1,
        }
    }

    /// Create a Fortran-order (column-major) layout
    #[must_use]
    pub fn fortran_contiguous<T>(shape: &[usize]) -> Self {
        let element_size = mem::size_of::<T>();
        let len: usize = shape.iter().product();
        let strides = Self::compute_f_strides(shape, element_size);

        Self {
            shape: shape.to_vec(),
            strides,
            len,
            element_size,
            alignment: Alignment::optimal_for_simd::<T>(),
            is_contiguous: true,
            is_c_contiguous: shape.len() <= 1,
            is_f_contiguous: true,
        }
    }

    /// Compute C-order (row-major) strides
    fn compute_c_strides(shape: &[usize], element_size: usize) -> Vec<isize> {
        let ndim = shape.len();
        if ndim == 0 {
            return vec![];
        }

        let mut strides = vec![0isize; ndim];
        strides[ndim - 1] = element_size as isize;

        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * (shape[i + 1] as isize);
        }

        strides
    }

    /// Compute Fortran-order (column-major) strides
    fn compute_f_strides(shape: &[usize], element_size: usize) -> Vec<isize> {
        let ndim = shape.len();
        if ndim == 0 {
            return vec![];
        }

        let mut strides = vec![0isize; ndim];
        strides[0] = element_size as isize;

        for i in 1..ndim {
            strides[i] = strides[i - 1] * (shape[i - 1] as isize);
        }

        strides
    }

    /// Get the number of dimensions
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Check if layouts are compatible for zero-copy operations
    #[must_use]
    pub fn is_compatible(&self, other: &Self) -> bool {
        self.shape == other.shape
            && self.element_size == other.element_size
            && self.is_contiguous
            && other.is_contiguous
    }

    /// Get total size in bytes
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.len * self.element_size
    }
}

/// Trait for types that have contiguous memory representation
pub trait ContiguousMemory {
    /// Get a pointer to the start of the memory
    fn as_ptr(&self) -> *const u8;

    /// Get the memory layout
    fn layout(&self) -> &MemoryLayout;

    /// Check if the memory is contiguous
    fn is_contiguous(&self) -> bool {
        self.layout().is_contiguous
    }

    /// Get the size in bytes
    fn size_bytes(&self) -> usize {
        self.layout().size_bytes()
    }
}

/// Trait for types that have mutable contiguous memory
pub trait ContiguousMemoryMut: ContiguousMemory {
    /// Get a mutable pointer to the start of the memory
    fn as_mut_ptr(&mut self) -> *mut u8;
}

/// Immutable zero-copy view into array data
#[derive(Debug)]
pub struct SharedArrayView<'a, T> {
    /// Pointer to the data
    ptr: *const T,
    /// Length of the data
    len: usize,
    /// Memory layout
    layout: MemoryLayout,
    /// Lifetime marker
    _marker: PhantomData<&'a T>,
}

impl<'a, T> SharedArrayView<'a, T> {
    /// Create a new view from a slice
    #[must_use]
    pub fn from_slice(data: &'a [T]) -> Self {
        let layout = MemoryLayout::contiguous::<T>(&[data.len()]);
        Self {
            ptr: data.as_ptr(),
            len: data.len(),
            layout,
            _marker: PhantomData,
        }
    }

    /// Create a view with custom layout
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - The pointer is valid for the entire layout
    /// - The layout correctly describes the memory
    /// - The data lives at least as long as 'a
    pub unsafe fn from_raw_parts(ptr: *const T, layout: MemoryLayout) -> Self {
        Self {
            ptr,
            len: layout.len,
            layout,
            _marker: PhantomData,
        }
    }

    /// Get the length of the view
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Check if the view is empty
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the memory layout
    #[must_use]
    pub const fn layout(&self) -> &MemoryLayout {
        &self.layout
    }

    /// Get the shape
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.layout.shape
    }

    /// Get a reference to element at index
    ///
    /// # Safety
    ///
    /// The index must be within bounds
    #[must_use]
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        &*self.ptr.add(index)
    }

    /// Get a reference to element at index with bounds checking
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len {
            // SAFETY: We just checked the bounds
            Some(unsafe { self.get_unchecked(index) })
        } else {
            None
        }
    }

    /// Convert to a slice if contiguous
    pub fn as_slice(&self) -> Option<&'a [T]> {
        if self.layout.is_contiguous {
            // SAFETY: Layout is contiguous, so data forms a valid slice
            Some(unsafe { slice::from_raw_parts(self.ptr, self.len) })
        } else {
            None
        }
    }

    /// Create a subview
    pub fn slice(&self, start: usize, end: usize) -> CoreResult<SharedArrayView<'a, T>> {
        if start > end || end > self.len {
            return Err(CoreError::ValidationError(
                ErrorContext::new(format!(
                    "Invalid slice range [{start}, {end}) for length {len}",
                    len = self.len
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        let new_len = end - start;
        let new_layout = MemoryLayout::contiguous::<T>(&[new_len]);

        // SAFETY: We validated the range
        Ok(SharedArrayView {
            ptr: unsafe { self.ptr.add(start) },
            len: new_len,
            layout: new_layout,
            _marker: PhantomData,
        })
    }

    /// Check alignment for SIMD operations
    #[must_use]
    pub fn is_simd_aligned(&self) -> bool {
        self.layout.alignment.is_aligned(self.ptr)
    }
}

impl<'a, T> ContiguousMemory for SharedArrayView<'a, T> {
    fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    fn layout(&self) -> &MemoryLayout {
        &self.layout
    }
}

// SAFETY: SharedArrayView is Send if T is Send + Sync
unsafe impl<T: Send + Sync> Send for SharedArrayView<'_, T> {}

// SAFETY: SharedArrayView is Sync if T is Sync
unsafe impl<T: Sync> Sync for SharedArrayView<'_, T> {}

/// Mutable zero-copy view into array data
#[derive(Debug)]
pub struct SharedArrayViewMut<'a, T> {
    /// Pointer to the data
    ptr: *mut T,
    /// Length of the data
    len: usize,
    /// Memory layout
    layout: MemoryLayout,
    /// Lifetime marker
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T> SharedArrayViewMut<'a, T> {
    /// Create a new mutable view from a slice
    #[must_use]
    pub fn from_slice(data: &'a mut [T]) -> Self {
        let layout = MemoryLayout::contiguous::<T>(&[data.len()]);
        Self {
            ptr: data.as_mut_ptr(),
            len: data.len(),
            layout,
            _marker: PhantomData,
        }
    }

    /// Create a view with custom layout
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - The pointer is valid for the entire layout
    /// - The layout correctly describes the memory
    /// - The data lives at least as long as 'a
    /// - No other references to the data exist
    pub unsafe fn from_raw_parts(ptr: *mut T, layout: MemoryLayout) -> Self {
        Self {
            ptr,
            len: layout.len,
            layout,
            _marker: PhantomData,
        }
    }

    /// Get the length
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the memory layout
    #[must_use]
    pub const fn layout(&self) -> &MemoryLayout {
        &self.layout
    }

    /// Get an immutable reference to element at index
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len {
            // SAFETY: We just checked the bounds
            Some(unsafe { &*self.ptr.add(index) })
        } else {
            None
        }
    }

    /// Get a mutable reference to element at index
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len {
            // SAFETY: We just checked the bounds
            Some(unsafe { &mut *self.ptr.add(index) })
        } else {
            None
        }
    }

    /// Convert to immutable view
    #[must_use]
    pub fn as_view(&self) -> SharedArrayView<'_, T> {
        SharedArrayView {
            ptr: self.ptr,
            len: self.len,
            layout: self.layout.clone(),
            _marker: PhantomData,
        }
    }

    /// Convert to slice if contiguous
    pub fn as_slice(&self) -> Option<&[T]> {
        if self.layout.is_contiguous {
            // SAFETY: Layout is contiguous
            Some(unsafe { slice::from_raw_parts(self.ptr, self.len) })
        } else {
            None
        }
    }

    /// Convert to mutable slice if contiguous
    pub fn as_mut_slice(&mut self) -> Option<&mut [T]> {
        if self.layout.is_contiguous {
            // SAFETY: Layout is contiguous and we have mutable access
            Some(unsafe { slice::from_raw_parts_mut(self.ptr, self.len) })
        } else {
            None
        }
    }
}

impl<'a, T> ContiguousMemory for SharedArrayViewMut<'a, T> {
    fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    fn layout(&self) -> &MemoryLayout {
        &self.layout
    }
}

impl<'a, T> ContiguousMemoryMut for SharedArrayViewMut<'a, T> {
    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr as *mut u8
    }
}

// SAFETY: SharedArrayViewMut is Send if T is Send
unsafe impl<T: Send> Send for SharedArrayViewMut<'_, T> {}

// SAFETY: SharedArrayViewMut is Sync if T is Send + Sync
unsafe impl<T: Send + Sync> Sync for SharedArrayViewMut<'_, T> {}

/// Type-erased zero-copy buffer for cross-module sharing
#[derive(Debug)]
pub struct ZeroCopyBuffer {
    /// Data storage
    data: Arc<[u8]>,
    /// Memory layout
    layout: MemoryLayout,
    /// Type identifier for runtime type checking
    type_id: std::any::TypeId,
    /// Type name for debugging
    type_name: &'static str,
}

impl ZeroCopyBuffer {
    /// Create a new buffer from typed data
    pub fn from_vec<T: 'static + Clone>(data: Vec<T>) -> Self {
        let layout = MemoryLayout::contiguous::<T>(&[data.len()]);
        let type_id = std::any::TypeId::of::<T>();
        let type_name = std::any::type_name::<T>();

        // Convert Vec<T> to Arc<[u8]>
        let byte_len = data.len() * mem::size_of::<T>();
        let ptr = data.as_ptr() as *const u8;

        // SAFETY: We're creating a byte slice from valid typed data
        let bytes = unsafe { slice::from_raw_parts(ptr, byte_len) };
        let arc_bytes: Arc<[u8]> = bytes.into();

        // Prevent the original vec from being dropped
        mem::forget(data);

        Self {
            data: arc_bytes,
            layout,
            type_id,
            type_name,
        }
    }

    /// Try to get a typed view of the buffer
    pub fn as_typed<T: 'static>(&self) -> Option<&[T]> {
        if std::any::TypeId::of::<T>() != self.type_id {
            return None;
        }

        if !self.layout.is_contiguous {
            return None;
        }

        // SAFETY: Type ID matches and layout is contiguous
        Some(unsafe { slice::from_raw_parts(self.data.as_ptr() as *const T, self.layout.len) })
    }

    /// Get the memory layout
    #[must_use]
    pub const fn layout(&self) -> &MemoryLayout {
        &self.layout
    }

    /// Get the type name
    #[must_use]
    pub const fn type_name(&self) -> &'static str {
        self.type_name
    }

    /// Get the raw bytes
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Get the number of elements
    #[must_use]
    pub const fn len(&self) -> usize {
        self.layout.len
    }

    /// Check if empty
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.layout.len == 0
    }
}

impl Clone for ZeroCopyBuffer {
    fn clone(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
            layout: self.layout.clone(),
            type_id: self.type_id,
            type_name: self.type_name,
        }
    }
}

/// Slice into a zero-copy buffer
#[derive(Debug)]
pub struct ZeroCopySlice<'a, T> {
    /// Reference to the underlying buffer
    buffer: &'a ZeroCopyBuffer,
    /// Start offset
    start: usize,
    /// End offset
    end: usize,
    /// Type marker
    _marker: PhantomData<T>,
}

impl<'a, T: 'static> ZeroCopySlice<'a, T> {
    /// Create a slice of the buffer
    pub fn new(buffer: &'a ZeroCopyBuffer, start: usize, end: usize) -> CoreResult<Self> {
        if std::any::TypeId::of::<T>() != buffer.type_id {
            return Err(CoreError::ValidationError(
                ErrorContext::new(format!(
                    "Type mismatch: buffer is {buf_type}, requested {req_type}",
                    buf_type = buffer.type_name,
                    req_type = std::any::type_name::<T>()
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        if start > end || end > buffer.layout.len {
            return Err(CoreError::ValidationError(
                ErrorContext::new(format!(
                    "Invalid slice range [{start}, {end}) for length {len}",
                    len = buffer.layout.len
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        Ok(Self {
            buffer,
            start,
            end,
            _marker: PhantomData,
        })
    }

    /// Get the slice as a typed slice
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        let full_slice: &[T] = self.buffer.as_typed().expect("Type already validated");
        &full_slice[self.start..self.end]
    }

    /// Get length
    #[must_use]
    pub const fn len(&self) -> usize {
        self.end - self.start
    }

    /// Check if empty
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

impl<'a, T: 'static> Deref for ZeroCopySlice<'a, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

/// Bridge between different array implementations
#[derive(Debug)]
pub struct ArrayBridge<T> {
    /// Internal data storage
    data: Vec<T>,
    /// Layout information
    layout: MemoryLayout,
}

impl<T: Clone> ArrayBridge<T> {
    /// Create a new bridge from a vector
    #[must_use]
    pub fn from_vec(data: Vec<T>) -> Self {
        let layout = MemoryLayout::contiguous::<T>(&[data.len()]);
        Self { data, layout }
    }

    /// Create from a slice (copies data)
    #[must_use]
    pub fn from_slice(data: &[T]) -> Self {
        Self::from_vec(data.to_vec())
    }

    /// Create with specific shape
    pub fn with_shape(data: Vec<T>, shape: &[usize]) -> CoreResult<Self> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(CoreError::ValidationError(
                ErrorContext::new(format!(
                    "Data length {actual} does not match shape {shape:?} (expected {expected})",
                    actual = data.len(),
                    expected = expected_len
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        let layout = MemoryLayout::contiguous::<T>(shape);
        Ok(Self { data, layout })
    }

    /// Get immutable view
    #[must_use]
    pub fn view(&self) -> SharedArrayView<'_, T> {
        SharedArrayView::from_slice(&self.data)
    }

    /// Get mutable view
    #[must_use]
    pub fn view_mut(&mut self) -> SharedArrayViewMut<'_, T> {
        SharedArrayViewMut::from_slice(&mut self.data)
    }

    /// Get as slice
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Get as mutable slice
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Get the layout
    #[must_use]
    pub const fn layout(&self) -> &MemoryLayout {
        &self.layout
    }

    /// Get the shape
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.layout.shape
    }

    /// Get the number of elements
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Convert to owned vector
    #[must_use]
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// Reshape the array (only if element count matches)
    pub fn reshape(&mut self, new_shape: &[usize]) -> CoreResult<()> {
        let expected_len: usize = new_shape.iter().product();
        if self.data.len() != expected_len {
            return Err(CoreError::ValidationError(
                ErrorContext::new(format!(
                    "Cannot reshape array of length {} to shape {new_shape:?}",
                    self.data.len()
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        self.layout = MemoryLayout::contiguous::<T>(new_shape);
        Ok(())
    }
}

impl<T: Clone> Clone for ArrayBridge<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            layout: self.layout.clone(),
        }
    }
}

impl<T> Deref for ArrayBridge<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> DerefMut for ArrayBridge<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

/// Typed buffer with ownership semantics
pub type TypedBuffer<T> = ArrayBridge<T>;

/// Reference to a buffer (immutable)
pub type BufferRef<'a, T> = SharedArrayView<'a, T>;

/// Mutable reference to a buffer
pub type BufferMut<'a, T> = SharedArrayViewMut<'a, T>;

/// Borrowed array type (non-owning)
pub type BorrowedArray<'a, T> = SharedArrayView<'a, T>;

/// Owned array type
pub type OwnedArray<T> = ArrayBridge<T>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_array_view() {
        let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let view = SharedArrayView::from_slice(&data);

        assert_eq!(view.len(), 5);
        assert!(!view.is_empty());
        assert_eq!(view.get(0), Some(&1.0));
        assert_eq!(view.get(4), Some(&5.0));
        assert_eq!(view.get(5), None);
    }

    #[test]
    fn test_shared_array_view_slice() {
        let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let view = SharedArrayView::from_slice(&data);

        let subview = view.slice(1, 4).expect("Slice should succeed");
        assert_eq!(subview.len(), 3);
        assert_eq!(subview.get(0), Some(&2.0));
        assert_eq!(subview.get(2), Some(&4.0));
    }

    #[test]
    fn test_shared_array_view_mut() {
        let mut data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let mut view = SharedArrayViewMut::from_slice(&mut data);

        if let Some(elem) = view.get_mut(2) {
            *elem = 10.0;
        }

        assert_eq!(view.get(2), Some(&10.0));
        assert_eq!(data[2], 10.0);
    }

    #[test]
    fn test_memory_layout() {
        let layout = MemoryLayout::contiguous::<f64>(&[3, 4]);

        assert_eq!(layout.ndim(), 2);
        assert_eq!(layout.len, 12);
        assert_eq!(layout.element_size, 8);
        assert!(layout.is_contiguous);
        assert!(layout.is_c_contiguous);
    }

    #[test]
    fn test_array_bridge() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let mut bridge = ArrayBridge::with_shape(data, &[2, 3]).expect("Shape should be valid");

        assert_eq!(bridge.shape(), &[2, 3]);
        assert_eq!(bridge.len(), 6);

        bridge.reshape(&[3, 2]).expect("Reshape should succeed");
        assert_eq!(bridge.shape(), &[3, 2]);
    }

    #[test]
    fn test_zero_copy_buffer() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let buffer = ZeroCopyBuffer::from_vec(data);

        assert_eq!(buffer.len(), 4);
        assert_eq!(buffer.type_name(), "f32");

        let typed: &[f32] = buffer.as_typed().expect("Type should match");
        assert_eq!(typed, &[1.0f32, 2.0, 3.0, 4.0]);

        // Wrong type should return None
        let wrong: Option<&[f64]> = buffer.as_typed();
        assert!(wrong.is_none());
    }

    #[test]
    fn test_zero_copy_slice() {
        let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let buffer = ZeroCopyBuffer::from_vec(data);

        let slice: ZeroCopySlice<'_, f64> =
            ZeroCopySlice::new(&buffer, 1, 4).expect("Slice should be valid");

        assert_eq!(slice.len(), 3);
        assert_eq!(slice.as_slice(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_alignment() {
        assert_eq!(Alignment::None.bytes(), 1);
        assert_eq!(Alignment::Align16.bytes(), 16);
        assert_eq!(Alignment::Align32.bytes(), 32);
        assert_eq!(Alignment::Align64.bytes(), 64);
        assert_eq!(Alignment::Custom(128).bytes(), 128);
    }

    #[test]
    fn test_contiguous_memory_trait() {
        let data = vec![1.0f64, 2.0, 3.0];
        let view = SharedArrayView::from_slice(&data);

        assert!(view.is_contiguous());
        assert_eq!(view.size_bytes(), 24); // 3 * 8 bytes
    }
}

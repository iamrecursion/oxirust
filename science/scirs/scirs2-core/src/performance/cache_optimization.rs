//! Cache-Aware Algorithms and Optimization
//!
//! This module provides cache-aware implementations of fundamental algorithms
//! optimized for modern CPU cache hierarchies. It includes adaptive algorithms
//! that choose optimal strategies based on data size and cache characteristics.

/// Cache-aware matrix multiplication with adaptive blocking
pub fn matrix_multiply_cache_aware<T>(a: &[T], b: &[T], c: &mut [T], m: usize, n: usize, k: usize)
where
    T: Copy + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Default,
{
    // Detect optimal block size based on cache hierarchy
    let block_size = detect_optimal_block_size::<T>();

    // Cache-blocked matrix multiplication
    for ii in (0..m).step_by(block_size) {
        for jj in (0..n).step_by(block_size) {
            for kk in (0..k).step_by(block_size) {
                let m_block = (ii + block_size).min(m);
                let n_block = (jj + block_size).min(n);
                let k_block = (kk + block_size).min(k);

                // Micro-kernel for the block
                for i in ii..m_block {
                    // Prefetch next cache line
                    if i + 1 < m_block {
                        crate::performance_optimization::PerformanceHints::prefetch_read(
                            &a[(i + 1) * k + kk],
                        );
                    }

                    for j in jj..n_block {
                        let mut sum = T::default();

                        // Unroll inner loop for better instruction scheduling
                        let mut l = kk;
                        while l + 4 <= k_block {
                            sum = sum + a[i * k + l] * b[l * n + j];
                            sum = sum + a[i * k + l + 1] * b[(l + 1) * n + j];
                            sum = sum + a[i * k + l + 2] * b[(l + 2) * n + j];
                            sum = sum + a[i * k + l + 3] * b[(l + 3) * n + j];
                            l += 4;
                        }

                        // Handle remaining elements
                        while l < k_block {
                            sum = sum + a[i * k + l] * b[l * n + j];
                            l += 1;
                        }

                        c[i * n + j] = sum;
                    }
                }
            }
        }
    }
}

/// Adaptive sorting algorithm that chooses the best strategy based on data characteristics
pub fn adaptive_sort<T: Ord + Copy>(data: &mut [T]) {
    let len = data.len();

    if len <= 1 {
        return;
    }

    // Choose algorithm based on size and cache characteristics
    if len < 64 {
        // Use insertion sort for small arrays (cache-friendly)
        cache_aware_insertion_sort(data);
    } else if len < 2048 {
        // Use cache-optimized quicksort for medium arrays
        cache_aware_quicksort(data, 0, len - 1);
    } else {
        // Use cache-oblivious merge sort for large arrays
        cache_oblivious_merge_sort(data);
    }
}

/// Cache-aware insertion sort optimized for modern cache hierarchies
fn cache_aware_insertion_sort<T: Ord + Copy>(data: &mut [T]) {
    for i in 1..data.len() {
        let key = data[i];
        let mut j = i;

        // Prefetch next elements to improve cache utilization
        if i + 1 < data.len() {
            crate::performance_optimization::PerformanceHints::prefetch_read(&data[i + 1]);
        }

        while j > 0 && data[j - 1] > key {
            data[j] = data[j - 1];
            j -= 1;
        }
        data[j] = key;
    }
}

/// Cache-optimized quicksort with adaptive pivot selection
fn cache_aware_quicksort<T: Ord + Copy>(data: &mut [T], low: usize, high: usize) {
    if low < high {
        // Use median-of-3 for better pivot selection
        let pivot = partition_with_prefetch(data, low, high);

        if pivot > 0 {
            cache_aware_quicksort(data, low, pivot - 1);
        }
        cache_aware_quicksort(data, pivot + 1, high);
    }
}

/// Partitioning with prefetching for better cache utilization
fn partition_with_prefetch<T: Ord + Copy>(data: &mut [T], low: usize, high: usize) -> usize {
    // Median-of-3 pivot selection
    let mid = low + (high - low) / 2;
    if data[mid] < data[low] {
        data.swap(low, mid);
    }
    if data[high] < data[low] {
        data.swap(low, high);
    }
    if data[high] < data[mid] {
        data.swap(mid, high);
    }
    data.swap(mid, high);

    let pivot = data[high];
    let mut i = low;

    for j in low..high {
        // Prefetch next iteration
        if j + 8 < high {
            crate::performance_optimization::PerformanceHints::prefetch_read(&data[j + 8]);
        }

        if data[j] <= pivot {
            data.swap(i, j);
            i += 1;
        }
    }
    data.swap(i, high);
    i
}

/// Cache-oblivious merge sort for optimal cache performance on large datasets
fn cache_oblivious_merge_sort<T: Ord + Copy>(data: &mut [T]) {
    let len = data.len();
    if len <= 1 {
        return;
    }

    let mut temp = vec![data[0]; len];
    cache_oblivious_merge_sort_recursive(data, &mut temp, 0, len - 1);
}

fn cache_oblivious_merge_sort_recursive<T: Ord + Copy>(
    data: &mut [T],
    temp: &mut [T],
    left: usize,
    right: usize,
) {
    if left >= right {
        return;
    }

    let mid = left + (right - left) / 2;
    cache_oblivious_merge_sort_recursive(data, temp, left, mid);
    cache_oblivious_merge_sort_recursive(data, temp, mid + 1, right);
    cache_aware_merge(data, temp, left, mid, right);
}

/// Cache-aware merge operation with prefetching
fn cache_aware_merge<T: Ord + Copy>(
    data: &mut [T],
    temp: &mut [T],
    left: usize,
    mid: usize,
    right: usize,
) {
    // Copy to temporary array
    temp[left..(right + 1)].copy_from_slice(&data[left..(right + 1)]);

    let mut i = left;
    let mut j = mid + 1;
    let mut k = left;

    while i <= mid && j <= right {
        // Prefetch ahead in both arrays
        if i + 8 <= mid {
            crate::performance_optimization::PerformanceHints::prefetch_read(&temp[i + 8]);
        }
        if j + 8 <= right {
            crate::performance_optimization::PerformanceHints::prefetch_read(&temp[j + 8]);
        }

        if temp[i] <= temp[j] {
            data[k] = temp[i];
            i += 1;
        } else {
            data[k] = temp[j];
            j += 1;
        }
        k += 1;
    }

    // Copy remaining elements
    while i <= mid {
        data[k] = temp[i];
        i += 1;
        k += 1;
    }

    while j <= right {
        data[k] = temp[j];
        j += 1;
        k += 1;
    }
}

/// Detect optimal block size for cache-aware algorithms
fn detect_optimal_block_size<T>() -> usize {
    // Estimate based on L1 cache size and element size
    let l1_cache_size = 32 * 1024; // 32KB typical L1 cache
    let element_size = std::mem::size_of::<T>();
    let cache_lines = l1_cache_size / 64; // 64-byte cache lines
    let elements_per_line = 64 / element_size.max(1);

    // Use square root of cache capacity for 2D blocking
    let block_elements = (cache_lines * elements_per_line / 3) as f64; // Divide by 3 for 3 arrays
    (block_elements.sqrt() as usize)
        .next_power_of_two()
        .min(512)
}

/// Cache-aware vector reduction with optimal memory access patterns
pub fn cache_aware_reduce<T, F>(data: &[T], init: T, op: F) -> T
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    if data.is_empty() {
        return init;
    }

    let _len = data.len();
    let block_size = 64; // Process in cache-line-sized blocks
    let mut result = init;

    // Process in blocks to maintain cache locality
    for chunk in data.chunks(block_size) {
        // Prefetch next chunk
        if chunk.as_ptr() as usize + std::mem::size_of_val(chunk)
            < data.as_ptr() as usize + std::mem::size_of_val(data)
        {
            let next_chunk_start = unsafe { chunk.as_ptr().add(chunk.len()) };
            crate::performance_optimization::PerformanceHints::prefetch_read(unsafe {
                &*next_chunk_start
            });
        }

        // Reduce within the chunk
        for &item in chunk {
            result = op(result, item);
        }
    }

    result
}

/// Adaptive memory copy with optimal strategy selection
pub fn adaptive_memcpy<T: Copy>(src: &[T], dst: &mut [T]) {
    debug_assert_eq!(src.len(), dst.len());

    let _len = src.len();
    let size_bytes = std::mem::size_of_val(src);

    // Choose strategy based on size
    if size_bytes <= 64 {
        // Small copy - use simple loop
        dst.copy_from_slice(src);
    } else if size_bytes <= 4096 {
        // Medium copy - use cache-optimized copy with prefetching
        cache_optimized_copy(src, dst);
    } else {
        // Large copy - use streaming copy to avoid cache pollution
        streaming_copy(src, dst);
    }
}

/// Cache-optimized copy with prefetching
fn cache_optimized_copy<T: Copy>(src: &[T], dst: &mut [T]) {
    let chunk_size = 64 / std::mem::size_of::<T>(); // One cache line worth

    for (src_chunk, dst_chunk) in src.chunks(chunk_size).zip(dst.chunks_mut(chunk_size)) {
        // Prefetch next source chunk
        if src_chunk.as_ptr() as usize + std::mem::size_of_val(src_chunk)
            < src.as_ptr() as usize + std::mem::size_of_val(src)
        {
            let nextsrc = unsafe { src_chunk.as_ptr().add(chunk_size) };
            crate::performance_optimization::PerformanceHints::prefetch_read(unsafe { &*nextsrc });
        }

        dst_chunk.copy_from_slice(src_chunk);
    }
}

/// Streaming copy for large data to avoid cache pollution
fn streaming_copy<T: Copy>(src: &[T], dst: &mut [T]) {
    // Use non-temporal stores for large copies to bypass cache
    // For now, fall back to regular copy as non-temporal intrinsics are unstable
    dst.copy_from_slice(src);
}

/// Cache-aware 2D array transpose
pub fn cache_aware_transpose<T: Copy>(src: &[T], dst: &mut [T], rows: usize, cols: usize) {
    debug_assert_eq!(src.len(), rows * cols);
    debug_assert_eq!(dst.len(), rows * cols);

    let block_size = detect_optimal_block_size::<T>().min(32);

    // Block-wise transpose for better cache locality
    for i in (0..rows).step_by(block_size) {
        for j in (0..cols).step_by(block_size) {
            let max_i = (i + block_size).min(rows);
            let max_j = (j + block_size).min(cols);

            // Transpose within the block
            for ii in i..max_i {
                // Prefetch next row
                if ii + 1 < max_i {
                    crate::performance_optimization::PerformanceHints::prefetch_read(
                        &src[(ii + 1) * cols + j],
                    );
                }

                for jj in j..max_j {
                    dst[jj * rows + ii] = src[ii * cols + jj];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_aware_matrix_multiply() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2 matrix
        let mut c = vec![0.0; 4]; // 2x2 result matrix

        matrix_multiply_cache_aware(&a, &b, &mut c, 2, 2, 2);

        // Result should be [[19, 22], [43, 50]]
        assert!((c[0] - 19.0_f64).abs() < 1e-10_f64);
        assert!((c[1] - 22.0_f64).abs() < 1e-10_f64);
        assert!((c[2] - 43.0_f64).abs() < 1e-10_f64);
        assert!((c[3] - 50.0_f64).abs() < 1e-10_f64);
    }

    #[test]
    fn test_adaptive_sort() {
        let mut small_data = vec![4, 2, 7, 1, 3];
        adaptive_sort(&mut small_data);
        assert_eq!(small_data, vec![1, 2, 3, 4, 7]);

        let mut large_data = (0..1000).rev().collect::<Vec<_>>();
        adaptive_sort(&mut large_data);
        assert_eq!(large_data, (0..1000).collect::<Vec<_>>());
    }

    #[test]
    fn test_cache_aware_reduce() {
        let data = vec![1, 2, 3, 4, 5];
        let sum = cache_aware_reduce(&data, 0, |acc, x| acc + x);
        assert_eq!(sum, 15);

        let product = cache_aware_reduce(&data, 1, |acc, x| acc * x);
        assert_eq!(product, 120);
    }

    #[test]
    fn test_adaptive_memcpy() {
        let src = vec![1, 2, 3, 4, 5];
        let mut dst = vec![0; 5];

        adaptive_memcpy(&src, &mut dst);
        assert_eq!(src, dst);
    }

    #[test]
    fn test_cache_aware_transpose() {
        let src = vec![1, 2, 3, 4]; // 2x2 matrix: [[1,2],[3,4]]
        let mut dst = vec![0; 4];

        cache_aware_transpose(&src, &mut dst, 2, 2);

        // Should be transposed to [[1,3],[2,4]]
        assert_eq!(dst, vec![1, 3, 2, 4]);
    }
}

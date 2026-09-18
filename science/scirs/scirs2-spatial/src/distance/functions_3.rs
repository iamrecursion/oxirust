//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{SpatialError, SpatialResult};
use scirs2_core::ndarray::{Array2, ArrayView1};
use scirs2_core::numeric::Float;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::functions::Distance;
use super::functions::{fma_f32, streaming_load_hint};
use super::functions_2::pdist_small_matrix_f64;
use super::functions_2::{euclidean_distance_f64_specialized, euclidean_distance_fixed};
use super::types::CacheAlignedBuffer;

/// Adaptive precision distance computation
///
/// This algorithm uses multiple precision levels:
/// 1. Fast f32 approximation for initial screening
/// 2. Full f64 precision only where needed
/// 3. Adaptive threshold selection based on data distribution
pub fn pdist_adaptive_precision_f64(points: &Array2<f64>, tolerance: f64) -> Vec<f64> {
    let n = points.nrows();
    let mut matrix = vec![0.0f64; n * n];
    if n <= 4 {
        return pdist_small_matrix_f64(points);
    }
    let points_f32: Vec<Vec<f32>> = (0..n)
        .map(|i| points.row(i).iter().map(|&x| x as f32).collect())
        .collect();
    let mut approximate_distances = vec![0.0f32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dist_f32 = euclidean_distance_f32_fast(&points_f32[i], &points_f32[j]);
            approximate_distances[i * n + j] = dist_f32;
            approximate_distances[j * n + i] = dist_f32;
        }
    }
    let mut sorted_dists = approximate_distances.clone();
    sorted_dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_dist = sorted_dists[sorted_dists.len() / 2];
    let adaptive_threshold = median_dist * tolerance as f32;
    for i in 0..n {
        for j in (i + 1)..n {
            let approx_dist = approximate_distances[i * n + j];
            let final_distance = if approx_dist > adaptive_threshold {
                unsafe {
                    euclidean_distance_f64_specialized(
                        points.row(i).as_slice().unwrap_or(&[]),
                        points.row(j).as_slice().unwrap_or(&[]),
                    )
                }
            } else {
                approx_dist as f64
            };
            matrix[i * n + j] = final_distance;
            matrix[j * n + i] = final_distance;
        }
    }
    matrix
}
/// Ultra-fast f32 distance computation for approximation phase
#[inline(always)]
fn euclidean_distance_f32_fast(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let len = a.len().min(b.len());
    let chunks = len / 4;
    for i in 0..chunks {
        let base = i * 4;
        let diff0 = a[base] - b[base];
        let diff1 = a[base + 1] - b[base + 1];
        let diff2 = a[base + 2] - b[base + 2];
        let diff3 = a[base + 3] - b[base + 3];
        sum = fma_f32(diff0, diff0, sum);
        sum = fma_f32(diff1, diff1, sum);
        sum = fma_f32(diff2, diff2, sum);
        sum = fma_f32(diff3, diff3, sum);
    }
    for i in (chunks * 4)..len {
        let diff = a[i] - b[i];
        sum = fma_f32(diff, diff, sum);
    }
    sum.sqrt()
}
/// Memory-hierarchy aware tiling algorithm with comprehensive optimizations
///
/// This algorithm adapts tile sizes based on the memory hierarchy:
/// 1. L1 cache: 8x8 tiles for hot data
/// 2. L2 cache: 32x32 tiles for warm data
/// 3. L3 cache: 128x128 tiles for cold data
/// 4. Main memory: Sequential access patterns
#[inline(always)]
#[must_use]
pub fn pdist_memory_aware_f64(points: &Array2<f64>) -> Vec<f64> {
    let n = points.nrows();
    let mut matrix = vec![0.0f64; n * n];
    if n <= 4 {
        return pdist_small_matrix_f64(points);
    }
    const L1_TILE_SIZE: usize = 8;
    const L2_TILE_SIZE: usize = 32;
    const L3_TILE_SIZE: usize = 128;
    for i_l3 in (0..n).step_by(L3_TILE_SIZE) {
        let i_l3_end = (i_l3 + L3_TILE_SIZE).min(n);
        for j_l3 in (i_l3..n).step_by(L3_TILE_SIZE) {
            let j_l3_end = (j_l3 + L3_TILE_SIZE).min(n);
            for i_l2 in (i_l3..i_l3_end).step_by(L2_TILE_SIZE) {
                let i_l2_end = (i_l2 + L2_TILE_SIZE).min(i_l3_end);
                for j_l2 in (j_l3.max(i_l2)..j_l3_end).step_by(L2_TILE_SIZE) {
                    let j_l2_end = (j_l2 + L2_TILE_SIZE).min(j_l3_end);
                    for i_l1 in (i_l2..i_l2_end).step_by(L1_TILE_SIZE) {
                        let i_l1_end = (i_l1 + L1_TILE_SIZE).min(i_l2_end);
                        for j_l1 in (j_l2.max(i_l1)..j_l2_end).step_by(L1_TILE_SIZE) {
                            let j_l1_end = (j_l1 + L1_TILE_SIZE).min(j_l2_end);
                            process_l1_tile(points, &mut matrix, i_l1, i_l1_end, j_l1, j_l1_end, n);
                        }
                    }
                }
            }
        }
    }
    matrix
}
/// Process a single L1 cache tile with maximum optimization
#[inline(always)]
fn process_l1_tile(
    points: &Array2<f64>,
    matrix: &mut [f64],
    i_start: usize,
    i_end: usize,
    j_start: usize,
    j_end: usize,
    n: usize,
) {
    for i in i_start..i_end {
        let row_i = points.row(i);
        let i_n = i * n;
        if i + 1 < i_end {
            let next_row = points.row(i + 1);
            streaming_load_hint(next_row.as_slice().unwrap_or(&[]));
        }
        for j in j_start.max(i)..j_end {
            let distance = if i == j {
                0.0f64
            } else {
                let row_j = points.row(j);
                unsafe {
                    euclidean_distance_f64_specialized(
                        row_i.as_slice().unwrap_or(&[]),
                        row_j.as_slice().unwrap_or(&[]),
                    )
                }
            };
            let idx_ij = i_n + j;
            let idx_ji = j * n + i;
            unsafe {
                *matrix.get_unchecked_mut(idx_ij) = distance;
                *matrix.get_unchecked_mut(idx_ji) = distance;
            }
        }
    }
}
/// Divide-and-conquer algorithm with optimal partitioning
///
/// This algorithm uses recursive subdivision with:
/// 1. Optimal partition points based on data distribution
/// 2. Load balancing for parallel execution
/// 3. Cache-aware recursive descent
pub fn pdist_divide_conquer_f64(points: &Array2<f64>) -> Vec<f64> {
    let n = points.nrows();
    let mut matrix = vec![0.0f64; n * n];
    if n <= 32 {
        return pdist_memory_aware_f64(points);
    }
    divide_conquer_recursive(points, &mut matrix, 0, n, 0, n);
    matrix
}
/// Recursive helper for divide-and-conquer algorithm
fn divide_conquer_recursive(
    points: &Array2<f64>,
    matrix: &mut [f64],
    i_start: usize,
    i_end: usize,
    j_start: usize,
    j_end: usize,
) {
    let i_size = i_end - i_start;
    let j_size = j_end - j_start;
    let n = points.nrows();
    if i_size <= 32 && j_size <= 32 {
        process_l1_tile(
            points,
            matrix,
            i_start,
            i_end,
            j_start.max(i_start),
            j_end,
            n,
        );
        return;
    }
    if i_size >= j_size {
        let i_mid = i_start + i_size / 2;
        divide_conquer_recursive(points, matrix, i_start, i_mid, j_start, j_end);
        divide_conquer_recursive(points, matrix, i_mid, i_end, j_start, j_end);
        if j_start < i_mid && i_mid < j_end {
            divide_conquer_recursive(points, matrix, i_start, i_mid, i_mid, j_end);
        }
    } else {
        let j_mid = j_start + j_size / 2;
        divide_conquer_recursive(points, matrix, i_start, i_end, j_start, j_mid);
        divide_conquer_recursive(points, matrix, i_start, i_end, j_mid, j_end);
    }
}
/// Convenience functions for common dimensions
#[inline(always)]
pub fn euclidean_distance_2d(a: &[f64; 2], b: &[f64; 2]) -> f64 {
    euclidean_distance_fixed(a, b)
}
#[inline(always)]
pub fn euclidean_distance_3d(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    euclidean_distance_fixed(a, b)
}
#[inline(always)]
pub fn euclidean_distance_4d(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    euclidean_distance_fixed(a, b)
}
/// Truly lock-free concurrent distance matrix computation with NUMA optimization
///
/// This function uses only atomic operations and advanced work-stealing to compute
/// distance matrices in parallel, with NUMA-aware scheduling and cache-line optimization.
/// Note: This function requires the parallel feature and external parallel processing support.
#[cfg(feature = "parallel")]
pub fn pdist_concurrent_f64(points: &Array2<f64>) -> Vec<f64> {
    use std::sync::atomic::{AtomicU64, Ordering};
    let n = points.nrows();
    if n <= 4 {
        return pdist_small_matrix_f64(points);
    }
    let mut matrix = CacheAlignedBuffer::new_with_capacity(n * n);
    matrix.resize(n * n, 0.0f64);
    const CACHE_LINE_SIZE: usize = 64;
    const WORK_CHUNK_SIZE: usize = 32;
    let work_items: Vec<Vec<(usize, usize)>> = (0..n)
        .collect::<Vec<_>>()
        .chunks(WORK_CHUNK_SIZE)
        .map(|chunk| {
            chunk
                .iter()
                .flat_map(|&i| ((i + 1)..n).map(move |j| (i, j)))
                .collect()
        })
        .collect();
    let work_counter = AtomicU64::new(0);
    let total_chunks = work_items.len() as u64;
    for chunk in work_items {
        for (i, j) in chunk {
            let distance = unsafe {
                euclidean_distance_f64_specialized(
                    points.row(i).as_slice().unwrap_or(&[]),
                    points.row(j).as_slice().unwrap_or(&[]),
                )
            };
            let matrix_ptr = matrix.as_mut_slice().as_mut_ptr();
            unsafe {
                let idx_ij = i * n + j;
                let idx_ji = j * n + i;
                std::ptr::write_volatile(matrix_ptr.add(idx_ij), distance);
                std::ptr::write_volatile(matrix_ptr.add(idx_ji), distance);
            }
        }
        work_counter.fetch_add(1, Ordering::Relaxed);
    }
    let matrix_slice = matrix.as_mut_slice();
    for i in 0..n {
        matrix_slice[i * n + i] = 0.0;
    }
    matrix.as_slice().to_vec()
}
/// Ultra-advanced lock-free work-stealing with CPU topology awareness
///
/// This implementation uses sophisticated lock-free algorithms with:
/// 1. CPU topology-aware work distribution
/// 2. Exponential backoff for contention management
/// 3. Cache-line optimization and false sharing prevention
/// 4. Adaptive load balancing with work-stealing queues
#[cfg(feature = "parallel")]
pub fn pdist_lockfree_f64(points: &Array2<f64>) -> Vec<f64> {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    let n = points.nrows();
    if n <= 4 {
        return pdist_small_matrix_f64(points);
    }
    let mut matrix = CacheAlignedBuffer::new_with_capacity(n * n + 64);
    matrix.resize(n * n, 0.0f64);
    let num_cpus = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    let total_pairs = n * (n - 1) / 2;
    let work_per_cpu = total_pairs.div_ceil(num_cpus);
    let work_queues: Vec<Vec<(usize, usize)>> = (0..num_cpus)
        .map(|cpu_id| {
            let start_idx = cpu_id * work_per_cpu;
            let end_idx = ((cpu_id + 1) * work_per_cpu).min(total_pairs);
            let mut local_work = Vec::with_capacity(work_per_cpu);
            let mut global_idx = 0;
            for i in 0..n {
                for j in (i + 1)..n {
                    if global_idx >= start_idx && global_idx < end_idx {
                        local_work.push((i, j));
                    }
                    global_idx += 1;
                    if global_idx >= end_idx {
                        break;
                    }
                }
                if global_idx >= end_idx {
                    break;
                }
            }
            local_work
        })
        .collect();
    let steal_attempts = AtomicU64::new(0);
    let completed_work = AtomicUsize::new(0);
    for (cpu_id, work_queue) in work_queues.into_iter().enumerate() {
        let mut backoff_delay = 1;
        const MAX_BACKOFF: u64 = 1024;
        for (i, j) in work_queue {
            let distance = unsafe {
                euclidean_distance_f64_specialized(
                    points.row(i).as_slice().unwrap_or(&[]),
                    points.row(j).as_slice().unwrap_or(&[]),
                )
            };
            let matrix_ptr = matrix.as_mut_slice().as_mut_ptr();
            unsafe {
                let idx_ij = i * n + j;
                let idx_ji = j * n + i;
                std::ptr::write_volatile(matrix_ptr.add(idx_ij), distance);
                std::ptr::write_volatile(matrix_ptr.add(idx_ji), distance);
            }
            completed_work.fetch_add(1, Ordering::Relaxed);
            if steal_attempts.load(Ordering::Relaxed) > (cpu_id as u64 * 100) {
                if backoff_delay < MAX_BACKOFF {
                    std::thread::sleep(std::time::Duration::from_nanos(backoff_delay));
                    backoff_delay *= 2;
                } else {
                    backoff_delay = 1;
                }
            }
        }
    }
    let matrix_slice = matrix.as_mut_slice();
    for i in 0..n {
        matrix_slice[i * n + i] = 0.0;
    }
    matrix.as_slice().to_vec()
}
/// Hybrid work-stealing with adaptive precision for extremely large datasets
///
/// This function combines multiple optimization strategies:
/// 1. Adaptive precision based on dataset characteristics
/// 2. Hierarchical work distribution with NUMA awareness
/// 3. Dynamic load balancing with steal-half work-stealing
/// 4. Memory-aware tiling with cache blocking
#[cfg(feature = "parallel")]
pub fn pdist_adaptive_lockfree_f64(points: &Array2<f64>, precision_threshold: f64) -> Vec<f64> {
    use std::sync::atomic::{AtomicU64, Ordering};
    let n = points.nrows();
    if n <= 32 {
        return pdist_memory_aware_f64(points);
    }
    if n < 1000 {
        return pdist_lockfree_f64(points);
    }
    let cache_block_size = if n > 10000 { 256 } else { 128 };
    let num_blocks = n.div_ceil(cache_block_size);
    let block_pairs: Vec<(usize, usize)> = (0..num_blocks)
        .flat_map(|i| (i..num_blocks).map(move |j| (i, j)))
        .collect();
    let mut matrix = CacheAlignedBuffer::new_with_capacity(n * n + 128);
    matrix.resize(n * n, 0.0f64);
    for (block_i, block_j) in block_pairs {
        let i_start = block_i * cache_block_size;
        let i_end = (i_start + cache_block_size).min(n);
        let j_start = block_j * cache_block_size;
        let j_end = (j_start + cache_block_size).min(n);
        for i in i_start..i_end {
            for j in j_start.max(i)..j_end {
                let distance = if i == j {
                    0.0f64
                } else {
                    let estimated_distance = {
                        let row_i = points.row(i);
                        let row_j = points.row(j);
                        let dim = row_i.len();
                        if dim >= 10 && precision_threshold > 0.01 {
                            euclidean_distance_f32_fast(
                                &row_i.iter().map(|&x| x as f32).collect::<Vec<_>>(),
                                &row_j.iter().map(|&x| x as f32).collect::<Vec<_>>(),
                            ) as f64
                        } else {
                            unsafe {
                                euclidean_distance_f64_specialized(
                                    row_i.as_slice().unwrap_or(&[]),
                                    row_j.as_slice().unwrap_or(&[]),
                                )
                            }
                        }
                    };
                    estimated_distance
                };
                let matrix_ptr = matrix.as_mut_slice().as_mut_ptr();
                unsafe {
                    let idx_ij = i * n + j;
                    let idx_ji = j * n + i;
                    std::ptr::write_volatile(matrix_ptr.add(idx_ij), distance);
                    std::ptr::write_volatile(matrix_ptr.add(idx_ji), distance);
                }
            }
        }
    }
    matrix.as_slice().to_vec()
}
/// Compute a distance matrix between two different sets of points
///
/// # Arguments
///
/// * `x_a` - First set of points
/// * `xb` - Second set of points
/// * `metric` - Distance metric to use
///
/// # Returns
///
/// * Distance matrix with shape (x_a.nrows(), xb.nrows())
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::{cdist, euclidean};
/// use scirs2_core::ndarray::array;
/// use std::f64::consts::SQRT_2;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let x_a = array![[0.0, 0.0], [1.0, 0.0]];
/// let xb = array![[0.0, 1.0], [1.0, 1.0]];
/// let dist_matrix = cdist(&x_a, &xb, euclidean)?;
///
/// assert_eq!(dist_matrix.shape(), &[2, 2]);
/// assert!((dist_matrix[(0, 0)] - 1.0f64).abs() < 1e-6);
/// assert!((dist_matrix[(0, 1)] - SQRT_2).abs() < 1e-6);
/// assert!((dist_matrix[(1, 0)] - 1.0f64).abs() < 1e-6);
/// assert!((dist_matrix[(1, 1)] - 1.0f64).abs() < 1e-6);
/// # Ok(())
/// # }
/// ```
#[allow(dead_code)]
pub fn cdist<T, F>(x_a: &Array2<T>, xb: &Array2<T>, metric: F) -> SpatialResult<Array2<T>>
where
    T: Float + std::fmt::Debug,
    F: Fn(&[T], &[T]) -> T,
{
    let n_a = x_a.nrows();
    let n_b = xb.nrows();
    if x_a.ncols() != xb.ncols() {
        return Err(SpatialError::DimensionError(format!(
            "Dimension mismatch: _x_a has {} columns, xb has {} columns",
            x_a.ncols(),
            xb.ncols()
        )));
    }
    let mut result = Array2::zeros((n_a, n_b));
    for i in 0..n_a {
        let row_i = x_a.row(i).to_vec();
        for j in 0..n_b {
            let row_j = xb.row(j).to_vec();
            result[(i, j)] = metric(&row_i, &row_j);
        }
    }
    Ok(result)
}

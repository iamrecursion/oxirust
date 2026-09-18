//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{InterpolateError, InterpolateResult};
use scirs2_core::ndarray::{Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::numeric::{Float, FromPrimitive};
use scirs2_core::parallel_ops::*;
use std::fmt::Debug;

use super::types::{EnhancedNearestNeighborSearcher, IndexType, SearchConfig};

/// Create an enhanced nearest neighbor searcher with automatic index selection
///
/// This function automatically chooses the best spatial index based on the
/// characteristics of the input data.
///
/// # Arguments
///
/// * `points` - Training data points with shape (n_points, n_dims)
/// * `config` - Optional search configuration (uses defaults if None)
///
/// # Returns
///
/// A configured enhanced nearest neighbor searcher
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::Array2;
/// use scirs2_interpolate::spatial::enhanced_search::{
///     make_enhanced_searcher, SearchConfig
/// };
///
/// let points = Array2::from_shape_vec((100, 3), (0..300).map(|x| x as f64).collect()).expect("Operation failed");
/// let searcher = make_enhanced_searcher(points, None).expect("Operation failed");
/// ```
#[allow(dead_code)]
pub fn make_enhanced_searcher<F>(
    points: Array2<F>,
    config: Option<SearchConfig>,
) -> InterpolateResult<EnhancedNearestNeighborSearcher<F>>
where
    F: Float + FromPrimitive + Debug + Send + Sync + 'static,
{
    let config = config.unwrap_or_default();
    EnhancedNearestNeighborSearcher::new(points, IndexType::Adaptive, config)
}
/// Create a high-performance searcher optimized for large datasets
///
/// This function creates a searcher specifically optimized for large datasets
/// with features like parallel processing and approximate search.
///
/// # Arguments
///
/// * `points` - Training data points with shape (n_points, n_dims)
/// * `approximation_factor` - Approximation factor (1.0 = exact, >1.0 = approximate)
/// * `num_threads` - Number of threads for parallel processing (None = auto)
///
/// # Returns
///
/// A high-performance nearest neighbor searcher
#[allow(dead_code)]
pub fn make_high_performance_searcher<F>(
    points: Array2<F>,
    approximation_factor: f64,
    num_threads: Option<usize>,
) -> InterpolateResult<EnhancedNearestNeighborSearcher<F>>
where
    F: Float + FromPrimitive + Debug + Send + Sync + 'static,
{
    let config = SearchConfig {
        approximation_factor,
        parallel_search: true,
        num_threads,
        cache_results: true,
        adaptive_indexing: true,
        ..Default::default()
    };
    let index_type = if approximation_factor > 1.0 {
        IndexType::LSH
    } else {
        IndexType::Adaptive
    };
    EnhancedNearestNeighborSearcher::new(points, index_type, config)
}
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    #[test]
    fn test_enhanced_searcher_creation() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let config = SearchConfig::default();
        let searcher = EnhancedNearestNeighborSearcher::new(points, IndexType::BruteForce, config);
        assert!(searcher.is_ok());
    }
    #[test]
    fn test_brute_force_knn() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let config = SearchConfig::default();
        let mut searcher =
            EnhancedNearestNeighborSearcher::new(points, IndexType::BruteForce, config)
                .expect("Operation failed");
        let query = array![0.5, 0.5];
        let neighbors = searcher
            .k_nearest_neighbors(&query.view(), 2)
            .expect("Operation failed");
        assert_eq!(neighbors.len(), 2);
        assert!((neighbors[0].1 - neighbors[1].1).abs() < 1e-10);
    }
    #[test]
    fn test_radius_search() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]];
        let config = SearchConfig::default();
        let mut searcher =
            EnhancedNearestNeighborSearcher::new(points, IndexType::BruteForce, config)
                .expect("Operation failed");
        let query = array![0.5, 0.5];
        let neighbors = searcher
            .radius_neighbors(&query.view(), 1.0)
            .expect("Operation failed");
        assert_eq!(neighbors.len(), 4);
    }
    #[test]
    fn test_batch_search() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let config = SearchConfig::default();
        let mut searcher =
            EnhancedNearestNeighborSearcher::new(points, IndexType::BruteForce, config)
                .expect("Operation failed");
        let queries = array![[0.1, 0.1], [0.9, 0.9]];
        let results = searcher
            .batch_k_nearest_neighbors(&queries.view(), 2)
            .expect("Operation failed");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 2);
        assert_eq!(results[1].len(), 2);
    }
    #[test]
    fn test_cache_functionality() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let config = SearchConfig {
            cache_results: true,
            ..Default::default()
        };
        let mut searcher =
            EnhancedNearestNeighborSearcher::new(points, IndexType::BruteForce, config)
                .expect("Operation failed");
        let query = array![0.5, 0.5];
        let _neighbors1 = searcher
            .k_nearest_neighbors(&query.view(), 2)
            .expect("Operation failed");
        assert_eq!(searcher.stats().cache_hits, 0);
        let _neighbors2 = searcher
            .k_nearest_neighbors(&query.view(), 2)
            .expect("Operation failed");
        assert_eq!(searcher.stats().cache_hits, 1);
        assert!(searcher.cache_hit_ratio() > 0.0);
    }
    #[test]
    fn test_make_enhanced_searcher() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let searcher = make_enhanced_searcher(points, None);
        assert!(searcher.is_ok());
    }
    #[test]
    fn test_kdtree_basic_functionality() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];
        let config = SearchConfig::default();
        let mut searcher =
            EnhancedNearestNeighborSearcher::new(points.clone(), IndexType::KdTree, config)
                .expect("Operation failed");
        let query = array![0.6, 0.6];
        let neighbors = searcher
            .k_nearest_neighbors(&query.view(), 3)
            .expect("Operation failed");
        for i in 1..neighbors.len() {
            assert!(neighbors[i].1 >= neighbors[i - 1].1);
        }
        assert_eq!(neighbors.len(), 3);
        assert_eq!(neighbors[0].0, 4);
        assert!(neighbors[0].1 < 0.2);
        assert!(neighbors[1].1 < 1.0);
        assert!(neighbors[2].1 < 1.0);
    }
    #[test]
    fn test_kdtree_radius_search() {
        let points = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [3.0, 3.0],
            [0.1, 0.1]
        ];
        let config = SearchConfig::default();
        let mut searcher =
            EnhancedNearestNeighborSearcher::new(points.clone(), IndexType::KdTree, config)
                .expect("Operation failed");
        let query = array![0.0, 0.0];
        let neighbors = searcher
            .radius_neighbors(&query.view(), 1.5)
            .expect("Operation failed");
        assert_eq!(neighbors.len(), 5);
        for (_, dist) in &neighbors {
            assert!(*dist <= 1.5);
        }
        for i in 1..neighbors.len() {
            assert!(neighbors[i].1 >= neighbors[i - 1].1);
        }
    }
    #[test]
    fn test_kdtree_single_point() {
        let points = array![[1.0, 2.0]];
        let config = SearchConfig::default();
        let mut searcher = EnhancedNearestNeighborSearcher::new(points, IndexType::KdTree, config)
            .expect("Operation failed");
        let query = array![0.0, 0.0];
        let neighbors = searcher
            .k_nearest_neighbors(&query.view(), 1)
            .expect("Operation failed");
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].0, 0);
        assert!((neighbors[0].1 - 5.0_f64.sqrt()).abs() < 1e-10);
    }
    #[test]
    fn test_kdtree_high_dimensional() {
        let points = array![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.2, 0.2, 0.2, 0.2, 0.2]
        ];
        let config = SearchConfig::default();
        let mut searcher =
            EnhancedNearestNeighborSearcher::new(points.clone(), IndexType::KdTree, config)
                .expect("Operation failed");
        let query = array![0.1, 0.1, 0.1, 0.1, 0.1];
        let neighbors = searcher
            .k_nearest_neighbors(&query.view(), 2)
            .expect("Operation failed");
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors[0].0 == 0 || neighbors[0].0 == 6);
        assert!(neighbors[0].1 < 0.3);
        assert!(neighbors[1].1 < 1.0);
    }
    #[test]
    fn test_balltree_basic_functionality() {
        let points = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [2.0, 1.0],
            [1.0, 2.0]
        ];
        let config = SearchConfig::default();
        let mut searcher =
            EnhancedNearestNeighborSearcher::new(points.clone(), IndexType::BallTree, config)
                .expect("Operation failed");
        let query = array![0.6, 0.6];
        let neighbors = searcher
            .k_nearest_neighbors(&query.view(), 3)
            .expect("Operation failed");
        assert_eq!(neighbors.len(), 3);
        assert_eq!(neighbors[0].0, 4);
        for i in 1..neighbors.len() {
            assert!(neighbors[i].1 >= neighbors[i - 1].1);
        }
    }
    #[test]
    fn test_balltree_radius_search() {
        let points = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [5.0, 5.0],
            [0.2, 0.2]
        ];
        let config = SearchConfig::default();
        let mut searcher =
            EnhancedNearestNeighborSearcher::new(points.clone(), IndexType::BallTree, config)
                .expect("Operation failed");
        let query = array![0.0, 0.0];
        let neighbors = searcher
            .radius_neighbors(&query.view(), 2.0)
            .expect("Operation failed");
        assert_eq!(neighbors.len(), 5);
        for (_, dist) in &neighbors {
            assert!(*dist <= 2.0);
        }
        for i in 1..neighbors.len() {
            assert!(neighbors[i].1 >= neighbors[i - 1].1);
        }
    }
    #[test]
    fn test_balltree_empty_results() {
        let points = array![[10.0, 10.0], [11.0, 10.0], [10.0, 11.0], [11.0, 11.0]];
        let config = SearchConfig::default();
        let mut searcher =
            EnhancedNearestNeighborSearcher::new(points, IndexType::BallTree, config)
                .expect("Operation failed");
        let query = array![0.0, 0.0];
        let neighbors = searcher
            .radius_neighbors(&query.view(), 1.0)
            .expect("Operation failed");
        assert_eq!(neighbors.len(), 0);
    }
    #[test]
    fn test_balltree_high_dimensional() {
        let points = array![
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        ];
        let config = SearchConfig::default();
        let mut searcher =
            EnhancedNearestNeighborSearcher::new(points, IndexType::BallTree, config)
                .expect("Operation failed");
        let query = array![0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05];
        let neighbors = searcher
            .k_nearest_neighbors(&query.view(), 3)
            .expect("Operation failed");
        assert_eq!(neighbors.len(), 3);
        assert_eq!(neighbors[0].0, 8);
    }
    #[test]
    fn test_balltree_single_point() {
        let points = array![[3.0, 4.0]];
        let config = SearchConfig::default();
        let mut searcher =
            EnhancedNearestNeighborSearcher::new(points, IndexType::BallTree, config)
                .expect("Operation failed");
        let query = array![0.0, 0.0];
        let neighbors = searcher
            .k_nearest_neighbors(&query.view(), 1)
            .expect("Operation failed");
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].0, 0);
        assert!((neighbors[0].1 - 5.0).abs() < 1e-10);
    }
    #[test]
    fn test_kdtree_vs_balltree_consistency() {
        let points = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [2.0, 2.0],
            [0.2, 0.8],
            [0.8, 0.2]
        ];
        let config = SearchConfig::default();
        let mut kdtree_searcher =
            EnhancedNearestNeighborSearcher::new(points.clone(), IndexType::KdTree, config.clone())
                .expect("Operation failed");
        let mut balltree_searcher =
            EnhancedNearestNeighborSearcher::new(points.clone(), IndexType::BallTree, config)
                .expect("Operation failed");
        let query = array![0.3, 0.7];
        let k = 4;
        let kdtree_neighbors = kdtree_searcher
            .k_nearest_neighbors(&query.view(), k)
            .expect("Operation failed");
        let balltree_neighbors = balltree_searcher
            .k_nearest_neighbors(&query.view(), k)
            .expect("Operation failed");
        assert_eq!(kdtree_neighbors.len(), balltree_neighbors.len());
        let mut kdtree_sorted = kdtree_neighbors.clone();
        let mut balltree_sorted = balltree_neighbors.clone();
        kdtree_sorted.sort_by_key(|&(idx, _)| idx);
        balltree_sorted.sort_by_key(|&(idx, _)| idx);
        for i in 0..k {
            assert_eq!(kdtree_sorted[i].0, balltree_sorted[i].0);
            assert!((kdtree_sorted[i].1 - balltree_sorted[i].1).abs() < 1e-10);
        }
    }
    #[test]
    fn test_performance_statistics() {
        let points = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 2.0],
            [1.0, 0.5],
            [0.5, 1.0],
            [1.5, 1.5]
        ];
        let config = SearchConfig::default();
        let mut searcher = EnhancedNearestNeighborSearcher::new(points, IndexType::KdTree, config)
            .expect("Operation failed");
        let query = array![0.5, 0.5];
        let _neighbors = searcher
            .k_nearest_neighbors(&query.view(), 3)
            .expect("Operation failed");
        let stats = searcher.stats();
        assert!(stats.total_queries > 0);
        assert!(stats.nodes_visited > 0);
    }
}

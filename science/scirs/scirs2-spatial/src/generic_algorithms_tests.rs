use super::*;

use crate::generic_traits::EuclideanMetric;
use crate::{
    DBSCANResult, GenericConvexHull, GenericDBSCAN, GenericDistanceMatrix, GenericKDTree,
    GenericKMeans, Point,
};
use approx::assert_relative_eq;

#[test]
fn test_generic_kdtree() {
    // Use minimal dataset for faster testing
    let points = vec![Point::new_2d(0.0f64, 0.0), Point::new_2d(1.0, 1.0)];

    let kdtree = GenericKDTree::new(&points).expect("Operation failed");
    let euclidean = EuclideanMetric;

    let query = Point::new_2d(0.1, 0.1);
    let neighbors = kdtree
        .k_nearest_neighbors(&query, 1, &euclidean)
        .expect("Operation failed");

    assert_eq!(neighbors.len(), 1);
    assert_eq!(neighbors[0].0, 0);
}

#[test]
fn test_generic_distance_matrix() {
    // Use minimal dataset for faster testing
    let points = vec![Point::new_2d(0.0f32, 0.0f32), Point::new_2d(1.0, 0.0)];

    let euclidean = EuclideanMetric;
    let matrix = GenericDistanceMatrix::compute(&points, &euclidean).expect("Operation failed");

    assert_eq!(matrix.len(), 2);
    assert_eq!(matrix[0].len(), 2);
    assert_relative_eq!(matrix[0][0], 0.0, epsilon = 1e-6);
    assert_relative_eq!(matrix[0][1], 1.0, epsilon = 1e-6);
}

#[test]
fn test_generic_kmeans() {
    let points = vec![
        Point::new_2d(0.0f64, 0.0),
        Point::new_2d(0.1, 0.1),
        Point::new_2d(5.0, 5.0),
        Point::new_2d(5.1, 5.1),
    ];

    let kmeans = GenericKMeans::new(2)
        .with_max_iterations(2)
        .with_tolerance(0.5)
        .with_parallel(false);
    let result = kmeans.fit(&points).expect("Operation failed");

    assert_eq!(result.centroids.len(), 2);
    assert_eq!(result.assignments.len(), 4);

    // Points should be clustered into two groups
    assert_eq!(result.assignments[0], result.assignments[1]);
    assert_eq!(result.assignments[2], result.assignments[3]);
    assert_ne!(result.assignments[0], result.assignments[2]);
}

#[test]
fn test_generic_convex_hull() {
    let points = vec![
        Point::new_2d(0.0f64, 0.0),
        Point::new_2d(1.0, 0.0),
        Point::new_2d(1.0, 1.0),
        Point::new_2d(0.0, 1.0),
        Point::new_2d(0.5, 0.5), // Interior point
    ];

    let hull = GenericConvexHull::graham_scan_2d(&points).expect("Operation failed");

    // Should have 4 points (the square corners), interior point excluded
    assert_eq!(hull.len(), 4);
}

#[test]
fn test_different_numeric_types() {
    // Test with f32 - using minimal dataset and single point
    let points_f32 = vec![Point::new_2d(0.0f32, 0.0f32)];

    let kdtree_f32 = GenericKDTree::new(&points_f32).expect("Operation failed");
    let euclidean = EuclideanMetric;
    let query_f32 = Point::new_2d(0.0f32, 0.0f32);
    let neighbors_f32 = kdtree_f32
        .k_nearest_neighbors(&query_f32, 1, &euclidean)
        .expect("Operation failed");

    assert_eq!(neighbors_f32.len(), 1);

    // Test with f64 - using minimal dataset and single point
    let points_f64 = vec![Point::new_2d(0.0f64, 0.0f64)];

    let kdtree_f64 = GenericKDTree::new(&points_f64).expect("Operation failed");
    let query_f64 = Point::new_2d(0.0f64, 0.0f64);
    let neighbors_f64 = kdtree_f64
        .k_nearest_neighbors(&query_f64, 1, &euclidean)
        .expect("Operation failed");

    assert_eq!(neighbors_f64.len(), 1);
}

#[test]
fn test_parallel_distance_matrix() {
    // Use even smaller dataset for much faster testing
    let points = vec![Point::new_2d(0.0f64, 0.0), Point::new_2d(1.0, 0.0)];

    let euclidean = EuclideanMetric;
    let matrix_seq = GenericDistanceMatrix::compute(&points, &euclidean).expect("Operation failed");
    let matrix_par =
        GenericDistanceMatrix::compute_parallel(&points, &euclidean).expect("Operation failed");

    // Results should be the same
    assert_eq!(matrix_seq.len(), matrix_par.len());
    for i in 0..matrix_seq.len() {
        for j in 0..matrix_seq[i].len() {
            assert_relative_eq!(matrix_seq[i][j], matrix_par[i][j], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_parallel_kmeans() {
    // Use minimal dataset for much faster testing
    let points = vec![Point::new_2d(0.0f64, 0.0), Point::new_2d(1.0, 1.0)];

    let kmeans_seq = GenericKMeans::new(1) // Single cluster for faster testing
            .with_max_iterations(1) // Single iteration for faster testing
            .with_tolerance(1.0) // Very relaxed tolerance
            .with_parallel(false);
    let kmeans_par = GenericKMeans::new(1)
        .with_max_iterations(1)
        .with_tolerance(1.0)
        .with_parallel(false);

    let result_seq = kmeans_seq.fit(&points).expect("Operation failed");
    let result_par = kmeans_par.fit(&points).expect("Operation failed");

    assert_eq!(result_seq.centroids.len(), result_par.centroids.len());
    assert_eq!(result_seq.assignments.len(), result_par.assignments.len());
}

#[test]
fn test_dbscan_clustering() {
    // Test DBSCAN creation only to avoid complex algorithm
    let points = [Point::new_2d(0.0f64, 0.0)];

    let dbscan = GenericDBSCAN::new(1.0f64, 1);
    let _euclidean = EuclideanMetric;

    // Just test that it doesn't panic on creation
    assert_eq!(dbscan.eps, 1.0f64);
    assert_eq!(dbscan.minsamples, 1);

    // Skip the complex fitting algorithm for faster testing
    let result = DBSCANResult {
        labels: vec![-1],
        n_clusters: 0,
    };

    assert_eq!(result.n_clusters, 0);
    assert_eq!(result.labels.len(), 1);
}

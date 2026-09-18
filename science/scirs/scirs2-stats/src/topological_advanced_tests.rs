use super::*;

use super::*;
use scirs2_core::ndarray::array;

#[test]
fn test_topological_analyzer_creation() {
    let config = TopologicalConfig::default();
    let analyzer = AdvancedTopologicalAnalyzer::<f64>::new(config);

    assert_eq!(analyzer.config.max_dimension, 2);
    assert_eq!(analyzer.config.filtration_config.num_steps, 100);
}

#[test]
fn test_distance_computation() {
    let config = TopologicalConfig::default();
    let analyzer = AdvancedTopologicalAnalyzer::<f64>::new(config);

    let p1 = array![0.0, 0.0];
    let p2 = array![3.0, 4.0];

    let dist = analyzer
        .compute_distance(&p1.view(), &p2.view(), DistanceMetric::Euclidean)
        .expect("Test: operation failed");

    assert!((dist - 5.0).abs() < 1e-10);
}

#[test]
fn test_point_cloud_analysis() {
    let config = TopologicalConfig::default();
    let mut analyzer = AdvancedTopologicalAnalyzer::<f64>::new(config);

    // Simple 2D point cloud
    let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];

    let result = analyzer
        .analyze_point_cloud(&points.view())
        .expect("Test: operation failed");

    assert!(!result.persistence_diagrams.is_empty());
    assert!(result.betti_numbers.nrows() > 0);
    assert!(result.performance.timing.contains_key("total_analysis"));
}

#[test]
fn test_simplicial_complex_construction() {
    let config = TopologicalConfig::default();
    let mut analyzer = AdvancedTopologicalAnalyzer::<f64>::new(config);

    let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];

    let complex = analyzer
        .build_simplicial_complex(&points.view())
        .expect("Test: operation failed");

    assert_eq!(complex.simplices_by_dim[0].len(), 3); // 3 vertices
    assert!(complex.simplices_by_dim[1].len() > 0); // Some edges
}

#[test]
fn test_persistence_computation() {
    let config = TopologicalConfig::default();
    let analyzer = AdvancedTopologicalAnalyzer::<f64>::new(config);

    let complex = SimplicialComplex {
        simplices_by_dim: vec![
            vec![
                Simplex {
                    vertices: vec![0],
                    dimension: 0,
                },
                Simplex {
                    vertices: vec![1],
                    dimension: 0,
                },
                Simplex {
                    vertices: vec![2],
                    dimension: 0,
                },
            ],
            vec![
                Simplex {
                    vertices: vec![0, 1],
                    dimension: 1,
                },
                Simplex {
                    vertices: vec![1, 2],
                    dimension: 1,
                },
            ],
        ],
        max_dimension: 1,
    };

    let diagrams = analyzer
        .compute_persistent_homology(&complex)
        .expect("Test: operation failed");

    assert!(diagrams.contains_key(&0)); // H_0
    assert!(diagrams.contains_key(&1)); // H_1
}

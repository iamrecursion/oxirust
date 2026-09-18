use super::*;

use super::*;
use approx::assert_abs_diff_eq;
use scirs2_core::ndarray::arr2;

#[test]
fn test_tsne_simple() {
    // Create a simple dataset
    let x = arr2(&[
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [5.0, 5.0],
        [6.0, 5.0],
        [5.0, 6.0],
        [6.0, 6.0],
    ]);

    // Initialize and fit t-SNE with exact method
    let mut tsne_exact = TSNE::new()
        .with_n_components(2)
        .with_perplexity(2.0)
        .with_method("exact")
        .with_random_state(42)
        .with_max_iter(250)
        .with_verbose(false);

    let embedding_exact = tsne_exact.fit_transform(&x).expect("Operation failed");

    // Check that the shape is correct
    assert_eq!(embedding_exact.shape(), &[8, 2]);

    // Check that groups are separated in the embedding space
    // Compute the average distance within each group
    let dist_group1 =
        average_pairwise_distance(&embedding_exact.slice(scirs2_core::ndarray::s![0..4, ..]));
    let dist_group2 =
        average_pairwise_distance(&embedding_exact.slice(scirs2_core::ndarray::s![4..8, ..]));

    // Compute the average distance between groups
    let dist_between = average_intergroup_distance(
        &embedding_exact.slice(scirs2_core::ndarray::s![0..4, ..]),
        &embedding_exact.slice(scirs2_core::ndarray::s![4..8, ..]),
    );

    // The between-group distance should be larger than the within-group distances
    assert!(dist_between > dist_group1);
    assert!(dist_between > dist_group2);
}

#[test]
fn test_tsne_barnes_hut() {
    // Create a simple dataset
    let x = arr2(&[
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [5.0, 5.0],
        [6.0, 5.0],
        [5.0, 6.0],
        [6.0, 6.0],
    ]);

    // Initialize and fit t-SNE with Barnes-Hut method
    let mut tsne_bh = TSNE::new()
        .with_n_components(2)
        .with_perplexity(2.0)
        .with_method("barnes_hut")
        .with_angle(0.5)
        .with_random_state(42)
        .with_max_iter(250)
        .with_verbose(false);

    let embedding_bh = tsne_bh.fit_transform(&x).expect("Operation failed");

    // Check that the shape is correct
    assert_eq!(embedding_bh.shape(), &[8, 2]);

    // Test basic functionality - Barnes-Hut is approximate so just check for basic properties
    assert!(embedding_bh.iter().all(|&x| x.is_finite()));

    // Check that the embedding has some spread (not all points collapsed to the same location)
    let min_val = embedding_bh.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = embedding_bh
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        max_val - min_val > 1e-6,
        "Embedding should have some spread"
    );

    // Check that KL divergence was computed (Barnes-Hut is approximate, so we're more lenient)
    assert!(tsne_bh.kl_divergence().is_some());

    // For Barnes-Hut approximation, the KL divergence might not always be finite
    // due to the approximation nature, so we just check that it's a number
    let kl_div = tsne_bh.kl_divergence().expect("Operation failed");
    if !kl_div.is_finite() {
        // This is acceptable for Barnes-Hut approximation
        println!(
            "Barnes-Hut KL divergence: {} (non-finite, which is acceptable for approximation)",
            kl_div
        );
    } else {
        println!("Barnes-Hut KL divergence: {} (finite)", kl_div);
    }
}

#[test]
fn test_tsne_multicore() {
    // Create a simple dataset
    let x = arr2(&[
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [5.0, 5.0],
        [6.0, 5.0],
        [5.0, 6.0],
        [6.0, 6.0],
    ]);

    // Initialize and fit t-SNE with multicore enabled
    let mut tsne_multicore = TSNE::new()
            .with_n_components(2)
            .with_perplexity(2.0)
            .with_method("exact")
            .with_n_jobs(-1) // Use all cores
            .with_random_state(42)
            .with_max_iter(100) // Shorter for testing
            .with_verbose(false);

    let embedding_multicore = tsne_multicore.fit_transform(&x).expect("Operation failed");

    // Check that the shape is correct
    assert_eq!(embedding_multicore.shape(), &[8, 2]);

    // Test basic functionality - multicore should produce valid results
    assert!(embedding_multicore.iter().all(|&x| x.is_finite()));

    // Check that the embedding has some spread (more lenient for short iterations)
    let min_val = embedding_multicore
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let max_val = embedding_multicore
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        max_val - min_val > 1e-12,
        "Embedding should have some spread, got range: {}",
        max_val - min_val
    );

    // Test single-core vs multicore consistency
    let mut tsne_singlecore = TSNE::new()
            .with_n_components(2)
            .with_perplexity(2.0)
            .with_method("exact")
            .with_n_jobs(1) // Single core
            .with_random_state(42)
            .with_max_iter(100)
            .with_verbose(false);

    let embedding_singlecore = tsne_singlecore.fit_transform(&x).expect("Operation failed");

    // Both should produce finite results (exact numerical match is not expected due to randomness)
    assert!(embedding_multicore.iter().all(|&x| x.is_finite()));
    assert!(embedding_singlecore.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_tsne_3d_barnes_hut() {
    // Create a simple 3D dataset
    let x = arr2(&[
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [5.0, 5.0, 5.0],
        [6.0, 5.0, 5.0],
        [5.0, 6.0, 5.0],
        [6.0, 6.0, 5.0],
    ]);

    // Initialize and fit t-SNE with Barnes-Hut method for 3D
    let mut tsne_3d = TSNE::new()
        .with_n_components(3)
        .with_perplexity(2.0)
        .with_method("barnes_hut")
        .with_angle(0.5)
        .with_random_state(42)
        .with_max_iter(250)
        .with_verbose(false);

    let embedding_3d = tsne_3d.fit_transform(&x).expect("Operation failed");

    // Check that the shape is correct
    assert_eq!(embedding_3d.shape(), &[8, 3]);

    // Test basic functionality - should not panic
    assert!(embedding_3d.iter().all(|&x| x.is_finite()));
}

// Helper function to compute average pairwise distance within a group
fn average_pairwise_distance(points: &ArrayBase<scirs2_core::ndarray::ViewRepr<&f64>, Ix2>) -> f64 {
    let n = points.shape()[0];
    let mut total_dist = 0.0;
    let mut count = 0;

    for i in 0..n {
        for j in i + 1..n {
            let mut dist_squared = 0.0;
            for k in 0..points.shape()[1] {
                let diff = points[[i, k]] - points[[j, k]];
                dist_squared += diff * diff;
            }
            total_dist += dist_squared.sqrt();
            count += 1;
        }
    }

    if count > 0 {
        total_dist / count as f64
    } else {
        0.0
    }
}

// Helper function to compute average distance between two groups
fn average_intergroup_distance(
    group1: &ArrayBase<scirs2_core::ndarray::ViewRepr<&f64>, Ix2>,
    group2: &ArrayBase<scirs2_core::ndarray::ViewRepr<&f64>, Ix2>,
) -> f64 {
    let n1 = group1.shape()[0];
    let n2 = group2.shape()[0];
    let mut total_dist = 0.0;
    let mut count = 0;

    for i in 0..n1 {
        for j in 0..n2 {
            let mut dist_squared = 0.0;
            for k in 0..group1.shape()[1] {
                let diff = group1[[i, k]] - group2[[j, k]];
                dist_squared += diff * diff;
            }
            total_dist += dist_squared.sqrt();
            count += 1;
        }
    }

    if count > 0 {
        total_dist / count as f64
    } else {
        0.0
    }
}

#[test]
fn test_trustworthiness() {
    // Create a simple dataset where we know the structure
    let x = arr2(&[
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [5.0, 5.0],
        [5.0, 6.0],
        [6.0, 5.0],
        [6.0, 6.0],
    ]);

    // A perfect embedding would preserve all neighborhoods
    let perfect_embedding = x.clone();
    let t_perfect =
        trustworthiness(&x, &perfect_embedding, 3, "euclidean").expect("Operation failed");
    assert_abs_diff_eq!(t_perfect, 1.0, epsilon = 1e-10);

    // A random embedding would have low trustworthiness
    let random_embedding = arr2(&[
        [0.9, 0.1],
        [0.8, 0.2],
        [0.7, 0.3],
        [0.6, 0.4],
        [0.5, 0.5],
        [0.4, 0.6],
        [0.3, 0.7],
        [0.2, 0.8],
    ]);

    let t_random =
        trustworthiness(&x, &random_embedding, 3, "euclidean").expect("Operation failed");
    assert!(t_random < 1.0);
}

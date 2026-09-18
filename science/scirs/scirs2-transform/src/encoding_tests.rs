use super::*;

use super::*;
use approx::assert_abs_diff_eq;
use scirs2_core::ndarray::Array;

#[test]
fn test_one_hot_encoder_basic() {
    // Create test data with categorical values
    let data = Array::from_shape_vec(
        (4, 2),
        vec![
            0.0, 1.0, // categories: [0, 1, 2] and [1, 2, 3]
            1.0, 2.0, 2.0, 3.0, 0.0, 1.0,
        ],
    )
    .expect("Test data construction failed");

    let mut encoder = OneHotEncoder::with_defaults();
    let encoded = encoder
        .fit_transform(&data)
        .expect("Test data construction failed");

    // Should have 3 + 3 = 6 output features
    assert_eq!(encoded.shape(), (4, 6));

    // Convert to dense for indexing
    let encoded_dense = encoded.to_dense();

    // Check first row: category 0 in feature 0, category 1 in feature 1
    assert_abs_diff_eq!(encoded_dense[[0, 0]], 1.0, epsilon = 1e-10); // cat 0, feature 0
    assert_abs_diff_eq!(encoded_dense[[0, 1]], 0.0, epsilon = 1e-10); // cat 1, feature 0
    assert_abs_diff_eq!(encoded_dense[[0, 2]], 0.0, epsilon = 1e-10); // cat 2, feature 0
    assert_abs_diff_eq!(encoded_dense[[0, 3]], 1.0, epsilon = 1e-10); // cat 1, feature 1
    assert_abs_diff_eq!(encoded_dense[[0, 4]], 0.0, epsilon = 1e-10); // cat 2, feature 1
    assert_abs_diff_eq!(encoded_dense[[0, 5]], 0.0, epsilon = 1e-10); // cat 3, feature 1

    // Check second row: category 1 in feature 0, category 2 in feature 1
    assert_abs_diff_eq!(encoded_dense[[1, 0]], 0.0, epsilon = 1e-10); // cat 0, feature 0
    assert_abs_diff_eq!(encoded_dense[[1, 1]], 1.0, epsilon = 1e-10); // cat 1, feature 0
    assert_abs_diff_eq!(encoded_dense[[1, 2]], 0.0, epsilon = 1e-10); // cat 2, feature 0
    assert_abs_diff_eq!(encoded_dense[[1, 3]], 0.0, epsilon = 1e-10); // cat 1, feature 1
    assert_abs_diff_eq!(encoded_dense[[1, 4]], 1.0, epsilon = 1e-10); // cat 2, feature 1
    assert_abs_diff_eq!(encoded_dense[[1, 5]], 0.0, epsilon = 1e-10); // cat 3, feature 1
}

#[test]
fn test_one_hot_encoder_drop_first() {
    // Create test data with categorical values
    let data = Array::from_shape_vec((3, 2), vec![0.0, 1.0, 1.0, 2.0, 2.0, 1.0])
        .expect("Test data construction failed");

    let mut encoder = OneHotEncoder::new(Some("first".to_string()), "error", false)
        .expect("Test data construction failed");
    let encoded = encoder
        .fit_transform(&data)
        .expect("Test data construction failed");

    // Should have (3-1) + (2-1) = 3 output features (dropped first category of each)
    assert_eq!(encoded.shape(), (3, 3));

    // Categories: feature 0: [0, 1, 2] -> keep [1, 2]
    //            feature 1: [1, 2] -> keep [2]
    let encoded_dense = encoded.to_dense();

    // First row: category 0 (dropped), category 1 (dropped)
    assert_abs_diff_eq!(encoded_dense[[0, 0]], 0.0, epsilon = 1e-10); // cat 1, feature 0
    assert_abs_diff_eq!(encoded_dense[[0, 1]], 0.0, epsilon = 1e-10); // cat 2, feature 0
    assert_abs_diff_eq!(encoded_dense[[0, 2]], 0.0, epsilon = 1e-10); // cat 2, feature 1

    // Second row: category 1, category 2
    assert_abs_diff_eq!(encoded_dense[[1, 0]], 1.0, epsilon = 1e-10); // cat 1, feature 0
    assert_abs_diff_eq!(encoded_dense[[1, 1]], 0.0, epsilon = 1e-10); // cat 2, feature 0
    assert_abs_diff_eq!(encoded_dense[[1, 2]], 1.0, epsilon = 1e-10); // cat 2, feature 1
}

#[test]
fn test_ordinal_encoder() {
    // Create test data with categorical values
    let data = Array::from_shape_vec(
        (4, 2),
        vec![
            2.0, 10.0, // categories will be mapped to ordinals
            1.0, 20.0, 3.0, 10.0, 2.0, 30.0,
        ],
    )
    .expect("Test data construction failed");

    let mut encoder = OrdinalEncoder::with_defaults();
    let encoded = encoder
        .fit_transform(&data)
        .expect("Test data construction failed");

    // Should preserve shape
    assert_eq!(encoded.shape(), &[4, 2]);

    // Categories for feature 0: [1, 2, 3] -> ordinals [0, 1, 2]
    // Categories for feature 1: [10, 20, 30] -> ordinals [0, 1, 2]

    // Check mappings
    assert_abs_diff_eq!(encoded[[0, 0]], 1.0, epsilon = 1e-10); // 2 -> ordinal 1
    assert_abs_diff_eq!(encoded[[0, 1]], 0.0, epsilon = 1e-10); // 10 -> ordinal 0
    assert_abs_diff_eq!(encoded[[1, 0]], 0.0, epsilon = 1e-10); // 1 -> ordinal 0
    assert_abs_diff_eq!(encoded[[1, 1]], 1.0, epsilon = 1e-10); // 20 -> ordinal 1
    assert_abs_diff_eq!(encoded[[2, 0]], 2.0, epsilon = 1e-10); // 3 -> ordinal 2
    assert_abs_diff_eq!(encoded[[2, 1]], 0.0, epsilon = 1e-10); // 10 -> ordinal 0
    assert_abs_diff_eq!(encoded[[3, 0]], 1.0, epsilon = 1e-10); // 2 -> ordinal 1
    assert_abs_diff_eq!(encoded[[3, 1]], 2.0, epsilon = 1e-10); // 30 -> ordinal 2
}

#[test]
fn test_unknown_category_handling() {
    let train_data =
        Array::from_shape_vec((2, 1), vec![1.0, 2.0]).expect("Test data construction failed");

    let test_data = Array::from_shape_vec(
        (1, 1),
        vec![3.0], // Unknown category
    )
    .expect("Test data construction failed");

    // Test error handling
    let mut encoder = OneHotEncoder::with_defaults(); // with_defaults is handleunknown="error"
    encoder
        .fit(&train_data)
        .expect("Test data construction failed");
    assert!(encoder.transform(&test_data).is_err());

    // Test ignore handling
    let mut encoder =
        OneHotEncoder::new(None, "ignore", false).expect("Test data construction failed");
    encoder
        .fit(&train_data)
        .expect("Test data construction failed");
    let encoded = encoder
        .transform(&test_data)
        .expect("Test data construction failed");

    // Should be all zeros (ignored unknown category)
    assert_eq!(encoded.shape(), (1, 2));
    let encoded_dense = encoded.to_dense();
    assert_abs_diff_eq!(encoded_dense[[0, 0]], 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(encoded_dense[[0, 1]], 0.0, epsilon = 1e-10);
}

#[test]
fn test_ordinal_encoder_unknown_value() {
    let train_data =
        Array::from_shape_vec((2, 1), vec![1.0, 2.0]).expect("Test data construction failed");

    let test_data = Array::from_shape_vec(
        (1, 1),
        vec![3.0], // Unknown category
    )
    .expect("Test data construction failed");

    let mut encoder = OrdinalEncoder::new("use_encoded_value", Some(-1.0))
        .expect("Test data construction failed");
    encoder
        .fit(&train_data)
        .expect("Test data construction failed");
    let encoded = encoder
        .transform(&test_data)
        .expect("Test data construction failed");

    // Should use the specified unknown value
    assert_eq!(encoded.shape(), &[1, 1]);
    assert_abs_diff_eq!(encoded[[0, 0]], -1.0, epsilon = 1e-10);
}

#[test]
fn test_get_feature_names() {
    let data = Array::from_shape_vec((2, 2), vec![1.0, 10.0, 2.0, 20.0])
        .expect("Test data construction failed");

    let mut encoder = OneHotEncoder::with_defaults();
    encoder.fit(&data).expect("Test data construction failed");

    let feature_names = encoder
        .get_feature_names(None)
        .expect("Test data construction failed");
    assert_eq!(feature_names.len(), 4); // 2 cats per feature * 2 features

    let custom_names = vec!["feat_a".to_string(), "feat_b".to_string()];
    let feature_names = encoder
        .get_feature_names(Some(&custom_names))
        .expect("Test data construction failed");
    assert!(feature_names[0].starts_with("feat_a_cat_"));
    assert!(feature_names[2].starts_with("feat_b_cat_"));
}

#[test]
fn test_target_encoder_mean_strategy() {
    // Create test data
    let x = Array::from_shape_vec((6, 1), vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
        .expect("Test data construction failed");
    let y = vec![1.0, 2.0, 3.0, 1.5, 2.5, 3.5];

    let mut encoder = TargetEncoder::new("mean", 0.0, 0.0).expect("Test data construction failed");
    let encoded = encoder
        .fit_transform(&x, &y)
        .expect("Test data construction failed");

    // Should preserve shape
    assert_eq!(encoded.shape(), &[6, 1]);

    // Check category encodings:
    // Category 0: targets [1.0, 1.5] -> mean = 1.25
    // Category 1: targets [2.0, 2.5] -> mean = 2.25
    // Category 2: targets [3.0, 3.5] -> mean = 3.25

    assert_abs_diff_eq!(encoded[[0, 0]], 1.25, epsilon = 1e-10);
    assert_abs_diff_eq!(encoded[[1, 0]], 2.25, epsilon = 1e-10);
    assert_abs_diff_eq!(encoded[[2, 0]], 3.25, epsilon = 1e-10);
    assert_abs_diff_eq!(encoded[[3, 0]], 1.25, epsilon = 1e-10);
    assert_abs_diff_eq!(encoded[[4, 0]], 2.25, epsilon = 1e-10);
    assert_abs_diff_eq!(encoded[[5, 0]], 3.25, epsilon = 1e-10);

    // Check global mean
    assert_abs_diff_eq!(encoder.global_mean(), 2.25, epsilon = 1e-10);
}

#[test]
fn test_target_encoder_median_strategy() {
    let x = Array::from_shape_vec((4, 1), vec![0.0, 1.0, 0.0, 1.0])
        .expect("Test data construction failed");
    let y = vec![1.0, 2.0, 3.0, 4.0];

    let mut encoder =
        TargetEncoder::new("median", 0.0, 0.0).expect("Test data construction failed");
    let encoded = encoder
        .fit_transform(&x, &y)
        .expect("Test data construction failed");

    // Category 0: targets [1.0, 3.0] -> median = 2.0
    // Category 1: targets [2.0, 4.0] -> median = 3.0

    assert_abs_diff_eq!(encoded[[0, 0]], 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(encoded[[1, 0]], 3.0, epsilon = 1e-10);
    assert_abs_diff_eq!(encoded[[2, 0]], 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(encoded[[3, 0]], 3.0, epsilon = 1e-10);
}

#[test]
fn test_target_encoder_count_strategy() {
    let x = Array::from_shape_vec((5, 1), vec![0.0, 1.0, 0.0, 2.0, 1.0])
        .expect("Test data construction failed");
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let mut encoder = TargetEncoder::new("count", 0.0, 0.0).expect("Test data construction failed");
    let encoded = encoder
        .fit_transform(&x, &y)
        .expect("Test data construction failed");

    // Category 0: appears 2 times
    // Category 1: appears 2 times
    // Category 2: appears 1 time

    assert_abs_diff_eq!(encoded[[0, 0]], 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(encoded[[1, 0]], 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(encoded[[2, 0]], 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(encoded[[3, 0]], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(encoded[[4, 0]], 2.0, epsilon = 1e-10);
}

#[test]
fn test_target_encoder_sum_strategy() {
    let x = Array::from_shape_vec((4, 1), vec![0.0, 1.0, 0.0, 1.0])
        .expect("Test data construction failed");
    let y = vec![1.0, 2.0, 3.0, 4.0];

    let mut encoder = TargetEncoder::new("sum", 0.0, 0.0).expect("Test data construction failed");
    let encoded = encoder
        .fit_transform(&x, &y)
        .expect("Test data construction failed");

    // Category 0: targets [1.0, 3.0] -> sum = 4.0
    // Category 1: targets [2.0, 4.0] -> sum = 6.0

    assert_abs_diff_eq!(encoded[[0, 0]], 4.0, epsilon = 1e-10);
    assert_abs_diff_eq!(encoded[[1, 0]], 6.0, epsilon = 1e-10);
    assert_abs_diff_eq!(encoded[[2, 0]], 4.0, epsilon = 1e-10);
    assert_abs_diff_eq!(encoded[[3, 0]], 6.0, epsilon = 1e-10);
}

#[test]
fn test_target_encoder_smoothing() {
    let x =
        Array::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).expect("Test data construction failed");
    let y = vec![1.0, 2.0, 3.0];

    let mut encoder = TargetEncoder::new("mean", 1.0, 0.0).expect("Test data construction failed");
    let encoded = encoder
        .fit_transform(&x, &y)
        .expect("Test data construction failed");

    // Global mean = (1+2+3)/3 = 2.0
    // Category 0: count=1, mean=1.0 -> smoothed = (1*1.0 + 1.0*2.0)/(1+1) = 1.5
    // Category 1: count=1, mean=2.0 -> smoothed = (1*2.0 + 1.0*2.0)/(1+1) = 2.0
    // Category 2: count=1, mean=3.0 -> smoothed = (1*3.0 + 1.0*2.0)/(1+1) = 2.5

    assert_abs_diff_eq!(encoded[[0, 0]], 1.5, epsilon = 1e-10);
    assert_abs_diff_eq!(encoded[[1, 0]], 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(encoded[[2, 0]], 2.5, epsilon = 1e-10);
}

#[test]
fn test_target_encoder_unknown_categories() {
    let train_x =
        Array::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).expect("Test data construction failed");
    let train_y = vec![1.0, 2.0, 3.0];

    let test_x =
        Array::from_shape_vec((2, 1), vec![3.0, 4.0]).expect("Test data construction failed"); // Unknown categories

    let mut encoder = TargetEncoder::new("mean", 0.0, -1.0).expect("Test data construction failed");
    encoder
        .fit(&train_x, &train_y)
        .expect("Test data construction failed");
    let encoded = encoder
        .transform(&test_x)
        .expect("Test data construction failed");

    // Should use globalstat for unknown categories
    assert_abs_diff_eq!(encoded[[0, 0]], -1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(encoded[[1, 0]], -1.0, epsilon = 1e-10);
}

#[test]
fn test_target_encoder_unknown_categories_global_mean() {
    let train_x =
        Array::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).expect("Test data construction failed");
    let train_y = vec![1.0, 2.0, 3.0];

    let test_x = Array::from_shape_vec((1, 1), vec![3.0]).expect("Test data construction failed"); // Unknown category

    let mut encoder = TargetEncoder::new("mean", 0.0, 0.0).expect("Test data construction failed"); // globalstat = 0.0
    encoder
        .fit(&train_x, &train_y)
        .expect("Test data construction failed");
    let encoded = encoder
        .transform(&test_x)
        .expect("Test data construction failed");

    // Should use global_mean for unknown categories when globalstat == 0.0
    assert_abs_diff_eq!(encoded[[0, 0]], 2.0, epsilon = 1e-10); // Global mean = 2.0
}

#[test]
fn test_target_encoder_multi_feature() {
    let x = Array::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
        .expect("Test data construction failed");
    let y = vec![1.0, 2.0, 3.0, 4.0];

    let mut encoder = TargetEncoder::new("mean", 0.0, 0.0).expect("Test data construction failed");
    let encoded = encoder
        .fit_transform(&x, &y)
        .expect("Test data construction failed");

    assert_eq!(encoded.shape(), &[4, 2]);

    // Feature 0: Category 0 -> targets [1.0, 3.0] -> mean = 2.0
    //           Category 1 -> targets [2.0, 4.0] -> mean = 3.0
    // Feature 1: Category 0 -> targets [1.0, 4.0] -> mean = 2.5
    //           Category 1 -> targets [2.0, 3.0] -> mean = 2.5

    assert_abs_diff_eq!(encoded[[0, 0]], 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(encoded[[0, 1]], 2.5, epsilon = 1e-10);
    assert_abs_diff_eq!(encoded[[1, 0]], 3.0, epsilon = 1e-10);
    assert_abs_diff_eq!(encoded[[1, 1]], 2.5, epsilon = 1e-10);
}

#[test]
fn test_target_encoder_cross_validation() {
    let x = Array::from_shape_vec(
        (10, 1),
        vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    )
    .expect("Test data construction failed");
    let y = vec![1.0, 2.0, 1.5, 2.5, 1.2, 2.2, 1.3, 2.3, 1.1, 2.1];

    let mut encoder = TargetEncoder::new("mean", 0.0, 0.0).expect("Test data construction failed");
    let encoded = encoder
        .fit_transform_cv(&x, &y, 5)
        .expect("Test data construction failed");

    // Should have same shape
    assert_eq!(encoded.shape(), &[10, 1]);

    // Results should be reasonable (not exact due to CV)
    // All category 0 samples should get similar values
    // All category 1 samples should get similar values
    assert!(encoded[[0, 0]] < encoded[[1, 0]]); // Category 0 < Category 1
    assert!(encoded[[2, 0]] < encoded[[3, 0]]);
}

#[test]
fn test_target_encoder_convenience_methods() {
    let _x = Array::from_shape_vec((4, 1), vec![0.0, 1.0, 0.0, 1.0])
        .expect("Test data construction failed");
    let _y = [1.0, 2.0, 3.0, 4.0];

    let encoder1 = TargetEncoder::with_mean(1.0);
    assert_eq!(encoder1.strategy, "mean");
    assert_abs_diff_eq!(encoder1.smoothing, 1.0, epsilon = 1e-10);

    let encoder2 = TargetEncoder::with_median(0.5);
    assert_eq!(encoder2.strategy, "median");
    assert_abs_diff_eq!(encoder2.smoothing, 0.5, epsilon = 1e-10);
}

#[test]
fn test_target_encoder_validation_errors() {
    // Invalid strategy
    assert!(TargetEncoder::new("invalid", 0.0, 0.0).is_err());

    // Negative smoothing
    assert!(TargetEncoder::new("mean", -1.0, 0.0).is_err());

    // Mismatched target length
    let x =
        Array::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).expect("Test data construction failed");
    let y = vec![1.0, 2.0]; // Wrong length

    let mut encoder = TargetEncoder::new("mean", 0.0, 0.0).expect("Test data construction failed");
    assert!(encoder.fit(&x, &y).is_err());

    // Transform before fit
    let encoder2 = TargetEncoder::new("mean", 0.0, 0.0).expect("Test data construction failed");
    assert!(encoder2.transform(&x).is_err());

    // Wrong number of features
    let train_x =
        Array::from_shape_vec((2, 1), vec![0.0, 1.0]).expect("Test data construction failed");
    let test_x = Array::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0])
        .expect("Test data construction failed");
    let train_y = vec![1.0, 2.0];

    let mut encoder = TargetEncoder::new("mean", 0.0, 0.0).expect("Test data construction failed");
    encoder
        .fit(&train_x, &train_y)
        .expect("Test data construction failed");
    assert!(encoder.transform(&test_x).is_err());

    // Invalid CV folds
    let x = Array::from_shape_vec((4, 1), vec![0.0, 1.0, 0.0, 1.0])
        .expect("Test data construction failed");
    let y = vec![1.0, 2.0, 3.0, 4.0];
    let mut encoder = TargetEncoder::new("mean", 0.0, 0.0).expect("Test data construction failed");
    assert!(encoder.fit_transform_cv(&x, &y, 1).is_err()); // cv_folds < 2
}

#[test]
fn test_target_encoder_accessors() {
    let x =
        Array::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).expect("Test data construction failed");
    let y = vec![1.0, 2.0, 3.0];

    let mut encoder = TargetEncoder::new("mean", 0.0, 0.0).expect("Test data construction failed");

    assert!(!encoder.is_fitted());
    assert!(encoder.encodings().is_none());

    encoder.fit(&x, &y).expect("Test data construction failed");

    assert!(encoder.is_fitted());
    assert!(encoder.encodings().is_some());
    assert_abs_diff_eq!(encoder.global_mean(), 2.0, epsilon = 1e-10);

    let encodings = encoder.encodings().expect("Test data construction failed");
    assert_eq!(encodings.len(), 1); // 1 feature
    assert_eq!(encodings[0].len(), 3); // 3 categories
}

#[test]
fn test_target_encoder_empty_data() {
    let empty_x = Array2::<f64>::zeros((0, 1));
    let empty_y = vec![];

    let mut encoder = TargetEncoder::new("mean", 0.0, 0.0).expect("Test data construction failed");
    assert!(encoder.fit(&empty_x, &empty_y).is_err());
}

// ===== BinaryEncoder Tests =====

#[test]
fn test_binary_encoder_basic() {
    // Test basic binary encoding with 4 categories (needs 2 bits)
    let data = Array::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0])
        .expect("Test data construction failed");

    let mut encoder = BinaryEncoder::with_defaults();
    let encoded = encoder
        .fit_transform(&data)
        .expect("Test data construction failed");

    // Should have 2 binary features (ceil(log2(4)) = 2)
    assert_eq!(encoded.shape(), &[4, 2]);

    // Check binary codes: 0=00, 1=01, 2=10, 3=11
    assert_abs_diff_eq!(encoded[[0, 0]], 0.0, epsilon = 1e-10); // 0 -> 00
    assert_abs_diff_eq!(encoded[[0, 1]], 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(encoded[[1, 0]], 0.0, epsilon = 1e-10); // 1 -> 01
    assert_abs_diff_eq!(encoded[[1, 1]], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(encoded[[2, 0]], 1.0, epsilon = 1e-10); // 2 -> 10
    assert_abs_diff_eq!(encoded[[2, 1]], 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(encoded[[3, 0]], 1.0, epsilon = 1e-10); // 3 -> 11
    assert_abs_diff_eq!(encoded[[3, 1]], 1.0, epsilon = 1e-10);
}

#[test]
fn test_binary_encoder_power_of_two() {
    // Test with exactly 8 categories (power of 2)
    let data = Array::from_shape_vec((8, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        .expect("Test data construction failed");

    let mut encoder = BinaryEncoder::with_defaults();
    let encoded = encoder
        .fit_transform(&data)
        .expect("Test data construction failed");

    // Should have 3 binary features (log2(8) = 3)
    assert_eq!(encoded.shape(), &[8, 3]);

    // Check some specific encodings
    assert_abs_diff_eq!(encoded[[0, 0]], 0.0, epsilon = 1e-10); // 0 -> 000
    assert_abs_diff_eq!(encoded[[0, 1]], 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(encoded[[0, 2]], 0.0, epsilon = 1e-10);

    assert_abs_diff_eq!(encoded[[7, 0]], 1.0, epsilon = 1e-10); // 7 -> 111
    assert_abs_diff_eq!(encoded[[7, 1]], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(encoded[[7, 2]], 1.0, epsilon = 1e-10);
}

#[test]
fn test_binary_encoder_non_power_of_two() {
    // Test with 5 categories (not power of 2, needs 3 bits)
    let data = Array::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0])
        .expect("Test data construction failed");

    let mut encoder = BinaryEncoder::with_defaults();
    let encoded = encoder
        .fit_transform(&data)
        .expect("Test data construction failed");

    // Should have 3 binary features (ceil(log2(5)) = 3)
    assert_eq!(encoded.shape(), &[5, 3]);
    assert_eq!(
        encoder
            .n_output_features()
            .expect("Failed to get output features count"),
        3
    );
}

#[test]
fn test_binary_encoder_single_category() {
    // Test edge case with only 1 category
    let data =
        Array::from_shape_vec((3, 1), vec![5.0, 5.0, 5.0]).expect("Test data construction failed");

    let mut encoder = BinaryEncoder::with_defaults();
    let encoded = encoder
        .fit_transform(&data)
        .expect("Test data construction failed");

    // Should have 1 binary feature for single category
    assert_eq!(encoded.shape(), &[3, 1]);
    assert_eq!(
        encoder
            .n_output_features()
            .expect("Failed to get output features count"),
        1
    );

    // All values should be encoded as 0
    for i in 0..3 {
        assert_abs_diff_eq!(encoded[[i, 0]], 0.0, epsilon = 1e-10);
    }
}

#[test]
fn test_binary_encoder_multi_feature() {
    // Test with multiple features
    let data = Array::from_shape_vec(
        (4, 2),
        vec![
            0.0, 10.0, // Feature 0: [0,1,2] (2 bits), Feature 1: [10,11] (1 bit)
            1.0, 11.0, 2.0, 10.0, 0.0, 11.0,
        ],
    )
    .expect("Test data construction failed");

    let mut encoder = BinaryEncoder::with_defaults();
    let encoded = encoder
        .fit_transform(&data)
        .expect("Test data construction failed");

    // Feature 0: 3 categories need 2 bits, Feature 1: 2 categories need 1 bit
    // Total: 2 + 1 = 3 features
    assert_eq!(encoded.shape(), &[4, 3]);
    assert_eq!(
        encoder
            .n_output_features()
            .expect("Failed to get output features count"),
        3
    );

    let n_binary_features = encoder
        .n_binary_features()
        .expect("Test data construction failed");
    assert_eq!(n_binary_features, &[2, 1]);
}

#[test]
fn test_binary_encoder_separate_fit_transform() {
    let train_data =
        Array::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).expect("Test data construction failed");
    let test_data =
        Array::from_shape_vec((2, 1), vec![1.0, 0.0]).expect("Test data construction failed");

    let mut encoder = BinaryEncoder::with_defaults();

    // Fit on training data
    encoder
        .fit(&train_data)
        .expect("Test data construction failed");
    assert!(encoder.is_fitted());

    // Transform test data
    let encoded = encoder
        .transform(&test_data)
        .expect("Test data construction failed");
    assert_eq!(encoded.shape(), &[2, 2]); // 3 categories need 2 bits

    // Check that mappings are consistent
    let train_encoded = encoder
        .transform(&train_data)
        .expect("Test data construction failed");
    assert_abs_diff_eq!(encoded[[0, 0]], train_encoded[[1, 0]], epsilon = 1e-10); // Same category 1
    assert_abs_diff_eq!(encoded[[0, 1]], train_encoded[[1, 1]], epsilon = 1e-10);
}

#[test]
fn test_binary_encoder_unknown_categories_error() {
    let train_data =
        Array::from_shape_vec((2, 1), vec![0.0, 1.0]).expect("Test data construction failed");
    let test_data =
        Array::from_shape_vec((1, 1), vec![2.0]).expect("Test data construction failed"); // Unknown category

    let mut encoder = BinaryEncoder::new("error").expect("Test data construction failed");
    encoder
        .fit(&train_data)
        .expect("Test data construction failed");

    // Should error on unknown category
    assert!(encoder.transform(&test_data).is_err());
}

#[test]
fn test_binary_encoder_unknown_categories_ignore() {
    let train_data =
        Array::from_shape_vec((2, 1), vec![0.0, 1.0]).expect("Test data construction failed");
    let test_data =
        Array::from_shape_vec((1, 1), vec![2.0]).expect("Test data construction failed"); // Unknown category

    let mut encoder = BinaryEncoder::new("ignore").expect("Test data construction failed");
    encoder
        .fit(&train_data)
        .expect("Test data construction failed");
    let encoded = encoder
        .transform(&test_data)
        .expect("Test data construction failed");

    // Unknown category should be encoded as all zeros
    assert_eq!(encoded.shape(), &[1, 1]); // 2 categories need 1 bit
    assert_abs_diff_eq!(encoded[[0, 0]], 0.0, epsilon = 1e-10);
}

#[test]
fn test_binary_encoder_categories_accessor() {
    let data = Array::from_shape_vec((3, 1), vec![10.0, 20.0, 30.0])
        .expect("Test data construction failed");

    let mut encoder = BinaryEncoder::with_defaults();

    // Before fitting
    assert!(!encoder.is_fitted());
    assert!(encoder.categories().is_none());
    assert!(encoder.n_binary_features().is_none());
    assert!(encoder.n_output_features().is_none());

    encoder.fit(&data).expect("Test data construction failed");

    // After fitting
    assert!(encoder.is_fitted());
    assert!(encoder.categories().is_some());
    assert!(encoder.n_binary_features().is_some());
    assert!(encoder.n_output_features().is_some());

    let categories = encoder.categories().expect("Test data construction failed");
    assert_eq!(categories.len(), 1); // 1 feature
    assert_eq!(categories[0].len(), 3); // 3 categories

    // Check that categories are mapped correctly
    let category_map = &categories[0];
    assert!(category_map.contains_key(&10));
    assert!(category_map.contains_key(&20));
    assert!(category_map.contains_key(&30));
}

#[test]
fn test_binary_encoder_int_to_binary() {
    // Test binary conversion utility function
    assert_eq!(BinaryEncoder::int_to_binary(0, 3), vec![0, 0, 0]);
    assert_eq!(BinaryEncoder::int_to_binary(1, 3), vec![0, 0, 1]);
    assert_eq!(BinaryEncoder::int_to_binary(2, 3), vec![0, 1, 0]);
    assert_eq!(BinaryEncoder::int_to_binary(3, 3), vec![0, 1, 1]);
    assert_eq!(BinaryEncoder::int_to_binary(7, 3), vec![1, 1, 1]);

    // Test with different bit lengths
    assert_eq!(BinaryEncoder::int_to_binary(5, 4), vec![0, 1, 0, 1]);
    assert_eq!(BinaryEncoder::int_to_binary(1, 1), vec![1]);
}

#[test]
fn test_binary_encoder_validation_errors() {
    // Invalid handleunknown parameter
    assert!(BinaryEncoder::new("invalid").is_err());

    // Empty data
    let empty_data = Array2::<f64>::zeros((0, 1));
    let mut encoder = BinaryEncoder::with_defaults();
    assert!(encoder.fit(&empty_data).is_err());

    // Transform before fit
    let data =
        Array::from_shape_vec((2, 1), vec![0.0, 1.0]).expect("Test data construction failed");
    let encoder = BinaryEncoder::with_defaults();
    assert!(encoder.transform(&data).is_err());

    // Wrong number of features in transform
    let train_data =
        Array::from_shape_vec((2, 1), vec![0.0, 1.0]).expect("Test data construction failed");
    let test_data = Array::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0])
        .expect("Test data construction failed");

    let mut encoder = BinaryEncoder::with_defaults();
    encoder
        .fit(&train_data)
        .expect("Test data construction failed");
    assert!(encoder.transform(&test_data).is_err());
}

#[test]
fn test_binary_encoder_consistency() {
    // Test that encoding is consistent across multiple calls
    let data = Array::from_shape_vec((4, 1), vec![3.0, 1.0, 4.0, 1.0])
        .expect("Test data construction failed");

    let mut encoder = BinaryEncoder::with_defaults();
    let encoded1 = encoder
        .fit_transform(&data)
        .expect("Test data construction failed");
    let encoded2 = encoder
        .transform(&data)
        .expect("Test data construction failed");

    // Both should be identical
    for i in 0..encoded1.shape()[0] {
        for j in 0..encoded1.shape()[1] {
            assert_abs_diff_eq!(encoded1[[i, j]], encoded2[[i, j]], epsilon = 1e-10);
        }
    }

    // Same categories should have same encoding
    assert_abs_diff_eq!(encoded1[[1, 0]], encoded1[[3, 0]], epsilon = 1e-10); // Both category 1
    assert_abs_diff_eq!(encoded1[[1, 1]], encoded1[[3, 1]], epsilon = 1e-10);
}

#[test]
fn test_binary_encoder_memory_efficiency() {
    // Test that binary encoding is more memory efficient than one-hot
    // For 10 categories: one-hot needs 10 features, binary needs 4 features
    let data = Array::from_shape_vec(
        (10, 1),
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .expect("Test data construction failed");

    let mut binary_encoder = BinaryEncoder::with_defaults();
    let binary_encoded = binary_encoder
        .fit_transform(&data)
        .expect("Test data construction failed");

    let mut onehot_encoder = OneHotEncoder::with_defaults();
    let onehot_encoded = onehot_encoder
        .fit_transform(&data)
        .expect("Test data construction failed");

    // Binary should use fewer features
    assert_eq!(binary_encoded.shape()[1], 4); // ceil(log2(10)) = 4
    assert_eq!(onehot_encoded.shape().1, 10); // 10 categories = 10 features
    assert!(binary_encoded.shape()[1] < onehot_encoded.shape().1);
}

#[test]
fn test_sparse_matrix_basic() {
    let mut sparse = SparseMatrix::new((3, 4));
    sparse.push(0, 1, 1.0);
    sparse.push(1, 2, 1.0);
    sparse.push(2, 0, 1.0);

    assert_eq!(sparse.shape, (3, 4));
    assert_eq!(sparse.nnz(), 3);

    let dense = sparse.to_dense();
    assert_eq!(dense.shape(), &[3, 4]);
    assert_eq!(dense[[0, 1]], 1.0);
    assert_eq!(dense[[1, 2]], 1.0);
    assert_eq!(dense[[2, 0]], 1.0);
    assert_eq!(dense[[0, 0]], 0.0); // Verify zeros
}

#[test]
fn test_onehot_sparse_output() {
    let data = Array::from_shape_vec((4, 2), vec![0.0, 1.0, 1.0, 2.0, 2.0, 0.0, 0.0, 1.0])
        .expect("Test data construction failed");

    // Test sparse output
    let mut encoder_sparse =
        OneHotEncoder::new(None, "error", true).expect("Test data construction failed");
    let result_sparse = encoder_sparse
        .fit_transform(&data)
        .expect("Test data construction failed");

    match &result_sparse {
        EncodedOutput::Sparse(sparse) => {
            assert_eq!(sparse.shape, (4, 6)); // 3 categories + 3 categories = 6 features
            assert_eq!(sparse.nnz(), 8); // 4 samples * 2 features = 8 non-zeros

            // Convert to dense for comparison
            let dense = sparse.to_dense();

            // First sample [0, 1] should have [1,0,0,0,1,0] (category 0 in col0, category 1 in col1)
            assert_eq!(dense[[0, 0]], 1.0); // category 0 in feature 0
            assert_eq!(dense[[0, 4]], 1.0); // category 1 in feature 1
            assert_eq!(dense[[0, 1]], 0.0); // not category 1 in feature 0
        }
        EncodedOutput::Dense(_) => assert!(false, "Expected sparse output, got dense"),
    }

    // Test dense output for comparison
    let mut encoder_dense =
        OneHotEncoder::new(None, "error", false).expect("Test data construction failed");
    let result_dense = encoder_dense
        .fit_transform(&data)
        .expect("Test data construction failed");

    match result_dense {
        EncodedOutput::Dense(dense) => {
            assert_eq!(dense.shape(), &[4, 6]);
            // Verify dense and sparse produce same results
            let sparse_as_dense = result_sparse.to_dense();
            for i in 0..4 {
                for j in 0..6 {
                    assert_abs_diff_eq!(dense[[i, j]], sparse_as_dense[[i, j]], epsilon = 1e-10);
                }
            }
        }
        EncodedOutput::Sparse(_) => assert!(false, "Expected dense output, got sparse"),
    }
}

#[test]
fn test_onehot_sparse_with_drop() {
    let data =
        Array::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).expect("Test data construction failed");

    let mut encoder = OneHotEncoder::new(Some("first".to_string()), "error", true)
        .expect("Test data construction failed");
    let result = encoder
        .fit_transform(&data)
        .expect("Test data construction failed");

    match result {
        EncodedOutput::Sparse(sparse) => {
            assert_eq!(sparse.shape, (3, 2)); // 3 categories - 1 dropped = 2 features
            assert_eq!(sparse.nnz(), 2); // Only categories 1 and 2 are encoded

            let dense = sparse.to_dense();
            assert_eq!(dense[[0, 0]], 0.0); // Category 0 dropped, all zeros
            assert_eq!(dense[[0, 1]], 0.0);
            assert_eq!(dense[[1, 0]], 1.0); // Category 1 maps to first output
            assert_eq!(dense[[2, 1]], 1.0); // Category 2 maps to second output
        }
        EncodedOutput::Dense(_) => assert!(false, "Expected sparse output, got dense"),
    }
}

#[test]
fn test_onehot_sparse_backward_compatibility() {
    let data =
        Array::from_shape_vec((2, 1), vec![0.0, 1.0]).expect("Test data construction failed");

    let mut encoder =
        OneHotEncoder::new(None, "error", true).expect("Test data construction failed");
    encoder.fit(&data).expect("Test data construction failed");

    // Test that the convenience methods work
    let dense_result = encoder
        .transform_dense(&data)
        .expect("Test data construction failed");
    assert_eq!(dense_result.shape(), &[2, 2]);
    assert_eq!(dense_result[[0, 0]], 1.0);
    assert_eq!(dense_result[[1, 1]], 1.0);

    let mut encoder2 =
        OneHotEncoder::new(None, "error", true).expect("Test data construction failed");
    let dense_result2 = encoder2
        .fit_transform_dense(&data)
        .expect("Test data construction failed");
    assert_eq!(dense_result2.shape(), &[2, 2]);

    // Results should be identical
    for i in 0..2 {
        for j in 0..2 {
            assert_abs_diff_eq!(dense_result[[i, j]], dense_result2[[i, j]], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_encoded_output_methods() {
    let dense_array = Array::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        .expect("Test data construction failed");
    let dense_output = EncodedOutput::Dense(dense_array);

    let mut sparse_matrix = SparseMatrix::new((2, 3));
    sparse_matrix.push(0, 0, 1.0);
    sparse_matrix.push(1, 1, 1.0);
    let sparse_output = EncodedOutput::Sparse(sparse_matrix);

    // Test shape method
    assert_eq!(dense_output.shape(), (2, 3));
    assert_eq!(sparse_output.shape(), (2, 3));

    // Test to_dense method
    let dense_from_dense = dense_output.to_dense();
    let dense_from_sparse = sparse_output.to_dense();

    assert_eq!(dense_from_dense.shape(), &[2, 3]);
    assert_eq!(dense_from_sparse.shape(), &[2, 3]);

    // Verify values are equivalent
    assert_eq!(dense_from_dense[[0, 0]], 1.0);
    assert_eq!(dense_from_sparse[[0, 0]], 1.0);
    assert_eq!(dense_from_dense[[1, 1]], 1.0);
    assert_eq!(dense_from_sparse[[1, 1]], 1.0);
}

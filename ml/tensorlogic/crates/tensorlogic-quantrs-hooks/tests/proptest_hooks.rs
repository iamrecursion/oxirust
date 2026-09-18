//! Proptest property-based tests for hook payloads, serialization, and normalization.
//!
//! Tests cover:
//! 1. Round-trip JSON serialize → deserialize for `DistributionExport` (via `Factor`)
//! 2. Deserializing arbitrary bytes never panics (`DistributionExport`, `ModelExport`, `QuantRSAssignment`)
//! 3. Factor normalization is idempotent
//! 4. `QuantRSAssignment` round-trips through JSON
//! 5. `AnnealingConfig` round-trips through JSON

use proptest::prelude::*;
use scirs2_core::ndarray::Array;
use tensorlogic_quantrs_hooks::{
    AnnealingConfig, DistributionExport, DistributionMetadata, Factor, ModelExport,
    QuantRSAssignment, QuantRSDistribution,
};

// ============================================================================
// Strategies
// ============================================================================

/// Strategy for a vector of positive f64 values (valid factor values).
fn positive_f64_vec(max_len: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(0.01f64..=10.0_f64, 1..=max_len)
}

/// Strategy for factor names and variable names.
fn ascii_name() -> impl Strategy<Value = String> {
    "[a-z][a-z0-9_]{0,7}".prop_map(|s| s)
}

/// Strategy producing a valid `DistributionExport` with a single binary variable.
fn binary_distribution_export_strategy() -> impl Strategy<Value = DistributionExport> {
    (0.01f64..=1.0_f64, 0.01f64..=1.0_f64, ascii_name()).prop_map(|(p0, p1, var_name)| {
        DistributionExport {
            variables: vec![var_name],
            cardinalities: vec![2],
            probabilities: vec![p0, p1],
            shape: vec![2],
            metadata: DistributionMetadata {
                distribution_type: "categorical".to_string(),
                normalized: false,
                parameter_names: vec![],
                tags: vec![],
            },
        }
    })
}

/// Strategy producing a valid `DistributionExport` with two binary variables (2×2 table).
fn joint_distribution_export_strategy() -> impl Strategy<Value = DistributionExport> {
    (positive_f64_vec(4), ascii_name(), ascii_name())
        .prop_filter("distinct variable names", |(_, a, b)| a != b)
        .prop_map(|(mut probs, v1, v2)| {
            probs.resize(4, 0.01);
            DistributionExport {
                variables: vec![v1, v2],
                cardinalities: vec![2, 2],
                probabilities: probs,
                shape: vec![2, 2],
                metadata: DistributionMetadata {
                    distribution_type: "categorical".to_string(),
                    normalized: false,
                    parameter_names: vec![],
                    tags: vec![],
                },
            }
        })
}

/// Strategy producing a valid `Factor` with one variable and cardinality 2.
fn unary_factor_strategy() -> impl Strategy<Value = Factor> {
    (positive_f64_vec(4), ascii_name()).prop_map(|(mut values, var)| {
        values.resize(2, 0.01);
        let arr = Array::from_shape_vec(vec![2], values)
            .expect("unary_factor_strategy: array creation")
            .into_dyn();
        Factor::new(format!("f_{}", var), vec![var], arr)
            .expect("unary_factor_strategy: factor creation")
    })
}

/// Strategy producing a valid `AnnealingConfig`.
fn annealing_config_strategy() -> impl Strategy<Value = AnnealingConfig> {
    (1usize..=200, 0.01f64..=100.0_f64, 1usize..=50).prop_map(
        |(num_steps, annealing_time, num_samples)| {
            AnnealingConfig::new(num_steps, annealing_time).with_samples(num_samples)
        },
    )
}

// ============================================================================
// Property 1: JSON round-trip for DistributionExport (binary variable)
// ============================================================================

proptest! {
    /// Round-trip: serialize a `DistributionExport` to JSON and deserialize back.
    /// Field equality is checked element-by-element (f64 values via exact bitwise
    /// comparison after JSON round-trip, which preserves full precision for finite
    /// values in this domain).
    #[test]
    fn prop_distribution_export_json_roundtrip(
        dist in binary_distribution_export_strategy()
    ) {
        let json = serde_json::to_string(&dist)
            .expect("prop_distribution_export_json_roundtrip: serialize");
        let restored: DistributionExport = serde_json::from_str(&json)
            .expect("prop_distribution_export_json_roundtrip: deserialize");

        prop_assert_eq!(&dist.variables, &restored.variables);
        prop_assert_eq!(&dist.cardinalities, &restored.cardinalities);
        prop_assert_eq!(&dist.shape, &restored.shape);
        prop_assert_eq!(dist.probabilities.len(), restored.probabilities.len());
        for (orig, got) in dist.probabilities.iter().zip(restored.probabilities.iter()) {
            prop_assert!(
                (orig - got).abs() < 1e-12,
                "probability mismatch: {} vs {}",
                orig,
                got
            );
        }
        prop_assert_eq!(
            &dist.metadata.distribution_type,
            &restored.metadata.distribution_type
        );
        prop_assert_eq!(dist.metadata.normalized, restored.metadata.normalized);
    }
}

// ============================================================================
// Property 2: deserializing arbitrary bytes never panics
// ============================================================================

proptest! {
    /// Deserializing arbitrary bytes as JSON into `DistributionExport` must never
    /// panic — all parse failures must surface as `Err`, not a panic.
    #[test]
    fn prop_no_panic_arbitrary_bytes_distribution_export(
        bytes in prop::collection::vec(any::<u8>(), 0..=256)
    ) {
        let _result: Result<DistributionExport, _> = serde_json::from_slice(&bytes);
        // Reaching here means no panic occurred.
    }

    /// Same no-panic guarantee for `ModelExport`.
    #[test]
    fn prop_no_panic_arbitrary_bytes_model_export(
        bytes in prop::collection::vec(any::<u8>(), 0..=256)
    ) {
        let _result: Result<ModelExport, _> = serde_json::from_slice(&bytes);
    }

    /// Same no-panic guarantee for `QuantRSAssignment`.
    #[test]
    fn prop_no_panic_arbitrary_bytes_assignment(
        bytes in prop::collection::vec(any::<u8>(), 0..=256)
    ) {
        let _result: Result<QuantRSAssignment, _> = serde_json::from_slice(&bytes);
    }
}

// ============================================================================
// Property 3: Factor normalization is idempotent
// ============================================================================

proptest! {
    /// Normalizing a factor once and then normalizing again must yield the same
    /// values (idempotence of `Factor::normalize()`).
    #[test]
    fn prop_normalize_idempotent(factor in unary_factor_strategy()) {
        let mut once = factor.clone();
        once.normalize();

        let mut twice = once.clone();
        twice.normalize();

        prop_assert_eq!(once.values.shape(), twice.values.shape());
        for (a, b) in once.values.iter().zip(twice.values.iter()) {
            prop_assert!(
                (a - b).abs() < 1e-12,
                "normalization not idempotent: {} vs {}",
                a,
                b
            );
        }
    }

    /// After normalization, the factor values must sum to 1 (provided all
    /// original values are positive, which the strategy guarantees).
    #[test]
    fn prop_normalize_sums_to_one(factor in unary_factor_strategy()) {
        let mut f = factor;
        f.normalize();
        let sum: f64 = f.values.iter().sum();
        prop_assert!(
            (sum - 1.0).abs() < 1e-10,
            "normalized factor does not sum to 1: sum = {}",
            sum
        );
    }
}

// ============================================================================
// Property 4: QuantRSAssignment JSON round-trip
// ============================================================================

proptest! {
    /// `QuantRSAssignment` must survive a JSON round-trip with identical
    /// assignment entries.
    #[test]
    fn prop_assignment_json_roundtrip(
        entries in prop::collection::hash_map(ascii_name(), 0usize..=7, 0..=8)
    ) {
        let assignment = QuantRSAssignment::new(entries.clone());
        let json = serde_json::to_string(&assignment)
            .expect("prop_assignment_json_roundtrip: serialize");
        let restored: QuantRSAssignment = serde_json::from_str(&json)
            .expect("prop_assignment_json_roundtrip: deserialize");

        prop_assert_eq!(assignment.to_hashmap(), restored.to_hashmap());
    }
}

// ============================================================================
// Property 5: AnnealingConfig JSON round-trip
// ============================================================================

proptest! {
    /// `AnnealingConfig` must survive a JSON round-trip with identical fields.
    #[test]
    fn prop_annealing_config_json_roundtrip(config in annealing_config_strategy()) {
        let json = serde_json::to_string(&config)
            .expect("prop_annealing_config_json_roundtrip: serialize");
        let restored: AnnealingConfig = serde_json::from_str(&json)
            .expect("prop_annealing_config_json_roundtrip: deserialize");

        prop_assert_eq!(config.num_steps, restored.num_steps);
        prop_assert_eq!(config.num_samples, restored.num_samples);
        prop_assert!(
            (config.annealing_time - restored.annealing_time).abs() < 1e-12,
            "annealing_time mismatch: {} vs {}",
            config.annealing_time,
            restored.annealing_time
        );
        prop_assert!(
            (config.initial_temperature - restored.initial_temperature).abs() < 1e-12,
            "initial_temperature mismatch"
        );
        prop_assert!(
            (config.final_temperature - restored.final_temperature).abs() < 1e-12,
            "final_temperature mismatch"
        );
    }
}

// ============================================================================
// Property 6: Factor → DistributionExport → Factor round-trip via QuantRSDistribution
// ============================================================================

proptest! {
    /// Converting a `Factor` to a `DistributionExport` and back must preserve
    /// the variable list and all values within floating-point tolerance.
    #[test]
    fn prop_factor_distribution_roundtrip(factor in unary_factor_strategy()) {
        let dist = factor
            .to_quantrs_distribution()
            .expect("prop_factor_distribution_roundtrip: to_quantrs_distribution");
        let restored = Factor::from_quantrs_distribution(&dist)
            .expect("prop_factor_distribution_roundtrip: from_quantrs_distribution");

        prop_assert_eq!(&factor.variables, &restored.variables);
        prop_assert_eq!(factor.values.shape(), restored.values.shape());
        for (orig, got) in factor.values.iter().zip(restored.values.iter()) {
            prop_assert!(
                (orig - got).abs() < 1e-12,
                "value mismatch in factor round-trip: {} vs {}",
                orig,
                got
            );
        }
    }

    /// Round-trip through JSON for a `DistributionExport` with a 2×2 joint table.
    #[test]
    fn prop_joint_distribution_export_json_roundtrip(
        dist in joint_distribution_export_strategy()
    ) {
        let json = serde_json::to_string(&dist)
            .expect("prop_joint_distribution_export_json_roundtrip: serialize");
        let restored: DistributionExport = serde_json::from_str(&json)
            .expect("prop_joint_distribution_export_json_roundtrip: deserialize");

        prop_assert_eq!(&dist.variables, &restored.variables);
        prop_assert_eq!(&dist.shape, &restored.shape);
        prop_assert_eq!(dist.probabilities.len(), restored.probabilities.len());
        for (a, b) in dist.probabilities.iter().zip(restored.probabilities.iter()) {
            prop_assert!((a - b).abs() < 1e-12, "{} vs {}", a, b);
        }
    }
}

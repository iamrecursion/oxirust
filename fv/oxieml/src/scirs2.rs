//! SciRS2 integration (feature: `scirs2`).
//!
//! Provides [`symbolic_regression()`] and [`symbolic_regression_with_names()`]
//! adapters that accept ndarray inputs from the SciRS2 ecosystem.

use ndarray::{Array1, Array2};

use crate::error::EmlError;
use crate::symreg::{DiscoveredFormula, SymRegConfig, SymRegEngine};

/// Run symbolic regression on ndarray data.
///
/// `features` is an (n_samples × n_features) matrix; `targets` is an n_samples vector.
/// Returns discovered formulas sorted by score (best first).
pub fn symbolic_regression(
    features: &Array2<f64>,
    targets: &Array1<f64>,
    config: &SymRegConfig,
) -> Result<Vec<DiscoveredFormula>, EmlError> {
    let n_samples = features.nrows();
    let n_features = features.ncols();

    if n_samples != targets.len() {
        return Err(EmlError::DimensionMismatch(n_samples, targets.len()));
    }
    if n_samples == 0 {
        return Err(EmlError::EmptyData);
    }

    // Convert ndarray rows to Vec<Vec<f64>>
    let inputs: Vec<Vec<f64>> = (0..n_samples).map(|i| features.row(i).to_vec()).collect();

    let targets_slice = targets
        .as_slice()
        .ok_or(EmlError::DimensionMismatch(n_samples, 0))?;

    let engine = SymRegEngine::new(config.clone());
    engine.discover(&inputs, targets_slice, n_features)
}

/// Run multi-output symbolic regression on ndarray data.
///
/// Each column of `targets_matrix` (n_samples × n_outputs) is treated as an
/// independent regression problem (`MultiOutputStrategy::Independent`).
/// Returns one [`Vec<DiscoveredFormula>`] per output column, sorted by score.
pub fn symbolic_regression_multi(
    features: &ndarray::Array2<f64>,
    targets_matrix: &ndarray::Array2<f64>,
    config: &SymRegConfig,
) -> Result<Vec<Vec<DiscoveredFormula>>, EmlError> {
    let n_samples = features.nrows();
    let n_features = features.ncols();
    let n_outputs = targets_matrix.ncols();

    if targets_matrix.nrows() != n_samples {
        return Err(EmlError::DimensionMismatch(
            n_samples,
            targets_matrix.nrows(),
        ));
    }
    if n_samples == 0 {
        return Err(EmlError::EmptyData);
    }

    // Convert feature matrix rows to Vec<Vec<f64>> (row-major, same as symbolic_regression).
    let inputs: Vec<Vec<f64>> = (0..n_samples).map(|i| features.row(i).to_vec()).collect();

    // Convert each output column to a Vec<f64>.
    let targets: Vec<Vec<f64>> = (0..n_outputs)
        .map(|j| targets_matrix.column(j).to_vec())
        .collect();

    let engine = SymRegEngine::new(config.clone());
    engine.discover_multi(&inputs, &targets, n_features)
}

/// Like [`symbolic_regression()`] but with named features for richer output.
///
/// `feature_names` labels each column; purely cosmetic (used in pretty-printing).
/// After regression, occurrences of `x0`, `x1`, … in the `pretty` field are
/// replaced with the corresponding names from `feature_names`.
pub fn symbolic_regression_with_names(
    features: &Array2<f64>,
    targets: &Array1<f64>,
    feature_names: &[&str],
    config: &SymRegConfig,
) -> Result<Vec<DiscoveredFormula>, EmlError> {
    if feature_names.len() != features.ncols() {
        return Err(EmlError::DimensionMismatch(
            features.ncols(),
            feature_names.len(),
        ));
    }

    let mut formulas = symbolic_regression(features, targets, config)?;

    // Replace x0, x1, … with feature names in pretty strings.
    // Process in reverse index order so that e.g. x10 is replaced before x1.
    for formula in &mut formulas {
        let mut pretty = formula.pretty.clone();
        for (i, name) in feature_names.iter().enumerate().rev() {
            pretty = pretty.replace(&format!("x{i}"), name);
        }
        formula.pretty = pretty;
    }

    Ok(formulas)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    #[test]
    fn test_ndarray_conversion() {
        // 3×2 matrix: rows are samples, cols are features
        let features =
            Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("test shape");
        let targets = array![10.0, 20.0, 30.0];

        // Use quick config for speed
        let config = SymRegConfig::quick();
        let result = symbolic_regression(&features, &targets, &config);
        // Should succeed (not error on conversion)
        assert!(result.is_ok());
    }

    #[test]
    fn test_dimension_mismatch() {
        let features =
            Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("test shape");
        // Wrong length: 2 targets for 3 samples
        let targets = array![10.0, 20.0];
        let config = SymRegConfig::quick();

        let result = symbolic_regression(&features, &targets, &config);
        assert!(result.is_err());
        match result {
            Err(EmlError::DimensionMismatch(3, 2)) => {}
            other => panic!("expected DimensionMismatch(3, 2), got {other:?}"),
        }
    }

    #[test]
    fn test_feature_names_mismatch() {
        let features =
            Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("test shape");
        let targets = array![10.0, 20.0, 30.0];
        let config = SymRegConfig::quick();

        // 3 names for 2 features
        let result = symbolic_regression_with_names(&features, &targets, &["a", "b", "c"], &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_symbolic_regression_linear() {
        // Generate y = 2*x data (single feature)
        let n = 20;
        let mut feat_data = Vec::with_capacity(n);
        let mut tgt_data = Vec::with_capacity(n);
        for i in 0..n {
            let x = (i as f64) * 0.5 + 0.1;
            feat_data.push(x);
            tgt_data.push(2.0 * x);
        }

        let features = Array2::from_shape_vec((n, 1), feat_data).expect("test shape");
        let targets = Array1::from_vec(tgt_data);

        let config = SymRegConfig {
            max_depth: 2,
            max_iter: 500,
            num_restarts: 2,
            ..SymRegConfig::default()
        };
        let formulas =
            symbolic_regression(&features, &targets, &config).expect("regression should succeed");

        assert!(!formulas.is_empty(), "should discover at least one formula");
        // The best formula should have reasonably low MSE for y=2x
        let best = &formulas[0];
        assert!(
            best.mse < 50.0,
            "best MSE should be reasonable, got {}",
            best.mse
        );
    }

    #[test]
    fn test_with_names_replaces_vars() {
        // Minimal test: 2 features named "mass" and "vel"
        let features = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0],
        )
        .expect("test shape");
        let targets = array![2.0, 6.0, 12.0, 20.0, 30.0]; // ~ mass * vel

        let config = SymRegConfig::quick();
        let formulas =
            symbolic_regression_with_names(&features, &targets, &["mass", "vel"], &config)
                .expect("regression should succeed");

        // If any formula mentions variables, they should use names not x0/x1
        for formula in &formulas {
            assert!(
                !formula.pretty.contains("x0") && !formula.pretty.contains("x1"),
                "pretty should use feature names, got: {}",
                formula.pretty
            );
        }
    }
}

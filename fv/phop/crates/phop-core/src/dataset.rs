//! Dataset ingestion: `(X, y)` pairs from arrays or CSV.

use crate::error::{PhopError, Result};
use crate::rng::SplitMix64;
use scirs2_core::ndarray::{Array1, Array2};

/// Per-column affine transform `(value - mean) / std` learned from a [`DataSet`].
///
/// Standardization helps gradient fitting when feature/target scales differ by orders of
/// magnitude. The stored means and stds let predictions made in standardized space be mapped
/// back to the original units via [`Standardizer::inverse_target`].
#[derive(Debug, Clone)]
pub struct Standardizer {
    /// Mean of each feature column.
    pub feat_mean: Vec<f64>,
    /// Standard deviation of each feature column (floored at a small epsilon).
    pub feat_std: Vec<f64>,
    /// Mean of the target.
    pub y_mean: f64,
    /// Standard deviation of the target (floored at a small epsilon).
    pub y_std: f64,
}

impl Standardizer {
    /// Map a target value from standardized space back to the original units.
    #[must_use]
    pub fn inverse_target(&self, y_std_space: f64) -> f64 {
        y_std_space * self.y_std + self.y_mean
    }

    /// Map a whole target vector back to the original units.
    #[must_use]
    pub fn inverse_targets(&self, ys: &Array1<f64>) -> Array1<f64> {
        ys.mapv(|v| self.inverse_target(v))
    }
}

/// Mean and (population) standard deviation of a slice, with the std floored at `eps` so the
/// transform stays finite for constant columns.
fn mean_std(values: impl Iterator<Item = f64>, n: usize) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let vals: Vec<f64> = values.collect();
    let mean = vals.iter().sum::<f64>() / n as f64;
    let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
    (mean, var.sqrt().max(1e-12))
}

/// A dataset of input features `x` (shape `[n_rows, n_vars]`) and targets `y` (`[n_rows]`).
#[derive(Debug, Clone)]
pub struct DataSet {
    /// Feature matrix, one row per observation.
    pub x: Array2<f64>,
    /// Target vector, one entry per observation.
    pub y: Array1<f64>,
    /// Names of the feature columns (length `n_vars`).
    pub feature_names: Vec<String>,
    /// Name of the target column.
    pub target_name: String,
}

impl DataSet {
    /// Build a dataset from in-memory arrays.
    ///
    /// # Errors
    /// Returns [`PhopError::ShapeMismatch`] if the row counts of `x` and `y` differ.
    pub fn from_arrays(x: Array2<f64>, y: Array1<f64>) -> Result<Self> {
        if x.nrows() != y.len() {
            return Err(PhopError::ShapeMismatch(format!(
                "x has {} rows but y has {} entries",
                x.nrows(),
                y.len()
            )));
        }
        let n_vars = x.ncols();
        let feature_names = (0..n_vars).map(|i| format!("x{i}")).collect();
        Ok(Self {
            x,
            y,
            feature_names,
            target_name: "y".to_string(),
        })
    }

    /// Number of feature variables.
    #[must_use]
    pub fn n_vars(&self) -> usize {
        self.x.ncols()
    }

    /// Number of observations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.y.len()
    }

    /// Whether the dataset is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.y.is_empty()
    }

    /// Produce a z-scored copy of the dataset together with the [`Standardizer`] that maps
    /// predictions back to the original target units.
    ///
    /// Each feature column and the target are centered and scaled to unit variance; constant
    /// columns (zero variance) are left centered with a unit scale so the transform stays finite.
    #[must_use]
    pub fn standardized(&self) -> (DataSet, Standardizer) {
        let n = self.len();
        let nv = self.n_vars();
        let mut feat_mean = vec![0.0; nv];
        let mut feat_std = vec![1.0; nv];
        for j in 0..nv {
            let (m, s) = mean_std(self.x.column(j).iter().copied(), n);
            feat_mean[j] = m;
            feat_std[j] = s;
        }
        let (y_mean, y_std) = mean_std(self.y.iter().copied(), n);

        let mut xz = self.x.clone();
        for j in 0..nv {
            let (m, s) = (feat_mean[j], feat_std[j]);
            xz.column_mut(j).mapv_inplace(|v| (v - m) / s);
        }
        let yz = self.y.mapv(|v| (v - y_mean) / y_std);

        let std = Standardizer {
            feat_mean,
            feat_std,
            y_mean,
            y_std,
        };
        let ds = DataSet {
            x: xz,
            y: yz,
            feature_names: self.feature_names.clone(),
            target_name: self.target_name.clone(),
        };
        (ds, std)
    }

    /// Build a sub-dataset from the given row indices (used by minibatching).
    ///
    /// # Errors
    /// Returns [`PhopError::ShapeMismatch`] if any index is out of range.
    pub fn select(&self, rows: &[usize]) -> Result<DataSet> {
        let nv = self.n_vars();
        let mut x_flat = Vec::with_capacity(rows.len() * nv);
        let mut y_vec = Vec::with_capacity(rows.len());
        for &r in rows {
            if r >= self.len() {
                return Err(PhopError::ShapeMismatch(format!(
                    "row index {r} out of range (len {})",
                    self.len()
                )));
            }
            for j in 0..nv {
                x_flat.push(self.x[[r, j]]);
            }
            y_vec.push(self.y[r]);
        }
        let x = Array2::from_shape_vec((rows.len(), nv), x_flat)
            .map_err(|e| PhopError::ShapeMismatch(e.to_string()))?;
        Ok(DataSet {
            x,
            y: Array1::from(y_vec),
            feature_names: self.feature_names.clone(),
            target_name: self.target_name.clone(),
        })
    }

    /// Partition the data axis into shuffled minibatches of (at most) `size` rows.
    ///
    /// The shuffle is seeded for reproducibility (Risk T3 mitigation: bounds per-step memory by
    /// letting the optimizer consume the data in chunks). A `size` of `0` or one `>=` the row
    /// count yields a single batch containing all rows.
    #[must_use]
    pub fn minibatches(&self, size: usize, seed: u64) -> Vec<DataSet> {
        let n = self.len();
        if n == 0 {
            return Vec::new();
        }
        let mut idx: Vec<usize> = (0..n).collect();
        SplitMix64::new(seed).shuffle(&mut idx);
        let chunk = if size == 0 { n } else { size.min(n) };
        idx.chunks(chunk)
            .map(|rows| self.select(rows).expect("indices in range"))
            .collect()
    }

    /// Load a dataset from a CSV file.
    ///
    /// The file is expected to have a header row. By default the **last** column is
    /// taken as the target `y` and all preceding columns as features `x`.
    ///
    /// # Errors
    /// Returns an error if the file cannot be read, parsed, or has fewer than two columns.
    pub fn from_csv<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        Self::from_csv_with_target(path, None)
    }

    /// Load a dataset from a CSV file, optionally choosing which column is the target.
    ///
    /// `target` is a 0-based column index; `None` selects the last column. All other columns
    /// become features `x`, preserving their header order.
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or parsed, has fewer than two columns, or if
    /// `target` is out of range.
    pub fn from_csv_with_target<P: AsRef<std::path::Path>>(
        path: P,
        target: Option<usize>,
    ) -> Result<Self> {
        let (headers, rows) = parse_csv(path)?;
        let n_cols = headers.len();
        let target_col = target.unwrap_or(n_cols - 1);
        if target_col >= n_cols {
            return Err(PhopError::ShapeMismatch(format!(
                "target column {target_col} out of range (CSV has {n_cols} columns)"
            )));
        }
        let features: Vec<usize> = (0..n_cols).filter(|&j| j != target_col).collect();
        Self::assemble(&headers, &rows, &features, target_col)
    }

    /// Load a dataset from a CSV file selecting an explicit subset of **feature** columns and a
    /// target column (all 0-based indices). Feature columns appear in the order given.
    ///
    /// # Errors
    /// Returns an error if the file cannot be read/parsed, has fewer than two columns, any index is
    /// out of range, or the target appears among the features.
    pub fn from_csv_columns<P: AsRef<std::path::Path>>(
        path: P,
        features: &[usize],
        target: usize,
    ) -> Result<Self> {
        let (headers, rows) = parse_csv(path)?;
        let n_cols = headers.len();
        if target >= n_cols || features.iter().any(|&j| j >= n_cols) {
            return Err(PhopError::ShapeMismatch(format!(
                "column index out of range (CSV has {n_cols} columns)"
            )));
        }
        if features.is_empty() {
            return Err(PhopError::ShapeMismatch(
                "at least one feature column is required".to_string(),
            ));
        }
        if features.contains(&target) {
            return Err(PhopError::ShapeMismatch(
                "target column cannot also be a feature".to_string(),
            ));
        }
        Self::assemble(&headers, &rows, features, target)
    }

    /// Build a dataset from parsed rows by selecting feature columns and a target column.
    fn assemble(
        headers: &[String],
        rows: &[Vec<f64>],
        features: &[usize],
        target: usize,
    ) -> Result<Self> {
        let n_rows = rows.len();
        let n_vars = features.len();
        let mut x_flat = Vec::with_capacity(n_rows * n_vars);
        let mut y_vec = Vec::with_capacity(n_rows);
        for row in rows {
            for &j in features {
                x_flat.push(row[j]);
            }
            y_vec.push(row[target]);
        }
        let x = Array2::from_shape_vec((n_rows, n_vars), x_flat)
            .map_err(|e| PhopError::ShapeMismatch(e.to_string()))?;
        let feature_names = features.iter().map(|&j| headers[j].clone()).collect();
        Ok(Self {
            x,
            y: Array1::from(y_vec),
            feature_names,
            target_name: headers[target].clone(),
        })
    }
}

/// Parse a headered numeric CSV into `(headers, rows)`; every field must parse as `f64`.
fn parse_csv<P: AsRef<std::path::Path>>(path: P) -> Result<(Vec<String>, Vec<Vec<f64>>)> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)?;
    let headers: Vec<String> = rdr.headers()?.iter().map(str::to_string).collect();
    let n_cols = headers.len();
    if n_cols < 2 {
        return Err(PhopError::ShapeMismatch(
            "CSV must have at least two columns (>=1 feature + target)".to_string(),
        ));
    }
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for rec in rdr.records() {
        let rec = rec?;
        if rec.len() != n_cols {
            return Err(PhopError::ShapeMismatch(format!(
                "row {} has {} fields, expected {n_cols}",
                rows.len(),
                rec.len()
            )));
        }
        let mut row = Vec::with_capacity(n_cols);
        for field in rec.iter() {
            row.push(
                field
                    .trim()
                    .parse::<f64>()
                    .map_err(|_| PhopError::Parse(format!("cannot parse '{field}' as f64")))?,
            );
        }
        rows.push(row);
    }
    Ok((headers, rows))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_arrays_checks_shape() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array1::from(vec![1.0, 2.0, 3.0]);
        let ds = DataSet::from_arrays(x, y).unwrap();
        assert_eq!(ds.n_vars(), 2);
        assert_eq!(ds.len(), 3);
        assert!(!ds.is_empty());
    }

    #[test]
    fn from_arrays_rejects_mismatch() {
        let x = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();
        let y = Array1::from(vec![1.0]);
        assert!(DataSet::from_arrays(x, y).is_err());
    }

    #[test]
    fn standardize_centers_and_scales() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from(vec![10.0, 20.0, 30.0, 40.0]);
        let ds = DataSet::from_arrays(x, y).unwrap();
        let (z, std) = ds.standardized();

        // Standardized feature column has ~zero mean and ~unit variance.
        let col_mean = z.x.column(0).iter().sum::<f64>() / 4.0;
        assert!(col_mean.abs() < 1e-9, "mean = {col_mean}");
        // Inverse transform round-trips the target.
        for (orig, zz) in ds.y.iter().zip(z.y.iter()) {
            assert!((std.inverse_target(*zz) - orig).abs() < 1e-9);
        }
    }

    #[test]
    fn minibatches_partition_all_rows() {
        let x = Array2::from_shape_vec((10, 1), (0..10).map(f64::from).collect()).unwrap();
        let y = Array1::from((0..10).map(f64::from).collect::<Vec<_>>());
        let ds = DataSet::from_arrays(x, y).unwrap();

        let batches = ds.minibatches(3, 123);
        assert_eq!(batches.len(), 4); // 3 + 3 + 3 + 1
        let total: usize = batches.iter().map(DataSet::len).sum();
        assert_eq!(total, 10);

        // Determinism: same seed -> same partition sizes and contents.
        let again = ds.minibatches(3, 123);
        for (a, b) in batches.iter().zip(&again) {
            assert_eq!(a.y, b.y);
        }
    }

    #[test]
    fn from_csv_with_target_selects_column() {
        use std::io::Write;
        let dir = std::env::temp_dir();
        let path = dir.join("phop_test_target.csv");
        let mut f = std::fs::File::create(&path).unwrap();
        // target is the FIRST column here.
        writeln!(f, "y,a,b").unwrap();
        writeln!(f, "10,1,2").unwrap();
        writeln!(f, "20,3,4").unwrap();
        drop(f);

        let ds = DataSet::from_csv_with_target(&path, Some(0)).unwrap();
        assert_eq!(ds.target_name, "y");
        assert_eq!(ds.feature_names, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(ds.n_vars(), 2);
        assert!((ds.y[0] - 10.0).abs() < 1e-12);
        assert!((ds.x[[1, 0]] - 3.0).abs() < 1e-12);

        // Feature subset: keep only column "b" (index 2) as the sole feature, target = col 0.
        let ds2 = DataSet::from_csv_columns(&path, &[2], 0).unwrap();
        assert_eq!(ds2.n_vars(), 1);
        assert_eq!(ds2.feature_names, vec!["b".to_string()]);
        assert!((ds2.x[[0, 0]] - 2.0).abs() < 1e-12);
        // Target listed as a feature is rejected.
        assert!(DataSet::from_csv_columns(&path, &[0], 0).is_err());
        std::fs::remove_file(&path).ok();
    }
}

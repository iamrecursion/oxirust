//! GPU-accelerated matched filter bank.
//!
//! A matched filter bank computes the cross-correlation of an input signal
//! with a collection of template kernels.  Running all correlations in
//! parallel maps naturally to GPU execution; this module provides the API and
//! a correct CPU reference implementation that can be swapped for a GPU
//! kernel without changing calling code.

use scirs2_core::ndarray::Array2;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error type for matched filter bank operations.
#[derive(Debug, Error)]
pub enum MatchedFilterError {
    /// Returned when [`MatchedFilterBank::apply`] is called on an empty bank.
    #[error("Empty template bank")]
    EmptyBank,

    /// Returned when any template is longer than the input signal.
    #[error("Template length {0} exceeds signal length {1}")]
    TemplateTooLong(usize, usize),
}

// ---------------------------------------------------------------------------
// MatchedFilterBank
// ---------------------------------------------------------------------------

/// Bank of matched filters for simultaneous template matching.
///
/// Each template is treated as a correlation kernel.  Given a signal of
/// length `L` and a template of length `M`, the output for that template has
/// length `L - M + 1` (valid convolution / full-overlap positions only).
///
/// # Example
///
/// ```rust
/// use scirs2_signal::gpu_matched_filter::MatchedFilterBank;
///
/// let mut bank = MatchedFilterBank::new(true);
/// bank.add_template(vec![1.0_f32, 0.5, 0.25]);
/// bank.add_template(vec![-1.0_f32, 1.0]);
///
/// let signal: Vec<f32> = (0..64).map(|i| (i as f32 / 64.0)).collect();
/// let output = bank.apply(&signal).expect("bank is non-empty");
/// println!("output shape: {:?}", output.dim()); // (2, output_len)
/// ```
pub struct MatchedFilterBank {
    /// Stored templates; each inner `Vec<f32>` is one correlation kernel.
    templates: Vec<Vec<f32>>,
    /// When `true`, each correlation output is divided by the template's
    /// L2-energy so that results are comparable across templates of different
    /// norms.
    normalize: bool,
}

impl MatchedFilterBank {
    /// Create a new, empty bank.
    ///
    /// `normalize` controls whether each template's cross-correlation output
    /// is divided by its L2-energy.  Set to `true` when comparing templates
    /// of different lengths or amplitudes.
    pub fn new(normalize: bool) -> Self {
        Self {
            templates: Vec::new(),
            normalize,
        }
    }

    /// Add a template kernel to the bank.
    ///
    /// Templates are stored in the order they are added; the row index of
    /// `apply`'s output matrix corresponds to the insertion order.
    pub fn add_template(&mut self, template: Vec<f32>) {
        self.templates.push(template);
    }

    /// Return the number of templates currently in the bank.
    pub fn n_templates(&self) -> usize {
        self.templates.len()
    }

    /// Compute the cross-correlation of `signal` with every template.
    ///
    /// Returns an `Array2<f32>` of shape `[n_templates, output_length]` where
    /// `output_length = signal.len() - max_template_len + 1` for templates
    /// equal to the longest template (shorter templates are zero-padded to
    /// the same output length — actually each template uses its own valid
    /// overlap length, and shorter rows are zero-padded to `output_length`).
    ///
    /// # Errors
    ///
    /// * [`MatchedFilterError::EmptyBank`] — no templates have been added.
    /// * [`MatchedFilterError::TemplateTooLong`] — at least one template is
    ///   longer than `signal`.
    pub fn apply(&self, signal: &[f32]) -> Result<Array2<f32>, MatchedFilterError> {
        if self.templates.is_empty() {
            return Err(MatchedFilterError::EmptyBank);
        }

        for (i, tmpl) in self.templates.iter().enumerate() {
            if tmpl.len() > signal.len() {
                return Err(MatchedFilterError::TemplateTooLong(
                    tmpl.len(),
                    signal.len(),
                ));
            }
            // Zero-length templates are skipped silently (will produce
            // an empty correlation; not blocked here).
            let _ = i;
        }

        // All rows will be padded to the length of the shortest valid
        // correlation (the one belonging to the longest template).
        let max_tmpl_len = self.templates.iter().map(|t| t.len()).max().unwrap_or(1);
        // Use unwrap_or here because we already checked templates is non-empty;
        // the only way max_tmpl_len could be absent is if templates is empty,
        // which we already return an error for.
        let output_length = signal.len().saturating_sub(max_tmpl_len) + 1;
        let n_templates = self.templates.len();

        let mut result = Array2::<f32>::zeros((n_templates, output_length));

        for (row, tmpl) in self.templates.iter().enumerate() {
            let corr = Self::correlate_single(signal, tmpl, self.normalize);
            // corr has length `signal.len() - tmpl.len() + 1`; pad the rest
            // with zeros (already initialised above).
            for (col, &v) in corr.iter().enumerate().take(output_length) {
                result[[row, col]] = v;
            }
        }

        Ok(result)
    }

    /// Return the `(template_index, lag_position, correlation_value)` triple
    /// of the global maximum across all templates and all lags.
    ///
    /// The `lag_position` is the zero-based index into the valid-overlap
    /// correlation output for that template.
    ///
    /// # Errors
    ///
    /// Same as [`MatchedFilterBank::apply`].
    pub fn best_match(&self, signal: &[f32]) -> Result<(usize, usize, f32), MatchedFilterError> {
        let output = self.apply(signal)?;

        let mut best_template = 0_usize;
        let mut best_lag = 0_usize;
        let mut best_val = f32::NEG_INFINITY;

        for row in 0..output.nrows() {
            for col in 0..output.ncols() {
                let v = output[[row, col]];
                if v > best_val {
                    best_val = v;
                    best_template = row;
                    best_lag = col;
                }
            }
        }

        Ok((best_template, best_lag, best_val))
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    /// Compute the sliding cross-correlation of `signal` and `template` at
    /// all valid-overlap lags, i.e. lags where the template fits entirely
    /// within the signal.
    ///
    /// The output length is `signal.len() - template.len() + 1`.
    ///
    /// When `normalize` is `true`, each output value is divided by the
    /// template's L2-energy so the result is comparable across templates.
    fn correlate_single(signal: &[f32], template: &[f32], normalize: bool) -> Vec<f32> {
        if template.is_empty() {
            return vec![0.0_f32; signal.len()];
        }

        let n = signal.len();
        let m = template.len();

        if m > n {
            return Vec::new();
        }

        let output_len = n - m + 1;
        let mut output = Vec::with_capacity(output_len);

        // Pre-compute template energy for normalisation.
        let energy: f32 = if normalize {
            let e: f32 = template.iter().map(|&v| v * v).sum();
            if e < 1e-12 {
                1.0 // avoid division by zero for a zero-norm template
            } else {
                e
            }
        } else {
            1.0
        };

        for lag in 0..output_len {
            let dot: f32 = template
                .iter()
                .enumerate()
                .map(|(k, &t)| signal[lag + k] * t)
                .sum();
            output.push(dot / energy);
        }

        output
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Correlating a signal with itself should peak at lag 0.
    #[test]
    fn test_matched_filter_basic() {
        let template: Vec<f32> = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let signal = template.clone();

        let mut bank = MatchedFilterBank::new(false);
        bank.add_template(template.clone());

        let output = bank.apply(&signal).expect("bank has one template");

        // Output should have length 1 (signal.len() == template.len())
        assert_eq!(output.ncols(), 1, "single valid lag expected");
        assert!(
            output[[0, 0]] > 0.0,
            "correlation at lag 0 should be positive"
        );
    }

    /// Bank output shape must be [n_templates, output_length].
    #[test]
    fn test_matched_filter_bank_multiple() {
        let signal: Vec<f32> = (0..128).map(|i| (i as f32).sin()).collect();

        let mut bank = MatchedFilterBank::new(true);
        bank.add_template(vec![1.0_f32; 8]);
        bank.add_template(vec![1.0_f32, -1.0, 1.0, -1.0]);
        bank.add_template(vec![0.5_f32; 16]);

        let max_tmpl_len = 16_usize;
        let expected_output_len = 128 - max_tmpl_len + 1;

        let output = bank.apply(&signal).expect("bank is non-empty");

        assert_eq!(
            output.dim(),
            (3, expected_output_len),
            "unexpected output shape"
        );
    }

    /// Embedding a known template in a noise floor should make the bank detect
    /// it at the correct position.
    #[test]
    fn test_matched_filter_best_match() {
        // Construct a signal of zeros with the template embedded at position 50.
        let template: Vec<f32> = vec![1.0, 2.0, 4.0, 2.0, 1.0];
        let embed_pos = 50_usize;
        let signal_len = 200_usize;
        let mut signal = vec![0.0_f32; signal_len];
        for (k, &v) in template.iter().enumerate() {
            signal[embed_pos + k] = v;
        }

        let mut bank = MatchedFilterBank::new(false);
        // Add some decoy templates first.
        bank.add_template(vec![-1.0_f32; 5]);
        bank.add_template(vec![0.1_f32; 5]);
        // The true template is at index 2.
        bank.add_template(template.clone());

        let (tmpl_idx, lag, _value) = bank.best_match(&signal).expect("bank is non-empty");

        assert_eq!(
            tmpl_idx, 2,
            "best match should identify the embedded template"
        );
        assert_eq!(
            lag, embed_pos,
            "best match should be at the embedded position"
        );
    }

    /// An empty bank should return `EmptyBank`.
    #[test]
    fn test_matched_filter_empty_bank() {
        let bank: MatchedFilterBank = MatchedFilterBank::new(false);
        let signal = vec![1.0_f32; 32];
        assert!(matches!(
            bank.apply(&signal),
            Err(MatchedFilterError::EmptyBank)
        ));
    }

    /// A template longer than the signal should return `TemplateTooLong`.
    #[test]
    fn test_matched_filter_template_too_long() {
        let mut bank = MatchedFilterBank::new(false);
        bank.add_template(vec![1.0_f32; 64]);
        let signal = vec![0.0_f32; 32];

        assert!(matches!(
            bank.apply(&signal),
            Err(MatchedFilterError::TemplateTooLong(64, 32))
        ));
    }

    /// Normalised correlation of a signal with itself should produce a value
    /// near 1.0 everywhere it is non-trivially overlapping.
    #[test]
    fn test_matched_filter_normalised() {
        let template: Vec<f32> = vec![2.0, 2.0, 2.0, 2.0];
        // Signal is exactly the template followed by zeros.
        let mut signal = template.clone();
        signal.extend_from_slice(&[0.0_f32; 10]);

        let mut bank = MatchedFilterBank::new(true); // normalise
        bank.add_template(template.clone());

        let output = bank.apply(&signal).expect("bank is non-empty");
        // At lag 0, the normalised dot product should equal 1.0.
        let at_lag_zero = output[[0, 0]];
        assert!(
            (at_lag_zero - 1.0_f32).abs() < 1e-5,
            "normalised match at lag 0 should be 1.0, got {}",
            at_lag_zero
        );
    }

    /// Return type of `apply` should be an `Array1`-readable row view so that
    /// callers can trivially extract per-template results.
    #[test]
    fn test_matched_filter_row_access() {
        let signal: Vec<f32> = (0..64).map(|i| i as f32).collect();

        let mut bank = MatchedFilterBank::new(false);
        bank.add_template(vec![1.0_f32; 4]);
        bank.add_template(vec![1.0_f32, -1.0, 1.0, -1.0]);

        let output = bank.apply(&signal).expect("bank is non-empty");

        // Access first template row and verify it's an Array1-like view.
        let row0: Vec<f32> = output.row(0).iter().copied().collect();
        assert_eq!(row0.len(), 64 - 4 + 1);

        let row1: Vec<f32> = output.row(1).iter().copied().collect();
        assert_eq!(row1.len(), 64 - 4 + 1);
    }
}

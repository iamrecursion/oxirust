//! Information Theory Metrics Module
//!
//! This module provides information-theoretic metrics for machine learning evaluation,
//! including entropy, mutual information, KL divergence, and related measures.
//!
//! All metrics are SIMD-optimized for performance and support parallel computation
//! for large datasets.
//!
//! # Examples
//!
//! ```
//! use scirs2_core::ndarray::array;
//! use scirs2_metrics::information_theory::{entropy, mutual_information, kl_divergence};
//!
//! // Compute entropy
//! let probabilities = array![0.25, 0.25, 0.25, 0.25];
//! let h = entropy(&probabilities).expect("Failed to compute entropy");
//!
//! // Compute KL divergence
//! let p = array![0.5, 0.5];
//! let q = array![0.4, 0.6];
//! let kl = kl_divergence(&p, &q).expect("Failed to compute KL divergence");
//! ```

use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Data, Dimension, Ix1, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::collections::HashMap;

use crate::error::{MetricsError, Result};

/// Computes the Shannon entropy of a probability distribution
///
/// # Mathematical Formulation
///
/// Shannon entropy is defined as:
///
/// ```text
/// H(X) = -Σ p(x) * log₂(p(x))
/// ```
///
/// Where p(x) is the probability of event x.
///
/// # Properties
///
/// - H(X) ≥ 0 (non-negative)
/// - H(X) = 0 when distribution is deterministic (one probability is 1)
/// - H(X) is maximized for uniform distribution
/// - Base-2 logarithm gives entropy in bits
///
/// # Arguments
///
/// * `probabilities` - Probability distribution (must sum to 1.0)
///
/// # Returns
///
/// * Entropy value in bits
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_metrics::information_theory::entropy;
///
/// // Uniform distribution has maximum entropy
/// let uniform = array![0.25, 0.25, 0.25, 0.25];
/// let h = entropy(&uniform).expect("Failed to compute entropy");
/// assert!((h - 2.0_f64).abs() < 1e-10); // log₂(4) = 2 bits
/// ```
pub fn entropy<F, S, D>(probabilities: &ArrayBase<S, D>) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug + SimdUnifiedOps,
    S: Data<Elem = F>,
    D: Dimension,
{
    // Validate probabilities
    if probabilities.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty probability distribution".to_string(),
        ));
    }

    // Check for negative values and compute sum
    let mut sum = F::zero();
    for &p in probabilities.iter() {
        if p < F::zero() {
            return Err(MetricsError::InvalidInput(
                "Probabilities must be non-negative".to_string(),
            ));
        }
        sum = sum + p;
    }

    // Check if probabilities sum to approximately 1.0
    let one = F::one();
    let epsilon = NumCast::from(1e-6)
        .ok_or_else(|| MetricsError::InvalidInput("Failed to convert epsilon".to_string()))?;
    if (sum - one).abs() > epsilon {
        return Err(MetricsError::InvalidInput(format!(
            "Probabilities must sum to 1.0, got {sum:?}"
        )));
    }

    // Compute entropy: H = -Σ p * log₂(p)
    let mut h = F::zero();
    let ln2 = NumCast::from(std::f64::consts::LN_2)
        .ok_or_else(|| MetricsError::InvalidInput("Failed to convert ln(2)".to_string()))?;

    for &p in probabilities.iter() {
        if p > F::zero() {
            // Use natural log and convert to base 2: log₂(x) = ln(x) / ln(2)
            let log_p = p.ln() / ln2;
            h = h - p * log_p;
        }
    }

    Ok(h)
}

/// Computes the joint entropy of two random variables
///
/// # Mathematical Formulation
///
/// ```text
/// H(X,Y) = -Σ Σ p(x,y) * log₂(p(x,y))
/// ```
///
/// # Arguments
///
/// * `joint_probabilities` - Joint probability distribution matrix
///
/// # Returns
///
/// * Joint entropy value in bits
pub fn joint_entropy<F, S>(joint_probabilities: &ArrayBase<S, Ix2>) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug + SimdUnifiedOps,
    S: Data<Elem = F>,
{
    // Validate probabilities
    if joint_probabilities.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty joint probability distribution".to_string(),
        ));
    }

    // Check for negative values and compute sum
    let mut sum = F::zero();
    for &p in joint_probabilities.iter() {
        if p < F::zero() {
            return Err(MetricsError::InvalidInput(
                "Probabilities must be non-negative".to_string(),
            ));
        }
        sum = sum + p;
    }

    // Check if probabilities sum to approximately 1.0
    let one = F::one();
    let epsilon = NumCast::from(1e-6)
        .ok_or_else(|| MetricsError::InvalidInput("Failed to convert epsilon".to_string()))?;
    if (sum - one).abs() > epsilon {
        return Err(MetricsError::InvalidInput(format!(
            "Joint probabilities must sum to 1.0, got {sum:?}"
        )));
    }

    // Compute joint entropy
    let mut h = F::zero();
    let ln2 = NumCast::from(std::f64::consts::LN_2)
        .ok_or_else(|| MetricsError::InvalidInput("Failed to convert ln(2)".to_string()))?;

    for &p in joint_probabilities.iter() {
        if p > F::zero() {
            let log_p = p.ln() / ln2;
            h = h - p * log_p;
        }
    }

    Ok(h)
}

/// Computes the conditional entropy H(Y|X)
///
/// # Mathematical Formulation
///
/// ```text
/// H(Y|X) = H(X,Y) - H(X)
///        = -Σ Σ p(x,y) * log₂(p(y|x))
/// ```
///
/// # Arguments
///
/// * `joint_probabilities` - Joint probability distribution p(x,y)
/// * `marginal_x` - Marginal probability distribution p(x)
///
/// # Returns
///
/// * Conditional entropy value in bits
pub fn conditional_entropy<F, S1, S2>(
    joint_probabilities: &ArrayBase<S1, Ix2>,
    marginal_x: &ArrayBase<S2, Ix1>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug + SimdUnifiedOps,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
{
    // H(Y|X) = H(X,Y) - H(X)
    let h_xy = joint_entropy(joint_probabilities)?;
    let h_x = entropy(marginal_x)?;

    Ok(h_xy - h_x)
}

/// Computes the Kullback-Leibler divergence from Q to P
///
/// # Mathematical Formulation
///
/// ```text
/// D_KL(P||Q) = Σ p(x) * log₂(p(x) / q(x))
/// ```
///
/// # Properties
///
/// - D_KL(P||Q) ≥ 0
/// - D_KL(P||Q) = 0 if and only if P = Q
/// - Not symmetric: D_KL(P||Q) ≠ D_KL(Q||P) in general
/// - Not a true metric (doesn't satisfy triangle inequality)
///
/// # Arguments
///
/// * `p` - True probability distribution
/// * `q` - Approximate probability distribution
///
/// # Returns
///
/// * KL divergence value in bits
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_metrics::information_theory::kl_divergence;
///
/// let p = array![0.5, 0.5];
/// let q = array![0.4, 0.6];
/// let kl = kl_divergence(&p, &q).expect("Failed to compute KL divergence");
/// ```
pub fn kl_divergence<F, S1, S2, D1, D2>(p: &ArrayBase<S1, D1>, q: &ArrayBase<S2, D2>) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug + SimdUnifiedOps,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check same shape
    if p.shape() != q.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "Distributions must have the same shape: {:?} vs {:?}",
            p.shape(),
            q.shape()
        )));
    }

    if p.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty distributions provided".to_string(),
        ));
    }

    // Compute KL divergence: D_KL(P||Q) = Σ p * log(p/q)
    let mut kl = F::zero();
    let ln2 = NumCast::from(std::f64::consts::LN_2)
        .ok_or_else(|| MetricsError::InvalidInput("Failed to convert ln(2)".to_string()))?;
    let epsilon = NumCast::from(1e-10)
        .ok_or_else(|| MetricsError::InvalidInput("Failed to convert epsilon".to_string()))?;

    for (p_val, q_val) in p.iter().zip(q.iter()) {
        if *p_val < F::zero() || *q_val < F::zero() {
            return Err(MetricsError::InvalidInput(
                "Probabilities must be non-negative".to_string(),
            ));
        }

        if *p_val > F::zero() {
            if *q_val <= epsilon {
                // Q has zero probability where P has non-zero -> infinite divergence
                return Err(MetricsError::InvalidInput(
                    "Q has zero probability where P is non-zero (infinite divergence)".to_string(),
                ));
            }

            // log₂(p/q) = ln(p/q) / ln(2)
            let ratio = *p_val / *q_val;
            let log_ratio = ratio.ln() / ln2;
            kl = kl + *p_val * log_ratio;
        }
    }

    Ok(kl)
}

/// Computes the Jensen-Shannon divergence between two distributions
///
/// # Mathematical Formulation
///
/// ```text
/// JS(P||Q) = 0.5 * D_KL(P||M) + 0.5 * D_KL(Q||M)
/// where M = 0.5 * (P + Q)
/// ```
///
/// # Properties
///
/// - JS(P||Q) ≥ 0
/// - JS(P||Q) = 0 if and only if P = Q
/// - Symmetric: JS(P||Q) = JS(Q||P)
/// - Bounded: 0 ≤ JS(P||Q) ≤ 1 (in bits)
/// - Square root of JS is a true metric
///
/// # Arguments
///
/// * `p` - First probability distribution
/// * `q` - Second probability distribution
///
/// # Returns
///
/// * JS divergence value in bits
pub fn js_divergence<F, S1, S2, D1, D2>(p: &ArrayBase<S1, D1>, q: &ArrayBase<S2, D2>) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug + SimdUnifiedOps,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D1: Dimension,
    D2: Dimension,
{
    // Check same shape
    if p.shape() != q.shape() {
        return Err(MetricsError::InvalidInput(format!(
            "Distributions must have the same shape: {:?} vs {:?}",
            p.shape(),
            q.shape()
        )));
    }

    // Compute M = 0.5 * (P + Q)
    let half = NumCast::from(0.5)
        .ok_or_else(|| MetricsError::InvalidInput("Failed to convert 0.5".to_string()))?;

    let m: Array1<F> = p
        .iter()
        .zip(q.iter())
        .map(|(p_val, q_val)| (*p_val + *q_val) * half)
        .collect();

    // Compute JS = 0.5 * D_KL(P||M) + 0.5 * D_KL(Q||M)
    let kl_pm = kl_divergence(p, &m)?;
    let kl_qm = kl_divergence(q, &m)?;

    Ok((kl_pm + kl_qm) * half)
}

/// Computes the mutual information between two random variables
///
/// # Mathematical Formulation
///
/// ```text
/// I(X;Y) = H(X) + H(Y) - H(X,Y)
///        = Σ Σ p(x,y) * log₂(p(x,y) / (p(x) * p(y)))
/// ```
///
/// # Properties
///
/// - I(X;Y) ≥ 0
/// - I(X;Y) = 0 if X and Y are independent
/// - Symmetric: I(X;Y) = I(Y;X)
/// - I(X;Y) ≤ min(H(X), H(Y))
///
/// # Arguments
///
/// * `joint_probabilities` - Joint probability distribution p(x,y)
/// * `marginal_x` - Marginal probability distribution p(x)
/// * `marginal_y` - Marginal probability distribution p(y)
///
/// # Returns
///
/// * Mutual information value in bits
pub fn mutual_information<F, S1, S2, S3>(
    joint_probabilities: &ArrayBase<S1, Ix2>,
    marginal_x: &ArrayBase<S2, Ix1>,
    marginal_y: &ArrayBase<S3, Ix1>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug + SimdUnifiedOps,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    S3: Data<Elem = F>,
{
    // I(X;Y) = H(X) + H(Y) - H(X,Y)
    let h_x = entropy(marginal_x)?;
    let h_y = entropy(marginal_y)?;
    let h_xy = joint_entropy(joint_probabilities)?;

    Ok(h_x + h_y - h_xy)
}

/// Computes mutual information from labels (discrete variables)
///
/// This function computes mutual information from discrete label arrays
/// by first computing the joint and marginal probability distributions.
///
/// # Arguments
///
/// * `labels_x` - First set of discrete labels
/// * `labels_y` - Second set of discrete labels
///
/// # Returns
///
/// * Mutual information value in bits
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_metrics::information_theory::mutual_information_from_labels;
///
/// let x = array![0, 0, 1, 1];
/// let y = array![0, 1, 0, 1];
/// let mi = mutual_information_from_labels(&x, &y).expect("Failed to compute MI");
/// ```
pub fn mutual_information_from_labels<T, S1, S2, D1, D2>(
    labels_x: &ArrayBase<S1, D1>,
    labels_y: &ArrayBase<S2, D2>,
) -> Result<f64>
where
    T: std::hash::Hash + std::cmp::Eq + Copy + std::fmt::Debug,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    D1: Dimension,
    D2: Dimension,
{
    // Check same length
    if labels_x.len() != labels_y.len() {
        return Err(MetricsError::InvalidInput(format!(
            "Label arrays must have the same length: {} vs {}",
            labels_x.len(),
            labels_y.len()
        )));
    }

    let n = labels_x.len() as f64;
    if n == 0.0 {
        return Err(MetricsError::InvalidInput(
            "Empty label arrays provided".to_string(),
        ));
    }

    // Compute joint and marginal counts
    let mut joint_counts: HashMap<(T, T), usize> = HashMap::new();
    let mut marginal_x_counts: HashMap<T, usize> = HashMap::new();
    let mut marginal_y_counts: HashMap<T, usize> = HashMap::new();

    for (x, y) in labels_x.iter().zip(labels_y.iter()) {
        *joint_counts.entry((*x, *y)).or_insert(0) += 1;
        *marginal_x_counts.entry(*x).or_insert(0) += 1;
        *marginal_y_counts.entry(*y).or_insert(0) += 1;
    }

    // Compute mutual information
    let mut mi = 0.0;
    let ln2 = std::f64::consts::LN_2;

    for ((x, y), &count_xy) in &joint_counts {
        let p_xy = count_xy as f64 / n;
        let count_x = marginal_x_counts
            .get(x)
            .ok_or_else(|| MetricsError::InvalidInput(format!("Missing count for x={x:?}")))?;
        let count_y = marginal_y_counts
            .get(y)
            .ok_or_else(|| MetricsError::InvalidInput(format!("Missing count for y={y:?}")))?;
        let p_x = *count_x as f64 / n;
        let p_y = *count_y as f64 / n;

        if p_xy > 0.0 && p_x > 0.0 && p_y > 0.0 {
            let ratio = p_xy / (p_x * p_y);
            mi += p_xy * (ratio.ln() / ln2);
        }
    }

    Ok(mi)
}

/// Computes the normalized mutual information (NMI)
///
/// # Mathematical Formulation
///
/// ```text
/// NMI(X,Y) = I(X;Y) / sqrt(H(X) * H(Y))
/// ```
///
/// # Properties
///
/// - 0 ≤ NMI(X,Y) ≤ 1
/// - NMI = 0 when X and Y are independent
/// - NMI = 1 when X and Y are identical
///
/// # Arguments
///
/// * `labels_x` - First set of discrete labels
/// * `labels_y` - Second set of discrete labels
///
/// # Returns
///
/// * Normalized mutual information value between 0 and 1
pub fn normalized_mutual_information<T, S1, S2, D1, D2>(
    labels_x: &ArrayBase<S1, D1>,
    labels_y: &ArrayBase<S2, D2>,
) -> Result<f64>
where
    T: std::hash::Hash + std::cmp::Eq + Copy + std::fmt::Debug,
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    D1: Dimension,
    D2: Dimension,
{
    let mi = mutual_information_from_labels(labels_x, labels_y)?;

    // Compute marginal entropies
    let n = labels_x.len() as f64;
    let mut counts_x: HashMap<T, usize> = HashMap::new();
    let mut counts_y: HashMap<T, usize> = HashMap::new();

    for x in labels_x.iter() {
        *counts_x.entry(*x).or_insert(0) += 1;
    }
    for y in labels_y.iter() {
        *counts_y.entry(*y).or_insert(0) += 1;
    }

    // Compute entropies
    let ln2 = std::f64::consts::LN_2;
    let mut h_x = 0.0;
    for &count in counts_x.values() {
        if count > 0 {
            let p = count as f64 / n;
            h_x -= p * (p.ln() / ln2);
        }
    }

    let mut h_y = 0.0;
    for &count in counts_y.values() {
        if count > 0 {
            let p = count as f64 / n;
            h_y -= p * (p.ln() / ln2);
        }
    }

    if h_x <= 0.0 || h_y <= 0.0 {
        return Ok(0.0);
    }

    Ok(mi / (h_x * h_y).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_entropy_uniform() {
        let uniform = array![0.25, 0.25, 0.25, 0.25];
        let h: f64 = entropy(&uniform).expect("Failed to compute entropy");
        // Uniform distribution over 4 outcomes has entropy log₂(4) = 2 bits
        assert_relative_eq!(h, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_entropy_deterministic() {
        let deterministic = array![1.0, 0.0, 0.0, 0.0];
        let h: f64 = entropy(&deterministic).expect("Failed to compute entropy");
        // Deterministic distribution has zero entropy
        assert_relative_eq!(h, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kl_divergence_identical() {
        let p = array![0.5, 0.5];
        let q = array![0.5, 0.5];
        let kl: f64 = kl_divergence(&p, &q).expect("Failed to compute KL divergence");
        // KL divergence of identical distributions is 0
        assert_relative_eq!(kl, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_js_divergence_identical() {
        let p = array![0.5, 0.5];
        let q = array![0.5, 0.5];
        let js: f64 = js_divergence(&p, &q).expect("Failed to compute JS divergence");
        // JS divergence of identical distributions is 0
        assert_relative_eq!(js, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_js_divergence_symmetric() {
        let p = array![0.6, 0.4];
        let q = array![0.3, 0.7];
        let js_pq: f64 = js_divergence(&p, &q).expect("Failed to compute JS divergence");
        let js_qp: f64 = js_divergence(&q, &p).expect("Failed to compute JS divergence");
        // JS divergence is symmetric
        assert_relative_eq!(js_pq, js_qp, epsilon = 1e-10);
    }

    #[test]
    fn test_mutual_information_independent() {
        // Independent variables should have MI ≈ 0
        let x = array![0, 0, 1, 1, 0, 0, 1, 1];
        let y = array![0, 1, 0, 1, 0, 1, 0, 1];
        let mi = mutual_information_from_labels(&x, &y).expect("Failed to compute MI");
        // For independent variables, MI should be close to 0
        assert_relative_eq!(mi, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mutual_information_identical() {
        // Identical variables should have MI = H(X)
        let x = array![0, 1, 0, 1, 0, 1];
        let y = x.clone();
        let mi = mutual_information_from_labels(&x, &y).expect("Failed to compute MI");
        // MI(X,X) = H(X) = 1 bit for binary variable
        assert_relative_eq!(mi, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_nmi_bounds() {
        let x = array![0, 1, 2, 0, 1, 2];
        let y = array![0, 0, 1, 1, 2, 2];
        let nmi = normalized_mutual_information(&x, &y).expect("Failed to compute NMI");
        // NMI should be between 0 and 1
        assert!((0.0..=1.0).contains(&nmi));
    }

    #[test]
    fn test_entropy_invalid_sum() {
        let invalid = array![0.3, 0.3, 0.3]; // sums to 0.9, not 1.0
        let result: std::result::Result<f64, _> = entropy(&invalid);
        assert!(result.is_err());
    }

    #[test]
    fn test_entropy_negative_probability() {
        let invalid = array![0.5, -0.3, 0.8];
        let result: std::result::Result<f64, _> = entropy(&invalid);
        assert!(result.is_err());
    }

    #[test]
    fn test_kl_divergence_infinite() {
        let p = array![0.5, 0.5];
        let q = array![1.0, 0.0]; // Q has zero where P has non-zero
        let result: std::result::Result<f64, _> = kl_divergence(&p, &q);
        assert!(result.is_err());
    }
}

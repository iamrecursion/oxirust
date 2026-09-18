//! Transform type selection.

/// Transform types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformType {
    /// Discrete Cosine Transform.
    Dct,
    /// Asymmetric DST.
    Adst,
    /// Identity transform.
    Identity,
    /// Hybrid DCT/ADST.
    Hybrid,
}

/// Transform selection optimizer.
pub struct TransformSelection {
    enable_adst: bool,
    enable_identity: bool,
}

impl Default for TransformSelection {
    fn default() -> Self {
        Self::new(true, false)
    }
}

impl TransformSelection {
    /// Creates a new transform selector.
    #[must_use]
    pub fn new(enable_adst: bool, enable_identity: bool) -> Self {
        Self {
            enable_adst,
            enable_identity,
        }
    }

    /// Selects the best transform for a block.
    #[allow(dead_code)]
    #[must_use]
    pub fn select(&self, residual: &[i16], is_intra: bool) -> TransformType {
        let candidates = self.candidate_transforms(is_intra);
        let mut best_transform = TransformType::Dct;
        let mut best_cost = f64::MAX;

        for &transform in &candidates {
            let cost = self.evaluate_transform(residual, transform);
            if cost < best_cost {
                best_cost = cost;
                best_transform = transform;
            }
        }

        best_transform
    }

    fn candidate_transforms(&self, is_intra: bool) -> Vec<TransformType> {
        let mut transforms = vec![TransformType::Dct];

        if self.enable_adst && is_intra {
            transforms.push(TransformType::Adst);
            transforms.push(TransformType::Hybrid);
        }

        if self.enable_identity {
            transforms.push(TransformType::Identity);
        }

        transforms
    }

    fn evaluate_transform(&self, residual: &[i16], transform: TransformType) -> f64 {
        match transform {
            TransformType::Dct => self.evaluate_dct(residual),
            TransformType::Adst => self.evaluate_adst(residual),
            TransformType::Identity => self.evaluate_identity(residual),
            TransformType::Hybrid => self.evaluate_hybrid(residual),
        }
    }

    fn evaluate_dct(&self, residual: &[i16]) -> f64 {
        // Simplified: count non-zero coefficients after transform
        residual.iter().filter(|&&x| x.abs() > 10).count() as f64
    }

    fn evaluate_adst(&self, residual: &[i16]) -> f64 {
        // ADST is better for directional content
        // Simplified evaluation
        self.evaluate_dct(residual) * 0.95
    }

    fn evaluate_identity(&self, residual: &[i16]) -> f64 {
        // Identity is only good for very low residuals
        let sum: i32 = residual.iter().map(|&x| i32::from(x.abs())).sum();
        if sum < 100 {
            f64::from(sum) * 0.5
        } else {
            f64::MAX // Avoid identity for high residuals
        }
    }

    fn evaluate_hybrid(&self, residual: &[i16]) -> f64 {
        // Hybrid can be beneficial for mixed content
        (self.evaluate_dct(residual) + self.evaluate_adst(residual)) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_selection_creation() {
        let selector = TransformSelection::default();
        assert!(selector.enable_adst);
        assert!(!selector.enable_identity);
    }

    #[test]
    fn test_candidate_transforms_intra() {
        let selector = TransformSelection::default();
        let transforms = selector.candidate_transforms(true);
        assert!(transforms.contains(&TransformType::Dct));
        assert!(transforms.contains(&TransformType::Adst));
    }

    #[test]
    fn test_candidate_transforms_inter() {
        let selector = TransformSelection::default();
        let transforms = selector.candidate_transforms(false);
        assert!(transforms.contains(&TransformType::Dct));
        assert!(!transforms.contains(&TransformType::Adst)); // ADST disabled for inter
    }

    #[test]
    fn test_transform_types() {
        assert_ne!(TransformType::Dct, TransformType::Adst);
        assert_eq!(TransformType::Dct, TransformType::Dct);
    }
}

//! Sample Adaptive Offset (SAO) optimization.

/// SAO types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaoType {
    /// No SAO.
    None,
    /// Band offset.
    BandOffset,
    /// Edge offset.
    EdgeOffset,
}

/// SAO parameters.
#[derive(Debug, Clone)]
pub struct SaoParams {
    /// SAO type.
    pub sao_type: SaoType,
    /// Band positions (for band offset).
    pub band_positions: Vec<u8>,
    /// Offsets.
    pub offsets: Vec<i8>,
    /// Edge class (for edge offset).
    pub edge_class: u8,
}

impl Default for SaoParams {
    fn default() -> Self {
        Self {
            sao_type: SaoType::None,
            band_positions: Vec::new(),
            offsets: Vec::new(),
            edge_class: 0,
        }
    }
}

/// SAO optimizer.
pub struct SaoOptimizer {
    enable_band_offset: bool,
    enable_edge_offset: bool,
    lambda: f64,
}

impl Default for SaoOptimizer {
    fn default() -> Self {
        Self::new(true, true, 1.0)
    }
}

impl SaoOptimizer {
    /// Creates a new SAO optimizer.
    #[must_use]
    pub fn new(enable_band_offset: bool, enable_edge_offset: bool, lambda: f64) -> Self {
        Self {
            enable_band_offset,
            enable_edge_offset,
            lambda,
        }
    }

    /// Optimizes SAO parameters for a block.
    #[allow(dead_code)]
    #[must_use]
    pub fn optimize(&self, original: &[u8], reconstructed: &[u8]) -> SaoParams {
        let mut best_params = SaoParams::default();
        let mut best_cost = self.calculate_cost(original, reconstructed, &best_params);

        if self.enable_band_offset {
            let band_params = self.optimize_band_offset(original, reconstructed);
            let cost = self.calculate_cost(original, reconstructed, &band_params);
            if cost < best_cost {
                best_cost = cost;
                best_params = band_params;
            }
        }

        if self.enable_edge_offset {
            let edge_params = self.optimize_edge_offset(original, reconstructed);
            let cost = self.calculate_cost(original, reconstructed, &edge_params);
            if cost < best_cost {
                best_params = edge_params;
            }
        }

        best_params
    }

    fn optimize_band_offset(&self, original: &[u8], reconstructed: &[u8]) -> SaoParams {
        // Simplified band offset optimization
        const NUM_BANDS: usize = 4;
        let mut offsets = vec![0i8; NUM_BANDS];
        let band_width = 256 / NUM_BANDS;

        for band in 0..NUM_BANDS {
            let mut sum_diff = 0i32;
            let mut count = 0;

            for (&orig, &recon) in original.iter().zip(reconstructed) {
                let band_idx = (usize::from(recon) / band_width).min(NUM_BANDS - 1);
                if band_idx == band {
                    sum_diff += i32::from(orig) - i32::from(recon);
                    count += 1;
                }
            }

            offsets[band] = if count > 0 {
                (sum_diff / count).clamp(-127, 127) as i8
            } else {
                0
            };
        }

        SaoParams {
            sao_type: SaoType::BandOffset,
            band_positions: (0..NUM_BANDS).map(|i| i as u8).collect(),
            offsets,
            edge_class: 0,
        }
    }

    fn optimize_edge_offset(&self, original: &[u8], reconstructed: &[u8]) -> SaoParams {
        // Simplified edge offset optimization
        let mut offsets = vec![0i8; 4]; // 4 edge classes

        // Calculate average difference for edge pixels
        let mut sum_diff = 0i32;
        let mut count = 0;

        for (&orig, &recon) in original.iter().zip(reconstructed) {
            sum_diff += i32::from(orig) - i32::from(recon);
            count += 1;
        }

        let avg_offset = if count > 0 {
            (sum_diff / count).clamp(-7, 7) as i8
        } else {
            0
        };

        offsets.fill(avg_offset);

        SaoParams {
            sao_type: SaoType::EdgeOffset,
            band_positions: Vec::new(),
            offsets,
            edge_class: 0,
        }
    }

    fn calculate_cost(&self, original: &[u8], reconstructed: &[u8], params: &SaoParams) -> f64 {
        // Calculate distortion
        let distortion: f64 = original
            .iter()
            .zip(reconstructed)
            .map(|(&o, &r)| {
                let diff = f64::from(o) - f64::from(r);
                diff * diff
            })
            .sum();

        // Calculate rate (simplified)
        let rate = match params.sao_type {
            SaoType::None => 0.0,
            SaoType::BandOffset => params.offsets.len() as f64 * 4.0,
            SaoType::EdgeOffset => params.offsets.len() as f64 * 3.0,
        };

        distortion + self.lambda * rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sao_optimizer_creation() {
        let optimizer = SaoOptimizer::default();
        assert!(optimizer.enable_band_offset);
        assert!(optimizer.enable_edge_offset);
    }

    #[test]
    fn test_sao_params_default() {
        let params = SaoParams::default();
        assert_eq!(params.sao_type, SaoType::None);
        assert!(params.offsets.is_empty());
    }

    #[test]
    fn test_band_offset_optimization() {
        let optimizer = SaoOptimizer::default();
        let original = vec![100u8; 64];
        let reconstructed = vec![95u8; 64];
        let params = optimizer.optimize_band_offset(&original, &reconstructed);
        assert_eq!(params.sao_type, SaoType::BandOffset);
        assert!(!params.offsets.is_empty());
    }

    #[test]
    fn test_edge_offset_optimization() {
        let optimizer = SaoOptimizer::default();
        let original = vec![100u8; 64];
        let reconstructed = vec![95u8; 64];
        let params = optimizer.optimize_edge_offset(&original, &reconstructed);
        assert_eq!(params.sao_type, SaoType::EdgeOffset);
        assert_eq!(params.offsets.len(), 4);
    }
}

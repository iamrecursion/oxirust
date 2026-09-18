//! Psychovisual adaptive quantization.

/// Psychovisual AQ parameters.
#[derive(Debug, Clone, Copy)]
pub struct PsychoAqParams {
    /// Edge preservation strength.
    pub edge_preservation: f64,
    /// Texture masking strength.
    pub texture_masking: f64,
    /// Contrast sensitivity.
    pub contrast_sensitivity: f64,
}

impl Default for PsychoAqParams {
    fn default() -> Self {
        Self {
            edge_preservation: 1.0,
            texture_masking: 1.0,
            contrast_sensitivity: 1.0,
        }
    }
}

/// Psychovisual AQ.
pub struct PsychoAq {
    params: PsychoAqParams,
}

impl Default for PsychoAq {
    fn default() -> Self {
        Self::new(PsychoAqParams::default())
    }
}

impl PsychoAq {
    /// Creates a new psychovisual AQ.
    #[must_use]
    pub fn new(params: PsychoAqParams) -> Self {
        Self { params }
    }

    /// Calculates psychovisual weight for a block.
    #[must_use]
    pub fn calculate_weight(&self, pixels: &[u8], width: usize) -> f64 {
        let edge_weight = self.calculate_edge_weight(pixels, width);
        let texture_weight = self.calculate_texture_weight(pixels);

        // Combine weights
        edge_weight * self.params.edge_preservation + texture_weight * self.params.texture_masking
    }

    fn calculate_edge_weight(&self, pixels: &[u8], width: usize) -> f64 {
        if pixels.len() < width * 2 {
            return 1.0;
        }

        let height = pixels.len() / width;
        let mut edge_strength = 0.0;
        let mut count = 0;

        for y in 0..height - 1 {
            for x in 0..width - 1 {
                let curr = pixels[y * width + x];
                let right = pixels[y * width + x + 1];
                let down = pixels[(y + 1) * width + x];

                let grad = u16::from(curr.abs_diff(right)) + u16::from(curr.abs_diff(down));
                edge_strength += f64::from(grad);
                count += 1;
            }
        }

        if count > 0 {
            let avg_edge = edge_strength / f64::from(count);
            // Normalize to 0-1 range (assuming max gradient ~100)
            (avg_edge / 100.0).min(1.0)
        } else {
            0.0
        }
    }

    fn calculate_texture_weight(&self, pixels: &[u8]) -> f64 {
        if pixels.is_empty() {
            return 1.0;
        }

        // Calculate variance as texture metric
        let mean = pixels.iter().map(|&p| f64::from(p)).sum::<f64>() / pixels.len() as f64;
        let variance = pixels
            .iter()
            .map(|&p| {
                let diff = f64::from(p) - mean;
                diff * diff
            })
            .sum::<f64>()
            / pixels.len() as f64;

        // High variance = high texture = allow more compression
        (variance / 500.0).min(1.5)
    }

    /// Converts psychovisual weight to QP offset.
    #[must_use]
    pub fn weight_to_qp_offset(&self, weight: f64, strength: f64) -> i8 {
        // Higher weight -> can compress more -> positive QP offset
        let offset = (weight - 1.0) * strength * 4.0;
        offset.clamp(-8.0, 8.0) as i8
    }

    /// Gets the parameters.
    #[must_use]
    pub fn params(&self) -> &PsychoAqParams {
        &self.params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psycho_aq_creation() {
        let aq = PsychoAq::default();
        assert_eq!(aq.params().edge_preservation, 1.0);
    }

    #[test]
    fn test_psycho_aq_params() {
        let params = PsychoAqParams {
            edge_preservation: 1.2,
            texture_masking: 0.8,
            contrast_sensitivity: 1.0,
        };
        let aq = PsychoAq::new(params);
        assert_eq!(aq.params().edge_preservation, 1.2);
    }

    #[test]
    fn test_edge_weight_flat() {
        let aq = PsychoAq::default();
        let flat = vec![128u8; 64];
        let weight = aq.calculate_edge_weight(&flat, 8);
        assert_eq!(weight, 0.0); // No edges
    }

    #[test]
    fn test_texture_weight_flat() {
        let aq = PsychoAq::default();
        let flat = vec![128u8; 64];
        let weight = aq.calculate_texture_weight(&flat);
        assert_eq!(weight, 0.0); // No texture
    }

    #[test]
    fn test_texture_weight_varied() {
        let aq = PsychoAq::default();
        let varied: Vec<u8> = (0..64).map(|i| i as u8).collect();
        let weight = aq.calculate_texture_weight(&varied);
        assert!(weight > 0.0);
    }

    #[test]
    fn test_weight_to_qp_offset() {
        let aq = PsychoAq::default();
        let offset_low = aq.weight_to_qp_offset(0.5, 1.0);
        let offset_high = aq.weight_to_qp_offset(1.5, 1.0);
        assert!(offset_low < 0); // Low weight -> preserve quality
        assert!(offset_high > 0); // High weight -> can compress more
    }
}

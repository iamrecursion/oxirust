//! Deblocking filter optimization.

/// Filter strength levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterStrength {
    /// No filtering.
    None,
    /// Weak filtering.
    Weak,
    /// Normal filtering.
    Normal,
    /// Strong filtering.
    Strong,
}

/// Deblocking filter optimizer.
pub struct DeblockOptimizer {
    beta_offset: i8,
    tc_offset: i8,
}

impl Default for DeblockOptimizer {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

impl DeblockOptimizer {
    /// Creates a new deblocking optimizer.
    #[must_use]
    pub const fn new(beta_offset: i8, tc_offset: i8) -> Self {
        Self {
            beta_offset,
            tc_offset,
        }
    }

    /// Determines optimal filter strength for an edge.
    #[allow(dead_code)]
    #[must_use]
    pub fn decide_strength(&self, edge_pixels: &[u8], qp: u8) -> FilterDecision {
        let gradient = self.calculate_gradient(edge_pixels);
        let beta = self.calculate_beta(qp);
        let tc = self.calculate_tc(qp);

        let strength = if gradient < beta / 4 {
            FilterStrength::None
        } else if gradient < beta / 2 {
            FilterStrength::Weak
        } else if gradient < beta {
            FilterStrength::Normal
        } else {
            FilterStrength::Strong
        };

        FilterDecision {
            strength,
            beta,
            tc,
            filter_enabled: strength != FilterStrength::None,
        }
    }

    fn calculate_gradient(&self, pixels: &[u8]) -> u32 {
        if pixels.len() < 2 {
            return 0;
        }

        pixels
            .windows(2)
            .map(|w| u32::from(w[0].abs_diff(w[1])))
            .sum()
    }

    fn calculate_beta(&self, qp: u8) -> u32 {
        // Simplified beta calculation
        let base = u32::from(qp) * 2;
        ((base as i32 + i32::from(self.beta_offset)).max(0)) as u32
    }

    fn calculate_tc(&self, qp: u8) -> u32 {
        // Simplified tc (threshold) calculation
        let base = u32::from(qp) / 2;
        ((base as i32 + i32::from(self.tc_offset)).max(0)) as u32
    }

    /// Applies deblocking filter to edge.
    #[allow(dead_code)]
    pub fn apply_filter(&self, pixels: &mut [u8], decision: &FilterDecision) {
        if !decision.filter_enabled || pixels.len() < 4 {
            return;
        }

        match decision.strength {
            FilterStrength::None => {}
            FilterStrength::Weak => self.apply_weak_filter(pixels, decision.tc),
            FilterStrength::Normal => self.apply_normal_filter(pixels, decision.tc),
            FilterStrength::Strong => self.apply_strong_filter(pixels, decision.tc),
        }
    }

    fn apply_weak_filter(&self, pixels: &mut [u8], tc: u32) {
        // Simplified weak filter
        let mid = pixels.len() / 2;
        if mid >= 1 && mid < pixels.len() {
            let avg = ((u16::from(pixels[mid - 1]) + u16::from(pixels[mid])) / 2) as u8;
            let delta = avg.saturating_sub(pixels[mid]).min(tc as u8);
            pixels[mid] = pixels[mid].saturating_add(delta);
        }
    }

    fn apply_normal_filter(&self, pixels: &mut [u8], tc: u32) {
        // Simplified normal filter
        self.apply_weak_filter(pixels, tc * 2);
    }

    fn apply_strong_filter(&self, pixels: &mut [u8], tc: u32) {
        // Simplified strong filter
        self.apply_weak_filter(pixels, tc * 3);
    }
}

/// Filter decision result.
#[derive(Debug, Clone, Copy)]
pub struct FilterDecision {
    /// Filter strength.
    pub strength: FilterStrength,
    /// Beta parameter.
    pub beta: u32,
    /// TC (threshold) parameter.
    pub tc: u32,
    /// Whether filtering is enabled.
    pub filter_enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deblock_optimizer_creation() {
        let optimizer = DeblockOptimizer::default();
        assert_eq!(optimizer.beta_offset, 0);
        assert_eq!(optimizer.tc_offset, 0);
    }

    #[test]
    fn test_gradient_calculation() {
        let optimizer = DeblockOptimizer::default();
        let flat = vec![128u8; 8];
        let gradient_flat = optimizer.calculate_gradient(&flat);
        assert_eq!(gradient_flat, 0);

        let edge = vec![100u8, 100, 100, 100, 200, 200, 200, 200];
        let gradient_edge = optimizer.calculate_gradient(&edge);
        assert!(gradient_edge > 0);
    }

    #[test]
    fn test_filter_strength_decision() {
        let optimizer = DeblockOptimizer::default();
        let flat = vec![128u8; 8];
        let decision = optimizer.decide_strength(&flat, 26);
        assert_eq!(decision.strength, FilterStrength::None);
    }

    #[test]
    fn test_beta_tc_calculation() {
        let optimizer = DeblockOptimizer::new(2, -1);
        let beta = optimizer.calculate_beta(26);
        let tc = optimizer.calculate_tc(26);
        assert!(beta > 0);
        assert!(tc > 0);
    }
}

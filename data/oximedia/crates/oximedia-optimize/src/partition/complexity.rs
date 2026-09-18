//! Block complexity analysis.

/// Block complexity metrics.
#[derive(Debug, Clone, Copy)]
pub struct BlockComplexity {
    /// Spatial variance.
    pub variance: f64,
    /// Edge density.
    pub edge_density: f64,
    /// Temporal complexity (for inter blocks).
    pub temporal_complexity: f64,
    /// Overall complexity score.
    pub complexity_score: f64,
}

impl Default for BlockComplexity {
    fn default() -> Self {
        Self {
            variance: 0.0,
            edge_density: 0.0,
            temporal_complexity: 0.0,
            complexity_score: 0.0,
        }
    }
}

/// Complexity analyzer.
pub struct ComplexityAnalyzer {
    edge_threshold: u8,
}

impl Default for ComplexityAnalyzer {
    fn default() -> Self {
        Self::new(10)
    }
}

impl ComplexityAnalyzer {
    /// Creates a new complexity analyzer.
    #[must_use]
    pub const fn new(edge_threshold: u8) -> Self {
        Self { edge_threshold }
    }

    /// Analyzes block complexity.
    #[allow(dead_code)]
    #[must_use]
    pub fn analyze(&self, pixels: &[u8], width: usize) -> BlockComplexity {
        let variance = self.calculate_variance(pixels);
        let edge_density = self.calculate_edge_density(pixels, width);
        let complexity_score = self.calculate_complexity_score(variance, edge_density);

        BlockComplexity {
            variance,
            edge_density,
            temporal_complexity: 0.0, // Would be set from motion analysis
            complexity_score,
        }
    }

    fn calculate_variance(&self, pixels: &[u8]) -> f64 {
        if pixels.is_empty() {
            return 0.0;
        }

        let mean = pixels.iter().map(|&p| f64::from(p)).sum::<f64>() / pixels.len() as f64;
        pixels
            .iter()
            .map(|&p| {
                let diff = f64::from(p) - mean;
                diff * diff
            })
            .sum::<f64>()
            / pixels.len() as f64
    }

    fn calculate_edge_density(&self, pixels: &[u8], width: usize) -> f64 {
        if pixels.len() < width * 2 {
            return 0.0;
        }

        let height = pixels.len() / width;
        let mut edge_count = 0;
        let mut total_count = 0;

        for y in 0..height - 1 {
            for x in 0..width - 1 {
                let curr = pixels[y * width + x];
                let right = pixels[y * width + x + 1];
                let down = pixels[(y + 1) * width + x];

                if curr.abs_diff(right) > self.edge_threshold
                    || curr.abs_diff(down) > self.edge_threshold
                {
                    edge_count += 1;
                }
                total_count += 1;
            }
        }

        if total_count > 0 {
            f64::from(edge_count) / f64::from(total_count)
        } else {
            0.0
        }
    }

    fn calculate_complexity_score(&self, variance: f64, edge_density: f64) -> f64 {
        // Weighted combination of metrics
        variance * 0.7 + edge_density * 1000.0 * 0.3
    }

    /// Determines if block should be split based on complexity.
    #[must_use]
    pub fn should_split(&self, complexity: &BlockComplexity, block_size: usize) -> bool {
        // Split if complexity is high relative to block size
        let size_factor = 128.0 / block_size as f64;
        complexity.complexity_score > 200.0 * size_factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complexity_analyzer_creation() {
        let analyzer = ComplexityAnalyzer::default();
        assert_eq!(analyzer.edge_threshold, 10);
    }

    #[test]
    fn test_variance_calculation() {
        let analyzer = ComplexityAnalyzer::default();
        let flat = vec![128u8; 64];
        assert_eq!(analyzer.calculate_variance(&flat), 0.0);

        let varied: Vec<u8> = (0..64).map(|i| i as u8).collect();
        assert!(analyzer.calculate_variance(&varied) > 0.0);
    }

    #[test]
    fn test_edge_density_flat() {
        let analyzer = ComplexityAnalyzer::default();
        let flat = vec![128u8; 64];
        let density = analyzer.calculate_edge_density(&flat, 8);
        assert_eq!(density, 0.0);
    }

    #[test]
    fn test_edge_density_checkerboard() {
        let analyzer = ComplexityAnalyzer::default();
        let mut checkerboard = vec![0u8; 64];
        for y in 0..8 {
            for x in 0..8 {
                checkerboard[y * 8 + x] = if (x + y) % 2 == 0 { 0 } else { 255 };
            }
        }
        let density = analyzer.calculate_edge_density(&checkerboard, 8);
        assert!(density > 0.5); // High edge density
    }

    #[test]
    fn test_complexity_analysis() {
        let analyzer = ComplexityAnalyzer::default();
        let pixels = vec![128u8; 64];
        let complexity = analyzer.analyze(&pixels, 8);
        assert_eq!(complexity.variance, 0.0);
        assert_eq!(complexity.edge_density, 0.0);
    }

    #[test]
    fn test_should_split_decision() {
        let analyzer = ComplexityAnalyzer::default();
        let low_complexity = BlockComplexity {
            variance: 10.0,
            edge_density: 0.1,
            temporal_complexity: 0.0,
            complexity_score: 50.0,
        };
        assert!(!analyzer.should_split(&low_complexity, 64));

        let high_complexity = BlockComplexity {
            variance: 500.0,
            edge_density: 0.8,
            temporal_complexity: 0.0,
            complexity_score: 1000.0,
        };
        assert!(analyzer.should_split(&high_complexity, 64));
    }
}

//! Sub-pixel motion estimation.

use super::MotionVector;

/// Sub-pixel precision levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubpelPrecision {
    /// Integer pixel precision.
    Integer,
    /// Half-pixel precision.
    HalfPel,
    /// Quarter-pixel precision.
    QuarterPel,
}

/// Sub-pixel motion optimizer.
pub struct SubpelOptimizer {
    precision: SubpelPrecision,
    #[allow(dead_code)]
    max_iterations: usize,
}

impl Default for SubpelOptimizer {
    fn default() -> Self {
        Self::new(SubpelPrecision::QuarterPel)
    }
}

impl SubpelOptimizer {
    /// Creates a new sub-pixel optimizer.
    #[must_use]
    pub fn new(precision: SubpelPrecision) -> Self {
        Self {
            precision,
            max_iterations: 8,
        }
    }

    /// Refines integer motion vector to sub-pixel precision.
    #[allow(dead_code)]
    #[must_use]
    pub fn refine(&self, integer_mv: MotionVector, src: &[u8], reference: &[u8]) -> MotionVector {
        match self.precision {
            SubpelPrecision::Integer => integer_mv,
            SubpelPrecision::HalfPel => self.refine_half_pel(integer_mv, src, reference),
            SubpelPrecision::QuarterPel => {
                let half_pel = self.refine_half_pel(integer_mv, src, reference);
                self.refine_quarter_pel(half_pel, src, reference)
            }
        }
    }

    fn refine_half_pel(
        &self,
        integer_mv: MotionVector,
        src: &[u8],
        _reference: &[u8],
    ) -> MotionVector {
        let mut best_mv = integer_mv;
        let mut best_cost = self.calculate_subpel_cost(src, best_mv);

        // Half-pel positions around integer position
        let half_pel_offsets = [
            (-2, 0),
            (2, 0),
            (0, -2),
            (0, 2),
            (-2, -2),
            (2, -2),
            (-2, 2),
            (2, 2),
        ];

        for &(dx, dy) in &half_pel_offsets {
            let mv = MotionVector::new(integer_mv.x + dx, integer_mv.y + dy);
            let cost = self.calculate_subpel_cost(src, mv);

            if cost < best_cost {
                best_cost = cost;
                best_mv = mv;
            }
        }

        best_mv
    }

    fn refine_quarter_pel(
        &self,
        half_pel_mv: MotionVector,
        src: &[u8],
        _reference: &[u8],
    ) -> MotionVector {
        let mut best_mv = half_pel_mv;
        let mut best_cost = self.calculate_subpel_cost(src, best_mv);

        // Quarter-pel positions around half-pel position
        let quarter_pel_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)];

        for &(dx, dy) in &quarter_pel_offsets {
            let mv = MotionVector::new(half_pel_mv.x + dx, half_pel_mv.y + dy);
            let cost = self.calculate_subpel_cost(src, mv);

            if cost < best_cost {
                best_cost = cost;
                best_mv = mv;
            }
        }

        best_mv
    }

    fn calculate_subpel_cost(&self, src: &[u8], _mv: MotionVector) -> f64 {
        // Simplified cost (would interpolate reference block in production)
        src.iter().map(|&x| f64::from(x)).sum::<f64>() / src.len() as f64
    }

    /// Performs bilinear interpolation for half-pel positions.
    #[allow(dead_code)]
    #[must_use]
    pub fn interpolate_half_pel(&self, reference: &[u8], width: usize, x: i16, y: i16) -> u8 {
        let x_int = (x / 2) as usize;
        let y_int = (y / 2) as usize;
        let x_frac = x % 2;
        let y_frac = y % 2;

        if x_frac == 0 && y_frac == 0 {
            // Integer position
            reference[y_int * width + x_int]
        } else if y_frac == 0 {
            // Horizontal half-pel
            let a = reference[y_int * width + x_int];
            let b = reference[y_int * width + x_int + 1];
            (u16::from(a) + u16::from(b)).div_ceil(2) as u8
        } else if x_frac == 0 {
            // Vertical half-pel
            let a = reference[y_int * width + x_int];
            let b = reference[(y_int + 1) * width + x_int];
            (u16::from(a) + u16::from(b)).div_ceil(2) as u8
        } else {
            // Diagonal half-pel
            let a = reference[y_int * width + x_int];
            let b = reference[y_int * width + x_int + 1];
            let c = reference[(y_int + 1) * width + x_int];
            let d = reference[(y_int + 1) * width + x_int + 1];
            ((u16::from(a) + u16::from(b) + u16::from(c) + u16::from(d) + 2) / 4) as u8
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subpel_optimizer_creation() {
        let optimizer = SubpelOptimizer::default();
        assert_eq!(optimizer.precision, SubpelPrecision::QuarterPel);
    }

    #[test]
    fn test_subpel_precision_levels() {
        let int_opt = SubpelOptimizer::new(SubpelPrecision::Integer);
        let half_opt = SubpelOptimizer::new(SubpelPrecision::HalfPel);
        let qpel_opt = SubpelOptimizer::new(SubpelPrecision::QuarterPel);

        assert_eq!(int_opt.precision, SubpelPrecision::Integer);
        assert_eq!(half_opt.precision, SubpelPrecision::HalfPel);
        assert_eq!(qpel_opt.precision, SubpelPrecision::QuarterPel);
    }

    #[test]
    fn test_interpolate_integer() {
        let optimizer = SubpelOptimizer::default();
        let reference = vec![100u8; 64];
        let pixel = optimizer.interpolate_half_pel(&reference, 8, 4, 4);
        assert_eq!(pixel, 100);
    }

    #[test]
    fn test_interpolate_half_pel_horizontal() {
        let optimizer = SubpelOptimizer::default();
        let mut reference = vec![100u8; 64];
        // x=17, y=0: x_int=8, y_int=0, x_frac=1, y_frac=0 → horizontal half-pel
        // reference[y_int*width + x_int] = reference[8], reference[9]
        reference[8] = 100;
        reference[9] = 200;
        let pixel = optimizer.interpolate_half_pel(&reference, 8, 17, 0); // x=17 is half-pel, y=0
        assert_eq!(pixel, 150); // (100 + 200 + 1) / 2 = 150
    }
}

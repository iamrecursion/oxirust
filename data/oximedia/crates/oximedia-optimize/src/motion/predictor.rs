//! Motion vector prediction.

use super::MotionVector;

/// Motion vector predictor.
#[derive(Debug, Clone, Copy, Default)]
pub struct MvPredictor {
    /// Left neighbor MV.
    pub left: Option<MotionVector>,
    /// Top neighbor MV.
    pub top: Option<MotionVector>,
    /// Top-right neighbor MV.
    pub top_right: Option<MotionVector>,
    /// Top-left neighbor MV.
    pub top_left: Option<MotionVector>,
}

impl MvPredictor {
    /// Creates a new MV predictor.
    #[must_use]
    pub const fn new(
        left: Option<MotionVector>,
        top: Option<MotionVector>,
        top_right: Option<MotionVector>,
        top_left: Option<MotionVector>,
    ) -> Self {
        Self {
            left,
            top,
            top_right,
            top_left,
        }
    }

    /// Calculates median predictor.
    #[must_use]
    pub fn median_predictor(&self) -> MotionVector {
        let mvs = [self.left, self.top, self.top_right];

        let x_values: Vec<i16> = mvs.iter().filter_map(|&mv| mv.map(|m| m.x)).collect();
        let y_values: Vec<i16> = mvs.iter().filter_map(|&mv| mv.map(|m| m.y)).collect();

        let median_x = Self::median(&x_values).unwrap_or(0);
        let median_y = Self::median(&y_values).unwrap_or(0);

        MotionVector::new(median_x, median_y)
    }

    /// Calculates spatial predictor.
    #[must_use]
    pub fn spatial_predictor(&self) -> MotionVector {
        // Use left if available, otherwise top
        if let Some(mv) = self.left {
            mv
        } else if let Some(mv) = self.top {
            mv
        } else if let Some(mv) = self.top_left {
            mv
        } else {
            MotionVector::zero()
        }
    }

    /// Calculates averaged predictor.
    #[must_use]
    pub fn averaged_predictor(&self) -> MotionVector {
        let mvs = [self.left, self.top, self.top_right, self.top_left];
        let valid_mvs: Vec<MotionVector> = mvs.iter().filter_map(|&mv| mv).collect();

        if valid_mvs.is_empty() {
            return MotionVector::zero();
        }

        let sum_x: i32 = valid_mvs.iter().map(|mv| i32::from(mv.x)).sum();
        let sum_y: i32 = valid_mvs.iter().map(|mv| i32::from(mv.y)).sum();
        let count = valid_mvs.len() as i32;

        MotionVector::new((sum_x / count) as i16, (sum_y / count) as i16)
    }

    fn median(values: &[i16]) -> Option<i16> {
        if values.is_empty() {
            return None;
        }

        let mut sorted = values.to_vec();
        sorted.sort_unstable();

        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            Some((i32::from(sorted[mid - 1]) + i32::from(sorted[mid])) as i16 / 2)
        } else {
            Some(sorted[mid])
        }
    }
}

/// MV predictor list for multiple references.
pub struct MvPredictorList {
    predictors: Vec<MvPredictor>,
}

impl MvPredictorList {
    /// Creates a new predictor list.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            predictors: Vec::with_capacity(capacity),
        }
    }

    /// Adds a predictor for a reference.
    pub fn add_predictor(&mut self, predictor: MvPredictor) {
        self.predictors.push(predictor);
    }

    /// Gets predictor for a reference index.
    #[must_use]
    pub fn get_predictor(&self, ref_idx: usize) -> Option<&MvPredictor> {
        self.predictors.get(ref_idx)
    }

    /// Gets the best predictor based on availability.
    #[must_use]
    pub fn best_predictor(&self) -> MotionVector {
        if let Some(predictor) = self.predictors.first() {
            predictor.median_predictor()
        } else {
            MotionVector::zero()
        }
    }
}

/// Advanced motion vector predictor with temporal prediction.
pub struct TemporalMvPredictor {
    spatial: MvPredictor,
    temporal: Option<MotionVector>,
    temporal_weight: f64,
}

impl TemporalMvPredictor {
    /// Creates a new temporal MV predictor.
    #[must_use]
    pub fn new(spatial: MvPredictor, temporal: Option<MotionVector>) -> Self {
        Self {
            spatial,
            temporal,
            temporal_weight: 0.5,
        }
    }

    /// Sets the temporal weight.
    pub fn set_temporal_weight(&mut self, weight: f64) {
        self.temporal_weight = weight.clamp(0.0, 1.0);
    }

    /// Calculates combined spatial-temporal predictor.
    #[must_use]
    pub fn predict(&self) -> MotionVector {
        let spatial_mv = self.spatial.median_predictor();

        if let Some(temporal_mv) = self.temporal {
            // Weighted combination of spatial and temporal
            let w_temporal = self.temporal_weight;
            let w_spatial = 1.0 - w_temporal;

            let x = (f64::from(spatial_mv.x) * w_spatial + f64::from(temporal_mv.x) * w_temporal)
                as i16;
            let y = (f64::from(spatial_mv.y) * w_spatial + f64::from(temporal_mv.y) * w_temporal)
                as i16;

            MotionVector::new(x, y)
        } else {
            spatial_mv
        }
    }

    /// Scales temporal MV for different temporal distances.
    #[must_use]
    pub fn scale_temporal_mv(&self, mv: MotionVector, distance_ratio: f64) -> MotionVector {
        let x = (f64::from(mv.x) * distance_ratio) as i16;
        let y = (f64::from(mv.y) * distance_ratio) as i16;
        MotionVector::new(x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mv_predictor_default() {
        let predictor = MvPredictor::default();
        assert!(predictor.left.is_none());
        assert!(predictor.top.is_none());
    }

    #[test]
    fn test_median_predictor_all_same() {
        let mv = MotionVector::new(16, 8);
        let predictor = MvPredictor::new(Some(mv), Some(mv), Some(mv), None);
        let median = predictor.median_predictor();
        assert_eq!(median.x, 16);
        assert_eq!(median.y, 8);
    }

    #[test]
    fn test_median_predictor_different() {
        let predictor = MvPredictor::new(
            Some(MotionVector::new(10, 10)),
            Some(MotionVector::new(20, 20)),
            Some(MotionVector::new(30, 30)),
            None,
        );
        let median = predictor.median_predictor();
        assert_eq!(median.x, 20);
        assert_eq!(median.y, 20);
    }

    #[test]
    fn test_spatial_predictor_priority() {
        let predictor = MvPredictor::new(
            Some(MotionVector::new(10, 10)),
            Some(MotionVector::new(20, 20)),
            None,
            None,
        );
        let spatial = predictor.spatial_predictor();
        assert_eq!(spatial.x, 10); // Left has priority
        assert_eq!(spatial.y, 10);
    }

    #[test]
    fn test_spatial_predictor_no_left() {
        let predictor = MvPredictor::new(None, Some(MotionVector::new(20, 20)), None, None);
        let spatial = predictor.spatial_predictor();
        assert_eq!(spatial.x, 20); // Falls back to top
        assert_eq!(spatial.y, 20);
    }

    #[test]
    fn test_averaged_predictor() {
        let predictor = MvPredictor::new(
            Some(MotionVector::new(10, 10)),
            Some(MotionVector::new(20, 20)),
            Some(MotionVector::new(30, 30)),
            None,
        );
        let avg = predictor.averaged_predictor();
        assert_eq!(avg.x, 20); // (10+20+30)/3 = 20
        assert_eq!(avg.y, 20);
    }

    #[test]
    fn test_mv_predictor_list() {
        let mut list = MvPredictorList::new(2);
        list.add_predictor(MvPredictor::default());
        list.add_predictor(MvPredictor::default());
        assert!(list.get_predictor(0).is_some());
        assert!(list.get_predictor(1).is_some());
        assert!(list.get_predictor(2).is_none());
    }

    #[test]
    fn test_temporal_predictor() {
        let spatial = MvPredictor::new(Some(MotionVector::new(10, 10)), None, None, None);
        let temporal = Some(MotionVector::new(20, 20));
        let predictor = TemporalMvPredictor::new(spatial, temporal);

        let predicted = predictor.predict();
        // Should be weighted combination
        assert!(predicted.x > 10 && predicted.x < 20);
        assert!(predicted.y > 10 && predicted.y < 20);
    }

    #[test]
    fn test_temporal_weight() {
        let spatial = MvPredictor::new(Some(MotionVector::new(10, 10)), None, None, None);
        let temporal = Some(MotionVector::new(20, 20));
        let mut predictor = TemporalMvPredictor::new(spatial, temporal);

        predictor.set_temporal_weight(0.0);
        let predicted = predictor.predict();
        assert_eq!(predicted.x, 10); // All spatial

        predictor.set_temporal_weight(1.0);
        let predicted = predictor.predict();
        assert_eq!(predicted.x, 20); // All temporal
    }

    #[test]
    fn test_scale_temporal_mv() {
        let predictor = TemporalMvPredictor::new(MvPredictor::default(), None);
        let mv = MotionVector::new(20, 20);
        let scaled = predictor.scale_temporal_mv(mv, 0.5);
        assert_eq!(scaled.x, 10);
        assert_eq!(scaled.y, 10);
    }
}

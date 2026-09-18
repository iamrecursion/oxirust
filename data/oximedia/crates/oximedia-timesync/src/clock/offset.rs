//! Clock offset measurement and filtering.

use std::collections::VecDeque;

/// Offset measurement filter.
pub struct OffsetFilter {
    /// Moving window of measurements
    measurements: VecDeque<i64>,
    /// Window size
    window_size: usize,
}

impl OffsetFilter {
    /// Create a new offset filter.
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            measurements: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    /// Add a measurement.
    pub fn add_measurement(&mut self, offset_ns: i64) {
        if self.measurements.len() >= self.window_size {
            self.measurements.pop_front();
        }
        self.measurements.push_back(offset_ns);
    }

    /// Get median offset.
    #[must_use]
    pub fn median(&self) -> Option<i64> {
        if self.measurements.is_empty() {
            return None;
        }

        let mut sorted: Vec<i64> = self.measurements.iter().copied().collect();
        sorted.sort_unstable();

        Some(sorted[sorted.len() / 2])
    }

    /// Get mean offset.
    #[must_use]
    pub fn mean(&self) -> Option<i64> {
        if self.measurements.is_empty() {
            return None;
        }

        let sum: i128 = self.measurements.iter().map(|&x| i128::from(x)).sum();
        Some((sum / self.measurements.len() as i128) as i64)
    }

    /// Get standard deviation.
    #[must_use]
    pub fn std_dev(&self) -> Option<f64> {
        if self.measurements.len() < 2 {
            return None;
        }

        let mean = self.mean()? as f64;
        let variance: f64 = self
            .measurements
            .iter()
            .map(|&x| {
                let diff = x as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / self.measurements.len() as f64;

        Some(variance.sqrt())
    }

    /// Get filtered offset using median filter.
    #[must_use]
    pub fn filtered_offset(&self) -> Option<i64> {
        self.median()
    }

    /// Clear all measurements.
    pub fn clear(&mut self) {
        self.measurements.clear();
    }

    /// Check if filter is full.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.measurements.len() >= self.window_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_offset_filter() {
        let mut filter = OffsetFilter::new(5);

        filter.add_measurement(100);
        filter.add_measurement(200);
        filter.add_measurement(150);

        assert_eq!(filter.median(), Some(150));
        assert_eq!(filter.mean(), Some(150));
    }

    #[test]
    fn test_offset_filter_median() {
        let mut filter = OffsetFilter::new(5);

        filter.add_measurement(100);
        filter.add_measurement(200);
        filter.add_measurement(300);
        filter.add_measurement(400);
        filter.add_measurement(500);

        assert_eq!(filter.median(), Some(300));
    }

    #[test]
    fn test_offset_filter_std_dev() {
        let mut filter = OffsetFilter::new(5);

        filter.add_measurement(100);
        filter.add_measurement(100);
        filter.add_measurement(100);

        let std_dev = filter.std_dev().expect("should succeed in test");
        assert!(std_dev < 0.001); // Should be ~0 for identical values
    }

    #[test]
    fn test_offset_filter_window() {
        let mut filter = OffsetFilter::new(3);

        filter.add_measurement(1);
        filter.add_measurement(2);
        filter.add_measurement(3);
        assert!(filter.is_full());

        filter.add_measurement(4); // Should push out 1
        assert_eq!(filter.median(), Some(3));
    }
}

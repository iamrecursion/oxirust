//! Reverse playback effect.

/// Reverse playback effect.
pub struct Reverse {
    duration: f64,
}

impl Reverse {
    /// Create a new reverse effect.
    #[must_use]
    pub const fn new(duration: f64) -> Self {
        Self { duration }
    }

    /// Map forward time to reverse time.
    #[must_use]
    pub fn map_time(&self, time: f64) -> f64 {
        self.duration - time
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reverse() {
        let reverse = Reverse::new(10.0);
        assert_eq!(reverse.map_time(0.0), 10.0);
        assert_eq!(reverse.map_time(10.0), 0.0);
        assert_eq!(reverse.map_time(5.0), 5.0);
    }
}

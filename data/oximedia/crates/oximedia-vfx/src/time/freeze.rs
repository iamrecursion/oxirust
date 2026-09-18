//! Freeze frame effect.

/// Freeze frame effect.
pub struct FreezeFrame {
    freeze_time: f64,
}

impl FreezeFrame {
    /// Create a new freeze frame effect.
    #[must_use]
    pub const fn new(freeze_time: f64) -> Self {
        Self { freeze_time }
    }

    /// Get the frozen time.
    #[must_use]
    pub const fn get_freeze_time(&self) -> f64 {
        self.freeze_time
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_freeze_frame() {
        let freeze = FreezeFrame::new(5.0);
        assert_eq!(freeze.get_freeze_time(), 5.0);
    }
}

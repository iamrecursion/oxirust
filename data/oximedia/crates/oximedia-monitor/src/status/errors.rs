//! Error logging and tracking.

/// Error logger.
pub struct ErrorLogger {
    errors: Vec<String>,
}

impl ErrorLogger {
    /// Create a new error logger.
    #[must_use]
    pub fn new() -> Self {
        Self { errors: Vec::new() }
    }

    /// Log an error.
    pub fn log(&mut self, error: String) {
        self.errors.push(error);
    }

    /// Get all errors.
    #[must_use]
    pub fn errors(&self) -> &[String] {
        &self.errors
    }

    /// Clear errors.
    pub fn clear(&mut self) {
        self.errors.clear();
    }
}

impl Default for ErrorLogger {
    fn default() -> Self {
        Self::new()
    }
}

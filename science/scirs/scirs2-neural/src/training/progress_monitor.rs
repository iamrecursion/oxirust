//! Progress monitoring utilities

/// Configuration for progress monitoring
#[derive(Debug, Clone)]
pub struct ProgressMonitorConfig {
    /// Whether to show progress bar
    pub show_progress: bool,
    /// Update frequency
    pub update_frequency: usize,
}

impl Default for ProgressMonitorConfig {
    fn default() -> Self {
        Self {
            show_progress: true,
            update_frequency: 1,
        }
    }
}

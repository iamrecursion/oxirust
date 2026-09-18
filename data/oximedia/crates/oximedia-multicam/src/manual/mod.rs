//! Manual control for multi-camera production.

pub mod control;
pub mod preview;

pub use control::ManualController;
pub use preview::PreviewManager;

/// Control mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlMode {
    /// Direct selection
    Direct,
    /// Preview + program (like video switchers)
    PreviewProgram,
    /// Multi-view with selection
    MultiView,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_mode() {
        let mode = ControlMode::Direct;
        assert_eq!(mode, ControlMode::Direct);
    }
}

//! Controller input capture.

use crate::GamingResult;

/// Controller capture.
pub struct ControllerCapture {
    controller_type: ControllerType,
}

/// Controller type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControllerType {
    /// Xbox controller
    Xbox,
    /// `PlayStation` controller
    PlayStation,
    /// Nintendo controller
    Nintendo,
    /// Generic controller
    Generic,
}

/// Controller state.
#[derive(Debug, Clone, Default)]
pub struct ControllerState {
    /// Button states
    pub buttons: Vec<bool>,
    /// Analog stick positions (-1.0 to 1.0)
    pub sticks: Vec<(f32, f32)>,
    /// Trigger values (0.0 to 1.0)
    pub triggers: Vec<f32>,
}

impl ControllerCapture {
    /// Create a new controller capture.
    #[must_use]
    pub fn new(controller_type: ControllerType) -> Self {
        Self { controller_type }
    }

    /// Get controller state.
    pub fn get_state(&self) -> GamingResult<ControllerState> {
        Ok(ControllerState::default())
    }

    /// Get controller type.
    #[must_use]
    pub fn controller_type(&self) -> ControllerType {
        self.controller_type
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_controller_creation() {
        let controller = ControllerCapture::new(ControllerType::Xbox);
        assert_eq!(controller.controller_type(), ControllerType::Xbox);
    }
}

//! Input capture and visualization.
//!
//! Captures and visualizes keyboard, mouse, and controller inputs.

pub mod capture;
pub mod controller;
pub mod overlay;

pub use capture::{InputCapture, InputEvent, KeyboardState, MouseState};
pub use controller::{ControllerCapture, ControllerState, ControllerType};
pub use overlay::{InputOverlay, OverlayStyle};

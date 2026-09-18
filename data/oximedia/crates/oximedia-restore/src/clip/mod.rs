//! Clipping detection and restoration.

pub mod detector;
pub mod restore;

pub use detector::{detect_clipping_derivative, ClipDetector, ClipDetectorConfig, ClippingRegion};
pub use restore::{declip_ar_prediction, declip_cubic_spline, BasicDeclipper, DeclipConfig};

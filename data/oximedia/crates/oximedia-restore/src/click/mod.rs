//! Click and pop detection and removal.

pub mod detector;
pub mod remover;

pub use detector::{
    detect_clicks_simple, Click, ClickDetector, ClickDetectorConfig, ClickSeverity,
};
pub use remover::{remove_click_ar, remove_click_median, ClickRemover};

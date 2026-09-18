//! Webcam capture and integration.

pub mod capture;
pub mod chroma;
pub mod pip;

pub use capture::{WebcamCapture, WebcamConfig};
pub use chroma::{ChromaKey, ChromaKeyConfig};
pub use pip::{PictureInPicture, PipPosition};

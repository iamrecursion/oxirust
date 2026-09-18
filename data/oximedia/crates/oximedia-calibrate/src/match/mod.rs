//! Color matching across cameras and scenes.
//!
//! This module provides tools for matching colors between multiple cameras,
//! scenes, and reference targets for consistent color reproduction.

pub mod camera;
pub mod reference;
pub mod scene;

pub use camera::{CameraMatch, CameraMatchConfig};
pub use reference::{ReferenceMatch, ReferenceTarget};
pub use scene::{SceneMatch, SceneMatchConfig};

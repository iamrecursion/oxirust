//! Data augmentation techniques for training.
//!
//! This module provides various data augmentation strategies to improve model generalization:
//! - Noise augmentation (Gaussian)
//! - Scale augmentation (random scaling)
//! - Rotation augmentation (placeholder for future)
//! - Mixup augmentation (interpolation between samples)
//! - CutMix augmentation (cutting and mixing patches)
//! - Random Erasing (randomly erase rectangular regions)
//! - CutOut (fixed-size random erasing)
//!
//! A second functional API (Part 2) is also provided for composable,
//! function-based augmentation with a pipeline abstraction.

pub mod error;
pub mod functional;
pub mod pipeline;
pub mod rng;
pub mod trait_api;

// Re-export all public types so the parent crate's existing `pub use augmentation::{...}`
// continue to work without modification.
pub use error::AugmentationError;
pub use functional::{
    center_crop_2d, clip, cutmix, denormalize, dropout, dropout_mask, gaussian_noise, mixup,
    normalize, random_crop_2d, random_hflip, random_vflip,
};
pub use pipeline::{AugStats, AugmentationPipeline, AugmentationStep};
pub use rng::AugRng;
pub use trait_api::{
    CompositeAugmenter, CutMixAugmenter, CutOutAugmenter, DataAugmenter, MixupAugmenter,
    NoAugmentation, NoiseAugmenter, RandomErasingAugmenter, RotationAugmenter, ScaleAugmenter,
};

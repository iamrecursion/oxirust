//! Distortion effects.

pub mod barrel;
pub mod lens;
pub mod lens_distort;
pub mod ripple;
pub mod wave;

pub use barrel::{BarrelDistortion, DistortionType};
pub use lens::{LensDistortion, LensModel};
pub use lens_distort::{
    apply_lens_distortion, bilinear_sample, distort_point, undistort_point, LensDistortParams,
};
pub use ripple::{Ripple, RipplePattern};
pub use wave::{Wave, WaveDirection};

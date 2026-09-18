//! Reverb effects module.
//!
//! Provides multiple reverb algorithms for different applications:
//!
//! - **Freeverb** - Classic algorithmic reverb, efficient and versatile
//! - **Plate Reverb** - Smooth, dense plate reverb simulation
//! - **Convolution Reverb** - Realistic spaces using impulse responses
//! - **Spring Reverb** - Physical waveguide spring reverb simulation
//! - **Cabinet Simulator** - Convolution-based speaker cabinet simulation

pub mod cabinet;
pub mod convolution;
pub mod freeverb;
pub mod overlap_add;
pub mod plate;
pub mod schroeder;
pub mod spring;

// Re-exports
pub use cabinet::{CabinetSimulator, CabinetType};
pub use convolution::{ConvolutionReverb, DoubleBufferConvolver};
pub use freeverb::{Freeverb, StereoMode};
pub use overlap_add::OverlapAddConvolver;
pub use plate::PlateReverb;
pub use schroeder::{SchroederReverb, SimpleConvolutionReverb};
pub use spring::{SpringReverb, SpringReverbConfig};

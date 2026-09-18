//! Format conversion module for sample rate, bit depth, and channel count.

pub mod bit_depth;
pub mod channels;
pub mod sample_rate;

pub use bit_depth::{BitDepthConverter, DitherType};
pub use channels::{ChannelConversionMode, ChannelCountConverter};
pub use sample_rate::{ConversionQuality, ConvertError, SampleRateConverter};

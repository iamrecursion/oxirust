//! Noise reduction modules.

pub mod gate;
pub mod profile;
pub mod subtract;
pub mod wiener;

pub use gate::{NoiseGate, NoiseGateConfig, SpectralGate};
pub use profile::{auto_learn_noise_profile, detect_silent_regions, NoiseProfile};
pub use subtract::{AdaptiveSpectralSubtraction, SpectralSubtraction, SpectralSubtractionConfig};
pub use wiener::{MmseFilter, WienerFilter, WienerFilterConfig};

//! Pitch analysis module using YIN algorithm.

pub mod contour;
pub mod track;
pub mod vibrato;

pub use contour::PitchContour;
pub use track::{PitchEstimate, PitchResult, PitchTracker};
pub use vibrato::{detect_vibrato, VibratoResult};

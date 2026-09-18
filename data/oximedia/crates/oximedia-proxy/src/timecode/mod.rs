//! Timecode management module.

pub mod preserve;
pub mod verify;

pub use preserve::TimecodePreserver;
pub use verify::{TimecodeVerifier, TimecodeVerifyResult};

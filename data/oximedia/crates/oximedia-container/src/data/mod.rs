//! Generic data track support.
//!
//! Provides data track handling for GPS, telemetry, and custom data.

#![forbid(unsafe_code)]

pub mod atom;
pub mod gps;
pub mod telemetry;
pub mod track;

pub use gps::{GpsCoordinate, GpsDataPoint, GpsTrack};
pub use telemetry::{ExposureData, ImuData, TelemetryData, TelemetryTrack};
pub use track::{DataSample, DataTrack, DataTrackBuilder, DataTrackType};

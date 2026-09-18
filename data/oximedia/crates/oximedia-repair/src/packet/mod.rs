//! Packet recovery and repair.
//!
//! This module provides tools for recovering missing or corrupt packets
//! in media streams.

pub mod discard;
pub mod interpolate;
pub mod recover;

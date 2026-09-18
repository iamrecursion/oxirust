//! Multi-track management.
//!
//! Provides sophisticated track management, selection, and routing.

#![forbid(unsafe_code)]

pub mod manager;
pub mod mapping;
pub mod selector;

pub use manager::{
    InterleavingCalculator, SyncMode, TrackInfo, TrackManager, TrackManagerConfig, TrackStats,
};
pub use mapping::{RoutingPresets, TrackMapping, TrackRouter, TrackRoutingBuilder};
pub use selector::{SelectionCriteria, SelectionPresets, TrackSelector, TrackType};

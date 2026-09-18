//! Core Neuromorphic Computing Components
//!
//! This module provides the fundamental building blocks for neuromorphic computing,
//! including spiking neurons, synapses, and spike events that form the basis of
//! brain-inspired spatial algorithms.

pub mod events;
pub mod neurons;
pub mod synapses;

// Re-export core types for convenient access
pub use events::SpikeEvent;
pub use neurons::SpikingNeuron;
pub use synapses::Synapse;

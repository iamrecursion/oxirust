//! Simulation pipeline and GPU scheduling descriptors.
//!
//! [`PhysicsPipeline`] runs the stages `BroadPhase -> NarrowPhase -> ConstraintSolve
//! -> Integration -> PostProcess` over a [`WorldState`] of spheres: a sort-and-sweep
//! broad phase (optionally sorted on the GPU), sphere-sphere contacts, a
//! sequential-impulse contact solver and semi-implicit Euler integration.  The
//! remaining types (compute pipeline descriptors, barriers, frame graph, memory
//! pool, timestamp queries) describe GPU work without needing a device.

pub mod asynccomputequeue_traits;
pub mod computeoverlapscheduler_traits;
pub mod functions;
#[cfg(test)]
mod physics_tests;
pub mod pipelinebuilder_traits;
pub mod pipelineconfig_traits;
pub mod types;

// Re-export all types
pub use types::*;

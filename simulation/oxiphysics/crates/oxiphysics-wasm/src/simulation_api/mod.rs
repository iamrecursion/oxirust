//! WASM Simulation API: high-level types and world management for
//! WebAssembly JavaScript interop.

mod functions;
mod trait_impls;
pub mod types;
pub mod types_3;
pub mod types_4;
pub(crate) mod vec_math;
mod world_contacts;
mod world_dynamics;
#[cfg(test)]
mod world_tests;

// Re-export the public API surface in the original flat shape.
pub use types::{
    BroadPhaseAlgorithm, IntegrationMethod, WasmColliderShape, WasmContactEvent, WasmJointHandle,
    WasmRaycastResult, WasmSceneSerializer, WasmSceneSnapshot,
};
pub use types_3::{SerializedBody, SerializedCollider, WasmPhysicsEvent, WasmWorld};
pub use types_4::{
    WasmBodyState, WasmBodyType, WasmColliderHandle, WasmJoint, WasmJointType, WasmOverlapResult,
    WasmRigidBodyHandle, WasmSimulationConfig,
};

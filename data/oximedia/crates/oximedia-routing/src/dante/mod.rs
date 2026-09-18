//! Dante audio-over-IP module (metadata only, respects Audinate IP).

pub mod discovery;
pub mod metadata;

pub use metadata::{
    DanteDeviceMetadata, DanteFlowMetadata, DanteRouting, DeviceStatus, FlowStatus, NetworkConfig,
};

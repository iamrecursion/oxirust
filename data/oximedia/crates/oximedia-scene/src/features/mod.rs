//! Feature extraction and descriptors.

pub mod descriptor;
pub mod extract;

pub use descriptor::{FeatureDescriptor, FeatureMatch};
pub use extract::{HogFeatures, LocalFeatures};

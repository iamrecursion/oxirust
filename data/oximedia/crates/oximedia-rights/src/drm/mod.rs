//! DRM metadata management module

pub mod encrypt;
pub mod metadata;

pub use encrypt::EncryptionPrep;
pub use metadata::{DrmMetadata, DrmType};

//! Content encryption preparation

use crate::drm::{DrmMetadata, DrmType};

/// Encryption preparation utilities
pub struct EncryptionPrep;

impl EncryptionPrep {
    /// Prepare DRM metadata for encryption
    pub fn prepare_metadata(asset_id: &str, drm_type: DrmType) -> DrmMetadata {
        DrmMetadata::new(asset_id, drm_type)
    }

    /// Generate content ID
    pub fn generate_content_id(asset_id: &str) -> String {
        format!("content_{asset_id}")
    }

    /// Generate encryption key ID
    pub fn generate_key_id(asset_id: &str) -> String {
        format!("key_{asset_id}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_id_generation() {
        let content_id = EncryptionPrep::generate_content_id("asset123");
        assert_eq!(content_id, "content_asset123");
    }

    #[test]
    fn test_key_id_generation() {
        let key_id = EncryptionPrep::generate_key_id("asset123");
        assert_eq!(key_id, "key_asset123");
    }
}

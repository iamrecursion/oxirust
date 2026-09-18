//! Packing List (PKL) document model – SMPTE ST 429-8.
//!
//! A PKL lists every asset in an IMF package together with its hash value so
//! that a receiver can verify integrity.

#![allow(dead_code)]

/// Categorises assets that appear in a PKL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PklAssetType {
    /// Main video / audio MXF track file.
    MxfEssence,
    /// Composition Playlist (CPL) document.
    Cpl,
    /// Output Profile List (OPL) document.
    Opl,
    /// Sidecar data (e.g. subtitles, captions).
    Sidecar,
    /// Unknown / unrecognised asset type.
    Unknown,
}

impl PklAssetType {
    /// Returns `true` when this asset type carries media essence.
    #[must_use]
    pub fn is_essence(self) -> bool {
        matches!(self, Self::MxfEssence)
    }

    /// MIME type string used in PKL XML.
    #[must_use]
    pub fn mime_type(self) -> &'static str {
        match self {
            Self::MxfEssence => "application/mxf",
            Self::Cpl => "text/xml",
            Self::Opl => "text/xml",
            Self::Sidecar => "application/octet-stream",
            Self::Unknown => "application/octet-stream",
        }
    }
}

// ---------------------------------------------------------------------------

/// A single asset entry within a [`PklDocument`].
#[derive(Debug, Clone)]
pub struct PklAsset {
    /// UUID identifying this asset (URN form, e.g. `urn:uuid:...`).
    pub id: String,
    /// Type of asset.
    pub asset_type: PklAssetType,
    /// Expected file size in bytes.
    pub size_bytes: u64,
    /// Expected SHA-1 hash (hex string, 40 chars).
    pub hash_sha1: String,
    /// Original file name (optional).
    pub original_filename: Option<String>,
}

impl PklAsset {
    /// Create a new asset entry.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        asset_type: PklAssetType,
        size_bytes: u64,
        hash_sha1: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            asset_type,
            size_bytes,
            hash_sha1: hash_sha1.into(),
            original_filename: None,
        }
    }

    /// Validate that the stored hash matches the provided computed hash.
    #[must_use]
    pub fn hash_ok(&self, computed_sha1: &str) -> bool {
        !self.hash_sha1.is_empty() && self.hash_sha1.eq_ignore_ascii_case(computed_sha1)
    }

    /// Returns `true` when the entry is minimally complete (non-empty id and hash).
    #[must_use]
    pub fn is_complete(&self) -> bool {
        !self.id.is_empty() && !self.hash_sha1.is_empty()
    }
}

// ---------------------------------------------------------------------------

/// An in-memory representation of a PKL document.
#[derive(Debug, Clone, Default)]
pub struct PklDocument {
    /// UUID of this PKL.
    pub pkl_id: String,
    /// UUID of the annotation text (optional).
    pub annotation_text: Option<String>,
    assets: Vec<PklAsset>,
}

impl PklDocument {
    /// Create a new PKL document with the given identifier.
    #[must_use]
    pub fn new(pkl_id: impl Into<String>) -> Self {
        Self {
            pkl_id: pkl_id.into(),
            annotation_text: None,
            assets: Vec::new(),
        }
    }

    /// Add an asset to the packing list.
    pub fn add_asset(&mut self, asset: PklAsset) {
        self.assets.push(asset);
    }

    /// Find an asset by its UUID. Returns the first match.
    #[must_use]
    pub fn find_by_id(&self, id: &str) -> Option<&PklAsset> {
        self.assets.iter().find(|a| a.id == id)
    }

    /// Total number of assets in this PKL.
    #[must_use]
    pub fn asset_count(&self) -> usize {
        self.assets.len()
    }

    /// Iterate over all assets.
    pub fn assets(&self) -> impl Iterator<Item = &PklAsset> {
        self.assets.iter()
    }

    /// Sum of all declared file sizes.
    #[must_use]
    pub fn total_size_bytes(&self) -> u64 {
        self.assets.iter().map(|a| a.size_bytes).sum()
    }

    /// Returns `true` if every asset is complete.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.assets.iter().all(|a| a.is_complete())
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_asset(id: &str) -> PklAsset {
        PklAsset::new(
            id,
            PklAssetType::MxfEssence,
            1_000_000,
            "aabbccdd1122334455667788990011223344556677",
        )
    }

    #[test]
    fn test_mxf_essence_is_essence() {
        assert!(PklAssetType::MxfEssence.is_essence());
    }

    #[test]
    fn test_cpl_is_not_essence() {
        assert!(!PklAssetType::Cpl.is_essence());
    }

    #[test]
    fn test_opl_is_not_essence() {
        assert!(!PklAssetType::Opl.is_essence());
    }

    #[test]
    fn test_mime_type_mxf() {
        assert_eq!(PklAssetType::MxfEssence.mime_type(), "application/mxf");
    }

    #[test]
    fn test_mime_type_cpl() {
        assert_eq!(PklAssetType::Cpl.mime_type(), "text/xml");
    }

    #[test]
    fn test_asset_hash_ok_matching() {
        let a = sample_asset("id-1");
        assert!(a.hash_ok("AABBCCDD1122334455667788990011223344556677"));
    }

    #[test]
    fn test_asset_hash_ok_mismatch() {
        let a = sample_asset("id-1");
        assert!(!a.hash_ok("000000"));
    }

    #[test]
    fn test_asset_is_complete() {
        let a = sample_asset("id-1");
        assert!(a.is_complete());
    }

    #[test]
    fn test_asset_not_complete_when_empty_id() {
        let a = PklAsset::new("", PklAssetType::Cpl, 100, "abc");
        assert!(!a.is_complete());
    }

    #[test]
    fn test_document_add_and_count() {
        let mut doc = PklDocument::new("pkl-001");
        doc.add_asset(sample_asset("a1"));
        doc.add_asset(sample_asset("a2"));
        assert_eq!(doc.asset_count(), 2);
    }

    #[test]
    fn test_document_find_by_id() {
        let mut doc = PklDocument::new("pkl-001");
        doc.add_asset(sample_asset("target-id"));
        let found = doc.find_by_id("target-id");
        assert!(found.is_some());
    }

    #[test]
    fn test_document_find_by_id_missing() {
        let doc = PklDocument::new("pkl-001");
        assert!(doc.find_by_id("nope").is_none());
    }

    #[test]
    fn test_document_total_size() {
        let mut doc = PklDocument::new("pkl-001");
        doc.add_asset(sample_asset("a1")); // 1_000_000
        doc.add_asset(sample_asset("a2")); // 1_000_000
        assert_eq!(doc.total_size_bytes(), 2_000_000);
    }

    #[test]
    fn test_document_is_complete() {
        let mut doc = PklDocument::new("pkl-001");
        doc.add_asset(sample_asset("a1"));
        assert!(doc.is_complete());
    }

    #[test]
    fn test_empty_document_is_complete() {
        let doc = PklDocument::new("pkl-001");
        // vacuously true
        assert!(doc.is_complete());
    }
}

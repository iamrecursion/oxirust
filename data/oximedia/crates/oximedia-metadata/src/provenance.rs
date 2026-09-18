//! Media provenance chain.
//!
//! This module provides types for tracking the full provenance history
//! of a media asset, including creation, edits, transcodes, and publications.

#![allow(dead_code)]

/// An action that was performed on the media asset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProvenanceAction {
    /// The asset was created.
    Created,
    /// The asset was edited.
    Edited,
    /// The asset was transcoded to another format.
    Transcoded,
    /// The asset was exported.
    Exported,
    /// The asset was published.
    Published,
    /// The asset was archived.
    Archived,
}

impl ProvenanceAction {
    /// Returns true if this action is potentially destructive to the original asset.
    pub fn is_destructive(&self) -> bool {
        matches!(self, Self::Edited | Self::Transcoded)
    }
}

/// A single entry in a provenance chain.
#[derive(Debug, Clone)]
pub struct ProvenanceEntry {
    /// Unix epoch timestamp of this action.
    pub timestamp_epoch: u64,
    /// The action that was performed.
    pub action: ProvenanceAction,
    /// The tool or software used to perform the action.
    pub tool: String,
    /// The human operator or system account responsible.
    pub operator: String,
    /// Hash of the asset before this action (if known).
    pub hash_before: Option<String>,
    /// Hash of the asset after this action (if known).
    pub hash_after: Option<String>,
}

impl ProvenanceEntry {
    /// Create a new provenance entry.
    pub fn new(
        timestamp_epoch: u64,
        action: ProvenanceAction,
        tool: String,
        operator: String,
        hash_before: Option<String>,
        hash_after: Option<String>,
    ) -> Self {
        Self {
            timestamp_epoch,
            action,
            tool,
            operator,
            hash_before,
            hash_after,
        }
    }

    /// Returns true if this entry was performed by an automated system
    /// (operator field starts with "auto:" or "system:").
    pub fn is_automated(&self) -> bool {
        self.operator.starts_with("auto:") || self.operator.starts_with("system:")
    }
}

/// A provenance chain tracking the full history of a media asset.
#[derive(Debug, Clone)]
pub struct ProvenanceChain {
    /// All provenance entries in chronological order.
    pub entries: Vec<ProvenanceEntry>,
    /// Unique identifier of the asset.
    pub asset_id: String,
}

impl ProvenanceChain {
    /// Create a new provenance chain for the given asset.
    pub fn new(asset_id: String) -> Self {
        Self {
            entries: Vec::new(),
            asset_id,
        }
    }

    /// Add a provenance entry to the chain.
    pub fn add_entry(&mut self, entry: ProvenanceEntry) {
        self.entries.push(entry);
    }

    /// Returns the first `Created` entry in the chain, if any.
    pub fn first_creation(&self) -> Option<&ProvenanceEntry> {
        self.entries
            .iter()
            .find(|e| e.action == ProvenanceAction::Created)
    }

    /// Returns the last entry in the chain, if any.
    pub fn last_action(&self) -> Option<&ProvenanceEntry> {
        self.entries.last()
    }

    /// Returns the total number of entries in the chain.
    pub fn action_count(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the chain contains any `Edited` or `Transcoded` entries.
    pub fn has_edits(&self) -> bool {
        self.entries.iter().any(|e| {
            e.action == ProvenanceAction::Edited || e.action == ProvenanceAction::Transcoded
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(
        ts: u64,
        action: ProvenanceAction,
        tool: &str,
        operator: &str,
    ) -> ProvenanceEntry {
        ProvenanceEntry::new(
            ts,
            action,
            tool.to_string(),
            operator.to_string(),
            None,
            None,
        )
    }

    fn make_entry_with_hashes(
        ts: u64,
        action: ProvenanceAction,
        tool: &str,
        operator: &str,
        before: Option<&str>,
        after: Option<&str>,
    ) -> ProvenanceEntry {
        ProvenanceEntry::new(
            ts,
            action,
            tool.to_string(),
            operator.to_string(),
            before.map(str::to_string),
            after.map(str::to_string),
        )
    }

    #[test]
    fn test_provenance_action_is_destructive() {
        assert!(!ProvenanceAction::Created.is_destructive());
        assert!(ProvenanceAction::Edited.is_destructive());
        assert!(ProvenanceAction::Transcoded.is_destructive());
        assert!(!ProvenanceAction::Exported.is_destructive());
        assert!(!ProvenanceAction::Published.is_destructive());
        assert!(!ProvenanceAction::Archived.is_destructive());
    }

    #[test]
    fn test_entry_is_automated_auto_prefix() {
        let entry = make_entry(0, ProvenanceAction::Transcoded, "ffmpeg", "auto:pipeline");
        assert!(entry.is_automated());
    }

    #[test]
    fn test_entry_is_automated_system_prefix() {
        let entry = make_entry(0, ProvenanceAction::Archived, "archiver", "system:backup");
        assert!(entry.is_automated());
    }

    #[test]
    fn test_entry_is_not_automated() {
        let entry = make_entry(0, ProvenanceAction::Edited, "Resolve", "jsmith");
        assert!(!entry.is_automated());
    }

    #[test]
    fn test_provenance_chain_empty() {
        let chain = ProvenanceChain::new("asset-001".to_string());
        assert_eq!(chain.action_count(), 0);
        assert!(chain.first_creation().is_none());
        assert!(chain.last_action().is_none());
        assert!(!chain.has_edits());
    }

    #[test]
    fn test_provenance_chain_add_entry() {
        let mut chain = ProvenanceChain::new("asset-002".to_string());
        chain.add_entry(make_entry(
            1000,
            ProvenanceAction::Created,
            "camera",
            "alice",
        ));
        assert_eq!(chain.action_count(), 1);
    }

    #[test]
    fn test_provenance_chain_first_creation() {
        let mut chain = ProvenanceChain::new("asset-003".to_string());
        chain.add_entry(make_entry(500, ProvenanceAction::Created, "camera", "bob"));
        chain.add_entry(make_entry(600, ProvenanceAction::Edited, "Premiere", "bob"));

        let first = chain.first_creation();
        assert!(first.is_some());
        assert_eq!(first.expect("should succeed in test").timestamp_epoch, 500);
    }

    #[test]
    fn test_provenance_chain_first_creation_none() {
        let mut chain = ProvenanceChain::new("asset-004".to_string());
        chain.add_entry(make_entry(100, ProvenanceAction::Exported, "tool", "user"));
        assert!(chain.first_creation().is_none());
    }

    #[test]
    fn test_provenance_chain_last_action() {
        let mut chain = ProvenanceChain::new("asset-005".to_string());
        chain.add_entry(make_entry(100, ProvenanceAction::Created, "cam", "alice"));
        chain.add_entry(make_entry(200, ProvenanceAction::Published, "cms", "alice"));

        let last = chain.last_action();
        assert!(last.is_some());
        assert_eq!(last.expect("should succeed in test").timestamp_epoch, 200);
        assert_eq!(
            last.expect("should succeed in test").action,
            ProvenanceAction::Published
        );
    }

    #[test]
    fn test_provenance_chain_has_edits_with_edit() {
        let mut chain = ProvenanceChain::new("asset-006".to_string());
        chain.add_entry(make_entry(100, ProvenanceAction::Created, "cam", "user"));
        chain.add_entry(make_entry(
            200,
            ProvenanceAction::Edited,
            "Resolve",
            "editor",
        ));
        assert!(chain.has_edits());
    }

    #[test]
    fn test_provenance_chain_has_edits_with_transcode() {
        let mut chain = ProvenanceChain::new("asset-007".to_string());
        chain.add_entry(make_entry(100, ProvenanceAction::Created, "cam", "user"));
        chain.add_entry(make_entry(
            200,
            ProvenanceAction::Transcoded,
            "ffmpeg",
            "auto:enc",
        ));
        assert!(chain.has_edits());
    }

    #[test]
    fn test_provenance_chain_no_edits() {
        let mut chain = ProvenanceChain::new("asset-008".to_string());
        chain.add_entry(make_entry(100, ProvenanceAction::Created, "cam", "user"));
        chain.add_entry(make_entry(
            200,
            ProvenanceAction::Published,
            "cms",
            "editor",
        ));
        chain.add_entry(make_entry(
            300,
            ProvenanceAction::Archived,
            "vault",
            "auto:archiver",
        ));
        assert!(!chain.has_edits());
    }

    #[test]
    fn test_provenance_chain_action_count() {
        let mut chain = ProvenanceChain::new("asset-009".to_string());
        for i in 0u64..5 {
            chain.add_entry(make_entry(
                i * 100,
                ProvenanceAction::Exported,
                "tool",
                "user",
            ));
        }
        assert_eq!(chain.action_count(), 5);
    }

    #[test]
    fn test_entry_hashes() {
        let entry = make_entry_with_hashes(
            1000,
            ProvenanceAction::Transcoded,
            "ffmpeg",
            "auto:pipe",
            Some("abc123"),
            Some("def456"),
        );
        assert_eq!(entry.hash_before, Some("abc123".to_string()));
        assert_eq!(entry.hash_after, Some("def456".to_string()));
    }

    #[test]
    fn test_provenance_chain_asset_id() {
        let chain = ProvenanceChain::new("my-unique-asset-id".to_string());
        assert_eq!(chain.asset_id, "my-unique-asset-id");
    }
}

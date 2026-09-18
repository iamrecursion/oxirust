//! Media provenance tracking — record and verify the chain of custody for a media asset.

#![allow(dead_code)]

use std::collections::HashMap;

/// An action performed on a media asset.
#[derive(Debug, Clone, PartialEq)]
pub enum ProvenanceAction {
    /// The asset was originally created.
    Created,
    /// The asset was edited with a named tool.
    Edited {
        /// Name of the editing tool used.
        tool: String,
    },
    /// The asset was converted to another format.
    Converted {
        /// Target format (e.g. "mp4", "wav").
        format: String,
    },
    /// The asset was exported to a destination.
    Exported {
        /// Destination path or URI.
        destination: String,
    },
    /// The asset's integrity was verified against a known hash.
    Verified {
        /// The hash that was checked.
        hash: String,
    },
}

impl ProvenanceAction {
    /// Return a short, human-readable tag for this action.
    pub fn label(&self) -> &str {
        match self {
            Self::Created => "Created",
            Self::Edited { .. } => "Edited",
            Self::Converted { .. } => "Converted",
            Self::Exported { .. } => "Exported",
            Self::Verified { .. } => "Verified",
        }
    }
}

/// A single event in a provenance chain.
#[derive(Debug, Clone)]
pub struct ProvenanceEvent {
    /// Unix epoch timestamp (seconds) when the event occurred.
    pub timestamp: u64,
    /// Identity of the actor (person, system, or tool) responsible.
    pub actor: String,
    /// The action that was performed.
    pub action: ProvenanceAction,
    /// Free-form key-value metadata attached to this event.
    pub metadata: HashMap<String, String>,
}

impl ProvenanceEvent {
    /// Convenience constructor with no extra metadata.
    pub fn new(timestamp: u64, actor: &str, action: ProvenanceAction) -> Self {
        Self {
            timestamp,
            actor: actor.to_string(),
            action,
            metadata: HashMap::new(),
        }
    }
}

/// An ordered chain of provenance events for a single media asset.
#[derive(Debug, Clone)]
pub struct ProvenanceChain {
    /// Unique identifier for the media asset.
    pub asset_id: String,
    /// Events in chronological order (oldest first).
    pub events: Vec<ProvenanceEvent>,
}

impl ProvenanceChain {
    /// Create a new, empty chain for the given asset ID.
    pub fn new(asset_id: &str) -> Self {
        Self {
            asset_id: asset_id.to_string(),
            events: Vec::new(),
        }
    }

    /// Append a new event to the chain.
    pub fn add_event(&mut self, event: ProvenanceEvent) {
        self.events.push(event);
    }

    /// Verify that the chain is internally consistent.
    ///
    /// A valid chain must:
    /// - Have at least one event.
    /// - Begin with a `Created` action.
    /// - Have non-decreasing timestamps.
    pub fn verify_chain(&self) -> bool {
        if self.events.is_empty() {
            return false;
        }
        if self.events[0].action != ProvenanceAction::Created {
            return false;
        }
        self.events
            .windows(2)
            .all(|w| w[1].timestamp >= w[0].timestamp)
    }

    /// Return the actor who performed the original `Created` event, if present.
    pub fn original_creator(&self) -> Option<&str> {
        self.events.iter().find_map(|e| {
            if e.action == ProvenanceAction::Created {
                Some(e.actor.as_str())
            } else {
                None
            }
        })
    }

    /// Return `true` if any `Edited` event appears in the chain.
    pub fn was_edited(&self) -> bool {
        self.events
            .iter()
            .any(|e| matches!(e.action, ProvenanceAction::Edited { .. }))
    }

    /// Return the number of `Exported` events in the chain.
    pub fn export_count(&self) -> usize {
        self.events
            .iter()
            .filter(|e| matches!(e.action, ProvenanceAction::Exported { .. }))
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chain() -> ProvenanceChain {
        let mut chain = ProvenanceChain::new("asset-001");
        chain.add_event(ProvenanceEvent::new(
            1_000,
            "alice",
            ProvenanceAction::Created,
        ));
        chain
    }

    #[test]
    fn test_new_chain_is_empty() {
        let chain = ProvenanceChain::new("x");
        assert!(chain.events.is_empty());
        assert_eq!(chain.asset_id, "x");
    }

    #[test]
    fn test_verify_chain_empty_is_invalid() {
        let chain = ProvenanceChain::new("x");
        assert!(!chain.verify_chain());
    }

    #[test]
    fn test_verify_chain_must_start_with_created() {
        let mut chain = ProvenanceChain::new("x");
        chain.add_event(ProvenanceEvent::new(
            100,
            "bob",
            ProvenanceAction::Exported {
                destination: "s3://bucket".to_string(),
            },
        ));
        assert!(!chain.verify_chain());
    }

    #[test]
    fn test_verify_chain_valid_single() {
        let chain = make_chain();
        assert!(chain.verify_chain());
    }

    #[test]
    fn test_verify_chain_non_monotonic_timestamps() {
        let mut chain = make_chain();
        chain.add_event(ProvenanceEvent::new(
            500, // before the Created timestamp of 1000
            "bob",
            ProvenanceAction::Edited {
                tool: "ffmpeg".to_string(),
            },
        ));
        assert!(!chain.verify_chain());
    }

    #[test]
    fn test_original_creator_present() {
        let chain = make_chain();
        assert_eq!(chain.original_creator(), Some("alice"));
    }

    #[test]
    fn test_original_creator_absent() {
        let chain = ProvenanceChain::new("x");
        assert!(chain.original_creator().is_none());
    }

    #[test]
    fn test_was_edited_false_initially() {
        let chain = make_chain();
        assert!(!chain.was_edited());
    }

    #[test]
    fn test_was_edited_after_edit_event() {
        let mut chain = make_chain();
        chain.add_event(ProvenanceEvent::new(
            2_000,
            "editor",
            ProvenanceAction::Edited {
                tool: "davinci".to_string(),
            },
        ));
        assert!(chain.was_edited());
    }

    #[test]
    fn test_export_count_zero() {
        let chain = make_chain();
        assert_eq!(chain.export_count(), 0);
    }

    #[test]
    fn test_export_count_multiple() {
        let mut chain = make_chain();
        for i in 0..3_u64 {
            chain.add_event(ProvenanceEvent::new(
                2_000 + i * 100,
                "delivery",
                ProvenanceAction::Exported {
                    destination: format!("dest-{i}"),
                },
            ));
        }
        assert_eq!(chain.export_count(), 3);
    }

    #[test]
    fn test_provenance_action_label() {
        assert_eq!(ProvenanceAction::Created.label(), "Created");
        assert_eq!(
            ProvenanceAction::Edited {
                tool: "x".to_string()
            }
            .label(),
            "Edited"
        );
        assert_eq!(
            ProvenanceAction::Converted {
                format: "mp4".to_string()
            }
            .label(),
            "Converted"
        );
    }

    #[test]
    fn test_event_metadata() {
        let mut event = ProvenanceEvent::new(1000, "tool", ProvenanceAction::Created);
        event
            .metadata
            .insert("key".to_string(), "value".to_string());
        assert_eq!(event.metadata.get("key").map(|s| s.as_str()), Some("value"));
    }
}

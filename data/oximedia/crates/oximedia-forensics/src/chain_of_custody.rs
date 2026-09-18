//! Media chain of custody and provenance tracking.
//!
//! Tracks the creation chain, modification history, and source attribution
//! of media assets to establish a verified chain of custody.
//!
//! # Data model
//!
//! - [`ChainOfCustody`] is the root record for one media asset (`asset_id`).
//!   It pairs an immutable [`SourceAttribution`] (who created the asset,
//!   when, under what license/copyright) with an append-only, ordered
//!   `Vec<`[`CustodyEvent`]`>` timeline.
//! - Each [`CustodyEvent`] records **who** (`actor`), **what**
//!   ([`CustodyEventType`]: `Creation`, `Transfer`, `Modification`,
//!   `Access`, `Verification`, `Archival`, `Export`, `OwnershipChange`),
//!   **when** (`timestamp_ms`), **where** (`location`: device/system/URL),
//!   and optionally a `content_hash` snapshot of the asset at that point —
//!   analogous to an evidence log entry in a physical chain-of-custody
//!   form.
//! - [`ProvenanceRegistry`] indexes many [`ChainOfCustody`] records by
//!   `asset_id`, so a caller managing a media library can look up, update,
//!   or audit ([`ProvenanceRegistry::find_broken_chains`],
//!   [`ProvenanceRegistry::find_modified_assets`]) every asset's custody
//!   trail in one place.
//!
//! Unlike [`crate::custody`]'s hash-chained `ChainOfCustody` (which
//! cryptographically links each event to its predecessor via an FNV-1a
//! checksum for tamper-evidence of the *log itself*), this module's model
//! favors a richer, queryable event schema — multiple event types, a
//! dedicated [`SourceAttribution`], and per-event free-text
//! `description` — at the cost of not itself being cryptographically
//! sealed. Pick the hash-chained model when the custody log's own
//! integrity must be provable; pick this model when you need structured
//! provenance queries (current custodian, event-type filtering, multi-asset
//! registries).
//!
//! # Verification process
//!
//! [`ChainOfCustody::add_event`] appends the new event and immediately
//! re-runs `verify_chain`, which walks every adjacent event pair and
//! records a [`ChainBreak`] wherever `events[i].timestamp_ms` is *earlier*
//! than `events[i - 1].timestamp_ms` (a [`BreakType::ChronologicalViolation`]
//! — evidence cannot be handed back in time). `chain_intact` is simply
//! `chain_breaks.is_empty()` after that scan, so it always reflects the
//! *entire* history, not just the most recent hop — a break introduced in
//! the middle of a long chain is still detected even if every event before
//! and after it is individually well-ordered (see
//! `test_multi_hop_chain_detects_break_in_the_middle`). Two additional
//! [`BreakType`] variants (`HashMismatch`, `TimeGap`, `UnknownActor`) are
//! defined for callers layering stronger checks (e.g. hash verification
//! against `content_hash`, or gap-duration policies) on top of this
//! baseline chronological check; `verify_chain` does not currently populate
//! them itself. To audit an asset end-to-end, combine `chain_intact` with
//! [`ChainOfCustody::has_modifications`] (were any `Modification` /
//! `OwnershipChange` events recorded?) and
//! [`ChainOfCustody::current_custodian`] (who is the last recorded
//! custodian, from the most recent `Transfer`/`OwnershipChange` event).

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_arguments)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single event in the chain of custody
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustodyEvent {
    /// Unique event ID
    pub event_id: String,
    /// Timestamp (Unix epoch, milliseconds)
    pub timestamp_ms: u64,
    /// Type of event
    pub event_type: CustodyEventType,
    /// Actor who performed this action (user/system)
    pub actor: String,
    /// Location (device, system, or URL)
    pub location: String,
    /// Optional description
    pub description: String,
    /// Hash of the media at this point (SHA-256 hex)
    pub content_hash: Option<String>,
}

impl CustodyEvent {
    /// Create a new custody event
    #[must_use]
    pub fn new(
        event_id: impl Into<String>,
        timestamp_ms: u64,
        event_type: CustodyEventType,
        actor: impl Into<String>,
        location: impl Into<String>,
    ) -> Self {
        Self {
            event_id: event_id.into(),
            timestamp_ms,
            event_type,
            actor: actor.into(),
            location: location.into(),
            description: String::new(),
            content_hash: None,
        }
    }

    /// Set the description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set the content hash
    pub fn with_hash(mut self, hash: impl Into<String>) -> Self {
        self.content_hash = Some(hash.into());
        self
    }
}

/// Type of custody event
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CustodyEventType {
    /// Media was created/captured
    Creation,
    /// Media was transferred between parties
    Transfer,
    /// Media was modified or processed
    Modification,
    /// Media was accessed/viewed
    Access,
    /// Media was verified/authenticated
    Verification,
    /// Media was archived or stored
    Archival,
    /// Media was exported or published
    Export,
    /// Ownership was changed
    OwnershipChange,
}

impl CustodyEventType {
    /// Human-readable name
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Creation => "Creation",
            Self::Transfer => "Transfer",
            Self::Modification => "Modification",
            Self::Access => "Access",
            Self::Verification => "Verification",
            Self::Archival => "Archival",
            Self::Export => "Export",
            Self::OwnershipChange => "OwnershipChange",
        }
    }

    /// Is this event a destructive/modifying event?
    #[must_use]
    pub fn is_modifying(&self) -> bool {
        matches!(self, Self::Modification | Self::OwnershipChange)
    }
}

/// Source attribution record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceAttribution {
    /// Originating source ID
    pub source_id: String,
    /// Original creator/author
    pub creator: String,
    /// Original creation timestamp
    pub created_at_ms: u64,
    /// Copyright holder
    pub copyright_holder: Option<String>,
    /// License under which content is used
    pub license: Option<String>,
    /// Geographic origin
    pub geographic_origin: Option<String>,
    /// Equipment used for capture
    pub capture_equipment: Option<String>,
}

impl SourceAttribution {
    /// Create a new source attribution
    #[must_use]
    pub fn new(
        source_id: impl Into<String>,
        creator: impl Into<String>,
        created_at_ms: u64,
    ) -> Self {
        Self {
            source_id: source_id.into(),
            creator: creator.into(),
            created_at_ms,
            copyright_holder: None,
            license: None,
            geographic_origin: None,
            capture_equipment: None,
        }
    }

    /// Set the copyright holder
    pub fn with_copyright(mut self, holder: impl Into<String>) -> Self {
        self.copyright_holder = Some(holder.into());
        self
    }

    /// Set the license
    pub fn with_license(mut self, license: impl Into<String>) -> Self {
        self.license = Some(license.into());
        self
    }

    /// Set geographic origin
    pub fn with_origin(mut self, origin: impl Into<String>) -> Self {
        self.geographic_origin = Some(origin.into());
        self
    }
}

/// Complete chain of custody for a media asset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainOfCustody {
    /// Asset ID (e.g., file hash or UUID)
    pub asset_id: String,
    /// Source attribution
    pub attribution: SourceAttribution,
    /// Ordered list of custody events
    pub events: Vec<CustodyEvent>,
    /// Whether the chain is verified intact
    pub chain_intact: bool,
    /// Any gaps or breaks detected in the chain
    pub chain_breaks: Vec<ChainBreak>,
}

impl ChainOfCustody {
    /// Create a new chain of custody
    #[must_use]
    pub fn new(asset_id: impl Into<String>, attribution: SourceAttribution) -> Self {
        Self {
            asset_id: asset_id.into(),
            attribution,
            events: Vec::new(),
            chain_intact: true,
            chain_breaks: Vec::new(),
        }
    }

    /// Add an event to the chain
    pub fn add_event(&mut self, event: CustodyEvent) {
        self.events.push(event);
        self.verify_chain();
    }

    /// Number of events in the chain
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Get events of a specific type
    #[must_use]
    pub fn events_of_type(&self, event_type: CustodyEventType) -> Vec<&CustodyEvent> {
        self.events
            .iter()
            .filter(|e| e.event_type == event_type)
            .collect()
    }

    /// Check if the chain has been modified (any modification events)
    #[must_use]
    pub fn has_modifications(&self) -> bool {
        self.events.iter().any(|e| e.event_type.is_modifying())
    }

    /// Get the current custodian (actor of the latest transfer event)
    #[must_use]
    pub fn current_custodian(&self) -> Option<&str> {
        self.events
            .iter()
            .rev()
            .find(|e| {
                matches!(
                    e.event_type,
                    CustodyEventType::Transfer | CustodyEventType::OwnershipChange
                )
            })
            .map(|e| e.actor.as_str())
    }

    /// Verify that events are in chronological order and hashes chain correctly
    fn verify_chain(&mut self) {
        self.chain_breaks.clear();

        // Check chronological order
        for i in 1..self.events.len() {
            if self.events[i].timestamp_ms < self.events[i - 1].timestamp_ms {
                self.chain_breaks.push(ChainBreak {
                    after_event_index: i - 1,
                    break_type: BreakType::ChronologicalViolation,
                    description: format!(
                        "Event {} has earlier timestamp than event {}",
                        self.events[i].event_id,
                        self.events[i - 1].event_id
                    ),
                });
            }
        }

        self.chain_intact = self.chain_breaks.is_empty();
    }

    /// Get time span of the chain in milliseconds
    #[must_use]
    pub fn time_span_ms(&self) -> Option<u64> {
        if self.events.len() < 2 {
            return None;
        }
        let first = self.events.first()?.timestamp_ms;
        let last = self.events.last()?.timestamp_ms;
        Some(last.saturating_sub(first))
    }
}

/// A break or gap detected in the chain of custody
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainBreak {
    /// Index of the event after which the break occurs
    pub after_event_index: usize,
    /// Type of break
    pub break_type: BreakType,
    /// Human-readable description
    pub description: String,
}

/// Type of chain break
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BreakType {
    /// Events are not in chronological order
    ChronologicalViolation,
    /// Content hash does not match expected
    HashMismatch,
    /// Gap in time with no recorded events
    TimeGap,
    /// Actor is unrecognized or untrusted
    UnknownActor,
}

/// Provenance registry for tracking multiple assets
#[derive(Debug, Clone)]
pub struct ProvenanceRegistry {
    /// Map of asset_id to chain of custody
    chains: HashMap<String, ChainOfCustody>,
}

impl ProvenanceRegistry {
    /// Create a new provenance registry
    #[must_use]
    pub fn new() -> Self {
        Self {
            chains: HashMap::new(),
        }
    }

    /// Register a new asset
    pub fn register(&mut self, chain: ChainOfCustody) {
        self.chains.insert(chain.asset_id.clone(), chain);
    }

    /// Get the chain for an asset
    #[must_use]
    pub fn get(&self, asset_id: &str) -> Option<&ChainOfCustody> {
        self.chains.get(asset_id)
    }

    /// Get a mutable chain for an asset
    pub fn get_mut(&mut self, asset_id: &str) -> Option<&mut ChainOfCustody> {
        self.chains.get_mut(asset_id)
    }

    /// Add an event to an asset's chain
    pub fn add_event(&mut self, asset_id: &str, event: CustodyEvent) -> bool {
        if let Some(chain) = self.chains.get_mut(asset_id) {
            chain.add_event(event);
            true
        } else {
            false
        }
    }

    /// Total number of tracked assets
    #[must_use]
    pub fn asset_count(&self) -> usize {
        self.chains.len()
    }

    /// Find all assets with broken chains
    #[must_use]
    pub fn find_broken_chains(&self) -> Vec<&str> {
        self.chains
            .values()
            .filter(|c| !c.chain_intact)
            .map(|c| c.asset_id.as_str())
            .collect()
    }

    /// Find all modified assets
    #[must_use]
    pub fn find_modified_assets(&self) -> Vec<&str> {
        self.chains
            .values()
            .filter(|c| c.has_modifications())
            .map(|c| c.asset_id.as_str())
            .collect()
    }

    /// Remove an asset from the registry
    pub fn remove(&mut self, asset_id: &str) -> Option<ChainOfCustody> {
        self.chains.remove(asset_id)
    }
}

impl Default for ProvenanceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_attribution(id: &str) -> SourceAttribution {
        SourceAttribution::new(id, "Test Creator", 1_000_000)
    }

    fn make_event(id: &str, ts: u64, etype: CustodyEventType) -> CustodyEvent {
        CustodyEvent::new(id, ts, etype, "actor", "location")
    }

    #[test]
    fn test_custody_event_creation() {
        let event = make_event("evt1", 5000, CustodyEventType::Creation);
        assert_eq!(event.event_id, "evt1");
        assert_eq!(event.timestamp_ms, 5000);
        assert_eq!(event.event_type, CustodyEventType::Creation);
    }

    #[test]
    fn test_custody_event_with_description() {
        let event =
            make_event("e1", 0, CustodyEventType::Access).with_description("Reviewed by editor");
        assert_eq!(event.description, "Reviewed by editor");
    }

    #[test]
    fn test_custody_event_with_hash() {
        let event = make_event("e1", 0, CustodyEventType::Verification).with_hash("abc123");
        assert_eq!(event.content_hash, Some("abc123".to_string()));
    }

    #[test]
    fn test_custody_event_type_as_str() {
        assert_eq!(CustodyEventType::Creation.as_str(), "Creation");
        assert_eq!(CustodyEventType::Modification.as_str(), "Modification");
        assert_eq!(CustodyEventType::Transfer.as_str(), "Transfer");
    }

    #[test]
    fn test_custody_event_type_is_modifying() {
        assert!(CustodyEventType::Modification.is_modifying());
        assert!(CustodyEventType::OwnershipChange.is_modifying());
        assert!(!CustodyEventType::Access.is_modifying());
        assert!(!CustodyEventType::Verification.is_modifying());
    }

    #[test]
    fn test_source_attribution_creation() {
        let attr = make_attribution("asset1");
        assert_eq!(attr.source_id, "asset1");
        assert_eq!(attr.creator, "Test Creator");
        assert_eq!(attr.created_at_ms, 1_000_000);
    }

    #[test]
    fn test_source_attribution_builder() {
        let attr = SourceAttribution::new("id", "creator", 0)
            .with_copyright("ACME Corp")
            .with_license("CC-BY-4.0")
            .with_origin("United States");
        assert_eq!(attr.copyright_holder, Some("ACME Corp".to_string()));
        assert_eq!(attr.license, Some("CC-BY-4.0".to_string()));
        assert_eq!(attr.geographic_origin, Some("United States".to_string()));
    }

    #[test]
    fn test_chain_of_custody_creation() {
        let chain = ChainOfCustody::new("asset1", make_attribution("asset1"));
        assert_eq!(chain.asset_id, "asset1");
        assert_eq!(chain.event_count(), 0);
        assert!(chain.chain_intact);
    }

    #[test]
    fn test_chain_add_events_chronological() {
        let mut chain = ChainOfCustody::new("a1", make_attribution("a1"));
        chain.add_event(make_event("e1", 1000, CustodyEventType::Creation));
        chain.add_event(make_event("e2", 2000, CustodyEventType::Transfer));
        chain.add_event(make_event("e3", 3000, CustodyEventType::Access));
        assert_eq!(chain.event_count(), 3);
        assert!(chain.chain_intact);
    }

    #[test]
    fn test_chain_detects_chronological_violation() {
        let mut chain = ChainOfCustody::new("a1", make_attribution("a1"));
        chain.add_event(make_event("e1", 2000, CustodyEventType::Creation));
        chain.add_event(make_event("e2", 1000, CustodyEventType::Transfer)); // Earlier timestamp!
        assert!(!chain.chain_intact);
        assert!(!chain.chain_breaks.is_empty());
        assert_eq!(
            chain.chain_breaks[0].break_type,
            BreakType::ChronologicalViolation
        );
    }

    #[test]
    fn test_chain_has_modifications() {
        let mut chain = ChainOfCustody::new("a1", make_attribution("a1"));
        chain.add_event(make_event("e1", 1000, CustodyEventType::Creation));
        assert!(!chain.has_modifications());
        chain.add_event(make_event("e2", 2000, CustodyEventType::Modification));
        assert!(chain.has_modifications());
    }

    #[test]
    fn test_chain_current_custodian() {
        let mut chain = ChainOfCustody::new("a1", make_attribution("a1"));
        assert!(chain.current_custodian().is_none());

        let mut e1 = make_event("e1", 1000, CustodyEventType::Transfer);
        e1.actor = "Alice".to_string();
        chain.add_event(e1);
        assert_eq!(chain.current_custodian(), Some("Alice"));

        let mut e2 = make_event("e2", 2000, CustodyEventType::Transfer);
        e2.actor = "Bob".to_string();
        chain.add_event(e2);
        assert_eq!(chain.current_custodian(), Some("Bob"));
    }

    #[test]
    fn test_chain_time_span() {
        let mut chain = ChainOfCustody::new("a1", make_attribution("a1"));
        assert!(chain.time_span_ms().is_none());
        chain.add_event(make_event("e1", 1000, CustodyEventType::Creation));
        assert!(chain.time_span_ms().is_none());
        chain.add_event(make_event("e2", 5000, CustodyEventType::Access));
        assert_eq!(chain.time_span_ms(), Some(4000));
    }

    #[test]
    fn test_chain_events_of_type() {
        let mut chain = ChainOfCustody::new("a1", make_attribution("a1"));
        chain.add_event(make_event("e1", 1000, CustodyEventType::Creation));
        chain.add_event(make_event("e2", 2000, CustodyEventType::Access));
        chain.add_event(make_event("e3", 3000, CustodyEventType::Access));

        let accesses = chain.events_of_type(CustodyEventType::Access);
        assert_eq!(accesses.len(), 2);
    }

    #[test]
    fn test_provenance_registry_creation() {
        let registry = ProvenanceRegistry::new();
        assert_eq!(registry.asset_count(), 0);
    }

    #[test]
    fn test_provenance_registry_register_and_get() {
        let mut registry = ProvenanceRegistry::new();
        let chain = ChainOfCustody::new("asset1", make_attribution("asset1"));
        registry.register(chain);
        assert_eq!(registry.asset_count(), 1);
        assert!(registry.get("asset1").is_some());
        assert!(registry.get("missing").is_none());
    }

    #[test]
    fn test_provenance_registry_add_event() {
        let mut registry = ProvenanceRegistry::new();
        let chain = ChainOfCustody::new("a1", make_attribution("a1"));
        registry.register(chain);
        let ok = registry.add_event("a1", make_event("e1", 1000, CustodyEventType::Creation));
        assert!(ok);
        assert_eq!(
            registry
                .get("a1")
                .expect("get should succeed")
                .event_count(),
            1
        );

        let not_ok =
            registry.add_event("missing", make_event("e2", 2000, CustodyEventType::Access));
        assert!(!not_ok);
    }

    #[test]
    fn test_provenance_registry_find_broken_chains() {
        let mut registry = ProvenanceRegistry::new();
        let mut chain = ChainOfCustody::new("a1", make_attribution("a1"));
        chain.add_event(make_event("e1", 2000, CustodyEventType::Creation));
        chain.add_event(make_event("e2", 1000, CustodyEventType::Transfer)); // Break!
        registry.register(chain);

        let intact = ChainOfCustody::new("a2", make_attribution("a2"));
        registry.register(intact);

        let broken = registry.find_broken_chains();
        assert_eq!(broken.len(), 1);
        assert_eq!(broken[0], "a1");
    }

    #[test]
    fn test_provenance_registry_remove() {
        let mut registry = ProvenanceRegistry::new();
        registry.register(ChainOfCustody::new("a1", make_attribution("a1")));
        let removed = registry.remove("a1");
        assert!(removed.is_some());
        assert_eq!(registry.asset_count(), 0);
    }

    // ── Multi-step / multi-hop custody transfer scenarios ─────────────────────

    /// A five-party evidence handoff: Creation → four sequential Transfers.
    /// The current custodian must track the *last* transfer at every hop, and
    /// the chain must stay intact throughout since all timestamps increase
    /// monotonically.
    #[test]
    fn test_multi_hop_transfer_chain_tracks_current_custodian_at_each_hop() {
        let mut chain = ChainOfCustody::new("evidence-1", make_attribution("evidence-1"));

        chain.add_event(make_event("e0", 1000, CustodyEventType::Creation));
        assert!(chain.current_custodian().is_none());

        let hops = [
            "Photographer",
            "Field Investigator",
            "Evidence Locker",
            "Forensic Lab",
        ];
        for (i, actor) in hops.iter().enumerate() {
            let mut evt = make_event(
                &format!("hop{i}"),
                2000 + (i as u64) * 1000,
                CustodyEventType::Transfer,
            );
            evt.actor = (*actor).to_string();
            chain.add_event(evt);

            // After each hop, the custodian must be exactly this hop's actor.
            assert_eq!(
                chain.current_custodian(),
                Some(*actor),
                "custodian mismatch after hop {i}"
            );
            assert!(
                chain.chain_intact,
                "chain must remain intact after monotonic hop {i}"
            );
        }

        // Creation + 4 transfers.
        assert_eq!(chain.event_count(), 5);
        assert_eq!(chain.events_of_type(CustodyEventType::Transfer).len(), 4);
        assert!(chain.chain_breaks.is_empty());
    }

    /// A long custody chain that interleaves Transfer, Modification, Access,
    /// and Verification events across many hops must still report
    /// `has_modifications() == true` (sticky once any modification occurs)
    /// while continuing to track the latest custodian through subsequent
    /// non-transfer hops.
    #[test]
    fn test_multi_hop_chain_with_interleaved_modifications_and_access() {
        let mut chain = ChainOfCustody::new("evidence-2", make_attribution("evidence-2"));

        chain.add_event(make_event("e0", 1000, CustodyEventType::Creation));

        let mut alice = make_event("e1", 2000, CustodyEventType::Transfer);
        alice.actor = "Alice".to_string();
        chain.add_event(alice);
        assert!(!chain.has_modifications());

        chain.add_event(make_event("e2", 2500, CustodyEventType::Access));
        // Access does not change custodian or modification status.
        assert_eq!(chain.current_custodian(), Some("Alice"));
        assert!(!chain.has_modifications());

        chain.add_event(make_event("e3", 3000, CustodyEventType::Modification));
        assert!(chain.has_modifications());
        // Modification alone is not a Transfer/OwnershipChange, so custodian
        // stays the last transfer actor (Alice).
        assert_eq!(chain.current_custodian(), Some("Alice"));

        let mut bob = make_event("e4", 4000, CustodyEventType::Transfer);
        bob.actor = "Bob".to_string();
        chain.add_event(bob);
        assert_eq!(chain.current_custodian(), Some("Bob"));

        chain.add_event(make_event("e5", 5000, CustodyEventType::Verification));
        assert_eq!(chain.current_custodian(), Some("Bob"));

        let mut carol = make_event("e6", 6000, CustodyEventType::OwnershipChange);
        carol.actor = "Carol".to_string();
        chain.add_event(carol);
        assert_eq!(chain.current_custodian(), Some("Carol"));

        // Sticky modification flag persists to the end of the chain.
        assert!(chain.has_modifications());
        assert!(chain.chain_intact);
        assert_eq!(chain.event_count(), 7);
    }

    /// A break introduced mid-chain (an out-of-order timestamp at hop 3 of 5)
    /// must be detected even though the hops before and after it are each
    /// individually monotonic — i.e. `verify_chain` must scan the whole
    /// sequence, not just the last transition.
    #[test]
    fn test_multi_hop_chain_detects_break_in_the_middle() {
        let mut chain = ChainOfCustody::new("evidence-3", make_attribution("evidence-3"));

        chain.add_event(make_event("e0", 1000, CustodyEventType::Creation));
        chain.add_event(make_event("e1", 2000, CustodyEventType::Transfer)); // hop 1: ok
        chain.add_event(make_event("e2", 3000, CustodyEventType::Transfer)); // hop 2: ok
        chain.add_event(make_event("e3", 1500, CustodyEventType::Transfer)); // hop 3: BREAK (earlier than e2)
        chain.add_event(make_event("e4", 4000, CustodyEventType::Transfer)); // hop 4: ok relative to e3.. but chain already broken

        assert!(!chain.chain_intact);
        assert_eq!(chain.chain_breaks.len(), 1);
        assert_eq!(
            chain.chain_breaks[0].break_type,
            BreakType::ChronologicalViolation
        );
        // The break is anchored between event index 2 (e2) and 3 (e3).
        assert_eq!(chain.chain_breaks[0].after_event_index, 2);

        // Despite the break, the chain still records every hop and the final
        // custodian is still derivable from the last Transfer event added.
        assert_eq!(chain.event_count(), 5);
        assert_eq!(chain.current_custodian(), Some("actor"));
    }

    /// Registering several assets, each with its own multi-hop transfer
    /// chain, must let the registry independently track per-asset custodian,
    /// modification, and chain-integrity state.
    #[test]
    fn test_provenance_registry_tracks_independent_multi_hop_chains() {
        let mut registry = ProvenanceRegistry::new();

        // Asset A: clean 3-hop transfer chain.
        let mut chain_a = ChainOfCustody::new("asset-a", make_attribution("asset-a"));
        chain_a.add_event(make_event("a0", 1000, CustodyEventType::Creation));
        registry.register(chain_a);
        for (i, actor) in ["A1", "A2", "A3"].iter().enumerate() {
            let mut evt = make_event(
                &format!("a{}", i + 1),
                2000 + (i as u64) * 1000,
                CustodyEventType::Transfer,
            );
            evt.actor = (*actor).to_string();
            assert!(registry.add_event("asset-a", evt));
        }

        // Asset B: 2-hop chain with a modification in between.
        let mut chain_b = ChainOfCustody::new("asset-b", make_attribution("asset-b"));
        chain_b.add_event(make_event("b0", 1000, CustodyEventType::Creation));
        registry.register(chain_b);
        assert!(registry.add_event(
            "asset-b",
            make_event("b1", 2000, CustodyEventType::Transfer)
        ));
        assert!(registry.add_event(
            "asset-b",
            make_event("b2", 3000, CustodyEventType::Modification)
        ));

        // Asset C: broken chain (out-of-order transfer).
        let mut chain_c = ChainOfCustody::new("asset-c", make_attribution("asset-c"));
        chain_c.add_event(make_event("c0", 5000, CustodyEventType::Creation));
        registry.register(chain_c);
        assert!(registry.add_event(
            "asset-c",
            make_event("c1", 1000, CustodyEventType::Transfer)
        ));

        assert_eq!(registry.asset_count(), 3);

        assert_eq!(
            registry
                .get("asset-a")
                .expect("asset-a must exist")
                .current_custodian(),
            Some("A3")
        );
        assert!(!registry
            .get("asset-a")
            .expect("asset-a must exist")
            .has_modifications());

        assert!(registry
            .get("asset-b")
            .expect("asset-b must exist")
            .has_modifications());

        let broken = registry.find_broken_chains();
        assert_eq!(broken, vec!["asset-c"]);

        let modified = registry.find_modified_assets();
        assert_eq!(modified, vec!["asset-b"]);
    }
}

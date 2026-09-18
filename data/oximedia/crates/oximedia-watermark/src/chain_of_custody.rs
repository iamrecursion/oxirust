//! Chain of custody tracking via watermarks.
//!
//! This module provides:
//! - `CustodyEvent` / `CustodyEventType`: events in the chain
//! - `CustodyChain`: ordered chain of events with monotonicity verification
//! - `CustodyWatermark`: compact watermark derived from the chain
//! - Embedding / extraction helpers

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// CustodyEventType
// ---------------------------------------------------------------------------

/// The type of a custody event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CustodyEventType {
    /// Asset was created.
    Created,
    /// Asset was modified.
    Modified,
    /// Asset was transferred to a new owner.
    Transferred,
    /// Asset was delivered to a recipient.
    Delivered,
    /// Asset was archived.
    Archived,
    /// Asset was deleted.
    Deleted,
}

// ---------------------------------------------------------------------------
// CustodyEvent
// ---------------------------------------------------------------------------

/// A single event in a chain of custody.
#[derive(Debug, Clone)]
pub struct CustodyEvent {
    /// Type of event.
    pub event_type: CustodyEventType,
    /// Actor (person, service, system) responsible.
    pub actor: String,
    /// Unix epoch timestamp in milliseconds.
    pub timestamp_ms: u64,
    /// Location or endpoint where the event occurred.
    pub location: String,
}

impl CustodyEvent {
    /// Create a new event.
    #[must_use]
    pub fn new(
        event_type: CustodyEventType,
        actor: impl Into<String>,
        timestamp_ms: u64,
        location: impl Into<String>,
    ) -> Self {
        Self {
            event_type,
            actor: actor.into(),
            timestamp_ms,
            location: location.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// CustodyChain
// ---------------------------------------------------------------------------

/// An ordered chain of custody events for a media asset.
#[derive(Debug, Clone)]
pub struct CustodyChain {
    /// Asset identifier.
    pub asset_id: String,
    /// Ordered list of events.
    pub events: Vec<CustodyEvent>,
}

impl CustodyChain {
    /// Create a new, empty chain for `asset_id`.
    #[must_use]
    pub fn new(asset_id: impl Into<String>) -> Self {
        Self {
            asset_id: asset_id.into(),
            events: Vec::new(),
        }
    }

    /// Append an event to the chain.
    pub fn add_event(&mut self, event: CustodyEvent) {
        self.events.push(event);
    }

    /// Verify that timestamps are monotonically non-decreasing.
    ///
    /// Returns `true` if the chain is valid, `false` otherwise.
    #[must_use]
    pub fn verify_chain(&self) -> bool {
        if self.events.len() < 2 {
            return true;
        }
        self.events
            .windows(2)
            .all(|w| w[0].timestamp_ms <= w[1].timestamp_ms)
    }

    /// Number of events in the chain.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// True if the chain has no events.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

// ---------------------------------------------------------------------------
// CustodyWatermark
// ---------------------------------------------------------------------------

/// A compact watermark representing the current state of a `CustodyChain`.
///
/// Fits in a small number of bytes for embedding into the metadata region
/// of a video frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CustodyWatermark {
    /// FNV-64 hash of the concatenated chain events.
    pub chain_hash: u64,
    /// Number of events in the chain.
    pub event_count: u32,
    /// FNV-32 hash of the last actor's name.
    pub last_actor_hash: u32,
}

impl CustodyWatermark {
    /// Derive a compact watermark from a `CustodyChain`.
    #[must_use]
    pub fn from_chain(chain: &CustodyChain) -> Self {
        let mut hasher = FnvHasher64::new();

        // Hash the asset ID
        hasher.update(chain.asset_id.as_bytes());

        // Hash each event
        for event in &chain.events {
            hasher.update(event.actor.as_bytes());
            hasher.update(event.location.as_bytes());
            hasher.update(&event.timestamp_ms.to_le_bytes());
            hasher.update(&[event.event_type as u8]);
        }

        let chain_hash = hasher.finish();

        // Last actor hash (FNV-32)
        let last_actor_hash = chain.events.last().map_or(0, |e| fnv32(e.actor.as_bytes()));

        Self {
            chain_hash,
            event_count: chain.events.len() as u32,
            last_actor_hash,
        }
    }

    /// Embed the watermark into the first row of `frame` as a metadata region.
    ///
    /// `width` is the frame width in pixels (RGBA: 4 bytes per pixel).
    /// Returns `true` if there was enough space to embed.
    pub fn embed(&self, frame: &mut Vec<u8>, width: u32) -> bool {
        // We need 16 bytes: 8 (chain_hash) + 4 (event_count) + 4 (last_actor_hash)
        let needed = 16usize;
        let available = (width as usize) * 4; // first row in RGBA

        if frame.len() < needed || available < needed {
            return false;
        }

        // Write into the first 16 bytes (first 4 pixels) of the frame
        let chain_bytes = self.chain_hash.to_le_bytes();
        let count_bytes = self.event_count.to_le_bytes();
        let actor_bytes = self.last_actor_hash.to_le_bytes();

        frame[0..8].copy_from_slice(&chain_bytes);
        frame[8..12].copy_from_slice(&count_bytes);
        frame[12..16].copy_from_slice(&actor_bytes);

        true
    }

    /// Extract a `CustodyWatermark` from the metadata region of `frame`.
    ///
    /// Returns `None` if the frame is too small.
    #[must_use]
    pub fn extract(frame: &[u8]) -> Option<Self> {
        if frame.len() < 16 {
            return None;
        }

        let chain_hash = u64::from_le_bytes(frame[0..8].try_into().ok()?);
        let event_count = u32::from_le_bytes(frame[8..12].try_into().ok()?);
        let last_actor_hash = u32::from_le_bytes(frame[12..16].try_into().ok()?);

        Some(Self {
            chain_hash,
            event_count,
            last_actor_hash,
        })
    }
}

// ---------------------------------------------------------------------------
// FNV helpers
// ---------------------------------------------------------------------------

struct FnvHasher64 {
    state: u64,
}

impl FnvHasher64 {
    fn new() -> Self {
        Self {
            state: 0xcbf2_9ce4_8422_2325u64,
        }
    }

    fn update(&mut self, data: &[u8]) {
        for &byte in data {
            self.state ^= u64::from(byte);
            self.state = self.state.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn finish(self) -> u64 {
        self.state
    }
}

fn fnv32(data: &[u8]) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for &byte in data {
        h ^= u32::from(byte);
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(ts: u64) -> CustodyEvent {
        CustodyEvent::new(CustodyEventType::Modified, "alice", ts, "server-1")
    }

    // --- CustodyChain tests ---

    #[test]
    fn test_chain_new_empty() {
        let chain = CustodyChain::new("asset-001");
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
    }

    #[test]
    fn test_chain_add_event() {
        let mut chain = CustodyChain::new("asset-001");
        chain.add_event(make_event(1000));
        assert_eq!(chain.len(), 1);
    }

    #[test]
    fn test_chain_verify_empty() {
        let chain = CustodyChain::new("x");
        assert!(chain.verify_chain());
    }

    #[test]
    fn test_chain_verify_single() {
        let mut chain = CustodyChain::new("x");
        chain.add_event(make_event(1000));
        assert!(chain.verify_chain());
    }

    #[test]
    fn test_chain_verify_monotone() {
        let mut chain = CustodyChain::new("x");
        chain.add_event(make_event(1000));
        chain.add_event(make_event(2000));
        chain.add_event(make_event(3000));
        assert!(chain.verify_chain());
    }

    #[test]
    fn test_chain_verify_equal_timestamps_ok() {
        let mut chain = CustodyChain::new("x");
        chain.add_event(make_event(1000));
        chain.add_event(make_event(1000)); // same timestamp is valid
        assert!(chain.verify_chain());
    }

    #[test]
    fn test_chain_verify_not_monotone() {
        let mut chain = CustodyChain::new("x");
        chain.add_event(make_event(3000));
        chain.add_event(make_event(1000)); // goes backwards
        assert!(!chain.verify_chain());
    }

    // --- CustodyWatermark tests ---

    #[test]
    fn test_watermark_from_chain_deterministic() {
        let mut chain = CustodyChain::new("asset-042");
        chain.add_event(make_event(500));
        chain.add_event(CustodyEvent::new(
            CustodyEventType::Transferred,
            "bob",
            1000,
            "cdn-edge",
        ));

        let wm1 = CustodyWatermark::from_chain(&chain);
        let wm2 = CustodyWatermark::from_chain(&chain);
        assert_eq!(wm1, wm2);
    }

    #[test]
    fn test_watermark_event_count() {
        let mut chain = CustodyChain::new("a");
        for ts in 0..5 {
            chain.add_event(make_event(ts * 100));
        }
        let wm = CustodyWatermark::from_chain(&chain);
        assert_eq!(wm.event_count, 5);
    }

    #[test]
    fn test_watermark_different_chains_differ() {
        let mut chain_a = CustodyChain::new("asset-A");
        chain_a.add_event(make_event(100));
        let mut chain_b = CustodyChain::new("asset-B");
        chain_b.add_event(make_event(200));

        let wm_a = CustodyWatermark::from_chain(&chain_a);
        let wm_b = CustodyWatermark::from_chain(&chain_b);
        assert_ne!(wm_a.chain_hash, wm_b.chain_hash);
    }

    #[test]
    fn test_watermark_embed_roundtrip() {
        let mut chain = CustodyChain::new("embed-test");
        chain.add_event(make_event(100));
        chain.add_event(make_event(200));

        let wm = CustodyWatermark::from_chain(&chain);

        // Frame: 100 pixels wide, 10 rows high (RGBA)
        let mut frame = vec![0u8; 100 * 10 * 4];
        let ok = wm.embed(&mut frame, 100);
        assert!(ok);

        let extracted = CustodyWatermark::extract(&frame).expect("should succeed in test");
        assert_eq!(extracted.chain_hash, wm.chain_hash);
        assert_eq!(extracted.event_count, wm.event_count);
        assert_eq!(extracted.last_actor_hash, wm.last_actor_hash);
    }

    #[test]
    fn test_watermark_embed_too_small() {
        let mut chain = CustodyChain::new("tiny");
        chain.add_event(make_event(0));
        let wm = CustodyWatermark::from_chain(&chain);

        let mut frame = vec![0u8; 4]; // only 1 pixel – not enough
        let ok = wm.embed(&mut frame, 1);
        assert!(!ok);
    }

    #[test]
    fn test_watermark_extract_too_small() {
        let result = CustodyWatermark::extract(&[0u8; 8]);
        assert!(result.is_none());
    }

    #[test]
    fn test_watermark_empty_chain() {
        let chain = CustodyChain::new("empty");
        let wm = CustodyWatermark::from_chain(&chain);
        assert_eq!(wm.event_count, 0);
        assert_eq!(wm.last_actor_hash, 0);
    }

    #[test]
    fn test_custody_event_types() {
        let types = [
            CustodyEventType::Created,
            CustodyEventType::Modified,
            CustodyEventType::Transferred,
            CustodyEventType::Delivered,
            CustodyEventType::Archived,
            CustodyEventType::Deleted,
        ];
        for t in &types {
            let event = CustodyEvent::new(*t, "system", 0, "dc1");
            assert_eq!(event.event_type, *t);
        }
    }
}

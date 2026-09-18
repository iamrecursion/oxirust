//! Conforming engine for relinking proxies to originals.

use super::edl::EdlConformer;
use crate::{ProxyLinkManager, Result};
use oximedia_edl::event::EdlEvent;
use oximedia_edl::Edl;
use std::path::Path;

/// Conforming engine for proxy-to-original workflows.
pub struct ConformEngine {
    link_manager: ProxyLinkManager,
}

impl ConformEngine {
    /// Create a new conform engine with the specified link database.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened.
    pub async fn new(db_path: impl AsRef<Path>) -> Result<Self> {
        let link_manager = ProxyLinkManager::new(db_path).await?;
        Ok(Self { link_manager })
    }

    /// Conform from an EDL file.
    ///
    /// # Errors
    ///
    /// Returns an error if conforming fails.
    pub async fn conform_from_edl(
        &self,
        edl_path: impl AsRef<Path>,
        output: impl AsRef<Path>,
    ) -> Result<ConformResult> {
        let conformer = EdlConformer::new(&self.link_manager);
        conformer.conform(edl_path, output).await
    }

    /// Relink a single proxy file to its original.
    ///
    /// # Errors
    ///
    /// Returns an error if no link exists for the proxy.
    pub fn relink(&self, proxy_path: impl AsRef<Path>) -> Result<&Path> {
        self.link_manager.get_original(proxy_path)
    }

    /// Get the link manager.
    #[must_use]
    pub const fn link_manager(&self) -> &ProxyLinkManager {
        &self.link_manager
    }

    /// Batch-conform multiple EDLs into a single merged timeline.
    ///
    /// Events from all input EDLs are collected and sorted by their record-in
    /// timecode. Overlapping events on the same track are resolved according to
    /// `strategy`. The method returns the merged event list together with a
    /// provenance map that identifies the source EDL index for each event.
    ///
    /// # Panics
    ///
    /// Does not panic; returns an empty result for an empty `edls` slice.
    #[must_use]
    pub fn batch_conform(&self, edls: &[Edl], strategy: MergeStrategy) -> BatchConformResult {
        // 1. Collect all events tagged with their source EDL index.
        let mut tagged: Vec<(usize, EdlEvent)> = edls
            .iter()
            .enumerate()
            .flat_map(|(src_idx, edl)| edl.events.iter().cloned().map(move |ev| (src_idx, ev)))
            .collect();

        // 2. Sort by record-in timecode (ascending).
        tagged.sort_by_key(|(_, ev)| ev.record_in.to_frames());

        // 3. Resolve overlaps per strategy.
        let merged: Vec<(usize, EdlEvent)> = match strategy {
            MergeStrategy::LayerToTracks => {
                // All events are kept; assign them sequential track layers based
                // on a per-source-EDL virtual track number (no dropping).
                tagged
            }
            MergeStrategy::PreferEarlier => {
                resolve_overlaps(tagged, |existing, candidate| {
                    // Keep whichever has the smaller record-in (earlier wins).
                    // Because we sorted by record-in the existing entry is
                    // always earlier or equal; drop the candidate.
                    let keep_existing =
                        existing.record_in.to_frames() <= candidate.record_in.to_frames();
                    if keep_existing {
                        KeepSide::Existing
                    } else {
                        KeepSide::Candidate
                    }
                })
            }
            MergeStrategy::PreferLonger => resolve_overlaps(tagged, |existing, candidate| {
                let existing_dur = existing
                    .record_out
                    .to_frames()
                    .saturating_sub(existing.record_in.to_frames());
                let candidate_dur = candidate
                    .record_out
                    .to_frames()
                    .saturating_sub(candidate.record_in.to_frames());
                if existing_dur >= candidate_dur {
                    KeepSide::Existing
                } else {
                    KeepSide::Candidate
                }
            }),
        };

        // 4. Build output structures.
        let mut events = Vec::with_capacity(merged.len());
        let mut provenance = Vec::with_capacity(merged.len());

        for (event_index, (source_edl_index, event)) in merged.into_iter().enumerate() {
            events.push(ConformedEvent {
                event,
                track_layer: source_edl_index,
            });
            provenance.push(EventProvenance {
                event_index,
                source_edl_index,
            });
        }

        BatchConformResult { events, provenance }
    }
}

/// Which side to keep when resolving an overlap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KeepSide {
    Existing,
    Candidate,
}

/// Walk through a sorted list of `(src_idx, EdlEvent)` and resolve overlapping
/// events on the same track using the provided `resolver` closure.
fn resolve_overlaps(
    tagged: Vec<(usize, EdlEvent)>,
    mut resolver: impl FnMut(&EdlEvent, &EdlEvent) -> KeepSide,
) -> Vec<(usize, EdlEvent)> {
    let mut result: Vec<(usize, EdlEvent)> = Vec::with_capacity(tagged.len());

    'outer: for (src_idx, candidate) in tagged {
        // Check whether `candidate` overlaps any already-accepted event on the
        // same track type.
        for slot in result.iter_mut() {
            if slot.1.overlaps_with(&candidate) {
                let keep = resolver(&slot.1, &candidate);
                match keep {
                    KeepSide::Existing => {
                        // Drop the candidate; move on to the next input event.
                        continue 'outer;
                    }
                    KeepSide::Candidate => {
                        // Replace the existing entry in-place.
                        *slot = (src_idx, candidate.clone());
                        continue 'outer;
                    }
                }
            }
        }
        // No overlap found; keep the candidate.
        result.push((src_idx, candidate));
    }

    result
}

/// Merge strategy for resolving overlapping events across multiple EDLs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Keep the event with the earlier record-in timecode.
    PreferEarlier,
    /// Keep the event with the longer duration.
    PreferLonger,
    /// Layer all events onto separate tracks (no dropping).
    LayerToTracks,
}

/// A single conformed event with its resolved track layer.
#[derive(Debug, Clone)]
pub struct ConformedEvent {
    /// The underlying EDL event.
    pub event: EdlEvent,
    /// The virtual track layer (0-based; equals source EDL index for
    /// `LayerToTracks`, 0 for single-winner strategies).
    pub track_layer: usize,
}

/// Source attribution for each event in the batch conform result.
#[derive(Debug, Clone)]
pub struct EventProvenance {
    /// Position of the event in `BatchConformResult::events`.
    pub event_index: usize,
    /// Index of the source EDL in the `edls` slice passed to `batch_conform`.
    pub source_edl_index: usize,
}

/// Result of a batch conform operation.
#[derive(Debug, Clone)]
pub struct BatchConformResult {
    /// Merged, sorted, and de-overlapped event list.
    pub events: Vec<ConformedEvent>,
    /// Per-event provenance map.
    pub provenance: Vec<EventProvenance>,
}

/// Result of a conform operation.
#[derive(Debug, Clone)]
pub struct ConformResult {
    /// Output file path.
    pub output_path: std::path::PathBuf,

    /// Number of clips relinked.
    pub clips_relinked: usize,

    /// Number of clips that couldn't be relinked.
    pub clips_failed: usize,

    /// Total duration in seconds.
    pub total_duration: f64,

    /// Frame-accurate conforming was successful.
    pub frame_accurate: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_edl::event::{EditType, TrackType};
    use oximedia_edl::timecode::{EdlFrameRate, EdlTimecode};
    use oximedia_edl::{Edl, EdlFormat};

    // ── helpers ──────────────────────────────────────────────────────────────

    fn fps() -> EdlFrameRate {
        EdlFrameRate::Fps25
    }

    fn tc(seconds: u8) -> EdlTimecode {
        EdlTimecode::new(0, 0, seconds, 0, fps()).expect("valid timecode")
    }

    /// Build a minimal cut event on the video track.
    fn cut(event_num: u32, rec_in_secs: u8, rec_out_secs: u8) -> EdlEvent {
        let src_in = tc(0);
        let src_out = tc(rec_out_secs.saturating_sub(rec_in_secs));
        let rec_in = tc(rec_in_secs);
        let rec_out = tc(rec_out_secs);
        EdlEvent::new(
            event_num,
            "A001".to_string(),
            TrackType::Video,
            EditType::Cut,
            src_in,
            src_out,
            rec_in,
            rec_out,
        )
    }

    fn single_event_edl(rec_in_secs: u8, rec_out_secs: u8, evt_num: u32) -> Edl {
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.events.push(cut(evt_num, rec_in_secs, rec_out_secs));
        edl
    }

    async fn make_engine() -> ConformEngine {
        let db = std::env::temp_dir().join(format!(
            "batch_conform_test_{}.json",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        let engine = ConformEngine::new(&db)
            .await
            .expect("engine creation should succeed");
        let _ = std::fs::remove_file(&db);
        engine
    }

    // ── engine creation ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_conform_engine_creation() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join("test_conform.json");

        let engine = ConformEngine::new(&db_path).await;
        assert!(engine.is_ok());

        // Clean up
        let _ = std::fs::remove_file(db_path);
    }

    // ── batch_conform tests ───────────────────────────────────────────────────

    /// Two EDLs with non-overlapping timecode ranges — result equals the
    /// concatenation of both EDLs' events, sorted by record-in.
    #[tokio::test]
    async fn test_batch_conform_non_overlapping() {
        let engine = make_engine().await;

        // EDL 0: 0–5 s
        let edl0 = single_event_edl(0, 5, 1);
        // EDL 1: 6–10 s (no overlap)
        let edl1 = single_event_edl(6, 10, 2);

        let result = engine.batch_conform(&[edl0, edl1], MergeStrategy::PreferEarlier);

        assert_eq!(result.events.len(), 2);
        // Events should be sorted by record-in
        assert!(
            result.events[0].event.record_in.to_frames()
                <= result.events[1].event.record_in.to_frames()
        );
        // Both EDL sources present
        assert!(result.provenance.iter().any(|p| p.source_edl_index == 0));
        assert!(result.provenance.iter().any(|p| p.source_edl_index == 1));
    }

    /// Overlapping events with PreferEarlier: the event with the smaller
    /// record-in timecode wins.
    #[tokio::test]
    async fn test_batch_conform_overlap_prefer_earlier() {
        let engine = make_engine().await;

        // EDL 0: event at 0–10 s (earlier record-in)
        let edl0 = single_event_edl(0, 10, 1);
        // EDL 1: event at 2–8 s (overlaps, later record-in)
        let edl1 = single_event_edl(2, 8, 2);

        let result = engine.batch_conform(&[edl0, edl1], MergeStrategy::PreferEarlier);

        // Only one event should survive (they overlap)
        assert_eq!(result.events.len(), 1);
        // The surviving event should be from EDL 0 (earlier record-in = 0 s)
        assert_eq!(result.events[0].event.record_in, tc(0));
        assert_eq!(result.provenance[0].source_edl_index, 0);
    }

    /// Overlapping events with PreferLonger: the longer event wins.
    #[tokio::test]
    async fn test_batch_conform_overlap_prefer_longer() {
        let engine = make_engine().await;

        // EDL 0: event at 0–3 s (duration 3 s — shorter)
        let edl0 = single_event_edl(0, 3, 1);
        // EDL 1: event at 1–8 s (duration 7 s — longer, but later record-in)
        let edl1 = single_event_edl(1, 8, 2);

        let result = engine.batch_conform(&[edl0, edl1], MergeStrategy::PreferLonger);

        // Only one event should survive
        assert_eq!(result.events.len(), 1);
        // The surviving event should be from EDL 1 (longer duration)
        assert_eq!(result.events[0].event.record_in, tc(1));
        assert_eq!(result.provenance[0].source_edl_index, 1);
    }

    /// Empty EDL slice must return an empty result without panicking.
    #[tokio::test]
    async fn test_batch_conform_empty() {
        let engine = make_engine().await;
        let result = engine.batch_conform(&[], MergeStrategy::PreferEarlier);
        assert!(result.events.is_empty());
        assert!(result.provenance.is_empty());
    }

    /// Provenance map correctly attributes each event to its source EDL index.
    #[tokio::test]
    async fn test_batch_conform_provenance() {
        let engine = make_engine().await;

        // Three non-overlapping EDLs with one event each
        let edl0 = single_event_edl(0, 2, 1); // source 0
        let edl1 = single_event_edl(3, 5, 2); // source 1
        let edl2 = single_event_edl(6, 9, 3); // source 2

        let result = engine.batch_conform(&[edl0, edl1, edl2], MergeStrategy::LayerToTracks);

        assert_eq!(result.events.len(), 3);
        assert_eq!(result.provenance.len(), 3);

        // The provenance event_index must match the position in events vec.
        for (pos, prov) in result.provenance.iter().enumerate() {
            assert_eq!(prov.event_index, pos);
        }

        // Sources must appear in order of record-in (all were non-overlapping
        // and EDL 0 < EDL 1 < EDL 2).
        assert_eq!(result.provenance[0].source_edl_index, 0);
        assert_eq!(result.provenance[1].source_edl_index, 1);
        assert_eq!(result.provenance[2].source_edl_index, 2);
    }
}

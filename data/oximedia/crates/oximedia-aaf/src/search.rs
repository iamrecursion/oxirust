//! AAF metadata search/query API
//!
//! Provides flexible search over ContentStorage mobs and clips using
//! composable query criteria including name patterns, mob type, and
//! timecode ranges.

use crate::object_model::MobType;
use crate::{AafError, CompositionMob, ContentStorage, Result};
use uuid::Uuid;

/// Reference to a mob found by search
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MobRef {
    /// Mob UUID
    pub mob_id: Uuid,
    /// Mob name
    pub name: String,
    /// Mob type
    pub mob_type: MobTypeKind,
    /// Duration in edit units (if available)
    pub duration: Option<i64>,
}

/// Mob type kind for search results
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MobTypeKind {
    /// Composition mob
    Composition,
    /// Master mob
    Master,
    /// Source mob
    Source,
}

impl From<MobType> for MobTypeKind {
    fn from(t: MobType) -> Self {
        match t {
            MobType::Composition => MobTypeKind::Composition,
            MobType::Master => MobTypeKind::Master,
            MobType::Source => MobTypeKind::Source,
        }
    }
}

/// Query for searching AAF mobs and clips
#[derive(Debug, Clone, Default)]
pub struct AafQuery {
    /// Optional case-sensitive name substring filter
    pub name_contains: Option<String>,
    /// Optional case-insensitive name substring filter
    pub name_contains_ci: Option<String>,
    /// Optional mob type filter
    pub mob_type: Option<MobTypeKind>,
    /// Optional timecode range: (start_inclusive, end_exclusive) in edit units
    pub timecode_range: Option<(i64, i64)>,
    /// Limit the number of results (None = unlimited)
    pub limit: Option<usize>,
}

impl AafQuery {
    /// Create a new empty query (matches everything)
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter by name containing this substring (case-sensitive)
    #[must_use]
    pub fn with_name_contains(mut self, pattern: impl Into<String>) -> Self {
        self.name_contains = Some(pattern.into());
        self
    }

    /// Filter by name containing this substring (case-insensitive)
    #[must_use]
    pub fn with_name_contains_ci(mut self, pattern: impl Into<String>) -> Self {
        self.name_contains_ci = Some(pattern.into());
        self
    }

    /// Filter by mob type
    #[must_use]
    pub fn with_mob_type(mut self, mob_type: MobTypeKind) -> Self {
        self.mob_type = Some(mob_type);
        self
    }

    /// Filter by timecode range in edit units [start, end)
    #[must_use]
    pub fn with_timecode_range(mut self, start: i64, end: i64) -> Self {
        self.timecode_range = Some((start, end));
        self
    }

    /// Limit maximum results returned
    #[must_use]
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Check whether a mob ref matches this query
    fn matches(&self, mob_ref: &MobRef) -> bool {
        // Case-sensitive name filter
        if let Some(ref pattern) = self.name_contains {
            if !mob_ref.name.contains(pattern.as_str()) {
                return false;
            }
        }

        // Case-insensitive name filter
        if let Some(ref pattern) = self.name_contains_ci {
            let name_lower = mob_ref.name.to_lowercase();
            let pat_lower = pattern.to_lowercase();
            if !name_lower.contains(&pat_lower) {
                return false;
            }
        }

        // Mob type filter
        if let Some(ref mob_type) = self.mob_type {
            if mob_ref.mob_type != *mob_type {
                return false;
            }
        }

        // Timecode range filter: mob must have a duration and it must overlap [start, end)
        if let Some((range_start, range_end)) = self.timecode_range {
            match mob_ref.duration {
                Some(dur) => {
                    // Mob spans [0, dur); check overlap with [range_start, range_end)
                    if dur <= range_start || 0 >= range_end {
                        return false;
                    }
                }
                None => {
                    // No duration info — exclude from range queries
                    return false;
                }
            }
        }

        true
    }
}

/// Searcher that executes queries against ContentStorage
pub struct AafSearcher;

impl AafSearcher {
    /// Execute a query against content storage and return matching mob refs
    ///
    /// # Errors
    ///
    /// Returns `AafError::InvalidFile` if a timecode range has start >= end.
    pub fn search(storage: &ContentStorage, query: &AafQuery) -> Result<Vec<MobRef>> {
        // Validate query
        if let Some((start, end)) = query.timecode_range {
            if start >= end {
                return Err(AafError::InvalidFile(format!(
                    "Timecode range start ({start}) must be less than end ({end})"
                )));
            }
        }

        let mut results: Vec<MobRef> = Vec::new();

        // Search composition mobs
        for comp_mob in storage.composition_mobs() {
            let mob_ref = mob_ref_from_composition(comp_mob);
            if query.matches(&mob_ref) {
                results.push(mob_ref);
            }
        }

        // Search master mobs
        for mob in storage.master_mobs() {
            let mob_ref = MobRef {
                mob_id: mob.mob_id(),
                name: mob.name().to_string(),
                mob_type: MobTypeKind::Master,
                duration: None,
            };
            if query.matches(&mob_ref) {
                results.push(mob_ref);
            }
        }

        // Search source mobs
        for mob in storage.source_mobs() {
            let mob_ref = MobRef {
                mob_id: mob.mob_id(),
                name: mob.name().to_string(),
                mob_type: MobTypeKind::Source,
                duration: None,
            };
            if query.matches(&mob_ref) {
                results.push(mob_ref);
            }
        }

        // Apply limit
        if let Some(limit) = query.limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    /// Search only composition mobs by name pattern (case-sensitive)
    ///
    /// # Errors
    ///
    /// Propagates query validation errors.
    pub fn find_compositions_by_name(storage: &ContentStorage, name: &str) -> Result<Vec<MobRef>> {
        let query = AafQuery::new()
            .with_name_contains(name)
            .with_mob_type(MobTypeKind::Composition);
        Self::search(storage, &query)
    }

    /// Search compositions by name pattern (case-insensitive)
    ///
    /// # Errors
    ///
    /// Propagates query validation errors.
    pub fn find_compositions_by_name_ci(
        storage: &ContentStorage,
        name: &str,
    ) -> Result<Vec<MobRef>> {
        let query = AafQuery::new()
            .with_name_contains_ci(name)
            .with_mob_type(MobTypeKind::Composition);
        Self::search(storage, &query)
    }
}

/// Build a MobRef from a CompositionMob
fn mob_ref_from_composition(comp_mob: &CompositionMob) -> MobRef {
    MobRef {
        mob_id: comp_mob.mob_id(),
        name: comp_mob.name().to_string(),
        mob_type: MobTypeKind::Composition,
        duration: comp_mob.duration(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition::{
        CompositionMob, Sequence, SequenceComponent, SourceClip, Track, TrackType,
    };
    use crate::dictionary::Auid;

    use crate::timeline::{EditRate, Position};
    use crate::ContentStorage;
    use uuid::Uuid;

    fn make_storage_with_compositions() -> ContentStorage {
        let mut storage = ContentStorage::new();

        let mut comp1 = CompositionMob::new(Uuid::new_v4(), "Main Edit");
        let mut seq = Sequence::new(Auid::PICTURE);
        seq.add_component(SequenceComponent::SourceClip(SourceClip::new(
            100,
            Position::zero(),
            Uuid::new_v4(),
            1,
        )));
        let mut track = Track::new(1, "Video", EditRate::PAL_25, TrackType::Picture);
        track.set_sequence(seq);
        comp1.add_track(track);
        storage.add_composition_mob(comp1);

        let comp2 = CompositionMob::new(Uuid::new_v4(), "Rough Cut");
        storage.add_composition_mob(comp2);

        let comp3 = CompositionMob::new(Uuid::new_v4(), "main_audio");
        storage.add_composition_mob(comp3);

        storage
    }

    #[test]
    fn test_search_all_compositions() {
        let storage = make_storage_with_compositions();
        let query = AafQuery::new().with_mob_type(MobTypeKind::Composition);
        let results = AafSearcher::search(&storage, &query).expect("search should succeed");
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_search_by_name_case_sensitive() {
        let storage = make_storage_with_compositions();
        let query = AafQuery::new()
            .with_name_contains("Main")
            .with_mob_type(MobTypeKind::Composition);
        let results = AafSearcher::search(&storage, &query).expect("search should succeed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Main Edit");
    }

    #[test]
    fn test_search_by_name_case_insensitive() {
        let storage = make_storage_with_compositions();
        let query = AafQuery::new()
            .with_name_contains_ci("main")
            .with_mob_type(MobTypeKind::Composition);
        let results = AafSearcher::search(&storage, &query).expect("search should succeed");
        // Should match "Main Edit" and "main_audio"
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_no_results() {
        let storage = make_storage_with_compositions();
        let query = AafQuery::new().with_name_contains("DoesNotExist");
        let results = AafSearcher::search(&storage, &query).expect("search should succeed");
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_timecode_range() {
        let storage = make_storage_with_compositions();
        // "Main Edit" has duration 100 → overlaps [0, 200)
        let query = AafQuery::new()
            .with_mob_type(MobTypeKind::Composition)
            .with_timecode_range(0, 200);
        let results = AafSearcher::search(&storage, &query).expect("search should succeed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Main Edit");
    }

    #[test]
    fn test_search_timecode_range_no_overlap() {
        let storage = make_storage_with_compositions();
        // "Main Edit" has duration 100 → does NOT overlap [200, 400)
        let query = AafQuery::new()
            .with_mob_type(MobTypeKind::Composition)
            .with_timecode_range(200, 400);
        let results = AafSearcher::search(&storage, &query).expect("search should succeed");
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_invalid_timecode_range() {
        let storage = make_storage_with_compositions();
        let query = AafQuery::new().with_timecode_range(100, 50);
        let result = AafSearcher::search(&storage, &query);
        assert!(result.is_err());
    }

    #[test]
    fn test_search_with_limit() {
        let storage = make_storage_with_compositions();
        let query = AafQuery::new()
            .with_mob_type(MobTypeKind::Composition)
            .with_limit(1);
        let results = AafSearcher::search(&storage, &query).expect("search should succeed");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_find_compositions_by_name_ci() {
        let storage = make_storage_with_compositions();
        let results =
            AafSearcher::find_compositions_by_name_ci(&storage, "cut").expect("should succeed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Rough Cut");
    }
}

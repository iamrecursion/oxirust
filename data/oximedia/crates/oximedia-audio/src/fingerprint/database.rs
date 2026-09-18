//! Fingerprint database for storage and matching.

use super::hash::Hash;
use super::matching::FingerprintMatcher;
use super::Fingerprint;
use std::collections::HashMap;

/// Fingerprint database for audio identification.
pub struct FingerprintDatabase {
    /// Hash index: hash -> [(track_id, time)]
    hash_index: HashMap<Hash, Vec<(String, f64)>>,
    /// Track metadata
    tracks: HashMap<String, TrackMetadata>,
    /// Matcher for finding matches
    matcher: FingerprintMatcher,
}

impl FingerprintDatabase {
    /// Create a new empty database.
    #[must_use]
    pub fn new() -> Self {
        Self {
            hash_index: HashMap::new(),
            tracks: HashMap::new(),
            matcher: FingerprintMatcher::default(),
        }
    }

    /// Create a database with custom matcher.
    #[must_use]
    pub fn with_matcher(matcher: FingerprintMatcher) -> Self {
        Self {
            hash_index: HashMap::new(),
            tracks: HashMap::new(),
            matcher,
        }
    }

    /// Add a fingerprint to the database.
    pub fn add_fingerprint(&mut self, track_id: impl Into<String>, fingerprint: Fingerprint) {
        let track_id = track_id.into();

        // Store track metadata
        self.tracks.insert(
            track_id.clone(),
            TrackMetadata {
                id: track_id.clone(),
                duration: fingerprint.duration,
                sample_rate: fingerprint.sample_rate,
                hash_count: fingerprint.hashes.len(),
            },
        );

        // Index all hashes
        for (hash, time) in fingerprint.hashes {
            self.hash_index
                .entry(hash)
                .or_default()
                .push((track_id.clone(), time));
        }
    }

    /// Remove a fingerprint from the database.
    pub fn remove_fingerprint(&mut self, track_id: &str) -> bool {
        if self.tracks.remove(track_id).is_none() {
            return false;
        }

        // Remove from hash index
        for entries in self.hash_index.values_mut() {
            entries.retain(|(id, _)| id != track_id);
        }

        // Clean up empty entries
        self.hash_index.retain(|_, entries| !entries.is_empty());

        true
    }

    /// Find matches for a query fingerprint.
    #[must_use]
    pub fn find_matches(&self, query: &Fingerprint, min_confidence: f64) -> Vec<Match> {
        // Build candidate matches by counting hash matches per track
        let mut candidate_scores: HashMap<String, Vec<(Hash, f64, f64)>> = HashMap::new();

        for (query_hash, query_time) in &query.hashes {
            if let Some(entries) = self.hash_index.get(query_hash) {
                for (track_id, ref_time) in entries {
                    candidate_scores.entry(track_id.clone()).or_default().push((
                        *query_hash,
                        *query_time,
                        *ref_time,
                    ));
                }
            }
        }

        // Process each candidate
        let mut matches = Vec::new();

        for (track_id, hash_matches) in candidate_scores {
            if let Some(metadata) = self.tracks.get(&track_id) {
                // Build temporary fingerprint for matching
                let ref_hashes: Vec<(Hash, f64)> = hash_matches
                    .iter()
                    .map(|(h, _, ref_time)| (*h, *ref_time))
                    .collect();

                let ref_fingerprint = Fingerprint::new(
                    ref_hashes,
                    metadata.sample_rate,
                    metadata.duration,
                    query.config.clone(),
                );

                // Match using the matcher
                if let Some(result) = self.matcher.match_fingerprint(query, &ref_fingerprint) {
                    if result.confidence >= min_confidence && self.matcher.verify_match(&result) {
                        matches.push(Match {
                            track_id: track_id.clone(),
                            confidence: result.confidence,
                            time_offset: result.time_offset,
                            match_count: result.match_count,
                            query_coverage: result.query_coverage(),
                            reference_coverage: result.reference_coverage(),
                        });
                    }
                }
            }
        }

        // Sort by confidence
        matches.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        matches
    }

    /// Find best match for a query.
    #[must_use]
    pub fn find_best_match(&self, query: &Fingerprint, min_confidence: f64) -> Option<Match> {
        self.find_matches(query, min_confidence).into_iter().next()
    }

    /// Check if a track exists in the database.
    #[must_use]
    pub fn contains_track(&self, track_id: &str) -> bool {
        self.tracks.contains_key(track_id)
    }

    /// Get track metadata.
    #[must_use]
    pub fn get_track_metadata(&self, track_id: &str) -> Option<&TrackMetadata> {
        self.tracks.get(track_id)
    }

    /// Get all track IDs.
    #[must_use]
    pub fn track_ids(&self) -> Vec<String> {
        self.tracks.keys().cloned().collect()
    }

    /// Get total number of tracks.
    #[must_use]
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Get total number of hashes.
    #[must_use]
    pub fn hash_count(&self) -> usize {
        self.hash_index.len()
    }

    /// Get database statistics.
    #[must_use]
    pub fn statistics(&self) -> DatabaseStatistics {
        let total_entries: usize = self.hash_index.values().map(Vec::len).sum();

        let avg_collisions = if !self.hash_index.is_empty() {
            total_entries as f64 / self.hash_index.len() as f64
        } else {
            0.0
        };

        let total_duration: f64 = self.tracks.values().map(|m| m.duration).sum();

        DatabaseStatistics {
            track_count: self.tracks.len(),
            unique_hash_count: self.hash_index.len(),
            total_hash_entries: total_entries,
            avg_hash_collisions: avg_collisions,
            total_duration,
        }
    }

    /// Clear the database.
    pub fn clear(&mut self) {
        self.hash_index.clear();
        self.tracks.clear();
    }

    /// Merge another database into this one.
    pub fn merge(&mut self, other: Self) {
        // Merge tracks
        self.tracks.extend(other.tracks);

        // Merge hash index
        for (hash, entries) in other.hash_index {
            self.hash_index.entry(hash).or_default().extend(entries);
        }
    }

    /// Optimize database (remove duplicates, compact storage).
    pub fn optimize(&mut self) {
        // Remove duplicate entries in hash index
        for entries in self.hash_index.values_mut() {
            entries.sort_by(|a, b| {
                a.0.cmp(&b.0)
                    .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            });
            entries.dedup();
        }

        // Remove empty entries
        self.hash_index.retain(|_, entries| !entries.is_empty());

        // Update track metadata hash counts
        let mut hash_counts: HashMap<String, usize> = HashMap::new();
        for entries in self.hash_index.values() {
            for (track_id, _) in entries {
                *hash_counts.entry(track_id.clone()).or_insert(0) += 1;
            }
        }

        for (track_id, metadata) in &mut self.tracks {
            if let Some(&count) = hash_counts.get(track_id) {
                metadata.hash_count = count;
            }
        }
    }

    /// Export database to serializable format.
    #[must_use]
    pub fn export(&self) -> DatabaseExport {
        let mut entries = Vec::new();

        for (hash, track_times) in &self.hash_index {
            for (track_id, time) in track_times {
                entries.push(HashEntry {
                    hash: hash.value(),
                    track_id: track_id.clone(),
                    time: *time,
                });
            }
        }

        DatabaseExport {
            tracks: self.tracks.values().cloned().collect(),
            entries,
        }
    }

    /// Import database from serializable format.
    pub fn import(export: DatabaseExport) -> Self {
        let mut db = Self::new();

        // Import tracks
        for metadata in export.tracks {
            db.tracks.insert(metadata.id.clone(), metadata);
        }

        // Import hash entries
        for entry in export.entries {
            let hash = Hash::from(entry.hash);
            db.hash_index
                .entry(hash)
                .or_default()
                .push((entry.track_id, entry.time));
        }

        db
    }

    /// Find duplicate tracks in the database.
    #[must_use]
    pub fn find_duplicates(&self, min_confidence: f64) -> Vec<(String, String, f64)> {
        let mut duplicates = Vec::new();
        let track_ids: Vec<_> = self.tracks.keys().collect();

        for i in 0..track_ids.len() {
            for j in (i + 1)..track_ids.len() {
                let id1 = track_ids[i];
                let id2 = track_ids[j];

                // Get fingerprints (reconstructed from hash index)
                let hashes1 = self.get_track_hashes(id1);
                let hashes2 = self.get_track_hashes(id2);

                if let (Some(meta1), Some(meta2)) = (self.tracks.get(id1), self.tracks.get(id2)) {
                    let fp1 = Fingerprint::new(
                        hashes1,
                        meta1.sample_rate,
                        meta1.duration,
                        Default::default(),
                    );

                    let fp2 = Fingerprint::new(
                        hashes2,
                        meta2.sample_rate,
                        meta2.duration,
                        Default::default(),
                    );

                    if let Some(result) = self.matcher.match_fingerprint(&fp1, &fp2) {
                        if result.confidence >= min_confidence {
                            duplicates.push((id1.clone(), id2.clone(), result.confidence));
                        }
                    }
                }
            }
        }

        duplicates
    }

    /// Get all hashes for a track.
    fn get_track_hashes(&self, track_id: &str) -> Vec<(Hash, f64)> {
        let mut hashes = Vec::new();

        for (hash, entries) in &self.hash_index {
            for (id, time) in entries {
                if id == track_id {
                    hashes.push((*hash, *time));
                }
            }
        }

        hashes.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        hashes
    }
}

impl Default for FingerprintDatabase {
    fn default() -> Self {
        Self::new()
    }
}

/// Track metadata.
#[derive(Clone, Debug)]
pub struct TrackMetadata {
    /// Track identifier.
    pub id: String,
    /// Duration in seconds.
    pub duration: f64,
    /// Sample rate.
    pub sample_rate: u32,
    /// Number of hashes.
    pub hash_count: usize,
}

/// Match result from database query.
#[derive(Clone, Debug)]
pub struct Match {
    /// Matched track ID.
    pub track_id: String,
    /// Confidence score (0-1).
    pub confidence: f64,
    /// Time offset (seconds).
    pub time_offset: f64,
    /// Number of matching hashes.
    pub match_count: usize,
    /// Query coverage (0-1).
    pub query_coverage: f64,
    /// Reference coverage (0-1).
    pub reference_coverage: f64,
}

impl Match {
    /// Check if this is a strong match.
    #[must_use]
    pub fn is_strong_match(&self) -> bool {
        self.confidence >= 0.7 && self.query_coverage >= 0.5
    }

    /// Check if this is likely a duplicate.
    #[must_use]
    pub fn is_likely_duplicate(&self) -> bool {
        self.confidence >= 0.9 && self.query_coverage >= 0.8
    }

    /// Check if this is a partial match.
    #[must_use]
    pub fn is_partial_match(&self) -> bool {
        self.confidence >= 0.3 && self.query_coverage < 0.5
    }
}

/// Database statistics.
#[derive(Clone, Debug, Default)]
pub struct DatabaseStatistics {
    /// Number of tracks in database.
    pub track_count: usize,
    /// Number of unique hashes.
    pub unique_hash_count: usize,
    /// Total hash entries (including collisions).
    pub total_hash_entries: usize,
    /// Average collisions per hash.
    pub avg_hash_collisions: f64,
    /// Total duration of all tracks.
    pub total_duration: f64,
}

impl DatabaseStatistics {
    /// Get database size estimate in bytes.
    #[must_use]
    pub fn estimated_size_bytes(&self) -> usize {
        // Hash: 8 bytes, track_id: ~32 bytes average, time: 8 bytes
        self.total_hash_entries * (8 + 32 + 8)
    }

    /// Get average hash count per track.
    #[must_use]
    pub fn avg_hashes_per_track(&self) -> f64 {
        if self.track_count > 0 {
            self.total_hash_entries as f64 / self.track_count as f64
        } else {
            0.0
        }
    }
}

/// Serializable database export format.
#[derive(Clone, Debug)]
pub struct DatabaseExport {
    /// Track metadata.
    pub tracks: Vec<TrackMetadata>,
    /// Hash entries.
    pub entries: Vec<HashEntry>,
}

/// Single hash entry for export.
#[derive(Clone, Debug)]
pub struct HashEntry {
    /// Hash value.
    pub hash: u64,
    /// Track identifier.
    pub track_id: String,
    /// Time position.
    pub time: f64,
}

/// In-memory cache for frequently queried fingerprints.
pub struct FingerprintCache {
    cache: HashMap<String, Fingerprint>,
    max_size: usize,
}

impl FingerprintCache {
    /// Create a new cache with maximum size.
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
        }
    }

    /// Get fingerprint from cache.
    #[must_use]
    pub fn get(&self, track_id: &str) -> Option<&Fingerprint> {
        self.cache.get(track_id)
    }

    /// Add fingerprint to cache.
    pub fn insert(&mut self, track_id: String, fingerprint: Fingerprint) {
        if self.cache.len() >= self.max_size {
            // Simple eviction: remove first entry
            if let Some(key) = self.cache.keys().next().cloned() {
                self.cache.remove(&key);
            }
        }

        self.cache.insert(track_id, fingerprint);
    }

    /// Clear cache.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Get cache size.
    #[must_use]
    pub fn size(&self) -> usize {
        self.cache.len()
    }
}

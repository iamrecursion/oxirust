//! Playlist shuffling: Fisher-Yates shuffle, weighted random selection,
//! and smart shuffle that avoids recently played repeats.
//!
//! # Weighted Shuffle
//!
//! The [`WeightedPlaylistShuffler`] extends the basic weighted shuffle with
//! configurable priority dimensions designed for broadcast/DJ playlists:
//!
//! - **Base priority**: explicit `0.0–10.0` priority set by the operator.
//! - **Genre affinity**: boost tracks whose genre matches a target genre.
//! - **Energy curve**: bias the shuffle toward high- or low-energy tracks
//!   (e.g., ramp up for a peak-hour set, or cool down at the end of a show).
//! - **Recency penalty**: automatically reduce the weight of recently played
//!   tracks so the same song does not repeat too soon (no global state required).
//! - **Request boost**: temporarily double the weight of user-requested tracks.
//!
//! Weights are combined multiplicatively so that any dimension can be tuned
//! independently without clamping artefacts.

#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};

// ── Fisher-Yates in-place shuffle ────────────────────────────────────────────

/// Shuffle a mutable slice in place using the Fisher-Yates algorithm.
/// Uses a deterministic seed for reproducibility in tests via a simple LCG.
pub fn fisher_yates<T>(items: &mut [T], seed: u64) {
    let n = items.len();
    if n < 2 {
        return;
    }
    let mut rng = LcgRng::new(seed);
    for i in (1..n).rev() {
        let j = rng.next_u64() as usize % (i + 1);
        items.swap(i, j);
    }
}

/// Minimal linear-congruential RNG for deterministic shuffling (no external deps).
#[derive(Debug, Clone)]
struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        // Knuth multiplicative LCG.
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }
}

// ── WeightedShuffler ─────────────────────────────────────────────────────────

/// An item with an associated weight for weighted random selection.
#[derive(Debug, Clone)]
pub struct WeightedItem<T> {
    /// The item value.
    pub item: T,
    /// Relative weight (must be > 0.0).
    pub weight: f64,
}

impl<T> WeightedItem<T> {
    /// Create a new weighted item.
    #[must_use]
    pub fn new(item: T, weight: f64) -> Self {
        Self {
            item,
            weight: weight.max(f64::EPSILON),
        }
    }
}

/// Produce a weighted shuffle of items (higher weight → more likely to appear early).
/// Uses a reservoir-sampling-inspired technique: sort by `u^(1/w)` with a LCG key.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn weighted_shuffle<T: Clone>(items: &[WeightedItem<T>], seed: u64) -> Vec<T> {
    if items.is_empty() {
        return Vec::new();
    }
    let mut rng = LcgRng::new(seed);
    let mut keyed: Vec<(f64, usize)> = items
        .iter()
        .enumerate()
        .map(|(i, wi)| {
            // u in (0, 1).
            let u = (rng.next_u64() as f64 / u64::MAX as f64).clamp(1e-10, 1.0 - 1e-10);
            let key = u.ln() / wi.weight;
            (key, i)
        })
        .collect();
    // Sort descending by key (highest key = appears first).
    keyed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    keyed
        .into_iter()
        .map(|(_, i)| items[i].item.clone())
        .collect()
}

// ── SmartShuffler ─────────────────────────────────────────────────────────────

/// A stateful shuffler that avoids replaying recently played tracks.
///
/// Internally maintains a queue of all track IDs in shuffled order, and a
/// "recent" window that prevents the same track from appearing until at least
/// `min_gap` other tracks have played.
#[derive(Debug)]
pub struct SmartShuffler {
    /// All available track IDs.
    track_ids: Vec<u64>,
    /// Current shuffle queue.
    queue: VecDeque<u64>,
    /// Recently played track IDs (oldest first).
    recent: VecDeque<u64>,
    /// Minimum number of distinct tracks between repeats.
    min_gap: usize,
    /// Seed for reproducibility.
    seed: u64,
    /// Counter incremented each re-shuffle to change the seed.
    generation: u64,
}

impl SmartShuffler {
    /// Create a new smart shuffler.
    ///
    /// * `track_ids` – all available tracks.
    /// * `min_gap`   – minimum distance between repeats (capped at track count − 1).
    /// * `seed`      – initial RNG seed.
    #[must_use]
    pub fn new(track_ids: Vec<u64>, min_gap: usize, seed: u64) -> Self {
        let cap = if track_ids.len() > 1 {
            track_ids.len() - 1
        } else {
            0
        };
        let min_gap = min_gap.min(cap);
        let mut s = Self {
            track_ids,
            queue: VecDeque::new(),
            recent: VecDeque::new(),
            min_gap,
            seed,
            generation: 0,
        };
        s.refill();
        s
    }

    /// Returns the next track ID, re-shuffling if the queue is exhausted.
    #[must_use]
    pub fn next(&mut self) -> Option<u64> {
        if self.track_ids.is_empty() {
            return None;
        }
        if self.queue.is_empty() {
            self.refill();
        }
        // Skip tracks in the recent window.
        loop {
            let candidate = *self.queue.front()?;
            if !self.recent.contains(&candidate) || self.recent.len() < self.min_gap {
                self.queue.pop_front();
                self.record_played(candidate);
                return Some(candidate);
            }
            // Move the blocked candidate to the back.
            self.queue.pop_front();
            self.queue.push_back(candidate);
        }
    }

    /// Record a track as played (updates the recent window).
    fn record_played(&mut self, id: u64) {
        self.recent.push_back(id);
        if self.recent.len() > self.min_gap {
            self.recent.pop_front();
        }
    }

    /// Refill the queue with a fresh shuffle of all tracks.
    fn refill(&mut self) {
        let mut ids = self.track_ids.clone();
        fisher_yates(&mut ids, self.seed.wrapping_add(self.generation));
        self.generation += 1;
        self.queue = ids.into();
    }

    /// Number of tracks remaining in the current shuffle pass.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.queue.len()
    }

    /// Total number of available tracks.
    #[must_use]
    pub fn track_count(&self) -> usize {
        self.track_ids.len()
    }
}

// ── Playlist-level weighted shuffle ─────────────────────────────────────────

/// Genre identifier.  String-keyed so callers can use their own taxonomy.
pub type Genre = String;

/// A playlist track descriptor for weighted shuffling.
///
/// Each field influences a distinct weighting dimension.  The final weight
/// applied to the reservoir-sampling algorithm is the product of all active
/// dimension weights, so individual dimensions compose cleanly.
#[derive(Debug, Clone)]
pub struct PlaylistTrack {
    /// Opaque track identifier (path, UUID, or database key).
    pub id: String,
    /// Operator-assigned priority in `[0.0, 10.0]`.  5.0 = neutral.
    pub priority: f64,
    /// Genre tags for this track (may be empty).
    pub genres: Vec<Genre>,
    /// Relative energy level in `[0.0, 1.0]`.  Used by the energy curve.
    pub energy: f64,
    /// Whether this track has been explicitly requested by a listener.
    pub requested: bool,
}

impl PlaylistTrack {
    /// Creates a track with neutral priority and no genre.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            priority: 5.0,
            genres: Vec::new(),
            energy: 0.5,
            requested: false,
        }
    }

    /// Sets the operator priority.
    #[must_use]
    pub fn with_priority(mut self, p: f64) -> Self {
        self.priority = p.clamp(0.0, 10.0);
        self
    }

    /// Adds a genre tag.
    #[must_use]
    pub fn with_genre(mut self, g: impl Into<Genre>) -> Self {
        self.genres.push(g.into());
        self
    }

    /// Sets the energy level.
    #[must_use]
    pub fn with_energy(mut self, e: f64) -> Self {
        self.energy = e.clamp(0.0, 1.0);
        self
    }

    /// Marks this track as listener-requested.
    #[must_use]
    pub fn requested(mut self) -> Self {
        self.requested = true;
        self
    }
}

/// Energy curve shape for biasing the shuffle toward high- or low-energy tracks.
#[derive(Debug, Clone, Copy)]
pub enum EnergyCurve {
    /// No energy bias — all tracks treated equally.
    Flat,
    /// Prefer high-energy tracks (ascending weight with energy).
    RampUp,
    /// Prefer low-energy tracks (descending weight with energy).
    RampDown,
    /// Prefer mid-energy tracks (peak at 0.5).
    Bell,
}

impl EnergyCurve {
    /// Compute the energy weight multiplier for a track with given `energy ∈ [0, 1]`.
    #[must_use]
    fn weight(self, energy: f64) -> f64 {
        let e = energy.clamp(0.0, 1.0);
        match self {
            Self::Flat => 1.0,
            Self::RampUp => 0.1 + 1.9 * e,           // 0.1 … 2.0
            Self::RampDown => 0.1 + 1.9 * (1.0 - e), // 0.1 … 2.0
            Self::Bell => {
                // Gaussian-like bell centred at 0.5: weight ∈ [~0.1, 1.0].
                let x = (e - 0.5) * 4.0; // scale to ~[-2, 2]
                (-x * x / 2.0).exp().max(0.1)
            }
        }
    }
}

/// Configuration for [`WeightedPlaylistShuffler`].
#[derive(Debug, Clone)]
pub struct ShuffleConfig {
    /// Genre to boost (e.g., current show genre).  `None` = no genre bias.
    pub target_genre: Option<Genre>,
    /// Multiplier applied to tracks matching `target_genre`. Default: `2.0`.
    pub genre_boost: f64,
    /// Energy curve bias.
    pub energy_curve: EnergyCurve,
    /// Multiplier applied to listener-requested tracks.  Default: `3.0`.
    pub request_boost: f64,
    /// Penalty applied to recently played tracks (fraction of normal weight).
    /// A value of `0.1` means recently played tracks are 10× less likely.
    pub recency_penalty: f64,
    /// Number of most-recently-played tracks to penalise.
    pub recency_window: usize,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for ShuffleConfig {
    fn default() -> Self {
        Self {
            target_genre: None,
            genre_boost: 2.0,
            energy_curve: EnergyCurve::Flat,
            request_boost: 3.0,
            recency_penalty: 0.1,
            recency_window: 5,
            seed: 0xC0FFEE_B0FFEE,
        }
    }
}

/// A stateful playlist shuffler that combines multiple weighting dimensions.
///
/// Call [`WeightedPlaylistShuffler::next_order`] to get a freshly weighted
/// shuffle of all registered tracks, or [`WeightedPlaylistShuffler::next`] to
/// consume tracks one at a time with automatic recency tracking.
#[derive(Debug)]
pub struct WeightedPlaylistShuffler {
    tracks: Vec<PlaylistTrack>,
    config: ShuffleConfig,
    /// Ring buffer of recently played track IDs for the recency penalty.
    recent: VecDeque<String>,
    /// Per-session play counts used to decay recency.
    play_counts: HashMap<String, u32>,
    /// Internal generation counter to evolve the RNG seed across calls.
    generation: u64,
}

impl WeightedPlaylistShuffler {
    /// Create a new shuffler with the given tracks and configuration.
    #[must_use]
    pub fn new(tracks: Vec<PlaylistTrack>, config: ShuffleConfig) -> Self {
        Self {
            tracks,
            config,
            recent: VecDeque::new(),
            play_counts: HashMap::new(),
            generation: 0,
        }
    }

    /// Compute the composite weight for a single track.
    fn track_weight(&self, track: &PlaylistTrack) -> f64 {
        // 1. Base priority: normalise [0, 10] → [0.1, 2.0].
        let priority_w = 0.1 + (track.priority / 10.0) * 1.9;

        // 2. Genre affinity.
        let genre_w = if let Some(target) = &self.config.target_genre {
            if track.genres.iter().any(|g| g == target) {
                self.config.genre_boost
            } else {
                1.0
            }
        } else {
            1.0
        };

        // 3. Energy curve.
        let energy_w = self.config.energy_curve.weight(track.energy);

        // 4. Request boost.
        let request_w = if track.requested {
            self.config.request_boost
        } else {
            1.0
        };

        // 5. Recency penalty.
        let recency_w = if self.recent.contains(&track.id) {
            self.config.recency_penalty.max(f64::EPSILON)
        } else {
            1.0
        };

        // Composite: multiplicative combination.
        (priority_w * genre_w * energy_w * request_w * recency_w).max(f64::EPSILON)
    }

    /// Produce a full weighted shuffle of all registered tracks.
    ///
    /// Returns a vector of track IDs ordered from most-preferred to least-preferred
    /// given the current weights and RNG state.
    #[must_use]
    pub fn next_order(&mut self) -> Vec<String> {
        if self.tracks.is_empty() {
            return Vec::new();
        }
        let seed = self
            .config
            .seed
            .wrapping_add(self.generation.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        self.generation = self.generation.wrapping_add(1);

        let weighted: Vec<WeightedItem<String>> = self
            .tracks
            .iter()
            .map(|t| WeightedItem::new(t.id.clone(), self.track_weight(t)))
            .collect();

        weighted_shuffle(&weighted, seed)
    }

    /// Return the next single track ID, updating the recency window.
    ///
    /// Generates a new full shuffle order and picks the first entry that is
    /// not already in the recency window (if possible).  Falls back to the
    /// first item in the order if all items are in the recency window.
    pub fn next(&mut self) -> Option<String> {
        if self.tracks.is_empty() {
            return None;
        }
        let order = self.next_order();
        // Find the best candidate not in the recency window.
        let chosen = order
            .iter()
            .find(|id| !self.recent.contains(*id))
            .or_else(|| order.first())
            .cloned()?;

        // Update recency tracking.
        self.recent.push_back(chosen.clone());
        if self.recent.len() > self.config.recency_window {
            self.recent.pop_front();
        }
        *self.play_counts.entry(chosen.clone()).or_insert(0) += 1;

        Some(chosen)
    }

    /// Returns the number of times a track has been played this session.
    #[must_use]
    pub fn play_count(&self, id: &str) -> u32 {
        self.play_counts.get(id).copied().unwrap_or(0)
    }

    /// Returns the total number of registered tracks.
    #[must_use]
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Update the target genre at runtime (e.g., for a show-hour transition).
    pub fn set_target_genre(&mut self, genre: Option<Genre>) {
        self.config.target_genre = genre;
    }

    /// Update the energy curve at runtime.
    pub fn set_energy_curve(&mut self, curve: EnergyCurve) {
        self.config.energy_curve = curve;
    }

    /// Mark a track as having been listener-requested.
    pub fn mark_requested(&mut self, id: &str) {
        if let Some(t) = self.tracks.iter_mut().find(|t| t.id == id) {
            t.requested = true;
        }
    }

    /// Clear all request flags (call after the requests have been honoured).
    pub fn clear_requests(&mut self) {
        for t in &mut self.tracks {
            t.requested = false;
        }
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fisher_yates_same_elements() {
        let mut items = vec![1, 2, 3, 4, 5];
        let original = items.clone();
        fisher_yates(&mut items, 42);
        let mut sorted = items.clone();
        sorted.sort();
        assert_eq!(sorted, original);
    }

    #[test]
    fn test_fisher_yates_deterministic() {
        let mut a = vec![1, 2, 3, 4, 5];
        let mut b = a.clone();
        fisher_yates(&mut a, 99);
        fisher_yates(&mut b, 99);
        assert_eq!(a, b);
    }

    #[test]
    fn test_fisher_yates_different_seeds() {
        let mut a = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let mut b = a.clone();
        fisher_yates(&mut a, 1);
        fisher_yates(&mut b, 2);
        // Different seeds should (almost always) produce different orders.
        // With 8 elements the probability of collision is 1/8! ≈ 0.002%.
        assert_ne!(a, b);
    }

    #[test]
    fn test_fisher_yates_empty() {
        let mut items: Vec<i32> = Vec::new();
        fisher_yates(&mut items, 1);
        assert!(items.is_empty());
    }

    #[test]
    fn test_fisher_yates_single() {
        let mut items = vec![42];
        fisher_yates(&mut items, 1);
        assert_eq!(items, vec![42]);
    }

    #[test]
    fn test_weighted_shuffle_preserves_elements() {
        let items = vec![
            WeightedItem::new(1u32, 1.0),
            WeightedItem::new(2u32, 2.0),
            WeightedItem::new(3u32, 0.5),
        ];
        let mut result = weighted_shuffle(&items, 7);
        result.sort();
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_weighted_shuffle_empty() {
        let items: Vec<WeightedItem<u32>> = Vec::new();
        assert!(weighted_shuffle(&items, 1).is_empty());
    }

    #[test]
    fn test_weighted_shuffle_deterministic() {
        let items = vec![
            WeightedItem::new("a", 1.0),
            WeightedItem::new("b", 10.0),
            WeightedItem::new("c", 0.1),
        ];
        let r1 = weighted_shuffle(&items, 42);
        let r2 = weighted_shuffle(&items, 42);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_weighted_shuffle_single() {
        let items = vec![WeightedItem::new(99u32, 5.0)];
        let result = weighted_shuffle(&items, 1);
        assert_eq!(result, vec![99]);
    }

    #[test]
    fn test_smart_shuffler_basic() {
        let ids: Vec<u64> = (1..=5).collect();
        let mut shuffler = SmartShuffler::new(ids.clone(), 2, 42);
        let mut seen = Vec::new();
        for _ in 0..5 {
            seen.push(shuffler.next().expect("should succeed in test"));
        }
        let mut sorted = seen.clone();
        sorted.sort();
        assert_eq!(sorted, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_smart_shuffler_empty() {
        let mut shuffler = SmartShuffler::new(vec![], 1, 1);
        assert!(shuffler.next().is_none());
    }

    #[test]
    fn test_smart_shuffler_single_track() {
        let mut shuffler = SmartShuffler::new(vec![42u64], 0, 1);
        assert_eq!(shuffler.next(), Some(42));
        assert_eq!(shuffler.next(), Some(42));
    }

    #[test]
    fn test_smart_shuffler_track_count() {
        let ids: Vec<u64> = (0..10).collect();
        let shuffler = SmartShuffler::new(ids, 3, 7);
        assert_eq!(shuffler.track_count(), 10);
    }

    #[test]
    fn test_smart_shuffler_continues_past_first_pass() {
        let ids: Vec<u64> = (1..=4).collect();
        let mut shuffler = SmartShuffler::new(ids, 1, 13);
        for _ in 0..8 {
            assert!(shuffler.next().is_some());
        }
    }

    #[test]
    fn test_lcg_rng_produces_different_values() {
        let mut rng = LcgRng::new(1);
        let a = rng.next_u64();
        let b = rng.next_u64();
        assert_ne!(a, b);
    }

    // ── WeightedPlaylistShuffler tests ───────────────────────────────────────

    fn make_tracks(n: usize) -> Vec<PlaylistTrack> {
        (0..n)
            .map(|i| PlaylistTrack::new(format!("track_{i}")).with_priority(5.0))
            .collect()
    }

    #[test]
    fn test_weighted_playlist_shuffler_produces_all_tracks() {
        let tracks = make_tracks(8);
        let mut shuffler = WeightedPlaylistShuffler::new(tracks, ShuffleConfig::default());
        let order = shuffler.next_order();
        assert_eq!(order.len(), 8);
        let mut sorted = order.clone();
        sorted.sort();
        let mut expected: Vec<String> = (0..8).map(|i| format!("track_{i}")).collect();
        expected.sort();
        assert_eq!(sorted, expected);
    }

    #[test]
    fn test_weighted_playlist_shuffler_empty() {
        let mut shuffler = WeightedPlaylistShuffler::new(vec![], ShuffleConfig::default());
        assert!(shuffler.next_order().is_empty());
        assert!(shuffler.next().is_none());
    }

    #[test]
    fn test_weighted_playlist_shuffler_single_track() {
        let tracks = vec![PlaylistTrack::new("only")];
        let mut shuffler = WeightedPlaylistShuffler::new(tracks, ShuffleConfig::default());
        assert_eq!(shuffler.next(), Some("only".to_string()));
        assert_eq!(shuffler.next(), Some("only".to_string()));
    }

    #[test]
    fn test_weighted_playlist_shuffler_play_count() {
        let tracks = make_tracks(5);
        let mut shuffler = WeightedPlaylistShuffler::new(tracks, ShuffleConfig::default());
        let id = shuffler.next().expect("should have a track");
        assert_eq!(shuffler.play_count(&id), 1);
    }

    #[test]
    fn test_weighted_playlist_shuffler_track_count() {
        let tracks = make_tracks(10);
        let shuffler = WeightedPlaylistShuffler::new(tracks, ShuffleConfig::default());
        assert_eq!(shuffler.track_count(), 10);
    }

    #[test]
    fn test_weighted_playlist_shuffler_genre_boost_orders_genre_first() {
        // Build tracks where one has the target genre.
        let mut tracks: Vec<PlaylistTrack> = (0..10)
            .map(|i| PlaylistTrack::new(format!("t{i}")).with_priority(5.0))
            .collect();
        tracks[5] = PlaylistTrack::new("t5")
            .with_priority(5.0)
            .with_genre("jazz");

        let config = ShuffleConfig {
            target_genre: Some("jazz".into()),
            genre_boost: 100.0, // extreme boost to ensure first place
            seed: 12345,
            ..Default::default()
        };
        let mut shuffler = WeightedPlaylistShuffler::new(tracks, config);
        let order = shuffler.next_order();
        // The jazz track should almost certainly be first with a 100× boost.
        assert_eq!(
            order[0],
            "t5",
            "Expected jazz track first, got {:?}",
            &order[..3]
        );
    }

    #[test]
    fn test_weighted_playlist_shuffler_request_boost() {
        let mut tracks: Vec<PlaylistTrack> = (0..10)
            .map(|i| PlaylistTrack::new(format!("t{i}")).with_priority(5.0))
            .collect();
        tracks[3] = PlaylistTrack::new("t3").with_priority(5.0).requested();

        let config = ShuffleConfig {
            request_boost: 1000.0, // extreme to guarantee first place
            seed: 9999,
            ..Default::default()
        };
        let mut shuffler = WeightedPlaylistShuffler::new(tracks, config);
        let order = shuffler.next_order();
        assert_eq!(order[0], "t3");
    }

    #[test]
    fn test_weighted_playlist_shuffler_mark_requested() {
        let tracks = make_tracks(5);
        let config = ShuffleConfig {
            request_boost: 1000.0,
            seed: 7777,
            ..Default::default()
        };
        let mut shuffler = WeightedPlaylistShuffler::new(tracks, config);
        shuffler.mark_requested("track_2");
        let order = shuffler.next_order();
        assert_eq!(order[0], "track_2");
        shuffler.clear_requests();
        // After clearing, the request flag should be gone.
        // (We just check it doesn't panic and still returns all tracks.)
        let order2 = shuffler.next_order();
        assert_eq!(order2.len(), 5);
    }

    #[test]
    fn test_weighted_playlist_shuffler_recency_avoidance() {
        let tracks = make_tracks(6);
        let config = ShuffleConfig {
            recency_window: 3,
            recency_penalty: 0.0001, // nearly zero weight for recent tracks
            seed: 42,
            ..Default::default()
        };
        let mut shuffler = WeightedPlaylistShuffler::new(tracks, config);
        let first = shuffler.next().expect("first");
        let second = shuffler.next().expect("second");
        let third = shuffler.next().expect("third");
        let fourth = shuffler.next().expect("fourth");
        // The fourth track should not equal the first three (recency window = 3).
        assert_ne!(fourth, first);
        assert_ne!(fourth, second);
        assert_ne!(fourth, third);
    }

    #[test]
    fn test_energy_curve_flat_weight() {
        assert!((EnergyCurve::Flat.weight(0.0) - 1.0).abs() < 1e-9);
        assert!((EnergyCurve::Flat.weight(1.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_energy_curve_ramp_up_prefers_high_energy() {
        assert!(EnergyCurve::RampUp.weight(1.0) > EnergyCurve::RampUp.weight(0.0));
    }

    #[test]
    fn test_energy_curve_ramp_down_prefers_low_energy() {
        assert!(EnergyCurve::RampDown.weight(0.0) > EnergyCurve::RampDown.weight(1.0));
    }

    #[test]
    fn test_energy_curve_bell_peaks_at_midpoint() {
        let mid = EnergyCurve::Bell.weight(0.5);
        let low = EnergyCurve::Bell.weight(0.0);
        let high = EnergyCurve::Bell.weight(1.0);
        assert!(mid > low, "Bell should peak at 0.5 vs 0.0");
        assert!(mid > high, "Bell should peak at 0.5 vs 1.0");
    }

    #[test]
    fn test_set_target_genre_updates_config() {
        let tracks = make_tracks(3);
        let mut shuffler = WeightedPlaylistShuffler::new(tracks, ShuffleConfig::default());
        shuffler.set_target_genre(Some("pop".into()));
        assert_eq!(shuffler.config.target_genre.as_deref(), Some("pop"));
        shuffler.set_target_genre(None);
        assert!(shuffler.config.target_genre.is_none());
    }

    #[test]
    fn test_set_energy_curve_updates_config() {
        let tracks = make_tracks(3);
        let mut shuffler = WeightedPlaylistShuffler::new(tracks, ShuffleConfig::default());
        shuffler.set_energy_curve(EnergyCurve::RampUp);
        assert!(matches!(shuffler.config.energy_curve, EnergyCurve::RampUp));
    }
}

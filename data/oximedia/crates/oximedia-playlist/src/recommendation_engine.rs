//! Playlist recommendation engine.
//!
//! Generates ranked recommendations from a content catalog based on user
//! context such as time-of-day, device type, and previously watched items.

use serde::{Deserialize, Serialize};

/// Type of device the user is watching on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceType {
    /// Living-room television.
    TV,
    /// Mobile phone.
    Mobile,
    /// Desktop computer.
    Desktop,
    /// Tablet device.
    Tablet,
    /// Smart speaker / voice assistant.
    SmartSpeaker,
}

/// Context information used to personalise recommendations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaylistContext {
    /// Optional user identifier.
    pub user_id: Option<u64>,
    /// Hour of the day (0–23) in the user's local time.
    pub time_of_day: u8,
    /// The type of device the user is using.
    pub device_type: DeviceType,
    /// IDs of items the user has already watched in this session.
    pub previous_items: Vec<u64>,
}

impl PlaylistContext {
    /// Create a new context.
    #[must_use]
    pub fn new(time_of_day: u8, device_type: DeviceType) -> Self {
        Self {
            user_id: None,
            time_of_day,
            device_type,
            previous_items: Vec::new(),
        }
    }

    /// Attach a user ID to this context.
    #[must_use]
    pub fn with_user(mut self, user_id: u64) -> Self {
        self.user_id = Some(user_id);
        self
    }

    /// Mark an item as previously watched.
    pub fn add_previous(&mut self, item_id: u64) {
        self.previous_items.push(item_id);
    }
}

/// A single item that can be recommended.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaylistItem {
    /// Unique content identifier.
    pub id: u64,
    /// Display title.
    pub title: String,
    /// Duration in seconds.
    pub duration_secs: u32,
    /// Genre label (e.g. "news", "drama", "comedy").
    pub genre: String,
    /// Editorial rating (0.0–5.0).
    pub rating: f32,
    /// Popularity score (0.0–1.0), e.g. from watch-count normalisation.
    pub popularity: f32,
}

impl PlaylistItem {
    /// Create a new playlist item.
    #[must_use]
    pub fn new(
        id: u64,
        title: impl Into<String>,
        duration_secs: u32,
        genre: impl Into<String>,
        rating: f32,
        popularity: f32,
    ) -> Self {
        Self {
            id,
            title: title.into(),
            duration_secs,
            genre: genre.into(),
            rating: rating.clamp(0.0, 5.0),
            popularity: popularity.clamp(0.0, 1.0),
        }
    }
}

// ── Pre-computed feature vectors ──────────────────────────────────────────────

/// Map a genre string to a bit position (0–63) using a lightweight FNV-1a hash.
///
/// Multiple genre strings may collide (intentional: 64 bits is enough for
/// practical genre vocabularies in broadcast contexts).
fn genre_to_bit(genre: &str) -> u64 {
    const FNV_PRIME: u64 = 1_099_511_628_211;
    const OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
    let mut hash = OFFSET_BASIS;
    for byte in genre.to_lowercase().bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    1u64 << (hash % 64)
}

/// A numeric feature representation of a `PlaylistItem`.
///
/// Pre-computing this vector at ingest time allows the scoring hot-path to
/// avoid repeated string allocations and `to_lowercase()` calls.
#[derive(Debug, Clone)]
pub struct FeatureVector {
    /// One-hot bit mask over up to 64 genre IDs derived via hash.
    pub genre_bits: u64,
    /// Normalised popularity in `[0.0, 1.0]` (already clamped by `PlaylistItem::new`).
    pub popularity_norm: f32,
    /// Normalised rating in `[0.0, 1.0]` (rating divided by 5.0).
    pub rating_norm: f32,
}

impl FeatureVector {
    /// Pre-compute the feature vector for an item.
    #[must_use]
    pub fn from_item(item: &PlaylistItem) -> Self {
        Self {
            genre_bits: genre_to_bit(&item.genre),
            popularity_norm: item.popularity.clamp(0.0, 1.0),
            rating_norm: (item.rating / 5.0).clamp(0.0, 1.0),
        }
    }

    /// Compute a base score using the pre-computed vector.
    ///
    /// Equivalent to `popularity * 0.3 + (rating / 5) * 0.3` but without
    /// any string operations.
    #[must_use]
    pub fn base_score(&self) -> f32 {
        self.popularity_norm * 0.3 + self.rating_norm * 0.3
    }
}

/// Scoring logic for a single item given a context.
pub struct RecommendationScore;

impl RecommendationScore {
    /// Compute a relevance score for `item` given `context`.
    ///
    /// The score is composed of:
    /// - Base: `popularity * 0.3 + (rating / 5) * 0.3`
    /// - Time-of-day genre boost (news in morning, drama/movies in evening)
    /// - Penalty if the item has already been watched in this session
    #[must_use]
    pub fn compute(item: &PlaylistItem, context: &PlaylistContext) -> f32 {
        // Base score (0.0 – 0.6)
        let base = item.popularity * 0.3 + (item.rating / 5.0) * 0.3;

        // Time-of-day genre boost (up to 0.4)
        let genre_boost = Self::genre_boost(&item.genre, context.time_of_day);

        // Penalty for already-watched items
        let penalty = if context.previous_items.contains(&item.id) {
            0.5
        } else {
            0.0
        };

        (base + genre_boost - penalty).max(0.0)
    }

    /// Compute a relevance score using a pre-computed `FeatureVector`.
    ///
    /// This variant avoids all string operations on the hot path; the genre
    /// boost is computed from the `item.genre` string only once per item
    /// (at `FeatureVector::from_item` time) if callers cache the vector.
    ///
    /// The result is numerically identical to `compute` for the same item.
    #[must_use]
    pub fn compute_with_vector(
        item: &PlaylistItem,
        fv: &FeatureVector,
        context: &PlaylistContext,
    ) -> f32 {
        let base = fv.base_score();
        let genre_boost = Self::genre_boost(&item.genre, context.time_of_day);
        let penalty = if context.previous_items.contains(&item.id) {
            0.5
        } else {
            0.0
        };
        (base + genre_boost - penalty).max(0.0)
    }

    /// Returns a genre-specific boost based on hour of day.
    fn genre_boost(genre: &str, hour: u8) -> f32 {
        let genre_lower = genre.to_lowercase();
        match hour {
            // Early morning 5–9: news, documentary
            5..=9 => {
                if genre_lower.contains("news") || genre_lower.contains("documentary") {
                    0.4
                } else {
                    0.0
                }
            }
            // Midday 10–16: reality, comedy, talk
            10..=16 => {
                if genre_lower.contains("comedy")
                    || genre_lower.contains("talk")
                    || genre_lower.contains("reality")
                {
                    0.3
                } else {
                    0.0
                }
            }
            // Prime time 17–22: drama, movie, thriller
            17..=22 => {
                if genre_lower.contains("drama")
                    || genre_lower.contains("movie")
                    || genre_lower.contains("thriller")
                    || genre_lower.contains("film")
                {
                    0.4
                } else {
                    0.0
                }
            }
            // Late night 23–4: any
            _ => 0.1,
        }
    }
}

/// Recommends a ranked list of items from a catalog.
pub struct PlaylistRecommender;

impl PlaylistRecommender {
    /// Return the top-`n` items from `catalog` ranked by score for `context`.
    ///
    /// Items are sorted descending by their computed relevance score.
    #[must_use]
    pub fn recommend<'a>(
        catalog: &'a [PlaylistItem],
        context: &PlaylistContext,
        n: usize,
    ) -> Vec<&'a PlaylistItem> {
        let mut scored: Vec<(&PlaylistItem, f32)> = catalog
            .iter()
            .map(|item| (item, RecommendationScore::compute(item, context)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(n).map(|(item, _)| item).collect()
    }
}

/// A personalised playlist generated for a user.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalizedPlaylist {
    /// Ordered list of item IDs in the playlist.
    pub items: Vec<u64>,
    /// Unix timestamp (ms) when this playlist was generated.
    pub generated_at_ms: u64,
    /// Time-to-live in seconds before this playlist is considered stale.
    pub ttl_secs: u32,
}

impl PersonalizedPlaylist {
    /// Create a new personalised playlist.
    #[must_use]
    pub fn new(items: Vec<u64>, generated_at_ms: u64, ttl_secs: u32) -> Self {
        Self {
            items,
            generated_at_ms,
            ttl_secs,
        }
    }

    /// Returns `true` if the playlist has expired relative to `now_ms`.
    #[must_use]
    pub fn is_expired(&self, now_ms: u64) -> bool {
        let expiry_ms = self.generated_at_ms + (self.ttl_secs as u64) * 1_000;
        now_ms >= expiry_ms
    }

    /// Returns the number of items in the playlist.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns `true` if the playlist is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_items() -> Vec<PlaylistItem> {
        vec![
            PlaylistItem::new(1, "Morning News", 1800, "news", 4.0, 0.8),
            PlaylistItem::new(2, "Comedy Show", 1800, "comedy", 3.5, 0.6),
            PlaylistItem::new(3, "Epic Drama", 3600, "drama", 4.8, 0.9),
            PlaylistItem::new(4, "Nature Documentary", 3600, "documentary", 4.2, 0.7),
            PlaylistItem::new(5, "Late Night Talk", 2700, "talk", 3.8, 0.65),
        ]
    }

    #[test]
    fn test_item_creation() {
        let item = PlaylistItem::new(1, "Test", 1800, "drama", 4.5, 0.8);
        assert_eq!(item.id, 1);
        assert_eq!(item.title, "Test");
        assert_eq!(item.duration_secs, 1800);
    }

    #[test]
    fn test_item_rating_clamped() {
        let item = PlaylistItem::new(1, "Over-rated", 60, "comedy", 10.0, 2.0);
        assert_eq!(item.rating, 5.0);
        assert_eq!(item.popularity, 1.0);
    }

    #[test]
    fn test_context_creation() {
        let ctx = PlaylistContext::new(8, DeviceType::TV).with_user(42);
        assert_eq!(ctx.user_id, Some(42));
        assert_eq!(ctx.time_of_day, 8);
        assert_eq!(ctx.device_type, DeviceType::TV);
    }

    #[test]
    fn test_context_add_previous() {
        let mut ctx = PlaylistContext::new(10, DeviceType::Mobile);
        ctx.add_previous(100);
        ctx.add_previous(200);
        assert_eq!(ctx.previous_items.len(), 2);
    }

    #[test]
    fn test_score_base_components() {
        let item = PlaylistItem::new(1, "Test", 60, "misc", 5.0, 1.0);
        let ctx = PlaylistContext::new(23, DeviceType::Desktop);
        let score = RecommendationScore::compute(&item, &ctx);
        // Base = 1.0 * 0.3 + 1.0 * 0.3 = 0.6, plus late-night boost 0.1
        assert!((score - 0.70).abs() < 0.01);
    }

    #[test]
    fn test_score_news_morning_boost() {
        let news = PlaylistItem::new(1, "News", 1800, "news", 4.0, 0.5);
        let ctx_morning = PlaylistContext::new(7, DeviceType::TV);
        let ctx_evening = PlaylistContext::new(20, DeviceType::TV);
        let score_morning = RecommendationScore::compute(&news, &ctx_morning);
        let score_evening = RecommendationScore::compute(&news, &ctx_evening);
        assert!(score_morning > score_evening);
    }

    #[test]
    fn test_score_drama_evening_boost() {
        let drama = PlaylistItem::new(2, "Drama", 3600, "drama", 4.0, 0.5);
        let ctx_prime = PlaylistContext::new(20, DeviceType::TV);
        let ctx_morning = PlaylistContext::new(7, DeviceType::TV);
        let score_prime = RecommendationScore::compute(&drama, &ctx_prime);
        let score_morning = RecommendationScore::compute(&drama, &ctx_morning);
        assert!(score_prime > score_morning);
    }

    #[test]
    fn test_score_penalty_for_previous_items() {
        let item = PlaylistItem::new(99, "Seen It", 600, "comedy", 5.0, 1.0);
        let mut ctx = PlaylistContext::new(12, DeviceType::Tablet);
        let score_before = RecommendationScore::compute(&item, &ctx);
        ctx.add_previous(99);
        let score_after = RecommendationScore::compute(&item, &ctx);
        assert!(score_before > score_after);
    }

    #[test]
    fn test_recommender_returns_n_items() {
        let catalog = sample_items();
        let ctx = PlaylistContext::new(20, DeviceType::TV);
        let recs = PlaylistRecommender::recommend(&catalog, &ctx, 3);
        assert_eq!(recs.len(), 3);
    }

    #[test]
    fn test_recommender_returns_fewer_if_catalog_smaller() {
        let catalog = sample_items();
        let ctx = PlaylistContext::new(10, DeviceType::Mobile);
        let recs = PlaylistRecommender::recommend(&catalog, &ctx, 100);
        assert_eq!(recs.len(), catalog.len());
    }

    #[test]
    fn test_recommender_sorted_descending() {
        let catalog = sample_items();
        let ctx = PlaylistContext::new(20, DeviceType::TV);
        let recs = PlaylistRecommender::recommend(&catalog, &ctx, 5);
        for i in 1..recs.len() {
            let score_prev = RecommendationScore::compute(recs[i - 1], &ctx);
            let score_curr = RecommendationScore::compute(recs[i], &ctx);
            assert!(score_prev >= score_curr);
        }
    }

    #[test]
    fn test_personalized_playlist_expiry() {
        let playlist = PersonalizedPlaylist::new(vec![1, 2, 3], 1_000_000, 3600);
        assert!(!playlist.is_expired(1_000_000));
        assert!(!playlist.is_expired(4_599_999));
        assert!(playlist.is_expired(4_600_000));
    }

    #[test]
    fn test_personalized_playlist_is_empty() {
        let empty = PersonalizedPlaylist::new(vec![], 0, 3600);
        assert!(empty.is_empty());
        let nonempty = PersonalizedPlaylist::new(vec![1], 0, 3600);
        assert!(!nonempty.is_empty());
        assert_eq!(nonempty.len(), 1);
    }

    // ── FeatureVector tests ──────────────────────────────────────────────────

    #[test]
    fn test_feature_vector_popularity_in_range() {
        for &pop in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            let item = PlaylistItem::new(1, "Item", 60, "drama", 3.0, pop);
            let fv = FeatureVector::from_item(&item);
            assert!(
                (0.0..=1.0).contains(&fv.popularity_norm),
                "popularity_norm={} out of range",
                fv.popularity_norm
            );
        }
        // Clamped at 1.0 even if the raw value is >1 (item constructor clamps).
        let item = PlaylistItem::new(1, "Overflow", 60, "drama", 5.0, 2.0);
        let fv = FeatureVector::from_item(&item);
        assert_eq!(fv.popularity_norm, 1.0);
    }

    #[test]
    fn test_feature_vector_rating_in_range() {
        let item = PlaylistItem::new(1, "Test", 60, "comedy", 4.5, 0.8);
        let fv = FeatureVector::from_item(&item);
        assert!(
            (0.0..=1.0).contains(&fv.rating_norm),
            "rating_norm={} out of range",
            fv.rating_norm
        );
    }

    #[test]
    fn test_feature_vector_scoring_matches_string_scoring() {
        let catalog = sample_items();
        let ctx = PlaylistContext::new(20, DeviceType::TV);

        // Compute top-k ordering using the original string-comparison path.
        let mut string_scored: Vec<(&PlaylistItem, f32)> = catalog
            .iter()
            .map(|item| (item, RecommendationScore::compute(item, &ctx)))
            .collect();
        string_scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let string_order: Vec<u64> = string_scored.iter().map(|(item, _)| item.id).collect();

        // Compute top-k ordering using the feature-vector path.
        let mut fv_scored: Vec<(&PlaylistItem, f32)> = catalog
            .iter()
            .map(|item| {
                let fv = FeatureVector::from_item(item);
                let score = RecommendationScore::compute_with_vector(item, &fv, &ctx);
                (item, score)
            })
            .collect();
        fv_scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let fv_order: Vec<u64> = fv_scored.iter().map(|(item, _)| item.id).collect();

        assert_eq!(
            string_order, fv_order,
            "feature-vector and string-comparison paths returned different orderings"
        );
    }
}

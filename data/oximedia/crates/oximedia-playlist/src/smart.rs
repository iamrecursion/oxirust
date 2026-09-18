//! Smart playlists that dynamically generate content based on rules.
//!
//! A [`SmartPlaylist`] evaluates a set of [`SmartPlaylistRule`]s against a
//! library of [`LibraryItem`]s, applies optional sorting, and returns a
//! filtered, ordered list of items up to an optional limit.

use serde::{Deserialize, Serialize};

/// Metadata held about each media item in the library.
///
/// A real implementation would typically load this from a database; here the
/// struct is self-contained for testability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryItem {
    /// Unique identifier.
    pub id: String,
    /// Display title.
    pub title: String,
    /// Artist / author.
    pub artist: Option<String>,
    /// Genre tags.
    pub genres: Vec<String>,
    /// Duration in seconds.
    pub duration_secs: f64,
    /// Unix timestamp when the item was added to the library.
    pub added_at: u64,
    /// Number of times the item has been played.
    pub play_count: u32,
    /// Arbitrary string tags.
    pub tags: Vec<String>,
    /// User rating (0.0–5.0, `None` if not rated).
    pub rating: Option<f32>,
}

impl LibraryItem {
    /// Creates a minimal library item for testing or bootstrapping.
    #[must_use]
    pub fn new<S: Into<String>>(id: S, title: S) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            artist: None,
            genres: Vec::new(),
            duration_secs: 0.0,
            added_at: 0,
            play_count: 0,
            tags: Vec::new(),
            rating: None,
        }
    }

    /// Sets the artist.
    #[must_use]
    pub fn with_artist<S: Into<String>>(mut self, artist: S) -> Self {
        self.artist = Some(artist.into());
        self
    }

    /// Sets the duration in seconds.
    #[must_use]
    pub const fn with_duration(mut self, secs: f64) -> Self {
        self.duration_secs = secs;
        self
    }

    /// Sets the `added_at` Unix timestamp.
    #[must_use]
    pub const fn with_added_at(mut self, ts: u64) -> Self {
        self.added_at = ts;
        self
    }

    /// Sets the play count.
    #[must_use]
    pub const fn with_play_count(mut self, count: u32) -> Self {
        self.play_count = count;
        self
    }

    /// Adds a genre.
    #[must_use]
    pub fn with_genre<S: Into<String>>(mut self, genre: S) -> Self {
        self.genres.push(genre.into());
        self
    }

    /// Adds a tag.
    #[must_use]
    pub fn with_tag<S: Into<String>>(mut self, tag: S) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Sets the user rating.
    #[must_use]
    pub const fn with_rating(mut self, rating: f32) -> Self {
        self.rating = Some(rating);
        self
    }
}

/// A rule that a [`LibraryItem`] must satisfy to be included in a [`SmartPlaylist`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SmartPlaylistRule {
    /// Item must belong to the given genre (case-insensitive).
    Genre(String),
    /// Item duration must be at least this many seconds.
    MinDuration(f64),
    /// Item duration must be at most this many seconds.
    MaxDuration(f64),
    /// Item must have been added after the given Unix timestamp.
    AddedAfter(u64),
    /// Item must have been added before the given Unix timestamp.
    AddedBefore(u64),
    /// Item must have been played fewer than `n` times.
    PlayCountLessThan(u32),
    /// Item must have been played at least `n` times.
    PlayCountAtLeast(u32),
    /// Item must have the given tag.
    HasTag(String),
    /// Item must **not** have the given tag.
    NotTag(String),
    /// Item rating must be at least `min_rating`.
    MinRating(f32),
    /// Item artist must contain the given substring (case-insensitive).
    ArtistContains(String),
    /// Item title must contain the given substring (case-insensitive).
    TitleContains(String),
    /// All of the given rules must match.
    All(Vec<SmartPlaylistRule>),
    /// Any of the given rules must match.
    Any(Vec<SmartPlaylistRule>),
    /// Negates a rule.
    Not(Box<SmartPlaylistRule>),
}

impl SmartPlaylistRule {
    /// Returns `true` if `item` satisfies this rule.
    #[must_use]
    pub fn matches(&self, item: &LibraryItem) -> bool {
        match self {
            Self::Genre(g) => item
                .genres
                .iter()
                .any(|genre| genre.to_lowercase() == g.to_lowercase()),

            Self::MinDuration(min) => item.duration_secs >= *min,
            Self::MaxDuration(max) => item.duration_secs <= *max,

            Self::AddedAfter(ts) => item.added_at > *ts,
            Self::AddedBefore(ts) => item.added_at < *ts,

            Self::PlayCountLessThan(n) => item.play_count < *n,
            Self::PlayCountAtLeast(n) => item.play_count >= *n,

            Self::HasTag(tag) => item.tags.iter().any(|t| t == tag),
            Self::NotTag(tag) => !item.tags.iter().any(|t| t == tag),

            Self::MinRating(min) => item.rating.map(|r| r >= *min).unwrap_or(false),

            Self::ArtistContains(substr) => item
                .artist
                .as_ref()
                .map(|a| a.to_lowercase().contains(&substr.to_lowercase()))
                .unwrap_or(false),

            Self::TitleContains(substr) => {
                item.title.to_lowercase().contains(&substr.to_lowercase())
            }

            Self::All(rules) => rules.iter().all(|r| r.matches(item)),
            Self::Any(rules) => rules.iter().any(|r| r.matches(item)),
            Self::Not(rule) => !rule.matches(item),
        }
    }
}

/// Sort order for a [`SmartPlaylist`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SmartPlaylistSort {
    /// Pseudo-random order (deterministic based on item count).
    #[default]
    Random,
    /// Alphabetical by title.
    Title,
    /// Alphabetical by artist then title.
    Artist,
    /// Shortest first.
    DurationAsc,
    /// Longest first.
    DurationDesc,
    /// Most recently added first.
    DateAddedDesc,
    /// Oldest first.
    DateAddedAsc,
    /// Most played first.
    PlayCountDesc,
    /// Least played first.
    PlayCountAsc,
    /// Highest rated first.
    RatingDesc,
}

/// A smart playlist that filters and sorts library items dynamically.
///
/// # Example
///
/// ```
/// use oximedia_playlist::smart::{SmartPlaylist, SmartPlaylistRule, SmartPlaylistSort, LibraryItem};
///
/// let library = vec![
///     LibraryItem::new("a", "Rock Track").with_genre("Rock").with_duration(200.0),
///     LibraryItem::new("b", "Jazz Track").with_genre("Jazz").with_duration(180.0),
///     LibraryItem::new("c", "Short Rock").with_genre("Rock").with_duration(60.0),
/// ];
///
/// let playlist = SmartPlaylist::new()
///     .with_rule(SmartPlaylistRule::Genre("Rock".to_string()))
///     .with_rule(SmartPlaylistRule::MinDuration(100.0))
///     .with_sort(SmartPlaylistSort::DurationAsc)
///     .with_limit(10);
///
/// let result = playlist.generate(&library);
/// assert_eq!(result.len(), 1);
/// assert_eq!(result[0].id, "a");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartPlaylist {
    /// Optional name for the smart playlist.
    pub name: String,
    /// Rules that items must satisfy (all rules must match — AND logic).
    pub rules: Vec<SmartPlaylistRule>,
    /// Sort order.
    pub sort: SmartPlaylistSort,
    /// Optional maximum number of items to return.
    pub limit: Option<usize>,
}

impl SmartPlaylist {
    /// Creates an empty smart playlist.
    #[must_use]
    pub fn new() -> Self {
        Self {
            name: "Smart Playlist".to_string(),
            rules: Vec::new(),
            sort: SmartPlaylistSort::default(),
            limit: None,
        }
    }

    /// Sets the name.
    #[must_use]
    pub fn with_name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = name.into();
        self
    }

    /// Adds a rule.
    #[must_use]
    pub fn with_rule(mut self, rule: SmartPlaylistRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Sets the sort order.
    #[must_use]
    pub fn with_sort(mut self, sort: SmartPlaylistSort) -> Self {
        self.sort = sort;
        self
    }

    /// Sets the maximum number of returned items.
    #[must_use]
    pub const fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Evaluates all rules against `library` and returns a filtered, sorted list.
    ///
    /// All rules must match (AND semantics).  Use [`SmartPlaylistRule::Any`] for
    /// OR semantics within a single rule.
    #[must_use]
    pub fn generate<'a>(&self, library: &'a [LibraryItem]) -> Vec<&'a LibraryItem> {
        let mut result: Vec<&LibraryItem> = library
            .iter()
            .filter(|item| self.rules.iter().all(|rule| rule.matches(item)))
            .collect();

        self.sort_items(&mut result);

        if let Some(limit) = self.limit {
            result.truncate(limit);
        }

        result
    }

    /// Like [`Self::generate`] but clones the items so the result is owned.
    #[must_use]
    pub fn generate_owned(&self, library: &[LibraryItem]) -> Vec<LibraryItem> {
        self.generate(library).into_iter().cloned().collect()
    }

    /// Returns the number of items that would be returned from `library`.
    #[must_use]
    pub fn count(&self, library: &[LibraryItem]) -> usize {
        let n = library
            .iter()
            .filter(|item| self.rules.iter().all(|rule| rule.matches(item)))
            .count();
        self.limit.map(|l| l.min(n)).unwrap_or(n)
    }

    fn sort_items(&self, items: &mut Vec<&LibraryItem>) {
        match &self.sort {
            SmartPlaylistSort::Random => {
                // Deterministic pseudo-shuffle based on item count
                let n = items.len();
                let mut seed: u64 = n as u64 ^ 0xABCD_EF01_2345_6789;
                for i in (1..n).rev() {
                    seed = seed
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let j = (seed >> 33) as usize % (i + 1);
                    items.swap(i, j);
                }
            }
            SmartPlaylistSort::Title => {
                items.sort_by(|a, b| a.title.cmp(&b.title));
            }
            SmartPlaylistSort::Artist => {
                items.sort_by(|a, b| {
                    let ak = (a.artist.as_deref().unwrap_or(""), a.title.as_str());
                    let bk = (b.artist.as_deref().unwrap_or(""), b.title.as_str());
                    ak.cmp(&bk)
                });
            }
            SmartPlaylistSort::DurationAsc => {
                items.sort_by(|a, b| {
                    a.duration_secs
                        .partial_cmp(&b.duration_secs)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            SmartPlaylistSort::DurationDesc => {
                items.sort_by(|a, b| {
                    b.duration_secs
                        .partial_cmp(&a.duration_secs)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            SmartPlaylistSort::DateAddedDesc => {
                items.sort_by(|a, b| b.added_at.cmp(&a.added_at));
            }
            SmartPlaylistSort::DateAddedAsc => {
                items.sort_by(|a, b| a.added_at.cmp(&b.added_at));
            }
            SmartPlaylistSort::PlayCountDesc => {
                items.sort_by(|a, b| b.play_count.cmp(&a.play_count));
            }
            SmartPlaylistSort::PlayCountAsc => {
                items.sort_by(|a, b| a.play_count.cmp(&b.play_count));
            }
            SmartPlaylistSort::RatingDesc => {
                items.sort_by(|a, b| {
                    let ar = a.rating.unwrap_or(0.0);
                    let br = b.rating.unwrap_or(0.0);
                    br.partial_cmp(&ar).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }
    }
}

impl Default for SmartPlaylist {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn library() -> Vec<LibraryItem> {
        vec![
            LibraryItem::new("a", "Rock Anthem")
                .with_genre("Rock")
                .with_duration(250.0)
                .with_artist("Band A")
                .with_added_at(1_000)
                .with_play_count(5)
                .with_tag("favourite")
                .with_rating(4.5),
            LibraryItem::new("b", "Jazz Night")
                .with_genre("Jazz")
                .with_duration(180.0)
                .with_artist("Trio B")
                .with_added_at(2_000)
                .with_play_count(1)
                .with_rating(3.0),
            LibraryItem::new("c", "Short Rock")
                .with_genre("Rock")
                .with_duration(90.0)
                .with_artist("Band A")
                .with_added_at(3_000)
                .with_play_count(10)
                .with_tag("favourite"),
            LibraryItem::new("d", "Classical Piece")
                .with_genre("Classical")
                .with_duration(600.0)
                .with_artist("Orchestra C")
                .with_added_at(500)
                .with_play_count(0),
        ]
    }

    #[test]
    fn test_empty_rules_returns_all() {
        let lib = library();
        let pl = SmartPlaylist::new();
        assert_eq!(pl.generate(&lib).len(), 4);
    }

    #[test]
    fn test_genre_filter() {
        let lib = library();
        let pl = SmartPlaylist::new().with_rule(SmartPlaylistRule::Genre("Rock".to_string()));
        let result = pl.generate(&lib);
        assert_eq!(result.len(), 2);
        assert!(result
            .iter()
            .all(|i| i.genres.contains(&"Rock".to_string())));
    }

    #[test]
    fn test_min_duration_filter() {
        let lib = library();
        let pl = SmartPlaylist::new().with_rule(SmartPlaylistRule::MinDuration(200.0));
        let result = pl.generate(&lib);
        assert_eq!(result.len(), 2); // a (250) and d (600)
    }

    #[test]
    fn test_max_duration_filter() {
        let lib = library();
        let pl = SmartPlaylist::new().with_rule(SmartPlaylistRule::MaxDuration(200.0));
        let result = pl.generate(&lib);
        assert_eq!(result.len(), 2); // b (180) and c (90)
    }

    #[test]
    fn test_play_count_less_than() {
        let lib = library();
        let pl = SmartPlaylist::new().with_rule(SmartPlaylistRule::PlayCountLessThan(5));
        let result = pl.generate(&lib);
        // a=5 excluded (not < 5), b=1 included, c=10 excluded, d=0 included
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_has_tag_filter() {
        let lib = library();
        let pl = SmartPlaylist::new().with_rule(SmartPlaylistRule::HasTag("favourite".to_string()));
        let result = pl.generate(&lib);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_not_tag_filter() {
        let lib = library();
        let pl = SmartPlaylist::new().with_rule(SmartPlaylistRule::NotTag("favourite".to_string()));
        let result = pl.generate(&lib);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_added_after_filter() {
        let lib = library();
        let pl = SmartPlaylist::new().with_rule(SmartPlaylistRule::AddedAfter(1_500));
        let result = pl.generate(&lib);
        // b=2000, c=3000 included; a=1000, d=500 excluded
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_added_before_filter() {
        let lib = library();
        let pl = SmartPlaylist::new().with_rule(SmartPlaylistRule::AddedBefore(1_500));
        let result = pl.generate(&lib);
        // a=1000, d=500 included
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_min_rating_filter() {
        let lib = library();
        let pl = SmartPlaylist::new().with_rule(SmartPlaylistRule::MinRating(4.0));
        let result = pl.generate(&lib);
        // only a has rating >= 4.0
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, "a");
    }

    #[test]
    fn test_artist_contains_filter() {
        let lib = library();
        let pl =
            SmartPlaylist::new().with_rule(SmartPlaylistRule::ArtistContains("band".to_string()));
        let result = pl.generate(&lib);
        // "Band A" contains "band" (case-insensitive)
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_title_contains_filter() {
        let lib = library();
        let pl =
            SmartPlaylist::new().with_rule(SmartPlaylistRule::TitleContains("rock".to_string()));
        let result = pl.generate(&lib);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_all_rule() {
        let lib = library();
        let pl = SmartPlaylist::new().with_rule(SmartPlaylistRule::All(vec![
            SmartPlaylistRule::Genre("Rock".to_string()),
            SmartPlaylistRule::MinDuration(200.0),
        ]));
        let result = pl.generate(&lib);
        // Only "Rock Anthem" (Rock + 250s)
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, "a");
    }

    #[test]
    fn test_any_rule() {
        let lib = library();
        let pl = SmartPlaylist::new().with_rule(SmartPlaylistRule::Any(vec![
            SmartPlaylistRule::Genre("Jazz".to_string()),
            SmartPlaylistRule::Genre("Classical".to_string()),
        ]));
        let result = pl.generate(&lib);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_not_rule() {
        let lib = library();
        let pl = SmartPlaylist::new().with_rule(SmartPlaylistRule::Not(Box::new(
            SmartPlaylistRule::Genre("Rock".to_string()),
        )));
        let result = pl.generate(&lib);
        assert_eq!(result.len(), 2); // Jazz + Classical
    }

    #[test]
    fn test_limit() {
        let lib = library();
        let pl = SmartPlaylist::new().with_limit(2);
        let result = pl.generate(&lib);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_count() {
        let lib = library();
        let pl = SmartPlaylist::new()
            .with_rule(SmartPlaylistRule::Genre("Rock".to_string()))
            .with_limit(1);
        assert_eq!(pl.count(&lib), 1);
    }

    #[test]
    fn test_sort_by_title() {
        let lib = library();
        let pl = SmartPlaylist::new().with_sort(SmartPlaylistSort::Title);
        let result = pl.generate(&lib);
        let titles: Vec<&str> = result.iter().map(|i| i.title.as_str()).collect();
        let mut sorted = titles.clone();
        sorted.sort_unstable();
        assert_eq!(titles, sorted);
    }

    #[test]
    fn test_sort_by_duration_asc() {
        let lib = library();
        let pl = SmartPlaylist::new().with_sort(SmartPlaylistSort::DurationAsc);
        let result = pl.generate(&lib);
        for pair in result.windows(2) {
            assert!(pair[0].duration_secs <= pair[1].duration_secs);
        }
    }

    #[test]
    fn test_sort_by_duration_desc() {
        let lib = library();
        let pl = SmartPlaylist::new().with_sort(SmartPlaylistSort::DurationDesc);
        let result = pl.generate(&lib);
        for pair in result.windows(2) {
            assert!(pair[0].duration_secs >= pair[1].duration_secs);
        }
    }

    #[test]
    fn test_sort_by_date_added_desc() {
        let lib = library();
        let pl = SmartPlaylist::new().with_sort(SmartPlaylistSort::DateAddedDesc);
        let result = pl.generate(&lib);
        for pair in result.windows(2) {
            assert!(pair[0].added_at >= pair[1].added_at);
        }
    }

    #[test]
    fn test_sort_by_play_count_asc() {
        let lib = library();
        let pl = SmartPlaylist::new().with_sort(SmartPlaylistSort::PlayCountAsc);
        let result = pl.generate(&lib);
        for pair in result.windows(2) {
            assert!(pair[0].play_count <= pair[1].play_count);
        }
    }

    #[test]
    fn test_sort_by_rating_desc() {
        let lib = library();
        let pl = SmartPlaylist::new().with_sort(SmartPlaylistSort::RatingDesc);
        let result = pl.generate(&lib);
        // Items without a rating get 0.0, so rated items should come first.
        let rated = result
            .iter()
            .position(|i| i.rating.is_none())
            .unwrap_or(result.len());
        let all_rated = &result[..rated];
        for pair in all_rated.windows(2) {
            let a = pair[0].rating.unwrap_or(0.0);
            let b = pair[1].rating.unwrap_or(0.0);
            assert!(a >= b);
        }
    }

    #[test]
    fn test_generate_owned() {
        let lib = library();
        let pl = SmartPlaylist::new().with_rule(SmartPlaylistRule::Genre("Rock".to_string()));
        let owned = pl.generate_owned(&lib);
        assert_eq!(owned.len(), 2);
    }

    #[test]
    fn test_combined_rules() {
        let lib = library();
        let pl = SmartPlaylist::new()
            .with_name("Favourites Rock")
            .with_rule(SmartPlaylistRule::Genre("Rock".to_string()))
            .with_rule(SmartPlaylistRule::HasTag("favourite".to_string()))
            .with_sort(SmartPlaylistSort::DurationDesc)
            .with_limit(5);
        let result = pl.generate(&lib);
        // Both "Rock Anthem" and "Short Rock" have Rock + favourite
        assert_eq!(result.len(), 2);
        // Sorted by duration descending: 250 first, 90 second
        assert_eq!(result[0].id, "a");
        assert_eq!(result[1].id, "c");
    }
}

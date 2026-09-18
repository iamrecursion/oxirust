//! Content-based filtering using feature vectors.
//!
//! Items are represented by tags, genre, duration, rating, and year.  Similarity
//! is computed via a cosine distance on a binary tag vector combined with a
//! simple genre match bonus.

#![allow(dead_code)]

/// Features describing a piece of content.
pub struct ContentFeatures {
    /// Unique item identifier.
    pub item_id: u64,
    /// Descriptive tags for this item.
    pub tags: Vec<String>,
    /// Genre string (e.g. `"drama"`, `"comedy"`).
    pub genre: String,
    /// Duration of the content in seconds.
    pub duration_s: u64,
    /// Average user rating (0.0 – 10.0 or similar).
    pub rating: f32,
    /// Release year.
    pub year: u32,
}

impl ContentFeatures {
    /// Create a new item with no tags, empty genre, zero duration, etc.
    #[must_use]
    pub fn new(id: u64) -> Self {
        Self {
            item_id: id,
            tags: Vec::new(),
            genre: String::new(),
            duration_s: 0,
            rating: 0.0,
            year: 0,
        }
    }

    /// Add a tag to this item (duplicates are silently ignored).
    pub fn add_tag(&mut self, tag: &str) {
        let t = tag.to_owned();
        if !self.tags.contains(&t) {
            self.tags.push(t);
        }
    }

    /// Build a binary tag vector aligned to `all_tags`.
    ///
    /// Index `i` in the returned vector is `1.0` if `all_tags[i]` appears in
    /// this item's tag list, and `0.0` otherwise.
    #[must_use]
    pub fn tag_vector(&self, all_tags: &[String]) -> Vec<f32> {
        all_tags
            .iter()
            .map(|t| {
                if self.tags.contains(t) {
                    1.0f32
                } else {
                    0.0f32
                }
            })
            .collect()
    }
}

/// Compute a similarity score between two items.
///
/// The score is the cosine similarity of their binary tag vectors.  If the
/// genres also match a bonus of `0.1` is added (clamped to 1.0).
#[must_use]
pub fn item_similarity(a: &ContentFeatures, b: &ContentFeatures, all_tags: &[String]) -> f64 {
    let va = a.tag_vector(all_tags);
    let vb = b.tag_vector(all_tags);

    let dot: f64 = va
        .iter()
        .zip(vb.iter())
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum();
    let norm_a: f64 = va
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt();
    let norm_b: f64 = vb
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt();

    let cosine = if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    };

    let genre_bonus = if !a.genre.is_empty() && a.genre == b.genre {
        0.1
    } else {
        0.0
    };
    (cosine + genre_bonus).min(1.0)
}

/// A content-based filter that stores items and supports similarity queries.
pub struct ContentFilter {
    /// Registered content items.
    pub items: Vec<ContentFeatures>,
    /// Union of all tags seen across all items.
    pub all_tags: Vec<String>,
}

impl ContentFilter {
    /// Create an empty filter.
    #[must_use]
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            all_tags: Vec::new(),
        }
    }

    /// Add an item to the filter, updating the global tag vocabulary.
    pub fn add_item(&mut self, item: ContentFeatures) {
        for tag in &item.tags {
            if !self.all_tags.contains(tag) {
                self.all_tags.push(tag.clone());
            }
        }
        self.items.push(item);
    }

    /// Return the `n` most similar items to `item_id`, excluding the item itself.
    ///
    /// Results are `(item_id, similarity)` pairs sorted by descending similarity.
    #[must_use]
    pub fn similar_to(&self, item_id: u64, n: usize) -> Vec<(u64, f64)> {
        let target = match self.items.iter().find(|i| i.item_id == item_id) {
            Some(t) => t,
            None => return Vec::new(),
        };

        let mut scored: Vec<(u64, f64)> = self
            .items
            .iter()
            .filter(|i| i.item_id != item_id)
            .map(|other| {
                let sim = item_similarity(target, other, &self.all_tags);
                (other.item_id, sim)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(n);
        scored
    }

    /// Return all items with a matching genre (case-sensitive).
    #[must_use]
    pub fn by_genre(&self, genre: &str) -> Vec<&ContentFeatures> {
        self.items.iter().filter(|i| i.genre == genre).collect()
    }

    /// Return all items that contain *all* of the specified tags.
    #[must_use]
    pub fn by_tags(&self, tags: &[&str]) -> Vec<&ContentFeatures> {
        self.items
            .iter()
            .filter(|i| tags.iter().all(|t| i.tags.contains(&(*t).to_owned())))
            .collect()
    }
}

impl Default for ContentFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_item(id: u64, tags: &[&str], genre: &str) -> ContentFeatures {
        let mut f = ContentFeatures::new(id);
        f.genre = genre.to_owned();
        for &t in tags {
            f.add_tag(t);
        }
        f
    }

    #[test]
    fn test_content_features_new() {
        let f = ContentFeatures::new(1);
        assert_eq!(f.item_id, 1);
        assert!(f.tags.is_empty());
        assert!(f.genre.is_empty());
    }

    #[test]
    fn test_add_tag_no_duplicates() {
        let mut f = ContentFeatures::new(1);
        f.add_tag("action");
        f.add_tag("action");
        assert_eq!(f.tags.len(), 1);
    }

    #[test]
    fn test_tag_vector_length() {
        let mut f = ContentFeatures::new(1);
        f.add_tag("a");
        f.add_tag("b");
        let vocab = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
        let v = f.tag_vector(&vocab);
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 1.0);
        assert_eq!(v[2], 0.0);
    }

    #[test]
    fn test_item_similarity_identical() {
        let a = make_item(1, &["x", "y"], "drama");
        let b = make_item(2, &["x", "y"], "drama");
        let vocab = vec!["x".to_owned(), "y".to_owned()];
        let sim = item_similarity(&a, &b, &vocab);
        // cosine = 1.0, genre bonus = 0.1, clamped → 1.0
        assert!((sim - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_item_similarity_no_overlap() {
        let a = make_item(1, &["x"], "drama");
        let b = make_item(2, &["y"], "comedy");
        let vocab = vec!["x".to_owned(), "y".to_owned()];
        let sim = item_similarity(&a, &b, &vocab);
        assert!(sim < 0.01);
    }

    #[test]
    fn test_item_similarity_empty_tags() {
        let a = ContentFeatures::new(1);
        let b = ContentFeatures::new(2);
        let vocab: Vec<String> = Vec::new();
        let sim = item_similarity(&a, &b, &vocab);
        assert!((sim).abs() < 1e-9);
    }

    #[test]
    fn test_content_filter_add_and_by_genre() {
        let mut cf = ContentFilter::new();
        cf.add_item(make_item(1, &["a"], "drama"));
        cf.add_item(make_item(2, &["b"], "comedy"));
        cf.add_item(make_item(3, &["c"], "drama"));
        let dramas = cf.by_genre("drama");
        assert_eq!(dramas.len(), 2);
    }

    #[test]
    fn test_by_tags_all_must_match() {
        let mut cf = ContentFilter::new();
        cf.add_item(make_item(1, &["action", "thriller"], "action"));
        cf.add_item(make_item(2, &["action"], "action"));
        let found = cf.by_tags(&["action", "thriller"]);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].item_id, 1);
    }

    #[test]
    fn test_similar_to_unknown_item() {
        let cf = ContentFilter::new();
        assert!(cf.similar_to(999, 5).is_empty());
    }

    #[test]
    fn test_similar_to_excludes_self() {
        let mut cf = ContentFilter::new();
        cf.add_item(make_item(1, &["a"], "drama"));
        cf.add_item(make_item(2, &["a"], "drama"));
        let sims = cf.similar_to(1, 5);
        assert!(!sims.iter().any(|&(id, _)| id == 1));
    }

    #[test]
    fn test_similar_to_ordering() {
        let mut cf = ContentFilter::new();
        cf.add_item(make_item(1, &["a", "b", "c"], "drama"));
        // item 2: shares all 3 tags with item 1
        cf.add_item(make_item(2, &["a", "b", "c"], "drama"));
        // item 3: shares only 1 tag
        cf.add_item(make_item(3, &["a"], "comedy"));
        let sims = cf.similar_to(1, 2);
        assert_eq!(sims[0].0, 2);
    }

    #[test]
    fn test_by_tags_empty_filter() {
        let cf = ContentFilter::new();
        assert!(cf.by_tags(&["action"]).is_empty());
    }

    #[test]
    fn test_all_tags_deduplicated() {
        let mut cf = ContentFilter::new();
        cf.add_item(make_item(1, &["a", "b"], "drama"));
        cf.add_item(make_item(2, &["b", "c"], "comedy"));
        // "b" should appear only once in all_tags
        let count = cf.all_tags.iter().filter(|t| t.as_str() == "b").count();
        assert_eq!(count, 1);
    }
}

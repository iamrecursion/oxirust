//! Search engine for clips.

use crate::clip::Clip;

/// Search engine for finding clips.
#[derive(Debug, Clone, Default)]
pub struct SearchEngine;

impl SearchEngine {
    /// Creates a new search engine.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Searches clips by query string.
    #[must_use]
    pub fn search<'a>(&self, clips: &'a [Clip], query: &str) -> Vec<&'a Clip> {
        let query_lower = query.to_lowercase();

        clips
            .iter()
            .filter(|clip| {
                // Search in name
                if clip.name.to_lowercase().contains(&query_lower) {
                    return true;
                }

                // Search in description
                if let Some(desc) = &clip.description {
                    if desc.to_lowercase().contains(&query_lower) {
                        return true;
                    }
                }

                // Search in keywords
                for keyword in &clip.keywords {
                    if keyword.to_lowercase().contains(&query_lower) {
                        return true;
                    }
                }

                // Search in file path
                if let Some(file_name) = clip.file_path.file_name() {
                    if let Some(name_str) = file_name.to_str() {
                        if name_str.to_lowercase().contains(&query_lower) {
                            return true;
                        }
                    }
                }

                false
            })
            .collect()
    }

    /// Searches clips by multiple terms (AND logic).
    #[must_use]
    pub fn search_multi<'a>(&self, clips: &'a [Clip], terms: &[&str]) -> Vec<&'a Clip> {
        clips
            .iter()
            .filter(|clip| {
                terms
                    .iter()
                    .all(|term| !self.search(std::slice::from_ref(clip), term).is_empty())
            })
            .collect()
    }

    /// Searches clips by keyword.
    #[must_use]
    pub fn search_by_keyword<'a>(&self, clips: &'a [Clip], keyword: &str) -> Vec<&'a Clip> {
        clips
            .iter()
            .filter(|clip| clip.keywords.contains(&keyword.to_string()))
            .collect()
    }

    /// Full-text search with scoring.
    #[must_use]
    pub fn search_scored<'a>(&self, clips: &'a [Clip], query: &str) -> Vec<(&'a Clip, f64)> {
        let query_lower = query.to_lowercase();
        let mut results: Vec<(&Clip, f64)> = clips
            .iter()
            .filter_map(|clip| {
                let score = Self::calculate_score(clip, &query_lower);
                if score > 0.0 {
                    Some((clip, score))
                } else {
                    None
                }
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    fn calculate_score(clip: &Clip, query: &str) -> f64 {
        let mut score = 0.0;

        // Name match (highest weight)
        if clip.name.to_lowercase().contains(query) {
            score += 10.0;
            // Exact match bonus
            if clip.name.to_lowercase() == query {
                score += 20.0;
            }
        }

        // Keyword match (high weight)
        for keyword in &clip.keywords {
            if keyword.to_lowercase().contains(query) {
                score += 5.0;
            }
        }

        // Description match (medium weight)
        if let Some(desc) = &clip.description {
            if desc.to_lowercase().contains(query) {
                score += 3.0;
            }
        }

        // File name match (low weight)
        if let Some(file_name) = clip.file_path.file_name() {
            if let Some(name_str) = file_name.to_str() {
                if name_str.to_lowercase().contains(query) {
                    score += 1.0;
                }
            }
        }

        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_search_by_name() {
        let mut clip1 = Clip::new(PathBuf::from("/test1.mov"));
        clip1.set_name("Interview John");

        let mut clip2 = Clip::new(PathBuf::from("/test2.mov"));
        clip2.set_name("Interview Jane");

        let clips = vec![clip1, clip2];
        let engine = SearchEngine::new();

        let results = engine.search(&clips, "john");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Interview John");
    }

    #[test]
    fn test_search_by_keyword() {
        let mut clip1 = Clip::new(PathBuf::from("/test1.mov"));
        clip1.add_keyword("interview");

        let mut clip2 = Clip::new(PathBuf::from("/test2.mov"));
        clip2.add_keyword("action");

        let clips = vec![clip1, clip2];
        let engine = SearchEngine::new();

        let results = engine.search_by_keyword(&clips, "interview");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_search_scored() {
        let mut clip1 = Clip::new(PathBuf::from("/test1.mov"));
        clip1.set_name("Test");

        let mut clip2 = Clip::new(PathBuf::from("/test2.mov"));
        clip2.set_name("Another Test");
        clip2.add_keyword("test");

        let clips = vec![clip1, clip2];
        let engine = SearchEngine::new();

        let results = engine.search_scored(&clips, "test");
        assert_eq!(results.len(), 2);
        // Clip with keyword should have higher score
        assert!(results[0].1 > 0.0);
    }
}

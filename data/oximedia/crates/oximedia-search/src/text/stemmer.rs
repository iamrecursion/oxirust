//! Text stemming for better search matching.

/// Simple stemmer using basic rules
pub struct Stemmer;

impl Stemmer {
    /// Create a new stemmer
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Stem a word to its root form
    #[must_use]
    pub fn stem(&self, word: &str) -> String {
        let mut stem = word.to_lowercase();

        // Remove common suffixes
        stem = self.remove_suffix(&stem, "ing");
        stem = self.remove_suffix(&stem, "ed");
        stem = self.remove_suffix(&stem, "ly");
        stem = self.remove_suffix(&stem, "s");
        stem = self.remove_suffix(&stem, "es");
        stem = self.remove_suffix(&stem, "ies");

        stem
    }

    /// Remove a suffix if present and result is long enough
    fn remove_suffix(&self, word: &str, suffix: &str) -> String {
        if word.len() > suffix.len() + 2 && word.ends_with(suffix) {
            word[..word.len() - suffix.len()].to_string()
        } else {
            word.to_string()
        }
    }
}

impl Default for Stemmer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stem_ing() {
        let stemmer = Stemmer::new();
        assert_eq!(stemmer.stem("running"), "runn");
        assert_eq!(stemmer.stem("walking"), "walk");
    }

    #[test]
    fn test_stem_ed() {
        let stemmer = Stemmer::new();
        assert_eq!(stemmer.stem("walked"), "walk");
        assert_eq!(stemmer.stem("jumped"), "jump");
    }

    #[test]
    fn test_stem_short_word() {
        let stemmer = Stemmer::new();
        // Short words should not be stemmed
        assert_eq!(stemmer.stem("is"), "is");
        assert_eq!(stemmer.stem("it"), "it");
    }
}

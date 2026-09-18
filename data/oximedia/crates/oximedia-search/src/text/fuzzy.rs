//! Fuzzy string matching for typo-tolerant search.

/// Fuzzy matcher using Levenshtein distance
pub struct FuzzyMatcher {
    max_distance: usize,
}

impl FuzzyMatcher {
    /// Create a new fuzzy matcher
    #[must_use]
    pub const fn new(max_distance: usize) -> Self {
        Self { max_distance }
    }

    /// Check if two strings match within the fuzzy threshold
    #[must_use]
    pub fn matches(&self, a: &str, b: &str) -> bool {
        self.distance(a, b) <= self.max_distance
    }

    /// Calculate Levenshtein distance between two strings
    #[must_use]
    pub fn distance(&self, a: &str, b: &str) -> usize {
        let a_len = a.len();
        let b_len = b.len();

        if a_len == 0 {
            return b_len;
        }
        if b_len == 0 {
            return a_len;
        }

        let mut matrix = vec![vec![0usize; b_len + 1]; a_len + 1];

        // Initialize first row and column
        for i in 0..=a_len {
            matrix[i][0] = i;
        }
        for j in 0..=b_len {
            matrix[0][j] = j;
        }

        // Fill the matrix
        for (i, a_char) in a.chars().enumerate() {
            for (j, b_char) in b.chars().enumerate() {
                let cost = usize::from(a_char != b_char);

                matrix[i + 1][j + 1] = (matrix[i][j + 1] + 1)
                    .min(matrix[i + 1][j] + 1)
                    .min(matrix[i][j] + cost);
            }
        }

        matrix[a_len][b_len]
    }

    /// Calculate similarity score (0.0 to 1.0)
    #[must_use]
    pub fn similarity(&self, a: &str, b: &str) -> f32 {
        let dist = self.distance(a, b);
        let max_len = a.len().max(b.len());

        if max_len == 0 {
            return 1.0;
        }

        1.0 - (dist as f32 / max_len as f32)
    }
}

impl Default for FuzzyMatcher {
    fn default() -> Self {
        Self::new(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match() {
        let matcher = FuzzyMatcher::new(2);
        assert_eq!(matcher.distance("hello", "hello"), 0);
        assert!(matcher.matches("hello", "hello"));
    }

    #[test]
    fn test_single_character_diff() {
        let matcher = FuzzyMatcher::new(2);
        assert_eq!(matcher.distance("hello", "hallo"), 1);
        assert!(matcher.matches("hello", "hallo"));
    }

    #[test]
    fn test_too_different() {
        let matcher = FuzzyMatcher::new(2);
        assert!(!matcher.matches("hello", "world"));
    }

    #[test]
    fn test_similarity() {
        let matcher = FuzzyMatcher::new(2);
        assert!((matcher.similarity("hello", "hello") - 1.0).abs() < f32::EPSILON);
        assert!(matcher.similarity("hello", "hallo") >= 0.8);
        assert!(matcher.similarity("hello", "world") < 0.5);
    }
}

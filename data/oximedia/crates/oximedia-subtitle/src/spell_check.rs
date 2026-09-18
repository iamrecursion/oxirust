//! Basic spell-checking framework for subtitle text.
//!
//! Provides dictionary lookup, edit-distance–based suggestion scoring,
//! and a simple correction pipeline.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::{HashMap, HashSet};

/// A simple dictionary backed by a `HashSet` of lowercase words.
#[derive(Clone, Debug, Default)]
pub struct Dictionary {
    words: HashSet<String>,
}

impl Dictionary {
    /// Create an empty dictionary.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a dictionary pre-loaded with `words`.
    #[must_use]
    pub fn from_words(words: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            words: words.into_iter().map(|w| w.into().to_lowercase()).collect(),
        }
    }

    /// Add a word to the dictionary.
    pub fn insert(&mut self, word: impl Into<String>) {
        self.words.insert(word.into().to_lowercase());
    }

    /// Returns `true` if the word (case-insensitive) is in the dictionary.
    #[must_use]
    pub fn contains(&self, word: &str) -> bool {
        self.words.contains(&word.to_lowercase())
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.words.len()
    }

    /// Returns `true` if the dictionary is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// Return all words in the dictionary as a sorted `Vec`.
    #[must_use]
    pub fn all_words(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.words.iter().map(String::as_str).collect();
        v.sort_unstable();
        v
    }
}

/// Compute the Levenshtein edit distance between two strings.
#[must_use]
pub fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let m = a.len();
    let n = b.len();

    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for (i, row) in dp.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in dp[0].iter_mut().enumerate() {
        *cell = j;
    }
    for i in 1..=m {
        for j in 1..=n {
            dp[i][j] = if a[i - 1] == b[j - 1] {
                dp[i - 1][j - 1]
            } else {
                1 + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1])
            };
        }
    }
    dp[m][n]
}

/// A single spelling suggestion with a score (lower = better).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Suggestion {
    /// The suggested word.
    pub word: String,
    /// Edit distance from the misspelled word.
    pub distance: usize,
}

impl PartialOrd for Suggestion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Suggestion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .cmp(&other.distance)
            .then_with(|| self.word.cmp(&other.word))
    }
}

/// A spell-checker that uses a [`Dictionary`] and edit-distance scoring.
#[derive(Clone, Debug)]
pub struct SpellChecker {
    dictionary: Dictionary,
    /// Maximum edit distance for a word to be considered a suggestion.
    pub max_distance: usize,
    /// Maximum number of suggestions returned per word.
    pub max_suggestions: usize,
}

impl SpellChecker {
    /// Create a `SpellChecker` with default settings.
    #[must_use]
    pub fn new(dictionary: Dictionary) -> Self {
        Self {
            dictionary,
            max_distance: 2,
            max_suggestions: 5,
        }
    }

    /// Check whether `word` is spelled correctly.
    #[must_use]
    pub fn is_correct(&self, word: &str) -> bool {
        self.dictionary.contains(word)
    }

    /// Return sorted suggestions for `word` (closest first).
    #[must_use]
    pub fn suggestions(&self, word: &str) -> Vec<Suggestion> {
        let lower = word.to_lowercase();
        let mut candidates: Vec<Suggestion> = self
            .dictionary
            .words
            .iter()
            .filter_map(|dict_word| {
                let dist = edit_distance(&lower, dict_word);
                if dist <= self.max_distance {
                    Some(Suggestion {
                        word: dict_word.clone(),
                        distance: dist,
                    })
                } else {
                    None
                }
            })
            .collect();
        candidates.sort();
        candidates.truncate(self.max_suggestions);
        candidates
    }

    /// Check a whole sentence and return a map of misspelled_word → suggestions.
    #[must_use]
    pub fn check_sentence(&self, sentence: &str) -> HashMap<String, Vec<Suggestion>> {
        let mut result: HashMap<String, Vec<Suggestion>> = HashMap::new();
        for token in sentence.split_whitespace() {
            let word = strip_punctuation(token);
            if word.is_empty() {
                continue;
            }
            if !self.is_correct(&word) {
                let sug = self.suggestions(&word);
                result.entry(word).or_insert(sug);
            }
        }
        result
    }

    /// Apply the best suggestion for every misspelled word in `sentence`,
    /// returning the corrected string.  If no suggestion exists the original
    /// token is kept.
    #[must_use]
    pub fn autocorrect(&self, sentence: &str) -> String {
        sentence
            .split_whitespace()
            .map(|token| {
                let stripped = strip_punctuation(token);
                if stripped.is_empty() || self.is_correct(&stripped) {
                    token.to_string()
                } else {
                    let sug = self.suggestions(&stripped);
                    if let Some(best) = sug.first() {
                        best.word.clone()
                    } else {
                        token.to_string()
                    }
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Strip leading/trailing punctuation from a word token.
fn strip_punctuation(token: &str) -> String {
    token
        .trim_matches(|c: char| c.is_ascii_punctuation())
        .to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn basic_dict() -> Dictionary {
        Dictionary::from_words(["hello", "world", "subtitle", "correct", "check", "spell"])
    }

    #[test]
    fn test_dictionary_contains_existing() {
        let dict = basic_dict();
        assert!(dict.contains("hello"));
    }

    #[test]
    fn test_dictionary_case_insensitive() {
        let dict = basic_dict();
        assert!(dict.contains("Hello"));
        assert!(dict.contains("WORLD"));
    }

    #[test]
    fn test_dictionary_missing_word() {
        let dict = basic_dict();
        assert!(!dict.contains("foobar"));
    }

    #[test]
    fn test_dictionary_insert() {
        let mut dict = Dictionary::new();
        dict.insert("test");
        assert!(dict.contains("test"));
    }

    #[test]
    fn test_dictionary_len() {
        let dict = basic_dict();
        assert_eq!(dict.len(), 6);
    }

    #[test]
    fn test_edit_distance_identical() {
        assert_eq!(edit_distance("hello", "hello"), 0);
    }

    #[test]
    fn test_edit_distance_one_insertion() {
        assert_eq!(edit_distance("helo", "hello"), 1);
    }

    #[test]
    fn test_edit_distance_one_substitution() {
        assert_eq!(edit_distance("hxllo", "hello"), 1);
    }

    #[test]
    fn test_edit_distance_empty_strings() {
        assert_eq!(edit_distance("", ""), 0);
    }

    #[test]
    fn test_edit_distance_one_vs_empty() {
        assert_eq!(edit_distance("a", ""), 1);
        assert_eq!(edit_distance("", "a"), 1);
    }

    #[test]
    fn test_spell_checker_correct_word() {
        let checker = SpellChecker::new(basic_dict());
        assert!(checker.is_correct("hello"));
    }

    #[test]
    fn test_spell_checker_incorrect_word() {
        let checker = SpellChecker::new(basic_dict());
        assert!(!checker.is_correct("helo"));
    }

    #[test]
    fn test_spell_checker_suggestions_close() {
        let checker = SpellChecker::new(basic_dict());
        let sug = checker.suggestions("helo");
        assert!(!sug.is_empty());
        assert_eq!(sug[0].word, "hello");
    }

    #[test]
    fn test_spell_checker_suggestions_too_far() {
        let checker = SpellChecker::new(basic_dict());
        // "xyz" is very far from all words
        let sug = checker.suggestions("xyz");
        assert!(sug.is_empty());
    }

    #[test]
    fn test_check_sentence_finds_misspelling() {
        let checker = SpellChecker::new(basic_dict());
        let result = checker.check_sentence("helo world");
        assert!(result.contains_key("helo"));
        assert!(!result.contains_key("world"));
    }

    #[test]
    fn test_autocorrect_fixes_word() {
        let checker = SpellChecker::new(basic_dict());
        let corrected = checker.autocorrect("helo world");
        assert!(corrected.contains("hello"));
    }

    #[test]
    fn test_strip_punctuation_removes_comma() {
        assert_eq!(strip_punctuation("hello,"), "hello");
    }

    #[test]
    fn test_suggestion_ordering() {
        let a = Suggestion {
            word: "a".to_string(),
            distance: 1,
        };
        let b = Suggestion {
            word: "b".to_string(),
            distance: 2,
        };
        assert!(a < b);
    }
}

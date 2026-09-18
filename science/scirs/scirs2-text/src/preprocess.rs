//! Text preprocessing utilities
//!
//! This module provides functionality for text normalization,
//! cleaning, and other preprocessing operations.

use crate::error::Result;
use lazy_static::lazy_static;
use regex::Regex;
use std::collections::HashSet;
use unicode_normalization::UnicodeNormalization;

lazy_static! {
    static ref SPECIAL_CHARS: Regex = Regex::new(r"[^\w\s]").expect("Operation failed");
    static ref WHITESPACE: Regex = Regex::new(r"\s+").expect("Operation failed");

    // Common English stopwords
    static ref DEFAULT_STOPWORDS: HashSet<String> = {
        let words = vec![
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
            "to", "was", "were", "will", "with"
        ];
        words.into_iter().map(String::from).collect()
    };
}

/// Trait for text normalization operations
pub trait TextNormalizer {
    /// Normalize the input text
    fn normalize(&self, text: &str) -> Result<String>;

    /// Normalize a batch of texts
    fn normalize_batch(&self, texts: &[&str]) -> Result<Vec<String>> {
        texts.iter().map(|text| self.normalize(text)).collect()
    }
}

/// Trait for text cleaning operations
pub trait TextCleaner {
    /// Clean the input text
    fn clean(&self, text: &str) -> Result<String>;

    /// Clean a batch of texts
    fn clean_batch(&self, texts: &[&str]) -> Result<Vec<String>> {
        texts.iter().map(|text| self.clean(text)).collect()
    }
}

/// Basic text normalizer that handles case folding and unicode normalization
#[derive(Debug, Clone)]
pub struct BasicNormalizer {
    lowercase: bool,
    unicode_normalization: bool,
}

impl BasicNormalizer {
    /// Create a new basic normalizer
    pub fn new(_lowercase: bool, unicodenormalization: bool) -> Self {
        Self {
            lowercase: _lowercase,
            unicode_normalization: unicodenormalization,
        }
    }
}

impl Default for BasicNormalizer {
    fn default() -> Self {
        Self::new(true, true)
    }
}

impl TextNormalizer for BasicNormalizer {
    fn normalize(&self, text: &str) -> Result<String> {
        let mut normalized = text.to_string();

        // Apply Unicode normalization (NFC form)
        if self.unicode_normalization {
            normalized = normalized.nfc().collect();
        }

        // Apply case folding
        if self.lowercase {
            normalized = normalized.to_lowercase();
        }

        Ok(normalized)
    }
}

/// Text cleaner for removing special characters, extra whitespace, and stopwords
#[derive(Debug, Clone)]
pub struct BasicTextCleaner {
    remove_special_chars: bool,
    remove_stopwords: bool,
    normalize_whitespace: bool,
    stopwords: HashSet<String>,
}

impl BasicTextCleaner {
    /// Create a new text cleaner
    pub fn new(
        remove_special_chars: bool,
        remove_stopwords: bool,
        normalize_whitespace: bool,
    ) -> Self {
        Self {
            remove_special_chars,
            remove_stopwords,
            normalize_whitespace: true,
            stopwords: DEFAULT_STOPWORDS.clone(),
        }
    }

    /// Create a text cleaner with custom stopwords
    pub fn with_stopwords(
        remove_special_chars: bool,
        remove_stopwords: bool,
        normalize_whitespace: bool,
        stopwords: HashSet<String>,
    ) -> Self {
        Self {
            remove_special_chars,
            remove_stopwords,
            normalize_whitespace,
            stopwords,
        }
    }

    /// Add stopwords to the cleaner
    pub fn add_stopwords(&mut self, words: &[&str]) {
        for word in words {
            self.stopwords.insert(word.to_string());
        }
    }

    /// Check if a word is a stopword
    pub fn is_stopword(&self, word: &str) -> bool {
        self.stopwords.contains(word)
    }
}

impl Default for BasicTextCleaner {
    fn default() -> Self {
        Self::new(true, true, true)
    }
}

impl TextCleaner for BasicTextCleaner {
    fn clean(&self, text: &str) -> Result<String> {
        let mut cleaned = text.to_string();

        // Remove special characters
        if self.remove_special_chars {
            cleaned = SPECIAL_CHARS.replace_all(&cleaned, " ").to_string();
        }

        // Normalize whitespace
        if self.normalize_whitespace {
            cleaned = WHITESPACE.replace_all(&cleaned, " ").trim().to_string();
        }

        // Remove stopwords
        if self.remove_stopwords {
            cleaned = cleaned
                .split_whitespace()
                .filter(|word| !self.is_stopword(word))
                .collect::<Vec<_>>()
                .join(" ");
        }

        Ok(cleaned)
    }
}

/// Pipeline for text preprocessing that combines normalization and cleaning
#[derive(Debug, Clone)]
pub struct TextPreprocessor {
    normalizer: BasicNormalizer,
    cleaner: BasicTextCleaner,
}

impl TextPreprocessor {
    /// Create a new text preprocessor
    pub fn new(normalizer: BasicNormalizer, cleaner: BasicTextCleaner) -> Self {
        Self {
            normalizer,
            cleaner,
        }
    }

    /// Process a text using the normalization and cleaning pipeline
    pub fn process(&self, text: &str) -> Result<String> {
        let normalized = self.normalizer.normalize(text)?;
        let cleaned = self.cleaner.clean(&normalized)?;
        Ok(cleaned)
    }

    /// Process a batch of texts
    pub fn process_batch(&self, texts: &[&str]) -> Result<Vec<String>> {
        texts.iter().map(|text| self.process(text)).collect()
    }
}

impl Default for TextPreprocessor {
    fn default() -> Self {
        Self::new(BasicNormalizer::default(), BasicTextCleaner::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_normalizer() {
        let normalizer = BasicNormalizer::default();
        let text = "Héllo, World!";
        let normalized = normalizer.normalize(text).expect("Operation failed");
        assert_eq!(normalized, "héllo, world!");
    }

    #[test]
    fn testtext_cleaner() {
        let cleaner = BasicTextCleaner::default();
        let text = "Hello, world! This is a test.";
        let cleaned = cleaner.clean(text).expect("Operation failed");
        assert_eq!(cleaned, "Hello world This test");
    }

    #[test]
    fn testtext_preprocessor() {
        let preprocessor = TextPreprocessor::default();
        let text = "Héllo, World! This is a test.";
        let processed = preprocessor.process(text).expect("Operation failed");
        assert_eq!(processed, "héllo world this test");
    }
}

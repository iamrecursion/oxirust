//! Vocabulary management for text processing
//!
//! This module provides functionality for building and managing vocabularies
//! used in text processing operations.

use crate::error::{Result, TextError};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

/// Vocabulary for mapping tokens to indices and vice versa
#[derive(Debug, Clone)]
pub struct Vocabulary {
    token_to_id: HashMap<String, usize>,
    id_to_token: HashMap<usize, String>,
    max_size: Option<usize>,
}

impl Vocabulary {
    /// Create a new empty vocabulary
    pub fn new() -> Self {
        Self {
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
            max_size: None,
        }
    }

    /// Create a new vocabulary with a maximum size
    pub fn with_maxsize(_maxsize: usize) -> Self {
        Self {
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
            max_size: Some(_maxsize),
        }
    }

    /// Create a vocabulary from a list of tokens
    pub fn fromtokens(tokens: &[String]) -> Self {
        let mut vocab = Self::new();
        for token in tokens {
            vocab.add_token(token);
        }
        vocab
    }

    /// Add a token to the vocabulary
    pub fn add_token(&mut self, token: &str) {
        // Check if we've reached the max size
        if let Some(max_size) = self.max_size {
            if self.token_to_id.len() >= max_size {
                return;
            }
        }

        // Add the token if it's not already in the vocabulary
        if !self.token_to_id.contains_key(token) {
            let id = self.token_to_id.len();
            self.token_to_id.insert(token.to_string(), id);
            self.id_to_token.insert(id, token.to_string());
        }
    }

    /// Get the index of a token
    pub fn get_index(&self, token: &str) -> Option<usize> {
        self.token_to_id.get(token).copied()
    }

    /// Get the token for an index
    pub fn get_token(&self, index: usize) -> Option<&str> {
        self.id_to_token.get(&index).map(|s| s.as_str())
    }

    /// Check if a token is in the vocabulary
    pub fn contains(&self, token: &str) -> bool {
        self.token_to_id.contains_key(token)
    }

    /// Get the number of tokens in the vocabulary
    pub fn len(&self) -> usize {
        self.token_to_id.len()
    }

    /// Check if the vocabulary is empty
    pub fn is_empty(&self) -> bool {
        self.token_to_id.is_empty()
    }

    /// Get all tokens in the vocabulary
    pub fn tokens(&self) -> HashSet<String> {
        self.token_to_id.keys().cloned().collect()
    }

    /// Get the vocabulary as a map from tokens to indices
    pub fn token_to_index(&self) -> &HashMap<String, usize> {
        &self.token_to_id
    }

    /// Save the vocabulary to a file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut file = File::create(path).map_err(|e| TextError::IoError(e.to_string()))?;

        // Sort by index
        let mut tokens: Vec<(usize, &String)> = self
            .id_to_token
            .iter()
            .map(|(id, token)| (*id, token))
            .collect();

        tokens.sort_by_key(|(id_, _)| *id_);

        // Write each token on a new line
        for (_, token) in tokens {
            writeln!(&mut file, "{token}").map_err(|e| TextError::IoError(e.to_string()))?;
        }

        Ok(())
    }

    /// Load a vocabulary from a file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path).map_err(|e| TextError::IoError(e.to_string()))?;

        let reader = BufReader::new(file);
        let mut vocab = Self::new();

        for (id, line) in reader.lines().enumerate() {
            let token = line.map_err(|e| TextError::IoError(e.to_string()))?;
            vocab.token_to_id.insert(token.clone(), id);
            vocab.id_to_token.insert(id, token);
        }

        Ok(vocab)
    }

    /// Prune the vocabulary to only include the most common tokens
    pub fn prune(&mut self, token_counts: &HashMap<String, usize>, mincount: usize) {
        // Create a new vocabulary with only tokens that meet the minimum _count
        let mut new_token_to_id = HashMap::new();
        let mut new_id_to_token = HashMap::new();

        let mut new_id = 0;
        for (token, count) in token_counts {
            if *count >= mincount && self.contains(token) {
                new_token_to_id.insert(token.clone(), new_id);
                new_id_to_token.insert(new_id, token.clone());
                new_id += 1;
            }
        }

        // Replace the current mappings
        self.token_to_id = new_token_to_id;
        self.id_to_token = new_id_to_token;
    }
}

impl Default for Vocabulary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Import needed for writeln! macro
    use tempfile::NamedTempFile;

    #[test]
    fn test_vocabulary_basics() {
        let mut vocab = Vocabulary::new();

        // Add tokens
        vocab.add_token("hello");
        vocab.add_token("world");
        vocab.add_token("test");

        // Check size
        assert_eq!(vocab.len(), 3);

        // Check indices
        assert_eq!(vocab.get_index("hello"), Some(0));
        assert_eq!(vocab.get_index("world"), Some(1));
        assert_eq!(vocab.get_index("test"), Some(2));

        // Check lookup by index
        assert_eq!(vocab.get_token(0), Some("hello"));
        assert_eq!(vocab.get_token(1), Some("world"));
        assert_eq!(vocab.get_token(2), Some("test"));

        // Check contains
        assert!(vocab.contains("hello"));
        assert!(!vocab.contains("unknown"));
    }

    #[test]
    fn test_vocabulary_fromtokens() {
        let tokens = vec![
            "hello".to_string(),
            "world".to_string(),
            "hello".to_string(), // Duplicate, should be ignored
            "test".to_string(),
        ];

        let vocab = Vocabulary::fromtokens(&tokens);

        // Check size (duplicates are removed)
        assert_eq!(vocab.len(), 3);

        // Check indices
        assert_eq!(vocab.get_index("hello"), Some(0));
        assert_eq!(vocab.get_index("world"), Some(1));
        assert_eq!(vocab.get_index("test"), Some(2));
    }

    #[test]
    fn test_vocabulary_maxsize() {
        let mut vocab = Vocabulary::with_maxsize(2);

        // Add tokens
        vocab.add_token("hello");
        vocab.add_token("world");
        vocab.add_token("test"); // Should be ignored due to max size

        // Check size
        assert_eq!(vocab.len(), 2);

        // Check that "test" wasn't added
        assert!(vocab.contains("hello"));
        assert!(vocab.contains("world"));
        assert!(!vocab.contains("test"));
    }

    #[test]
    fn test_vocabulary_file_io() {
        // Create a temporary file
        let temp_file = NamedTempFile::new().expect("Operation failed");

        // Create a vocabulary
        let mut vocab = Vocabulary::new();
        vocab.add_token("hello");
        vocab.add_token("world");
        vocab.add_token("test");

        // Save to file
        vocab
            .save_to_file(temp_file.path())
            .expect("Operation failed");

        // Load from file
        let loaded_vocab = Vocabulary::load_from_file(temp_file.path()).expect("Operation failed");

        // Check that it matches
        assert_eq!(loaded_vocab.len(), vocab.len());
        assert_eq!(loaded_vocab.get_index("hello"), vocab.get_index("hello"));
        assert_eq!(loaded_vocab.get_index("world"), vocab.get_index("world"));
        assert_eq!(loaded_vocab.get_index("test"), vocab.get_index("test"));
    }

    #[test]
    fn test_vocabulary_prune() {
        let mut vocab = Vocabulary::new();
        vocab.add_token("rare");
        vocab.add_token("common");
        vocab.add_token("frequent");

        // Create token counts
        let mut token_counts = HashMap::new();
        token_counts.insert("rare".to_string(), 1);
        token_counts.insert("common".to_string(), 5);
        token_counts.insert("frequent".to_string(), 10);

        // Prune tokens with count < 5
        vocab.prune(&token_counts, 5);

        // Check that "rare" was removed
        assert_eq!(vocab.len(), 2);
        assert!(!vocab.contains("rare"));
        assert!(vocab.contains("common"));
        assert!(vocab.contains("frequent"));
    }
}

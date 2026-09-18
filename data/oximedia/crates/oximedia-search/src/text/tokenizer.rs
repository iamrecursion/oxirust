//! Text tokenization for search.

use serde::{Deserialize, Serialize};

/// Token from text
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Token {
    /// Token text
    pub text: String,
    /// Start position in original text
    pub start: usize,
    /// End position in original text
    pub end: usize,
}

/// Text tokenizer
pub struct Tokenizer {
    /// Lowercase tokens
    lowercase: bool,
    /// Remove stopwords
    remove_stopwords: bool,
}

impl Tokenizer {
    /// Create a new tokenizer
    #[must_use]
    pub const fn new(lowercase: bool, remove_stopwords: bool) -> Self {
        Self {
            lowercase,
            remove_stopwords,
        }
    }

    /// Tokenize text into tokens
    #[must_use]
    pub fn tokenize(&self, text: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut current_start = 0;
        let mut current_token = String::new();

        for (i, c) in text.char_indices() {
            if c.is_alphanumeric() || c == '_' {
                if current_token.is_empty() {
                    current_start = i;
                }
                current_token.push(if self.lowercase {
                    c.to_lowercase().next().unwrap_or(c)
                } else {
                    c
                });
            } else if !current_token.is_empty() {
                if !self.is_stopword(&current_token) {
                    tokens.push(Token {
                        text: current_token.clone(),
                        start: current_start,
                        end: i,
                    });
                }
                current_token.clear();
            }
        }

        // Add last token if any
        if !current_token.is_empty() && !self.is_stopword(&current_token) {
            tokens.push(Token {
                text: current_token,
                start: current_start,
                end: text.len(),
            });
        }

        tokens
    }

    /// Check if a word is a stopword
    fn is_stopword(&self, word: &str) -> bool {
        if !self.remove_stopwords {
            return false;
        }

        // Common English stopwords
        const STOPWORDS: &[&str] = &[
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in",
            "is", "it", "its", "of", "on", "that", "the", "to", "was", "will", "with",
        ];

        STOPWORDS.contains(&word)
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new(true, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_basic() {
        let tokenizer = Tokenizer::new(true, false);
        let tokens = tokenizer.tokenize("Hello World");

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");
    }

    #[test]
    fn test_tokenize_with_stopwords() {
        let tokenizer = Tokenizer::new(true, true);
        let tokens = tokenizer.tokenize("the quick brown fox");

        assert_eq!(tokens.len(), 3); // "the" is filtered out
        assert_eq!(tokens[0].text, "quick");
    }

    #[test]
    fn test_token_positions() {
        let tokenizer = Tokenizer::new(false, false);
        let tokens = tokenizer.tokenize("Hello World");

        assert_eq!(tokens[0].start, 0);
        assert_eq!(tokens[0].end, 5);
        assert_eq!(tokens[1].start, 6);
    }
}

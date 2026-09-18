//! Language-agnostic Unicode tokenizer.
//!
//! Provides tokenization that is aware of Unicode character categories,
//! CJK ideographs, and accent stripping — implemented entirely in pure Rust
//! without any external Unicode normalization library.
//!
//! # Example
//!
//! ```rust
//! use scirs2_text::tokenizers::unicode::{UnicodeTokenizer, UnicodeTokenizerConfig};
//!
//! let tok = UnicodeTokenizer::new(UnicodeTokenizerConfig {
//!     lowercase: true,
//!     handle_cjk: true,
//!     split_on_punctuation: true,
//!     ..Default::default()
//! });
//!
//! let tokens = tok.tokenize("Hello, 世界! How are you?");
//! assert!(!tokens.is_empty());
//! ```

use std::collections::HashMap;

// ── UnicodeTokenizerConfig ────────────────────────────────────────────────────

/// Configuration for [`UnicodeTokenizer`].
#[derive(Debug, Clone)]
pub struct UnicodeTokenizerConfig {
    /// Convert all characters to lowercase before tokenizing. Default: true.
    pub lowercase: bool,
    /// Strip combining accents (approximation via Unicode range exclusion).
    /// Default: false.
    pub strip_accents: bool,
    /// Split on whitespace. Default: true.
    pub split_on_whitespace: bool,
    /// Split on punctuation characters. Default: true.
    pub split_on_punctuation: bool,
    /// Insert spaces around CJK characters so they become individual tokens.
    /// Default: true.
    pub handle_cjk: bool,
    /// Discard tokens shorter than this many bytes. Default: 1.
    pub min_token_length: usize,
    /// Maximum number of tokens to return. `None` = unlimited. Default: None.
    pub max_tokens: Option<usize>,
}

impl Default for UnicodeTokenizerConfig {
    fn default() -> Self {
        UnicodeTokenizerConfig {
            lowercase: true,
            strip_accents: false,
            split_on_whitespace: true,
            split_on_punctuation: true,
            handle_cjk: true,
            min_token_length: 1,
            max_tokens: None,
        }
    }
}

// ── UnicodeTokenizer ──────────────────────────────────────────────────────────

/// Language-agnostic Unicode-aware tokenizer.
///
/// Handles:
/// - ASCII and multi-byte Unicode text
/// - CJK ideographs (each character becomes a separate token)
/// - Punctuation splitting
/// - Optional accent stripping (approximate, pure Rust — no ICU)
/// - Optional lowercasing
pub struct UnicodeTokenizer {
    config: UnicodeTokenizerConfig,
}

impl UnicodeTokenizer {
    /// Create a new tokenizer with the given configuration.
    pub fn new(config: UnicodeTokenizerConfig) -> Self {
        UnicodeTokenizer { config }
    }

    /// Tokenize `text` into a `Vec<String>` of Unicode-aware tokens.
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return vec![];
        }

        // Step 1: optional lowercasing
        let working: String = if self.config.lowercase {
            text.to_lowercase()
        } else {
            text.to_string()
        };

        // Step 2: insert spaces around CJK characters
        let working: String = if self.config.handle_cjk {
            self.add_cjk_spaces(&working)
        } else {
            working
        };

        // Step 3: optional accent stripping
        let working: String = if self.config.strip_accents {
            Self::strip_accents_approx(&working)
        } else {
            working
        };

        // Step 4: whitespace split
        let raw_tokens: Vec<String> = if self.config.split_on_whitespace {
            working.split_whitespace().map(|s| s.to_string()).collect()
        } else {
            vec![working]
        };

        // Step 5: optionally split each token on punctuation
        let tokens: Vec<String> = if self.config.split_on_punctuation {
            raw_tokens
                .into_iter()
                .flat_map(|tok| self.split_on_punct(tok))
                .collect()
        } else {
            raw_tokens
        };

        // Step 6: filter by min length
        let mut tokens: Vec<String> = tokens
            .into_iter()
            .filter(|t| t.len() >= self.config.min_token_length)
            .collect();

        // Step 7: optional max_tokens truncation
        if let Some(max) = self.config.max_tokens {
            tokens.truncate(max);
        }

        tokens
    }

    /// Tokenize `text` and convert tokens to vocabulary indices.
    ///
    /// Unknown tokens are silently dropped.
    pub fn encode(&self, text: &str, vocab: &HashMap<String, usize>) -> Vec<usize> {
        self.tokenize(text)
            .iter()
            .filter_map(|tok| vocab.get(tok).copied())
            .collect()
    }

    // ── Private helpers ───────────────────────────────────────────────────

    /// Insert a space before and after every CJK ideograph.
    fn add_cjk_spaces(&self, s: &str) -> String {
        let mut out = String::with_capacity(s.len() + s.chars().count());
        for c in s.chars() {
            if Self::is_cjk(c) {
                out.push(' ');
                out.push(c);
                out.push(' ');
            } else {
                out.push(c);
            }
        }
        out
    }

    /// Split a token string at punctuation boundaries.
    fn split_on_punct(&self, tok: String) -> Vec<String> {
        let mut parts: Vec<String> = Vec::new();
        let mut current = String::new();
        for c in tok.chars() {
            if Self::is_punctuation(c) {
                if !current.is_empty() {
                    parts.push(current.clone());
                    current.clear();
                }
                parts.push(c.to_string());
            } else {
                current.push(c);
            }
        }
        if !current.is_empty() {
            parts.push(current);
        }
        parts
    }

    /// Returns `true` for CJK Unified Ideographs and CJK Extension A/B.
    ///
    /// Covers:
    /// - `U+4E00`–`U+9FFF`: CJK Unified Ideographs
    /// - `U+3400`–`U+4DBF`: CJK Extension A
    /// - `U+20000`–`U+2A6DF`: CJK Extension B
    #[inline]
    pub fn is_cjk(c: char) -> bool {
        matches!(c as u32,
            0x4E00..=0x9FFF   // CJK Unified Ideographs
            | 0x3400..=0x4DBF // CJK Extension A
            | 0x20000..=0x2A6DF // CJK Extension B
        )
    }

    /// Returns `true` for ASCII punctuation and common Unicode punctuation.
    #[inline]
    pub fn is_punctuation(c: char) -> bool {
        let cp = c as u32;
        // ASCII punctuation ranges
        if matches!(cp, 33..=47 | 58..=64 | 91..=96 | 123..=126) {
            return true;
        }
        // Unicode general punctuation: U+2000–U+206F
        if (0x2000..=0x206F).contains(&cp) {
            return true;
        }
        // CJK punctuation: U+3000–U+303F
        if (0x3000..=0x303F).contains(&cp) {
            return true;
        }
        // Fullwidth and other common punctuation: U+FF00–U+FFEF
        if (0xFF00..=0xFFEF).contains(&cp) {
            return true;
        }
        false
    }

    /// Approximate accent stripping: remove characters in the Unicode
    /// combining diacritics range `U+0300`–`U+036F`.
    ///
    /// This removes the *combining marks* but does not perform NFKD
    /// decomposition — for most Latin-script text the result is correct.
    /// Full NFKD would require the `unicode-normalization` crate.
    pub fn strip_accents_approx(s: &str) -> String {
        s.chars()
            .filter(|&c| {
                let cp = c as u32;
                !(0x0300..=0x036F).contains(&cp)
            })
            .collect()
    }
}

impl std::fmt::Debug for UnicodeTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnicodeTokenizer")
            .field("lowercase", &self.config.lowercase)
            .field("handle_cjk", &self.config.handle_cjk)
            .field("split_on_punctuation", &self.config.split_on_punctuation)
            .finish()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_tok() -> UnicodeTokenizer {
        UnicodeTokenizer::new(UnicodeTokenizerConfig::default())
    }

    // ── test_unicode_tokenizer_empty ──────────────────────────────────────

    #[test]
    fn test_unicode_tokenizer_empty() {
        let tok = default_tok();
        let tokens = tok.tokenize("");
        assert!(
            tokens.is_empty(),
            "empty string must produce empty token list"
        );
    }

    // ── test_unicode_tokenizer_cjk ────────────────────────────────────────

    #[test]
    fn test_unicode_tokenizer_cjk() {
        let tok = UnicodeTokenizer::new(UnicodeTokenizerConfig {
            lowercase: false,
            handle_cjk: true,
            split_on_punctuation: false,
            ..Default::default()
        });
        let tokens = tok.tokenize("你好world");
        // CJK chars 你 and 好 must be individual tokens
        assert!(
            tokens.contains(&"你".to_string()),
            "CJK char '你' should be its own token"
        );
        assert!(
            tokens.contains(&"好".to_string()),
            "CJK char '好' should be its own token"
        );
    }

    // ── test_unicode_tokenizer_punctuation ────────────────────────────────

    #[test]
    fn test_unicode_tokenizer_punctuation() {
        let tok = UnicodeTokenizer::new(UnicodeTokenizerConfig {
            lowercase: false,
            handle_cjk: false,
            split_on_punctuation: true,
            ..Default::default()
        });
        let tokens = tok.tokenize("Hello,world!");
        // "," and "!" should be separate tokens
        assert!(
            tokens.contains(&",".to_string()),
            "comma must be a separate token"
        );
        assert!(
            tokens.contains(&"!".to_string()),
            "exclamation must be a separate token"
        );
        assert!(
            tokens.contains(&"Hello".to_string()),
            "Hello must remain a token"
        );
    }

    #[test]
    fn test_unicode_tokenizer_lowercase() {
        let tok = UnicodeTokenizer::new(UnicodeTokenizerConfig {
            lowercase: true,
            split_on_punctuation: false,
            handle_cjk: false,
            ..Default::default()
        });
        let tokens = tok.tokenize("Hello World");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
    }

    #[test]
    fn test_unicode_tokenizer_max_tokens() {
        let tok = UnicodeTokenizer::new(UnicodeTokenizerConfig {
            max_tokens: Some(2),
            split_on_punctuation: false,
            handle_cjk: false,
            ..Default::default()
        });
        let tokens = tok.tokenize("one two three four five");
        assert_eq!(tokens.len(), 2);
    }

    #[test]
    fn test_unicode_tokenizer_min_length() {
        let tok = UnicodeTokenizer::new(UnicodeTokenizerConfig {
            min_token_length: 3,
            split_on_punctuation: false,
            handle_cjk: false,
            ..Default::default()
        });
        let tokens = tok.tokenize("a bb ccc dddd");
        // "a" (len 1) and "bb" (len 2) should be filtered out
        for t in &tokens {
            assert!(t.len() >= 3, "token '{t}' is too short");
        }
    }

    #[test]
    fn test_is_cjk_basic() {
        assert!(UnicodeTokenizer::is_cjk('中')); // U+4E2D
        assert!(UnicodeTokenizer::is_cjk('日')); // U+65E5
        assert!(!UnicodeTokenizer::is_cjk('A'));
        assert!(!UnicodeTokenizer::is_cjk('é'));
    }

    #[test]
    fn test_is_punctuation_ascii() {
        assert!(UnicodeTokenizer::is_punctuation(','));
        assert!(UnicodeTokenizer::is_punctuation('!'));
        assert!(UnicodeTokenizer::is_punctuation(';'));
        assert!(!UnicodeTokenizer::is_punctuation('a'));
        assert!(!UnicodeTokenizer::is_punctuation('5'));
    }

    #[test]
    fn test_strip_accents_approx() {
        // Test that combining diacritics are stripped
        // 'é' in decomposed form = 'e' + combining acute (U+0301)
        let decomposed = "e\u{0301}"; // 'e' + combining acute
        let stripped = UnicodeTokenizer::strip_accents_approx(decomposed);
        assert_eq!(stripped, "e", "combining accent should be stripped");
    }

    #[test]
    fn test_encode_returns_vocab_indices() {
        let tok = UnicodeTokenizer::new(UnicodeTokenizerConfig {
            split_on_punctuation: false,
            handle_cjk: false,
            ..Default::default()
        });
        let mut vocab = HashMap::new();
        vocab.insert("hello".to_string(), 0usize);
        vocab.insert("world".to_string(), 1usize);
        let indices = tok.encode("Hello World", &vocab);
        assert_eq!(indices, vec![0, 1]);
    }

    #[test]
    fn test_tokenize_mixed_script() {
        let tok = UnicodeTokenizer::new(UnicodeTokenizerConfig {
            handle_cjk: true,
            split_on_punctuation: true,
            lowercase: true,
            ..Default::default()
        });
        let tokens = tok.tokenize("Hello 世界 world!");
        assert!(!tokens.is_empty());
        // world and exclamation must be separate
        assert!(
            tokens.iter().any(|t| t == "world"),
            "world should be a token"
        );
    }
}

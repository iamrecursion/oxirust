//! N-gram Language Models
//!
//! This module provides statistical language models based on n-grams.
//! N-gram language models estimate the probability of word sequences
//! and can be used for text generation, auto-completion, and more.
//!
//! ## Overview
//!
//! An n-gram is a contiguous sequence of n items from a given text.
//! The n-gram model estimates:
//!
//! P(w_n | w_1, w_2, ..., w_{n-1})
//!
//! ## Supported Models
//!
//! - **Unigram**: P(word)
//! - **Bigram**: P(word | previous_word)
//! - **Trigram**: P(word | previous_two_words)
//! - **N-gram**: P(word | previous_n-1_words)
//!
//! ## Smoothing Techniques
//!
//! - **Laplace (Add-1) Smoothing**: Adds 1 to all counts
//! - **Add-k Smoothing**: Adds k to all counts
//! - **Kneser-Ney Smoothing**: Advanced smoothing based on continuation probability
//!
//! ## Quick Start
//!
//! ```rust
//! use scirs2_text::language_model::{NgramModel, SmoothingMethod};
//!
//! // Create a bigram model
//! let texts = vec![
//!     "the quick brown fox jumps over the lazy dog",
//!     "the dog was lazy but the fox was quick"
//! ];
//!
//! let mut model = NgramModel::new(2, SmoothingMethod::Laplace);
//! model.train(&texts).expect("Training failed");
//!
//! // Calculate probability
//! let prob = model.probability(&["the"], "quick").expect("Failed to get probability");
//! println!("P(quick | the) = {}", prob);
//!
//! // Generate text
//! let text = model.generate(10, Some("the")).expect("Generation failed");
//! println!("Generated: {}", text);
//! ```

use crate::error::{Result, TextError};
use crate::tokenize::{Tokenizer, WordTokenizer};
use scirs2_core::random::prelude::*;
use std::collections::HashMap;
use std::fmt::Debug;

/// Smoothing methods for n-gram models
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SmoothingMethod {
    /// No smoothing (maximum likelihood estimation)
    None,
    /// Laplace (add-1) smoothing
    Laplace,
    /// Add-k smoothing with custom k value
    AddK(f64),
    /// Kneser-Ney smoothing with discount parameter
    KneserNey(f64),
}

/// N-gram language model
pub struct NgramModel {
    /// Order of the n-gram model (n)
    n: usize,
    /// Smoothing method
    smoothing: SmoothingMethod,
    /// N-gram counts: (context, word) -> count
    ngram_counts: HashMap<Vec<String>, HashMap<String, usize>>,
    /// Context counts for normalization
    context_counts: HashMap<Vec<String>, usize>,
    /// Vocabulary
    vocabulary: Vec<String>,
    /// Total word count
    total_words: usize,
    /// Tokenizer
    tokenizer: Box<dyn Tokenizer + Send + Sync>,
}

impl Debug for NgramModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NgramModel")
            .field("n", &self.n)
            .field("smoothing", &self.smoothing)
            .field("vocabulary_size", &self.vocabulary.len())
            .field("total_words", &self.total_words)
            .finish()
    }
}

impl Clone for NgramModel {
    fn clone(&self) -> Self {
        Self {
            n: self.n,
            smoothing: self.smoothing,
            ngram_counts: self.ngram_counts.clone(),
            context_counts: self.context_counts.clone(),
            vocabulary: self.vocabulary.clone(),
            total_words: self.total_words,
            tokenizer: Box::new(WordTokenizer::default()),
        }
    }
}

impl NgramModel {
    /// Create a new n-gram model
    ///
    /// # Arguments
    ///
    /// * `n` - Order of the model (1 for unigram, 2 for bigram, etc.)
    /// * `smoothing` - Smoothing method to use
    pub fn new(n: usize, smoothing: SmoothingMethod) -> Self {
        if n == 0 {
            panic!("N-gram order must be at least 1");
        }

        Self {
            n,
            smoothing,
            ngram_counts: HashMap::new(),
            context_counts: HashMap::new(),
            vocabulary: Vec::new(),
            total_words: 0,
            tokenizer: Box::new(WordTokenizer::default()),
        }
    }

    /// Set a custom tokenizer
    pub fn with_tokenizer(mut self, tokenizer: Box<dyn Tokenizer + Send + Sync>) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /// Train the model on a corpus
    pub fn train(&mut self, texts: &[&str]) -> Result<()> {
        if texts.is_empty() {
            return Err(TextError::InvalidInput(
                "No texts provided for training".into(),
            ));
        }

        // Clear existing data
        self.ngram_counts.clear();
        self.context_counts.clear();
        self.vocabulary.clear();
        self.total_words = 0;

        // Collect vocabulary
        let mut vocab_set = std::collections::HashSet::new();

        for &text in texts {
            let tokens = self.tokenizer.tokenize(text)?;

            // Add start and end markers
            let mut augmented_tokens = vec!["<START>".to_string(); self.n - 1];
            augmented_tokens.extend(tokens);
            augmented_tokens.push("<END>".to_string());

            // Build vocabulary
            for token in &augmented_tokens {
                vocab_set.insert(token.clone());
            }

            // Count n-grams
            for i in (self.n - 1)..augmented_tokens.len() {
                let context = augmented_tokens[i - (self.n - 1)..i].to_vec();
                let word = &augmented_tokens[i];

                // Update n-gram counts
                *self
                    .ngram_counts
                    .entry(context.clone())
                    .or_default()
                    .entry(word.clone())
                    .or_insert(0) += 1;

                // Update context counts
                *self.context_counts.entry(context).or_insert(0) += 1;

                self.total_words += 1;
            }
        }

        self.vocabulary = vocab_set.into_iter().collect();
        self.vocabulary.sort();

        Ok(())
    }

    /// Calculate the probability of a word given its context
    ///
    /// # Arguments
    ///
    /// * `context` - The previous n-1 words
    /// * `word` - The word to predict
    ///
    /// # Returns
    ///
    /// The probability P(word | context)
    pub fn probability(&self, context: &[&str], word: &str) -> Result<f64> {
        if context.len() != self.n - 1 {
            return Err(TextError::InvalidInput(format!(
                "Context must have exactly {} words for {}-gram model",
                self.n - 1,
                self.n
            )));
        }

        let context_vec: Vec<String> = context.iter().map(|s| s.to_string()).collect();
        let vocab_size = self.vocabulary.len();

        match self.smoothing {
            SmoothingMethod::None => {
                // Maximum likelihood estimation
                let context_count = self.context_counts.get(&context_vec).copied().unwrap_or(0);

                if context_count == 0 {
                    return Ok(0.0);
                }

                let ngram_count = self
                    .ngram_counts
                    .get(&context_vec)
                    .and_then(|words| words.get(word))
                    .copied()
                    .unwrap_or(0);

                Ok(ngram_count as f64 / context_count as f64)
            }
            SmoothingMethod::Laplace => {
                // Add-1 smoothing
                let context_count = self.context_counts.get(&context_vec).copied().unwrap_or(0);

                let ngram_count = self
                    .ngram_counts
                    .get(&context_vec)
                    .and_then(|words| words.get(word))
                    .copied()
                    .unwrap_or(0);

                Ok((ngram_count + 1) as f64 / (context_count + vocab_size) as f64)
            }
            SmoothingMethod::AddK(k) => {
                // Add-k smoothing
                let context_count = self.context_counts.get(&context_vec).copied().unwrap_or(0);

                let ngram_count = self
                    .ngram_counts
                    .get(&context_vec)
                    .and_then(|words| words.get(word))
                    .copied()
                    .unwrap_or(0);

                Ok((ngram_count as f64 + k) / (context_count as f64 + k * vocab_size as f64))
            }
            SmoothingMethod::KneserNey(discount) => {
                // Simplified Kneser-Ney smoothing
                let context_count = self.context_counts.get(&context_vec).copied().unwrap_or(0);

                if context_count == 0 {
                    return Ok(1.0 / vocab_size as f64);
                }

                let ngram_count = self
                    .ngram_counts
                    .get(&context_vec)
                    .and_then(|words| words.get(word))
                    .copied()
                    .unwrap_or(0);

                let adjusted_count = (ngram_count as f64 - discount).max(0.0);
                let lambda = discount
                    * self
                        .ngram_counts
                        .get(&context_vec)
                        .map(|m| m.len())
                        .unwrap_or(0) as f64
                    / context_count as f64;

                let continuation_prob = 1.0 / vocab_size as f64;

                Ok(adjusted_count / context_count as f64 + lambda * continuation_prob)
            }
        }
    }

    /// Calculate perplexity on a test corpus
    ///
    /// Perplexity is a measure of how well the model predicts the test data.
    /// Lower perplexity indicates better performance.
    pub fn perplexity(&self, texts: &[&str]) -> Result<f64> {
        if texts.is_empty() {
            return Err(TextError::InvalidInput("No test texts provided".into()));
        }

        let mut log_prob_sum = 0.0;
        let mut word_count = 0;

        for &text in texts {
            let tokens = self.tokenizer.tokenize(text)?;

            let mut augmented_tokens = vec!["<START>".to_string(); self.n - 1];
            augmented_tokens.extend(tokens);
            augmented_tokens.push("<END>".to_string());

            for i in (self.n - 1)..augmented_tokens.len() {
                let context: Vec<&str> = augmented_tokens[i - (self.n - 1)..i]
                    .iter()
                    .map(|s| s.as_str())
                    .collect();
                let word = &augmented_tokens[i];

                let prob = self.probability(&context, word)?;

                if prob > 0.0 {
                    log_prob_sum += prob.ln();
                    word_count += 1;
                } else {
                    // Avoid log(0) by using a small probability
                    log_prob_sum += f64::ln(1e-10);
                    word_count += 1;
                }
            }
        }

        if word_count == 0 {
            return Ok(f64::INFINITY);
        }

        Ok((-log_prob_sum / word_count as f64).exp())
    }

    /// Generate text using the language model
    ///
    /// # Arguments
    ///
    /// * `max_length` - Maximum number of words to generate
    /// * `start_context` - Optional starting context (must have n-1 words)
    ///
    /// # Returns
    ///
    /// Generated text as a string
    pub fn generate(&self, max_length: usize, start_context: Option<&str>) -> Result<String> {
        let mut rng = scirs2_core::random::rng();
        let mut generated = Vec::new();

        // Initialize context
        let mut context: Vec<String> = if let Some(start) = start_context {
            let tokens = self.tokenizer.tokenize(start)?;
            if tokens.len() < self.n - 1 {
                let mut ctx = vec!["<START>".to_string(); self.n - 1 - tokens.len()];
                ctx.extend(tokens);
                ctx
            } else {
                tokens.into_iter().rev().take(self.n - 1).rev().collect()
            }
        } else {
            vec!["<START>".to_string(); self.n - 1]
        };

        // Generate words
        for _ in 0..max_length {
            let context_refs: Vec<&str> = context.iter().map(|s| s.as_str()).collect();

            // Get possible next words and their probabilities
            let candidates = match self.ngram_counts.get(&context) {
                Some(words) => words,
                None => {
                    // If context not found, sample from vocabulary
                    break;
                }
            };

            if candidates.is_empty() {
                break;
            }

            // Sample next word based on probabilities
            let total: usize = candidates.values().sum();
            let mut threshold = rng.random_range(0..total);
            let mut next_word = String::new();

            for (word, &count) in candidates {
                if threshold < count {
                    next_word = word.clone();
                    break;
                }
                threshold -= count;
            }

            if next_word == "<END>" {
                break;
            }

            if next_word != "<START>" {
                generated.push(next_word.clone());
            }

            // Update context
            context.remove(0);
            context.push(next_word);
        }

        Ok(generated.join(" "))
    }

    /// Get the most likely next words given a context
    ///
    /// # Arguments
    ///
    /// * `context` - The previous n-1 words
    /// * `top_n` - Number of suggestions to return
    ///
    /// # Returns
    ///
    /// Vector of (word, probability) pairs, sorted by probability (descending)
    pub fn suggest_next(&self, context: &[&str], top_n: usize) -> Result<Vec<(String, f64)>> {
        if context.len() != self.n - 1 {
            return Err(TextError::InvalidInput(format!(
                "Context must have exactly {} words",
                self.n - 1
            )));
        }

        let context_vec: Vec<String> = context.iter().map(|s| s.to_string()).collect();

        let candidates = match self.ngram_counts.get(&context_vec) {
            Some(words) => words,
            None => {
                return Ok(Vec::new());
            }
        };

        let mut suggestions: Vec<(String, f64)> = candidates
            .keys()
            .map(|word| {
                let prob = self.probability(context, word).unwrap_or(0.0);
                (word.clone(), prob)
            })
            .collect();

        suggestions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(suggestions.into_iter().take(top_n).collect())
    }

    /// Get the n-gram order
    pub fn order(&self) -> usize {
        self.n
    }

    /// Get the vocabulary size
    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unigram_model() {
        let texts = vec!["the cat sat on the mat", "the dog sat on the log"];

        let mut model = NgramModel::new(1, SmoothingMethod::Laplace);
        model.train(&texts).expect("Training failed");

        // "the" appears 4 times out of ~14 total words
        let prob = model
            .probability(&[], "the")
            .expect("Failed to get probability");
        assert!(prob > 0.0);
    }

    #[test]
    fn test_bigram_model() {
        let texts = vec!["the cat sat", "the dog sat"];

        let mut model = NgramModel::new(2, SmoothingMethod::Laplace);
        model.train(&texts).expect("Training failed");

        // P(cat | the) should be non-zero
        let prob = model
            .probability(&["the"], "cat")
            .expect("Failed to get probability");
        assert!(prob > 0.0);

        // P(dog | the) should be non-zero
        let prob = model
            .probability(&["the"], "dog")
            .expect("Failed to get probability");
        assert!(prob > 0.0);
    }

    #[test]
    fn test_trigram_model() {
        let texts = vec!["the quick brown fox", "the quick red fox"];

        let mut model = NgramModel::new(3, SmoothingMethod::Laplace);
        model.train(&texts).expect("Training failed");

        // P(brown | the quick)
        let prob = model
            .probability(&["the", "quick"], "brown")
            .expect("Failed to get probability");
        assert!(prob > 0.0);
    }

    #[test]
    fn test_smoothing_methods() {
        let texts = vec!["the cat sat"];

        // Test Laplace smoothing
        let mut model_laplace = NgramModel::new(2, SmoothingMethod::Laplace);
        model_laplace.train(&texts).expect("Training failed");

        let prob_laplace = model_laplace
            .probability(&["the"], "dog")
            .expect("Failed to get probability");
        assert!(
            prob_laplace > 0.0,
            "Laplace smoothing should give non-zero probability to unseen n-grams"
        );

        // Test Add-k smoothing
        let mut model_addk = NgramModel::new(2, SmoothingMethod::AddK(0.5));
        model_addk.train(&texts).expect("Training failed");

        let prob_addk = model_addk
            .probability(&["the"], "dog")
            .expect("Failed to get probability");
        assert!(prob_addk > 0.0);
    }

    #[test]
    fn test_text_generation() {
        let texts = vec![
            "the quick brown fox jumps over the lazy dog",
            "the quick brown dog runs fast",
        ];

        let mut model = NgramModel::new(2, SmoothingMethod::Laplace);
        model.train(&texts).expect("Training failed");

        let generated = model.generate(10, Some("the")).expect("Generation failed");
        assert!(!generated.is_empty());
    }

    #[test]
    fn test_perplexity() {
        let train_texts = vec!["the cat sat on the mat"];
        let test_texts = vec!["the cat sat"];

        let mut model = NgramModel::new(2, SmoothingMethod::Laplace);
        model.train(&train_texts).expect("Training failed");

        let perplexity = model
            .perplexity(&test_texts)
            .expect("Failed to calculate perplexity");
        assert!(perplexity > 0.0);
        assert!(perplexity.is_finite());
    }

    #[test]
    fn test_suggest_next() {
        let texts = vec!["the cat sat", "the cat ran", "the dog sat"];

        let mut model = NgramModel::new(2, SmoothingMethod::Laplace);
        model.train(&texts).expect("Training failed");

        let suggestions = model
            .suggest_next(&["the"], 3)
            .expect("Failed to get suggestions");

        assert!(!suggestions.is_empty());
        // "cat" and "dog" should be among the suggestions
        assert!(suggestions.iter().any(|(word, _)| word == "cat"));
    }
}

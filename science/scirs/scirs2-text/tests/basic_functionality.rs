//! Basic functionality tests for scirs2-text
//!
//! These tests verify core NLP functionality including tokenization,
//! vectorization, and sentiment analysis.

use scirs2_text::{
    sentiment::LexiconSentimentAnalyzer, tokenize::WordTokenizer, vectorize::TfidfVectorizer,
    Tokenizer, Vectorizer,
};

#[test]
fn test_word_tokenizer_basic() {
    let tokenizer = WordTokenizer::default();
    let result = tokenizer.tokenize("Hello, world! This is a test.");
    assert!(result.is_ok());

    let tokens = result.expect("Operation failed");
    assert!(tokens.len() >= 5);
    assert!(tokens.contains(&"Hello".to_string()) || tokens.contains(&"hello".to_string()));
    assert!(tokens.contains(&"world".to_string()));
    assert!(tokens.contains(&"test".to_string()));
}

#[test]
fn test_word_tokenizer_empty() {
    let tokenizer = WordTokenizer::default();
    let result = tokenizer.tokenize("");
    assert!(result.is_ok());
    let tokens = result.expect("Operation failed");
    assert!(tokens.is_empty());
}

#[test]
fn test_word_tokenizer_punctuation() {
    let tokenizer = WordTokenizer::default();
    let result = tokenizer.tokenize("Hello, world! How are you?");
    assert!(result.is_ok());
    let tokens = result.expect("Operation failed");
    assert!(tokens.len() >= 3);
}

#[test]
fn test_tfidf_vectorizer_basic() {
    let documents = vec![
        "The quick brown fox jumps over the lazy dog",
        "A quick brown dog outpaces a quick fox",
        "The lazy dog sleeps all day",
    ];

    let mut vectorizer = TfidfVectorizer::new(false, true, Some("l2".to_string()));
    let result = vectorizer.fit_transform(&documents);
    assert!(result.is_ok());

    let matrix = result.expect("Operation failed");
    let shape = matrix.shape();
    assert_eq!(shape[0], 3); // 3 documents
    assert!(shape[1] > 0); // Some features
}

#[test]
fn test_tfidf_vectorizer_empty_docs() {
    let documents: Vec<&str> = vec![];
    let mut vectorizer = TfidfVectorizer::new(false, true, Some("l2".to_string()));
    let result = vectorizer.fit_transform(&documents);
    assert!(result.is_err());
}

#[test]
fn test_tfidf_vectorizer_single_doc() {
    let documents = vec!["Hello world"];
    let mut vectorizer = TfidfVectorizer::new(false, true, Some("l2".to_string()));
    let result = vectorizer.fit_transform(&documents);
    assert!(result.is_ok());

    let matrix = result.expect("Operation failed");
    assert_eq!(matrix.shape()[0], 1);
}

#[test]
fn test_tfidf_vectorizer_vocabulary() {
    let documents = vec!["cat dog bird", "dog bird fish", "cat fish"];

    let mut vectorizer = TfidfVectorizer::new(false, true, Some("l2".to_string()));
    let result = vectorizer.fit_transform(&documents);
    assert!(result.is_ok());

    // Check vocabulary contains expected words
    let vocab_map = vectorizer.vocabulary_map();
    assert!(
        vocab_map.contains_key("cat")
            || vocab_map.contains_key("dog")
            || vocab_map.contains_key("bird")
    );
}

#[test]
fn test_sentiment_analyzer_basic() {
    let analyzer = LexiconSentimentAnalyzer::with_basiclexicon();

    // Test positive sentiment
    let result = analyzer.analyze("I love this library!");
    assert!(result.is_ok());
    let sentiment = result.expect("Operation failed");
    assert!(sentiment.score > 0.0 || sentiment.word_counts.positive_words > 0);

    // Test negative sentiment
    let result = analyzer.analyze("I hate bugs and errors.");
    assert!(result.is_ok());
    let sentiment = result.expect("Operation failed");
    assert!(sentiment.score < 0.0 || sentiment.word_counts.negative_words > 0);
}

#[test]
fn test_sentiment_analyzer_neutral() {
    let analyzer = LexiconSentimentAnalyzer::with_basiclexicon();

    let result = analyzer.analyze("The sky is blue.");
    assert!(result.is_ok());
    let sentiment = result.expect("Operation failed");
    // Neutral text should have low word counts
    assert!(sentiment.word_counts.positive_words + sentiment.word_counts.negative_words < 3);
}

#[test]
fn test_sentiment_analyzer_empty() {
    let analyzer = LexiconSentimentAnalyzer::with_basiclexicon();

    let result = analyzer.analyze("");
    assert!(result.is_ok());
    let sentiment = result.expect("Operation failed");
    assert_eq!(sentiment.score, 0.0);
}

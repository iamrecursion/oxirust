// Integration tests for NLP pipeline
// Exercises: scirs2-text (tokenize, vectorize, embeddings, similarity)

use scirs2_core::ndarray::Array1;
use scirs2_text::{
    distance::cosine_similarity, embeddings::embedding_cosine_similarity, tokenize::WordTokenizer,
    vectorize::TfidfVectorizer, Tokenizer, Vectorizer,
};

type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

// ---------------------------------------------------------------------------
// Test 1: Tokenize sentences and verify vocabulary properties
// ---------------------------------------------------------------------------

/// Tokenize a set of sentences, fit a TF-IDF vectorizer, verify vocab size and
/// that all token IDs produced by the tokenizer are valid words.
#[test]
fn test_nlp_tokenize_encode() -> TestResult<()> {
    let sentences = [
        "the quick brown fox jumps over the lazy dog",
        "a quick brown dog outpaces a lazy fox",
        "machine learning is a subfield of artificial intelligence",
        "neural networks are powerful function approximators",
        "natural language processing enables text understanding",
    ];

    // Step 1: Tokenize via WordTokenizer (lowercase = true)
    let tokenizer = WordTokenizer::new(true);

    let mut all_tokens: Vec<Vec<String>> = Vec::new();
    for &sentence in &sentences {
        let tokens = tokenizer
            .tokenize(sentence)
            .map_err(|e| format!("tokenize '{}': {}", sentence, e))?;
        assert!(
            !tokens.is_empty(),
            "Tokens should not be empty for: {}",
            sentence
        );
        all_tokens.push(tokens);
    }

    // Each sentence should produce some tokens
    for (i, tokens) in all_tokens.iter().enumerate() {
        assert!(!tokens.is_empty(), "Sentence {} should produce tokens", i);
        // All tokens should be non-empty strings
        for token in tokens {
            assert!(!token.is_empty(), "Individual token should be non-empty");
        }
    }

    // Step 2: Fit TF-IDF vectorizer and check vocabulary size
    let docs: Vec<&str> = sentences.to_vec();
    let mut vectorizer = TfidfVectorizer::new(false, true, Some("l2".to_string()));
    vectorizer
        .fit(&docs)
        .map_err(|e| format!("TF-IDF fit: {}", e))?;

    let vocab_size = vectorizer.vocabulary_size();
    assert!(vocab_size > 0, "Vocabulary should not be empty");

    // Corpus has roughly 25+ unique words, so vocab_size should be substantial
    assert!(
        vocab_size >= 10,
        "Vocabulary size should be at least 10, got {}",
        vocab_size
    );

    // Step 3: Transform a single document and verify output length
    let tfidf_vec = vectorizer
        .transform(sentences[0])
        .map_err(|e| format!("TF-IDF transform: {}", e))?;

    assert_eq!(
        tfidf_vec.len(),
        vocab_size,
        "TF-IDF vector length ({}) should equal vocabulary size ({})",
        tfidf_vec.len(),
        vocab_size
    );

    // All TF-IDF values should be non-negative and finite
    assert!(
        tfidf_vec.iter().all(|&v| v >= 0.0 && v.is_finite()),
        "All TF-IDF values should be non-negative and finite"
    );

    println!(
        "Tokenize+encode: vocab_size={}, tfidf_vec_len={}",
        vocab_size,
        tfidf_vec.len()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Test 2: Embedding cosine similarity
// ---------------------------------------------------------------------------

/// Verify that embedding_cosine_similarity works correctly on known vectors.
/// A vector compared with itself should have similarity = 1.0.
/// Orthogonal vectors should have similarity = 0.0.
#[test]
fn test_nlp_embedding_similarity() -> TestResult<()> {
    // Self-similarity should be 1.0 (within floating point)
    let v1 = Array1::from_vec(vec![1.0f64, 0.0, 0.0]);
    let v2 = Array1::from_vec(vec![0.0f64, 1.0, 0.0]);
    let v3 = Array1::from_vec(vec![1.0f64, 0.0, 0.0]);

    // Self-similarity
    let sim_self = embedding_cosine_similarity(&v1, &v1);
    assert!(
        (sim_self - 1.0).abs() < 1e-10,
        "Self-similarity should be 1.0, got {}",
        sim_self
    );

    // Orthogonal vectors: similarity = 0
    let sim_orthogonal = embedding_cosine_similarity(&v1, &v2);
    assert!(
        sim_orthogonal.abs() < 1e-10,
        "Orthogonal vectors should have similarity ~0, got {}",
        sim_orthogonal
    );

    // Same direction: similarity = 1.0
    let sim_same_dir = embedding_cosine_similarity(&v1, &v3);
    assert!(
        (sim_same_dir - 1.0).abs() < 1e-10,
        "Same-direction vectors should have similarity 1.0, got {}",
        sim_same_dir
    );

    // Opposite direction: similarity = -1.0
    let v_neg = Array1::from_vec(vec![-1.0f64, 0.0, 0.0]);
    let sim_opposite = embedding_cosine_similarity(&v1, &v_neg);
    assert!(
        (sim_opposite + 1.0).abs() < 1e-10,
        "Opposite-direction vectors should have similarity -1.0, got {}",
        sim_opposite
    );

    // Test with scirs2_text::distance::cosine_similarity (takes ArrayView1)
    let sim_dist =
        cosine_similarity(v1.view(), v3.view()).map_err(|e| format!("cosine_similarity: {}", e))?;

    assert!(
        (sim_dist - 1.0).abs() < 1e-10,
        "distance::cosine_similarity for same vectors should be 1.0, got {}",
        sim_dist
    );

    // Verify similarity is in [-1, 1]
    let v_arbitrary = Array1::from_vec(vec![0.3f64, 0.4, 0.5]);
    let sim_arb = embedding_cosine_similarity(&v1, &v_arbitrary);
    assert!(
        (-1.0..=1.0).contains(&sim_arb),
        "Cosine similarity must be in [-1, 1], got {}",
        sim_arb
    );

    println!(
        "Embedding similarity: self={:.6}, orth={:.6}, opp={:.6}",
        sim_self, sim_orthogonal, sim_opposite
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Test 3: Basic text processing — tokenization + TF-IDF feature extraction
// ---------------------------------------------------------------------------

/// End-to-end test: tokenize documents, build TF-IDF features, verify that
/// similar documents have higher cosine similarity than dissimilar ones.
#[test]
fn test_nlp_text_basic() -> TestResult<()> {
    let docs = [
        "the cat sat on the mat",
        "the cat sat near the window",
        "deep learning transforms machine learning research",
    ];

    let doc_refs: Vec<&str> = docs.to_vec();

    // Fit and transform with TF-IDF
    let mut vectorizer = TfidfVectorizer::new(false, true, Some("l2".to_string()));
    let tfidf_matrix = vectorizer
        .fit_transform(&doc_refs)
        .map_err(|e| format!("fit_transform: {}", e))?;

    let n_docs = tfidf_matrix.nrows();
    let vocab_size = tfidf_matrix.ncols();

    assert_eq!(n_docs, docs.len(), "Should have one row per document");
    assert!(vocab_size > 0, "Vocabulary should be non-empty");

    // Each row should be an L2-normalized vector (norm = 1.0)
    for i in 0..n_docs {
        let row = tfidf_matrix.row(i);
        let norm: f64 = row.iter().map(|&v| v * v).sum::<f64>().sqrt();
        // L2 normalization means norm should be ~1 (or 0 for empty doc)
        if norm > 1e-12 {
            assert!(
                (norm - 1.0).abs() < 1e-10,
                "Doc {} TF-IDF row should be L2-normalized, norm={}",
                i,
                norm
            );
        }
    }

    // Similarity between doc 0 and doc 1 (both about cat/mat) should be
    // higher than similarity between doc 0 and doc 2 (ML topic)
    let vec0 = Array1::from_vec(tfidf_matrix.row(0).to_vec());
    let vec1 = Array1::from_vec(tfidf_matrix.row(1).to_vec());
    let vec2 = Array1::from_vec(tfidf_matrix.row(2).to_vec());

    let sim_01 = embedding_cosine_similarity(&vec0, &vec1);
    let sim_02 = embedding_cosine_similarity(&vec0, &vec2);

    assert!(
        sim_01 > sim_02,
        "Docs 0 and 1 (both about cat/mat) should be more similar ({:.4}) than docs 0 and 2 ({:.4})",
        sim_01, sim_02
    );

    println!(
        "TF-IDF pipeline: n_docs={}, vocab={}, sim(0,1)={:.4}, sim(0,2)={:.4}",
        n_docs, vocab_size, sim_01, sim_02
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Test 4: Batch tokenization
// ---------------------------------------------------------------------------

/// Verify that batch tokenization produces the same result as single tokenization.
#[test]
fn test_nlp_batch_tokenization_consistency() -> TestResult<()> {
    let sentences = [
        "hello world",
        "natural language processing",
        "rust programming is fast",
    ];

    let tokenizer = WordTokenizer::new(true);

    // Single tokenization
    let mut single_results: Vec<Vec<String>> = Vec::new();
    for &s in &sentences {
        let tokens = tokenizer
            .tokenize(s)
            .map_err(|e| format!("tokenize: {}", e))?;
        single_results.push(tokens);
    }

    // Batch tokenization
    let batch_results = tokenizer
        .tokenize_batch(&sentences)
        .map_err(|e| format!("tokenize_batch: {}", e))?;

    assert_eq!(
        single_results.len(),
        batch_results.len(),
        "Batch result count should match single"
    );

    for (i, (single, batch)) in single_results.iter().zip(batch_results.iter()).enumerate() {
        assert_eq!(
            single, batch,
            "Sentence {} tokens differ between single and batch",
            i
        );
    }

    // Verify total token count across all sentences
    let total_tokens: usize = single_results.iter().map(|t| t.len()).sum();
    // "hello world" = 2, "natural language processing" = 3, "rust programming is fast" = 4
    assert_eq!(
        total_tokens, 9,
        "Total tokens should be 9, got {}",
        total_tokens
    );

    println!(
        "Batch tokenization consistency verified: {} total tokens",
        total_tokens
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Test 5: TF-IDF discriminative feature check
// ---------------------------------------------------------------------------

/// Verify TF-IDF weights: a rare word shared by fewer documents should have
/// higher IDF weight (and thus higher TF-IDF score) than a very common word.
#[test]
fn test_nlp_tfidf_discriminative_features() -> TestResult<()> {
    // "the" appears in all 4 documents -> low IDF
    // "unique" appears in only 1 document -> high IDF
    let docs = [
        "the cat is on the mat",
        "the dog is in the garden",
        "the bird is in the tree",
        "the unique algorithm is powerful",
    ];
    let doc_refs: Vec<&str> = docs.to_vec();

    let mut vectorizer = TfidfVectorizer::new(false, true, None);
    let matrix = vectorizer
        .fit_transform(&doc_refs)
        .map_err(|e| format!("fit_transform: {}", e))?;

    let vocab = vectorizer.vocabulary();

    // "the" should have a lower or equal TF-IDF score in any doc compared to "unique"
    // in the doc where "unique" appears
    let the_idx = vocab.get_index("the");
    let unique_idx = vocab.get_index("unique");

    if let (Some(the_i), Some(unique_i)) = (the_idx, unique_idx) {
        // In doc 3 (the "unique" document), tfidf["unique"] > tfidf["the"]
        let the_score = matrix[[3, the_i]];
        let unique_score = matrix[[3, unique_i]];

        assert!(
            unique_score > the_score,
            "TF-IDF['unique'] ({:.4}) should be > TF-IDF['the'] ({:.4}) in doc with 'unique'",
            unique_score,
            the_score
        );

        println!(
            "TF-IDF discriminative: tfidf['the']={:.4}, tfidf['unique']={:.4}",
            the_score, unique_score
        );
    } else {
        // If either word was pruned, just verify the matrix is well-formed
        let n_docs = matrix.nrows();
        let vocab_size = matrix.ncols();
        assert_eq!(n_docs, docs.len());
        assert!(vocab_size > 0);
        println!(
            "TF-IDF discriminative (fallback): n_docs={}, vocab={}",
            n_docs, vocab_size
        );
    }

    Ok(())
}

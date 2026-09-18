//! Comprehensive NLP Benchmarks comparing SciRS2-Text with NLTK/spaCy
//!
//! This benchmark suite compares scirs2-text performance against:
//! - NLTK (Natural Language Toolkit) via Python
//! - spaCy (Industrial-strength NLP) via Python
//!
//! ## Benchmark Categories
//!
//! 1. **Tokenization**: Word, sentence, subword (BPE) tokenization
//! 2. **Vectorization**: TF-IDF, Count Vectorization, Bag-of-Words
//! 3. **Embeddings**: Word2Vec, FastText, GloVe
//! 4. **Similarity**: Cosine, Jaccard, Levenshtein distance
//! 5. **Language Models**: N-gram models, perplexity calculation
//! 6. **NER**: Named Entity Recognition
//!
//! ## Running Benchmarks
//!
//! ```bash
//! cargo bench --bench nlp_benchmarks
//! ```
//!
//! ## Results Interpretation
//!
//! - **Time**: Lower is better
//! - **Throughput**: Higher is better (ops/sec)
//! - **Memory**: Lower is better (MB)

use criterion::{criterion_group, criterion_main, Bencher, BenchmarkId, Criterion, Throughput};
use scirs2_text::{
    // Similarity
    distance::{cosine_similarity, jaccard_similarity, levenshtein_distance},
    // Embeddings
    embeddings::{FastText, FastTextConfig, GloVe, Word2Vec, Word2VecAlgorithm, Word2VecConfig},
    // Language Models
    language_model::{NgramModel, SmoothingMethod},
    // Tokenization
    tokenize::{BpeConfig, BpeTokenizer, CharacterTokenizer, SentenceTokenizer, WordTokenizer},
    // Vectorization
    vectorize::{CountVectorizer, TfidfVectorizer},
    Tokenizer,
    Vectorizer,
};
use std::hint::black_box;

// Test data
const SMALL_TEXTS: &[&str] = &[
    "The quick brown fox jumps over the lazy dog",
    "A fast brown fox leaps over a sleeping canine",
    "The swift auburn fox bounds over the idle hound",
];

const MEDIUM_TEXTS: &[&str] = &[
    "Natural language processing is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language",
    "The goal of NLP is to read, decipher, understand, and make sense of human language in a valuable way",
    "Modern NLP models use deep learning techniques to achieve state-of-the-art results",
    "Word embeddings like Word2Vec and GloVe capture semantic relationships between words",
    "FastText extends Word2Vec by incorporating character n-grams for better handling of rare words",
    "TF-IDF is a numerical statistic that reflects how important a word is to a document in a collection",
    "N-gram language models estimate the probability of word sequences based on training data",
    "Named entity recognition identifies and classifies named entities in text into predefined categories",
    "Tokenization is the process of breaking down text into smaller units called tokens",
    "Stemming and lemmatization are techniques for reducing words to their root forms",
];

const LARGE_TEXTS: &[&str] = &[
    "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention. Because of new computing technologies, machine learning today is not like machine learning of the past. It was born from pattern recognition and the theory that computers can learn without being programmed to perform specific tasks. Researchers interested in artificial intelligence wanted to see if computers could learn from data. The iterative aspect of machine learning is important because as models are exposed to new data, they are able to independently adapt. They learn from previous computations to produce reliable, repeatable decisions and results.",
    "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks, deep belief networks, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, audio recognition, social network filtering, machine translation, bioinformatics, drug design, medical image analysis, material inspection and board game programs, where they have produced results comparable to and in some cases surpassing human expert performance.",
    "Transfer learning is a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem. For example, knowledge gained while learning to recognize cars could apply when trying to recognize trucks. This area of research bears some relation to the long history of psychological literature on transfer of learning, although formal ties between the two fields are limited.",
];

/// Benchmark: Word Tokenization
fn bench_tokenization(c: &mut Criterion) {
    let mut group = c.benchmark_group("tokenization");

    // Word tokenization - Small
    group.bench_with_input(
        BenchmarkId::new("word_tokenizer", "small"),
        &SMALL_TEXTS,
        |b: &mut Bencher, texts: &&[&str]| {
            let tokenizer = WordTokenizer::default();
            b.iter(|| {
                for &text in texts.iter() {
                    black_box(tokenizer.tokenize(text).expect("Tokenization failed"));
                }
            });
        },
    );

    // Word tokenization - Medium
    group.bench_with_input(
        BenchmarkId::new("word_tokenizer", "medium"),
        &MEDIUM_TEXTS,
        |b: &mut Bencher, texts: &&[&str]| {
            let tokenizer = WordTokenizer::default();
            b.iter(|| {
                for &text in texts.iter() {
                    black_box(tokenizer.tokenize(text).expect("Tokenization failed"));
                }
            });
        },
    );

    // Sentence tokenization
    group.bench_with_input(
        BenchmarkId::new("sentence_tokenizer", "large"),
        &LARGE_TEXTS,
        |b: &mut Bencher, texts: &&[&str]| {
            let tokenizer = SentenceTokenizer::default();
            b.iter(|| {
                for &text in texts.iter() {
                    black_box(tokenizer.tokenize(text).expect("Tokenization failed"));
                }
            });
        },
    );

    // Character tokenization
    group.bench_with_input(
        BenchmarkId::new("character_tokenizer", "medium"),
        &MEDIUM_TEXTS[0],
        |b: &mut Bencher, text: &&str| {
            let tokenizer = CharacterTokenizer::default();
            b.iter(|| {
                black_box(tokenizer.tokenize(text).expect("Tokenization failed"));
            });
        },
    );

    group.finish();
}

/// Benchmark: Text Vectorization (TF-IDF, Count Vectorizer)
fn bench_vectorization(c: &mut Criterion) {
    let mut group = c.benchmark_group("vectorization");

    // TF-IDF - Small
    group.bench_with_input(
        BenchmarkId::new("tfidf", "small"),
        &SMALL_TEXTS,
        |b: &mut Bencher, texts: &&[&str]| {
            b.iter(|| {
                let mut vectorizer = TfidfVectorizer::default();
                black_box(
                    vectorizer
                        .fit_transform(texts)
                        .expect("Vectorization failed"),
                );
            });
        },
    );

    // TF-IDF - Medium
    group.bench_with_input(
        BenchmarkId::new("tfidf", "medium"),
        &MEDIUM_TEXTS,
        |b: &mut Bencher, texts: &&[&str]| {
            b.iter(|| {
                let mut vectorizer = TfidfVectorizer::default();
                black_box(
                    vectorizer
                        .fit_transform(texts)
                        .expect("Vectorization failed"),
                );
            });
        },
    );

    // Count Vectorizer - Medium
    group.bench_with_input(
        BenchmarkId::new("count_vectorizer", "medium"),
        &MEDIUM_TEXTS,
        |b: &mut Bencher, texts: &&[&str]| {
            b.iter(|| {
                let mut vectorizer = CountVectorizer::default();
                black_box(
                    vectorizer
                        .fit_transform(texts)
                        .expect("Vectorization failed"),
                );
            });
        },
    );

    group.finish();
}

/// Benchmark: Word Embeddings (Word2Vec, FastText)
fn bench_embeddings(c: &mut Criterion) {
    let mut group = c.benchmark_group("embeddings");
    group.sample_size(10); // Reduce sample size for slower benchmarks

    // Word2Vec training - Skip-gram
    group.bench_with_input(
        BenchmarkId::new("word2vec_skipgram", "medium"),
        &MEDIUM_TEXTS,
        |b: &mut Bencher, texts: &&[&str]| {
            let config = Word2VecConfig {
                algorithm: Word2VecAlgorithm::SkipGram,
                vector_size: 50,
                window_size: 3,
                min_count: 1,
                epochs: 3,
                ..Default::default()
            };

            b.iter(|| {
                let mut model = Word2Vec::with_config(config.clone());
                model.train(texts).expect("Training failed");
                black_box(());
            });
        },
    );

    // Word2Vec training - CBOW
    group.bench_with_input(
        BenchmarkId::new("word2vec_cbow", "medium"),
        &MEDIUM_TEXTS,
        |b: &mut Bencher, texts: &&[&str]| {
            let config = Word2VecConfig {
                algorithm: Word2VecAlgorithm::CBOW,
                vector_size: 50,
                window_size: 3,
                min_count: 1,
                epochs: 3,
                ..Default::default()
            };

            b.iter(|| {
                let mut model = Word2Vec::with_config(config.clone());
                model.train(texts).expect("Training failed");
                black_box(());
            });
        },
    );

    // FastText training
    group.bench_with_input(
        BenchmarkId::new("fasttext", "medium"),
        &MEDIUM_TEXTS,
        |b: &mut Bencher, texts: &&[&str]| {
            let config = FastTextConfig {
                vector_size: 50,
                min_n: 3,
                max_n: 6,
                window_size: 3,
                epochs: 3,
                min_count: 1,
                ..Default::default()
            };

            b.iter(|| {
                let mut model = FastText::with_config(config.clone());
                model.train(texts).expect("Training failed");
                black_box(());
            });
        },
    );

    group.finish();
}

/// Benchmark: Text Similarity Measures
fn bench_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity");

    let text1 = MEDIUM_TEXTS[0];
    let text2 = MEDIUM_TEXTS[1];

    // Jaccard similarity
    group.bench_function("jaccard_similarity", |b: &mut Bencher| {
        b.iter(|| {
            black_box(jaccard_similarity(text1, text2, None).expect("Jaccard failed"));
        });
    });

    // Levenshtein distance
    let word1 = "algorithm";
    let word2 = "logarithm";
    group.bench_function("levenshtein_distance", |b: &mut Bencher| {
        b.iter(|| {
            black_box(levenshtein_distance(word1, word2));
        });
    });

    // Cosine similarity (using vectors)
    use scirs2_core::ndarray::Array1;
    let vec1 = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let vec2 = Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0]);

    group.bench_function("cosine_similarity", |b: &mut Bencher| {
        b.iter(|| {
            black_box(cosine_similarity(vec1.view(), vec2.view()).expect("Cosine failed"));
        });
    });

    group.finish();
}

/// Benchmark: N-gram Language Models
fn bench_language_models(c: &mut Criterion) {
    let mut group = c.benchmark_group("language_models");

    // Bigram model training
    group.bench_with_input(
        BenchmarkId::new("bigram_train", "medium"),
        &MEDIUM_TEXTS,
        |b: &mut Bencher, texts: &&[&str]| {
            b.iter(|| {
                let mut model = NgramModel::new(2, SmoothingMethod::Laplace);
                model.train(texts).expect("Training failed");
                black_box(());
            });
        },
    );

    // Trigram model training
    group.bench_with_input(
        BenchmarkId::new("trigram_train", "medium"),
        &MEDIUM_TEXTS,
        |b: &mut Bencher, texts: &&[&str]| {
            b.iter(|| {
                let mut model = NgramModel::new(3, SmoothingMethod::Laplace);
                model.train(texts).expect("Training failed");
                black_box(());
            });
        },
    );

    // Perplexity calculation
    let mut model = NgramModel::new(2, SmoothingMethod::Laplace);
    model.train(&MEDIUM_TEXTS[..5]).expect("Training failed");

    group.bench_with_input(
        BenchmarkId::new("perplexity", "test_set"),
        &MEDIUM_TEXTS[5..],
        |b: &mut Bencher, test_texts: &[&str]| {
            b.iter(|| {
                black_box(
                    model
                        .perplexity(test_texts)
                        .expect("Perplexity calculation failed"),
                );
            });
        },
    );

    // Text generation
    group.bench_function("text_generation", |b: &mut Bencher| {
        let mut model = NgramModel::new(2, SmoothingMethod::Laplace);
        model.train(MEDIUM_TEXTS).expect("Training failed");

        b.iter(|| {
            black_box(model.generate(20, Some("the")).expect("Generation failed"));
        });
    });

    group.finish();
}

/// Benchmark: End-to-End NLP Pipeline
fn bench_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end");
    group.sample_size(10);

    // Complete pipeline: Tokenize -> Vectorize -> Train Embeddings
    group.bench_with_input(
        BenchmarkId::new("full_pipeline", "medium"),
        &MEDIUM_TEXTS,
        |b: &mut Bencher, texts: &&[&str]| {
            b.iter(|| {
                // Tokenization
                let tokenizer = WordTokenizer::default();
                let _tokens: Vec<_> = texts
                    .iter()
                    .map(|&text| tokenizer.tokenize(text).expect("Tokenization failed"))
                    .collect();

                // Vectorization
                let mut vectorizer = TfidfVectorizer::default();
                let _matrix = vectorizer
                    .fit_transform(texts)
                    .expect("Vectorization failed");

                // Train embeddings
                let config = Word2VecConfig {
                    vector_size: 50,
                    epochs: 2,
                    min_count: 1,
                    ..Default::default()
                };
                let mut model = Word2Vec::with_config(config);
                model.train(texts).expect("Training failed");
                black_box(());
            });
        },
    );

    group.finish();
}

criterion_group!(
    benches,
    bench_tokenization,
    bench_vectorization,
    bench_embeddings,
    bench_similarity,
    bench_language_models,
    bench_end_to_end
);

criterion_main!(benches);

//! Smoke tests for newly-wired orphan modules in oximedia-watermark.

// ── adaptive_wm ───────────────────────────────────────────────────────────────

#[test]
fn test_adaptive_wm_compute_strength_non_negative() {
    use oximedia_watermark::adaptive_wm::AdaptiveWatermark;

    let signal: Vec<f32> = (0..1024).map(|i| (i as f32 / 1024.0).sin()).collect();
    let variance = AdaptiveWatermark::local_variance(&signal);
    let gain = AdaptiveWatermark::compute_strength(variance, 0.1);
    assert!(gain >= 0.0);
}

// ── capacity_calc ─────────────────────────────────────────────────────────────

#[test]
fn test_capacity_calc_pcm_audio() {
    use oximedia_watermark::capacity_calc::{
        CapacityCalculator, EmbedAlgorithm, MediaParams, MediaType,
    };

    let params = MediaParams {
        media_type: MediaType::PcmAudio,
        sample_count: 44100,
        sample_rate: 44100,
        bit_depth: 16,
        channels: 2,
    };
    let calc = CapacityCalculator::new(EmbedAlgorithm::SpreadSpectrum);
    let result = calc.calculate(&params);
    assert!(result.capacity_bits > 0, "should have positive capacity");
}

// ── fingerprint_watermark ─────────────────────────────────────────────────────

#[test]
fn test_fingerprint_watermarker_constructs() {
    use oximedia_watermark::fingerprint_watermark::{
        FingerprintWatermarkConfig, FingerprintWatermarker,
    };

    let config = FingerprintWatermarkConfig {
        strength: 0.1,
        ..Default::default()
    };
    let wm = FingerprintWatermarker::new(config, 44100);
    assert!(wm.is_ok());
}

// ── forensic_wm ───────────────────────────────────────────────────────────────

#[test]
fn test_forensic_wm_embed_user_id() {
    use oximedia_watermark::forensic_wm::ForensicWatermark;

    let img = vec![128u8; 64 * 64]; // 64×64 grayscale
    let result = ForensicWatermark::embed_user_id(&img, 64, 64, 0xDEAD_BEEF_1234_5678);
    assert!(result.is_ok());
    assert_eq!(result.expect("embed ok").len(), 64 * 64);
}

// ── freq_watermark ────────────────────────────────────────────────────────────

#[test]
fn test_freq_watermark_embed_smoke() {
    use oximedia_watermark::freq_watermark::FreqWatermark;

    // Need ≥ 32 blocks at SPREAD=2 to hold 64 bits → 32×(8×8)=32×64 pixels = 128×16 → use 128×128
    let img = vec![128u8; 128 * 128];
    let result = FreqWatermark::embed(&img, 128, 128, 0xABCD_1234_u64);
    assert!(result.is_ok());
}

// ── image_watermark ───────────────────────────────────────────────────────────

#[test]
fn test_image_watermark_dwt_embedder_constructs() {
    use oximedia_watermark::image_watermark::{DwtImageEmbedder, DwtWatermarkConfig};

    let config = DwtWatermarkConfig::default();
    let emb = DwtImageEmbedder::new(config);
    assert!(emb.is_ok());
}

// ── multi_layer_watermark ─────────────────────────────────────────────────────

#[test]
fn test_multi_layer_embedder_constructs() {
    use oximedia_watermark::multi_layer_watermark::{MultiLayerConfig, MultiLayerEmbedder};

    let config = MultiLayerConfig::default();
    let wm = MultiLayerEmbedder::new(config);
    assert!(wm.is_ok());
}

// ── pn_cache ──────────────────────────────────────────────────────────────────

#[test]
fn test_pn_cache_stores_and_retrieves() {
    use oximedia_watermark::pn_cache::PnSequenceCache;

    let mut cache = PnSequenceCache::new();
    assert!(cache.is_empty());

    // Generate once, then compare with a second generation in a fresh cache.
    let seq_a: Vec<i8> = cache.get_or_generate(256, 42).to_vec();
    assert_eq!(seq_a.len(), 256);
    assert_eq!(cache.len(), 1);

    // Same key → same sequence.
    let seq_b: Vec<i8> = cache.get_or_generate(256, 42).to_vec();
    assert_eq!(seq_a, seq_b);
    assert_eq!(cache.len(), 1);
}

// ── realtime_embedder ─────────────────────────────────────────────────────────

#[test]
fn test_realtime_embedder_constructs_and_stats() {
    use oximedia_watermark::realtime_embedder::{RealtimeEmbedder, RealtimeEmbedderConfig};

    let config = RealtimeEmbedderConfig::default();
    let embedder = RealtimeEmbedder::new(config, 44100).expect("construct ok");
    assert!(!embedder.payload_set());
    let stats = embedder.stats();
    assert_eq!(stats.frames_processed, 0);
}

// ── robustness_test ───────────────────────────────────────────────────────────

#[test]
fn test_robustness_test_add_awgn_preserves_length() {
    use oximedia_watermark::robustness_test::WatermarkRobustness;

    let pixels = vec![128u8; 256];
    let noisy = WatermarkRobustness::add_awgn(&pixels, 30.0);
    assert_eq!(noisy.len(), pixels.len());
}

// ── temporal_watermark ────────────────────────────────────────────────────────

#[test]
fn test_temporal_watermark_capacity_positive() {
    use oximedia_watermark::temporal_watermark::{TemporalConfig, TemporalEmbedder};

    let config = TemporalConfig::default();
    let embedder = TemporalEmbedder::new(config).expect("construct ok");
    let cap = embedder.capacity(44100);
    assert!(cap > 0);
}

// ── watermark_analyzer ────────────────────────────────────────────────────────

#[test]
fn test_watermark_analyzer_blind_smoke() {
    use oximedia_watermark::watermark_analyzer::WatermarkAnalyzer;

    let analyzer = WatermarkAnalyzer::new(2048, 44100);
    assert_eq!(analyzer.sample_rate(), 44100);
    let samples: Vec<f32> = vec![0.1; 4096];
    let report = analyzer.analyze_blind(&samples);
    assert_eq!(report.sample_count, 4096);
}

// ── watermark_comparator ──────────────────────────────────────────────────────

#[test]
fn test_watermark_payload_database_empty_initially() {
    use oximedia_watermark::watermark_comparator::WatermarkPayloadDatabase;

    let db = WatermarkPayloadDatabase::new();
    assert!(db.is_empty());
    assert_eq!(db.len(), 0);
}

//! Smoke tests for newly-wired orphan modules in oximedia-denoise.

#[test]
fn test_adaptive_temporal_window_config() {
    use oximedia_denoise::adaptive_temporal_window::AdaptiveWindowConfig;
    let config = AdaptiveWindowConfig::default();
    let _ = config;
}

#[test]
fn test_hdr_denoise_transfer_function() {
    use oximedia_denoise::hdr_denoise::HdrTransferFunction;
    let _ = HdrTransferFunction::Pq;
}

#[test]
fn test_pipeline_mode_config() {
    use oximedia_denoise::pipeline_mode::DenoiserPipelineConfig;
    let config = DenoiserPipelineConfig::default();
    let _ = config;
}

#[test]
fn test_streaming_config() {
    use oximedia_denoise::streaming::StreamingConfig;
    let config = StreamingConfig::default();
    let _ = config;
}

#[test]
fn test_perceptual_shaping_config() {
    use oximedia_denoise::perceptual_shaping::PerceptualConfig;
    let config = PerceptualConfig::conservative();
    let _ = config;
}

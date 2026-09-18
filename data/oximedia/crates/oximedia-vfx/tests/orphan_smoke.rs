//! Smoke tests for newly-wired orphan modules in oximedia-vfx.

#[test]
fn test_audio_reactive_frequency_bands() {
    use oximedia_vfx::audio_reactive::FrequencyBand;
    let bands = FrequencyBand::all_bands();
    assert!(!bands.is_empty());
}

#[test]
fn test_color_management_white_point() {
    use oximedia_vfx::color_management::WhitePoint;
    let _ = WhitePoint::D65;
}

#[test]
fn test_draft_preview_downsampled_size() {
    use oximedia_vfx::draft_preview::DownsampleFactor;
    let (w, h) = DownsampleFactor::Half.downsampled_size(1920, 1080);
    assert_eq!(w, 960);
    assert_eq!(h, 540);
}

#[test]
fn test_gpu_backend_availability() {
    use oximedia_vfx::gpu_backend::GpuAvailability;
    let _ = GpuAvailability::NotAvailable;
}

#[test]
fn test_grain_film_grain_new() {
    use oximedia_vfx::grain::FilmGrain;
    let grain = FilmGrain::new(0.3, 42);
    assert!((grain.strength() - 0.3).abs() < 1e-6);
}

#[test]
fn test_param_track_cache_construct() {
    use oximedia_vfx::param_track_cache::CachedParameterTrack;
    use oximedia_vfx::ParameterTrack;
    let track = ParameterTrack::new();
    let cached = CachedParameterTrack::new(track);
    let _ = cached;
}

#[test]
fn test_retro_film_preset() {
    use oximedia_vfx::retro_film::RetroFilmPreset;
    let config = RetroFilmPreset::Vhs.to_config();
    let _ = config;
}

#[test]
fn test_tilt_shift_config() {
    use oximedia_vfx::tilt_shift::TiltShiftConfig;
    let config = TiltShiftConfig::default();
    let _ = config;
}

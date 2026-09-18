//! Smoke tests for newly-wired orphan modules in oximedia-scopes.

#[test]
fn test_audio_phase_meter_new() {
    use oximedia_scopes::audio_phase_meter::PhaseCorrelationMeter;
    let meter = PhaseCorrelationMeter::new(48000, 0.3);
    let _ = meter;
}

#[test]
fn test_cie_xy_point() {
    use oximedia_scopes::cie_xy_diagram::CieXyPoint;
    let pt = CieXyPoint::new(0.3127, 0.3290, 1.0);
    assert!((pt.x - 0.3127).abs() < 1e-6);
}

#[test]
fn test_color_checker_scope_linear_rgb() {
    use oximedia_scopes::color_checker_scope::LinearRgb;
    let c = LinearRgb::from_srgb8(128, 128, 128);
    assert!(c.r > 0.0 && c.r < 1.0);
}

#[test]
fn test_gamut_scope_overlay_line_style() {
    use oximedia_scopes::gamut_scope_overlay::OverlayLineStyle;
    let style = OverlayLineStyle::new([255, 255, 0, 255], 2);
    assert_eq!(style.thickness, 2);
}

#[test]
fn test_luma_parade_channel_rgba() {
    use oximedia_scopes::luma_parade::ParadeChannel;
    let color = ParadeChannel::Luma.rgba_color();
    assert_eq!(color.len(), 4);
}

#[test]
fn test_noise_floor_config() {
    use oximedia_scopes::noise_floor_scope::NoiseFloorConfig;
    let config = NoiseFloorConfig::default();
    let _ = config;
}

#[test]
fn test_rgb_parade_config() {
    use oximedia_scopes::rgb_parade::RgbParadeConfig;
    let config = RgbParadeConfig::default();
    let _ = config;
}

#[test]
fn test_scope_snapshot_store_type() {
    use oximedia_scopes::scope_snapshot_store::ScopeType;
    let _ = ScopeType::Waveform;
}

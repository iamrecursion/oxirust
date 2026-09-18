//! Smoke tests for newly-wired orphan modules in oximedia-presets.
#[test]
fn test_film_grain_presets_count() {
    use oximedia_presets::film_grain::all_presets;
    assert!(!all_presets().is_empty());
}
#[test]
fn test_preset_api_matcher() {
    let _ = std::any::type_name::<oximedia_presets::preset_api::PresetMatcher>();
}
#[test]
fn test_preset_compatibility_report() {
    let _ = std::any::type_name::<oximedia_presets::preset_compatibility::CompatibilityReport>();
}
#[test]
fn test_preset_derived_accessible() {
    let _ = std::any::type_name::<oximedia_presets::preset_derived::DerivedPreset>();
}
#[test]
fn test_preset_recommendation_engine() {
    use oximedia_presets::preset_recommendation::RecommendationEngine;
    use oximedia_presets::PresetLibrary;
    let lib = PresetLibrary::new();
    let engine = RecommendationEngine::from_library(&lib);
    let _ = engine;
}

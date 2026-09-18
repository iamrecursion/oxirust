//! Smoke tests for newly-wired orphan modules in oximedia-analytics.
#[test]
fn test_replay_frame_accessible() {
    let _ = std::any::type_name::<oximedia_analytics::replay::ReplayFrame>();
}
#[test]
fn test_uniformity_result_accessible() {
    let _ = std::any::type_name::<oximedia_analytics::uniformity::UniformityResult>();
}

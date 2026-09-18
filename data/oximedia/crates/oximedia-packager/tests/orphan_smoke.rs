//! Smoke tests for newly-wired orphan modules in oximedia-packager.
#[test]
fn test_metadata_cache_accessible() {
    let _ = std::any::type_name::<oximedia_packager::metadata_cache::MetadataCache>();
}
#[test]
fn test_pre_roll_config() {
    let _ = std::any::type_name::<oximedia_packager::pre_roll::PreRollConfig>();
}
#[test]
fn test_ssai_ad_break_type() {
    use oximedia_packager::ssai::AdBreakType;
    let _ = AdBreakType::PreRoll;
}
#[test]
fn test_streaming_output_config() {
    let _ = std::any::type_name::<oximedia_packager::streaming_output::SegmentStreamConfig>();
}

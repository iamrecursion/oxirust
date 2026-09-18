//! Smoke tests for newly-wired orphan modules in oximedia-conform.
#[test]
fn test_clip_resolver_accessible() {
    let _ = std::any::type_name::<oximedia_conform::clip_resolver::ClipResolver>();
}
#[test]
fn test_conform_enhancements_xml_clip() {
    let _ = std::any::type_name::<oximedia_conform::conform_enhancements::XmlClip>();
}
#[test]
fn test_conform_snapshot_accessible() {
    let _ = std::any::type_name::<oximedia_conform::conform_snapshot::ConformSnapshot>();
}

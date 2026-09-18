//! Smoke tests for newly-wired orphan modules in oximedia-automation.

#[test]
fn test_ad_insertion_ad_avail() {
    let _ = std::any::type_name::<oximedia_automation::ad_insertion::AdAvail>();
}
#[test]
fn test_equipment_inventory_status() {
    use oximedia_automation::equipment_inventory::EquipmentStatus;
    let _ = EquipmentStatus::Operational;
}
#[test]
fn test_gap_detector_fill_strategy() {
    use oximedia_automation::gap_detector::FillStrategy;
    let _ = FillStrategy::FillerClip;
}
#[test]
fn test_graphics_overlay_kind() {
    use oximedia_automation::graphics_overlay::OverlayKind;
    let _ = OverlayKind::LowerThird;
}
#[test]
fn test_multi_site_site_status() {
    use oximedia_automation::multi_site::SiteStatus;
    let _ = SiteStatus::Online;
}
#[test]
fn test_operator_journal_action_kind() {
    let _ = std::any::type_name::<oximedia_automation::operator_journal::OperatorActionKind>();
}
#[test]
fn test_regulatory_compliance_content_rating() {
    let _ = std::any::type_name::<oximedia_automation::regulatory_compliance::ContentRating>();
}
#[test]
fn test_timecode_scheduler_action_kind() {
    let _ = std::any::type_name::<oximedia_automation::timecode_scheduler::TimecodeActionKind>();
}
#[test]
fn test_timecode_sync_standard() {
    use oximedia_automation::timecode_sync::TimecodeStandard;
    let _ = TimecodeStandard::Ltc30;
}

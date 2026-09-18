//! Smoke tests for newly-wired orphan modules in oximedia-edl.
#[test]
fn test_diff_module_accessible() {
    let _ = std::any::type_name::<oximedia_edl::diff::EdlChange>();
}
#[test]
fn test_filter_module_accessible() {
    let _ = std::any::type_name::<oximedia_edl::filter::EdlFilter>();
}
#[test]
fn test_tc_list_exporter_accessible() {
    let _ = std::any::type_name::<oximedia_edl::tc_list::TcListExporter>();
}
#[test]
fn test_to_timeline_accessible() {
    let _ = std::any::type_name::<oximedia_edl::to_timeline::Timeline>();
}

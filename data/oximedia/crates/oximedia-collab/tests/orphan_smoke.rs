//! Smoke tests for newly-wired orphan modules in oximedia-collab.

#[test]
fn change_feed_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_collab::change_feed));
}

#[test]
fn collab_bus_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_collab::collab_bus));
}

#[test]
fn conflict_resolution_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_collab::conflict_resolution));
}

#[test]
fn conflict_resolver_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_collab::conflict_resolver));
}

#[test]
fn cursor_interpolation_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_collab::cursor_interpolation));
}

#[test]
fn cursor_sharing_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_collab::cursor_sharing));
}

#[test]
fn dag_index_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_collab::dag_index));
}

#[test]
fn export_coordinator_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_collab::export_coordinator));
}

#[test]
fn history_replay_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_collab::history_replay));
}

#[test]
fn incremental_sync_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_collab::incremental_sync));
}

#[test]
fn lock_escalation_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_collab::lock_escalation));
}

#[test]
fn op_batcher_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_collab::op_batcher));
}

#[test]
fn opt_lock_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_collab::opt_lock));
}

#[test]
fn session_lifecycle_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_collab::session_lifecycle));
}

#[test]
fn session_recording_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_collab::session_recording));
}

#[test]
fn timeline_collab_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_collab::timeline_collab));
}

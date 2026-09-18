//! Smoke tests for newly-wired orphan modules in oximedia-plugin.

#[test]
fn capability_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_plugin::capability));
}

#[test]
fn config_persist_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_plugin::config_persist));
}

#[test]
fn config_persistence_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_plugin::config_persistence));
}

#[test]
fn filter_plugin_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_plugin::filter_plugin));
}

#[test]
fn graceful_reload_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_plugin::graceful_reload));
}

#[test]
fn harness_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_plugin::harness));
}

#[test]
fn health_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_plugin::health));
}

#[test]
fn health_check_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_plugin::health_check));
}

#[test]
fn health_monitor_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_plugin::health_monitor));
}

#[test]
fn lazy_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_plugin::lazy));
}

#[test]
fn lazy_init_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_plugin::lazy_init));
}

#[test]
fn plugin_config_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_plugin::plugin_config));
}

#[test]
fn plugin_telemetry_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_plugin::plugin_telemetry));
}

#[test]
fn pool_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_plugin::pool));
}

#[test]
fn priority_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_plugin::priority));
}

#[test]
fn resources_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_plugin::resources));
}

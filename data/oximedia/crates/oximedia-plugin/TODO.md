# oximedia-plugin TODO

## Current Status
- 24 modules: capability, config_persist, config_persistence, error, filter_plugin, graceful_reload, harness, health, health_check, health_monitor, hot_reload, lazy, lazy_init, manifest, plugin_config, plugin_telemetry, pool, priority, registry, resources, sandbox, static_plugin, traits, version_resolver, loader (feature-gated)
- Supports static and dynamic plugin registration via `CodecPlugin` trait
- Features: hot reload with file watching, plugin manifest with dependency resolution, sandboxing with permission sets, version constraint solving
- Filter/transform plugin type (`FilterPlugin` trait + `FilterRegistry`)
- Health check monitoring (`PluginHealthMonitor` with periodic liveness probes, history window, degradation detection)
- Configuration persistence (`PluginConfigStore` — typed JSON-backed key-value store per plugin)
- Feature gate: `dynamic-loading` (libloading) for shared library loading
- 271+ tests passing (160 unit + 33 fuzz_manifest + 9 hot_reload + 16 integration + 37 sandbox + 13 version_graph + 3 doc)

## Enhancements
- [x] Add plugin priority/ordering in `PluginRegistry` for codec conflict resolution (multiple plugins for same codec) (verified 2026-05-16; src/priority.rs:47 PluginPriority(u32), lower value = higher precedence:35)
- [x] Implement plugin health checks in `hot_reload::HotReloadManager` (periodic liveness probe) (verified 2026-05-16; src/health_monitor.rs PluginHealthMonitor, src/health_check.rs periodic liveness probes)
- [x] Extend `sandbox::PermissionSet` with fine-grained filesystem path restrictions (not just PERM_FILESYSTEM) (verified 2026-05-16; src/sandbox.rs:57 path allow-list, add_path:119, is_path_permitted:148)
- [x] Add plugin resource usage tracking (memory, CPU time) in `SandboxContext` (verified 2026-05-16; src/resources.rs:31 ResourceUsage cpu_time_ms:35, ResourceTracker, ResourceLimit)
- [x] Implement plugin dependency conflict detection in `version_resolver` (diamond dependency problem) (verified 2026-06-01: register_with_deps+provider_deps+BFS transitive propagation in version_resolver.rs)
- [x] Add graceful degradation in `GracefulReload` — serve from old plugin during new plugin initialization (verified 2026-05-16; src/graceful_reload.rs:80 serves old plugin during init, InProgress state:35)
- [x] Extend `PluginManifest` with minimum OxiMedia version requirement field (verified 2026-05-16; src/manifest.rs:399 min_host_version: Option<String>)

## New Features
- [ ] Add WASM plugin support — load plugins compiled to WebAssembly via wasmtime/wasmer (verified-open 2026-05-16: no wasmtime/wasmer in plugin crate)
- [ ] Implement plugin marketplace protocol — discovery, download, and verification of remote plugins (verified-open 2026-05-16: no marketplace/remote discovery module)
- [x] Add plugin configuration persistence — save/load plugin settings between sessions (verified 2026-05-16; src/config_persistence.rs:598 PluginConfigStore JSON-backed key-value store)
- [x] Implement plugin telemetry collection — anonymous usage stats for plugin authors (verified 2026-05-16; src/plugin_telemetry.rs:747 lines PluginTelemetry)
- [x] Add plugin test harness — standardized testing framework for plugin developers (verified 2026-05-16; src/harness.rs:589 lines plugin test harness)
- [ ] Implement plugin isolation via process sandboxing (run plugin in subprocess with IPC) (verified-open 2026-05-16: sandbox.rs uses bitmask/path restrictions, not subprocess IPC isolation)
- [x] Add filter/transform plugin type alongside codec plugins (video/audio filter plugins) (verified 2026-05-16; src/filter_plugin.rs:442 FilterPlugin trait, FilterRegistry)

## Performance
- [x] Cache plugin capability lookups in `PluginRegistry` with invalidation on register/unregister (verified 2026-06-01: registry.rs:44 CapabilityCache struct with invalidate():59, rebuild():65, wired at line 584)
- [x] Implement lazy plugin initialization — defer codec creation until first use (verified 2026-05-16; src/lazy_init.rs:320 lazy plugin init, src/lazy.rs lazy loading)
- [x] Add plugin instance pooling for codecs that are expensive to initialize (verified 2026-05-16; src/pool.rs:378 plugin instance pool)
- [x] Optimize `compute_hash` in hot_reload to use memory-mapped I/O for large plugin files — `compute_hash_mmap` with `MMAP_THRESHOLD_BYTES` (4 MiB) page-streaming strategy; 9 new tests

## Testing
- [x] Add integration test for full plugin lifecycle (register -> lookup -> use -> unregister) — 10 new tests in `tests/integration.rs` covering priority ordering, failover, clear, re-registration
- [x] Test `hot_reload` with simulated file modification events and verify seamless reload (verified 2026-06-01: tests/hot_reload_test.rs:29 test_hot_reload_seamless_lifecycle)
- [x] Add fuzz testing for `PluginManifest` parsing with malformed JSON/TOML (verified 2026-06-01: tests/fuzz_manifest.rs 33 tests; tests/manifest_fuzz.rs 5 additional injection/large-input tests)
- [x] Test `sandbox` permission enforcement — verify blocked operations raise `SandboxError` — 13 new tests in `tests/sandbox_test.rs` covering path allow-list, CPU quota, combined enforcement
- [x] Add tests for `version_resolver` with complex dependency graphs (10+ interdependent plugins) (verified 2026-06-01: version_resolver.rs test_large_dep_graph_chain+test_diamond_conflict_transitive+test_diamond_compatible_transitive; tests/version_graph_test.rs 13 tests)

## Documentation
- [ ] Add plugin development guide with step-by-step shared library plugin creation
- [ ] Document `declare_plugin!` macro usage with complete working example
- [ ] Add security model documentation for sandbox permissions and trust levels

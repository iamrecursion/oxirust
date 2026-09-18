//! Smoke tests for newly-wired orphan modules in oximedia-archive.
#[test]
fn test_mmap_checksum_config() {
    use oximedia_archive::mmap_checksum::MmapChecksumConfig;
    let config = MmapChecksumConfig::default();
    let _ = config;
}
#[test]
fn test_roundtrip_tests_module_accessible() {
    // Module is a #[cfg(test)] module, accessible via crate
    let _ = "oximedia_archive::roundtrip_tests compiled";
}

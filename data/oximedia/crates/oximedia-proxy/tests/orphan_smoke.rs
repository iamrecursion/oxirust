//! Smoke tests verifying newly wired proxy orphan modules compile and expose
//! at least one public item from each module.

use oximedia_proxy::{
    cleanup::ProxyCleanupScheduler,
    cloud_proxy::{CloudProxyManager, DispatchStrategy},
    frame_map::ProxyFrameMap,
    nle_format_select::{detect_nle_from_app_name, Nle, NleFormatSelector},
    proxy_api::{ProxySpec, SmartProxyGenerator},
    proxy_audit::{AuditAction, AuditLog},
    proxy_checksum::{checksum, ChecksumAlgo},
    proxy_pool::ProxyWorkerPool,
    proxy_streaming::{ProxyStreamingServer, StreamProtocol},
    tiers::{ProxyTier, ProxyTierSelector},
};

#[test]
fn test_cleanup_scheduler_is_expired() {
    let scheduler = ProxyCleanupScheduler::new(30); // 30-day max age
                                                    // A proxy from 31 days ago should be expired.
    let now = 100_000_000u64; // large enough so we can subtract 31 days
    let old = now - 31 * 86_400u64;
    assert!(scheduler.is_expired(old, now));
    // A proxy from 1 day ago should not be expired.
    let recent = now - 86_400;
    assert!(!scheduler.is_expired(recent, now));
}

#[test]
fn test_cloud_proxy_manager_new() {
    let manager = CloudProxyManager::new(DispatchStrategy::LeastLoaded);
    // Manager should start with no workers.
    let _ = manager;
}

#[test]
fn test_proxy_frame_map_identity() {
    let map = ProxyFrameMap::new(30.0, 30.0);
    assert!(map.is_identity());
    assert_eq!(map.proxy_frame_to_original(10), 10);
}

#[test]
fn test_proxy_frame_map_ratio() {
    // Proxy at 15 fps, original at 30 fps → 2× ratio.
    let map = ProxyFrameMap::new(15.0, 30.0);
    assert!(!map.is_identity());
    assert_eq!(map.proxy_frame_to_original(5), 10);
}

#[test]
fn test_detect_nle_from_app_name() {
    let nle = detect_nle_from_app_name("Resolve");
    assert_eq!(nle, Nle::DaVinciResolve);
}

#[test]
fn test_nle_format_selector_new() {
    let _selector = NleFormatSelector::new();
}

#[test]
fn test_smart_proxy_generator_new() {
    let _gen = SmartProxyGenerator;
}

#[test]
fn test_proxy_spec_new() {
    let spec = ProxySpec::new(1920, 1080, "h264", 5_000);
    assert_eq!(spec.width, 1920);
    assert_eq!(spec.height, 1080);
}

#[test]
fn test_audit_log_new() {
    let log = AuditLog::new();
    assert_eq!(log.len(), 0);
}

#[test]
fn test_audit_action_enum() {
    let action = AuditAction::Created;
    assert_eq!(action, AuditAction::Created);
}

#[test]
fn test_checksum_fnv1a64() {
    let data = b"hello world";
    let h1 = checksum(ChecksumAlgo::Fnv1a64, data);
    let h2 = checksum(ChecksumAlgo::Fnv1a64, data);
    // Deterministic hash.
    assert_eq!(h1, h2);
}

#[test]
fn test_checksum_crc32() {
    let data = b"oximedia proxy checksum test";
    let h = checksum(ChecksumAlgo::Crc32, data);
    assert_ne!(h, 0);
}

#[test]
fn test_proxy_worker_pool_new() {
    let pool = ProxyWorkerPool::new(4);
    assert_eq!(pool.worker_count(), 4);
    assert_eq!(pool.idle_count(), 4);
}

#[test]
fn test_proxy_streaming_server_new() {
    let server = ProxyStreamingServer::new();
    let _ = server;
}

#[test]
fn test_stream_protocol_http1() {
    let proto = StreamProtocol::Http1;
    assert_eq!(proto, StreamProtocol::Http1);
}

#[test]
fn test_proxy_tier_selector_new() {
    let selector = ProxyTierSelector::new();
    // Tier selector starts empty.
    assert_eq!(selector.tiers().len(), 0);
}

#[test]
fn test_proxy_tier_selector_standard() {
    let selector = ProxyTierSelector::standard();
    assert!(!selector.tiers().is_empty());
}

#[test]
fn test_proxy_tier_new() {
    let tier = ProxyTier::new("quarter", 480, 500);
    assert_eq!(tier.max_width, 480);
}

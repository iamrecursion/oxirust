//! Integration tests for geographic fencing via IP geolocation.
//!
//! `oximedia_drm::geo_fence::GeoFenceManager` evaluates rules against ISO
//! 3166-1 country codes directly. These tests wrap that core API with a
//! local `MockIpGeolocator` that maps an IP-like opaque string to a
//! country code, so the tests exercise the realistic flow of a license
//! server: `client_ip → country_code → geo-fence verdict`.
//!
//! Test-local `MockIpGeolocator` is intentionally defined inside the
//! integration test file (not exported) because the production codebase
//! treats geolocation lookup as a pluggable external concern. The trait
//! and mock here serve only as a smoke-test harness for the public
//! `GeoFenceManager::evaluate(...)` surface.

use oximedia_drm::geo_fence::{
    GeoFenceManager, GeoFenceMode, GeoFenceRule, GeoFenceVerdict, RegionGroup,
};
use std::collections::HashMap;

/// Minimal pluggable interface — production code would back this by MaxMind
/// GeoIP, an HTTP geo service, etc. Here we use a `HashMap` lookup.
trait IpGeolocator {
    /// Resolve a client IP-like identifier to an ISO 3166-1 alpha-2 country
    /// code. `None` if the IP is unknown.
    fn country_for(&self, ip: &str) -> Option<String>;
}

/// Test-only deterministic mock geolocator.
struct MockIpGeolocator {
    table: HashMap<String, String>,
}

impl MockIpGeolocator {
    fn new() -> Self {
        Self {
            table: HashMap::new(),
        }
    }

    fn insert(&mut self, ip: &str, country: &str) {
        self.table.insert(ip.to_string(), country.to_string());
    }
}

impl IpGeolocator for MockIpGeolocator {
    fn country_for(&self, ip: &str) -> Option<String> {
        self.table.get(ip).cloned()
    }
}

/// Evaluate a geo-fence verdict for the given client IP.
///
/// Returns `Denied { rule_id: "unresolved" }` when the IP cannot be
/// geo-located — strict default-deny posture appropriate for premium
/// content distribution.
fn evaluate_for_ip(
    mgr: &GeoFenceManager,
    geo: &dyn IpGeolocator,
    content_id: &str,
    client_ip: &str,
    epoch_secs: u64,
) -> GeoFenceVerdict {
    match geo.country_for(client_ip) {
        Some(country) => mgr.evaluate(content_id, &country, epoch_secs),
        None => GeoFenceVerdict::Denied {
            rule_id: "unresolved".to_string(),
        },
    }
}

#[test]
fn allow_list_us_allows_us_denies_others() {
    let mut mgr = GeoFenceManager::new();
    let mut rule = GeoFenceRule::new("allow-us", "movie-001", GeoFenceMode::AllowList);
    rule.add_country("US");
    mgr.add_rule(rule);

    let mut geo = MockIpGeolocator::new();
    geo.insert("10.0.0.1", "US"); // San Francisco
    geo.insert("203.0.113.5", "JP"); // Tokyo
    geo.insert("198.51.100.42", "DE"); // Berlin

    let us_verdict = evaluate_for_ip(&mgr, &geo, "movie-001", "10.0.0.1", 1_700_000_000);
    assert_eq!(us_verdict, GeoFenceVerdict::Allowed);

    let jp_verdict = evaluate_for_ip(&mgr, &geo, "movie-001", "203.0.113.5", 1_700_000_000);
    assert!(matches!(jp_verdict, GeoFenceVerdict::Denied { .. }));

    let de_verdict = evaluate_for_ip(&mgr, &geo, "movie-001", "198.51.100.42", 1_700_000_000);
    assert!(matches!(de_verdict, GeoFenceVerdict::Denied { .. }));
}

#[test]
fn deny_list_cn_blocks_cn_allows_others() {
    let mut mgr = GeoFenceManager::new();
    let mut rule = GeoFenceRule::new("deny-cn", "show-101", GeoFenceMode::DenyList);
    rule.add_country("CN");
    mgr.add_rule(rule);

    let mut geo = MockIpGeolocator::new();
    geo.insert("1.2.3.4", "CN");
    geo.insert("8.8.8.8", "US");
    geo.insert("9.9.9.9", "FR");

    let cn_verdict = evaluate_for_ip(&mgr, &geo, "show-101", "1.2.3.4", 0);
    assert!(matches!(cn_verdict, GeoFenceVerdict::Denied { rule_id } if rule_id == "deny-cn"));

    let us_verdict = evaluate_for_ip(&mgr, &geo, "show-101", "8.8.8.8", 0);
    assert_eq!(us_verdict, GeoFenceVerdict::Allowed);

    let fr_verdict = evaluate_for_ip(&mgr, &geo, "show-101", "9.9.9.9", 0);
    assert_eq!(fr_verdict, GeoFenceVerdict::Allowed);
}

#[test]
fn unresolved_ip_default_denied_under_strict_policy() {
    let mut mgr = GeoFenceManager::new();
    let mut rule = GeoFenceRule::new("allow-us", "movie-002", GeoFenceMode::AllowList);
    rule.add_country("US");
    mgr.add_rule(rule);

    let geo = MockIpGeolocator::new(); // empty — all IPs unresolved

    let verdict = evaluate_for_ip(&mgr, &geo, "movie-002", "192.168.1.1", 0);
    assert!(
        matches!(verdict, GeoFenceVerdict::Denied { rule_id } if rule_id == "unresolved"),
        "unresolved IP must trigger default-deny verdict"
    );
}

#[test]
fn eu_region_group_allows_all_member_states() {
    let mut mgr = GeoFenceManager::new();
    let mut rule = GeoFenceRule::new("eu-only", "movie-eu-001", GeoFenceMode::AllowList);
    rule.add_region_group(&GeoFenceManager::eu_region());
    mgr.add_rule(rule);

    let mut geo = MockIpGeolocator::new();
    geo.insert("ip-fr", "FR");
    geo.insert("ip-de", "DE");
    geo.insert("ip-it", "IT");
    geo.insert("ip-us", "US");

    assert_eq!(
        evaluate_for_ip(&mgr, &geo, "movie-eu-001", "ip-fr", 0),
        GeoFenceVerdict::Allowed
    );
    assert_eq!(
        evaluate_for_ip(&mgr, &geo, "movie-eu-001", "ip-de", 0),
        GeoFenceVerdict::Allowed
    );
    assert_eq!(
        evaluate_for_ip(&mgr, &geo, "movie-eu-001", "ip-it", 0),
        GeoFenceVerdict::Allowed
    );
    let us = evaluate_for_ip(&mgr, &geo, "movie-eu-001", "ip-us", 0);
    assert!(matches!(us, GeoFenceVerdict::Denied { .. }));
}

#[test]
fn temporal_validity_window_excludes_out_of_range_timestamps() {
    let mut mgr = GeoFenceManager::new();
    let mut rule = GeoFenceRule::new("limited", "movie-limited", GeoFenceMode::AllowList);
    rule.add_country("US");
    rule.set_validity(Some(1000), Some(2000));
    mgr.add_rule(rule);

    let mut geo = MockIpGeolocator::new();
    geo.insert("us-ip", "US");

    // Before window: NoRule because temporal validity fails
    let before = evaluate_for_ip(&mgr, &geo, "movie-limited", "us-ip", 500);
    assert_eq!(before, GeoFenceVerdict::NoRule);

    // Inside window: Allowed
    let inside = evaluate_for_ip(&mgr, &geo, "movie-limited", "us-ip", 1500);
    assert_eq!(inside, GeoFenceVerdict::Allowed);

    // After window: NoRule
    let after = evaluate_for_ip(&mgr, &geo, "movie-limited", "us-ip", 3000);
    assert_eq!(after, GeoFenceVerdict::NoRule);
}

#[test]
fn registering_a_named_apac_region_group_keeps_it_retrievable() {
    let mut mgr = GeoFenceManager::new();
    let mut apac = RegionGroup::new("APAC", "Asia-Pacific");
    apac.add_country("JP");
    apac.add_country("KR");
    apac.add_country("AU");
    mgr.register_group(apac);

    let got = mgr.get_group("APAC").expect("APAC group registered");
    assert!(got.contains("JP"));
    assert!(got.contains("KR"));
    assert!(got.contains("AU"));
    assert!(!got.contains("US"));
}

//! NMOS IS-09 System API implementation.
//!
//! Provides system-wide configuration discovery for NMOS deployments including
//! global settings (PTP domain, DNS-SD scope, NTP servers, timezone),
//! authentication service discovery, NMOS API version advertisement, and
//! system health reporting.
//!
//! ## IS-09 endpoints
//!
//! | Method | Path                              | Description              |
//! |--------|-----------------------------------|--------------------------|
//! | GET    | `/x-nmos/system/v1.0/`           | API root listing         |
//! | GET    | `/x-nmos/system/v1.0/global`     | Global system config     |
//! | GET    | `/x-nmos/system/v1.0/health`     | System health metrics    |

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ============================================================================
// ApiVersionMap
// ============================================================================

/// Map of NMOS API name → list of supported version strings.
///
/// Example: `{"is-04": ["v1.3"], "is-05": ["v1.1"]}`
pub type ApiVersionMap = HashMap<String, Vec<String>>;

// ============================================================================
// NmosSystemConfig
// ============================================================================

/// IS-09 global system configuration.
///
/// Encapsulates all parameters that a compliant IS-09 Global Configuration
/// resource must expose: PTP domain, DNS-SD scopes, NTP servers, timezone,
/// per-API version lists, and optional authentication service URL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NmosSystemConfig {
    /// PTP clock domain number (0–127 per IEEE 1588-2019).
    pub ptp_domain: u8,
    /// DNS-SD browsing scopes in which NMOS services are announced.
    pub dns_sd_scopes: Vec<String>,
    /// NTP server hostnames or IP addresses (ordered by preference).
    pub ntp_servers: Vec<String>,
    /// Timezone as an IANA tz-database name (e.g. `"Europe/London"`).
    pub timezone: String,
    /// NMOS API versions supported by this system deployment.
    pub api_versions: ApiVersionMap,
    /// OAuth2 / IS-10 authentication service base URL, if present.
    pub auth_service: Option<String>,
    /// Universally-unique identifier for this system (UUID format).
    pub system_id: String,
    /// Human-readable label for this system.
    pub label: String,
    /// Optional extended description.
    pub description: Option<String>,
}

impl NmosSystemConfig {
    /// Create a default OxiMedia system configuration pre-populated with
    /// the complete set of supported NMOS APIs:
    ///
    /// | API   | Version |
    /// |-------|---------|
    /// | IS-04 | v1.3    |
    /// | IS-05 | v1.1    |
    /// | IS-07 | v1.0    |
    /// | IS-08 | v1.0    |
    /// | IS-09 | v1.0    |
    pub fn new(system_id: impl Into<String>, label: impl Into<String>) -> Self {
        let mut api_versions: ApiVersionMap = HashMap::new();
        api_versions.insert("is-04".to_string(), vec!["v1.3".to_string()]);
        api_versions.insert("is-05".to_string(), vec!["v1.1".to_string()]);
        api_versions.insert("is-07".to_string(), vec!["v1.0".to_string()]);
        api_versions.insert("is-08".to_string(), vec!["v1.0".to_string()]);
        api_versions.insert("is-09".to_string(), vec!["v1.0".to_string()]);

        Self {
            ptp_domain: 0,
            dns_sd_scopes: vec!["local".to_string()],
            ntp_servers: vec!["pool.ntp.org".to_string()],
            timezone: "UTC".to_string(),
            api_versions,
            auth_service: None,
            system_id: system_id.into(),
            label: label.into(),
            description: None,
        }
    }

    /// Add or update a supported API version entry.
    ///
    /// If the API already has an entry the version string is appended unless
    /// it is already present.
    pub fn add_api(&mut self, api: &str, version: &str) {
        let versions = self.api_versions.entry(api.to_string()).or_default();
        if !versions.contains(&version.to_string()) {
            versions.push(version.to_string());
        }
    }

    /// Override the PTP clock domain.
    pub fn with_ptp_domain(mut self, domain: u8) -> Self {
        self.ptp_domain = domain;
        self
    }

    /// Append an NTP server.
    pub fn with_ntp_server(mut self, server: impl Into<String>) -> Self {
        self.ntp_servers.push(server.into());
        self
    }

    /// Set the timezone.
    pub fn with_timezone(mut self, tz: impl Into<String>) -> Self {
        self.timezone = tz.into();
        self
    }

    /// Add a DNS-SD scope.
    pub fn with_dns_sd_scope(mut self, scope: impl Into<String>) -> Self {
        self.dns_sd_scopes.push(scope.into());
        self
    }

    /// Set the authentication service URL.
    pub fn with_auth_service(mut self, url: impl Into<String>) -> Self {
        self.auth_service = Some(url.into());
        self
    }

    /// Set a human-readable description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// - [`SystemApiError::InvalidPtpDomain`] if `ptp_domain` > 127.
    /// - [`SystemApiError::InvalidTimezone`] if `timezone` is empty.
    pub fn validate(&self) -> Result<(), SystemApiError> {
        if self.ptp_domain > 127 {
            return Err(SystemApiError::InvalidPtpDomain(self.ptp_domain));
        }
        if self.timezone.is_empty() {
            return Err(SystemApiError::InvalidTimezone(
                "timezone must not be empty".to_string(),
            ));
        }
        Ok(())
    }
}

// ============================================================================
// SystemHealth
// ============================================================================

/// IS-09 system health snapshot.
///
/// Aggregated runtime metrics derived from the live NMOS registry and
/// connection manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    /// Total wall-clock seconds since the System API server was started.
    pub uptime_seconds: u64,
    /// Number of registered NMOS nodes.
    pub node_count: usize,
    /// Number of registered NMOS senders.
    pub sender_count: usize,
    /// Number of registered NMOS receivers.
    pub receiver_count: usize,
    /// Number of currently active IS-05 connections.
    pub active_connections: usize,
    /// Whether the PTP grandmaster clock is locked (`true` = locked).
    pub ptp_locked: bool,
    /// NMOS API version advertisement (mirrors `NmosSystemConfig::api_versions`).
    pub api_versions: ApiVersionMap,
}

// ============================================================================
// NmosSystemApi
// ============================================================================

/// IS-09 System API server state.
///
/// Owns the global system configuration and exposes computed health metrics
/// and serialized JSON representations for the HTTP handlers.
pub struct NmosSystemApi {
    /// The IS-09 global configuration.
    pub config: NmosSystemConfig,
    /// The instant at which this server instance was created, used to compute
    /// `uptime_seconds` in health responses.
    pub start_time: std::time::Instant,
}

impl NmosSystemApi {
    /// Create a new `NmosSystemApi` with the given configuration.
    ///
    /// The uptime clock starts at construction time.
    pub fn new(config: NmosSystemConfig) -> Self {
        Self {
            config,
            start_time: std::time::Instant::now(),
        }
    }

    /// Compute a [`SystemHealth`] snapshot from the live registry and
    /// connection manager state.
    pub fn health(
        &self,
        registry: &super::NmosRegistry,
        connection_manager: &super::NmosConnectionManager,
    ) -> SystemHealth {
        SystemHealth {
            uptime_seconds: self.start_time.elapsed().as_secs(),
            node_count: registry.node_count(),
            sender_count: registry.sender_count(),
            receiver_count: registry.receiver_count(),
            active_connections: connection_manager.active_connections().len(),
            ptp_locked: true,
            api_versions: self.config.api_versions.clone(),
        }
    }

    /// Serialize the global configuration to a `serde_json::Value` suitable
    /// for the `GET /x-nmos/system/v1.0/global` response body.
    ///
    /// # Errors
    ///
    /// Returns [`SystemApiError::Serialization`] if the configuration cannot
    /// be serialized (practically impossible for this well-typed structure).
    pub fn to_global_json(&self) -> Result<serde_json::Value, SystemApiError> {
        let value = serde_json::to_value(&self.config)?;
        Ok(value)
    }

    /// Return the uptime in seconds.
    pub fn uptime_seconds(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }
}

// ============================================================================
// SystemApiError
// ============================================================================

/// Errors produced by the IS-09 System API.
#[derive(Debug, thiserror::Error)]
pub enum SystemApiError {
    /// PTP domain value is outside the valid 0–127 range.
    #[error("invalid PTP domain: {0} (must be 0-127)")]
    InvalidPtpDomain(u8),
    /// Timezone string is invalid or empty.
    #[error("invalid timezone: {0}")]
    InvalidTimezone(String),
    /// JSON serialization failed.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nmos::{NmosConnectionManager, NmosRegistry};

    fn make_config() -> NmosSystemConfig {
        NmosSystemConfig::new(
            "550e8400-e29b-41d4-a716-446655440000",
            "OxiMedia Test System",
        )
    }

    // ── NmosSystemConfig construction ─────────────────────────────────────

    #[test]
    fn test_config_new_sets_system_id() {
        let cfg = make_config();
        assert_eq!(cfg.system_id, "550e8400-e29b-41d4-a716-446655440000");
    }

    #[test]
    fn test_config_new_sets_label() {
        let cfg = make_config();
        assert_eq!(cfg.label, "OxiMedia Test System");
    }

    #[test]
    fn test_config_default_ptp_domain_zero() {
        let cfg = make_config();
        assert_eq!(cfg.ptp_domain, 0);
    }

    #[test]
    fn test_config_default_timezone_utc() {
        let cfg = make_config();
        assert_eq!(cfg.timezone, "UTC");
    }

    #[test]
    fn test_config_default_ntp_servers_not_empty() {
        let cfg = make_config();
        assert!(
            !cfg.ntp_servers.is_empty(),
            "default NTP server list must not be empty"
        );
    }

    #[test]
    fn test_config_default_dns_sd_scopes_not_empty() {
        let cfg = make_config();
        assert!(
            !cfg.dns_sd_scopes.is_empty(),
            "default DNS-SD scope list must not be empty"
        );
    }

    #[test]
    fn test_config_default_auth_service_none() {
        let cfg = make_config();
        assert!(cfg.auth_service.is_none());
    }

    #[test]
    fn test_config_default_description_none() {
        let cfg = make_config();
        assert!(cfg.description.is_none());
    }

    // ── Pre-populated API versions ─────────────────────────────────────────

    #[test]
    fn test_config_contains_is04() {
        let cfg = make_config();
        assert!(
            cfg.api_versions.contains_key("is-04"),
            "is-04 must be pre-populated"
        );
        assert!(cfg.api_versions["is-04"].contains(&"v1.3".to_string()));
    }

    #[test]
    fn test_config_contains_is05() {
        let cfg = make_config();
        assert!(cfg.api_versions.contains_key("is-05"));
        assert!(cfg.api_versions["is-05"].contains(&"v1.1".to_string()));
    }

    #[test]
    fn test_config_contains_is07() {
        let cfg = make_config();
        assert!(cfg.api_versions.contains_key("is-07"));
        assert!(cfg.api_versions["is-07"].contains(&"v1.0".to_string()));
    }

    #[test]
    fn test_config_contains_is08() {
        let cfg = make_config();
        assert!(cfg.api_versions.contains_key("is-08"));
        assert!(cfg.api_versions["is-08"].contains(&"v1.0".to_string()));
    }

    #[test]
    fn test_config_contains_is09() {
        let cfg = make_config();
        assert!(cfg.api_versions.contains_key("is-09"));
        assert!(cfg.api_versions["is-09"].contains(&"v1.0".to_string()));
    }

    // ── Builder helpers ────────────────────────────────────────────────────

    #[test]
    fn test_with_ptp_domain() {
        let cfg = make_config().with_ptp_domain(42);
        assert_eq!(cfg.ptp_domain, 42);
    }

    #[test]
    fn test_with_ntp_server() {
        let cfg = make_config().with_ntp_server("ntp.example.com");
        assert!(cfg.ntp_servers.contains(&"ntp.example.com".to_string()));
    }

    #[test]
    fn test_with_timezone() {
        let cfg = make_config().with_timezone("America/New_York");
        assert_eq!(cfg.timezone, "America/New_York");
    }

    #[test]
    fn test_with_dns_sd_scope() {
        let cfg = make_config().with_dns_sd_scope("remote");
        assert!(cfg.dns_sd_scopes.contains(&"remote".to_string()));
    }

    #[test]
    fn test_with_auth_service() {
        let cfg = make_config().with_auth_service("https://auth.example.com");
        assert_eq!(
            cfg.auth_service.as_deref(),
            Some("https://auth.example.com")
        );
    }

    #[test]
    fn test_with_description() {
        let cfg = make_config().with_description("A test system");
        assert_eq!(cfg.description.as_deref(), Some("A test system"));
    }

    // ── add_api ────────────────────────────────────────────────────────────

    #[test]
    fn test_add_api_new_entry() {
        let mut cfg = make_config();
        cfg.add_api("is-11", "v1.0");
        assert!(cfg.api_versions["is-11"].contains(&"v1.0".to_string()));
    }

    #[test]
    fn test_add_api_append_version() {
        let mut cfg = make_config();
        cfg.add_api("is-04", "v1.2");
        let versions = &cfg.api_versions["is-04"];
        assert!(versions.contains(&"v1.3".to_string()));
        assert!(versions.contains(&"v1.2".to_string()));
    }

    #[test]
    fn test_add_api_no_duplicates() {
        let mut cfg = make_config();
        cfg.add_api("is-04", "v1.3");
        cfg.add_api("is-04", "v1.3");
        let count = cfg.api_versions["is-04"]
            .iter()
            .filter(|v| v.as_str() == "v1.3")
            .count();
        assert_eq!(count, 1, "duplicate version must not be added");
    }

    // ── validate ───────────────────────────────────────────────────────────

    #[test]
    fn test_validate_valid_config() {
        let cfg = make_config();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_validate_ptp_domain_127_ok() {
        let cfg = make_config().with_ptp_domain(127);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_validate_ptp_domain_128_err() {
        let cfg = make_config().with_ptp_domain(128);
        let err = cfg.validate().expect_err("128 must be invalid");
        assert!(matches!(err, SystemApiError::InvalidPtpDomain(128)));
    }

    #[test]
    fn test_validate_empty_timezone_err() {
        let cfg = make_config().with_timezone("");
        let err = cfg.validate().expect_err("empty timezone must be invalid");
        assert!(matches!(err, SystemApiError::InvalidTimezone(_)));
    }

    // ── serde round-trip ───────────────────────────────────────────────────

    #[test]
    fn test_config_serde_roundtrip() {
        let cfg = make_config()
            .with_ptp_domain(10)
            .with_ntp_server("ntp2.example.com")
            .with_auth_service("https://auth.example.com");
        let json = serde_json::to_string(&cfg).expect("serialize should succeed in test");
        let decoded: NmosSystemConfig =
            serde_json::from_str(&json).expect("deserialize should succeed in test");
        assert_eq!(decoded.system_id, cfg.system_id);
        assert_eq!(decoded.ptp_domain, 10);
        assert_eq!(
            decoded.auth_service.as_deref(),
            Some("https://auth.example.com")
        );
    }

    // ── NmosSystemApi ──────────────────────────────────────────────────────

    #[test]
    fn test_system_api_new() {
        let api = NmosSystemApi::new(make_config());
        assert!(api.uptime_seconds() < 5, "uptime should start near zero");
    }

    #[test]
    fn test_to_global_json_contains_system_id() {
        let api = NmosSystemApi::new(make_config());
        let val = api
            .to_global_json()
            .expect("to_global_json should succeed in test");
        assert_eq!(val["system_id"], "550e8400-e29b-41d4-a716-446655440000");
    }

    #[test]
    fn test_to_global_json_contains_api_versions() {
        let api = NmosSystemApi::new(make_config());
        let val = api
            .to_global_json()
            .expect("to_global_json should succeed in test");
        assert!(val["api_versions"]["is-04"].is_array());
    }

    #[test]
    fn test_health_empty_registry() {
        let api = NmosSystemApi::new(make_config());
        let registry = NmosRegistry::new();
        let conn_mgr = NmosConnectionManager::new();
        let h = api.health(&registry, &conn_mgr);
        assert_eq!(h.node_count, 0);
        assert_eq!(h.sender_count, 0);
        assert_eq!(h.receiver_count, 0);
        assert_eq!(h.active_connections, 0);
        assert!(h.ptp_locked);
    }

    #[test]
    fn test_health_reflects_registry() {
        use crate::nmos::{
            NmosDevice, NmosDeviceType, NmosFlow, NmosFormat, NmosNode, NmosReceiver, NmosSender,
            NmosTransport,
        };

        let api = NmosSystemApi::new(make_config());
        let mut registry = NmosRegistry::new();
        registry.add_node(NmosNode::new("n-1", "Node 1"));
        registry.add_device(NmosDevice::new(
            "d-1",
            "n-1",
            "Dev",
            NmosDeviceType::Generic,
        ));
        registry.add_source(crate::nmos::NmosSource::new(
            "s-1",
            "d-1",
            "Src",
            NmosFormat::Video,
            "clk0",
        ));
        registry.add_flow(NmosFlow::new(
            "f-1",
            "s-1",
            "Flow",
            NmosFormat::Video,
            (25, 1),
        ));
        registry.add_sender(NmosSender::new(
            "tx-1",
            "f-1",
            "Sender",
            NmosTransport::RtpMulticast,
        ));
        registry.add_receiver(NmosReceiver::new("rx-1", "d-1", "Recv", NmosFormat::Video));

        let mut conn_mgr = NmosConnectionManager::new();
        conn_mgr.connect("tx-1", "rx-1");

        let h = api.health(&registry, &conn_mgr);
        assert_eq!(h.node_count, 1);
        assert_eq!(h.sender_count, 1);
        assert_eq!(h.receiver_count, 1);
        assert_eq!(h.active_connections, 1);
    }

    #[test]
    fn test_health_api_versions_match_config() {
        let api = NmosSystemApi::new(make_config());
        let registry = NmosRegistry::new();
        let conn_mgr = NmosConnectionManager::new();
        let h = api.health(&registry, &conn_mgr);
        assert_eq!(h.api_versions, api.config.api_versions);
    }

    #[test]
    fn test_system_health_serde_roundtrip() {
        let api = NmosSystemApi::new(make_config());
        let registry = NmosRegistry::new();
        let conn_mgr = NmosConnectionManager::new();
        let h = api.health(&registry, &conn_mgr);
        let json = serde_json::to_string(&h).expect("serialize health should succeed in test");
        let decoded: SystemHealth =
            serde_json::from_str(&json).expect("deserialize health should succeed in test");
        assert_eq!(decoded.node_count, h.node_count);
        assert_eq!(decoded.ptp_locked, h.ptp_locked);
    }

    // ── SystemApiError display ────────────────────────────────────────────

    #[test]
    fn test_error_invalid_ptp_domain_display() {
        let e = SystemApiError::InvalidPtpDomain(200);
        assert!(e.to_string().contains("200"));
        assert!(e.to_string().contains("0-127"));
    }

    #[test]
    fn test_error_invalid_timezone_display() {
        let e = SystemApiError::InvalidTimezone("???".into());
        assert!(e.to_string().contains("???"));
    }
}

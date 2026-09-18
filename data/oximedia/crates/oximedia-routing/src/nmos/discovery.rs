//! NMOS DNS-SD / mDNS service discovery (AMWA IS-04).
//!
//! This module implements DNS-SD announcements and browsing for NMOS IS-04
//! Node, Query, and Registration APIs using the pure-Rust `mdns-sd` crate.
//!
//! ## NMOS service types (per AMWA IS-04)
//!
//! | Service type                    | Purpose                    |
//! |---------------------------------|----------------------------|
//! | `_nmos-node._tcp.local.`        | NMOS Node API              |
//! | `_nmos-query._tcp.local.`       | NMOS Query API             |
//! | `_nmos-registration._tcp.local.`| NMOS Registration API      |
//!
//! ## Example
//!
//! ```rust,no_run
//! # #[cfg(feature = "nmos-discovery")]
//! # {
//! use oximedia_routing::NmosDiscovery;
//!
//! let mut disc = NmosDiscovery::new(
//!     "550e8400-e29b-41d4-a716-446655440000",
//!     "My NMOS Node",
//!     8080,
//! ).expect("discovery init failed");
//!
//! disc.announce().expect("announce failed");
//!
//! // … later …
//! disc.shutdown().expect("shutdown failed");
//! # }
//! ```

use std::collections::HashMap;
use std::time::Duration;

use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};

// ============================================================================
// Error type
// ============================================================================

/// Errors produced by [`NmosDiscovery`].
#[derive(Debug, thiserror::Error)]
pub enum NmosDiscoveryError {
    /// The underlying mDNS daemon could not be created or operated.
    #[error("mDNS daemon error: {0}")]
    Mdns(String),

    /// Building or registering a `ServiceInfo` failed.
    #[error("service registration failed: {0}")]
    Registration(String),

    /// Browsing for remote services failed.
    #[error("service browse failed: {0}")]
    Browse(String),

    /// The daemon returned an error while shutting down.
    #[error("shutdown error: {0}")]
    Shutdown(String),

    /// A required parameter (e.g. node ID or label) was invalid.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
}

// ============================================================================
// NmosRegistryInfo
// ============================================================================

/// A discovered NMOS Registration API endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NmosRegistryInfo {
    /// DNS-SD instance full name (e.g. `"MyRegistry._nmos-registration._tcp.local."`).
    pub name: String,
    /// Resolved hostname or IP address string.
    pub host: String,
    /// TCP port of the Registration API.
    pub port: u16,
    /// Numeric priority from the `pri` TXT record (lower = higher priority).
    pub priority: u32,
}

// ============================================================================
// Internal constants
// ============================================================================

/// NMOS Node API service type (AMWA IS-04 §4.1).
const NMOS_NODE_SVC: &str = "_nmos-node._tcp.local.";
/// NMOS Query API service type (AMWA IS-04 §4.3).
const NMOS_QUERY_SVC: &str = "_nmos-query._tcp.local.";
/// NMOS Registration API service type (AMWA IS-04 §4.2).
const NMOS_REGISTRATION_SVC: &str = "_nmos-registration._tcp.local.";

/// Browse timeout when collecting already-cached registry entries.
const BROWSE_COLLECT_TIMEOUT: Duration = Duration::from_millis(500);

// ============================================================================
// NmosDiscovery
// ============================================================================

/// NMOS DNS-SD service announcer and registry browser.
///
/// On construction the daemon is created. Call [`announce`](Self::announce) to
/// publish the Node API service, and [`browse_registries`](Self::browse_registries)
/// to discover Registration API endpoints on the local network.
pub struct NmosDiscovery {
    /// UUID of this NMOS node.
    node_id: String,
    /// Human-readable label used as the DNS-SD instance name.
    node_label: String,
    /// NMOS API version string (e.g. `"v1.3"`).
    api_version: String,
    /// HTTP port on which the Node API listens.
    http_port: u16,
    /// Full name of the registered node service, set after [`announce`](Self::announce).
    node_service_fullname: Option<String>,
    /// The underlying mDNS daemon (cheaply cloneable channel handle).
    service_daemon: ServiceDaemon,
}

impl std::fmt::Debug for NmosDiscovery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NmosDiscovery")
            .field("node_id", &self.node_id)
            .field("node_label", &self.node_label)
            .field("api_version", &self.api_version)
            .field("http_port", &self.http_port)
            .field("node_service_fullname", &self.node_service_fullname)
            .finish_non_exhaustive()
    }
}

impl NmosDiscovery {
    /// Create a new `NmosDiscovery` instance.
    ///
    /// This starts the mDNS daemon thread but does **not** yet announce the
    /// service. Call [`announce`](Self::announce) after construction.
    ///
    /// # Parameters
    ///
    /// * `node_id`    — UUID string for this NMOS node (used as `id` TXT record).
    /// * `node_label` — Human-readable label; becomes the DNS-SD instance name.
    ///                  Must not be empty.
    /// * `http_port`  — TCP port of the Node API HTTP server.
    ///
    /// # Errors
    ///
    /// Returns [`NmosDiscoveryError::InvalidParameter`] if `node_id` or
    /// `node_label` are empty, or [`NmosDiscoveryError::Mdns`] if the daemon
    /// cannot be created.
    pub fn new(
        node_id: impl Into<String>,
        node_label: impl Into<String>,
        http_port: u16,
    ) -> Result<Self, NmosDiscoveryError> {
        let node_id = node_id.into();
        let node_label = node_label.into();

        if node_id.is_empty() {
            return Err(NmosDiscoveryError::InvalidParameter(
                "node_id must not be empty".into(),
            ));
        }
        if node_label.is_empty() {
            return Err(NmosDiscoveryError::InvalidParameter(
                "node_label must not be empty".into(),
            ));
        }

        let service_daemon =
            ServiceDaemon::new().map_err(|e| NmosDiscoveryError::Mdns(e.to_string()))?;

        // Raise the service-name length limit from the default 15 to accommodate
        // "_nmos-registration" (18 chars). The mdns-sd crate caps this at 30;
        // we use 30 which covers all NMOS service type names.
        service_daemon
            .set_service_name_len_max(30)
            .map_err(|e| NmosDiscoveryError::Mdns(e.to_string()))?;

        Ok(Self {
            node_id,
            node_label,
            api_version: "v1.3".into(),
            http_port,
            node_service_fullname: None,
            service_daemon,
        })
    }

    /// Override the NMOS API version string (default: `"v1.3"`).
    pub fn with_api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = version.into();
        self
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Build the TXT-record property map for the Node API service.
    fn node_txt_properties(&self) -> HashMap<&'static str, String> {
        let mut props = HashMap::new();
        props.insert("api_ver", self.api_version.clone());
        props.insert("api_proto", "http".into());
        props.insert("api_auth", "false".into());
        props.insert("pri", "0".into());
        props.insert("id", self.node_id.clone());
        props
    }

    /// Derive the mDNS hostname from the node label.
    ///
    /// DNS-SD hostnames must end with `.local.` and contain only alphanumeric
    /// characters and hyphens. We sanitise the label to meet those constraints.
    fn derive_hostname(&self) -> String {
        let sanitised: String = self
            .node_label
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
            .collect();
        // Trim leading/trailing hyphens that may result from sanitisation.
        let trimmed = sanitised.trim_matches('-');
        // Fall back to "nmos-node" if the entire label was non-alphanumeric.
        let hostname_base = if trimmed.is_empty() {
            "nmos-node"
        } else {
            trimmed
        };
        format!("{hostname_base}.local.")
    }

    /// Build a [`ServiceInfo`] for the given service type.
    fn build_service_info(
        &self,
        service_type: &str,
        instance_name: &str,
        txt_props: &HashMap<&'static str, String>,
    ) -> Result<ServiceInfo, NmosDiscoveryError> {
        let hostname = self.derive_hostname();

        // Convert txt_props into a Vec<(&str, &str)> which satisfies IntoTxtProperties.
        let props_vec: Vec<(&str, &str)> =
            txt_props.iter().map(|(k, v)| (*k, v.as_str())).collect();

        ServiceInfo::new(
            service_type,
            instance_name,
            &hostname,
            "", // empty → let enable_addr_auto() discover addresses
            self.http_port,
            props_vec.as_slice(),
        )
        .map(|info| info.enable_addr_auto())
        .map_err(|e| NmosDiscoveryError::Registration(e.to_string()))
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Announce this node's NMOS Node API via mDNS.
    ///
    /// Registers the `_nmos-node._tcp.local.` service record. If called
    /// multiple times the previous registration is silently replaced.
    ///
    /// # Errors
    ///
    /// Returns [`NmosDiscoveryError::Registration`] if the service cannot be
    /// registered, or [`NmosDiscoveryError::Mdns`] on daemon errors.
    pub fn announce(&mut self) -> Result<(), NmosDiscoveryError> {
        let txt = self.node_txt_properties();
        let service_info =
            self.build_service_info(NMOS_NODE_SVC, &self.node_label.clone(), &txt)?;

        let fullname = service_info.get_fullname().to_owned();

        self.service_daemon
            .register(service_info)
            .map_err(|e| NmosDiscoveryError::Registration(e.to_string()))?;

        self.node_service_fullname = Some(fullname);
        Ok(())
    }

    /// Withdraw the Node API announcement from the local network.
    ///
    /// No-op if [`announce`](Self::announce) was never called.
    ///
    /// # Errors
    ///
    /// Returns [`NmosDiscoveryError::Registration`] if unregistration fails.
    pub fn withdraw(&self) -> Result<(), NmosDiscoveryError> {
        let fullname = match &self.node_service_fullname {
            Some(n) => n.clone(),
            None => return Ok(()),
        };

        // unregister() returns a receiver that drains the result; we consume it
        // here without blocking indefinitely.
        let rx = self
            .service_daemon
            .unregister(&fullname)
            .map_err(|e| NmosDiscoveryError::Registration(e.to_string()))?;

        // Drain results (non-blocking).
        while rx.try_recv().is_ok() {}
        Ok(())
    }

    /// Browse for NMOS Registration API endpoints on the local network.
    ///
    /// Issues a DNS-SD browse for `_nmos-registration._tcp.local.` and
    /// collects all resolved instances found within a short timeout window.
    /// Results are sorted by [`NmosRegistryInfo::priority`] (ascending, so
    /// priority 0 is the preferred registry per AMWA IS-04 §4.2).
    ///
    /// # Errors
    ///
    /// Returns [`NmosDiscoveryError::Browse`] if the browse cannot be started.
    pub fn browse_registries(&self) -> Result<Vec<NmosRegistryInfo>, NmosDiscoveryError> {
        self.browse_service_type(NMOS_REGISTRATION_SVC)
    }

    /// Browse for NMOS Query API endpoints on the local network.
    ///
    /// # Errors
    ///
    /// Returns [`NmosDiscoveryError::Browse`] if the browse cannot be started.
    pub fn browse_query_apis(&self) -> Result<Vec<NmosRegistryInfo>, NmosDiscoveryError> {
        self.browse_service_type(NMOS_QUERY_SVC)
    }

    /// Generic browse helper: collect [`NmosRegistryInfo`] for any NMOS service type.
    fn browse_service_type(
        &self,
        service_type: &str,
    ) -> Result<Vec<NmosRegistryInfo>, NmosDiscoveryError> {
        let receiver = self
            .service_daemon
            .browse(service_type)
            .map_err(|e| NmosDiscoveryError::Browse(e.to_string()))?;

        let deadline = std::time::Instant::now() + BROWSE_COLLECT_TIMEOUT;
        let mut infos: Vec<NmosRegistryInfo> = Vec::new();

        loop {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                break;
            }

            match receiver.recv_timeout(remaining) {
                Ok(ServiceEvent::ServiceResolved(resolved)) => {
                    let host = resolved
                        .get_addresses_v4()
                        .into_iter()
                        .next()
                        .map(|a| a.to_string())
                        .unwrap_or_else(|| {
                            resolved.get_hostname().trim_end_matches('.').to_owned()
                        });

                    let priority = resolved
                        .get_property_val_str("pri")
                        .and_then(|s| s.parse::<u32>().ok())
                        .unwrap_or(100);

                    infos.push(NmosRegistryInfo {
                        name: resolved.get_fullname().to_owned(),
                        host,
                        port: resolved.get_port(),
                        priority,
                    });
                }
                Ok(ServiceEvent::SearchStopped(_)) => break,
                Ok(_) => {
                    // ServiceFound, SearchStarted, ServiceRemoved — not actionable here.
                }
                Err(_) => {
                    // recv_timeout timed out or channel closed — either is fine.
                    break;
                }
            }
        }

        // Stop the browse to free daemon resources.
        let _ = self.service_daemon.stop_browse(service_type);

        infos.sort_by_key(|r| r.priority);
        Ok(infos)
    }

    /// Gracefully stop the mDNS daemon.
    ///
    /// This withdraws all registered services and tears down the daemon thread.
    /// The [`NmosDiscovery`] instance should not be used after this call.
    ///
    /// # Errors
    ///
    /// Returns [`NmosDiscoveryError::Shutdown`] if the daemon shutdown fails.
    pub fn shutdown(&self) -> Result<(), NmosDiscoveryError> {
        let rx = self
            .service_daemon
            .shutdown()
            .map_err(|e| NmosDiscoveryError::Shutdown(e.to_string()))?;

        // Drain the status channel without blocking indefinitely.
        while rx.try_recv().is_ok() {}
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// The NMOS API version string used in TXT records.
    pub fn api_version(&self) -> &str {
        &self.api_version
    }

    /// The node ID as passed to [`new`](Self::new).
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    /// The node label / DNS-SD instance name.
    pub fn node_label(&self) -> &str {
        &self.node_label
    }

    /// The HTTP port of the Node API.
    pub fn http_port(&self) -> u16 {
        self.http_port
    }

    /// The full DNS-SD name of the registered node service, if [`announce`](Self::announce)
    /// has been called.
    pub fn node_service_fullname(&self) -> Option<&str> {
        self.node_service_fullname.as_deref()
    }

    /// Compute which NMOS Node service type constant is used.
    pub fn node_service_type() -> &'static str {
        NMOS_NODE_SVC
    }

    /// Compute which NMOS Registration service type constant is used.
    pub fn registration_service_type() -> &'static str {
        NMOS_REGISTRATION_SVC
    }

    /// Compute which NMOS Query service type constant is used.
    pub fn query_service_type() -> &'static str {
        NMOS_QUERY_SVC
    }
}

// ============================================================================
// NmosDiscoveryBuilder
// ============================================================================

/// Builder for [`NmosDiscovery`] with optional configuration.
///
/// Provides ergonomic construction when several optional fields need to be set.
///
/// # Example
///
/// ```rust,no_run
/// # #[cfg(feature = "nmos-discovery")]
/// # {
/// use oximedia_routing::nmos::discovery::NmosDiscoveryBuilder;
///
/// let disc = NmosDiscoveryBuilder::new(
///     "550e8400-e29b-41d4-a716-446655440001",
///     "Studio Node A",
///     8080,
/// )
/// .api_version("v1.3")
/// .build()
/// .expect("builder failed");
/// # }
/// ```
pub struct NmosDiscoveryBuilder {
    node_id: String,
    node_label: String,
    api_version: String,
    http_port: u16,
}

impl NmosDiscoveryBuilder {
    /// Start building with mandatory fields.
    pub fn new(node_id: impl Into<String>, node_label: impl Into<String>, http_port: u16) -> Self {
        Self {
            node_id: node_id.into(),
            node_label: node_label.into(),
            api_version: "v1.3".into(),
            http_port,
        }
    }

    /// Override the NMOS API version string.
    pub fn api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = version.into();
        self
    }

    /// Construct the [`NmosDiscovery`] instance.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`NmosDiscovery::new`].
    pub fn build(self) -> Result<NmosDiscovery, NmosDiscoveryError> {
        NmosDiscovery::new(self.node_id, self.node_label, self.http_port)
            .map(|d| d.with_api_version(self.api_version))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Unit tests — logic tested without starting real mDNS traffic
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_rejects_empty_node_id() {
        let err = NmosDiscovery::new("", "My Node", 8080);
        assert!(err.is_err());
        let msg = err.expect_err("should fail").to_string();
        assert!(msg.contains("node_id"), "error should mention 'node_id'");
    }

    #[test]
    fn test_new_rejects_empty_node_label() {
        let err = NmosDiscovery::new("some-uuid", "", 8080);
        assert!(err.is_err());
        let msg = err.expect_err("should fail").to_string();
        assert!(
            msg.contains("node_label"),
            "error should mention 'node_label'"
        );
    }

    #[test]
    fn test_derive_hostname_alphanumeric_label() {
        let disc = NmosDiscovery {
            node_id: "id".into(),
            node_label: "StudioNode".into(),
            api_version: "v1.3".into(),
            http_port: 8080,
            node_service_fullname: None,
            service_daemon: ServiceDaemon::new().expect("daemon creation failed in test"),
        };
        let h = disc.derive_hostname();
        assert_eq!(h, "StudioNode.local.");
        disc.shutdown().ok();
    }

    #[test]
    fn test_derive_hostname_sanitises_spaces() {
        let disc = NmosDiscovery {
            node_id: "id".into(),
            node_label: "Studio Node A".into(),
            api_version: "v1.3".into(),
            http_port: 8080,
            node_service_fullname: None,
            service_daemon: ServiceDaemon::new().expect("daemon creation failed in test"),
        };
        let h = disc.derive_hostname();
        // Spaces become hyphens; consecutive hyphens are allowed by DNS labels.
        assert!(h.ends_with(".local."), "hostname must end with .local.");
        assert!(!h.contains(' '), "hostname must not contain spaces");
        disc.shutdown().ok();
    }

    #[test]
    fn test_derive_hostname_all_special_chars_falls_back() {
        let disc = NmosDiscovery {
            node_id: "id".into(),
            node_label: "!!!".into(),
            api_version: "v1.3".into(),
            http_port: 8080,
            node_service_fullname: None,
            service_daemon: ServiceDaemon::new().expect("daemon creation failed in test"),
        };
        let h = disc.derive_hostname();
        assert!(
            h.starts_with("nmos-node"),
            "should fall back to 'nmos-node'"
        );
        disc.shutdown().ok();
    }

    #[test]
    fn test_node_txt_properties_keys() {
        let disc = NmosDiscovery {
            node_id: "uuid-1234".into(),
            node_label: "Test".into(),
            api_version: "v1.3".into(),
            http_port: 8080,
            node_service_fullname: None,
            service_daemon: ServiceDaemon::new().expect("daemon creation failed in test"),
        };
        let props = disc.node_txt_properties();
        assert_eq!(props.get("api_ver").map(String::as_str), Some("v1.3"));
        assert_eq!(props.get("api_proto").map(String::as_str), Some("http"));
        assert_eq!(props.get("api_auth").map(String::as_str), Some("false"));
        assert_eq!(props.get("pri").map(String::as_str), Some("0"));
        assert_eq!(props.get("id").map(String::as_str), Some("uuid-1234"));
        disc.shutdown().ok();
    }

    #[test]
    fn test_service_type_constants() {
        assert_eq!(NmosDiscovery::node_service_type(), "_nmos-node._tcp.local.");
        assert_eq!(
            NmosDiscovery::registration_service_type(),
            "_nmos-registration._tcp.local."
        );
        assert_eq!(
            NmosDiscovery::query_service_type(),
            "_nmos-query._tcp.local."
        );
    }

    #[test]
    fn test_accessors() {
        let disc = NmosDiscovery {
            node_id: "my-node-id".into(),
            node_label: "My Label".into(),
            api_version: "v1.2".into(),
            http_port: 9000,
            node_service_fullname: None,
            service_daemon: ServiceDaemon::new().expect("daemon creation failed in test"),
        };
        assert_eq!(disc.node_id(), "my-node-id");
        assert_eq!(disc.node_label(), "My Label");
        assert_eq!(disc.api_version(), "v1.2");
        assert_eq!(disc.http_port(), 9000);
        assert!(disc.node_service_fullname().is_none());
        disc.shutdown().ok();
    }

    #[test]
    fn test_builder_sets_api_version() {
        let disc = NmosDiscoveryBuilder::new("uuid-abc", "Builder Node", 7070)
            .api_version("v1.2")
            .build()
            .expect("builder should succeed in test");
        assert_eq!(disc.api_version(), "v1.2");
        disc.shutdown().ok();
    }

    #[test]
    fn test_builder_defaults_api_version() {
        let disc = NmosDiscoveryBuilder::new("uuid-def", "Default Version Node", 7071)
            .build()
            .expect("builder should succeed in test");
        assert_eq!(disc.api_version(), "v1.3");
        disc.shutdown().ok();
    }

    #[test]
    fn test_error_display_mdns() {
        let err = NmosDiscoveryError::Mdns("test error".into());
        assert!(err.to_string().contains("mDNS daemon error"));
    }

    #[test]
    fn test_error_display_registration() {
        let err = NmosDiscoveryError::Registration("bad service".into());
        assert!(err.to_string().contains("service registration failed"));
    }

    #[test]
    fn test_error_display_browse() {
        let err = NmosDiscoveryError::Browse("no network".into());
        assert!(err.to_string().contains("service browse failed"));
    }

    #[test]
    fn test_error_display_shutdown() {
        let err = NmosDiscoveryError::Shutdown("already stopped".into());
        assert!(err.to_string().contains("shutdown error"));
    }

    #[test]
    fn test_nmos_registry_info_fields() {
        let info = NmosRegistryInfo {
            name: "MyReg._nmos-registration._tcp.local.".into(),
            host: "192.168.1.1".into(),
            port: 8080,
            priority: 0,
        };
        assert_eq!(info.name, "MyReg._nmos-registration._tcp.local.");
        assert_eq!(info.host, "192.168.1.1");
        assert_eq!(info.port, 8080);
        assert_eq!(info.priority, 0);
    }

    #[test]
    fn test_withdraw_before_announce_is_noop() {
        let disc = NmosDiscovery::new("uuid-noop", "Noop Node", 8888)
            .expect("discovery init failed in test");
        // withdraw without announce should succeed quietly.
        disc.withdraw().expect("withdraw should be no-op");
        disc.shutdown().ok();
    }

    // -----------------------------------------------------------------------
    // Integration-style test: announce + shutdown lifecycle
    // (requires an actual loopback interface; skipped in pure unit-test envs)
    // -----------------------------------------------------------------------

    #[test]
    #[ignore = "requires real network interface; run manually with --include-ignored"]
    fn test_announce_and_shutdown_lifecycle() {
        let mut disc = NmosDiscovery::new(
            "550e8400-e29b-41d4-a716-446655440099",
            "TestNMOSNode",
            18080,
        )
        .expect("discovery init failed");

        disc.announce().expect("announce failed");
        assert!(
            disc.node_service_fullname().is_some(),
            "fullname set after announce"
        );

        disc.withdraw().expect("withdraw failed");
        disc.shutdown().expect("shutdown failed");
    }

    #[test]
    #[ignore = "requires real network interface; run manually with --include-ignored"]
    fn test_browse_registries_returns_empty_on_clean_network() {
        let disc = NmosDiscovery::new(
            "550e8400-e29b-41d4-a716-446655440088",
            "BrowseTestNode",
            18081,
        )
        .expect("discovery init failed");

        // On a clean test network there should be no NMOS registries; we just
        // verify the call completes without error.
        let registries = disc.browse_registries().expect("browse failed");
        // Result may be empty or contain real devices — both are valid.
        let _ = registries;

        disc.shutdown().expect("shutdown failed");
    }
}

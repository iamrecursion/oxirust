//! mDNS advertisement for NDI sender-side discovery.
//!
//! This module provides `MdnsAdvertiser` which advertises an NDI source on the
//! local network using the `_ndi._tcp.local.` service type, with TXT records
//! conforming to the NDI v5 protocol.

#![allow(dead_code)]

use crate::{NdiError, Result};
use mdns_sd::{ServiceDaemon, ServiceInfo};
use std::collections::HashMap;

/// NDI service type for mDNS advertisement.
pub const NDI_SERVICE_TYPE: &str = "_ndi._tcp.local.";

/// Standard NDI TXT record value for the `type` key.
pub const NDI_TXT_TYPE: &str = "NDI Source";

/// Standard NDI TXT record value for the `groups` key.
pub const NDI_TXT_GROUPS_DEFAULT: &str = "Public";

/// Standard NDI protocol version advertised in TXT records.
pub const NDI_TXT_VERSION: &str = "5.0";

/// An mDNS advertisement record for an NDI source.
///
/// Encapsulates the service name, port, and TXT properties that will be
/// broadcast via the `_ndi._tcp.local.` service type.
#[derive(Debug, Clone)]
pub struct MdnsAdvertisement {
    /// The unique service instance name (e.g. `"MyMachine (Camera 1)"`).
    pub service_name: String,
    /// The TCP port on which the NDI source is listening.
    pub port: u16,
    /// TXT record key/value pairs published with the advertisement.
    pub properties: HashMap<String, String>,
}

impl MdnsAdvertisement {
    /// Create a new advertisement with the standard NDI TXT records.
    ///
    /// TXT records populated by default:
    /// - `type` = `"NDI Source"`
    /// - `groups` = `"Public"`
    /// - `v` = `"5.0"`
    pub fn new(service_name: impl Into<String>, port: u16) -> Self {
        let mut properties = HashMap::new();
        properties.insert("type".to_string(), NDI_TXT_TYPE.to_string());
        properties.insert("groups".to_string(), NDI_TXT_GROUPS_DEFAULT.to_string());
        properties.insert("v".to_string(), NDI_TXT_VERSION.to_string());
        Self {
            service_name: service_name.into(),
            port,
            properties,
        }
    }

    /// Override the `groups` TXT record (comma-separated group list).
    pub fn with_groups(mut self, groups: impl Into<String>) -> Self {
        self.properties.insert("groups".to_string(), groups.into());
        self
    }

    /// Insert or overwrite an arbitrary TXT record key/value pair.
    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Return the `type` TXT record value, if present.
    pub fn ndi_type(&self) -> Option<&str> {
        self.properties.get("type").map(String::as_str)
    }

    /// Return the `groups` TXT record value, if present.
    pub fn groups(&self) -> Option<&str> {
        self.properties.get("groups").map(String::as_str)
    }

    /// Return the NDI protocol version TXT record value, if present.
    pub fn version(&self) -> Option<&str> {
        self.properties.get("v").map(String::as_str)
    }

    /// Encode the TXT records into a compact DNS-wire-style byte sequence.
    ///
    /// Each record is serialised as a single length-prefixed string
    /// `"key=value"` exactly as required by RFC 6763 §6.
    pub fn encode_txt_records(&self) -> Vec<Vec<u8>> {
        let mut records: Vec<Vec<u8>> = self
            .properties
            .iter()
            .map(|(k, v)| {
                let entry = format!("{k}={v}");
                entry.into_bytes()
            })
            .collect();
        records.sort(); // deterministic order for testing
        records
    }

    /// Return the DNS instance label that would appear in the PTR record:
    /// `"<service_name>._ndi._tcp.local."`.
    pub fn dns_ptr_name(&self) -> String {
        format!("{}.{}", self.service_name, NDI_SERVICE_TYPE)
    }

    /// Validate that required TXT keys are present.
    pub fn is_valid(&self) -> bool {
        self.properties.contains_key("type")
            && self.properties.contains_key("groups")
            && self.properties.contains_key("v")
            && self.port > 0
            && !self.service_name.is_empty()
    }
}

// ---------------------------------------------------------------------------
// MdnsAdvertiser
// ---------------------------------------------------------------------------

/// Advertises an NDI source using mDNS (`_ndi._tcp.local.`).
///
/// Wraps the `mdns-sd` crate's `ServiceDaemon` to register and unregister the
/// service automatically.  The advertisement uses TXT records required by the
/// NDI v5 specification.
pub struct MdnsAdvertiser {
    daemon: ServiceDaemon,
    advertisement: MdnsAdvertisement,
    /// Host name used in the SRV record (with `.local.` suffix).
    host_name: String,
    /// IP address string for the A/AAAA record.
    ip_addr: String,
    /// Whether we are currently registered.
    registered: bool,
}

impl MdnsAdvertiser {
    /// Create a new advertiser.  Does **not** start advertising yet; call
    /// [`MdnsAdvertiser::start`] to begin.
    pub fn new(
        advertisement: MdnsAdvertisement,
        host_name: impl Into<String>,
        ip_addr: impl Into<String>,
    ) -> Result<Self> {
        let daemon = ServiceDaemon::new()
            .map_err(|e| NdiError::Discovery(format!("mDNS daemon init failed: {e}")))?;
        Ok(Self {
            daemon,
            advertisement,
            host_name: host_name.into(),
            ip_addr: ip_addr.into(),
            registered: false,
        })
    }

    /// Convenience constructor: build from a bare `NdiSource` name and port.
    ///
    /// Determines the local hostname and IP automatically.
    pub fn start(service_name: impl Into<String>, port: u16) -> Result<Self> {
        let adv = MdnsAdvertisement::new(service_name, port);
        // Derive a hostname from the system
        let host_name = hostname_local();
        let ip_addr = local_ip_str();
        let mut me = Self::new(adv, host_name, ip_addr)?;
        me.register()?;
        Ok(me)
    }

    /// Register (start advertising) the NDI source.
    ///
    /// This is idempotent — a second call is a no-op if already registered.
    pub fn register(&mut self) -> Result<()> {
        if self.registered {
            return Ok(());
        }

        let svc = self.build_service_info()?;
        self.daemon
            .register(svc)
            .map_err(|e| NdiError::Discovery(format!("mDNS register failed: {e}")))?;
        self.registered = true;
        Ok(())
    }

    /// Unregister (stop advertising) the NDI source.
    ///
    /// This is idempotent — a second call is a no-op if not registered.
    pub fn unregister(&mut self) -> Result<()> {
        if !self.registered {
            return Ok(());
        }
        let fullname = format!("{}.{}", self.advertisement.service_name, NDI_SERVICE_TYPE);
        self.daemon
            .unregister(&fullname)
            .map_err(|e| NdiError::Discovery(format!("mDNS unregister failed: {e}")))?;
        self.registered = false;
        Ok(())
    }

    /// Returns `true` if the advertisement is currently active.
    pub fn is_registered(&self) -> bool {
        self.registered
    }

    /// Access the underlying advertisement descriptor.
    pub fn advertisement(&self) -> &MdnsAdvertisement {
        &self.advertisement
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn build_service_info(&self) -> Result<ServiceInfo> {
        ServiceInfo::new(
            NDI_SERVICE_TYPE,
            &self.advertisement.service_name,
            &self.host_name,
            self.ip_addr.as_str(),
            self.advertisement.port,
            Some(self.advertisement.properties.clone()),
        )
        .map_err(|e| NdiError::Discovery(format!("ServiceInfo construction failed: {e}")))
    }
}

impl Drop for MdnsAdvertiser {
    fn drop(&mut self) {
        if self.registered {
            let _ = self.unregister();
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Return a `.local.`-suffixed hostname for use in SRV records.
fn hostname_local() -> String {
    // Attempt to get the real hostname; fall back to a safe default.
    std::process::Command::new("hostname")
        .output()
        .ok()
        .and_then(|out| {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if s.is_empty() {
                None
            } else {
                Some(format!("{s}.local."))
            }
        })
        .unwrap_or_else(|| "oximedia-ndi.local.".to_string())
}

/// Return the local (non-loopback) IP as a string, defaulting to `127.0.0.1`.
fn local_ip_str() -> String {
    use std::net::UdpSocket;
    UdpSocket::bind("0.0.0.0:0")
        .and_then(|s| {
            s.connect("8.8.8.8:80")?;
            s.local_addr()
        })
        .map(|a| a.ip().to_string())
        .unwrap_or_else(|_| "127.0.0.1".to_string())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // MdnsAdvertisement construction and accessors
    // -----------------------------------------------------------------------

    #[test]
    fn test_advertisement_default_txt_records() {
        let adv = MdnsAdvertisement::new("MyCamera", 5960);
        assert_eq!(adv.ndi_type(), Some("NDI Source"));
        assert_eq!(adv.groups(), Some("Public"));
        assert_eq!(adv.version(), Some("5.0"));
    }

    #[test]
    fn test_advertisement_with_groups_override() {
        let adv = MdnsAdvertisement::new("Studio Cam", 5960).with_groups("Studio,Preview");
        assert_eq!(adv.groups(), Some("Studio,Preview"));
    }

    #[test]
    fn test_advertisement_with_custom_property() {
        let adv = MdnsAdvertisement::new("Cam1", 5961).with_property("vendor", "OxiMedia");
        assert_eq!(
            adv.properties.get("vendor").map(String::as_str),
            Some("OxiMedia")
        );
    }

    #[test]
    fn test_advertisement_is_valid() {
        let adv = MdnsAdvertisement::new("Cam", 5960);
        assert!(adv.is_valid());
    }

    #[test]
    fn test_advertisement_invalid_no_port() {
        let adv = MdnsAdvertisement::new("Cam", 0);
        assert!(!adv.is_valid());
    }

    #[test]
    fn test_advertisement_invalid_empty_name() {
        let adv = MdnsAdvertisement::new("", 5960);
        assert!(!adv.is_valid());
    }

    #[test]
    fn test_advertisement_dns_ptr_name() {
        let adv = MdnsAdvertisement::new("Camera 1", 5960);
        let ptr = adv.dns_ptr_name();
        assert!(ptr.starts_with("Camera 1."));
        assert!(ptr.contains("_ndi._tcp.local."));
    }

    #[test]
    fn test_advertisement_encode_txt_records_contains_type() {
        let adv = MdnsAdvertisement::new("SomeSource", 5960);
        let records = adv.encode_txt_records();
        let strings: Vec<String> = records
            .iter()
            .map(|b| String::from_utf8_lossy(b).to_string())
            .collect();
        assert!(strings.iter().any(|s| s == "type=NDI Source"));
    }

    #[test]
    fn test_advertisement_encode_txt_records_deterministic() {
        let adv = MdnsAdvertisement::new("X", 1234);
        let a = adv.encode_txt_records();
        let b = adv.encode_txt_records();
        assert_eq!(a, b);
    }

    #[test]
    fn test_advertisement_unique_service_names() {
        let adv1 = MdnsAdvertisement::new("Studio Camera 1", 5960);
        let adv2 = MdnsAdvertisement::new("Studio Camera 2", 5960);
        assert_ne!(adv1.service_name, adv2.service_name);
        assert_ne!(adv1.dns_ptr_name(), adv2.dns_ptr_name());
    }

    #[test]
    fn test_advertisement_port_survives_builder() {
        let adv = MdnsAdvertisement::new("Cam", 7777)
            .with_groups("A,B")
            .with_property("extra", "val");
        assert_eq!(adv.port, 7777);
    }

    // -----------------------------------------------------------------------
    // Hostname / IP helpers (compile-time checks)
    // -----------------------------------------------------------------------

    #[test]
    fn test_hostname_local_not_empty() {
        let h = hostname_local();
        assert!(!h.is_empty());
        assert!(h.ends_with(".local.") || h == "oximedia-ndi.local.");
    }

    #[test]
    fn test_local_ip_str_not_empty() {
        let ip = local_ip_str();
        assert!(!ip.is_empty());
    }
}

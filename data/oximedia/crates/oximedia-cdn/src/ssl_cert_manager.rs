//! TLS certificate expiration tracking for CDN edge nodes.
//!
//! # Overview
//!
//! `SslCertManager` maintains a registry of `CertRecord`s, one per edge
//! node.  Each record stores the certificate's not-before/not-after window,
//! the associated domain names, and metadata (issuer, serial number,
//! fingerprint).  The manager can:
//!
//! - Detect certificates that have already expired.
//! - Warn about certificates expiring within a configurable horizon.
//! - Compute days remaining until expiry.
//! - Export a Prometheus-compatible gauge (`ssl_cert_days_remaining`).
//!
//! # Design
//!
//! No external X.509 parser is used.  Callers supply pre-parsed certificate
//! metadata when calling `SslCertManager::register`.  The manager's job is
//! purely to track and report on expiration state.
//!
//! Thread-safety is provided by an `RwLock`-protected `HashMap` so reads
//! (status queries, Prometheus export) can proceed concurrently.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use thiserror::Error;

// ─── Errors ───────────────────────────────────────────────────────────────────

/// Errors from certificate management operations.
#[derive(Debug, Error)]
pub enum CertError {
    /// A certificate with the given edge-node ID is not registered.
    #[error("no certificate registered for edge node '{0}'")]
    NotFound(String),
    /// The certificate's `not_before` is after `not_after`.
    #[error(
        "invalid certificate validity window: not_before={not_before} > not_after={not_after}"
    )]
    InvalidWindow {
        /// `not_before` unix timestamp.
        not_before: u64,
        /// `not_after` unix timestamp.
        not_after: u64,
    },
    /// An internal lock was poisoned.
    #[error("internal lock poisoned")]
    LockPoisoned,
}

// ─── CertStatus ───────────────────────────────────────────────────────────────

/// Current status of a TLS certificate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CertStatus {
    /// Certificate is valid and not expiring soon.
    Valid,
    /// Certificate will expire within the warning horizon.
    ExpiringSoon {
        /// Days remaining until expiry (floor).
        days_remaining: u32,
    },
    /// Certificate has already expired.
    Expired,
    /// Certificate's `not_before` is in the future (clock skew or pre-issued).
    NotYetValid,
}

impl fmt::Display for CertStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Valid => f.write_str("valid"),
            Self::ExpiringSoon { days_remaining } => {
                write!(f, "expiring_soon({days_remaining}d)")
            }
            Self::Expired => f.write_str("expired"),
            Self::NotYetValid => f.write_str("not_yet_valid"),
        }
    }
}

// ─── CertRecord ───────────────────────────────────────────────────────────────

/// Metadata for a single TLS certificate deployed on an edge node.
#[derive(Debug, Clone)]
pub struct CertRecord {
    /// The edge node this certificate is deployed on.
    pub edge_node_id: String,
    /// Primary domain name (CN or first SAN).
    pub domain: String,
    /// All Subject Alternative Names.
    pub sans: Vec<String>,
    /// Certificate issuer (Common Name of the CA).
    pub issuer: String,
    /// Serial number as a hex string.
    pub serial: String,
    /// SHA-256 fingerprint as a colon-separated hex string.
    pub fingerprint: String,
    /// `not_before` as a Unix timestamp (seconds since epoch).
    pub not_before_unix: u64,
    /// `not_after` as a Unix timestamp (seconds since epoch).
    pub not_after_unix: u64,
    /// Whether OCSP stapling is enabled for this certificate.
    pub ocsp_stapling: bool,
    /// Whether the certificate is a wildcard (`*.domain`).
    pub is_wildcard: bool,
}

impl CertRecord {
    /// Create a new [`CertRecord`].
    ///
    /// Returns [`CertError::InvalidWindow`] if `not_before_unix > not_after_unix`.
    pub fn new(
        edge_node_id: impl Into<String>,
        domain: impl Into<String>,
        not_before_unix: u64,
        not_after_unix: u64,
    ) -> Result<Self, CertError> {
        let not_before = not_before_unix;
        let not_after = not_after_unix;
        if not_before > not_after {
            return Err(CertError::InvalidWindow {
                not_before,
                not_after,
            });
        }
        let domain = domain.into();
        let is_wildcard = domain.starts_with("*.");
        Ok(Self {
            edge_node_id: edge_node_id.into(),
            domain,
            sans: Vec::new(),
            issuer: String::new(),
            serial: String::new(),
            fingerprint: String::new(),
            not_before_unix,
            not_after_unix,
            ocsp_stapling: false,
            is_wildcard,
        })
    }

    /// Set the issuer CN.
    pub fn with_issuer(mut self, issuer: impl Into<String>) -> Self {
        self.issuer = issuer.into();
        self
    }

    /// Set the serial number.
    pub fn with_serial(mut self, serial: impl Into<String>) -> Self {
        self.serial = serial.into();
        self
    }

    /// Set the SHA-256 fingerprint.
    pub fn with_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.fingerprint = fingerprint.into();
        self
    }

    /// Add a Subject Alternative Name.
    pub fn with_san(mut self, san: impl Into<String>) -> Self {
        self.sans.push(san.into());
        self
    }

    /// Enable OCSP stapling flag.
    pub fn with_ocsp_stapling(mut self, enabled: bool) -> Self {
        self.ocsp_stapling = enabled;
        self
    }

    /// Compute the number of days remaining until expiry at `now_unix`.
    ///
    /// Returns `0` if the certificate has already expired.
    pub fn days_remaining(&self, now_unix: u64) -> u32 {
        if now_unix >= self.not_after_unix {
            return 0;
        }
        let remaining_secs = self.not_after_unix - now_unix;
        (remaining_secs / 86_400) as u32
    }

    /// Compute the certificate validity duration.
    pub fn validity_duration(&self) -> Duration {
        Duration::from_secs(self.not_after_unix.saturating_sub(self.not_before_unix))
    }

    /// Determine the [`CertStatus`] at `now_unix` with the given warning horizon.
    ///
    /// `warning_horizon_days` controls when `ExpiringSoon` is triggered.
    pub fn status_at(&self, now_unix: u64, warning_horizon_days: u32) -> CertStatus {
        if now_unix < self.not_before_unix {
            return CertStatus::NotYetValid;
        }
        if now_unix >= self.not_after_unix {
            return CertStatus::Expired;
        }
        let days = self.days_remaining(now_unix);
        if days <= warning_horizon_days {
            CertStatus::ExpiringSoon {
                days_remaining: days,
            }
        } else {
            CertStatus::Valid
        }
    }
}

// ─── CertSnapshot ─────────────────────────────────────────────────────────────

/// Point-in-time view of a certificate's status.
#[derive(Debug, Clone)]
pub struct CertSnapshot {
    /// Edge node identifier.
    pub edge_node_id: String,
    /// Primary domain.
    pub domain: String,
    /// Current status.
    pub status: CertStatus,
    /// Days remaining (0 if expired).
    pub days_remaining: u32,
    /// `not_after` unix timestamp.
    pub not_after_unix: u64,
    /// Certificate issuer.
    pub issuer: String,
    /// Serial number.
    pub serial: String,
}

// ─── CertManagerConfig ────────────────────────────────────────────────────────

/// Configuration for [`SslCertManager`].
#[derive(Debug, Clone)]
pub struct CertManagerConfig {
    /// Number of days before expiry to start warning.
    pub warning_horizon_days: u32,
    /// Number of days before expiry to start a critical alert.
    pub critical_horizon_days: u32,
}

impl Default for CertManagerConfig {
    fn default() -> Self {
        Self {
            warning_horizon_days: 30,
            critical_horizon_days: 7,
        }
    }
}

// ─── SslCertManager ───────────────────────────────────────────────────────────

/// Registry for TLS certificates across CDN edge nodes.
pub struct SslCertManager {
    certs: RwLock<HashMap<String, Arc<CertRecord>>>,
    /// Configuration for expiration thresholds.
    pub config: CertManagerConfig,
}

impl SslCertManager {
    /// Create an empty manager.
    pub fn new(config: CertManagerConfig) -> Self {
        Self {
            certs: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Register (or replace) a certificate for an edge node.
    pub fn register(&self, record: CertRecord) -> Result<(), CertError> {
        let mut guard = self.certs.write().map_err(|_| CertError::LockPoisoned)?;
        guard.insert(record.edge_node_id.clone(), Arc::new(record));
        Ok(())
    }

    /// Remove a certificate registration.  Returns `true` if one was removed.
    pub fn deregister(&self, edge_node_id: &str) -> bool {
        self.certs
            .write()
            .map(|mut g| g.remove(edge_node_id).is_some())
            .unwrap_or(false)
    }

    /// Get a certificate record by edge node ID.
    pub fn get(&self, edge_node_id: &str) -> Result<Arc<CertRecord>, CertError> {
        let guard = self.certs.read().map_err(|_| CertError::LockPoisoned)?;
        guard
            .get(edge_node_id)
            .cloned()
            .ok_or_else(|| CertError::NotFound(edge_node_id.to_string()))
    }

    /// Get a snapshot of all registered certificates evaluated at `now_unix`.
    pub fn snapshots_at(&self, now_unix: u64) -> Vec<CertSnapshot> {
        let guard = match self.certs.read() {
            Ok(g) => g,
            Err(_) => return Vec::new(),
        };
        guard
            .values()
            .map(|r| CertSnapshot {
                edge_node_id: r.edge_node_id.clone(),
                domain: r.domain.clone(),
                status: r.status_at(now_unix, self.config.warning_horizon_days),
                days_remaining: r.days_remaining(now_unix),
                not_after_unix: r.not_after_unix,
                issuer: r.issuer.clone(),
                serial: r.serial.clone(),
            })
            .collect()
    }

    /// Get a snapshot evaluated using the current system clock.
    pub fn snapshots(&self) -> Vec<CertSnapshot> {
        let now = current_unix_ts();
        self.snapshots_at(now)
    }

    /// Return all edge nodes whose certificate status at `now_unix` matches
    /// `status`.
    pub fn nodes_with_status_at(
        &self,
        status_filter: CertStatus,
        now_unix: u64,
    ) -> Vec<CertSnapshot> {
        self.snapshots_at(now_unix)
            .into_iter()
            .filter(|s| {
                // Compare by variant, not exact value (ExpiringSoon days may differ).
                std::mem::discriminant(&s.status) == std::mem::discriminant(&status_filter)
            })
            .collect()
    }

    /// Count certificates in each status category at `now_unix`.
    pub fn status_counts_at(&self, now_unix: u64) -> StatusCounts {
        let snaps = self.snapshots_at(now_unix);
        let mut counts = StatusCounts::default();
        for s in &snaps {
            match s.status {
                CertStatus::Valid => counts.valid += 1,
                CertStatus::ExpiringSoon { .. } => counts.expiring_soon += 1,
                CertStatus::Expired => counts.expired += 1,
                CertStatus::NotYetValid => counts.not_yet_valid += 1,
            }
        }
        counts
    }

    /// Return the number of registered certificates.
    pub fn len(&self) -> usize {
        self.certs.read().map(|g| g.len()).unwrap_or(0)
    }

    /// Returns `true` if no certificates are registered.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Render Prometheus gauge metrics for all certificates.
    ///
    /// Emits `ssl_cert_days_remaining{node="...",domain="..."}` and
    /// `ssl_cert_status{node="...",domain="...",status="..."}`.
    pub fn to_prometheus_at(&self, now_unix: u64) -> String {
        let snaps = self.snapshots_at(now_unix);
        let mut out = String::with_capacity(snaps.len() * 128);

        if !snaps.is_empty() {
            out.push_str(
                "# HELP ssl_cert_days_remaining Days until TLS certificate expiration.\n\
                 # TYPE ssl_cert_days_remaining gauge\n",
            );
            for s in &snaps {
                out.push_str(&format!(
                    "ssl_cert_days_remaining{{node=\"{}\",domain=\"{}\"}} {}\n",
                    s.edge_node_id, s.domain, s.days_remaining
                ));
            }

            out.push_str(
                "# HELP ssl_cert_expired 1 if certificate is expired, 0 otherwise.\n\
                 # TYPE ssl_cert_expired gauge\n",
            );
            for s in &snaps {
                let expired = if s.status == CertStatus::Expired {
                    1
                } else {
                    0
                };
                out.push_str(&format!(
                    "ssl_cert_expired{{node=\"{}\",domain=\"{}\"}} {}\n",
                    s.edge_node_id, s.domain, expired
                ));
            }
        }

        out
    }

    /// Render Prometheus metrics using the current system clock.
    pub fn to_prometheus(&self) -> String {
        self.to_prometheus_at(current_unix_ts())
    }
}

impl Default for SslCertManager {
    fn default() -> Self {
        Self::new(CertManagerConfig::default())
    }
}

// ─── StatusCounts ─────────────────────────────────────────────────────────────

/// Aggregate counts of certificates in each status category.
#[derive(Debug, Clone, Default)]
pub struct StatusCounts {
    /// Valid certificates.
    pub valid: usize,
    /// Certificates expiring soon.
    pub expiring_soon: usize,
    /// Expired certificates.
    pub expired: usize,
    /// Certificates not yet valid.
    pub not_yet_valid: usize,
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn current_unix_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a CertRecord with a known window relative to `anchor`.
    fn make_cert(
        node: &str,
        domain: &str,
        anchor: u64,
        before_offset_days: i64,
        after_offset_days: i64,
    ) -> CertRecord {
        let before = (anchor as i64 + before_offset_days * 86_400) as u64;
        let after = (anchor as i64 + after_offset_days * 86_400) as u64;
        CertRecord::new(node, domain, before, after).expect("valid window")
    }

    const NOW: u64 = 1_700_000_000; // arbitrary fixed "now"

    // 1. New CertRecord has correct fields.
    #[test]
    fn test_cert_record_new() {
        let r = CertRecord::new("pop-iad", "cdn.example.com", 1_000_000, 2_000_000).expect("ok");
        assert_eq!(r.edge_node_id, "pop-iad");
        assert_eq!(r.domain, "cdn.example.com");
        assert_eq!(r.not_before_unix, 1_000_000);
        assert_eq!(r.not_after_unix, 2_000_000);
        assert!(!r.is_wildcard);
    }

    // 2. Wildcard domain detected.
    #[test]
    fn test_cert_record_wildcard() {
        let r = CertRecord::new("n", "*.example.com", 0, 1_000).expect("ok");
        assert!(r.is_wildcard);
    }

    // 3. InvalidWindow error when not_before > not_after.
    #[test]
    fn test_cert_record_invalid_window() {
        let err = CertRecord::new("n", "d", 2_000, 1_000).unwrap_err();
        assert!(matches!(err, CertError::InvalidWindow { .. }));
    }

    // 4. Builder methods set fields.
    #[test]
    fn test_cert_record_builders() {
        let r = CertRecord::new("n", "d", 0, 1_000)
            .expect("ok")
            .with_issuer("Let's Encrypt R3")
            .with_serial("aabb1122")
            .with_fingerprint("AA:BB:CC")
            .with_san("www.example.com")
            .with_ocsp_stapling(true);
        assert_eq!(r.issuer, "Let's Encrypt R3");
        assert_eq!(r.serial, "aabb1122");
        assert_eq!(r.fingerprint, "AA:BB:CC");
        assert_eq!(r.sans, vec!["www.example.com"]);
        assert!(r.ocsp_stapling);
    }

    // 5. days_remaining returns 0 for expired cert.
    #[test]
    fn test_days_remaining_expired() {
        let r = make_cert("n", "d", NOW, -365, -1);
        assert_eq!(r.days_remaining(NOW), 0);
    }

    // 6. days_remaining returns correct value for future cert.
    #[test]
    fn test_days_remaining_future() {
        let r = make_cert("n", "d", NOW, -365, 30); // expires in 30 days
        let days = r.days_remaining(NOW);
        assert_eq!(days, 30);
    }

    // 7. validity_duration is correct.
    #[test]
    fn test_validity_duration() {
        let r = CertRecord::new("n", "d", 0, 90 * 86_400).expect("ok");
        assert_eq!(r.validity_duration(), Duration::from_secs(90 * 86_400));
    }

    // ── CertStatus ────────────────────────────────────────────────────────

    // 8. status_at: Valid.
    #[test]
    fn test_status_at_valid() {
        let r = make_cert("n", "d", NOW, -30, 60);
        assert_eq!(r.status_at(NOW, 30), CertStatus::Valid);
    }

    // 9. status_at: ExpiringSoon.
    #[test]
    fn test_status_at_expiring_soon() {
        let r = make_cert("n", "d", NOW, -335, 29);
        let status = r.status_at(NOW, 30);
        assert!(
            matches!(status, CertStatus::ExpiringSoon { days_remaining: 29 }),
            "status={status}"
        );
    }

    // 10. status_at: Expired.
    #[test]
    fn test_status_at_expired() {
        let r = make_cert("n", "d", NOW, -365, -1);
        assert_eq!(r.status_at(NOW, 30), CertStatus::Expired);
    }

    // 11. status_at: NotYetValid.
    #[test]
    fn test_status_at_not_yet_valid() {
        let r = make_cert("n", "d", NOW, 1, 365);
        assert_eq!(r.status_at(NOW, 30), CertStatus::NotYetValid);
    }

    // 12. CertStatus Display.
    #[test]
    fn test_cert_status_display() {
        assert_eq!(CertStatus::Valid.to_string(), "valid");
        assert_eq!(CertStatus::Expired.to_string(), "expired");
        assert_eq!(CertStatus::NotYetValid.to_string(), "not_yet_valid");
        assert_eq!(
            CertStatus::ExpiringSoon { days_remaining: 14 }.to_string(),
            "expiring_soon(14d)"
        );
    }

    // ── SslCertManager ────────────────────────────────────────────────────

    // 13. Empty manager has len=0.
    #[test]
    fn test_manager_empty() {
        let mgr = SslCertManager::default();
        assert!(mgr.is_empty());
        assert_eq!(mgr.len(), 0);
    }

    // 14. register inserts a record.
    #[test]
    fn test_manager_register() {
        let mgr = SslCertManager::default();
        let r = make_cert("pop-lax", "lax.cdn.com", NOW, -30, 60);
        mgr.register(r).expect("ok");
        assert_eq!(mgr.len(), 1);
    }

    // 15. get returns the registered record.
    #[test]
    fn test_manager_get() {
        let mgr = SslCertManager::default();
        let r = make_cert("pop-fra", "fra.cdn.com", NOW, -30, 60);
        mgr.register(r).expect("ok");
        let rec = mgr.get("pop-fra").expect("found");
        assert_eq!(rec.domain, "fra.cdn.com");
    }

    // 16. get returns NotFound for unknown node.
    #[test]
    fn test_manager_get_not_found() {
        let mgr = SslCertManager::default();
        let err = mgr.get("ghost").unwrap_err();
        assert!(matches!(err, CertError::NotFound(_)));
    }

    // 17. deregister removes the record.
    #[test]
    fn test_manager_deregister() {
        let mgr = SslCertManager::default();
        mgr.register(make_cert("n", "d", NOW, -30, 60)).expect("ok");
        assert!(mgr.deregister("n"));
        assert!(mgr.is_empty());
        assert!(!mgr.deregister("n")); // already gone
    }

    // 18. register replaces existing record.
    #[test]
    fn test_manager_register_replaces() {
        let mgr = SslCertManager::default();
        mgr.register(make_cert("n", "old.com", NOW, -30, 60))
            .expect("ok");
        mgr.register(make_cert("n", "new.com", NOW, -30, 90))
            .expect("ok");
        let r = mgr.get("n").expect("ok");
        assert_eq!(r.domain, "new.com");
        assert_eq!(mgr.len(), 1);
    }

    // 19. snapshots_at returns correct statuses.
    #[test]
    fn test_snapshots_at() {
        let mgr = SslCertManager::new(CertManagerConfig {
            warning_horizon_days: 30,
            ..CertManagerConfig::default()
        });
        mgr.register(make_cert("valid-node", "valid.com", NOW, -30, 90))
            .expect("ok");
        mgr.register(make_cert("expiring-node", "expiring.com", NOW, -335, 20))
            .expect("ok");
        mgr.register(make_cert("expired-node", "expired.com", NOW, -365, -1))
            .expect("ok");

        let snaps = mgr.snapshots_at(NOW);
        assert_eq!(snaps.len(), 3);

        let find = |id: &str| snaps.iter().find(|s| s.edge_node_id == id).expect("found");
        assert_eq!(find("valid-node").status, CertStatus::Valid);
        assert!(matches!(
            find("expiring-node").status,
            CertStatus::ExpiringSoon { .. }
        ));
        assert_eq!(find("expired-node").status, CertStatus::Expired);
    }

    // 20. status_counts_at sums correctly.
    #[test]
    fn test_status_counts_at() {
        let mgr = SslCertManager::default();
        mgr.register(make_cert("n1", "d1", NOW, -30, 90))
            .expect("ok"); // valid
        mgr.register(make_cert("n2", "d2", NOW, -335, 20))
            .expect("ok"); // expiring
        mgr.register(make_cert("n3", "d3", NOW, -365, -1))
            .expect("ok"); // expired
        mgr.register(make_cert("n4", "d4", NOW, 1, 365))
            .expect("ok"); // not yet valid

        let counts = mgr.status_counts_at(NOW);
        assert_eq!(counts.valid, 1);
        assert_eq!(counts.expiring_soon, 1);
        assert_eq!(counts.expired, 1);
        assert_eq!(counts.not_yet_valid, 1);
    }

    // 21. nodes_with_status_at filters correctly.
    #[test]
    fn test_nodes_with_status_at_expired() {
        let mgr = SslCertManager::default();
        mgr.register(make_cert("good", "d", NOW, -30, 90))
            .expect("ok");
        mgr.register(make_cert("bad", "d", NOW, -365, -1))
            .expect("ok");
        let expired = mgr.nodes_with_status_at(CertStatus::Expired, NOW);
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].edge_node_id, "bad");
    }

    // 22. to_prometheus_at emits correct metric names.
    #[test]
    fn test_to_prometheus_at() {
        let mgr = SslCertManager::default();
        mgr.register(make_cert("pop-tok", "tok.cdn.com", NOW, -30, 60))
            .expect("ok");
        let prom = mgr.to_prometheus_at(NOW);
        assert!(
            prom.contains("ssl_cert_days_remaining"),
            "missing metric: {prom}"
        );
        assert!(prom.contains("pop-tok"), "missing node: {prom}");
        assert!(prom.contains("tok.cdn.com"), "missing domain: {prom}");
        assert!(
            prom.contains("ssl_cert_expired"),
            "missing expired gauge: {prom}"
        );
    }

    // 23. Expired cert shows 0 days_remaining in snapshot.
    #[test]
    fn test_snapshot_expired_zero_days() {
        let mgr = SslCertManager::default();
        mgr.register(make_cert("n", "d", NOW, -365, -1))
            .expect("ok");
        let snaps = mgr.snapshots_at(NOW);
        assert_eq!(snaps[0].days_remaining, 0);
    }

    // 24. Prometheus output is empty when no certs registered.
    #[test]
    fn test_prometheus_empty_manager() {
        let mgr = SslCertManager::default();
        assert!(mgr.to_prometheus_at(NOW).is_empty());
    }

    // 25. Multiple SANs stored correctly.
    #[test]
    fn test_cert_multiple_sans() {
        let r = CertRecord::new("n", "example.com", 0, 1_000)
            .expect("ok")
            .with_san("www.example.com")
            .with_san("api.example.com")
            .with_san("cdn.example.com");
        assert_eq!(r.sans.len(), 3);
    }

    // 26. CertManagerConfig defaults.
    #[test]
    fn test_cert_manager_config_defaults() {
        let cfg = CertManagerConfig::default();
        assert_eq!(cfg.warning_horizon_days, 30);
        assert_eq!(cfg.critical_horizon_days, 7);
    }

    // 27. StatusCounts default is all zero.
    #[test]
    fn test_status_counts_default() {
        let c = StatusCounts::default();
        assert_eq!(c.valid, 0);
        assert_eq!(c.expiring_soon, 0);
        assert_eq!(c.expired, 0);
        assert_eq!(c.not_yet_valid, 0);
    }

    // 28. Prometheus expired metric is 1 for expired cert.
    #[test]
    fn test_prometheus_expired_is_one() {
        let mgr = SslCertManager::default();
        mgr.register(make_cert("edge-expired", "exp.com", NOW, -365, -1))
            .expect("ok");
        let prom = mgr.to_prometheus_at(NOW);
        assert!(
            prom.contains("ssl_cert_expired{node=\"edge-expired\",domain=\"exp.com\"} 1"),
            "prom={prom}"
        );
    }

    // 29. Prometheus expired metric is 0 for valid cert.
    #[test]
    fn test_prometheus_expired_is_zero_for_valid() {
        let mgr = SslCertManager::default();
        mgr.register(make_cert("edge-ok", "ok.com", NOW, -30, 90))
            .expect("ok");
        let prom = mgr.to_prometheus_at(NOW);
        assert!(
            prom.contains("ssl_cert_expired{node=\"edge-ok\",domain=\"ok.com\"} 0"),
            "prom={prom}"
        );
    }

    // 30. CertRecord not_after_unix is preserved in snapshot.
    #[test]
    fn test_snapshot_not_after_unix() {
        let mgr = SslCertManager::default();
        let cert = make_cert("n", "d", NOW, -30, 60);
        let expected_not_after = cert.not_after_unix;
        mgr.register(cert).expect("ok");
        let snaps = mgr.snapshots_at(NOW);
        assert_eq!(snaps[0].not_after_unix, expected_not_after);
    }
}

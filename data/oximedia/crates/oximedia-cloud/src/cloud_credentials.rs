#![allow(dead_code)]
//! Cloud credential management.
//!
//! Provides [`CloudCredential`] — a provider-agnostic credential record — and
//! [`CredentialStore`] for storing, retrieving, and rotating credentials keyed
//! by a logical name.  Expiry tracking lets callers pro-actively refresh
//! tokens before they become invalid.

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ─────────────────────────────────────────────────────────────────────────────
// CredentialType
// ─────────────────────────────────────────────────────────────────────────────

/// The kind of cloud credential.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CredentialType {
    /// Static access-key / secret-key pair (AWS-style).
    StaticKey,
    /// Short-lived bearer token (OAuth 2.0 / STS assume-role).
    BearerToken,
    /// Service-account JSON key (GCP style).
    ServiceAccountKey,
    /// Azure shared access signature.
    SharedAccessSignature,
    /// Instance / workload identity — no explicit secret required.
    InstanceIdentity,
    /// Custom credential format identified by an arbitrary tag.
    Custom(String),
}

impl CredentialType {
    /// Human-readable description.
    pub fn description(&self) -> &str {
        match self {
            Self::StaticKey => "Static access-key",
            Self::BearerToken => "Bearer token",
            Self::ServiceAccountKey => "Service-account key",
            Self::SharedAccessSignature => "Shared access signature",
            Self::InstanceIdentity => "Instance identity",
            Self::Custom(_) => "Custom",
        }
    }

    /// Returns `true` for credential types that typically have a short TTL.
    pub fn is_ephemeral(&self) -> bool {
        matches!(
            self,
            Self::BearerToken | Self::SharedAccessSignature | Self::InstanceIdentity
        )
    }
}

impl std::fmt::Display for CredentialType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.description())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CloudCredential
// ─────────────────────────────────────────────────────────────────────────────

/// A provider-agnostic cloud credential.
///
/// The `secret` field contains the raw credential material (token, key, …).
/// For [`CredentialType::InstanceIdentity`] credentials the secret may be an
/// empty string.
#[derive(Debug, Clone)]
pub struct CloudCredential {
    /// Logical name / alias for this credential.
    pub name: String,
    /// The credential kind.
    pub kind: CredentialType,
    /// Primary key or principal identifier (access key ID, client ID, …).
    pub key_id: String,
    /// Secret material (secret key, token, …).
    pub secret: String,
    /// When this credential was issued / last refreshed.
    pub issued_at: Instant,
    /// How long the credential is valid for.  `None` means indefinite.
    pub ttl: Option<Duration>,
    /// Optional metadata tags.
    pub tags: HashMap<String, String>,
}

impl CloudCredential {
    /// Create a new credential that never expires.
    pub fn new(
        name: impl Into<String>,
        kind: CredentialType,
        key_id: impl Into<String>,
        secret: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            kind,
            key_id: key_id.into(),
            secret: secret.into(),
            issued_at: Instant::now(),
            ttl: None,
            tags: HashMap::new(),
        }
    }

    /// Attach a time-to-live to this credential.
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    /// Attach a metadata tag.
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }

    /// Returns `true` when the credential's TTL has elapsed.
    pub fn is_expired(&self) -> bool {
        match self.ttl {
            Some(ttl) => self.issued_at.elapsed() >= ttl,
            None => false,
        }
    }

    /// Seconds remaining until expiry, or `None` if no TTL is set.
    pub fn seconds_remaining(&self) -> Option<f64> {
        self.ttl.map(|ttl| {
            let elapsed = self.issued_at.elapsed().as_secs_f64();
            (ttl.as_secs_f64() - elapsed).max(0.0)
        })
    }

    /// Returns `true` when the credential will expire within `window`.
    pub fn expires_within(&self, window: Duration) -> bool {
        match self.ttl {
            Some(ttl) => {
                let remaining = ttl.saturating_sub(self.issued_at.elapsed());
                remaining <= window
            }
            None => false,
        }
    }

    /// Re-issue the credential with fresh material, resetting the issue time.
    pub fn refresh(&mut self, new_secret: impl Into<String>) {
        self.secret = new_secret.into();
        self.issued_at = Instant::now();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CredentialStore
// ─────────────────────────────────────────────────────────────────────────────

/// Errors returned by [`CredentialStore`] operations.
#[derive(Debug, PartialEq, Eq)]
pub enum CredentialStoreError {
    /// No credential is registered under the given name.
    NotFound(String),
    /// A credential with the same name already exists.
    AlreadyExists(String),
    /// The credential is expired and cannot be used.
    Expired(String),
}

impl std::fmt::Display for CredentialStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(n) => write!(f, "credential '{n}' not found"),
            Self::AlreadyExists(n) => write!(f, "credential '{n}' already exists"),
            Self::Expired(n) => write!(f, "credential '{n}' has expired"),
        }
    }
}

/// In-memory registry for named [`CloudCredential`] objects.
///
/// ```rust
/// use oximedia_cloud::cloud_credentials::{
///     CloudCredential, CredentialStore, CredentialType,
/// };
///
/// let mut store = CredentialStore::new();
/// let cred = CloudCredential::new("prod-s3", CredentialType::StaticKey, "AKIA…", "secret");
/// store.register(cred).expect("register must succeed");
///
/// let found = store.get("prod-s3").expect("lookup must succeed");
/// assert_eq!(found.key_id, "AKIA…");
/// ```
pub struct CredentialStore {
    entries: HashMap<String, CloudCredential>,
    /// Stored refresh hooks keyed by credential name.
    refreshers: HashMap<String, Box<dyn Fn() -> CloudCredential + Send + Sync>>,
}

impl std::fmt::Debug for CredentialStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CredentialStore")
            .field("entries", &self.entries)
            .field("refreshers_count", &self.refreshers.len())
            .finish()
    }
}

impl Default for CredentialStore {
    fn default() -> Self {
        Self::new()
    }
}

impl CredentialStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            refreshers: HashMap::new(),
        }
    }

    /// Register a credential.  Returns an error if the name is already taken.
    pub fn register(&mut self, cred: CloudCredential) -> Result<(), CredentialStoreError> {
        if self.entries.contains_key(&cred.name) {
            return Err(CredentialStoreError::AlreadyExists(cred.name.clone()));
        }
        self.entries.insert(cred.name.clone(), cred);
        Ok(())
    }

    /// Register a credential together with a refresh hook.
    ///
    /// The `refresher` closure is called automatically by [`get_fresh`] when the
    /// stored credential will expire within the requested `refresh_window`.
    ///
    /// Returns an error if a credential with the same name already exists.
    ///
    /// [`get_fresh`]: CredentialStore::get_fresh
    pub fn register_with_refresher<F>(
        &mut self,
        cred: CloudCredential,
        refresher: F,
    ) -> Result<(), CredentialStoreError>
    where
        F: Fn() -> CloudCredential + Send + Sync + 'static,
    {
        let name = cred.name.clone();
        self.register(cred)?;
        self.refreshers.insert(name, Box::new(refresher));
        Ok(())
    }

    /// Retrieve a credential, proactively refreshing it when it falls inside
    /// `refresh_window`.
    ///
    /// If a refresh hook was registered via [`register_with_refresher`] and the
    /// credential will expire within `refresh_window` (or is already expired),
    /// the hook is invoked and the returned credential is the newly refreshed
    /// one.  If no hook is registered the current credential is returned as-is
    /// (the caller may still receive a [`CredentialStoreError::Expired`] error if the
    /// credential has already elapsed).
    ///
    /// Returns [`CredentialStoreError::NotFound`] when no credential exists
    /// under `name`.
    ///
    /// [`register_with_refresher`]: CredentialStore::register_with_refresher
    pub fn get_fresh(
        &mut self,
        name: &str,
        refresh_window: Duration,
    ) -> Result<&CloudCredential, CredentialStoreError> {
        // Determine whether a refresh is warranted.
        let needs_refresh = self
            .entries
            .get(name)
            .map(|c| c.expires_within(refresh_window) || c.is_expired())
            .unwrap_or(false);

        if needs_refresh {
            // Invoke the refresher outside of the borrow on `self.entries`.
            if let Some(refresher) = self.refreshers.get(name) {
                let new_cred = refresher();
                self.entries.insert(name.to_string(), new_cred);
            }
        }

        // Delegate to the regular (expiry-checking) get.
        self.entries
            .get(name)
            .ok_or_else(|| CredentialStoreError::NotFound(name.to_string()))
    }

    /// Retrieve a credential by name.
    ///
    /// Returns [`CredentialStoreError::Expired`] if the credential exists but
    /// its TTL has elapsed.
    pub fn get(&self, name: &str) -> Result<&CloudCredential, CredentialStoreError> {
        match self.entries.get(name) {
            None => Err(CredentialStoreError::NotFound(name.to_string())),
            Some(cred) if cred.is_expired() => Err(CredentialStoreError::Expired(name.to_string())),
            Some(cred) => Ok(cred),
        }
    }

    /// Retrieve a credential for mutation (e.g. to call [`CloudCredential::refresh`]).
    pub fn get_mut(&mut self, name: &str) -> Result<&mut CloudCredential, CredentialStoreError> {
        match self.entries.get_mut(name) {
            None => Err(CredentialStoreError::NotFound(name.to_string())),
            Some(cred) => Ok(cred),
        }
    }

    /// Upsert a credential, replacing any existing entry with the same name.
    pub fn upsert(&mut self, cred: CloudCredential) {
        self.entries.insert(cred.name.clone(), cred);
    }

    /// Remove a credential, returning it if it existed.
    pub fn remove(&mut self, name: &str) -> Option<CloudCredential> {
        self.entries.remove(name)
    }

    /// Number of credentials in the store (including expired ones).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the store is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove all expired credentials and return the count removed.
    pub fn purge_expired(&mut self) -> usize {
        let before = self.entries.len();
        self.entries.retain(|_, cred| !cred.is_expired());
        before - self.entries.len()
    }

    /// Names of all credentials that will expire within `window`.
    pub fn expiring_within(&self, window: Duration) -> Vec<&str> {
        self.entries
            .values()
            .filter(|c| c.expires_within(window))
            .map(|c| c.name.as_str())
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cred(name: &str) -> CloudCredential {
        CloudCredential::new(name, CredentialType::StaticKey, "key_id", "secret")
    }

    #[test]
    fn test_credential_type_description() {
        assert!(CredentialType::StaticKey.description().contains("Static"));
        assert!(CredentialType::BearerToken.description().contains("Bearer"));
    }

    #[test]
    fn test_credential_type_display() {
        assert_eq!(
            CredentialType::ServiceAccountKey.to_string(),
            "Service-account key"
        );
    }

    #[test]
    fn test_credential_type_is_ephemeral() {
        assert!(CredentialType::BearerToken.is_ephemeral());
        assert!(!CredentialType::StaticKey.is_ephemeral());
        assert!(CredentialType::InstanceIdentity.is_ephemeral());
    }

    #[test]
    fn test_credential_not_expired_by_default() {
        let cred = make_cred("test");
        assert!(!cred.is_expired());
    }

    #[test]
    fn test_credential_with_long_ttl_not_expired() {
        let cred = make_cred("test").with_ttl(Duration::from_secs(3600));
        assert!(!cred.is_expired());
    }

    #[test]
    fn test_credential_seconds_remaining_none_when_no_ttl() {
        let cred = make_cred("test");
        assert!(cred.seconds_remaining().is_none());
    }

    #[test]
    fn test_credential_seconds_remaining_positive() {
        let cred = make_cred("test").with_ttl(Duration::from_secs(3600));
        let rem = cred.seconds_remaining().expect("rem should be valid");
        assert!(rem > 3590.0 && rem <= 3600.0);
    }

    #[test]
    fn test_credential_does_not_expire_within_short_window() {
        let cred = make_cred("test").with_ttl(Duration::from_secs(3600));
        assert!(!cred.expires_within(Duration::from_secs(60)));
    }

    #[test]
    fn test_credential_with_tag() {
        let cred = make_cred("test").with_tag("env", "prod");
        assert_eq!(cred.tags.get("env").map(String::as_str), Some("prod"));
    }

    #[test]
    fn test_credential_refresh_updates_secret() {
        let mut cred = make_cred("test");
        cred.refresh("new_secret");
        assert_eq!(cred.secret, "new_secret");
    }

    #[test]
    fn test_store_register_and_get() {
        let mut store = CredentialStore::new();
        store
            .register(make_cred("alpha"))
            .expect("test expectation failed");
        let cred = store.get("alpha").expect("cred should be valid");
        assert_eq!(cred.name, "alpha");
    }

    #[test]
    fn test_store_duplicate_register_fails() {
        let mut store = CredentialStore::new();
        store
            .register(make_cred("dup"))
            .expect("test expectation failed");
        let err = store.register(make_cred("dup")).unwrap_err();
        assert!(matches!(err, CredentialStoreError::AlreadyExists(_)));
    }

    #[test]
    fn test_store_not_found_error() {
        let store = CredentialStore::new();
        assert!(matches!(
            store.get("missing"),
            Err(CredentialStoreError::NotFound(_))
        ));
    }

    #[test]
    fn test_store_upsert_replaces() {
        let mut store = CredentialStore::new();
        store
            .register(make_cred("x"))
            .expect("test expectation failed");
        let updated =
            CloudCredential::new("x", CredentialType::BearerToken, "new_key", "new_secret");
        store.upsert(updated);
        let cred = store.get("x").expect("cred should be valid");
        assert!(matches!(cred.kind, CredentialType::BearerToken));
    }

    #[test]
    fn test_store_remove() {
        let mut store = CredentialStore::new();
        store
            .register(make_cred("rm"))
            .expect("test expectation failed");
        let removed = store.remove("rm");
        assert!(removed.is_some());
        assert!(store.is_empty());
    }

    #[test]
    fn test_store_len_and_is_empty() {
        let mut store = CredentialStore::new();
        assert!(store.is_empty());
        store
            .register(make_cred("a"))
            .expect("test expectation failed");
        store
            .register(make_cred("b"))
            .expect("test expectation failed");
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_credential_store_error_display() {
        assert!(CredentialStoreError::NotFound("k".to_string())
            .to_string()
            .contains("not found"));
        assert!(CredentialStoreError::Expired("k".to_string())
            .to_string()
            .contains("expired"));
        assert!(CredentialStoreError::AlreadyExists("k".to_string())
            .to_string()
            .contains("already exists"));
    }

    // ── get_fresh tests ──────────────────────────────────────────────────────

    #[test]
    fn test_get_fresh_returns_existing_when_not_near_expiry() {
        use std::sync::{Arc, Mutex};

        let call_count = Arc::new(Mutex::new(0u32));
        let counter = Arc::clone(&call_count);

        let mut store = CredentialStore::new();
        let cred = make_cred("long-lived").with_ttl(Duration::from_secs(3600));
        store
            .register_with_refresher(cred, move || {
                let mut n = counter.lock().expect("lock must not be poisoned");
                *n += 1;
                CloudCredential::new(
                    "long-lived",
                    CredentialType::StaticKey,
                    "refreshed_key",
                    "refreshed_secret",
                )
                .with_ttl(Duration::from_secs(3600))
            })
            .expect("register_with_refresher must succeed");

        // refresh_window is only 5 seconds; credential has 3600 s remaining.
        let found = store
            .get_fresh("long-lived", Duration::from_secs(5))
            .expect("should return credential");
        assert_eq!(found.key_id, "key_id");
        assert_eq!(*call_count.lock().expect("lock must not be poisoned"), 0);
    }

    #[test]
    fn test_get_fresh_calls_refresher_when_within_window() {
        use std::sync::{Arc, Mutex};

        let call_count = Arc::new(Mutex::new(0u32));
        let counter = Arc::clone(&call_count);

        let mut store = CredentialStore::new();
        // Credential expires in 5 seconds — well within a 10 s window.
        let cred = make_cred("short-lived").with_ttl(Duration::from_secs(5));
        store
            .register_with_refresher(cred, move || {
                let mut n = counter.lock().expect("lock must not be poisoned");
                *n += 1;
                CloudCredential::new(
                    "short-lived",
                    CredentialType::BearerToken,
                    "new_key",
                    "new_token",
                )
                .with_ttl(Duration::from_secs(3600))
            })
            .expect("register_with_refresher must succeed");

        let found = store
            .get_fresh("short-lived", Duration::from_secs(10))
            .expect("should return refreshed credential");

        assert_eq!(found.key_id, "new_key");
        assert_eq!(found.secret, "new_token");
        assert_eq!(*call_count.lock().expect("lock must not be poisoned"), 1);
    }

    #[test]
    fn test_get_fresh_no_refresher_returns_existing_cred() {
        let mut store = CredentialStore::new();
        store
            .register(make_cred("no-refresher").with_ttl(Duration::from_secs(3600)))
            .expect("register must succeed");

        let found = store
            .get_fresh("no-refresher", Duration::from_secs(60))
            .expect("should return existing credential");
        assert_eq!(found.name, "no-refresher");
    }

    #[test]
    fn test_get_fresh_expired_cred_with_refresher_refreshes() {
        use std::sync::{Arc, Mutex};

        let call_count = Arc::new(Mutex::new(0u32));
        let counter = Arc::clone(&call_count);

        let mut store = CredentialStore::new();
        // TTL of 1 ns guarantees the credential is expired immediately.
        let cred = make_cred("expired-cred").with_ttl(Duration::from_nanos(1));
        store
            .register_with_refresher(cred, move || {
                let mut n = counter.lock().expect("lock must not be poisoned");
                *n += 1;
                CloudCredential::new(
                    "expired-cred",
                    CredentialType::BearerToken,
                    "fresh_key",
                    "fresh_token",
                )
                .with_ttl(Duration::from_secs(3600))
            })
            .expect("register_with_refresher must succeed");

        let found = store
            .get_fresh("expired-cred", Duration::from_secs(30))
            .expect("should return freshly minted credential");

        assert_eq!(found.key_id, "fresh_key");
        assert_eq!(*call_count.lock().expect("lock must not be poisoned"), 1);
    }

    #[test]
    fn test_get_fresh_unknown_name_returns_error() {
        let mut store = CredentialStore::new();
        let err = store
            .get_fresh("nonexistent", Duration::from_secs(60))
            .unwrap_err();
        assert!(matches!(err, CredentialStoreError::NotFound(_)));
    }
}

//! Authentication middleware for the media server.
//!
//! Provides `AuthScheme`, `AuthToken`, `AuthMiddleware`, and `AuthResult`
//! for validating incoming HTTP requests.

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ── AuthScheme ────────────────────────────────────────────────────────────────

/// Supported HTTP authentication schemes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthScheme {
    /// RFC 6750 bearer token.
    Bearer,
    /// HTTP Basic authentication (base64 encoded user:pass).
    Basic,
    /// Proprietary API-key header scheme.
    ApiKey,
    /// Digest authentication (RFC 7616).
    Digest,
}

impl AuthScheme {
    /// Returns `true` when this scheme uses a bearer token.
    pub fn is_bearer(&self) -> bool {
        matches!(self, Self::Bearer)
    }

    /// Returns `true` when this scheme uses plain credentials (Basic / ApiKey).
    pub fn is_credential_based(&self) -> bool {
        matches!(self, Self::Basic | Self::ApiKey)
    }

    /// Parse a scheme from the `Authorization` header prefix string.
    pub fn from_header_prefix(prefix: &str) -> Option<Self> {
        match prefix.trim().to_lowercase().as_str() {
            "bearer" => Some(Self::Bearer),
            "basic" => Some(Self::Basic),
            "apikey" => Some(Self::ApiKey),
            "digest" => Some(Self::Digest),
            _ => None,
        }
    }
}

// ── AuthToken ────────────────────────────────────────────────────────────────

/// A parsed authentication token with expiry tracking.
#[derive(Debug, Clone)]
pub struct AuthToken {
    /// Raw token string.
    pub value: String,
    /// Associated subject (user ID or service name).
    pub subject: String,
    /// Unix timestamp (seconds) when this token expires. `None` = never expires.
    pub expires_at: Option<u64>,
    /// Granted permission scopes.
    pub scopes: Vec<String>,
}

impl AuthToken {
    /// Create a new token.
    pub fn new(value: impl Into<String>, subject: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            subject: subject.into(),
            expires_at: None,
            scopes: Vec::new(),
        }
    }

    /// Attach an expiry duration relative to now.
    #[must_use]
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.expires_at = Some(now + ttl.as_secs());
        self
    }

    /// Add a scope to the token.
    #[must_use]
    pub fn with_scope(mut self, scope: impl Into<String>) -> Self {
        self.scopes.push(scope.into());
        self
    }

    /// Returns `true` if the token has expired at `unix_now` (seconds).
    pub fn is_expired_at(&self, unix_now: u64) -> bool {
        self.expires_at.is_some_and(|exp| unix_now >= exp)
    }

    /// Returns `true` if the token is currently expired.
    pub fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.is_expired_at(now)
    }

    /// Returns `true` if the token carries the given scope.
    pub fn has_scope(&self, scope: &str) -> bool {
        self.scopes.iter().any(|s| s == scope)
    }
}

// ── AuthResult ───────────────────────────────────────────────────────────────

/// Outcome of an authentication validation attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthResult {
    /// The request is authenticated and authorized.
    Authorized {
        /// Authenticated subject (user ID or service name).
        subject: String,
    },
    /// The token / credential was recognised but is expired.
    Expired,
    /// The scheme or credential format was invalid.
    InvalidCredential,
    /// A required scope was missing.
    InsufficientScope {
        /// The scope that was required but not present.
        required: String,
    },
    /// No `Authorization` header was present.
    Missing,
}

impl AuthResult {
    /// Returns `true` when the result is `Authorized`.
    pub fn is_authorized(&self) -> bool {
        matches!(self, Self::Authorized { .. })
    }

    /// Returns `true` for any failure variant.
    pub fn is_failure(&self) -> bool {
        !self.is_authorized()
    }

    /// Convenience: extract the subject string, or `None` on failure.
    pub fn subject(&self) -> Option<&str> {
        if let Self::Authorized { subject } = self {
            Some(subject.as_str())
        } else {
            None
        }
    }
}

// ── AuthMiddleware ───────────────────────────────────────────────────────────

/// Configuration for the authentication middleware.
#[derive(Debug, Clone)]
pub struct AuthMiddlewareConfig {
    /// Which scheme this middleware handles.
    pub scheme: AuthScheme,
    /// Required scopes for every validated request.
    pub required_scopes: Vec<String>,
    /// Whether to allow requests that have no `Authorization` header.
    pub allow_anonymous: bool,
}

impl Default for AuthMiddlewareConfig {
    fn default() -> Self {
        Self {
            scheme: AuthScheme::Bearer,
            required_scopes: Vec::new(),
            allow_anonymous: false,
        }
    }
}

/// In-process authentication middleware.
///
/// Holds a registry of known tokens and validates incoming requests.
pub struct AuthMiddleware {
    config: AuthMiddlewareConfig,
    /// token_value → AuthToken
    registry: HashMap<String, AuthToken>,
}

impl AuthMiddleware {
    /// Create a new middleware with the given configuration.
    pub fn new(config: AuthMiddlewareConfig) -> Self {
        Self {
            config,
            registry: HashMap::new(),
        }
    }

    /// Register a known token so that `validate` can look it up.
    pub fn register_token(&mut self, token: AuthToken) {
        self.registry.insert(token.value.clone(), token);
    }

    /// Revoke a previously registered token.
    pub fn revoke_token(&mut self, value: &str) -> bool {
        self.registry.remove(value).is_some()
    }

    /// Validate the raw value extracted from the `Authorization` header.
    ///
    /// Returns an `AuthResult` indicating the outcome.
    pub fn validate(&self, raw_value: Option<&str>) -> AuthResult {
        let value = match raw_value {
            Some(v) if !v.is_empty() => v,
            _ => {
                return if self.config.allow_anonymous {
                    AuthResult::Authorized {
                        subject: "anonymous".to_string(),
                    }
                } else {
                    AuthResult::Missing
                };
            }
        };

        let Some(token) = self.registry.get(value) else {
            return AuthResult::InvalidCredential;
        };

        if token.is_expired() {
            return AuthResult::Expired;
        }

        for scope in &self.config.required_scopes {
            if !token.has_scope(scope) {
                return AuthResult::InsufficientScope {
                    required: scope.clone(),
                };
            }
        }

        AuthResult::Authorized {
            subject: token.subject.clone(),
        }
    }

    /// Number of registered tokens.
    pub fn token_count(&self) -> usize {
        self.registry.len()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_middleware() -> AuthMiddleware {
        AuthMiddleware::new(AuthMiddlewareConfig::default())
    }

    // AuthScheme

    #[test]
    fn scheme_bearer_is_bearer() {
        assert!(AuthScheme::Bearer.is_bearer());
    }

    #[test]
    fn scheme_basic_not_bearer() {
        assert!(!AuthScheme::Basic.is_bearer());
    }

    #[test]
    fn scheme_credential_based() {
        assert!(AuthScheme::Basic.is_credential_based());
        assert!(AuthScheme::ApiKey.is_credential_based());
        assert!(!AuthScheme::Bearer.is_credential_based());
    }

    #[test]
    fn scheme_from_header_prefix() {
        assert_eq!(
            AuthScheme::from_header_prefix("Bearer"),
            Some(AuthScheme::Bearer)
        );
        assert_eq!(
            AuthScheme::from_header_prefix("basic"),
            Some(AuthScheme::Basic)
        );
        assert_eq!(
            AuthScheme::from_header_prefix("APIKEY"),
            Some(AuthScheme::ApiKey)
        );
        assert_eq!(AuthScheme::from_header_prefix("unknown"), None);
    }

    // AuthToken

    #[test]
    fn token_not_expired_without_ttl() {
        let token = AuthToken::new("tok", "user1");
        assert!(!token.is_expired());
        assert!(!token.is_expired_at(9_999_999_999));
    }

    #[test]
    fn token_expired_at_past_timestamp() {
        let token = AuthToken::new("tok", "user1").with_ttl(Duration::from_secs(1));
        // A timestamp far in the future should mark it expired if expires_at <= unix_now.
        // We set expires_at = now+1, so now+100 is past it.
        let far_future = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("should succeed in test")
            .as_secs()
            + 100;
        assert!(token.is_expired_at(far_future));
    }

    #[test]
    fn token_not_expired_before_expiry() {
        let token = AuthToken::new("tok", "user1").with_ttl(Duration::from_secs(3600));
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("should succeed in test")
            .as_secs();
        assert!(!token.is_expired_at(now));
    }

    #[test]
    fn token_scope_check() {
        let token = AuthToken::new("tok", "u")
            .with_scope("read")
            .with_scope("write");
        assert!(token.has_scope("read"));
        assert!(token.has_scope("write"));
        assert!(!token.has_scope("admin"));
    }

    // AuthResult

    #[test]
    fn auth_result_authorized_is_authorized() {
        let r = AuthResult::Authorized {
            subject: "u1".into(),
        };
        assert!(r.is_authorized());
        assert!(!r.is_failure());
        assert_eq!(r.subject(), Some("u1"));
    }

    #[test]
    fn auth_result_expired_is_failure() {
        let r = AuthResult::Expired;
        assert!(!r.is_authorized());
        assert!(r.is_failure());
        assert_eq!(r.subject(), None);
    }

    #[test]
    fn auth_result_missing_is_failure() {
        assert!(AuthResult::Missing.is_failure());
    }

    // AuthMiddleware

    #[test]
    fn middleware_missing_without_anonymous() {
        let mw = make_middleware();
        assert_eq!(mw.validate(None), AuthResult::Missing);
        assert_eq!(mw.validate(Some("")), AuthResult::Missing);
    }

    #[test]
    fn middleware_anonymous_allowed_when_configured() {
        let cfg = AuthMiddlewareConfig {
            allow_anonymous: true,
            ..Default::default()
        };
        let mw = AuthMiddleware::new(cfg);
        assert!(mw.validate(None).is_authorized());
    }

    #[test]
    fn middleware_invalid_credential() {
        let mw = make_middleware();
        let result = mw.validate(Some("bad-token"));
        assert_eq!(result, AuthResult::InvalidCredential);
    }

    #[test]
    fn middleware_valid_token_authorized() {
        let mut mw = make_middleware();
        let token = AuthToken::new("secret123", "alice");
        mw.register_token(token);
        let result = mw.validate(Some("secret123"));
        assert!(result.is_authorized());
        assert_eq!(result.subject(), Some("alice"));
    }

    #[test]
    fn middleware_revoke_token() {
        let mut mw = make_middleware();
        mw.register_token(AuthToken::new("tok", "bob"));
        assert_eq!(mw.token_count(), 1);
        assert!(mw.revoke_token("tok"));
        assert_eq!(mw.token_count(), 0);
        assert_eq!(mw.validate(Some("tok")), AuthResult::InvalidCredential);
    }

    #[test]
    fn middleware_scope_enforcement() {
        let cfg = AuthMiddlewareConfig {
            required_scopes: vec!["admin".into()],
            ..Default::default()
        };
        let mut mw = AuthMiddleware::new(cfg);
        mw.register_token(AuthToken::new("tok", "user").with_scope("read"));
        let result = mw.validate(Some("tok"));
        assert!(matches!(result, AuthResult::InsufficientScope { .. }));
    }

    #[test]
    fn middleware_scope_satisfied() {
        let cfg = AuthMiddlewareConfig {
            required_scopes: vec!["read".into()],
            ..Default::default()
        };
        let mut mw = AuthMiddleware::new(cfg);
        mw.register_token(AuthToken::new("tok", "user").with_scope("read"));
        assert!(mw.validate(Some("tok")).is_authorized());
    }
}

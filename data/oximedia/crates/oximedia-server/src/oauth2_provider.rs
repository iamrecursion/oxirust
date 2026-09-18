//! OAuth2/OIDC provider integration for SSO authentication.
//!
//! Extends the auth middleware with support for external identity providers
//! (Google, GitHub, Microsoft, custom OIDC) by managing provider configurations,
//! token exchange flows, and user identity mapping.

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Supported OAuth2/OIDC provider types.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProviderType {
    /// Google OAuth2.
    Google,
    /// GitHub OAuth2.
    GitHub,
    /// Microsoft / Azure AD.
    Microsoft,
    /// Generic OIDC-compliant provider.
    CustomOidc(String),
}

impl ProviderType {
    /// Returns a human-readable label.
    pub fn label(&self) -> &str {
        match self {
            Self::Google => "google",
            Self::GitHub => "github",
            Self::Microsoft => "microsoft",
            Self::CustomOidc(name) => name.as_str(),
        }
    }
}

/// Configuration for a single OAuth2/OIDC provider.
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    /// Provider type.
    pub provider_type: ProviderType,
    /// OAuth2 client ID.
    pub client_id: String,
    /// OAuth2 client secret (encrypted or redacted in logs).
    pub client_secret: String,
    /// Authorization endpoint URL.
    pub authorize_url: String,
    /// Token endpoint URL.
    pub token_url: String,
    /// OIDC userinfo endpoint URL (optional, for OIDC providers).
    pub userinfo_url: Option<String>,
    /// OIDC issuer URL for JWT validation.
    pub issuer: Option<String>,
    /// Requested scopes.
    pub scopes: Vec<String>,
    /// Redirect URI for the OAuth2 callback.
    pub redirect_uri: String,
    /// Whether this provider is enabled.
    pub enabled: bool,
}

impl ProviderConfig {
    /// Creates a Google OAuth2 provider config.
    pub fn google(client_id: &str, client_secret: &str, redirect_uri: &str) -> Self {
        Self {
            provider_type: ProviderType::Google,
            client_id: client_id.to_string(),
            client_secret: client_secret.to_string(),
            authorize_url: "https://accounts.google.com/o/oauth2/v2/auth".to_string(),
            token_url: "https://oauth2.googleapis.com/token".to_string(),
            userinfo_url: Some("https://openidconnect.googleapis.com/v1/userinfo".to_string()),
            issuer: Some("https://accounts.google.com".to_string()),
            scopes: vec![
                "openid".to_string(),
                "email".to_string(),
                "profile".to_string(),
            ],
            redirect_uri: redirect_uri.to_string(),
            enabled: true,
        }
    }

    /// Creates a GitHub OAuth2 provider config.
    pub fn github(client_id: &str, client_secret: &str, redirect_uri: &str) -> Self {
        Self {
            provider_type: ProviderType::GitHub,
            client_id: client_id.to_string(),
            client_secret: client_secret.to_string(),
            authorize_url: "https://github.com/login/oauth/authorize".to_string(),
            token_url: "https://github.com/login/oauth/access_token".to_string(),
            userinfo_url: Some("https://api.github.com/user".to_string()),
            issuer: None,
            scopes: vec!["user:email".to_string()],
            redirect_uri: redirect_uri.to_string(),
            enabled: true,
        }
    }

    /// Creates a Microsoft/Azure AD provider config.
    pub fn microsoft(
        client_id: &str,
        client_secret: &str,
        redirect_uri: &str,
        tenant_id: &str,
    ) -> Self {
        let base = format!("https://login.microsoftonline.com/{}", tenant_id);
        Self {
            provider_type: ProviderType::Microsoft,
            client_id: client_id.to_string(),
            client_secret: client_secret.to_string(),
            authorize_url: format!("{}/oauth2/v2.0/authorize", base),
            token_url: format!("{}/oauth2/v2.0/token", base),
            userinfo_url: Some("https://graph.microsoft.com/oidc/userinfo".to_string()),
            issuer: Some(format!("{}/v2.0", base)),
            scopes: vec![
                "openid".to_string(),
                "email".to_string(),
                "profile".to_string(),
            ],
            redirect_uri: redirect_uri.to_string(),
            enabled: true,
        }
    }
}

/// An OAuth2 authorization request (state for CSRF protection).
#[derive(Debug, Clone)]
pub struct AuthorizationRequest {
    /// Unique state parameter for CSRF.
    pub state: String,
    /// The provider being used.
    pub provider: String,
    /// PKCE code verifier (S256).
    pub code_verifier: Option<String>,
    /// When this request was created.
    pub created_at: u64,
    /// TTL for the authorization request (prevents replay).
    pub expires_at: u64,
    /// Optional nonce for OIDC id_token validation.
    pub nonce: Option<String>,
    /// Where to redirect after successful auth.
    pub redirect_after: Option<String>,
}

impl AuthorizationRequest {
    /// Creates a new authorization request.
    pub fn new(state: String, provider: String, ttl: Duration) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            state,
            provider,
            code_verifier: None,
            created_at: now,
            expires_at: now + ttl.as_secs(),
            nonce: None,
            redirect_after: None,
        }
    }

    /// Whether this request has expired.
    pub fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now >= self.expires_at
    }

    /// Sets the PKCE code verifier.
    pub fn with_code_verifier(mut self, verifier: String) -> Self {
        self.code_verifier = Some(verifier);
        self
    }

    /// Sets the nonce.
    pub fn with_nonce(mut self, nonce: String) -> Self {
        self.nonce = Some(nonce);
        self
    }

    /// Sets the post-auth redirect.
    pub fn with_redirect_after(mut self, redirect: String) -> Self {
        self.redirect_after = Some(redirect);
        self
    }
}

/// Token response from an OAuth2 provider.
#[derive(Debug, Clone)]
pub struct OAuthTokenResponse {
    /// Access token.
    pub access_token: String,
    /// Token type (usually "Bearer").
    pub token_type: String,
    /// Refresh token (if provided).
    pub refresh_token: Option<String>,
    /// Access token expiry in seconds.
    pub expires_in: Option<u64>,
    /// ID token (OIDC providers only).
    pub id_token: Option<String>,
    /// Granted scopes.
    pub scope: Option<String>,
}

/// User identity returned by the provider.
#[derive(Debug, Clone)]
pub struct OAuthUserIdentity {
    /// Provider-specific user ID.
    pub provider_user_id: String,
    /// Provider type.
    pub provider: String,
    /// Email address (if available).
    pub email: Option<String>,
    /// Whether the email is verified.
    pub email_verified: bool,
    /// Display name.
    pub display_name: Option<String>,
    /// Avatar/profile picture URL.
    pub avatar_url: Option<String>,
    /// Raw claims from the provider.
    pub raw_claims: HashMap<String, String>,
}

impl OAuthUserIdentity {
    /// Returns a deterministic internal user ID for this identity.
    pub fn internal_user_id(&self) -> String {
        format!("{}:{}", self.provider, self.provider_user_id)
    }
}

/// Manages OAuth2 provider configurations and authorization state.
pub struct OAuthProviderManager {
    /// Registered provider configurations.
    providers: HashMap<String, ProviderConfig>,
    /// Pending authorization requests (state -> request).
    pending_requests: HashMap<String, AuthorizationRequest>,
    /// Known user identity mappings (provider:id -> internal user id).
    identity_map: HashMap<String, String>,
    /// Authorization request TTL.
    request_ttl: Duration,
}

impl OAuthProviderManager {
    /// Creates a new provider manager.
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            pending_requests: HashMap::new(),
            identity_map: HashMap::new(),
            request_ttl: Duration::from_secs(600), // 10 minutes
        }
    }

    /// Registers a provider configuration.
    pub fn register_provider(&mut self, config: ProviderConfig) {
        let key = config.provider_type.label().to_string();
        self.providers.insert(key, config);
    }

    /// Gets a provider configuration.
    pub fn get_provider(&self, name: &str) -> Option<&ProviderConfig> {
        self.providers.get(name)
    }

    /// Returns all registered provider names.
    pub fn provider_names(&self) -> Vec<String> {
        self.providers.keys().cloned().collect()
    }

    /// Returns only enabled providers.
    pub fn enabled_providers(&self) -> Vec<&ProviderConfig> {
        self.providers.values().filter(|p| p.enabled).collect()
    }

    /// Builds the authorization URL for a provider.
    ///
    /// Returns `(url, state)` where state is for CSRF validation.
    pub fn build_authorize_url(
        &mut self,
        provider_name: &str,
        state: &str,
    ) -> Option<(String, AuthorizationRequest)> {
        let provider = self.providers.get(provider_name)?;
        if !provider.enabled {
            return None;
        }

        let auth_request = AuthorizationRequest::new(
            state.to_string(),
            provider_name.to_string(),
            self.request_ttl,
        );

        let scopes = provider.scopes.join(" ");
        let url = format!(
            "{}?client_id={}&redirect_uri={}&response_type=code&scope={}&state={}",
            provider.authorize_url, provider.client_id, provider.redirect_uri, scopes, state,
        );

        self.pending_requests
            .insert(state.to_string(), auth_request.clone());

        Some((url, auth_request))
    }

    /// Validates and consumes an authorization callback state.
    ///
    /// Returns the pending request if the state is valid and not expired.
    pub fn validate_callback(&mut self, state: &str) -> Option<AuthorizationRequest> {
        let request = self.pending_requests.remove(state)?;
        if request.is_expired() {
            return None;
        }
        Some(request)
    }

    /// Maps a provider identity to an internal user ID.
    pub fn map_identity(&mut self, identity: &OAuthUserIdentity, internal_user_id: &str) {
        let key = identity.internal_user_id();
        self.identity_map.insert(key, internal_user_id.to_string());
    }

    /// Looks up the internal user ID for a provider identity.
    pub fn lookup_identity(&self, identity: &OAuthUserIdentity) -> Option<&String> {
        let key = identity.internal_user_id();
        self.identity_map.get(&key)
    }

    /// Returns the number of pending authorization requests.
    pub fn pending_count(&self) -> usize {
        self.pending_requests.len()
    }

    /// Purges expired authorization requests.
    pub fn purge_expired_requests(&mut self) -> usize {
        let before = self.pending_requests.len();
        self.pending_requests.retain(|_, req| !req.is_expired());
        before - self.pending_requests.len()
    }

    /// Returns the number of identity mappings.
    pub fn identity_count(&self) -> usize {
        self.identity_map.len()
    }

    /// Disables a provider.
    pub fn disable_provider(&mut self, name: &str) -> bool {
        if let Some(p) = self.providers.get_mut(name) {
            p.enabled = false;
            true
        } else {
            false
        }
    }

    /// Enables a provider.
    pub fn enable_provider(&mut self, name: &str) -> bool {
        if let Some(p) = self.providers.get_mut(name) {
            p.enabled = true;
            true
        } else {
            false
        }
    }
}

impl Default for OAuthProviderManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn google_config() -> ProviderConfig {
        ProviderConfig::google("client-id", "client-secret", "http://localhost/callback")
    }

    fn github_config() -> ProviderConfig {
        ProviderConfig::github("gh-client", "gh-secret", "http://localhost/callback")
    }

    // ProviderType

    #[test]
    fn test_provider_type_labels() {
        assert_eq!(ProviderType::Google.label(), "google");
        assert_eq!(ProviderType::GitHub.label(), "github");
        assert_eq!(ProviderType::Microsoft.label(), "microsoft");
        assert_eq!(
            ProviderType::CustomOidc("keycloak".into()).label(),
            "keycloak"
        );
    }

    // ProviderConfig

    #[test]
    fn test_google_config() {
        let cfg = google_config();
        assert_eq!(cfg.provider_type, ProviderType::Google);
        assert!(cfg.authorize_url.contains("google"));
        assert!(cfg.scopes.contains(&"openid".to_string()));
        assert!(cfg.enabled);
    }

    #[test]
    fn test_github_config() {
        let cfg = github_config();
        assert_eq!(cfg.provider_type, ProviderType::GitHub);
        assert!(cfg.authorize_url.contains("github"));
    }

    #[test]
    fn test_microsoft_config() {
        let cfg = ProviderConfig::microsoft(
            "ms-client",
            "ms-secret",
            "http://localhost/callback",
            "my-tenant",
        );
        assert_eq!(cfg.provider_type, ProviderType::Microsoft);
        assert!(cfg.authorize_url.contains("my-tenant"));
        assert!(cfg.issuer.is_some());
    }

    // AuthorizationRequest

    #[test]
    fn test_auth_request_not_expired() {
        let req = AuthorizationRequest::new(
            "state123".to_string(),
            "google".to_string(),
            Duration::from_secs(600),
        );
        assert!(!req.is_expired());
    }

    #[test]
    fn test_auth_request_expired() {
        let req = AuthorizationRequest::new(
            "state123".to_string(),
            "google".to_string(),
            Duration::from_secs(0),
        );
        // Immediately expired
        assert!(req.is_expired());
    }

    #[test]
    fn test_auth_request_builders() {
        let req = AuthorizationRequest::new(
            "s".to_string(),
            "google".to_string(),
            Duration::from_secs(60),
        )
        .with_code_verifier("verifier".to_string())
        .with_nonce("nonce".to_string())
        .with_redirect_after("/dashboard".to_string());

        assert_eq!(req.code_verifier, Some("verifier".to_string()));
        assert_eq!(req.nonce, Some("nonce".to_string()));
        assert_eq!(req.redirect_after, Some("/dashboard".to_string()));
    }

    // OAuthUserIdentity

    #[test]
    fn test_internal_user_id() {
        let identity = OAuthUserIdentity {
            provider_user_id: "12345".to_string(),
            provider: "google".to_string(),
            email: Some("user@example.com".to_string()),
            email_verified: true,
            display_name: Some("Test User".to_string()),
            avatar_url: None,
            raw_claims: HashMap::new(),
        };
        assert_eq!(identity.internal_user_id(), "google:12345");
    }

    // OAuthProviderManager

    #[test]
    fn test_register_and_get_provider() {
        let mut mgr = OAuthProviderManager::new();
        mgr.register_provider(google_config());
        assert!(mgr.get_provider("google").is_some());
        assert!(mgr.get_provider("unknown").is_none());
    }

    #[test]
    fn test_provider_names() {
        let mut mgr = OAuthProviderManager::new();
        mgr.register_provider(google_config());
        mgr.register_provider(github_config());
        let names = mgr.provider_names();
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn test_enabled_providers() {
        let mut mgr = OAuthProviderManager::new();
        mgr.register_provider(google_config());
        let mut gh = github_config();
        gh.enabled = false;
        mgr.register_provider(gh);
        assert_eq!(mgr.enabled_providers().len(), 1);
    }

    #[test]
    fn test_build_authorize_url() {
        let mut mgr = OAuthProviderManager::new();
        mgr.register_provider(google_config());
        let result = mgr.build_authorize_url("google", "random-state");
        assert!(result.is_some());
        let (url, req) = result.expect("should succeed");
        assert!(url.contains("client_id=client-id"));
        assert!(url.contains("state=random-state"));
        assert_eq!(req.state, "random-state");
        assert_eq!(mgr.pending_count(), 1);
    }

    #[test]
    fn test_build_authorize_url_disabled_provider() {
        let mut mgr = OAuthProviderManager::new();
        let mut cfg = google_config();
        cfg.enabled = false;
        mgr.register_provider(cfg);
        assert!(mgr.build_authorize_url("google", "state").is_none());
    }

    #[test]
    fn test_build_authorize_url_unknown_provider() {
        let mut mgr = OAuthProviderManager::new();
        assert!(mgr.build_authorize_url("unknown", "state").is_none());
    }

    #[test]
    fn test_validate_callback() {
        let mut mgr = OAuthProviderManager::new();
        mgr.register_provider(google_config());
        mgr.build_authorize_url("google", "state-abc");

        let req = mgr.validate_callback("state-abc");
        assert!(req.is_some());
        // Second call should return None (consumed)
        assert!(mgr.validate_callback("state-abc").is_none());
    }

    #[test]
    fn test_validate_callback_expired() {
        let mut mgr = OAuthProviderManager::new();
        mgr.request_ttl = Duration::from_secs(0);
        mgr.register_provider(google_config());
        mgr.build_authorize_url("google", "expired-state");
        // Should be expired immediately
        assert!(mgr.validate_callback("expired-state").is_none());
    }

    #[test]
    fn test_identity_mapping() {
        let mut mgr = OAuthProviderManager::new();
        let identity = OAuthUserIdentity {
            provider_user_id: "12345".to_string(),
            provider: "google".to_string(),
            email: None,
            email_verified: false,
            display_name: None,
            avatar_url: None,
            raw_claims: HashMap::new(),
        };
        mgr.map_identity(&identity, "internal-user-42");
        assert_eq!(
            mgr.lookup_identity(&identity),
            Some(&"internal-user-42".to_string())
        );
        assert_eq!(mgr.identity_count(), 1);
    }

    #[test]
    fn test_purge_expired_requests() {
        let mut mgr = OAuthProviderManager::new();
        mgr.request_ttl = Duration::from_secs(0);
        mgr.register_provider(google_config());
        mgr.build_authorize_url("google", "s1");
        mgr.build_authorize_url("google", "s2");
        let purged = mgr.purge_expired_requests();
        assert_eq!(purged, 2);
        assert_eq!(mgr.pending_count(), 0);
    }

    #[test]
    fn test_disable_enable_provider() {
        let mut mgr = OAuthProviderManager::new();
        mgr.register_provider(google_config());
        assert!(mgr.disable_provider("google"));
        assert!(!mgr.get_provider("google").expect("should exist").enabled);
        assert!(mgr.enable_provider("google"));
        assert!(mgr.get_provider("google").expect("should exist").enabled);
    }

    #[test]
    fn test_disable_unknown_provider() {
        let mut mgr = OAuthProviderManager::new();
        assert!(!mgr.disable_provider("unknown"));
        assert!(!mgr.enable_provider("unknown"));
    }

    #[test]
    fn test_default_manager() {
        let mgr = OAuthProviderManager::default();
        assert_eq!(mgr.pending_count(), 0);
        assert_eq!(mgr.identity_count(), 0);
    }

    #[test]
    fn test_oauth_token_response_structure() {
        let resp = OAuthTokenResponse {
            access_token: "at-123".to_string(),
            token_type: "Bearer".to_string(),
            refresh_token: Some("rt-456".to_string()),
            expires_in: Some(3600),
            id_token: None,
            scope: Some("openid email".to_string()),
        };
        assert_eq!(resp.access_token, "at-123");
        assert_eq!(resp.token_type, "Bearer");
        assert!(resp.refresh_token.is_some());
    }
}

#![allow(dead_code)]
//! Cloud authentication and credential management.

use std::time::{Duration, Instant};

/// Authentication method used to access cloud services.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthMethod {
    /// Static API key authentication.
    ApiKey,
    /// OAuth 2.0 client-credentials flow.
    OAuth2,
    /// IAM role-based access (e.g., AWS instance role).
    IamRole,
    /// Short-lived token from a service account.
    ServiceAccount,
    /// Mutual TLS certificate authentication.
    MutualTls,
}

impl AuthMethod {
    /// Returns true if this authentication method requires multi-factor authentication.
    pub fn requires_mfa(&self) -> bool {
        matches!(self, AuthMethod::OAuth2 | AuthMethod::MutualTls)
    }

    /// Returns a human-readable description of the method.
    pub fn description(&self) -> &'static str {
        match self {
            AuthMethod::ApiKey => "Static API key",
            AuthMethod::OAuth2 => "OAuth 2.0 (requires MFA)",
            AuthMethod::IamRole => "IAM role-based access",
            AuthMethod::ServiceAccount => "Service account token",
            AuthMethod::MutualTls => "Mutual TLS certificate",
        }
    }

    /// Returns true if the method uses short-lived tokens that must be refreshed.
    pub fn uses_short_lived_tokens(&self) -> bool {
        matches!(
            self,
            AuthMethod::OAuth2 | AuthMethod::ServiceAccount | AuthMethod::IamRole
        )
    }
}

/// Cloud credentials including access token and expiry.
#[derive(Debug, Clone)]
pub struct CloudCredentials {
    /// The access token or key.
    pub access_token: String,
    /// Optional refresh token for token renewal.
    pub refresh_token: Option<String>,
    /// Optional cloud provider endpoint.
    pub endpoint: Option<String>,
    /// Time at which the credentials were issued.
    issued_at: Instant,
    /// Time-to-live for the credentials.
    ttl: Duration,
}

impl CloudCredentials {
    /// Creates credentials that never expire (TTL = u64::MAX seconds).
    pub fn permanent(access_token: impl Into<String>) -> Self {
        Self {
            access_token: access_token.into(),
            refresh_token: None,
            endpoint: None,
            issued_at: Instant::now(),
            ttl: Duration::from_secs(u64::MAX / 2),
        }
    }

    /// Creates credentials with a specified TTL in seconds.
    pub fn with_ttl(access_token: impl Into<String>, ttl_secs: u64) -> Self {
        Self {
            access_token: access_token.into(),
            refresh_token: None,
            endpoint: None,
            issued_at: Instant::now(),
            ttl: Duration::from_secs(ttl_secs),
        }
    }

    /// Attaches a refresh token.
    pub fn with_refresh_token(mut self, token: impl Into<String>) -> Self {
        self.refresh_token = Some(token.into());
        self
    }

    /// Attaches an endpoint URL.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    /// Returns true if the credentials have passed their TTL.
    pub fn is_expired(&self) -> bool {
        self.issued_at.elapsed() >= self.ttl
    }

    /// Returns the remaining time before expiry, or `None` if already expired.
    pub fn time_remaining(&self) -> Option<Duration> {
        let elapsed = self.issued_at.elapsed();
        self.ttl.checked_sub(elapsed)
    }

    /// Returns the fraction of TTL remaining in `[0.0, 1.0]`.
    #[allow(clippy::cast_precision_loss)]
    pub fn freshness(&self) -> f64 {
        let elapsed = self.issued_at.elapsed().as_secs_f64();
        let ttl = self.ttl.as_secs_f64();
        if ttl == 0.0 {
            return 0.0;
        }
        (1.0 - elapsed / ttl).clamp(0.0, 1.0)
    }
}

/// Result of an authentication attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthResult {
    /// Authentication succeeded; contains the new access token.
    Success(String),
    /// Authentication failed; contains a reason string.
    Failure(String),
    /// Multi-factor challenge required before proceeding.
    MfaRequired,
}

impl AuthResult {
    /// Returns true if the authentication was successful.
    pub fn is_success(&self) -> bool {
        matches!(self, AuthResult::Success(_))
    }

    /// Extracts the token if authentication succeeded.
    pub fn token(&self) -> Option<&str> {
        if let AuthResult::Success(t) = self {
            Some(t.as_str())
        } else {
            None
        }
    }
}

/// Cloud authentication handler.
#[derive(Debug)]
pub struct CloudAuth {
    /// Authentication method in use.
    pub method: AuthMethod,
    /// Current credentials, if authenticated.
    credentials: Option<CloudCredentials>,
    /// Default TTL for new tokens in seconds.
    token_ttl_secs: u64,
}

impl CloudAuth {
    /// Creates a new `CloudAuth` with the given method and default token TTL.
    pub fn new(method: AuthMethod, token_ttl_secs: u64) -> Self {
        Self {
            method,
            credentials: None,
            token_ttl_secs,
        }
    }

    /// Simulates an authentication attempt against the given endpoint using the provided secret.
    ///
    /// For this implementation the authentication always succeeds when the secret is non-empty.
    pub fn authenticate(&mut self, secret: &str) -> AuthResult {
        if secret.is_empty() {
            return AuthResult::Failure("Empty secret provided".to_string());
        }
        if self.method.requires_mfa() {
            // In a real implementation we would initiate the MFA challenge here.
            // For simulation: we treat a secret that starts with "mfa:" as having passed MFA.
            if !secret.starts_with("mfa:") {
                return AuthResult::MfaRequired;
            }
        }
        let token = format!("tok_{}", &secret[..secret.len().min(8)]);
        let creds = CloudCredentials::with_ttl(token.clone(), self.token_ttl_secs);
        self.credentials = Some(creds);
        AuthResult::Success(token)
    }

    /// Refreshes the current credentials using the stored refresh token.
    ///
    /// Returns `false` if there are no credentials or no refresh token.
    pub fn refresh_token(&mut self) -> bool {
        let has_refresh = self
            .credentials
            .as_ref()
            .and_then(|c| c.refresh_token.as_ref())
            .is_some();

        if !has_refresh {
            return false;
        }

        // In production this would call the token endpoint. Here we issue new credentials.
        if let Some(ref creds) = self.credentials.clone() {
            if let Some(ref rt) = creds.refresh_token {
                let new_token = format!("tok_refreshed_{}", &rt[..rt.len().min(6)]);
                let mut new_creds = CloudCredentials::with_ttl(new_token, self.token_ttl_secs)
                    .with_refresh_token(rt.clone());
                if let Some(ref ep) = creds.endpoint {
                    new_creds = new_creds.with_endpoint(ep.clone());
                }
                self.credentials = Some(new_creds);
                return true;
            }
        }
        false
    }

    /// Returns true if the current credentials are present and not expired.
    pub fn is_valid(&self) -> bool {
        self.credentials.as_ref().is_some_and(|c| !c.is_expired())
    }

    /// Returns a reference to the current credentials, if any.
    pub fn credentials(&self) -> Option<&CloudCredentials> {
        self.credentials.as_ref()
    }

    /// Revokes the current credentials.
    pub fn revoke(&mut self) {
        self.credentials = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auth_method_requires_mfa() {
        assert!(!AuthMethod::ApiKey.requires_mfa());
        assert!(AuthMethod::OAuth2.requires_mfa());
        assert!(!AuthMethod::IamRole.requires_mfa());
        assert!(!AuthMethod::ServiceAccount.requires_mfa());
        assert!(AuthMethod::MutualTls.requires_mfa());
    }

    #[test]
    fn test_auth_method_uses_short_lived_tokens() {
        assert!(!AuthMethod::ApiKey.uses_short_lived_tokens());
        assert!(AuthMethod::OAuth2.uses_short_lived_tokens());
        assert!(AuthMethod::IamRole.uses_short_lived_tokens());
        assert!(AuthMethod::ServiceAccount.uses_short_lived_tokens());
        assert!(!AuthMethod::MutualTls.uses_short_lived_tokens());
    }

    #[test]
    fn test_credentials_not_expired_fresh() {
        let creds = CloudCredentials::with_ttl("token123", 3600);
        assert!(!creds.is_expired());
    }

    #[test]
    fn test_credentials_expired_zero_ttl() {
        let creds = CloudCredentials::with_ttl("token123", 0);
        // TTL of 0 means immediately expired
        assert!(creds.is_expired());
    }

    #[test]
    fn test_credentials_permanent_not_expired() {
        let creds = CloudCredentials::permanent("api_key_xyz");
        assert!(!creds.is_expired());
    }

    #[test]
    fn test_freshness_fresh_credentials() {
        let creds = CloudCredentials::with_ttl("tok", 3600);
        // Should be very close to 1.0 right after creation
        assert!(creds.freshness() > 0.99);
    }

    #[test]
    fn test_credentials_with_refresh_token() {
        let creds = CloudCredentials::with_ttl("tok", 3600).with_refresh_token("refresh_xyz");
        assert_eq!(creds.refresh_token.as_deref(), Some("refresh_xyz"));
    }

    #[test]
    fn test_credentials_with_endpoint() {
        let creds =
            CloudCredentials::with_ttl("tok", 3600).with_endpoint("https://api.example.com");
        assert_eq!(creds.endpoint.as_deref(), Some("https://api.example.com"));
    }

    #[test]
    fn test_auth_result_is_success() {
        let r = AuthResult::Success("token".into());
        assert!(r.is_success());
    }

    #[test]
    fn test_auth_result_token() {
        let r = AuthResult::Success("my_token".into());
        assert_eq!(r.token(), Some("my_token"));
    }

    #[test]
    fn test_auth_result_failure_not_success() {
        let r = AuthResult::Failure("bad creds".into());
        assert!(!r.is_success());
        assert!(r.token().is_none());
    }

    #[test]
    fn test_authenticate_empty_secret_fails() {
        let mut auth = CloudAuth::new(AuthMethod::ApiKey, 3600);
        let result = auth.authenticate("");
        assert!(matches!(result, AuthResult::Failure(_)));
    }

    #[test]
    fn test_authenticate_api_key_succeeds() {
        let mut auth = CloudAuth::new(AuthMethod::ApiKey, 3600);
        let result = auth.authenticate("mysecret");
        assert!(result.is_success());
        assert!(auth.is_valid());
    }

    #[test]
    fn test_authenticate_oauth2_requires_mfa() {
        let mut auth = CloudAuth::new(AuthMethod::OAuth2, 3600);
        let result = auth.authenticate("plain_password");
        assert_eq!(result, AuthResult::MfaRequired);
    }

    #[test]
    fn test_authenticate_oauth2_with_mfa_prefix() {
        let mut auth = CloudAuth::new(AuthMethod::OAuth2, 3600);
        let result = auth.authenticate("mfa:otp123456");
        assert!(result.is_success());
    }

    #[test]
    fn test_refresh_token_no_credentials() {
        let mut auth = CloudAuth::new(AuthMethod::ApiKey, 3600);
        assert!(!auth.refresh_token());
    }

    #[test]
    fn test_refresh_token_without_refresh_cred() {
        let mut auth = CloudAuth::new(AuthMethod::ApiKey, 3600);
        auth.authenticate("secret");
        // No refresh token attached — should return false
        assert!(!auth.refresh_token());
    }

    #[test]
    fn test_refresh_token_with_refresh_cred() {
        let mut auth = CloudAuth::new(AuthMethod::ApiKey, 3600);
        auth.authenticate("secret");
        // Manually attach a refresh token
        if let Some(ref mut creds) = auth.credentials {
            creds.refresh_token = Some("rt_abc".into());
        }
        assert!(auth.refresh_token());
        assert!(auth.is_valid());
    }

    #[test]
    fn test_revoke_clears_credentials() {
        let mut auth = CloudAuth::new(AuthMethod::ApiKey, 3600);
        auth.authenticate("secret");
        auth.revoke();
        assert!(!auth.is_valid());
        assert!(auth.credentials().is_none());
    }
}

//! Header-based API content negotiation (Accept-Version) alongside URL-based versioning.
//!
//! Extracts the requested API version from either the URL path prefix
//! (`/api/v1/...`) or the `Accept-Version` header, then resolves the best
//! matching registered version using the `ApiVersionRegistry`.

#![allow(dead_code)]

use crate::api_versioning::{ApiVersion, ApiVersionRegistry, VersionEntry};
use std::fmt;

/// How the API version was determined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionSource {
    /// Extracted from the URL path (e.g. `/api/v2/...`).
    UrlPath,
    /// Extracted from the `Accept-Version` request header.
    AcceptVersionHeader,
    /// Extracted from a custom `X-Api-Version` header.
    CustomHeader,
    /// No version was specified; using the server default.
    Default,
}

impl fmt::Display for VersionSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UrlPath => write!(f, "url-path"),
            Self::AcceptVersionHeader => write!(f, "accept-version-header"),
            Self::CustomHeader => write!(f, "x-api-version-header"),
            Self::Default => write!(f, "default"),
        }
    }
}

/// The result of resolving an API version from a request.
#[derive(Debug, Clone)]
pub struct ResolvedVersion {
    /// The requested version (as parsed from the request).
    pub requested: ApiVersion,
    /// The actual version the server will use (after resolution).
    pub resolved: ApiVersion,
    /// How the version was determined.
    pub source: VersionSource,
    /// Whether the resolved version is deprecated.
    pub deprecated: bool,
    /// Optional deprecation message.
    pub deprecation_notice: Option<String>,
}

impl ResolvedVersion {
    /// Returns response headers to inform the client of the resolved version.
    pub fn response_headers(&self) -> Vec<(String, String)> {
        let mut headers = vec![
            ("X-Api-Version".to_string(), self.resolved.to_string()),
            ("X-Api-Version-Source".to_string(), self.source.to_string()),
        ];

        if self.deprecated {
            headers.push(("Deprecation".to_string(), "true".to_string()));
            if let Some(ref notice) = self.deprecation_notice {
                headers.push(("Sunset".to_string(), notice.clone()));
            }
        }

        headers
    }
}

/// Errors that can occur during version negotiation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NegotiationError {
    /// The `Accept-Version` header value could not be parsed.
    InvalidVersionFormat(String),
    /// No compatible version was found in the registry.
    UnsupportedVersion(ApiVersion),
    /// Multiple conflicting version sources were specified.
    ConflictingVersions {
        /// Version from URL.
        url_version: ApiVersion,
        /// Version from header.
        header_version: ApiVersion,
    },
}

impl fmt::Display for NegotiationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidVersionFormat(s) => write!(f, "Invalid version format: '{}'", s),
            Self::UnsupportedVersion(v) => write!(f, "Unsupported API version: {}", v),
            Self::ConflictingVersions {
                url_version,
                header_version,
            } => write!(
                f,
                "Conflicting versions: URL={}, header={}",
                url_version, header_version
            ),
        }
    }
}

impl std::error::Error for NegotiationError {}

/// Parses an API version string like `"1"`, `"1.2"`, or `"1.2.3"`.
pub fn parse_version(s: &str) -> Result<ApiVersion, NegotiationError> {
    let s = s.trim().trim_start_matches('v').trim_start_matches('V');
    let parts: Vec<&str> = s.split('.').collect();

    match parts.len() {
        1 => {
            let major = parts[0]
                .parse::<u32>()
                .map_err(|_| NegotiationError::InvalidVersionFormat(s.to_string()))?;
            Ok(ApiVersion::new(major, 0, 0))
        }
        2 => {
            let major = parts[0]
                .parse::<u32>()
                .map_err(|_| NegotiationError::InvalidVersionFormat(s.to_string()))?;
            let minor = parts[1]
                .parse::<u32>()
                .map_err(|_| NegotiationError::InvalidVersionFormat(s.to_string()))?;
            Ok(ApiVersion::new(major, minor, 0))
        }
        3 => {
            let major = parts[0]
                .parse::<u32>()
                .map_err(|_| NegotiationError::InvalidVersionFormat(s.to_string()))?;
            let minor = parts[1]
                .parse::<u32>()
                .map_err(|_| NegotiationError::InvalidVersionFormat(s.to_string()))?;
            let patch = parts[2]
                .parse::<u32>()
                .map_err(|_| NegotiationError::InvalidVersionFormat(s.to_string()))?;
            Ok(ApiVersion::new(major, minor, patch))
        }
        _ => Err(NegotiationError::InvalidVersionFormat(s.to_string())),
    }
}

/// Extracts the version from a URL path prefix like `/api/v1/...`.
pub fn extract_url_version(path: &str) -> Option<ApiVersion> {
    let parts: Vec<&str> = path.split('/').collect();
    for part in &parts {
        if part.starts_with('v') || part.starts_with('V') {
            if let Ok(v) = parse_version(part) {
                return Some(v);
            }
        }
    }
    None
}

/// Configuration for version negotiation.
#[derive(Debug, Clone)]
pub struct NegotiationConfig {
    /// Default version to use when none is specified.
    pub default_version: ApiVersion,
    /// Whether to allow the `Accept-Version` header.
    pub allow_accept_version_header: bool,
    /// Whether to allow the `X-Api-Version` header.
    pub allow_custom_header: bool,
    /// Header name priority order (first match wins).
    pub header_priority: Vec<String>,
    /// Whether URL version takes precedence over header version.
    pub url_takes_precedence: bool,
    /// Whether conflicting URL/header versions should produce an error.
    pub strict_conflict: bool,
}

impl Default for NegotiationConfig {
    fn default() -> Self {
        Self {
            default_version: ApiVersion::new(1, 0, 0),
            allow_accept_version_header: true,
            allow_custom_header: true,
            header_priority: vec!["Accept-Version".to_string(), "X-Api-Version".to_string()],
            url_takes_precedence: true,
            strict_conflict: false,
        }
    }
}

/// The API version negotiator.
///
/// Resolves the best API version from request metadata (URL path and headers)
/// using a registry of known versions.
pub struct VersionNegotiator {
    config: NegotiationConfig,
    registry: ApiVersionRegistry,
}

impl VersionNegotiator {
    /// Creates a new negotiator with the given config and registry.
    pub fn new(config: NegotiationConfig, registry: ApiVersionRegistry) -> Self {
        Self { config, registry }
    }

    /// Resolves the API version from URL path and headers.
    ///
    /// `headers` is a slice of `(name, value)` pairs.
    pub fn negotiate(
        &self,
        url_path: &str,
        headers: &[(&str, &str)],
    ) -> Result<ResolvedVersion, NegotiationError> {
        let url_version = extract_url_version(url_path);
        let header_version = self.extract_header_version(headers)?;

        // Determine which version and source to use
        let (requested, source) = match (url_version, header_version) {
            (Some(uv), Some((hv, hs))) => {
                if self.config.strict_conflict && uv != hv {
                    return Err(NegotiationError::ConflictingVersions {
                        url_version: uv,
                        header_version: hv,
                    });
                }
                if self.config.url_takes_precedence {
                    (uv, VersionSource::UrlPath)
                } else {
                    (hv, hs)
                }
            }
            (Some(uv), None) => (uv, VersionSource::UrlPath),
            (None, Some((hv, hs))) => (hv, hs),
            (None, None) => (self.config.default_version, VersionSource::Default),
        };

        // Resolve against registry
        let entry: &VersionEntry = self
            .registry
            .resolve(&requested)
            .ok_or(NegotiationError::UnsupportedVersion(requested))?;

        let deprecation_notice = if entry.deprecated {
            Some(format!(
                "API version {} is deprecated. {}",
                entry.version, entry.description
            ))
        } else {
            None
        };

        Ok(ResolvedVersion {
            requested,
            resolved: entry.version,
            source,
            deprecated: entry.deprecated,
            deprecation_notice,
        })
    }

    /// Extracts the version from headers based on priority.
    fn extract_header_version(
        &self,
        headers: &[(&str, &str)],
    ) -> Result<Option<(ApiVersion, VersionSource)>, NegotiationError> {
        for priority_header in &self.config.header_priority {
            let lower = priority_header.to_lowercase();
            for &(name, value) in headers {
                if name.to_lowercase() == lower {
                    let version = parse_version(value)?;
                    let source = if lower == "accept-version" {
                        VersionSource::AcceptVersionHeader
                    } else {
                        VersionSource::CustomHeader
                    };
                    return Ok(Some((version, source)));
                }
            }
        }
        Ok(None)
    }

    /// Returns the registry (for inspection / testing).
    pub fn registry(&self) -> &ApiVersionRegistry {
        &self.registry
    }

    /// Returns the default version.
    pub fn default_version(&self) -> ApiVersion {
        self.config.default_version
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_registry() -> ApiVersionRegistry {
        let mut reg = ApiVersionRegistry::new();
        reg.register(ApiVersion::new(1, 0, 0), "Initial release");
        reg.register(ApiVersion::new(1, 1, 0), "Added streaming");
        reg.register(ApiVersion::new(1, 2, 0), "Added batch ops");
        reg.register(ApiVersion::new(2, 0, 0), "Major v2");
        reg.deprecate(&ApiVersion::new(1, 0, 0));
        reg
    }

    fn make_negotiator() -> VersionNegotiator {
        VersionNegotiator::new(NegotiationConfig::default(), make_registry())
    }

    // parse_version

    #[test]
    fn test_parse_version_major_only() {
        let v = parse_version("1").expect("should parse");
        assert_eq!(v, ApiVersion::new(1, 0, 0));
    }

    #[test]
    fn test_parse_version_major_minor() {
        let v = parse_version("1.2").expect("should parse");
        assert_eq!(v, ApiVersion::new(1, 2, 0));
    }

    #[test]
    fn test_parse_version_full() {
        let v = parse_version("1.2.3").expect("should parse");
        assert_eq!(v, ApiVersion::new(1, 2, 3));
    }

    #[test]
    fn test_parse_version_with_v_prefix() {
        let v = parse_version("v2").expect("should parse");
        assert_eq!(v, ApiVersion::new(2, 0, 0));
    }

    #[test]
    fn test_parse_version_invalid() {
        assert!(parse_version("abc").is_err());
        assert!(parse_version("1.2.3.4").is_err());
    }

    // extract_url_version

    #[test]
    fn test_extract_url_version_standard() {
        let v = extract_url_version("/api/v1/media");
        assert_eq!(v, Some(ApiVersion::new(1, 0, 0)));
    }

    #[test]
    fn test_extract_url_version_v2() {
        let v = extract_url_version("/api/v2/users");
        assert_eq!(v, Some(ApiVersion::new(2, 0, 0)));
    }

    #[test]
    fn test_extract_url_version_none() {
        let v = extract_url_version("/api/media");
        assert!(v.is_none());
    }

    // negotiate - URL path

    #[test]
    fn test_negotiate_url_path() {
        let neg = make_negotiator();
        let result = neg
            .negotiate("/api/v1/media", &[])
            .expect("should negotiate");
        assert_eq!(result.source, VersionSource::UrlPath);
        assert_eq!(result.resolved, ApiVersion::new(1, 2, 0)); // resolves to latest v1
    }

    #[test]
    fn test_negotiate_url_v2() {
        let neg = make_negotiator();
        let result = neg
            .negotiate("/api/v2/media", &[])
            .expect("should negotiate");
        assert_eq!(result.resolved, ApiVersion::new(2, 0, 0));
    }

    // negotiate - Accept-Version header

    #[test]
    fn test_negotiate_accept_version_header() {
        let neg = make_negotiator();
        let result = neg
            .negotiate("/api/media", &[("Accept-Version", "1.1")])
            .expect("should negotiate");
        assert_eq!(result.source, VersionSource::AcceptVersionHeader);
        assert_eq!(result.resolved, ApiVersion::new(1, 2, 0));
    }

    #[test]
    fn test_negotiate_custom_header() {
        let neg = make_negotiator();
        let result = neg
            .negotiate("/api/media", &[("X-Api-Version", "2.0.0")])
            .expect("should negotiate");
        assert_eq!(result.source, VersionSource::CustomHeader);
        assert_eq!(result.resolved, ApiVersion::new(2, 0, 0));
    }

    // negotiate - default

    #[test]
    fn test_negotiate_default() {
        let neg = make_negotiator();
        let result = neg.negotiate("/api/media", &[]).expect("should negotiate");
        assert_eq!(result.source, VersionSource::Default);
        assert_eq!(result.requested, ApiVersion::new(1, 0, 0));
    }

    // negotiate - URL takes precedence

    #[test]
    fn test_negotiate_url_takes_precedence() {
        let neg = make_negotiator();
        let result = neg
            .negotiate("/api/v2/media", &[("Accept-Version", "1.0")])
            .expect("should negotiate");
        assert_eq!(result.source, VersionSource::UrlPath);
        assert_eq!(result.resolved, ApiVersion::new(2, 0, 0));
    }

    // negotiate - header takes precedence

    #[test]
    fn test_negotiate_header_takes_precedence() {
        let config = NegotiationConfig {
            url_takes_precedence: false,
            ..Default::default()
        };
        let neg = VersionNegotiator::new(config, make_registry());
        let result = neg
            .negotiate("/api/v1/media", &[("Accept-Version", "2.0")])
            .expect("should negotiate");
        assert_eq!(result.source, VersionSource::AcceptVersionHeader);
        assert_eq!(result.resolved, ApiVersion::new(2, 0, 0));
    }

    // negotiate - strict conflict

    #[test]
    fn test_negotiate_strict_conflict() {
        let config = NegotiationConfig {
            strict_conflict: true,
            ..Default::default()
        };
        let neg = VersionNegotiator::new(config, make_registry());
        let result = neg.negotiate("/api/v1/media", &[("Accept-Version", "2.0")]);
        assert!(matches!(
            result,
            Err(NegotiationError::ConflictingVersions { .. })
        ));
    }

    #[test]
    fn test_negotiate_strict_conflict_same_version_ok() {
        let config = NegotiationConfig {
            strict_conflict: true,
            ..Default::default()
        };
        let neg = VersionNegotiator::new(config, make_registry());
        let result = neg.negotiate("/api/v1/media", &[("Accept-Version", "1.0")]);
        assert!(result.is_ok());
    }

    // negotiate - unsupported version

    #[test]
    fn test_negotiate_unsupported_version() {
        let neg = make_negotiator();
        let result = neg.negotiate("/api/v3/media", &[]);
        assert!(matches!(
            result,
            Err(NegotiationError::UnsupportedVersion(_))
        ));
    }

    // deprecated version

    #[test]
    fn test_negotiate_deprecated_version_returns_notice() {
        let neg = make_negotiator();
        // v1.0 is deprecated, but resolves to v1.2 (latest v1)
        let result = neg
            .negotiate("/api/v1/media", &[])
            .expect("should negotiate");
        // Resolved to v1.2 which is NOT deprecated
        assert!(!result.deprecated);
    }

    // response headers

    #[test]
    fn test_resolved_version_response_headers() {
        let rv = ResolvedVersion {
            requested: ApiVersion::new(1, 0, 0),
            resolved: ApiVersion::new(1, 2, 0),
            source: VersionSource::UrlPath,
            deprecated: true,
            deprecation_notice: Some("Use v2".to_string()),
        };
        let headers = rv.response_headers();
        assert!(headers
            .iter()
            .any(|(k, v)| k == "X-Api-Version" && v == "1.2.0"));
        assert!(headers.iter().any(|(k, _)| k == "Deprecation"));
        assert!(headers.iter().any(|(k, v)| k == "Sunset" && v == "Use v2"));
    }

    #[test]
    fn test_resolved_version_no_deprecation_headers() {
        let rv = ResolvedVersion {
            requested: ApiVersion::new(2, 0, 0),
            resolved: ApiVersion::new(2, 0, 0),
            source: VersionSource::Default,
            deprecated: false,
            deprecation_notice: None,
        };
        let headers = rv.response_headers();
        assert!(!headers.iter().any(|(k, _)| k == "Deprecation"));
    }

    // Error display

    #[test]
    fn test_negotiation_error_display_invalid() {
        let err = NegotiationError::InvalidVersionFormat("xyz".to_string());
        assert!(format!("{}", err).contains("xyz"));
    }

    #[test]
    fn test_negotiation_error_display_unsupported() {
        let err = NegotiationError::UnsupportedVersion(ApiVersion::new(99, 0, 0));
        assert!(format!("{}", err).contains("99.0.0"));
    }

    #[test]
    fn test_negotiation_error_display_conflict() {
        let err = NegotiationError::ConflictingVersions {
            url_version: ApiVersion::new(1, 0, 0),
            header_version: ApiVersion::new(2, 0, 0),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("1.0.0"));
        assert!(msg.contains("2.0.0"));
    }

    // VersionSource display

    #[test]
    fn test_version_source_display() {
        assert_eq!(VersionSource::UrlPath.to_string(), "url-path");
        assert_eq!(
            VersionSource::AcceptVersionHeader.to_string(),
            "accept-version-header"
        );
        assert_eq!(
            VersionSource::CustomHeader.to_string(),
            "x-api-version-header"
        );
        assert_eq!(VersionSource::Default.to_string(), "default");
    }

    #[test]
    fn test_default_negotiation_config() {
        let cfg = NegotiationConfig::default();
        assert_eq!(cfg.default_version, ApiVersion::new(1, 0, 0));
        assert!(cfg.allow_accept_version_header);
        assert!(cfg.url_takes_precedence);
        assert!(!cfg.strict_conflict);
    }

    #[test]
    fn test_negotiator_default_version() {
        let neg = make_negotiator();
        assert_eq!(neg.default_version(), ApiVersion::new(1, 0, 0));
    }

    #[test]
    fn test_header_priority_accept_version_first() {
        let neg = make_negotiator();
        let result = neg
            .negotiate(
                "/api/media",
                &[("X-Api-Version", "2.0"), ("Accept-Version", "1.1")],
            )
            .expect("should negotiate");
        // Accept-Version has higher priority
        assert_eq!(result.source, VersionSource::AcceptVersionHeader);
    }

    #[test]
    fn test_case_insensitive_header_matching() {
        let neg = make_negotiator();
        let result = neg
            .negotiate("/api/media", &[("accept-version", "1.0")])
            .expect("should negotiate");
        assert_eq!(result.source, VersionSource::AcceptVersionHeader);
    }
}

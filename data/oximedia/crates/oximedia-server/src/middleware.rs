//! HTTP middleware chain for the OxiMedia server.
//!
//! Provides composable middleware components for logging, compression,
//! CORS, authentication, and request tracing.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// HTTP method
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Delete,
    Patch,
    Head,
    Options,
    Other(String),
}

impl std::fmt::Display for HttpMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HttpMethod::Get => write!(f, "GET"),
            HttpMethod::Post => write!(f, "POST"),
            HttpMethod::Put => write!(f, "PUT"),
            HttpMethod::Delete => write!(f, "DELETE"),
            HttpMethod::Patch => write!(f, "PATCH"),
            HttpMethod::Head => write!(f, "HEAD"),
            HttpMethod::Options => write!(f, "OPTIONS"),
            HttpMethod::Other(s) => write!(f, "{s}"),
        }
    }
}

impl From<&str> for HttpMethod {
    fn from(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "GET" => HttpMethod::Get,
            "POST" => HttpMethod::Post,
            "PUT" => HttpMethod::Put,
            "DELETE" => HttpMethod::Delete,
            "PATCH" => HttpMethod::Patch,
            "HEAD" => HttpMethod::Head,
            "OPTIONS" => HttpMethod::Options,
            other => HttpMethod::Other(other.to_string()),
        }
    }
}

/// Simulated HTTP request context for middleware processing
#[derive(Debug, Clone)]
pub struct RequestContext {
    /// Request ID (unique per request)
    pub request_id: String,
    /// HTTP method
    pub method: HttpMethod,
    /// Request path
    pub path: String,
    /// Remote IP address
    pub remote_addr: String,
    /// Request headers
    pub headers: HashMap<String, String>,
    /// Authenticated user ID (set by auth middleware)
    pub user_id: Option<String>,
    /// Request timestamp
    pub started_at: Instant,
}

impl RequestContext {
    /// Creates a new request context
    pub fn new(
        request_id: impl Into<String>,
        method: HttpMethod,
        path: impl Into<String>,
        remote_addr: impl Into<String>,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            method,
            path: path.into(),
            remote_addr: remote_addr.into(),
            headers: HashMap::new(),
            user_id: None,
            started_at: Instant::now(),
        }
    }

    /// Returns the elapsed time since the request started
    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Returns elapsed time in milliseconds as f64
    pub fn elapsed_ms(&self) -> f64 {
        self.elapsed().as_secs_f64() * 1000.0
    }

    /// Adds a header to the context
    #[must_use]
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }
}

/// Result of middleware processing
#[derive(Debug, Clone)]
pub struct MiddlewareResult {
    /// Whether the request should continue to the next layer
    pub proceed: bool,
    /// HTTP status code (if terminating the chain)
    pub status: Option<u16>,
    /// Response body (if terminating early)
    pub body: Option<String>,
    /// Headers to add to the response
    pub response_headers: HashMap<String, String>,
}

impl MiddlewareResult {
    /// Creates a result that passes through to the next middleware
    pub fn proceed() -> Self {
        Self {
            proceed: true,
            status: None,
            body: None,
            response_headers: HashMap::new(),
        }
    }

    /// Creates a result that terminates the chain with the given status
    pub fn terminate(status: u16, body: impl Into<String>) -> Self {
        Self {
            proceed: false,
            status: Some(status),
            body: Some(body.into()),
            response_headers: HashMap::new(),
        }
    }

    /// Adds a response header
    #[must_use]
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.response_headers.insert(name.into(), value.into());
        self
    }
}

/// Logging middleware — records method, path, and elapsed time
pub struct LoggingMiddleware {
    /// Minimum elapsed time (ms) before logging a "slow request" warning
    pub slow_threshold_ms: f64,
    /// Whether to log request headers
    pub log_headers: bool,
}

impl LoggingMiddleware {
    /// Creates a new logging middleware with default thresholds
    pub fn new() -> Self {
        Self {
            slow_threshold_ms: 500.0,
            log_headers: false,
        }
    }

    /// Processes the request context
    pub fn process(&self, ctx: &RequestContext) -> MiddlewareResult {
        let elapsed = ctx.elapsed_ms();
        if elapsed >= self.slow_threshold_ms {
            tracing::warn!(
                request_id = %ctx.request_id,
                method = %ctx.method,
                path = %ctx.path,
                elapsed_ms = elapsed,
                "Slow request detected"
            );
        } else {
            tracing::info!(
                request_id = %ctx.request_id,
                method = %ctx.method,
                path = %ctx.path,
                "Request received"
            );
        }
        MiddlewareResult::proceed()
    }
}

impl Default for LoggingMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

/// CORS configuration for the CORS middleware
#[derive(Debug, Clone)]
pub struct CorsConfig {
    /// Allowed origins (use `["*"]` for all)
    pub allowed_origins: Vec<String>,
    /// Allowed HTTP methods
    pub allowed_methods: Vec<String>,
    /// Allowed headers
    pub allowed_headers: Vec<String>,
    /// Whether to allow credentials
    pub allow_credentials: bool,
    /// Max age for preflight cache (seconds)
    pub max_age_secs: u64,
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            allowed_origins: vec!["*".to_string()],
            allowed_methods: vec![
                "GET".to_string(),
                "POST".to_string(),
                "PUT".to_string(),
                "DELETE".to_string(),
                "OPTIONS".to_string(),
            ],
            allowed_headers: vec![
                "Content-Type".to_string(),
                "Authorization".to_string(),
                "X-Request-ID".to_string(),
            ],
            allow_credentials: false,
            max_age_secs: 86400,
        }
    }
}

/// CORS middleware — adds appropriate cross-origin headers
pub struct CorsMiddleware {
    config: CorsConfig,
}

impl CorsMiddleware {
    /// Creates a new CORS middleware with the given configuration
    pub fn new(config: CorsConfig) -> Self {
        Self { config }
    }

    /// Checks whether the given origin is allowed
    pub fn is_origin_allowed(&self, origin: &str) -> bool {
        self.config
            .allowed_origins
            .iter()
            .any(|o| o == "*" || o == origin)
    }

    /// Processes a request, attaching CORS headers as needed
    pub fn process(&self, ctx: &RequestContext) -> MiddlewareResult {
        let origin = ctx.headers.get("Origin").cloned().unwrap_or_default();
        let mut result = MiddlewareResult::proceed();

        if self.is_origin_allowed(&origin) {
            result = result
                .with_header("Access-Control-Allow-Origin", &origin)
                .with_header(
                    "Access-Control-Allow-Methods",
                    self.config.allowed_methods.join(", "),
                )
                .with_header(
                    "Access-Control-Allow-Headers",
                    self.config.allowed_headers.join(", "),
                )
                .with_header(
                    "Access-Control-Max-Age",
                    self.config.max_age_secs.to_string(),
                );

            if self.config.allow_credentials {
                result = result.with_header("Access-Control-Allow-Credentials", "true");
            }
        }

        // Handle preflight
        if ctx.method == HttpMethod::Options {
            return MiddlewareResult::terminate(204, "")
                .with_header("Access-Control-Allow-Origin", &origin)
                .with_header(
                    "Access-Control-Allow-Methods",
                    self.config.allowed_methods.join(", "),
                )
                .with_header(
                    "Access-Control-Allow-Headers",
                    self.config.allowed_headers.join(", "),
                );
        }

        result
    }
}

/// Authentication middleware — validates Bearer tokens
pub struct AuthMiddleware {
    /// Paths that are exempt from authentication
    pub public_paths: Vec<String>,
    /// Expected token prefix (default: "Bearer ")
    pub token_prefix: String,
}

impl AuthMiddleware {
    /// Creates a new auth middleware
    pub fn new(public_paths: Vec<String>) -> Self {
        Self {
            public_paths,
            token_prefix: "Bearer ".to_string(),
        }
    }

    /// Checks whether the given path is public (no auth required)
    pub fn is_public_path(&self, path: &str) -> bool {
        self.public_paths
            .iter()
            .any(|p| path.starts_with(p.as_str()))
    }

    /// Extracts the raw token from the Authorization header value
    pub fn extract_token<'a>(&self, auth_header: &'a str) -> Option<&'a str> {
        auth_header.strip_prefix(&self.token_prefix)
    }

    /// Processes the request context for authentication
    pub fn process(&self, ctx: &RequestContext) -> MiddlewareResult {
        if self.is_public_path(&ctx.path) {
            return MiddlewareResult::proceed();
        }
        match ctx.headers.get("Authorization") {
            Some(auth) => {
                if self.extract_token(auth).is_some() {
                    MiddlewareResult::proceed()
                } else {
                    MiddlewareResult::terminate(401, r#"{"error":"invalid_token"}"#)
                }
            }
            None => MiddlewareResult::terminate(401, r#"{"error":"missing_token"}"#),
        }
    }
}

/// Compression settings for the compression middleware
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Enable gzip compression
    pub enable_gzip: bool,
    /// Enable brotli compression
    pub enable_brotli: bool,
    /// Minimum response size (bytes) to compress
    pub min_size: usize,
    /// MIME types eligible for compression
    pub compressible_types: Vec<String>,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enable_gzip: true,
            enable_brotli: true,
            min_size: 1024,
            compressible_types: vec![
                "text/plain".to_string(),
                "text/html".to_string(),
                "application/json".to_string(),
                "application/javascript".to_string(),
                "text/css".to_string(),
            ],
        }
    }
}

/// Compression middleware — negotiates content encoding with the client
pub struct CompressionMiddleware {
    config: CompressionConfig,
}

impl CompressionMiddleware {
    /// Creates a new compression middleware
    pub fn new(config: CompressionConfig) -> Self {
        Self { config }
    }

    /// Returns the best accepted encoding for the given Accept-Encoding header value
    pub fn negotiate_encoding(&self, accept_encoding: &str) -> Option<&'static str> {
        if self.config.enable_brotli && accept_encoding.contains("br") {
            return Some("br");
        }
        if self.config.enable_gzip && accept_encoding.contains("gzip") {
            return Some("gzip");
        }
        None
    }

    /// Checks whether the given content type is compressible
    pub fn is_compressible(&self, content_type: &str) -> bool {
        self.config
            .compressible_types
            .iter()
            .any(|t| content_type.starts_with(t.as_str()))
    }

    /// Processes the request context
    pub fn process(&self, ctx: &RequestContext) -> MiddlewareResult {
        let accept = ctx
            .headers
            .get("Accept-Encoding")
            .map(String::as_str)
            .unwrap_or("");
        let mut result = MiddlewareResult::proceed();
        if let Some(encoding) = self.negotiate_encoding(accept) {
            result = result.with_header("Content-Encoding", encoding);
        }
        result
    }
}

/// Middleware chain that runs each middleware in order
pub struct MiddlewareChain {
    logging: Option<LoggingMiddleware>,
    cors: Option<CorsMiddleware>,
    auth: Option<AuthMiddleware>,
    compression: Option<CompressionMiddleware>,
}

impl MiddlewareChain {
    /// Creates an empty middleware chain
    pub fn new() -> Self {
        Self {
            logging: None,
            cors: None,
            auth: None,
            compression: None,
        }
    }

    /// Adds logging middleware to the chain
    #[must_use]
    pub fn with_logging(mut self, mw: LoggingMiddleware) -> Self {
        self.logging = Some(mw);
        self
    }

    /// Adds CORS middleware to the chain
    #[must_use]
    pub fn with_cors(mut self, mw: CorsMiddleware) -> Self {
        self.cors = Some(mw);
        self
    }

    /// Adds authentication middleware to the chain
    #[must_use]
    pub fn with_auth(mut self, mw: AuthMiddleware) -> Self {
        self.auth = Some(mw);
        self
    }

    /// Adds compression middleware to the chain
    #[must_use]
    pub fn with_compression(mut self, mw: CompressionMiddleware) -> Self {
        self.compression = Some(mw);
        self
    }

    /// Runs the chain against the given request context.
    ///
    /// Returns the first non-proceeding result, or a proceed result
    /// with all accumulated response headers if every layer passes.
    pub fn run(&self, ctx: &RequestContext) -> MiddlewareResult {
        let mut accumulated_headers: HashMap<String, String> = HashMap::new();

        #[allow(clippy::type_complexity)]
        let layers: Vec<Box<dyn Fn(&RequestContext) -> MiddlewareResult>> = {
            let mut v: Vec<Box<dyn Fn(&RequestContext) -> MiddlewareResult>> = Vec::new();
            if let Some(ref mw) = self.logging {
                v.push(Box::new(|c| mw.process(c)));
            }
            if let Some(ref mw) = self.cors {
                v.push(Box::new(|c| mw.process(c)));
            }
            if let Some(ref mw) = self.auth {
                v.push(Box::new(|c| mw.process(c)));
            }
            if let Some(ref mw) = self.compression {
                v.push(Box::new(|c| mw.process(c)));
            }
            v
        };

        for layer in &layers {
            let result = layer(ctx);
            for (k, v) in &result.response_headers {
                accumulated_headers.insert(k.clone(), v.clone());
            }
            if !result.proceed {
                let mut final_result = result;
                final_result.response_headers.extend(accumulated_headers);
                return final_result;
            }
        }

        MiddlewareResult {
            proceed: true,
            status: None,
            body: None,
            response_headers: accumulated_headers,
        }
    }
}

impl Default for MiddlewareChain {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctx(method: &str, path: &str) -> RequestContext {
        RequestContext::new("req-001", HttpMethod::from(method), path, "127.0.0.1")
    }

    #[test]
    fn test_http_method_from_str() {
        assert_eq!(HttpMethod::from("GET"), HttpMethod::Get);
        assert_eq!(HttpMethod::from("post"), HttpMethod::Post);
        assert_eq!(HttpMethod::from("delete"), HttpMethod::Delete);
    }

    #[test]
    fn test_http_method_display() {
        assert_eq!(HttpMethod::Get.to_string(), "GET");
        assert_eq!(HttpMethod::Post.to_string(), "POST");
        assert_eq!(
            HttpMethod::Other("CUSTOM".to_string()).to_string(),
            "CUSTOM"
        );
    }

    #[test]
    fn test_request_context_new() {
        let ctx = make_ctx("GET", "/api/v1/media");
        assert_eq!(ctx.method, HttpMethod::Get);
        assert_eq!(ctx.path, "/api/v1/media");
        assert_eq!(ctx.remote_addr, "127.0.0.1");
        assert!(ctx.user_id.is_none());
    }

    #[test]
    fn test_request_context_with_header() {
        let ctx = make_ctx("GET", "/").with_header("Authorization", "Bearer token123");
        assert_eq!(
            ctx.headers.get("Authorization").map(String::as_str),
            Some("Bearer token123")
        );
    }

    #[test]
    fn test_middleware_result_proceed() {
        let r = MiddlewareResult::proceed();
        assert!(r.proceed);
        assert!(r.status.is_none());
    }

    #[test]
    fn test_middleware_result_terminate() {
        let r = MiddlewareResult::terminate(401, "Unauthorized");
        assert!(!r.proceed);
        assert_eq!(r.status, Some(401));
        assert_eq!(r.body.as_deref(), Some("Unauthorized"));
    }

    #[test]
    fn test_logging_middleware_process() {
        let mw = LoggingMiddleware::new();
        let ctx = make_ctx("GET", "/health");
        let result = mw.process(&ctx);
        assert!(result.proceed);
    }

    #[test]
    fn test_cors_allowed_origin_wildcard() {
        let mw = CorsMiddleware::new(CorsConfig::default());
        assert!(mw.is_origin_allowed("https://example.com"));
        assert!(mw.is_origin_allowed("http://localhost:3000"));
    }

    #[test]
    fn test_cors_restricted_origin() {
        let config = CorsConfig {
            allowed_origins: vec!["https://trusted.com".to_string()],
            ..CorsConfig::default()
        };
        let mw = CorsMiddleware::new(config);
        assert!(mw.is_origin_allowed("https://trusted.com"));
        assert!(!mw.is_origin_allowed("https://evil.com"));
    }

    #[test]
    fn test_cors_preflight_terminates() {
        let mw = CorsMiddleware::new(CorsConfig::default());
        let ctx = make_ctx("OPTIONS", "/api/v1/media").with_header("Origin", "https://example.com");
        let result = mw.process(&ctx);
        assert!(!result.proceed);
        assert_eq!(result.status, Some(204));
    }

    #[test]
    fn test_auth_middleware_public_path() {
        let mw = AuthMiddleware::new(vec!["/health".to_string(), "/ready".to_string()]);
        let ctx = make_ctx("GET", "/health");
        let result = mw.process(&ctx);
        assert!(result.proceed);
    }

    #[test]
    fn test_auth_middleware_missing_token() {
        let mw = AuthMiddleware::new(vec!["/health".to_string()]);
        let ctx = make_ctx("GET", "/api/v1/media");
        let result = mw.process(&ctx);
        assert!(!result.proceed);
        assert_eq!(result.status, Some(401));
    }

    #[test]
    fn test_auth_middleware_valid_token() {
        let mw = AuthMiddleware::new(vec![]);
        let ctx =
            make_ctx("GET", "/api/v1/media").with_header("Authorization", "Bearer valid_token_123");
        let result = mw.process(&ctx);
        assert!(result.proceed);
    }

    #[test]
    fn test_auth_extract_token() {
        let mw = AuthMiddleware::new(vec![]);
        assert_eq!(mw.extract_token("Bearer my_jwt"), Some("my_jwt"));
        assert!(mw.extract_token("Basic dXNlcjpwYXNz").is_none());
    }

    #[test]
    fn test_compression_negotiate_brotli() {
        let mw = CompressionMiddleware::new(CompressionConfig::default());
        assert_eq!(mw.negotiate_encoding("br, gzip, deflate"), Some("br"));
    }

    #[test]
    fn test_compression_negotiate_gzip_fallback() {
        let config = CompressionConfig {
            enable_brotli: false,
            ..CompressionConfig::default()
        };
        let mw = CompressionMiddleware::new(config);
        assert_eq!(mw.negotiate_encoding("gzip, deflate"), Some("gzip"));
    }

    #[test]
    fn test_compression_no_encoding() {
        let mw = CompressionMiddleware::new(CompressionConfig::default());
        assert!(mw.negotiate_encoding("identity").is_none());
    }

    #[test]
    fn test_compression_compressible_type() {
        let mw = CompressionMiddleware::new(CompressionConfig::default());
        assert!(mw.is_compressible("application/json"));
        assert!(mw.is_compressible("text/html; charset=utf-8"));
        assert!(!mw.is_compressible("video/mp4"));
    }

    #[test]
    fn test_middleware_chain_proceeds_through_all() {
        let chain = MiddlewareChain::new()
            .with_logging(LoggingMiddleware::new())
            .with_cors(CorsMiddleware::new(CorsConfig::default()))
            .with_compression(CompressionMiddleware::new(CompressionConfig::default()));

        let ctx = make_ctx("GET", "/api/v1/media")
            .with_header("Origin", "https://example.com")
            .with_header("Accept-Encoding", "br, gzip");
        let result = chain.run(&ctx);
        assert!(result.proceed);
        // CORS and compression headers should be accumulated
        assert!(result
            .response_headers
            .contains_key("Access-Control-Allow-Origin"));
    }

    #[test]
    fn test_middleware_chain_auth_blocks() {
        let chain = MiddlewareChain::new()
            .with_logging(LoggingMiddleware::new())
            .with_auth(AuthMiddleware::new(vec![]));

        let ctx = make_ctx("POST", "/api/v1/media");
        let result = chain.run(&ctx);
        assert!(!result.proceed);
        assert_eq!(result.status, Some(401));
    }
}

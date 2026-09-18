//! Request validation middleware for the OxiMedia server.
//!
//! Provides composable validation rules for HTTP requests including
//! field presence checks, content-type negotiation, body size limits,
//! and structured validation error reporting.

#![allow(dead_code)]
#![allow(missing_docs)]

/// A single validation error describing what went wrong and where.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationError {
    /// The field or source that triggered the error (e.g. "Content-Type").
    pub field: String,
    /// Human-readable explanation of the problem.
    pub message: String,
    /// HTTP status code that should be returned (typically 400 or 415).
    pub status_code: u16,
}

impl ValidationError {
    /// Creates a new validation error.
    pub fn new(field: impl Into<String>, message: impl Into<String>, status_code: u16) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
            status_code,
        }
    }

    /// Shortcut for a 400 Bad Request error.
    pub fn bad_request(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(field, message, 400)
    }

    /// Shortcut for a 415 Unsupported Media Type error.
    pub fn unsupported_media_type(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(field, message, 415)
    }

    /// Shortcut for a 413 Payload Too Large error.
    pub fn payload_too_large(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(field, message, 413)
    }
}

/// Outcome of running the validator against a request.
#[derive(Debug, Clone)]
pub enum ValidationOutcome {
    /// All rules passed; the request is valid.
    Valid,
    /// One or more rules failed.
    Invalid(Vec<ValidationError>),
}

impl ValidationOutcome {
    /// Returns `true` when all rules passed.
    pub fn is_valid(&self) -> bool {
        matches!(self, ValidationOutcome::Valid)
    }

    /// Returns the list of errors, or an empty slice when valid.
    pub fn errors(&self) -> &[ValidationError] {
        match self {
            ValidationOutcome::Valid => &[],
            ValidationOutcome::Invalid(errs) => errs,
        }
    }

    /// Returns the highest HTTP status code across all errors,
    /// or 200 when there are no errors.
    pub fn http_status(&self) -> u16 {
        match self {
            ValidationOutcome::Valid => 200,
            ValidationOutcome::Invalid(errs) => {
                errs.iter().map(|e| e.status_code).max().unwrap_or(400)
            }
        }
    }
}

/// A minimal snapshot of an HTTP request used for validation.
#[derive(Debug, Clone)]
pub struct RequestSnapshot {
    /// HTTP method string (e.g. "GET", "POST").
    pub method: String,
    /// Request path (e.g. "/api/v1/media").
    pub path: String,
    /// Header name-value pairs.
    pub headers: Vec<(String, String)>,
    /// Body size in bytes (0 if there is no body).
    pub body_bytes: usize,
}

impl RequestSnapshot {
    /// Creates a new `RequestSnapshot`.
    pub fn new(
        method: impl Into<String>,
        path: impl Into<String>,
        headers: Vec<(String, String)>,
        body_bytes: usize,
    ) -> Self {
        Self {
            method: method.into(),
            path: path.into(),
            headers,
            body_bytes,
        }
    }

    /// Looks up a header by name (case-insensitive).
    pub fn header(&self, name: &str) -> Option<&str> {
        let lower = name.to_lowercase();
        self.headers
            .iter()
            .find(|(k, _)| k.to_lowercase() == lower)
            .map(|(_, v)| v.as_str())
    }

    /// Returns `true` if the request carries a body (POST/PUT/PATCH with bytes > 0).
    pub fn has_body(&self) -> bool {
        self.body_bytes > 0
    }
}

/// Validates that the `Content-Type` header matches one of the accepted types.
#[derive(Debug, Clone)]
pub struct ContentTypeRule {
    /// Accepted MIME types (e.g. `["application/json"]`).
    pub accepted: Vec<String>,
}

impl ContentTypeRule {
    /// Creates a new rule accepting the given MIME types.
    pub fn new(accepted: Vec<&str>) -> Self {
        Self {
            accepted: accepted.iter().map(|s| (*s).to_string()).collect(),
        }
    }

    /// JSON-only shortcut.
    pub fn json() -> Self {
        Self::new(vec!["application/json"])
    }

    /// Multipart-only shortcut (e.g. for file uploads).
    pub fn multipart() -> Self {
        Self::new(vec!["multipart/form-data"])
    }

    /// Runs the rule against `req`. Returns `None` when the rule passes.
    pub fn check(&self, req: &RequestSnapshot) -> Option<ValidationError> {
        // Only validate bodies that actually carry content.
        if !req.has_body() {
            return None;
        }
        let ct = req.header("Content-Type").unwrap_or("");
        let ok = self.accepted.iter().any(|a| ct.starts_with(a.as_str()));
        if ok {
            None
        } else {
            Some(ValidationError::unsupported_media_type(
                "Content-Type",
                format!("Expected one of {:?}, got '{ct}'", self.accepted),
            ))
        }
    }
}

/// Validates that the request body does not exceed a maximum size.
#[derive(Debug, Clone)]
pub struct BodySizeRule {
    /// Maximum allowed body size in bytes.
    pub max_bytes: usize,
}

impl BodySizeRule {
    /// Creates a new size rule.
    pub fn new(max_bytes: usize) -> Self {
        Self { max_bytes }
    }

    /// Runs the rule. Returns `None` when the rule passes.
    pub fn check(&self, req: &RequestSnapshot) -> Option<ValidationError> {
        if req.body_bytes > self.max_bytes {
            Some(ValidationError::payload_too_large(
                "body",
                format!(
                    "Body size {} bytes exceeds maximum {} bytes",
                    req.body_bytes, self.max_bytes
                ),
            ))
        } else {
            None
        }
    }
}

/// Validates that required headers are present on the request.
#[derive(Debug, Clone)]
pub struct RequiredHeadersRule {
    /// Header names that must be present.
    pub required: Vec<String>,
}

impl RequiredHeadersRule {
    /// Creates a new rule requiring the specified headers.
    pub fn new(required: Vec<&str>) -> Self {
        Self {
            required: required.iter().map(|s| (*s).to_string()).collect(),
        }
    }

    /// Runs the rule. Returns errors for each missing header.
    pub fn check(&self, req: &RequestSnapshot) -> Vec<ValidationError> {
        self.required
            .iter()
            .filter(|name| req.header(name).is_none())
            .map(|name| {
                ValidationError::bad_request(
                    name.clone(),
                    format!("Required header '{name}' is missing"),
                )
            })
            .collect()
    }
}

/// Validates that the HTTP method is one of the allowed methods for the path.
#[derive(Debug, Clone)]
pub struct MethodRule {
    /// Allowed HTTP methods (uppercase).
    pub allowed: Vec<String>,
}

impl MethodRule {
    /// Creates a new rule allowing the given methods.
    pub fn new(allowed: Vec<&str>) -> Self {
        Self {
            allowed: allowed.iter().map(|s| s.to_uppercase()).collect(),
        }
    }

    /// Runs the rule. Returns `None` when the method is allowed.
    pub fn check(&self, req: &RequestSnapshot) -> Option<ValidationError> {
        let method = req.method.to_uppercase();
        if self.allowed.contains(&method) {
            None
        } else {
            Some(ValidationError::new(
                "method",
                format!(
                    "Method '{method}' is not allowed; accepted: {:?}",
                    self.allowed
                ),
                405,
            ))
        }
    }
}

/// A composable request validator that runs multiple rules.
#[derive(Debug, Default)]
pub struct RequestValidator {
    content_type: Option<ContentTypeRule>,
    body_size: Option<BodySizeRule>,
    required_headers: Option<RequiredHeadersRule>,
    method: Option<MethodRule>,
}

impl RequestValidator {
    /// Creates an empty validator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Attaches a content-type rule.
    #[must_use]
    pub fn with_content_type(mut self, rule: ContentTypeRule) -> Self {
        self.content_type = Some(rule);
        self
    }

    /// Attaches a body size rule.
    #[must_use]
    pub fn with_body_size(mut self, rule: BodySizeRule) -> Self {
        self.body_size = Some(rule);
        self
    }

    /// Attaches a required-headers rule.
    #[must_use]
    pub fn with_required_headers(mut self, rule: RequiredHeadersRule) -> Self {
        self.required_headers = Some(rule);
        self
    }

    /// Attaches a method rule.
    #[must_use]
    pub fn with_method(mut self, rule: MethodRule) -> Self {
        self.method = Some(rule);
        self
    }

    /// Runs all attached rules against `req` and returns an aggregated outcome.
    pub fn validate(&self, req: &RequestSnapshot) -> ValidationOutcome {
        let mut errors: Vec<ValidationError> = Vec::new();

        if let Some(ref rule) = self.content_type {
            if let Some(err) = rule.check(req) {
                errors.push(err);
            }
        }
        if let Some(ref rule) = self.body_size {
            if let Some(err) = rule.check(req) {
                errors.push(err);
            }
        }
        if let Some(ref rule) = self.required_headers {
            errors.extend(rule.check(req));
        }
        if let Some(ref rule) = self.method {
            if let Some(err) = rule.check(req) {
                errors.push(err);
            }
        }

        if errors.is_empty() {
            ValidationOutcome::Valid
        } else {
            ValidationOutcome::Invalid(errors)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn json_post(body_bytes: usize) -> RequestSnapshot {
        RequestSnapshot::new(
            "POST",
            "/api/v1/media",
            vec![("Content-Type".to_string(), "application/json".to_string())],
            body_bytes,
        )
    }

    fn empty_get() -> RequestSnapshot {
        RequestSnapshot::new("GET", "/api/v1/media", vec![], 0)
    }

    // --- ValidationError ---

    #[test]
    fn test_validation_error_bad_request() {
        let e = ValidationError::bad_request("field", "missing");
        assert_eq!(e.status_code, 400);
        assert_eq!(e.field, "field");
    }

    #[test]
    fn test_validation_error_unsupported_media_type() {
        let e = ValidationError::unsupported_media_type("Content-Type", "bad type");
        assert_eq!(e.status_code, 415);
    }

    #[test]
    fn test_validation_error_payload_too_large() {
        let e = ValidationError::payload_too_large("body", "too big");
        assert_eq!(e.status_code, 413);
    }

    // --- ValidationOutcome ---

    #[test]
    fn test_outcome_valid_is_valid() {
        assert!(ValidationOutcome::Valid.is_valid());
    }

    #[test]
    fn test_outcome_invalid_is_not_valid() {
        let errs = vec![ValidationError::bad_request("f", "m")];
        assert!(!ValidationOutcome::Invalid(errs).is_valid());
    }

    #[test]
    fn test_outcome_http_status_valid() {
        assert_eq!(ValidationOutcome::Valid.http_status(), 200);
    }

    #[test]
    fn test_outcome_http_status_picks_highest() {
        let errs = vec![
            ValidationError::new("a", "msg", 400),
            ValidationError::new("b", "msg", 415),
        ];
        assert_eq!(ValidationOutcome::Invalid(errs).http_status(), 415);
    }

    #[test]
    fn test_outcome_errors_empty_on_valid() {
        assert!(ValidationOutcome::Valid.errors().is_empty());
    }

    // --- RequestSnapshot ---

    #[test]
    fn test_snapshot_header_case_insensitive() {
        let req = json_post(100);
        assert_eq!(req.header("content-type"), Some("application/json"));
        assert_eq!(req.header("CONTENT-TYPE"), Some("application/json"));
    }

    #[test]
    fn test_snapshot_header_missing() {
        let req = empty_get();
        assert!(req.header("Authorization").is_none());
    }

    #[test]
    fn test_snapshot_has_body_true() {
        assert!(json_post(50).has_body());
    }

    #[test]
    fn test_snapshot_has_body_false() {
        assert!(!empty_get().has_body());
    }

    // --- ContentTypeRule ---

    #[test]
    fn test_content_type_rule_passes_json() {
        let rule = ContentTypeRule::json();
        assert!(rule.check(&json_post(10)).is_none());
    }

    #[test]
    fn test_content_type_rule_fails_wrong_type() {
        let rule = ContentTypeRule::json();
        let req = RequestSnapshot::new(
            "POST",
            "/upload",
            vec![("Content-Type".to_string(), "text/plain".to_string())],
            20,
        );
        assert!(rule.check(&req).is_some());
        assert_eq!(
            rule.check(&req)
                .expect("should succeed in test")
                .status_code,
            415
        );
    }

    #[test]
    fn test_content_type_rule_skips_no_body() {
        let rule = ContentTypeRule::json();
        assert!(rule.check(&empty_get()).is_none());
    }

    // --- BodySizeRule ---

    #[test]
    fn test_body_size_rule_passes_within_limit() {
        let rule = BodySizeRule::new(1024);
        assert!(rule.check(&json_post(512)).is_none());
    }

    #[test]
    fn test_body_size_rule_fails_over_limit() {
        let rule = BodySizeRule::new(100);
        let err = rule.check(&json_post(200)).expect("should succeed in test");
        assert_eq!(err.status_code, 413);
    }

    #[test]
    fn test_body_size_rule_passes_exact_limit() {
        let rule = BodySizeRule::new(100);
        assert!(rule.check(&json_post(100)).is_none());
    }

    // --- RequiredHeadersRule ---

    #[test]
    fn test_required_headers_all_present() {
        let rule = RequiredHeadersRule::new(vec!["Content-Type"]);
        let errs = rule.check(&json_post(10));
        assert!(errs.is_empty());
    }

    #[test]
    fn test_required_headers_missing_one() {
        let rule = RequiredHeadersRule::new(vec!["Authorization"]);
        let errs = rule.check(&json_post(10));
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].field, "Authorization");
    }

    // --- MethodRule ---

    #[test]
    fn test_method_rule_allowed() {
        let rule = MethodRule::new(vec!["GET", "HEAD"]);
        assert!(rule.check(&empty_get()).is_none());
    }

    #[test]
    fn test_method_rule_not_allowed() {
        let rule = MethodRule::new(vec!["GET"]);
        let err = rule.check(&json_post(10)).expect("should succeed in test");
        assert_eq!(err.status_code, 405);
    }

    // --- RequestValidator composition ---

    #[test]
    fn test_validator_all_pass() {
        let v = RequestValidator::new()
            .with_content_type(ContentTypeRule::json())
            .with_body_size(BodySizeRule::new(10_000))
            .with_method(MethodRule::new(vec!["POST"]));
        let outcome = v.validate(&json_post(500));
        assert!(outcome.is_valid());
    }

    #[test]
    fn test_validator_multiple_failures_collected() {
        let v = RequestValidator::new()
            .with_content_type(ContentTypeRule::json())
            .with_required_headers(RequiredHeadersRule::new(vec![
                "Authorization",
                "X-Request-ID",
            ]));
        let req = RequestSnapshot::new(
            "POST",
            "/api",
            vec![("Content-Type".to_string(), "text/xml".to_string())],
            100,
        );
        let outcome = v.validate(&req);
        assert!(!outcome.is_valid());
        // content-type failure + 2 missing headers
        assert!(outcome.errors().len() >= 2);
    }
}

//! Lightweight HTTP request router: method matching, pattern matching, and route resolution.

#![allow(dead_code)]

/// HTTP method enum covering the most common verbs.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Patch,
    Delete,
    Head,
    Options,
    Trace,
    Connect,
}

impl HttpMethod {
    /// Returns `true` for methods that are considered "safe" (read-only by spec):
    /// GET, HEAD, OPTIONS, TRACE.
    pub fn is_safe(&self) -> bool {
        matches!(self, Self::Get | Self::Head | Self::Options | Self::Trace)
    }

    /// Returns `true` for idempotent methods: GET, HEAD, OPTIONS, TRACE, PUT, DELETE.
    pub fn is_idempotent(&self) -> bool {
        matches!(
            self,
            Self::Get | Self::Head | Self::Options | Self::Trace | Self::Put | Self::Delete
        )
    }

    /// Parses an HTTP method string (case-insensitive).  Returns `None` on unknown.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_uppercase().as_str() {
            "GET" => Some(Self::Get),
            "POST" => Some(Self::Post),
            "PUT" => Some(Self::Put),
            "PATCH" => Some(Self::Patch),
            "DELETE" => Some(Self::Delete),
            "HEAD" => Some(Self::Head),
            "OPTIONS" => Some(Self::Options),
            "TRACE" => Some(Self::Trace),
            "CONNECT" => Some(Self::Connect),
            _ => None,
        }
    }
}

/// A URL route pattern supporting static segments and named parameters
/// (`:param`).
///
/// Example: `/api/v1/media/:id/thumb`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutePattern {
    raw: String,
    segments: Vec<Segment>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Segment {
    Static(String),
    Param(String),
}

impl RoutePattern {
    /// Parses a route pattern string.
    pub fn new(pattern: impl Into<String>) -> Self {
        let raw: String = pattern.into();
        let segments = raw
            .split('/')
            .filter(|s| !s.is_empty())
            .map(|s| {
                if let Some(name) = s.strip_prefix(':') {
                    Segment::Param(name.to_string())
                } else {
                    Segment::Static(s.to_string())
                }
            })
            .collect();
        Self { raw, segments }
    }

    /// Returns `true` if `path` matches this pattern.
    ///
    /// Static segments must match exactly; `:param` segments match any
    /// non-empty path segment.
    pub fn matches(&self, path: &str) -> bool {
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        if parts.len() != self.segments.len() {
            return false;
        }
        parts
            .iter()
            .zip(self.segments.iter())
            .all(|(part, seg)| match seg {
                Segment::Static(s) => s == part,
                Segment::Param(_) => !part.is_empty(),
            })
    }

    /// Extracts named parameter values from `path`, returning a `Vec<(name, value)>`.
    pub fn extract_params<'a>(&self, path: &'a str) -> Vec<(String, &'a str)> {
        let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        if parts.len() != self.segments.len() {
            return vec![];
        }
        parts
            .iter()
            .zip(self.segments.iter())
            .filter_map(|(part, seg)| {
                if let Segment::Param(name) = seg {
                    Some((name.clone(), *part))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Returns the raw pattern string.
    pub fn as_str(&self) -> &str {
        &self.raw
    }
}

/// Result of a route lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteResult {
    /// A matching route was found; carries the handler name and extracted params.
    Found {
        handler: String,
        params: Vec<(String, String)>,
    },
    /// No route matched.
    NotFound,
    /// A path matched but not for the requested HTTP method.
    MethodNotAllowed,
}

impl RouteResult {
    /// Returns `true` if a handler was found.
    pub fn is_found(&self) -> bool {
        matches!(self, Self::Found { .. })
    }

    /// Returns the handler name if found.
    pub fn handler(&self) -> Option<&str> {
        if let Self::Found { handler, .. } = self {
            Some(handler.as_str())
        } else {
            None
        }
    }
}

/// A registered route entry.
#[derive(Debug, Clone)]
struct RouteEntry {
    method: HttpMethod,
    pattern: RoutePattern,
    handler: String,
}

/// Simple trie-less HTTP router backed by a linear route table.
#[derive(Debug, Default)]
pub struct Router {
    routes: Vec<RouteEntry>,
}

impl Router {
    /// Creates an empty router.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a route.
    pub fn add_route(
        &mut self,
        method: HttpMethod,
        pattern: impl Into<String>,
        handler: impl Into<String>,
    ) {
        self.routes.push(RouteEntry {
            method,
            pattern: RoutePattern::new(pattern),
            handler: handler.into(),
        });
    }

    /// Resolves `method` + `path` to a `RouteResult`.
    pub fn route(&self, method: &HttpMethod, path: &str) -> RouteResult {
        let mut path_matched = false;
        for entry in &self.routes {
            if entry.pattern.matches(path) {
                path_matched = true;
                if &entry.method == method {
                    let params = entry
                        .pattern
                        .extract_params(path)
                        .into_iter()
                        .map(|(k, v)| (k, v.to_string()))
                        .collect();
                    return RouteResult::Found {
                        handler: entry.handler.clone(),
                        params,
                    };
                }
            }
        }
        if path_matched {
            RouteResult::MethodNotAllowed
        } else {
            RouteResult::NotFound
        }
    }

    /// Returns the number of registered routes.
    pub fn route_count(&self) -> usize {
        self.routes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_http_method_is_safe_get() {
        assert!(HttpMethod::Get.is_safe());
    }

    #[test]
    fn test_http_method_is_safe_post() {
        assert!(!HttpMethod::Post.is_safe());
    }

    #[test]
    fn test_http_method_is_idempotent_put() {
        assert!(HttpMethod::Put.is_idempotent());
    }

    #[test]
    fn test_http_method_is_idempotent_post() {
        assert!(!HttpMethod::Post.is_idempotent());
    }

    #[test]
    fn test_http_method_parse_valid() {
        assert_eq!(HttpMethod::parse("get"), Some(HttpMethod::Get));
        assert_eq!(HttpMethod::parse("DELETE"), Some(HttpMethod::Delete));
    }

    #[test]
    fn test_http_method_parse_invalid() {
        assert_eq!(HttpMethod::parse("BREW"), None);
    }

    #[test]
    fn test_route_pattern_static_match() {
        let p = RoutePattern::new("/api/v1/health");
        assert!(p.matches("/api/v1/health"));
        assert!(!p.matches("/api/v1/other"));
    }

    #[test]
    fn test_route_pattern_param_match() {
        let p = RoutePattern::new("/media/:id");
        assert!(p.matches("/media/42"));
        assert!(!p.matches("/media/42/thumb"));
    }

    #[test]
    fn test_route_pattern_extract_params() {
        let p = RoutePattern::new("/media/:id/variant/:v");
        let params = p.extract_params("/media/99/variant/hd");
        assert_eq!(
            params,
            vec![("id".to_string(), "99"), ("v".to_string(), "hd")]
        );
    }

    #[test]
    fn test_route_result_is_found() {
        let r = RouteResult::Found {
            handler: "h".to_string(),
            params: vec![],
        };
        assert!(r.is_found());
    }

    #[test]
    fn test_route_result_not_found_is_not_found() {
        assert!(!RouteResult::NotFound.is_found());
    }

    #[test]
    fn test_router_add_and_count() {
        let mut router = Router::new();
        router.add_route(HttpMethod::Get, "/health", "health_handler");
        router.add_route(HttpMethod::Post, "/upload", "upload_handler");
        assert_eq!(router.route_count(), 2);
    }

    #[test]
    fn test_router_route_found() {
        let mut router = Router::new();
        router.add_route(HttpMethod::Get, "/files/:id", "get_file");
        let result = router.route(&HttpMethod::Get, "/files/123");
        assert!(result.is_found());
        assert_eq!(result.handler(), Some("get_file"));
    }

    #[test]
    fn test_router_route_not_found() {
        let router = Router::new();
        let result = router.route(&HttpMethod::Get, "/nowhere");
        assert_eq!(result, RouteResult::NotFound);
    }

    #[test]
    fn test_router_route_method_not_allowed() {
        let mut router = Router::new();
        router.add_route(HttpMethod::Get, "/items/:id", "get_item");
        let result = router.route(&HttpMethod::Delete, "/items/7");
        assert_eq!(result, RouteResult::MethodNotAllowed);
    }
}

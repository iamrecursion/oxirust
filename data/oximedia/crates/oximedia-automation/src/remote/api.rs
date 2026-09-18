//! REST API for remote control.

use crate::remote::server::RemoteConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

/// API request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiRequest {
    /// Request method
    pub method: String,
    /// Request path
    pub path: String,
    /// Request parameters
    pub params: HashMap<String, String>,
    /// Request body
    pub body: Option<String>,
}

/// API response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse {
    /// Response status code
    pub status: u16,
    /// Response body
    pub body: String,
    /// Response headers
    pub headers: HashMap<String, String>,
}

impl ApiResponse {
    /// Create success response.
    pub fn ok(body: String) -> Self {
        Self {
            status: 200,
            body,
            headers: HashMap::new(),
        }
    }

    /// Create error response.
    pub fn error(status: u16, message: String) -> Self {
        Self {
            status,
            body: serde_json::json!({ "error": message }).to_string(),
            headers: HashMap::new(),
        }
    }

    /// Create not found response.
    pub fn not_found() -> Self {
        Self::error(404, "Not Found".to_string())
    }
}

/// API router.
#[allow(dead_code)]
pub struct ApiRouter {
    config: RemoteConfig,
}

impl ApiRouter {
    /// Create a new API router.
    pub fn new(config: RemoteConfig) -> Self {
        info!("Creating API router");

        Self { config }
    }

    /// Handle API request.
    pub async fn handle(&self, request: ApiRequest) -> ApiResponse {
        debug!("Handling API request: {} {}", request.method, request.path);

        // Route the request
        match request.path.as_str() {
            "/api/status" => self.handle_status().await,
            "/api/channels" => self.handle_channels(&request).await,
            "/api/playlist" => self.handle_playlist(&request).await,
            "/api/failover" => self.handle_failover(&request).await,
            "/api/eas" => self.handle_eas(&request).await,
            _ => ApiResponse::not_found(),
        }
    }

    /// Handle status request.
    async fn handle_status(&self) -> ApiResponse {
        let status = serde_json::json!({
            "status": "running",
            "version": "0.1.0",
        });

        ApiResponse::ok(status.to_string())
    }

    /// Handle channels request.
    async fn handle_channels(&self, request: &ApiRequest) -> ApiResponse {
        match request.method.as_str() {
            "GET" => {
                let channels = serde_json::json!({
                    "channels": []
                });
                ApiResponse::ok(channels.to_string())
            }
            _ => ApiResponse::error(405, "Method Not Allowed".to_string()),
        }
    }

    /// Handle playlist request.
    async fn handle_playlist(&self, request: &ApiRequest) -> ApiResponse {
        match request.method.as_str() {
            "GET" => {
                let playlist = serde_json::json!({
                    "items": []
                });
                ApiResponse::ok(playlist.to_string())
            }
            "POST" => {
                // Add item to playlist
                ApiResponse::ok("{}".to_string())
            }
            _ => ApiResponse::error(405, "Method Not Allowed".to_string()),
        }
    }

    /// Handle failover request.
    async fn handle_failover(&self, request: &ApiRequest) -> ApiResponse {
        match request.method.as_str() {
            "POST" => {
                // Trigger failover
                ApiResponse::ok("{}".to_string())
            }
            _ => ApiResponse::error(405, "Method Not Allowed".to_string()),
        }
    }

    /// Handle EAS request.
    async fn handle_eas(&self, request: &ApiRequest) -> ApiResponse {
        match request.method.as_str() {
            "POST" => {
                // Handle EAS alert
                ApiResponse::ok("{}".to_string())
            }
            _ => ApiResponse::error(405, "Method Not Allowed".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_api_router() {
        let config = RemoteConfig::default();
        let router = ApiRouter::new(config);

        let request = ApiRequest {
            method: "GET".to_string(),
            path: "/api/status".to_string(),
            params: HashMap::new(),
            body: None,
        };

        let response = router.handle(request).await;
        assert_eq!(response.status, 200);
    }

    #[tokio::test]
    async fn test_not_found() {
        let config = RemoteConfig::default();
        let router = ApiRouter::new(config);

        let request = ApiRequest {
            method: "GET".to_string(),
            path: "/api/unknown".to_string(),
            params: HashMap::new(),
            body: None,
        };

        let response = router.handle(request).await;
        assert_eq!(response.status, 404);
    }

    #[test]
    fn test_api_response() {
        let response = ApiResponse::ok("test".to_string());
        assert_eq!(response.status, 200);

        let error = ApiResponse::error(500, "error".to_string());
        assert_eq!(error.status, 500);
    }
}

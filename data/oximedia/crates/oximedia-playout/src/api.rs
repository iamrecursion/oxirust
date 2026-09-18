//! REST API and control interface
//!
//! Provides HTTP REST API and WebSocket support for remote control
//! and real-time status updates.

use crate::{PlayoutError, PlayoutServer, PlayoutState, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};
use uuid::Uuid;

/// API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    /// API server bind address
    pub bind_address: String,

    /// API server port
    pub port: u16,

    /// Enable authentication
    pub auth_enabled: bool,

    /// API key
    pub api_key: Option<String>,

    /// Enable WebSocket
    pub websocket_enabled: bool,

    /// CORS allowed origins
    pub cors_origins: Vec<String>,

    /// Request timeout in seconds
    pub request_timeout_sec: u32,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0".to_string(),
            port: 8080,
            auth_enabled: true,
            api_key: None,
            websocket_enabled: true,
            cors_origins: vec!["*".to_string()],
            request_timeout_sec: 30,
        }
    }
}

/// API server
pub struct ApiServer {
    config: ApiConfig,
    #[allow(dead_code)]
    playout: Arc<RwLock<Option<Arc<PlayoutServer>>>>,
}

impl ApiServer {
    /// Create new API server
    pub fn new(config: ApiConfig) -> Self {
        Self {
            config,
            playout: Arc::new(RwLock::new(None)),
        }
    }

    /// Start API server
    pub async fn start(&self) -> Result<()> {
        info!(
            "Starting API server on {}:{}",
            self.config.bind_address, self.config.port
        );

        // In real implementation, this would start actual HTTP server
        // For now, just log

        Ok(())
    }

    /// Stop API server
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping API server");
        Ok(())
    }

    /// Register playout server
    pub async fn register_playout(&self, server: Arc<PlayoutServer>) {
        *self.playout.write().await = Some(server);
    }
}

/// API request types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum ApiRequest {
    /// Get server status
    GetStatus,

    /// Start playout
    Start,

    /// Stop playout
    Stop,

    /// Pause playout
    Pause,

    /// Resume playout
    Resume,

    /// Load playlist
    LoadPlaylist { path: String },

    /// Get current playlist
    GetPlaylist,

    /// Get channel list
    GetChannels,

    /// Get channel status
    GetChannelStatus { channel_id: Uuid },

    /// Get content list
    GetContent { filter: Option<ContentFilter> },

    /// Ingest content
    IngestContent { path: String },

    /// Trigger manual failover
    TriggerFailover,

    /// Get as-run log
    GetAsRun {
        start: chrono::DateTime<chrono::Utc>,
        end: chrono::DateTime<chrono::Utc>,
    },
}

/// Content filter for queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentFilter {
    /// Filter by content type
    pub content_type: Option<String>,

    /// Filter by availability
    pub available_only: bool,

    /// Search query
    pub query: Option<String>,

    /// Limit results
    pub limit: Option<usize>,
}

/// API response types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", content = "data")]
pub enum ApiResponse {
    /// Success response
    Success {
        message: String,
        data: Option<serde_json::Value>,
    },

    /// Error response
    Error { code: String, message: String },

    /// Status response
    Status(StatusResponse),

    /// Playlist response
    Playlist(PlaylistResponse),

    /// Channels response
    Channels(ChannelsResponse),

    /// Content response
    Content(ContentResponse),
}

/// Status response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusResponse {
    /// Server state
    pub state: PlayoutState,

    /// Uptime in seconds
    pub uptime_sec: u64,

    /// Current playlist ID
    pub current_playlist: Option<Uuid>,

    /// Current item ID
    pub current_item: Option<Uuid>,

    /// System health
    pub health: HealthMetrics,
}

/// Health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetrics {
    /// CPU usage percentage
    pub cpu_usage: f32,

    /// Memory usage percentage
    pub memory_usage: f32,

    /// Disk usage percentage
    pub disk_usage: f32,

    /// Network status
    pub network_ok: bool,

    /// Active connections
    pub active_connections: u32,
}

/// Playlist response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaylistResponse {
    /// Playlist ID
    pub id: Uuid,

    /// Playlist name
    pub name: String,

    /// Number of items
    pub item_count: usize,

    /// Total duration in milliseconds
    pub total_duration_ms: u64,

    /// Current position
    pub current_position: Option<usize>,
}

/// Channels response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelsResponse {
    /// List of channels
    pub channels: Vec<ChannelInfo>,

    /// Total count
    pub total: usize,
}

/// Channel information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelInfo {
    /// Channel ID
    pub id: Uuid,

    /// Channel name
    pub name: String,

    /// Channel number
    pub number: u16,

    /// Current state
    pub state: String,

    /// Enabled flag
    pub enabled: bool,
}

/// Content response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentResponse {
    /// List of content items
    pub items: Vec<ContentInfo>,

    /// Total count
    pub total: usize,
}

/// Content information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentInfo {
    /// Content ID
    pub id: Uuid,

    /// Title
    pub title: String,

    /// Duration in milliseconds
    pub duration_ms: u64,

    /// Content type
    pub content_type: String,

    /// Availability
    pub available: bool,
}

/// WebSocket message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum WebSocketMessage {
    /// Status update
    StatusUpdate(StatusResponse),

    /// Playlist changed
    PlaylistChanged { playlist_id: Uuid },

    /// Item started
    ItemStarted { item_id: Uuid, title: String },

    /// Item ended
    ItemEnded { item_id: Uuid },

    /// Error occurred
    Error { message: String },

    /// Alert notification
    Alert { severity: String, message: String },
}

/// API client for testing
pub struct ApiClient {
    base_url: String,
    #[allow(dead_code)]
    api_key: Option<String>,
}

impl ApiClient {
    /// Create new API client
    pub fn new(base_url: String, api_key: Option<String>) -> Self {
        Self { base_url, api_key }
    }

    /// Send request
    pub async fn send_request(&self, _request: ApiRequest) -> Result<ApiResponse> {
        // In real implementation, this would make HTTP request
        debug!("Sending API request to {}", self.base_url);

        Ok(ApiResponse::Success {
            message: "Request processed".to_string(),
            data: None,
        })
    }

    /// Get server status
    pub async fn get_status(&self) -> Result<StatusResponse> {
        let response = self.send_request(ApiRequest::GetStatus).await?;

        match response {
            ApiResponse::Status(status) => Ok(status),
            _ => Err(PlayoutError::Config("Unexpected response type".to_string())),
        }
    }

    /// Start playout
    pub async fn start(&self) -> Result<()> {
        self.send_request(ApiRequest::Start).await?;
        Ok(())
    }

    /// Stop playout
    pub async fn stop(&self) -> Result<()> {
        self.send_request(ApiRequest::Stop).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_config_default() {
        let config = ApiConfig::default();
        assert_eq!(config.port, 8080);
        assert!(config.auth_enabled);
        assert!(config.websocket_enabled);
    }

    #[test]
    fn test_api_server_creation() {
        let config = ApiConfig::default();
        let server = ApiServer::new(config);
        assert_eq!(server.config.port, 8080);
    }

    #[test]
    fn test_api_client_creation() {
        let client = ApiClient::new(
            "http://localhost:8080".to_string(),
            Some("test-key".to_string()),
        );
        assert_eq!(client.base_url, "http://localhost:8080");
    }

    #[test]
    fn test_api_request_serialization() {
        let request = ApiRequest::GetStatus;
        let json = serde_json::to_string(&request).expect("should succeed in test");
        assert!(json.contains("GetStatus"));
    }

    #[test]
    fn test_api_response_serialization() {
        let response = ApiResponse::Success {
            message: "OK".to_string(),
            data: None,
        };
        let json = serde_json::to_string(&response).expect("should succeed in test");
        assert!(json.contains("Success"));
    }

    #[test]
    fn test_status_response_creation() {
        let status = StatusResponse {
            state: PlayoutState::Running,
            uptime_sec: 3600,
            current_playlist: Some(Uuid::new_v4()),
            current_item: None,
            health: HealthMetrics {
                cpu_usage: 45.0,
                memory_usage: 60.0,
                disk_usage: 30.0,
                network_ok: true,
                active_connections: 5,
            },
        };

        assert_eq!(status.uptime_sec, 3600);
        assert_eq!(status.health.cpu_usage, 45.0);
    }

    #[test]
    fn test_content_filter_creation() {
        let filter = ContentFilter {
            content_type: Some("video".to_string()),
            available_only: true,
            query: Some("test".to_string()),
            limit: Some(10),
        };

        assert_eq!(filter.content_type, Some("video".to_string()));
        assert!(filter.available_only);
    }

    #[test]
    fn test_channel_info_creation() {
        let info = ChannelInfo {
            id: Uuid::new_v4(),
            name: "Channel 1".to_string(),
            number: 1,
            state: "Running".to_string(),
            enabled: true,
        };

        assert_eq!(info.name, "Channel 1");
        assert_eq!(info.number, 1);
    }

    #[test]
    fn test_websocket_message_serialization() {
        let msg = WebSocketMessage::Alert {
            severity: "warning".to_string(),
            message: "Test alert".to_string(),
        };

        let json = serde_json::to_string(&msg).expect("should succeed in test");
        assert!(json.contains("Alert"));
        assert!(json.contains("warning"));
    }

    #[test]
    fn test_playlist_response_creation() {
        let response = PlaylistResponse {
            id: Uuid::new_v4(),
            name: "Daily Playlist".to_string(),
            item_count: 10,
            total_duration_ms: 3600000,
            current_position: Some(3),
        };

        assert_eq!(response.item_count, 10);
        assert_eq!(response.total_duration_ms, 3600000);
    }

    #[test]
    fn test_channels_response_creation() {
        let response = ChannelsResponse {
            channels: vec![ChannelInfo {
                id: Uuid::new_v4(),
                name: "Channel 1".to_string(),
                number: 1,
                state: "Running".to_string(),
                enabled: true,
            }],
            total: 1,
        };

        assert_eq!(response.total, 1);
        assert_eq!(response.channels.len(), 1);
    }

    #[test]
    fn test_content_info_creation() {
        let info = ContentInfo {
            id: Uuid::new_v4(),
            title: "Test Video".to_string(),
            duration_ms: 60000,
            content_type: "video".to_string(),
            available: true,
        };

        assert_eq!(info.title, "Test Video");
        assert!(info.available);
    }

    #[test]
    fn test_health_metrics_creation() {
        let metrics = HealthMetrics {
            cpu_usage: 50.0,
            memory_usage: 70.0,
            disk_usage: 40.0,
            network_ok: true,
            active_connections: 10,
        };

        assert_eq!(metrics.cpu_usage, 50.0);
        assert!(metrics.network_ok);
    }

    #[tokio::test]
    async fn test_api_client_request() {
        let client = ApiClient::new("http://localhost:8080".to_string(), None);

        let result = client.send_request(ApiRequest::GetStatus).await;
        assert!(result.is_ok());
    }
}

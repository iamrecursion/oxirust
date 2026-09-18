//! NMOS HTTP REST API server (IS-04, IS-05, IS-07).
//!
//! This module implements the AMWA NMOS HTTP REST APIs:
//! - IS-04 v1.3 Node API (discovery and registration)
//! - IS-05 v1.1 Connection API (device connection management)
//! - IS-07 v1.0 Event API (event and tally)

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio::sync::RwLock;

use super::channel_mapping::ChannelMappingRegistry;
use super::compatibility::CompatibilityRegistry;
use super::system::{NmosSystemApi, NmosSystemConfig};
use super::{
    Is07EventBus, Is07EventType, Is07Source, NmosConnectionManager, NmosFormat, NmosRegistry,
    NmosTransport, TallyController,
};

// ============================================================================
// Error type
// ============================================================================

/// Errors that can occur during NMOS HTTP server operation.
#[derive(Debug, thiserror::Error)]
pub enum NmosHttpError {
    /// 404 Not Found
    #[error("not found: {0}")]
    NotFound(String),
    /// 405 Method Not Allowed
    #[error("method not allowed")]
    MethodNotAllowed,
    /// 400 Bad Request
    #[error("bad request: {0}")]
    BadRequest(String),
    /// 500 Internal Server Error
    #[error("internal server error: {0}")]
    Internal(String),
    /// IO error (e.g. bind failure)
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// Hyper transport error
    #[error("hyper error: {0}")]
    Hyper(#[from] hyper::Error),
    /// JSON serialization/deserialization error
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    /// Body collection error
    #[error("body error: {0}")]
    Body(String),
    /// 422 Unprocessable Entity — constraint violation
    #[error("constraint violation: {0}")]
    ConstraintViolation(String),
}

impl NmosHttpError {
    pub(super) fn status_code(&self) -> StatusCode {
        match self {
            NmosHttpError::NotFound(_) => StatusCode::NOT_FOUND,
            NmosHttpError::MethodNotAllowed => StatusCode::METHOD_NOT_ALLOWED,
            NmosHttpError::BadRequest(_) => StatusCode::BAD_REQUEST,
            NmosHttpError::ConstraintViolation(_) => StatusCode::UNPROCESSABLE_ENTITY,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    pub(super) fn to_json_body(&self) -> String {
        let code = self.status_code().as_u16();
        json!({
            "code": code,
            "error": self.to_string(),
            "debug": null
        })
        .to_string()
    }
}

// ============================================================================
// IS-05 staged parameter types
// ============================================================================

/// IS-05 staged transport parameters for a sender.
#[derive(Debug, Clone, Default)]
pub struct StagedSenderParams {
    /// Whether the sender is master-enabled.
    pub master_enable: bool,
    /// Target receiver ID (if unicast).
    pub receiver_id: Option<String>,
    /// Destination IP address.
    pub destination_ip: Option<String>,
    /// Destination port.
    pub destination_port: Option<u16>,
    /// Source IP address.
    pub source_ip: Option<String>,
}

impl StagedSenderParams {
    pub(super) fn to_json(&self) -> Value {
        json!({
            "master_enable": self.master_enable,
            "receiver_id": self.receiver_id,
            "transport_params": [{
                "destination_ip": self.destination_ip,
                "destination_port": self.destination_port,
                "source_ip": self.source_ip,
                "rtp_enabled": true
            }],
            "activation": {
                "mode": null,
                "requested_time": null,
                "activation_time": null
            }
        })
    }
}

/// IS-05 staged connection parameters for a receiver.
#[derive(Debug, Clone, Default)]
pub struct StagedReceiverParams {
    /// Whether the receiver is master-enabled.
    pub master_enable: bool,
    /// Sender ID to subscribe to (None = disconnected).
    pub sender_id: Option<String>,
    /// Interface IP.
    pub interface_ip: Option<String>,
    /// Multicast group.
    pub multicast_ip: Option<String>,
    /// Source port.
    pub source_port: Option<u16>,
}

impl StagedReceiverParams {
    pub(super) fn to_json(&self) -> Value {
        json!({
            "master_enable": self.master_enable,
            "sender_id": self.sender_id,
            "transport_params": [{
                "interface_ip": self.interface_ip,
                "multicast_ip": self.multicast_ip,
                "source_port": self.source_port,
                "rtp_enabled": true
            }],
            "activation": {
                "mode": null,
                "requested_time": null,
                "activation_time": null
            }
        })
    }
}

// ============================================================================
// Route enum
// ============================================================================

/// Parsed route from an HTTP request.
#[derive(Debug, PartialEq, Eq)]
enum Route {
    // IS-04 Node API
    NodeRoot,
    NodeSelf,
    DeviceList,
    DeviceById(String),
    SourceList,
    SourceById(String),
    FlowList,
    FlowById(String),
    SenderList,
    SenderById(String),
    ReceiverList,
    ReceiverById(String),
    // IS-05 Connection API
    ConnectionRoot,
    ConnectionSenderList,
    ConnectionSenderRoot(String),
    ConnectionSenderStaged(String),
    ConnectionSenderActive(String),
    ConnectionSenderConstraints(String),
    ConnectionReceiverList,
    ConnectionReceiverRoot(String),
    ConnectionReceiverStaged(String),
    ConnectionReceiverActive(String),
    ConnectionReceiverConstraints(String),
    // IS-07 Event API
    EventRoot,
    EventSourceList,
    EventSourceById(String),
    EventSourceState(String),
    // IS-08 Channel Mapping API
    ChannelMappingRoot,
    ChannelMappingActivationList,
    ChannelMappingActivationDevice(String),
    ChannelMappingActivationActive(String),
    ChannelMappingActivationStaged(String),
    ChannelMappingIo,
    // IS-09 System API
    SystemRoot,
    SystemGlobal,
    SystemHealth,
    // IS-11 Stream Compatibility Management API
    StreamCompatRoot,
    StreamCompatSenderList,
    StreamCompatSenderActiveConstraints(String),
    StreamCompatSenderStatus(String),
    StreamCompatReceiverList,
    StreamCompatReceiverActiveConstraints(String),
    StreamCompatReceiverStatus(String),
    // Error routes
    NotFound,
    MethodNotAllowed,
}

/// Parse an incoming request into a `Route`.
fn resolve_route(method: &str, path: &str) -> Route {
    // Strip trailing slash for uniform matching (except root paths).
    let path = path.trim_end_matches('/');
    let parts: Vec<&str> = path.trim_start_matches('/').split('/').collect();

    match (method, parts.as_slice()) {
        // IS-04 Node API ─────────────────────────────────────────────────────
        ("GET", ["x-nmos", "node", "v1.3"]) => Route::NodeRoot,
        ("GET", ["x-nmos", "node", "v1.3", "self"]) => Route::NodeSelf,
        ("GET", ["x-nmos", "node", "v1.3", "devices"]) => Route::DeviceList,
        ("GET", ["x-nmos", "node", "v1.3", "devices", id]) => Route::DeviceById((*id).to_string()),
        ("GET", ["x-nmos", "node", "v1.3", "sources"]) => Route::SourceList,
        ("GET", ["x-nmos", "node", "v1.3", "sources", id]) => Route::SourceById((*id).to_string()),
        ("GET", ["x-nmos", "node", "v1.3", "flows"]) => Route::FlowList,
        ("GET", ["x-nmos", "node", "v1.3", "flows", id]) => Route::FlowById((*id).to_string()),
        ("GET", ["x-nmos", "node", "v1.3", "senders"]) => Route::SenderList,
        ("GET", ["x-nmos", "node", "v1.3", "senders", id]) => Route::SenderById((*id).to_string()),
        ("GET", ["x-nmos", "node", "v1.3", "receivers"]) => Route::ReceiverList,
        ("GET", ["x-nmos", "node", "v1.3", "receivers", id]) => {
            Route::ReceiverById((*id).to_string())
        }

        // IS-05 Connection API ────────────────────────────────────────────────
        ("GET", ["x-nmos", "connection", "v1.1"]) => Route::ConnectionRoot,
        ("GET", ["x-nmos", "connection", "v1.1", "single", "senders"]) => {
            Route::ConnectionSenderList
        }
        ("GET", ["x-nmos", "connection", "v1.1", "single", "senders", id]) => {
            Route::ConnectionSenderRoot((*id).to_string())
        }
        ("GET", ["x-nmos", "connection", "v1.1", "single", "senders", id, "staged"]) => {
            Route::ConnectionSenderStaged((*id).to_string())
        }
        ("PATCH", ["x-nmos", "connection", "v1.1", "single", "senders", id, "staged"]) => {
            Route::ConnectionSenderStaged((*id).to_string())
        }
        ("GET", ["x-nmos", "connection", "v1.1", "single", "senders", id, "active"]) => {
            Route::ConnectionSenderActive((*id).to_string())
        }
        ("GET", ["x-nmos", "connection", "v1.1", "single", "senders", id, "constraints"]) => {
            Route::ConnectionSenderConstraints((*id).to_string())
        }
        ("GET", ["x-nmos", "connection", "v1.1", "single", "receivers"]) => {
            Route::ConnectionReceiverList
        }
        ("GET", ["x-nmos", "connection", "v1.1", "single", "receivers", id]) => {
            Route::ConnectionReceiverRoot((*id).to_string())
        }
        ("GET", ["x-nmos", "connection", "v1.1", "single", "receivers", id, "staged"]) => {
            Route::ConnectionReceiverStaged((*id).to_string())
        }
        ("PATCH", ["x-nmos", "connection", "v1.1", "single", "receivers", id, "staged"]) => {
            Route::ConnectionReceiverStaged((*id).to_string())
        }
        ("GET", ["x-nmos", "connection", "v1.1", "single", "receivers", id, "active"]) => {
            Route::ConnectionReceiverActive((*id).to_string())
        }
        ("GET", ["x-nmos", "connection", "v1.1", "single", "receivers", id, "constraints"]) => {
            Route::ConnectionReceiverConstraints((*id).to_string())
        }

        // IS-07 Event API ─────────────────────────────────────────────────────
        ("GET", ["x-nmos", "events", "v1.0"]) => Route::EventRoot,
        ("GET", ["x-nmos", "events", "v1.0", "sources"]) => Route::EventSourceList,
        ("GET", ["x-nmos", "events", "v1.0", "sources", id]) => {
            Route::EventSourceById((*id).to_string())
        }
        ("POST", ["x-nmos", "events", "v1.0", "sources", id, "state"]) => {
            Route::EventSourceState((*id).to_string())
        }

        // IS-08 Channel Mapping API ───────────────────────────────────────────
        ("GET", ["x-nmos", "channelmapping", "v1.0"]) => Route::ChannelMappingRoot,
        ("GET", ["x-nmos", "channelmapping", "v1.0", "map", "activations"]) => {
            Route::ChannelMappingActivationList
        }
        ("GET", ["x-nmos", "channelmapping", "v1.0", "map", "activations", device_id]) => {
            Route::ChannelMappingActivationDevice((*device_id).to_string())
        }
        (
            "GET",
            ["x-nmos", "channelmapping", "v1.0", "map", "activations", device_id, "active"],
        ) => Route::ChannelMappingActivationActive((*device_id).to_string()),
        (
            "GET",
            ["x-nmos", "channelmapping", "v1.0", "map", "activations", device_id, "staged"],
        ) => Route::ChannelMappingActivationStaged((*device_id).to_string()),
        (
            "POST",
            ["x-nmos", "channelmapping", "v1.0", "map", "activations", device_id, "staged"],
        ) => Route::ChannelMappingActivationStaged((*device_id).to_string()),
        ("GET", ["x-nmos", "channelmapping", "v1.0", "io"]) => Route::ChannelMappingIo,

        // IS-09 System API ─────────────────────────────────────────────────
        ("GET", ["x-nmos", "system", "v1.0"]) => Route::SystemRoot,
        ("GET", ["x-nmos", "system", "v1.0", "global"]) => Route::SystemGlobal,
        ("GET", ["x-nmos", "system", "v1.0", "health"]) => Route::SystemHealth,

        // IS-11 Stream Compatibility Management API ────────────────────────
        ("GET", ["x-nmos", "streamcompatibility", "v1.0"]) => Route::StreamCompatRoot,
        ("GET", ["x-nmos", "streamcompatibility", "v1.0", "senders"]) => {
            Route::StreamCompatSenderList
        }
        (
            "GET" | "PUT",
            ["x-nmos", "streamcompatibility", "v1.0", "senders", id, "active_constraints"],
        ) => Route::StreamCompatSenderActiveConstraints((*id).to_string()),
        ("GET", ["x-nmos", "streamcompatibility", "v1.0", "senders", id, "status"]) => {
            Route::StreamCompatSenderStatus((*id).to_string())
        }
        ("GET", ["x-nmos", "streamcompatibility", "v1.0", "receivers"]) => {
            Route::StreamCompatReceiverList
        }
        (
            "GET" | "PUT",
            ["x-nmos", "streamcompatibility", "v1.0", "receivers", id, "active_constraints"],
        ) => Route::StreamCompatReceiverActiveConstraints((*id).to_string()),
        ("GET", ["x-nmos", "streamcompatibility", "v1.0", "receivers", id, "status"]) => {
            Route::StreamCompatReceiverStatus((*id).to_string())
        }

        // Method not allowed for known paths but wrong method
        (_, ["x-nmos", ..]) => Route::MethodNotAllowed,

        _ => Route::NotFound,
    }
}

// ============================================================================
// JSON helpers
// ============================================================================

/// Build a JSON HTTP response with the given status code and body string.
pub(super) fn json_response(status: u16, body: String) -> Response<Full<Bytes>> {
    let status_code = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    Response::builder()
        .status(status_code)
        .header("Content-Type", "application/json")
        .header("Access-Control-Allow-Origin", "*")
        .header("Access-Control-Allow-Methods", "GET, PATCH, POST, OPTIONS")
        .header("Access-Control-Allow-Headers", "Content-Type")
        .body(Full::new(Bytes::from(body)))
        .unwrap_or_else(|_| {
            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Full::new(Bytes::from(
                    r#"{"code":500,"error":"response build failure","debug":null}"#,
                )))
                .expect("fallback response always valid")
        })
}

/// Serialize a NmosFormat to its NMOS URN string.
fn format_urn(f: &NmosFormat) -> &'static str {
    match f {
        NmosFormat::Video => "urn:x-nmos:format:video",
        NmosFormat::Audio => "urn:x-nmos:format:audio",
        NmosFormat::Data => "urn:x-nmos:format:data",
        NmosFormat::Mux => "urn:x-nmos:format:mux",
    }
}

/// Serialize a NmosTransport to its NMOS URN string.
fn transport_urn(t: &NmosTransport) -> &'static str {
    match t {
        NmosTransport::RtpMulticast => "urn:x-nmos:transport:rtp.mcast",
        NmosTransport::RtpUnicast => "urn:x-nmos:transport:rtp.ucast",
        NmosTransport::Dash => "urn:x-nmos:transport:dash",
        NmosTransport::Hls => "urn:x-nmos:transport:hls",
        NmosTransport::Srt => "urn:x-nmos:transport:srt",
    }
}

/// Serialize an Is07EventType to its NMOS type string.
fn event_type_str(t: &Is07EventType) -> &'static str {
    match t {
        Is07EventType::Boolean => "boolean",
        Is07EventType::Number => "number",
        Is07EventType::String => "string",
        Is07EventType::Enum => "string/enum",
    }
}

// ============================================================================
// Shared server state (Arc-wrapped for clone into service_fn closure)
// ============================================================================

#[derive(Clone)]
pub(super) struct ServerState {
    pub(super) registry: Arc<RwLock<NmosRegistry>>,
    pub(super) connection_manager: Arc<RwLock<NmosConnectionManager>>,
    pub(super) event_bus: Arc<RwLock<Is07EventBus>>,
    pub(super) tally: Arc<RwLock<TallyController>>,
    pub(super) event_sources: Arc<RwLock<HashMap<String, Is07Source>>>,
    pub(super) sender_staged: Arc<RwLock<HashMap<String, StagedSenderParams>>>,
    pub(super) receiver_staged: Arc<RwLock<HashMap<String, StagedReceiverParams>>>,
    /// IS-08 channel mapping registry.
    pub(super) channel_mapping: Arc<RwLock<ChannelMappingRegistry>>,
    /// IS-09 System API state.
    pub(super) system_api: Arc<RwLock<NmosSystemApi>>,
    /// IS-11 stream compatibility registry.
    pub(super) compatibility: Arc<RwLock<CompatibilityRegistry>>,
    pub(super) node_id: String,
}

// ============================================================================
// NmosHttpServer
// ============================================================================

/// NMOS HTTP server implementing IS-04, IS-05, IS-07, IS-08, IS-09, and IS-11 REST APIs.
pub struct NmosHttpServer {
    registry: Arc<RwLock<NmosRegistry>>,
    connection_manager: Arc<RwLock<NmosConnectionManager>>,
    event_bus: Arc<RwLock<Is07EventBus>>,
    tally: Arc<RwLock<TallyController>>,
    event_sources: Arc<RwLock<HashMap<String, Is07Source>>>,
    sender_staged: Arc<RwLock<HashMap<String, StagedSenderParams>>>,
    receiver_staged: Arc<RwLock<HashMap<String, StagedReceiverParams>>>,
    /// IS-08 channel mapping registry.
    channel_mapping: Arc<RwLock<ChannelMappingRegistry>>,
    /// IS-09 System API state.
    system_api: Arc<RwLock<NmosSystemApi>>,
    /// IS-11 stream compatibility registry.
    compatibility: Arc<RwLock<CompatibilityRegistry>>,
    bind_addr: SocketAddr,
    node_id: String,
}

impl NmosHttpServer {
    /// Create a new NMOS HTTP server.
    ///
    /// A default IS-09 [`NmosSystemConfig`] is constructed from `node_id` and
    /// `node_label`.  Use [`set_system_config`](Self::set_system_config) to
    /// replace it after construction.
    pub fn new(
        registry: Arc<RwLock<NmosRegistry>>,
        connection_manager: Arc<RwLock<NmosConnectionManager>>,
        event_bus: Arc<RwLock<Is07EventBus>>,
        tally: Arc<RwLock<TallyController>>,
        bind_addr: SocketAddr,
        node_id: impl Into<String>,
    ) -> Self {
        let node_id = node_id.into();
        let default_system_cfg =
            NmosSystemConfig::new(node_id.clone(), format!("OxiMedia Node {node_id}"));
        Self {
            registry,
            connection_manager,
            event_bus,
            tally,
            event_sources: Arc::new(RwLock::new(HashMap::new())),
            sender_staged: Arc::new(RwLock::new(HashMap::new())),
            receiver_staged: Arc::new(RwLock::new(HashMap::new())),
            channel_mapping: Arc::new(RwLock::new(ChannelMappingRegistry::new())),
            system_api: Arc::new(RwLock::new(NmosSystemApi::new(default_system_cfg))),
            compatibility: Arc::new(RwLock::new(CompatibilityRegistry::new())),
            bind_addr,
            node_id,
        }
    }

    /// Replace the IS-09 system configuration.
    pub async fn set_system_config(&self, config: NmosSystemConfig) {
        let mut guard = self.system_api.write().await;
        *guard = NmosSystemApi::new(config);
    }

    /// Register an IS-07 event source that the HTTP API will expose.
    pub async fn register_event_source(&self, source: Is07Source) {
        let mut guard = self.event_sources.write().await;
        guard.insert(source.id.clone(), source);
    }

    /// Start listening and serving requests.  Returns when the server shuts
    /// down or if a fatal bind/accept error occurs.
    pub async fn serve(self) -> Result<(), NmosHttpError> {
        let listener = TcpListener::bind(self.bind_addr).await?;

        let state = ServerState {
            registry: Arc::clone(&self.registry),
            connection_manager: Arc::clone(&self.connection_manager),
            event_bus: Arc::clone(&self.event_bus),
            tally: Arc::clone(&self.tally),
            event_sources: Arc::clone(&self.event_sources),
            sender_staged: Arc::clone(&self.sender_staged),
            receiver_staged: Arc::clone(&self.receiver_staged),
            channel_mapping: Arc::clone(&self.channel_mapping),
            system_api: Arc::clone(&self.system_api),
            compatibility: Arc::clone(&self.compatibility),
            node_id: self.node_id.clone(),
        };

        loop {
            let (stream, _addr) = listener.accept().await?;
            let state = state.clone();

            tokio::spawn(async move {
                let io = hyper_util::rt::TokioIo::new(stream);
                let svc = service_fn(move |req| {
                    let state = state.clone();
                    async move { handle_request(req, state).await }
                });
                if let Err(e) = http1::Builder::new().serve_connection(io, svc).await {
                    // Log but don't propagate individual connection errors.
                    eprintln!("NMOS HTTP connection error: {e}");
                }
            });
        }
    }
}

// ============================================================================
// Request handler
// ============================================================================

async fn handle_request(
    req: Request<hyper::body::Incoming>,
    state: ServerState,
) -> Result<Response<Full<Bytes>>, NmosHttpError> {
    let method = req.method().as_str().to_uppercase();
    let path = req.uri().path().to_string();

    // Handle CORS preflight.
    if method == "OPTIONS" {
        return Ok(json_response(204, String::new()));
    }

    let route = resolve_route(&method, &path);

    let resp = match route {
        // ── IS-04 Node API ─────────────────────────────────────────────────
        Route::NodeRoot => handle_node_root(),
        Route::NodeSelf => handle_node_self(&state).await,
        Route::DeviceList => handle_device_list(&state).await,
        Route::DeviceById(id) => handle_device_by_id(&state, &id).await,
        Route::SourceList => handle_source_list(&state).await,
        Route::SourceById(id) => handle_source_by_id(&state, &id).await,
        Route::FlowList => handle_flow_list(&state).await,
        Route::FlowById(id) => handle_flow_by_id(&state, &id).await,
        Route::SenderList => handle_sender_list(&state).await,
        Route::SenderById(id) => handle_sender_by_id(&state, &id).await,
        Route::ReceiverList => handle_receiver_list(&state).await,
        Route::ReceiverById(id) => handle_receiver_by_id(&state, &id).await,

        // ── IS-05 Connection API ────────────────────────────────────────────
        Route::ConnectionRoot => http_is05::handle_connection_root(),
        Route::ConnectionSenderList => http_is05::handle_connection_sender_list(&state).await,
        Route::ConnectionSenderRoot(id) => {
            http_is05::handle_connection_sender_root(&state, &id).await
        }
        Route::ConnectionSenderStaged(id) => {
            if method == "PATCH" {
                http_is05::handle_patch_sender_staged(req, &state, &id).await
            } else {
                http_is05::handle_get_sender_staged(&state, &id).await
            }
        }
        Route::ConnectionSenderActive(id) => http_is05::handle_get_sender_active(&state, &id).await,
        Route::ConnectionSenderConstraints(id) => {
            http_is05::handle_get_sender_constraints(&state, &id).await
        }
        Route::ConnectionReceiverList => http_is05::handle_connection_receiver_list(&state).await,
        Route::ConnectionReceiverRoot(id) => {
            http_is05::handle_connection_receiver_root(&state, &id).await
        }
        Route::ConnectionReceiverStaged(id) => {
            if method == "PATCH" {
                http_is05::handle_patch_receiver_staged(req, &state, &id).await
            } else {
                http_is05::handle_get_receiver_staged(&state, &id).await
            }
        }
        Route::ConnectionReceiverActive(id) => {
            http_is05::handle_get_receiver_active(&state, &id).await
        }
        Route::ConnectionReceiverConstraints(id) => {
            http_is05::handle_get_receiver_constraints(&state, &id).await
        }

        // ── IS-07 Event API ────────────────────────────────────────────────
        Route::EventRoot => handle_event_root(),
        Route::EventSourceList => handle_event_source_list(&state).await,
        Route::EventSourceById(id) => handle_event_source_by_id(&state, &id).await,
        Route::EventSourceState(id) => handle_post_event_source_state(req, &state, &id).await,

        // ── IS-08 Channel Mapping API ──────────────────────────────────────
        Route::ChannelMappingRoot => http_handlers::handle_channel_mapping_root(),
        Route::ChannelMappingActivationList => {
            http_handlers::handle_channel_mapping_activation_list(&state).await
        }
        Route::ChannelMappingActivationDevice(id) => {
            http_handlers::handle_channel_mapping_activation_device(&state, &id).await
        }
        Route::ChannelMappingActivationActive(id) => {
            http_handlers::handle_channel_mapping_activation_active(&state, &id).await
        }
        Route::ChannelMappingActivationStaged(id) => {
            if method == "POST" {
                http_handlers::handle_post_channel_mapping_staged(req, &state, &id).await
            } else {
                http_handlers::handle_get_channel_mapping_staged(&state, &id).await
            }
        }
        Route::ChannelMappingIo => http_handlers::handle_channel_mapping_io(&state).await,

        // ── IS-09 System API ───────────────────────────────────────────────
        Route::SystemRoot => http_handlers::handle_system_root(),
        Route::SystemGlobal => http_handlers::handle_system_global(&state).await,
        Route::SystemHealth => http_handlers::handle_system_health(&state).await,

        // ── IS-11 Stream Compatibility Management API ──────────────────────
        Route::StreamCompatRoot => http_handlers::handle_stream_compat_root(),
        Route::StreamCompatSenderList => {
            http_handlers::handle_stream_compat_sender_list(&state).await
        }
        Route::StreamCompatSenderActiveConstraints(id) => {
            if method == "PUT" {
                http_handlers::handle_put_stream_compat_sender_constraints(req, &state, &id).await
            } else {
                http_handlers::handle_get_stream_compat_sender_constraints(&state, &id).await
            }
        }
        Route::StreamCompatSenderStatus(id) => {
            http_handlers::handle_stream_compat_sender_status(&state, &id).await
        }
        Route::StreamCompatReceiverList => {
            http_handlers::handle_stream_compat_receiver_list(&state).await
        }
        Route::StreamCompatReceiverActiveConstraints(id) => {
            if method == "PUT" {
                http_handlers::handle_put_stream_compat_receiver_constraints(req, &state, &id).await
            } else {
                http_handlers::handle_get_stream_compat_receiver_constraints(&state, &id).await
            }
        }
        Route::StreamCompatReceiverStatus(id) => {
            http_handlers::handle_stream_compat_receiver_status(&state, &id).await
        }

        // ── Error routes ───────────────────────────────────────────────────
        Route::NotFound => {
            let err = NmosHttpError::NotFound(path);
            return Ok(json_response(
                err.status_code().as_u16(),
                err.to_json_body(),
            ));
        }
        Route::MethodNotAllowed => {
            let err = NmosHttpError::MethodNotAllowed;
            return Ok(json_response(
                err.status_code().as_u16(),
                err.to_json_body(),
            ));
        }
    };

    Ok(resp)
}

// ============================================================================
// IS-04 handlers
// ============================================================================

fn handle_node_root() -> Response<Full<Bytes>> {
    let body = json!([
        "self/",
        "devices/",
        "sources/",
        "flows/",
        "senders/",
        "receivers/"
    ])
    .to_string();
    json_response(200, body)
}

async fn handle_node_self(state: &ServerState) -> Response<Full<Bytes>> {
    let reg = state.registry.read().await;
    let body = if let Some(node) = reg.get_node(&state.node_id) {
        json!({
            "id": node.id,
            "label": node.label,
            "description": node.description,
            "tags": node.tags,
            "version": "1.3",
            "href": format!("http://localhost/{}", node.id),
            "hostname": node.label,
            "caps": {},
            "services": [],
            "clocks": [],
            "interfaces": []
        })
        .to_string()
    } else {
        let err = NmosHttpError::NotFound(format!("node {}", state.node_id));
        return json_response(err.status_code().as_u16(), err.to_json_body());
    };
    json_response(200, body)
}

async fn handle_device_list(state: &ServerState) -> Response<Full<Bytes>> {
    let reg = state.registry.read().await;
    // Access the private fields indirectly via the public API.
    // NmosRegistry doesn't expose an iterator over devices, so we must
    // collect by listing all device IDs.  Since the registry is opaque,
    // we serialize what the registry will give us.
    let devices: Vec<Value> = reg
        .all_devices()
        .iter()
        .map(|d| {
            json!({
                "id": d.id,
                "node_id": d.node_id,
                "label": d.label,
                "type": format!("{:?}", d.device_type),
                "senders": [],
                "receivers": []
            })
        })
        .collect();
    json_response(200, json!(devices).to_string())
}

async fn handle_device_by_id(state: &ServerState, id: &str) -> Response<Full<Bytes>> {
    let reg = state.registry.read().await;
    match reg.get_device(id) {
        Some(d) => {
            let body = json!({
                "id": d.id,
                "node_id": d.node_id,
                "label": d.label,
                "type": format!("{:?}", d.device_type),
                "senders": [],
                "receivers": []
            })
            .to_string();
            json_response(200, body)
        }
        None => {
            let err = NmosHttpError::NotFound(format!("device {id}"));
            json_response(err.status_code().as_u16(), err.to_json_body())
        }
    }
}

async fn handle_source_list(state: &ServerState) -> Response<Full<Bytes>> {
    let reg = state.registry.read().await;
    let sources: Vec<Value> = reg
        .all_sources()
        .iter()
        .map(|s| {
            json!({
                "id": s.id,
                "device_id": s.device_id,
                "label": s.label,
                "format": format_urn(&s.format),
                "clock_name": s.clock_name,
                "grain_rate": null,
                "description": "",
                "tags": {}
            })
        })
        .collect();
    json_response(200, json!(sources).to_string())
}

async fn handle_source_by_id(state: &ServerState, id: &str) -> Response<Full<Bytes>> {
    let reg = state.registry.read().await;
    match reg.get_source(id) {
        Some(s) => {
            let body = json!({
                "id": s.id,
                "device_id": s.device_id,
                "label": s.label,
                "format": format_urn(&s.format),
                "clock_name": s.clock_name,
                "grain_rate": null,
                "description": "",
                "tags": {}
            })
            .to_string();
            json_response(200, body)
        }
        None => {
            let err = NmosHttpError::NotFound(format!("source {id}"));
            json_response(err.status_code().as_u16(), err.to_json_body())
        }
    }
}

async fn handle_flow_list(state: &ServerState) -> Response<Full<Bytes>> {
    let reg = state.registry.read().await;
    let flows: Vec<Value> = reg
        .all_flows()
        .iter()
        .map(|f| {
            json!({
                "id": f.id,
                "source_id": f.source_id,
                "label": f.label,
                "format": format_urn(&f.format),
                "frame_rate": {
                    "numerator": f.frame_rate.0,
                    "denominator": f.frame_rate.1
                },
                "description": "",
                "tags": {}
            })
        })
        .collect();
    json_response(200, json!(flows).to_string())
}

async fn handle_flow_by_id(state: &ServerState, id: &str) -> Response<Full<Bytes>> {
    let reg = state.registry.read().await;
    match reg.get_flow(id) {
        Some(f) => {
            let body = json!({
                "id": f.id,
                "source_id": f.source_id,
                "label": f.label,
                "format": format_urn(&f.format),
                "frame_rate": {
                    "numerator": f.frame_rate.0,
                    "denominator": f.frame_rate.1
                },
                "description": "",
                "tags": {}
            })
            .to_string();
            json_response(200, body)
        }
        None => {
            let err = NmosHttpError::NotFound(format!("flow {id}"));
            json_response(err.status_code().as_u16(), err.to_json_body())
        }
    }
}

async fn handle_sender_list(state: &ServerState) -> Response<Full<Bytes>> {
    let reg = state.registry.read().await;
    let senders: Vec<Value> = reg
        .all_senders()
        .iter()
        .map(|s| {
            json!({
                "id": s.id,
                "flow_id": s.flow_id,
                "label": s.label,
                "transport": transport_urn(&s.transport),
                "device_id": null,
                "manifest_href": null,
                "description": "",
                "tags": {},
                "subscription": {
                    "receiver_id": null,
                    "active": false
                },
                "interface_bindings": []
            })
        })
        .collect();
    json_response(200, json!(senders).to_string())
}

async fn handle_sender_by_id(state: &ServerState, id: &str) -> Response<Full<Bytes>> {
    let reg = state.registry.read().await;
    match reg.get_sender(id) {
        Some(s) => {
            let body = json!({
                "id": s.id,
                "flow_id": s.flow_id,
                "label": s.label,
                "transport": transport_urn(&s.transport),
                "device_id": null,
                "manifest_href": null,
                "description": "",
                "tags": {},
                "subscription": {
                    "receiver_id": null,
                    "active": false
                },
                "interface_bindings": []
            })
            .to_string();
            json_response(200, body)
        }
        None => {
            let err = NmosHttpError::NotFound(format!("sender {id}"));
            json_response(err.status_code().as_u16(), err.to_json_body())
        }
    }
}

async fn handle_receiver_list(state: &ServerState) -> Response<Full<Bytes>> {
    let reg = state.registry.read().await;
    let receivers: Vec<Value> = reg
        .all_receivers()
        .iter()
        .map(|r| {
            json!({
                "id": r.id,
                "device_id": r.device_id,
                "label": r.label,
                "format": format_urn(&r.format),
                "transport": "urn:x-nmos:transport:rtp",
                "description": "",
                "tags": {},
                "caps": {
                    "media_types": [r.format.media_type()]
                },
                "interface_bindings": [],
                "subscription": {
                    "sender_id": r.subscription,
                    "active": r.subscription.is_some()
                }
            })
        })
        .collect();
    json_response(200, json!(receivers).to_string())
}

async fn handle_receiver_by_id(state: &ServerState, id: &str) -> Response<Full<Bytes>> {
    let reg = state.registry.read().await;
    match reg.get_receiver(id) {
        Some(r) => {
            let body = json!({
                "id": r.id,
                "device_id": r.device_id,
                "label": r.label,
                "format": format_urn(&r.format),
                "transport": "urn:x-nmos:transport:rtp",
                "description": "",
                "tags": {},
                "caps": {
                    "media_types": [r.format.media_type()]
                },
                "interface_bindings": [],
                "subscription": {
                    "sender_id": r.subscription,
                    "active": r.subscription.is_some()
                }
            })
            .to_string();
            json_response(200, body)
        }
        None => {
            let err = NmosHttpError::NotFound(format!("receiver {id}"));
            json_response(err.status_code().as_u16(), err.to_json_body())
        }
    }
}

// IS-05 handlers are in the `http_is05` sibling module.
use super::http_is05;

// ============================================================================
// IS-07 handlers
// ============================================================================

fn handle_event_root() -> Response<Full<Bytes>> {
    let body = json!(["sources/"]).to_string();
    json_response(200, body)
}

async fn handle_event_source_list(state: &ServerState) -> Response<Full<Bytes>> {
    let sources = state.event_sources.read().await;
    let ids: Vec<&str> = sources.keys().map(String::as_str).collect();
    json_response(200, json!(ids).to_string())
}

async fn handle_event_source_by_id(state: &ServerState, id: &str) -> Response<Full<Bytes>> {
    let sources = state.event_sources.read().await;
    match sources.get(id) {
        Some(src) => {
            let current_state = build_is07_state(src);
            let body = json!({
                "id": src.id,
                "label": src.label,
                "event_type": event_type_str(&src.event_type),
                "state": current_state
            })
            .to_string();
            json_response(200, body)
        }
        None => {
            let err = NmosHttpError::NotFound(format!("event source {id}"));
            json_response(err.status_code().as_u16(), err.to_json_body())
        }
    }
}

/// Build the IS-07 state object for an event source.
fn build_is07_state(src: &Is07Source) -> Value {
    match src.event_type {
        Is07EventType::Boolean => json!({
            "value": src.bool_state.unwrap_or(false)
        }),
        Is07EventType::Number => json!({
            "value": src.number_state.unwrap_or(0.0)
        }),
        Is07EventType::String | Is07EventType::Enum => json!({
            "value": src.string_state.as_deref().unwrap_or("")
        }),
    }
}

async fn handle_post_event_source_state(
    req: Request<hyper::body::Incoming>,
    state: &ServerState,
    id: &str,
) -> Response<Full<Bytes>> {
    // Verify source exists.
    {
        let sources = state.event_sources.read().await;
        if !sources.contains_key(id) {
            let err = NmosHttpError::NotFound(format!("event source {id}"));
            return json_response(err.status_code().as_u16(), err.to_json_body());
        }
    }

    let body_bytes = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => {
            let err = NmosHttpError::Body(e.to_string());
            return json_response(err.status_code().as_u16(), err.to_json_body());
        }
    };

    let payload: Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => {
            let err = NmosHttpError::BadRequest(e.to_string());
            return json_response(err.status_code().as_u16(), err.to_json_body());
        }
    };

    let mut sources = state.event_sources.write().await;
    let src = match sources.get_mut(id) {
        Some(s) => s,
        None => {
            let err = NmosHttpError::NotFound(format!("event source {id}"));
            return json_response(err.status_code().as_u16(), err.to_json_body());
        }
    };

    // Update state and emit event on the bus.
    let mut bus = state.event_bus.write().await;
    match src.event_type {
        Is07EventType::Boolean => {
            if let Some(v) = payload.get("value").and_then(|v| v.as_bool()) {
                src.bool_state = Some(v);
                bus.emit_boolean(id.to_string(), v);
            } else {
                let err = NmosHttpError::BadRequest(
                    "expected {\"value\": <bool>} for boolean source".to_string(),
                );
                return json_response(err.status_code().as_u16(), err.to_json_body());
            }
        }
        Is07EventType::Number => {
            if let Some(v) = payload.get("value").and_then(|v| v.as_f64()) {
                src.number_state = Some(v);
                bus.emit_number(id.to_string(), v);
            } else {
                let err = NmosHttpError::BadRequest(
                    "expected {\"value\": <number>} for number source".to_string(),
                );
                return json_response(err.status_code().as_u16(), err.to_json_body());
            }
        }
        Is07EventType::String | Is07EventType::Enum => {
            if let Some(v) = payload.get("value").and_then(|v| v.as_str()) {
                let v = v.to_string();
                bus.emit_string(id.to_string(), v.clone());
                src.string_state = Some(v);
            } else {
                let err = NmosHttpError::BadRequest(
                    "expected {\"value\": \"<string>\"} for string source".to_string(),
                );
                return json_response(err.status_code().as_u16(), err.to_json_body());
            }
        }
    }

    let new_state = build_is07_state(src);
    let body = json!({
        "id": src.id,
        "state": new_state
    })
    .to_string();
    json_response(200, body)
}

// IS-08, IS-09, and IS-11 handlers are in the `http_handlers` sibling module.
use super::http_handlers;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::super::channel_mapping::ChannelMappingRegistry;
    use super::*;
    use crate::nmos::{
        NmosDevice, NmosDeviceType, NmosFlow, NmosNode, NmosReceiver, NmosSender, NmosSource,
    };

    fn make_test_state() -> ServerState {
        let mut reg = NmosRegistry::new();
        reg.add_node(NmosNode::new("node-1", "Test Node"));
        reg.add_device(NmosDevice::new(
            "dev-1",
            "node-1",
            "Camera",
            NmosDeviceType::Output,
        ));
        reg.add_source(NmosSource::new(
            "src-1",
            "dev-1",
            "Cam Source",
            NmosFormat::Video,
            "clk-0",
        ));
        reg.add_flow(NmosFlow::new(
            "flow-1",
            "src-1",
            "Cam Flow",
            NmosFormat::Video,
            (25, 1),
        ));
        reg.add_sender(NmosSender::new(
            "sender-1",
            "flow-1",
            "Cam Sender",
            NmosTransport::RtpMulticast,
        ));
        reg.add_receiver(NmosReceiver::new(
            "rx-1",
            "dev-1",
            "Monitor A",
            NmosFormat::Video,
        ));

        let mut event_sources = HashMap::new();
        event_sources.insert(
            "evt-1".to_string(),
            Is07Source::new_boolean("evt-1", "Tally Button"),
        );

        let mut cm_reg = ChannelMappingRegistry::new();
        cm_reg.add_device("dev-cm-1", 4);

        let system_cfg = NmosSystemConfig::new("node-1", "Test System");
        let mut compat_reg = CompatibilityRegistry::new();
        compat_reg.register_sender(
            "sender-1".to_string(),
            super::super::compatibility::MediaCapability::video_raw(1920, 1080, (25, 1)),
        );
        compat_reg.register_receiver(
            "rx-1".to_string(),
            super::super::compatibility::MediaCapability::video_raw(1920, 1080, (25, 1)),
        );
        ServerState {
            registry: Arc::new(RwLock::new(reg)),
            connection_manager: Arc::new(RwLock::new(NmosConnectionManager::new())),
            event_bus: Arc::new(RwLock::new(Is07EventBus::new())),
            tally: Arc::new(RwLock::new(TallyController::new())),
            event_sources: Arc::new(RwLock::new(event_sources)),
            sender_staged: Arc::new(RwLock::new(HashMap::new())),
            receiver_staged: Arc::new(RwLock::new(HashMap::new())),
            channel_mapping: Arc::new(RwLock::new(cm_reg)),
            system_api: Arc::new(RwLock::new(NmosSystemApi::new(system_cfg))),
            compatibility: Arc::new(RwLock::new(compat_reg)),
            node_id: "node-1".to_string(),
        }
    }

    // ── json_response helper ────────────────────────────────────────────────

    #[test]
    fn test_json_response_status_200() {
        let resp = json_response(200, r#"{"ok":true}"#.to_string());
        assert_eq!(resp.status().as_u16(), 200);
    }

    #[test]
    fn test_json_response_status_404() {
        let resp = json_response(404, r#"{"error":"not found"}"#.to_string());
        assert_eq!(resp.status().as_u16(), 404);
    }

    #[test]
    fn test_json_response_content_type() {
        let resp = json_response(200, "{}".to_string());
        assert_eq!(
            resp.headers()
                .get("Content-Type")
                .and_then(|v| v.to_str().ok()),
            Some("application/json")
        );
    }

    #[test]
    fn test_json_response_cors_header() {
        let resp = json_response(200, "{}".to_string());
        assert_eq!(
            resp.headers()
                .get("Access-Control-Allow-Origin")
                .and_then(|v| v.to_str().ok()),
            Some("*")
        );
    }

    // ── Route resolution ────────────────────────────────────────────────────

    #[test]
    fn test_route_node_root() {
        assert_eq!(resolve_route("GET", "/x-nmos/node/v1.3"), Route::NodeRoot);
        // Trailing slash should also work.
        assert_eq!(resolve_route("GET", "/x-nmos/node/v1.3/"), Route::NodeRoot);
    }

    #[test]
    fn test_route_node_self() {
        assert_eq!(
            resolve_route("GET", "/x-nmos/node/v1.3/self"),
            Route::NodeSelf
        );
    }

    #[test]
    fn test_route_device_list() {
        assert_eq!(
            resolve_route("GET", "/x-nmos/node/v1.3/devices"),
            Route::DeviceList
        );
    }

    #[test]
    fn test_route_device_by_id() {
        assert_eq!(
            resolve_route("GET", "/x-nmos/node/v1.3/devices/abc-123"),
            Route::DeviceById("abc-123".to_string())
        );
    }

    #[test]
    fn test_route_sender_list() {
        assert_eq!(
            resolve_route("GET", "/x-nmos/node/v1.3/senders"),
            Route::SenderList
        );
    }

    #[test]
    fn test_route_connection_sender_staged_get() {
        assert_eq!(
            resolve_route("GET", "/x-nmos/connection/v1.1/single/senders/s-1/staged"),
            Route::ConnectionSenderStaged("s-1".to_string())
        );
    }

    #[test]
    fn test_route_connection_receiver_staged_patch() {
        assert_eq!(
            resolve_route(
                "PATCH",
                "/x-nmos/connection/v1.1/single/receivers/r-1/staged"
            ),
            Route::ConnectionReceiverStaged("r-1".to_string())
        );
    }

    #[test]
    fn test_route_event_source_state_post() {
        assert_eq!(
            resolve_route("POST", "/x-nmos/events/v1.0/sources/e-1/state"),
            Route::EventSourceState("e-1".to_string())
        );
    }

    #[test]
    fn test_route_not_found() {
        assert_eq!(resolve_route("GET", "/unknown/path"), Route::NotFound);
    }

    #[test]
    fn test_route_method_not_allowed() {
        // POST on a GET-only NMOS path
        assert_eq!(
            resolve_route("POST", "/x-nmos/node/v1.3/self"),
            Route::MethodNotAllowed
        );
    }

    // ── Async handler tests ─────────────────────────────────────────────────

    #[tokio::test]
    async fn test_handle_node_root_response() {
        let resp = handle_node_root();
        assert_eq!(resp.status().as_u16(), 200);
    }

    #[tokio::test]
    async fn test_handle_node_self_found() {
        let state = make_test_state();
        let resp = handle_node_self(&state).await;
        assert_eq!(resp.status().as_u16(), 200);
    }

    #[tokio::test]
    async fn test_handle_node_self_not_found() {
        let mut state = make_test_state();
        state.node_id = "nonexistent-node".to_string();
        let resp = handle_node_self(&state).await;
        assert_eq!(resp.status().as_u16(), 404);
    }

    #[tokio::test]
    async fn test_handle_device_list() {
        let state = make_test_state();
        let resp = handle_device_list(&state).await;
        assert_eq!(resp.status().as_u16(), 200);
    }

    #[tokio::test]
    async fn test_handle_device_by_id_found() {
        let state = make_test_state();
        let resp = handle_device_by_id(&state, "dev-1").await;
        assert_eq!(resp.status().as_u16(), 200);
    }

    #[tokio::test]
    async fn test_handle_device_by_id_not_found() {
        let state = make_test_state();
        let resp = handle_device_by_id(&state, "nonexistent").await;
        assert_eq!(resp.status().as_u16(), 404);
    }

    #[tokio::test]
    async fn test_handle_source_list() {
        let state = make_test_state();
        let resp = handle_source_list(&state).await;
        assert_eq!(resp.status().as_u16(), 200);
    }

    #[tokio::test]
    async fn test_handle_source_by_id_found() {
        let state = make_test_state();
        let resp = handle_source_by_id(&state, "src-1").await;
        assert_eq!(resp.status().as_u16(), 200);
    }

    #[tokio::test]
    async fn test_handle_source_by_id_not_found() {
        let state = make_test_state();
        let resp = handle_source_by_id(&state, "no-such-src").await;
        assert_eq!(resp.status().as_u16(), 404);
    }

    #[tokio::test]
    async fn test_handle_flow_list() {
        let state = make_test_state();
        let resp = handle_flow_list(&state).await;
        assert_eq!(resp.status().as_u16(), 200);
    }

    #[tokio::test]
    async fn test_handle_flow_by_id_found() {
        let state = make_test_state();
        let resp = handle_flow_by_id(&state, "flow-1").await;
        assert_eq!(resp.status().as_u16(), 200);
    }

    #[tokio::test]
    async fn test_handle_flow_by_id_not_found() {
        let state = make_test_state();
        let resp = handle_flow_by_id(&state, "no-flow").await;
        assert_eq!(resp.status().as_u16(), 404);
    }

    #[tokio::test]
    async fn test_handle_sender_list() {
        let state = make_test_state();
        let resp = handle_sender_list(&state).await;
        assert_eq!(resp.status().as_u16(), 200);
    }

    #[tokio::test]
    async fn test_handle_sender_by_id_found() {
        let state = make_test_state();
        let resp = handle_sender_by_id(&state, "sender-1").await;
        assert_eq!(resp.status().as_u16(), 200);
    }

    #[tokio::test]
    async fn test_handle_sender_by_id_not_found() {
        let state = make_test_state();
        let resp = handle_sender_by_id(&state, "no-sender").await;
        assert_eq!(resp.status().as_u16(), 404);
    }

    #[tokio::test]
    async fn test_handle_receiver_list() {
        let state = make_test_state();
        let resp = handle_receiver_list(&state).await;
        assert_eq!(resp.status().as_u16(), 200);
    }

    #[tokio::test]
    async fn test_handle_receiver_by_id_found() {
        let state = make_test_state();
        let resp = handle_receiver_by_id(&state, "rx-1").await;
        assert_eq!(resp.status().as_u16(), 200);
    }

    #[tokio::test]
    async fn test_handle_receiver_by_id_not_found() {
        let state = make_test_state();
        let resp = handle_receiver_by_id(&state, "no-rx").await;
        assert_eq!(resp.status().as_u16(), 404);
    }

    #[tokio::test]
    async fn test_handle_connection_root() {
        let resp = http_is05::handle_connection_root();
        assert_eq!(resp.status().as_u16(), 200);
    }

    #[tokio::test]
    async fn test_handle_connection_sender_list() {
        let state = make_test_state();
        let resp = http_is05::handle_connection_sender_list(&state).await;
        assert_eq!(resp.status().as_u16(), 200);
    }

    #[tokio::test]
    async fn test_handle_connection_sender_root_found() {
        let state = make_test_state();
        let resp = http_is05::handle_connection_sender_root(&state, "sender-1").await;
        assert_eq!(resp.status().as_u16(), 200);
    }

    #[tokio::test]
    async fn test_handle_connection_sender_root_not_found() {
        let state = make_test_state();
        let resp = http_is05::handle_connection_sender_root(&state, "no-sender").await;
        assert_eq!(resp.status().as_u16(), 404);
    }

    #[tokio::test]
    async fn test_handle_get_sender_staged_default() {
        let state = make_test_state();
        let resp = http_is05::handle_get_sender_staged(&state, "sender-1").await;
        assert_eq!(resp.status().as_u16(), 200);
    }

    #[tokio::test]
    async fn test_handle_get_sender_staged_not_found() {
        let state = make_test_state();
        let resp = http_is05::handle_get_sender_staged(&state, "no-sender").await;
        assert_eq!(resp.status().as_u16(), 404);
    }

    #[tokio::test]
    async fn test_handle_get_sender_active() {
        let state = make_test_state();
        let resp = http_is05::handle_get_sender_active(&state, "sender-1").await;
        assert_eq!(resp.status().as_u16(), 200);
    }

    #[tokio::test]
    async fn test_handle_connection_receiver_list() {
        let state = make_test_state();
        let resp = http_is05::handle_connection_receiver_list(&state).await;
        assert_eq!(resp.status().as_u16(), 200);
    }

    #[tokio::test]
    async fn test_handle_connection_receiver_root_found() {
        let state = make_test_state();
        let resp = http_is05::handle_connection_receiver_root(&state, "rx-1").await;
        assert_eq!(resp.status().as_u16(), 200);
    }

    #[tokio::test]
    async fn test_handle_connection_receiver_root_not_found() {
        let state = make_test_state();
        let resp = http_is05::handle_connection_receiver_root(&state, "no-rx").await;
        assert_eq!(resp.status().as_u16(), 404);
    }

    #[tokio::test]
    async fn test_handle_get_receiver_staged_default() {
        let state = make_test_state();
        let resp = http_is05::handle_get_receiver_staged(&state, "rx-1").await;
        assert_eq!(resp.status().as_u16(), 200);
    }

    #[tokio::test]
    async fn test_handle_get_receiver_active() {
        let state = make_test_state();
        let resp = http_is05::handle_get_receiver_active(&state, "rx-1").await;
        assert_eq!(resp.status().as_u16(), 200);
    }

    #[tokio::test]
    async fn test_handle_event_root() {
        let resp = handle_event_root();
        assert_eq!(resp.status().as_u16(), 200);
    }

    #[tokio::test]
    async fn test_handle_event_source_list() {
        let state = make_test_state();
        let resp = handle_event_source_list(&state).await;
        assert_eq!(resp.status().as_u16(), 200);
    }

    #[tokio::test]
    async fn test_handle_event_source_by_id_found() {
        let state = make_test_state();
        let resp = handle_event_source_by_id(&state, "evt-1").await;
        assert_eq!(resp.status().as_u16(), 200);
    }

    #[tokio::test]
    async fn test_handle_event_source_by_id_not_found() {
        let state = make_test_state();
        let resp = handle_event_source_by_id(&state, "no-evt").await;
        assert_eq!(resp.status().as_u16(), 404);
    }

    // ── Staged params struct tests ──────────────────────────────────────────

    #[test]
    fn test_staged_sender_params_default() {
        let p = StagedSenderParams::default();
        assert!(!p.master_enable);
        assert!(p.receiver_id.is_none());
        assert!(p.destination_ip.is_none());
        assert!(p.destination_port.is_none());
        assert!(p.source_ip.is_none());
    }

    #[test]
    fn test_staged_receiver_params_default() {
        let p = StagedReceiverParams::default();
        assert!(!p.master_enable);
        assert!(p.sender_id.is_none());
        assert!(p.interface_ip.is_none());
        assert!(p.multicast_ip.is_none());
        assert!(p.source_port.is_none());
    }

    #[test]
    fn test_staged_sender_params_to_json() {
        let p = StagedSenderParams {
            master_enable: true,
            receiver_id: Some("r-1".to_string()),
            destination_ip: Some("239.0.0.1".to_string()),
            destination_port: Some(5004),
            source_ip: Some("10.0.0.1".to_string()),
        };
        let v = p.to_json();
        assert_eq!(v["master_enable"], true);
        assert_eq!(v["receiver_id"], "r-1");
    }

    #[test]
    fn test_staged_receiver_params_to_json() {
        let p = StagedReceiverParams {
            master_enable: true,
            sender_id: Some("s-1".to_string()),
            interface_ip: Some("10.0.0.2".to_string()),
            multicast_ip: Some("239.0.0.1".to_string()),
            source_port: Some(5004),
        };
        let v = p.to_json();
        assert_eq!(v["master_enable"], true);
        assert_eq!(v["sender_id"], "s-1");
    }

    // ── NmosHttpError tests ─────────────────────────────────────────────────

    #[test]
    fn test_error_not_found_status() {
        let e = NmosHttpError::NotFound("test".to_string());
        assert_eq!(e.status_code().as_u16(), 404);
    }

    #[test]
    fn test_error_method_not_allowed_status() {
        let e = NmosHttpError::MethodNotAllowed;
        assert_eq!(e.status_code().as_u16(), 405);
    }

    #[test]
    fn test_error_bad_request_status() {
        let e = NmosHttpError::BadRequest("bad json".to_string());
        assert_eq!(e.status_code().as_u16(), 400);
    }

    #[test]
    fn test_error_internal_status() {
        let e = NmosHttpError::Internal("whoops".to_string());
        assert_eq!(e.status_code().as_u16(), 500);
    }

    #[test]
    fn test_error_to_json_body_has_code() {
        let e = NmosHttpError::NotFound("x".to_string());
        let body = e.to_json_body();
        let v: Value = serde_json::from_str(&body).expect("valid json");
        assert_eq!(v["code"], 404);
    }

    // ── NmosHttpServer construction ─────────────────────────────────────────

    #[test]
    fn test_nmos_http_server_new() {
        let registry = Arc::new(RwLock::new(NmosRegistry::new()));
        let cm = Arc::new(RwLock::new(NmosConnectionManager::new()));
        let eb = Arc::new(RwLock::new(Is07EventBus::new()));
        let tc = Arc::new(RwLock::new(TallyController::new()));
        let addr: SocketAddr = "127.0.0.1:0".parse().expect("valid addr");
        let server = NmosHttpServer::new(registry, cm, eb, tc, addr, "node-test");
        assert_eq!(server.node_id, "node-test");
    }

    // ── Format / transport serialization ───────────────────────────────────

    #[test]
    fn test_format_urn_video() {
        assert_eq!(format_urn(&NmosFormat::Video), "urn:x-nmos:format:video");
    }

    #[test]
    fn test_format_urn_audio() {
        assert_eq!(format_urn(&NmosFormat::Audio), "urn:x-nmos:format:audio");
    }

    #[test]
    fn test_format_urn_data() {
        assert_eq!(format_urn(&NmosFormat::Data), "urn:x-nmos:format:data");
    }

    #[test]
    fn test_format_urn_mux() {
        assert_eq!(format_urn(&NmosFormat::Mux), "urn:x-nmos:format:mux");
    }

    #[test]
    fn test_transport_urn_rtp_multicast() {
        assert_eq!(
            transport_urn(&NmosTransport::RtpMulticast),
            "urn:x-nmos:transport:rtp.mcast"
        );
    }

    #[test]
    fn test_transport_urn_rtp_unicast() {
        assert_eq!(
            transport_urn(&NmosTransport::RtpUnicast),
            "urn:x-nmos:transport:rtp.ucast"
        );
    }

    #[test]
    fn test_transport_urn_srt() {
        assert_eq!(
            transport_urn(&NmosTransport::Srt),
            "urn:x-nmos:transport:srt"
        );
    }

    #[test]
    fn test_event_type_str_boolean() {
        assert_eq!(event_type_str(&Is07EventType::Boolean), "boolean");
    }

    #[test]
    fn test_event_type_str_number() {
        assert_eq!(event_type_str(&Is07EventType::Number), "number");
    }

    #[test]
    fn test_event_type_str_enum() {
        assert_eq!(event_type_str(&Is07EventType::Enum), "string/enum");
    }

    // ── build_is07_state ───────────────────────────────────────────────────

    #[test]
    fn test_build_is07_state_boolean_true() {
        let mut src = Is07Source::new_boolean("s", "S");
        src.bool_state = Some(true);
        let v = build_is07_state(&src);
        assert_eq!(v["value"], true);
    }

    #[test]
    fn test_build_is07_state_number() {
        let mut src = Is07Source::new_number("s", "S");
        src.number_state = Some(42.0);
        let v = build_is07_state(&src);
        assert!((v["value"].as_f64().unwrap_or(0.0) - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_build_is07_state_string() {
        let mut src = Is07Source::new_string("s", "S");
        src.string_state = Some("hello".to_string());
        let v = build_is07_state(&src);
        assert_eq!(v["value"], "hello");
    }
}

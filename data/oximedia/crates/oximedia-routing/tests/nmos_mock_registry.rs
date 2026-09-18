//! Integration tests for NMOS IS-04 Query API + IS-05 Connection API
//! using an in-process mock HTTP server.
//!
//! The mock server listens on an ephemeral port (127.0.0.1:0), so tests can
//! run in parallel without port conflicts.  The HTTP client is built directly
//! on hyper 1.x + tokio (plain TCP, no TLS) to avoid the need for a
//! rustls crypto provider.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::client::conn::http1 as client_http1;
use hyper::server::conn::http1 as server_http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode, Uri};
use hyper_util::rt::TokioIo;
use serde_json::Value;
use tokio::net::{TcpListener, TcpStream};

// ============================================================================
// Mock server state
// ============================================================================

#[derive(Clone, Default)]
struct MockState {
    senders: Arc<Mutex<Vec<Value>>>,
    receivers: Arc<Mutex<Vec<Value>>>,
    /// Staged IS-05 sender params: sender_id → staged body.
    staged: Arc<Mutex<HashMap<String, Value>>>,
}

impl MockState {
    fn new() -> Self {
        Self::default()
    }
}

// ============================================================================
// HTTP router (server side)
// ============================================================================

fn json_response(status: StatusCode, body: &Value) -> Response<Full<Bytes>> {
    Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Full::new(Bytes::from(body.to_string())))
        .expect("response builder is infallible for these parameters")
}

async fn handle_request(
    req: Request<hyper::body::Incoming>,
    state: Arc<MockState>,
) -> Result<Response<Full<Bytes>>, std::convert::Infallible> {
    let method = req.method().clone();
    let path = req.uri().path().to_owned();

    let path_trimmed = path.trim_end_matches('/');
    let parts: Vec<&str> = path_trimmed
        .trim_start_matches('/')
        .split('/')
        .collect::<Vec<_>>();

    let resp = match (method.clone(), parts.as_slice()) {
        // ── IS-04 Query API ───────────────────────────────────────────────────
        (Method::GET, ["x-nmos", "query", "v1.3", "senders"]) => {
            let list = state
                .senders
                .lock()
                .expect("mutex should not be poisoned")
                .clone();
            json_response(StatusCode::OK, &Value::Array(list))
        }

        (Method::GET, ["x-nmos", "query", "v1.3", "receivers"]) => {
            let list = state
                .receivers
                .lock()
                .expect("mutex should not be poisoned")
                .clone();
            json_response(StatusCode::OK, &Value::Array(list))
        }

        // ── IS-04 Registration API ────────────────────────────────────────────
        (Method::POST, ["x-nmos", "registration", "v1.3", "resource"]) => {
            let body_bytes = match req.collect().await {
                Ok(b) => b.to_bytes(),
                Err(_) => {
                    return Ok(json_response(
                        StatusCode::BAD_REQUEST,
                        &serde_json::json!({"error": "failed to read body"}),
                    ))
                }
            };

            let parsed: Value = match serde_json::from_slice(&body_bytes) {
                Ok(v) => v,
                Err(_) => {
                    return Ok(json_response(
                        StatusCode::BAD_REQUEST,
                        &serde_json::json!({"error": "invalid JSON"}),
                    ))
                }
            };

            let resource_type = parsed
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned();

            let data = parsed.get("data").cloned().unwrap_or(Value::Null);

            match resource_type.as_str() {
                "sender" => {
                    state
                        .senders
                        .lock()
                        .expect("mutex should not be poisoned")
                        .push(data);
                }
                "receiver" => {
                    state
                        .receivers
                        .lock()
                        .expect("mutex should not be poisoned")
                        .push(data);
                }
                _ => {
                    return Ok(json_response(
                        StatusCode::BAD_REQUEST,
                        &serde_json::json!({"error": "unknown resource type"}),
                    ))
                }
            }

            json_response(
                StatusCode::CREATED,
                &serde_json::json!({"status": "created"}),
            )
        }

        // ── IS-05 Connection API ──────────────────────────────────────────────
        (
            Method::POST,
            ["x-nmos", "connection", "v1.0", "single", "senders", sender_id, "staged"],
        ) => {
            let sid = sender_id.to_string();

            let body_bytes = match req.collect().await {
                Ok(b) => b.to_bytes(),
                Err(_) => {
                    return Ok(json_response(
                        StatusCode::BAD_REQUEST,
                        &serde_json::json!({"error": "failed to read body"}),
                    ))
                }
            };

            let parsed: Value =
                serde_json::from_slice(&body_bytes).unwrap_or(serde_json::json!({}));

            // Echo back the staged params, adding the sender_id field.
            let mut echo = parsed.clone();
            if let Value::Object(ref mut map) = echo {
                map.insert("sender_id".to_owned(), Value::String(sid.clone()));
            }

            state
                .staged
                .lock()
                .expect("mutex should not be poisoned")
                .insert(sid, echo.clone());

            json_response(StatusCode::OK, &echo)
        }

        // ── Fallthrough → 404 ─────────────────────────────────────────────────
        _ => json_response(
            StatusCode::NOT_FOUND,
            &serde_json::json!({"error": "not found", "path": path}),
        ),
    };

    Ok(resp)
}

// ============================================================================
// Mock server launcher
// ============================================================================

struct MockNmosRegistry {
    state: Arc<MockState>,
}

impl MockNmosRegistry {
    fn new() -> Self {
        Self {
            state: Arc::new(MockState::new()),
        }
    }

    /// Bind to an ephemeral port and start serving.
    ///
    /// Returns `(bound_addr, join_handle)`.  The join handle is intentionally
    /// not awaited in the test — the server runs until the test exits.
    async fn start(self) -> (SocketAddr, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("binding to ephemeral port succeeds in tests");
        let addr = listener
            .local_addr()
            .expect("local_addr succeeds after successful bind");

        let state = self.state.clone();

        let handle = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };

                let io = TokioIo::new(stream);
                let state_clone = state.clone();

                tokio::spawn(async move {
                    let svc = service_fn(move |req| {
                        let sc = state_clone.clone();
                        handle_request(req, sc)
                    });
                    let _ = server_http1::Builder::new().serve_connection(io, svc).await;
                });
            }
        });

        (addr, handle)
    }
}

// ============================================================================
// HTTP/1.1 helper — plain TCP, no TLS
// ============================================================================

/// Send a request using hyper's low-level HTTP/1.1 client over plain TCP.
/// Returns `(status_code, body_bytes)`.
async fn send(addr: SocketAddr, req: Request<Full<Bytes>>) -> (u16, Bytes) {
    let stream = TcpStream::connect(addr)
        .await
        .expect("TCP connect to mock server succeeds");
    let io = TokioIo::new(stream);

    let (mut sender, conn) = client_http1::handshake(io)
        .await
        .expect("HTTP/1.1 handshake succeeds");

    // Drive the connection in a background task.
    tokio::spawn(async move {
        let _ = conn.await;
    });

    let resp = sender
        .send_request(req)
        .await
        .expect("send_request succeeds");

    let status = resp.status().as_u16();
    let body = resp
        .collect()
        .await
        .expect("response body collection succeeds")
        .to_bytes();

    (status, body)
}

/// Convenience: build a GET request.
fn get(addr: SocketAddr, path: &str) -> Request<Full<Bytes>> {
    let uri: Uri = format!("http://{addr}{path}")
        .parse()
        .expect("URI should parse correctly");
    Request::builder()
        .method(Method::GET)
        .uri(uri)
        .header("Host", addr.to_string())
        .body(Full::new(Bytes::new()))
        .expect("GET request builds correctly")
}

/// Convenience: build a POST request with a JSON body.
fn post_json(addr: SocketAddr, path: &str, body: &Value) -> Request<Full<Bytes>> {
    let uri: Uri = format!("http://{addr}{path}")
        .parse()
        .expect("URI should parse correctly");
    let body_bytes = Bytes::from(body.to_string());
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("Host", addr.to_string())
        .header("Content-Type", "application/json")
        .header("Content-Length", body_bytes.len().to_string())
        .body(Full::new(body_bytes))
        .expect("POST request builds correctly")
}

// ============================================================================
// Integration tests
// ============================================================================

/// Register a sender via IS-04 Registration API, then query it via IS-04
/// Query API and assert it appears in the list.
#[tokio::test]
async fn mock_registers_sender_then_queries() {
    let registry = MockNmosRegistry::new();
    let (addr, _handle) = registry.start().await;

    // POST a sender registration.
    let sender_body = serde_json::json!({
        "type": "sender",
        "data": {
            "id": "test-sender-1",
            "label": "Test Sender",
            "format": "urn:x-nmos:format:video"
        }
    });

    let (status, _) = send(
        addr,
        post_json(addr, "/x-nmos/registration/v1.3/resource", &sender_body),
    )
    .await;
    assert_eq!(status, 201, "registration should return 201 Created");

    // GET senders — the registered sender should appear.
    let (status, body) = send(addr, get(addr, "/x-nmos/query/v1.3/senders")).await;
    assert_eq!(status, 200, "sender list should return 200 OK");

    let senders: Vec<Value> =
        serde_json::from_slice(&body).expect("response body should be a JSON array");

    assert!(
        !senders.is_empty(),
        "senders list should contain the registered sender"
    );

    let ids: Vec<&str> = senders
        .iter()
        .filter_map(|s| s.get("id").and_then(Value::as_str))
        .collect();

    assert!(
        ids.contains(&"test-sender-1"),
        "registered sender id should appear in the query response; got {ids:?}"
    );
}

/// Register a receiver via IS-04, then perform an IS-05 staged connection
/// for a sender.  Both operations should succeed with correct status codes.
#[tokio::test]
async fn mock_registers_receiver_then_subscribes() {
    let registry = MockNmosRegistry::new();
    let (addr, _handle) = registry.start().await;

    // Register a receiver.
    let receiver_body = serde_json::json!({
        "type": "receiver",
        "data": {
            "id": "test-receiver-1",
            "label": "Test Receiver",
            "format": "urn:x-nmos:format:video"
        }
    });

    let (status, _) = send(
        addr,
        post_json(addr, "/x-nmos/registration/v1.3/resource", &receiver_body),
    )
    .await;
    assert_eq!(status, 201, "receiver registration → 201 Created");

    // Confirm it appears in the receiver query list.
    let (status, body) = send(addr, get(addr, "/x-nmos/query/v1.3/receivers")).await;
    assert_eq!(status, 200, "receiver list → 200 OK");

    let receivers: Vec<Value> = serde_json::from_slice(&body).expect("body should be a JSON array");
    let ids: Vec<&str> = receivers
        .iter()
        .filter_map(|r| r.get("id").and_then(Value::as_str))
        .collect();
    assert!(
        ids.contains(&"test-receiver-1"),
        "registered receiver should appear in list; got {ids:?}"
    );

    // IS-05 — stage a connection on a sender.
    let staged_body = serde_json::json!({
        "master_enable": true,
        "receiver_id": "test-receiver-1",
        "transport_params": [{ "destination_ip": "239.0.0.1", "rtp_enabled": true }]
    });

    let (status, body) = send(
        addr,
        post_json(
            addr,
            "/x-nmos/connection/v1.0/single/senders/sender-xyz/staged",
            &staged_body,
        ),
    )
    .await;
    assert_eq!(status, 200, "IS-05 staged connection should return 200 OK");

    let echo: Value = serde_json::from_slice(&body).expect("response should be JSON");
    assert_eq!(
        echo.get("sender_id").and_then(Value::as_str),
        Some("sender-xyz"),
        "echo body should include the sender_id"
    );
    assert_eq!(
        echo.get("receiver_id").and_then(Value::as_str),
        Some("test-receiver-1"),
        "echo body should preserve receiver_id from the staged params"
    );
}

/// Requesting an unknown path should return 404.
#[tokio::test]
async fn mock_handles_404_for_unknown_resource() {
    let registry = MockNmosRegistry::new();
    let (addr, _handle) = registry.start().await;

    let (status, body) = send(addr, get(addr, "/nonexistent/path")).await;
    assert_eq!(status, 404, "unknown path should return 404 Not Found");

    let parsed: Value = serde_json::from_slice(&body).expect("404 body should be JSON");
    assert!(
        parsed.get("error").is_some(),
        "404 body should contain an 'error' field"
    );
}

/// Eight concurrent clients all querying senders simultaneously should all
/// receive 200 OK responses — validates that the shared `Mutex` state is not
/// deadlocked under contention.
#[tokio::test]
async fn mock_handles_concurrent_clients() {
    let registry = MockNmosRegistry::new();
    let (addr, _handle) = registry.start().await;

    // Pre-register one sender so the list is non-trivially populated.
    let pre_body = serde_json::json!({
        "type": "sender",
        "data": { "id": "concurrent-sender", "label": "Concurrent Sender" }
    });
    let (status, _) = send(
        addr,
        post_json(addr, "/x-nmos/registration/v1.3/resource", &pre_body),
    )
    .await;
    assert_eq!(status, 201, "pre-registration should succeed");

    let mut handles = Vec::with_capacity(8);
    for _ in 0..8 {
        handles.push(tokio::spawn(async move {
            let (status, body) = send(addr, get(addr, "/x-nmos/query/v1.3/senders")).await;
            assert_eq!(status, 200, "concurrent sender query → 200 OK");
            let list: Vec<Value> =
                serde_json::from_slice(&body).expect("concurrent response should be JSON array");
            assert!(
                !list.is_empty(),
                "each concurrent client should see the pre-registered sender"
            );
        }));
    }

    for h in handles {
        h.await.expect("concurrent task should not panic");
    }
}

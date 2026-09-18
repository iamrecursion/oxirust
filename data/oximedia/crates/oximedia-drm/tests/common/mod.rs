//! Shared mock-server scaffolding for PlayReady and FairPlay integration tests.
//!
//! This module is `#[path]`-included from each integration test binary so that
//! the same TCP mock server logic is reused without duplication.

#![allow(dead_code)]

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::server::conn::http1 as server_http1;
use hyper::service::service_fn;
use hyper::{Response, StatusCode};
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;

// ─────────────────────────────────────────────────────────────────────────────
// MockBehavior — what the server does for the next incoming request
// ─────────────────────────────────────────────────────────────────────────────

/// Behaviour the mock server should adopt for the next incoming POST.
#[derive(Clone)]
pub enum MockBehavior {
    /// Return `200 OK` with the supplied body bytes.
    Ok(Vec<u8>),
    /// Return the supplied non-2xx HTTP status with a textual body.
    Status { status: u16, body: String },
    /// Hold the connection open without responding for `delay_ms` milliseconds
    /// before finally returning 200 with an empty body. Used to trigger
    /// client-side timeouts.
    Stall { delay_ms: u64 },
}

// ─────────────────────────────────────────────────────────────────────────────
// CapturedRequest / MockState
// ─────────────────────────────────────────────────────────────────────────────

/// A single request captured by the mock server.
#[derive(Clone, Default)]
pub struct CapturedRequest {
    pub body: Vec<u8>,
    pub path: String,
    pub content_type: Option<String>,
    pub soap_action: Option<String>,
}

/// Shared state between the spawned mock server and the test body.
#[derive(Clone, Default)]
pub struct MockState {
    pub captured: Arc<Mutex<Vec<CapturedRequest>>>,
}

impl MockState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the most recently captured request (or a default if none).
    pub fn last_captured(&self) -> CapturedRequest {
        self.captured
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .last()
            .cloned()
            .unwrap_or_default()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// spawn_mock_server
// ─────────────────────────────────────────────────────────────────────────────

/// Bind a one-shot mock server on `127.0.0.1:0` and return its address.
///
/// The server handles **exactly one** incoming request according to
/// `behavior` and then exits.
pub async fn spawn_mock_server(behavior: MockBehavior, state: MockState) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local_addr after bind");

    tokio::spawn(async move {
        if let Ok((stream, _)) = listener.accept().await {
            let io = TokioIo::new(stream);
            let state_clone = state.clone();
            let behavior_clone = behavior.clone();

            let svc = service_fn(move |req| {
                let state_inner = state_clone.clone();
                let behavior_inner = behavior_clone.clone();
                async move {
                    let path = req.uri().path().to_owned();
                    let content_type = req
                        .headers()
                        .get("Content-Type")
                        .and_then(|v| v.to_str().ok())
                        .map(|s| s.to_owned());
                    let soap_action = req
                        .headers()
                        .get("SOAPAction")
                        .and_then(|v| v.to_str().ok())
                        .map(|s| s.to_owned());

                    let body_bytes = req
                        .collect()
                        .await
                        .map(|b| b.to_bytes().to_vec())
                        .unwrap_or_default();

                    state_inner
                        .captured
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push(CapturedRequest {
                            body: body_bytes,
                            path,
                            content_type,
                            soap_action,
                        });

                    let response: Response<Full<Bytes>> = match behavior_inner {
                        MockBehavior::Ok(payload) => Response::builder()
                            .status(StatusCode::OK)
                            .header("Content-Type", "application/json")
                            .body(Full::new(Bytes::from(payload)))
                            .expect("response builds"),
                        MockBehavior::Status { status, body } => {
                            let code = StatusCode::from_u16(status)
                                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                            Response::builder()
                                .status(code)
                                .body(Full::new(Bytes::from(body)))
                                .expect("response builds")
                        }
                        MockBehavior::Stall { delay_ms } => {
                            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                            Response::builder()
                                .status(StatusCode::OK)
                                .body(Full::new(Bytes::new()))
                                .expect("response builds")
                        }
                    };

                    Ok::<_, std::convert::Infallible>(response)
                }
            });

            let _ = server_http1::Builder::new().serve_connection(io, svc).await;
        }
    });

    addr
}

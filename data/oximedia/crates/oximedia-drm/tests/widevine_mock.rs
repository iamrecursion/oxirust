//! End-to-end integration tests for the Widevine license RPC layer.
//!
//! These tests drive a plain-TCP `hyper` mock server bound to an ephemeral
//! port and pair it with the `HyperPlainLicenseClient` transport so we can
//! verify:
//!
//! 1. `WidevineCdm::acquire_license` performs the full request → response →
//!    key-registration flow against a real (in-process) HTTP server.
//! 2. Non-2xx HTTP responses bubble up as `DrmError::LicenseDenied`.
//! 3. Malformed response bodies surface as JSON parse errors.
//! 4. The bytes sent on the wire match `WidevineLicenseRequest::to_bytes`.
//! 5. Per-phase timeouts are honoured when the server stalls.
//!
//! No TLS is involved — the production TLS client (`HyperRustlsLicenseClient`)
//! is exercised separately via the unit-level URL parser tests in
//! `widevine_rpc.rs`.

#![cfg(feature = "widevine")]

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

use oximedia_drm::widevine::{
    LicenseType, WidevineCdm, WidevineKey, WidevineLicenseRequest, WidevineLicenseResponse,
};
use oximedia_drm::{DrmError, HyperPlainLicenseClient};

// ============================================================================
// Mock-server scaffolding
// ============================================================================

/// Behaviour the mock server should adopt for the next incoming POST.
#[derive(Clone)]
enum MockBehavior {
    /// Return `200 OK` with the supplied body bytes.
    Ok(Vec<u8>),
    /// Return the supplied non-2xx HTTP status with a textual body.
    Status { status: u16, body: String },
    /// Hold the connection open without responding for `delay_ms` milliseconds
    /// before finally returning 200 with empty body. Used to trigger
    /// client-side timeouts.
    Stall { delay_ms: u64 },
}

#[derive(Clone, Default)]
struct CapturedRequest {
    body: Vec<u8>,
    path: String,
    content_type: Option<String>,
}

#[derive(Clone, Default)]
struct MockState {
    captured: Arc<Mutex<Vec<CapturedRequest>>>,
}

impl MockState {
    fn new() -> Self {
        Self::default()
    }

    fn last_captured(&self) -> CapturedRequest {
        self.captured
            .lock()
            .expect("mutex should not be poisoned")
            .last()
            .cloned()
            .unwrap_or_default()
    }
}

/// Bind a one-shot mock server on `127.0.0.1:0` and return its address.
///
/// The server consumes a single request, applies `behavior`, and exits.
async fn spawn_mock_server(behavior: MockBehavior, state: MockState) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local_addr after bind");

    tokio::spawn(async move {
        // Accept a single connection.
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

                    let body_bytes = match req.collect().await {
                        Ok(b) => b.to_bytes().to_vec(),
                        Err(_) => Vec::new(),
                    };

                    state_inner
                        .captured
                        .lock()
                        .expect("mutex should not be poisoned")
                        .push(CapturedRequest {
                            body: body_bytes,
                            path,
                            content_type,
                        });

                    let response: Response<Full<Bytes>> = match behavior_inner {
                        MockBehavior::Ok(payload) => Response::builder()
                            .status(StatusCode::OK)
                            .header("Content-Type", "application/octet-stream")
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

// ============================================================================
// Helpers — sample request / response payloads
// ============================================================================

fn sample_response_bytes() -> Vec<u8> {
    let key = WidevineKey::new(vec![0x01; 16], vec![0xAB; 16]);
    let resp = WidevineLicenseResponse::new(vec![key])
        .with_license_duration(86_400)
        .with_playback_duration(7_200);
    resp.to_bytes().expect("serialise response")
}

fn build_url(addr: SocketAddr) -> String {
    format!("http://{addr}/widevine/license")
}

// ============================================================================
// Tests
// ============================================================================

#[tokio::test]
async fn test_widevine_full_license_flow_via_mock_server() {
    let state = MockState::new();
    let server_body = sample_response_bytes();
    let addr = spawn_mock_server(MockBehavior::Ok(server_body.clone()), state.clone()).await;

    let client = HyperPlainLicenseClient::new().with_timeout_ms(5_000);
    let mut cdm = WidevineCdm::new(b"client-id-bytes".to_vec());

    let session_id = vec![0xDE, 0xAD, 0xBE, 0xEF];
    let content_id = b"content-1".to_vec();
    let key_id = vec![0x01u8; 16];

    let keys = cdm
        .acquire_license(
            session_id.clone(),
            LicenseType::Streaming,
            content_id,
            vec![key_id.clone()],
            &build_url(addr),
            &client,
            &[],
        )
        .await
        .expect("license acquisition should succeed");

    assert_eq!(keys.len(), 1, "exactly one key should have been delivered");
    assert_eq!(
        keys.get(&key_id),
        Some(&vec![0xAB; 16]),
        "the delivered key bytes should match the mock server payload"
    );
    assert_eq!(
        cdm.get_key(&session_id, &key_id),
        Some(vec![0xAB; 16]),
        "CDM should store the key under the session id"
    );

    // The server should have observed exactly one POST with octet-stream type.
    let captured = state.last_captured();
    assert_eq!(captured.path, "/widevine/license");
    assert_eq!(
        captured.content_type.as_deref(),
        Some("application/octet-stream"),
        "client must POST with Content-Type: application/octet-stream"
    );
    assert!(
        !captured.body.is_empty(),
        "server must have received a non-empty request body"
    );
}

#[tokio::test]
async fn test_widevine_license_denied_403() {
    let state = MockState::new();
    let addr = spawn_mock_server(
        MockBehavior::Status {
            status: 403,
            body: "device revoked".to_string(),
        },
        state.clone(),
    )
    .await;

    let client = HyperPlainLicenseClient::new().with_timeout_ms(5_000);
    let mut cdm = WidevineCdm::new(b"client-id".to_vec());

    let err = cdm
        .acquire_license(
            vec![0xAA, 0xBB],
            LicenseType::Streaming,
            b"content-x".to_vec(),
            vec![vec![0x02; 16]],
            &build_url(addr),
            &client,
            &[],
        )
        .await
        .expect_err("acquire_license must fail on HTTP 403");

    match err {
        DrmError::LicenseDenied { status, body } => {
            assert_eq!(status, 403, "denial status should be 403");
            assert!(
                body.contains("device revoked"),
                "denial body should propagate server message, got: {body:?}"
            );
        }
        other => panic!("expected DrmError::LicenseDenied, got {other:?}"),
    }
}

#[tokio::test]
async fn test_widevine_malformed_response() {
    let state = MockState::new();
    let garbage = b"not a valid widevine response at all".to_vec();
    let addr = spawn_mock_server(MockBehavior::Ok(garbage), state.clone()).await;

    let client = HyperPlainLicenseClient::new().with_timeout_ms(5_000);
    let mut cdm = WidevineCdm::new(b"client-id".to_vec());

    let err = cdm
        .acquire_license(
            vec![0xCA, 0xFE],
            LicenseType::Streaming,
            b"content-y".to_vec(),
            vec![vec![0x03; 16]],
            &build_url(addr),
            &client,
            &[],
        )
        .await
        .expect_err("acquire_license must reject garbage body");

    match err {
        DrmError::JsonError(_) => {}
        other => panic!("expected DrmError::JsonError, got {other:?}"),
    }
}

#[tokio::test]
async fn test_widevine_request_body_round_trip() {
    let state = MockState::new();
    let server_body = sample_response_bytes();
    let addr = spawn_mock_server(MockBehavior::Ok(server_body), state.clone()).await;

    let client = HyperPlainLicenseClient::new().with_timeout_ms(5_000);
    let mut cdm = WidevineCdm::new(b"client-fingerprint".to_vec());

    let session_id = vec![0x11, 0x22, 0x33];
    let content_id = b"super-content".to_vec();
    let key_id = vec![0x01u8; 16];

    cdm.acquire_license(
        session_id.clone(),
        LicenseType::Offline,
        content_id.clone(),
        vec![key_id.clone()],
        &build_url(addr),
        &client,
        &[],
    )
    .await
    .expect("license acquisition should succeed");

    let captured = state.last_captured();
    let parsed = WidevineLicenseRequest::from_bytes(&captured.body)
        .expect("captured body must round-trip through WidevineLicenseRequest");

    assert_eq!(parsed.license_type, LicenseType::Offline);
    assert_eq!(parsed.content_id, content_id);
    assert_eq!(parsed.key_ids, vec![key_id]);
    assert_eq!(
        parsed.session_id,
        Some(session_id),
        "session_id must be attached to the wire request"
    );
    assert_eq!(
        parsed.client_id,
        Some(b"client-fingerprint".to_vec()),
        "CDM client_id must be attached to the wire request"
    );
}

#[tokio::test]
async fn test_widevine_timeout_respected() {
    let state = MockState::new();
    let addr = spawn_mock_server(MockBehavior::Stall { delay_ms: 2_000 }, state.clone()).await;

    let client = HyperPlainLicenseClient::new().with_timeout_ms(150);
    let mut cdm = WidevineCdm::new(b"client".to_vec());

    let started = std::time::Instant::now();
    let err = cdm
        .acquire_license(
            vec![0xFE, 0xED],
            LicenseType::Streaming,
            b"content".to_vec(),
            vec![vec![0x04; 16]],
            &build_url(addr),
            &client,
            &[],
        )
        .await
        .expect_err("stalled server must trigger a client-side timeout");
    let elapsed = started.elapsed();

    match err {
        DrmError::NetworkError(msg) => {
            assert!(
                msg.contains("timed out"),
                "timeout error must mention 'timed out', got: {msg}"
            );
        }
        other => panic!("expected DrmError::NetworkError, got {other:?}"),
    }

    assert!(
        elapsed < Duration::from_millis(1_500),
        "client must give up well before server's 2s stall (took {elapsed:?})"
    );
}

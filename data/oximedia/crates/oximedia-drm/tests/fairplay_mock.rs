//! End-to-end integration tests for the FairPlay KSM transport layer.
//!
//! These tests drive a plain-TCP `hyper` mock server bound to an ephemeral
//! port and pair it with the `HyperPlainFairPlayClient` transport to verify:
//!
//! 1. `FairPlayClient::request_key_from_server` performs the full JSON POST →
//!    CKC decode → cache flow against a real (in-process) HTTP server.
//! 2. Non-2xx HTTP responses bubble up as `DrmError::LicenseDenied`.
//! 3. A missing `"ckc"` field in the response surfaces as `DrmError::LicenseError`.
//! 4. The JSON request body is correctly encoded (Base64 SPC, asset_id field).
//! 5. Per-phase timeouts are honoured when the server stalls.

#![cfg(feature = "fairplay")]

#[path = "common/mod.rs"]
mod common;

use std::net::SocketAddr;
use std::time::Duration;

use base64::engine::general_purpose::STANDARD;
use base64::Engine;

use common::{spawn_mock_server, MockBehavior, MockState};

use oximedia_drm::fairplay::FairPlayClient;
use oximedia_drm::fairplay_rpc::{
    build_ksm_json, parse_ckc_response, FairPlayClientExt, HyperPlainFairPlayClient,
};
use oximedia_drm::DrmError;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn build_url(addr: SocketAddr) -> String {
    format!("http://{addr}/fps/key")
}

/// Build a valid JSON CKC response body.
fn sample_ckc_response(ckc_bytes: &[u8]) -> Vec<u8> {
    let b64 = STANDARD.encode(ckc_bytes);
    serde_json::to_vec(&serde_json::json!({ "ckc": b64 })).expect("json")
}

// ─────────────────────────────────────────────────────────────────────────────
// Integration tests
// ─────────────────────────────────────────────────────────────────────────────

/// Full KSM flow: SPC → JSON POST → CKC decode → cache.
#[tokio::test]
async fn test_fairplay_full_key_flow_via_mock_server() {
    let ckc_bytes = vec![0xCA, 0xFE, 0xBA, 0xBE, 0x01, 0x02, 0x03, 0x04];
    let response_body = sample_ckc_response(&ckc_bytes);

    let state = MockState::new();
    let addr = spawn_mock_server(MockBehavior::Ok(response_body), state.clone()).await;

    let transport = HyperPlainFairPlayClient::new().with_timeout_ms(5_000);
    let mut fp_client = FairPlayClient::new(b"test-certificate".to_vec());

    let asset_id = "asset-integration-001".to_string();

    let returned_ckc = fp_client
        .request_key_from_server(&build_url(addr), asset_id.clone(), &transport, &[])
        .await
        .expect("request_key_from_server must succeed");

    assert_eq!(
        returned_ckc, ckc_bytes,
        "returned CKC must match mock payload"
    );

    // CKC must be cached
    assert!(
        fp_client.get_ckc(&asset_id).is_some(),
        "CKC must be cached after successful KSM round-trip"
    );

    // Verify the request the server observed
    let captured = state.last_captured();
    assert_eq!(captured.path, "/fps/key");
    let ct = captured.content_type.as_deref().unwrap_or("");
    assert!(
        ct.starts_with("application/json"),
        "Content-Type must be application/json, got: {ct}"
    );
    assert!(
        !captured.body.is_empty(),
        "server must receive a non-empty JSON body"
    );
}

/// Non-2xx response maps to `DrmError::LicenseDenied`.
#[tokio::test]
async fn test_fairplay_key_denied_403() {
    let state = MockState::new();
    let addr = spawn_mock_server(
        MockBehavior::Status {
            status: 403,
            body: "asset not licensed".to_string(),
        },
        state.clone(),
    )
    .await;

    let transport = HyperPlainFairPlayClient::new().with_timeout_ms(5_000);
    let mut fp_client = FairPlayClient::new(b"cert".to_vec());

    let err = fp_client
        .request_key_from_server(&build_url(addr), "asset-x".to_string(), &transport, &[])
        .await
        .expect_err("must fail on HTTP 403");

    match err {
        DrmError::LicenseDenied { status, body } => {
            assert_eq!(status, 403, "denial status must be 403");
            assert!(
                body.contains("asset not licensed"),
                "denial body must propagate server message, got: {body:?}"
            );
        }
        other => panic!("expected DrmError::LicenseDenied, got {other:?}"),
    }
}

/// A response with missing `"ckc"` field surfaces as `DrmError::LicenseError`.
#[tokio::test]
async fn test_fairplay_missing_ckc_field() {
    let bad_response = br#"{"status":"ok"}"#.to_vec();
    let state = MockState::new();
    let addr = spawn_mock_server(MockBehavior::Ok(bad_response), state.clone()).await;

    let transport = HyperPlainFairPlayClient::new().with_timeout_ms(5_000);
    let mut fp_client = FairPlayClient::new(b"cert".to_vec());

    let err = fp_client
        .request_key_from_server(&build_url(addr), "asset-y".to_string(), &transport, &[])
        .await
        .expect_err("must fail when 'ckc' field is missing");

    match err {
        DrmError::LicenseError(msg) => {
            assert!(
                msg.contains("ckc"),
                "error must mention 'ckc' field, got: {msg}"
            );
        }
        other => panic!("expected DrmError::LicenseError, got {other:?}"),
    }
}

/// The JSON body sent on the wire has the correct `asset_id` and SPC Base64.
#[tokio::test]
async fn test_fairplay_request_json_encoding() {
    let ckc_bytes = b"round-trip-ckc";
    let response_body = sample_ckc_response(ckc_bytes);
    let state = MockState::new();
    let addr = spawn_mock_server(MockBehavior::Ok(response_body), state.clone()).await;

    let transport = HyperPlainFairPlayClient::new().with_timeout_ms(5_000);
    let mut fp_client = FairPlayClient::new(b"my-cert".to_vec());

    fp_client
        .request_key_from_server(
            &build_url(addr),
            "asset-roundtrip".to_string(),
            &transport,
            &[],
        )
        .await
        .expect("must succeed");

    let captured = state.last_captured();
    let v: serde_json::Value =
        serde_json::from_slice(&captured.body).expect("body must be valid JSON");

    assert_eq!(
        v["asset_id"].as_str().unwrap_or(""),
        "asset-roundtrip",
        "asset_id must match"
    );

    // The `spc` field must be a non-empty Base64 string
    let spc_b64 = v["spc"].as_str().expect("spc field must be a string");
    let spc_decoded = STANDARD.decode(spc_b64).expect("spc must be valid Base64");
    assert!(!spc_decoded.is_empty(), "decoded SPC must not be empty");
}

/// A stalling server triggers a timeout.
#[tokio::test]
async fn test_fairplay_timeout_respected() {
    let state = MockState::new();
    let addr = spawn_mock_server(MockBehavior::Stall { delay_ms: 2_000 }, state.clone()).await;

    let transport = HyperPlainFairPlayClient::new().with_timeout_ms(150);
    let mut fp_client = FairPlayClient::new(b"cert".to_vec());

    let started = std::time::Instant::now();
    let err = fp_client
        .request_key_from_server(&build_url(addr), "asset-stall".to_string(), &transport, &[])
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

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests (no network)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_build_ksm_json_encodes_spc() {
    let spc = b"binary-spc";
    let json_bytes = build_ksm_json("asset-unit", spc, None).expect("build_ksm_json");
    let v: serde_json::Value = serde_json::from_slice(&json_bytes).expect("valid JSON");

    let decoded = STANDARD
        .decode(v["spc"].as_str().expect("spc"))
        .expect("base64");
    assert_eq!(decoded, spc.to_vec());
    assert_eq!(v["asset_id"].as_str().unwrap_or(""), "asset-unit");
}

#[test]
fn test_parse_ckc_response_ok() {
    let ckc = b"decoded-ckc-data";
    let resp =
        serde_json::to_vec(&serde_json::json!({ "ckc": STANDARD.encode(ckc) })).expect("json");
    let result = parse_ckc_response(&resp).expect("parse_ckc_response");
    assert_eq!(result, ckc.to_vec());
}

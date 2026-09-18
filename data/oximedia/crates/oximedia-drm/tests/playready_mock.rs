//! End-to-end integration tests for the PlayReady license RPC layer.
//!
//! These tests drive a plain-TCP `hyper` mock server bound to an ephemeral
//! port and pair it with the `HyperPlainPlayReadyClient` transport to verify:
//!
//! 1. `PlayReadyClient::acquire_license` performs the full SOAP request →
//!    response → license-parse flow against a real (in-process) HTTP server.
//! 2. Non-2xx HTTP responses bubble up as `DrmError::LicenseDenied`.
//! 3. Malformed response bodies surface as `DrmError::XmlError`.
//! 4. The wire request contains the required SOAP headers and challenge.
//! 5. Per-phase timeouts are honoured when the server stalls.

#![cfg(feature = "playready")]

#[path = "common/mod.rs"]
mod common;

use std::net::SocketAddr;
use std::time::Duration;

use base64::engine::general_purpose::STANDARD;
use base64::Engine;

use common::{spawn_mock_server, MockBehavior, MockState};

use oximedia_drm::playready::{PlayReadyLicense, PlayReadyLicenseChallenge};
use oximedia_drm::playready_rpc::{
    build_soap_envelope, HyperPlainPlayReadyClient, PlayReadyClient,
};
use oximedia_drm::DrmError;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn build_url(addr: SocketAddr) -> String {
    format!("http://{addr}/playready/license")
}

/// Build a minimal but valid SOAP response containing a Base64-encoded license.
fn sample_soap_response(license_bytes: &[u8]) -> Vec<u8> {
    let b64 = STANDARD.encode(license_bytes);
    format!(
        r#"<?xml version="1.0"?>
<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
  <soap:Body>
    <AcquireLicenseResponse>
      <License>{b64}</License>
    </AcquireLicenseResponse>
  </soap:Body>
</soap:Envelope>"#
    )
    .into_bytes()
}

fn make_client_and_challenge() -> (PlayReadyClient, Vec<u8>) {
    let license_bytes = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04];
    let challenge = PlayReadyLicenseChallenge::new(b"challenge-data".to_vec());
    let session_id = vec![0x11, 0x22, 0x33, 0x44];
    let client = PlayReadyClient::new(session_id, challenge);
    (client, license_bytes)
}

// ─────────────────────────────────────────────────────────────────────────────
// Integration tests
// ─────────────────────────────────────────────────────────────────────────────

/// Full license acquisition flow: challenge → SOAP POST → license parse.
#[tokio::test]
async fn test_playready_full_license_flow_via_mock_server() {
    let (pr_client, license_bytes) = make_client_and_challenge();
    let soap_response = sample_soap_response(&license_bytes);

    let state = MockState::new();
    let addr = spawn_mock_server(MockBehavior::Ok(soap_response), state.clone()).await;

    let transport = HyperPlainPlayReadyClient::new().with_timeout_ms(5_000);

    let license: PlayReadyLicense = pr_client
        .acquire_license(&build_url(addr), &transport, &[])
        .await
        .expect("acquire_license must succeed");

    assert_eq!(
        license.data, license_bytes,
        "decoded license bytes must match mock payload"
    );

    let captured = state.last_captured();
    assert_eq!(captured.path, "/playready/license");

    // Content-Type must be text/xml for SOAP
    let ct = captured.content_type.as_deref().unwrap_or("");
    assert!(
        ct.starts_with("text/xml"),
        "Content-Type must be text/xml, got: {ct}"
    );

    // SOAPAction header must be present with embedded quotes
    let sa = captured.soap_action.as_deref().unwrap_or("");
    assert!(
        sa.contains("AcquireLicense"),
        "SOAPAction must reference AcquireLicense, got: {sa}"
    );
    assert!(
        !captured.body.is_empty(),
        "server must receive a non-empty SOAP body"
    );
}

/// Non-2xx response maps to `DrmError::LicenseDenied`.
#[tokio::test]
async fn test_playready_license_denied_403() {
    let (pr_client, _) = make_client_and_challenge();
    let state = MockState::new();
    let addr = spawn_mock_server(
        MockBehavior::Status {
            status: 403,
            body: "device not registered".to_string(),
        },
        state.clone(),
    )
    .await;

    let transport = HyperPlainPlayReadyClient::new().with_timeout_ms(5_000);

    let err = pr_client
        .acquire_license(&build_url(addr), &transport, &[])
        .await
        .expect_err("acquire_license must fail on HTTP 403");

    match err {
        DrmError::LicenseDenied { status, body } => {
            assert_eq!(status, 403, "denial status must be 403");
            assert!(
                body.contains("device not registered"),
                "denial body must contain server message, got: {body:?}"
            );
        }
        other => panic!("expected DrmError::LicenseDenied, got {other:?}"),
    }
}

/// A non-XML response body surfaces as `DrmError::XmlError`.
#[tokio::test]
async fn test_playready_malformed_response() {
    let (pr_client, _) = make_client_and_challenge();
    let garbage = b"this is not valid XML or SOAP at all".to_vec();
    let state = MockState::new();
    let addr = spawn_mock_server(MockBehavior::Ok(garbage), state.clone()).await;

    let transport = HyperPlainPlayReadyClient::new().with_timeout_ms(5_000);

    let err = pr_client
        .acquire_license(&build_url(addr), &transport, &[])
        .await
        .expect_err("acquire_license must reject non-XML body");

    match err {
        DrmError::XmlError(_) => {}
        other => panic!("expected DrmError::XmlError, got {other:?}"),
    }
}

/// The SOAP envelope sent on the wire can be parsed back to recover the challenge.
#[tokio::test]
async fn test_playready_request_body_contains_soap_envelope() {
    let license_bytes = b"roundtrip-license";
    let challenge_data = b"roundtrip-challenge";
    let challenge = PlayReadyLicenseChallenge::new(challenge_data.to_vec());
    let pr_client = PlayReadyClient::new(vec![0xAA, 0xBB], challenge);

    let soap_response = sample_soap_response(license_bytes);
    let state = MockState::new();
    let addr = spawn_mock_server(MockBehavior::Ok(soap_response), state.clone()).await;

    let transport = HyperPlainPlayReadyClient::new().with_timeout_ms(5_000);

    pr_client
        .acquire_license(&build_url(addr), &transport, &[])
        .await
        .expect("acquire_license must succeed");

    let captured = state.last_captured();
    let body_str = std::str::from_utf8(&captured.body).expect("body must be UTF-8");

    assert!(
        body_str.contains("soap:Envelope"),
        "request body must contain soap:Envelope"
    );
    assert!(
        body_str.contains("AcquireLicense"),
        "request body must contain AcquireLicense"
    );

    // The challenge bytes must be Base64-encoded somewhere in the envelope.
    let expected_b64 = STANDARD.encode(challenge_data);
    assert!(
        body_str.contains(&expected_b64),
        "request body must contain the Base64-encoded challenge"
    );
}

/// A stalling server triggers `DrmError::NetworkError` before the 2s stall.
#[tokio::test]
async fn test_playready_timeout_respected() {
    let (pr_client, _) = make_client_and_challenge();
    let state = MockState::new();
    let addr = spawn_mock_server(MockBehavior::Stall { delay_ms: 2_000 }, state.clone()).await;

    let transport = HyperPlainPlayReadyClient::new().with_timeout_ms(150);

    let started = std::time::Instant::now();
    let err = pr_client
        .acquire_license(&build_url(addr), &transport, &[])
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
// Unit tests (SOAP envelope / URL parser — no network)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_soap_envelope_roundtrip_challenge_bytes() {
    let challenge_bytes = b"test-challenge-bytes-for-envelope";
    let b64 = STANDARD.encode(challenge_bytes);
    let envelope = build_soap_envelope(&b64);

    assert!(
        envelope.contains(&b64),
        "envelope must embed the challenge b64"
    );
    assert!(
        envelope.contains("soap:Envelope"),
        "must have soap:Envelope"
    );
    assert!(
        envelope.contains("AcquireLicense"),
        "must have AcquireLicense"
    );
    assert!(envelope.contains("soap:Body"), "must have soap:Body");
}

#[test]
fn test_playready_client_session_id() {
    let session_id = vec![0x01, 0x02, 0x03];
    let challenge = PlayReadyLicenseChallenge::new(b"data".to_vec());
    let client = PlayReadyClient::new(session_id.clone(), challenge);
    assert_eq!(client.session_id(), session_id.as_slice());
}

#[test]
fn test_soap_response_parse_missing_license_element() {
    use oximedia_drm::playready_rpc::parse_soap_response;
    let response = b"<soap:Body><NoLicenseHere/></soap:Body>";
    let err = parse_soap_response(response).expect_err("must fail");
    match err {
        DrmError::XmlError(_) => {}
        other => panic!("expected XmlError, got {other:?}"),
    }
}

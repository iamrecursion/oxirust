//! Integration tests for the W3C ClearKey EME (Encrypted Media Extensions)
//! JSON license format.
//!
//! Test vectors here follow the W3C Encrypted Media Extensions ClearKey
//! specification, which mandates a JSON wire format:
//!
//!   Request:  `{"kids":["<base64url>", ...], "type":"temporary"}`
//!   Response: `{"keys":[{"kty":"oct","k":"<base64url>","kid":"<base64url>"}], "type":"temporary"}`
//!
//! Source: https://www.w3.org/TR/encrypted-media/#clear-key
//!
//! These integration tests smoke-test the public API surface of
//! `oximedia_drm::clearkey` end-to-end, without using any internal helpers.

#![cfg(feature = "clearkey")]

use oximedia_drm::clearkey::{
    generate_clearkey_license, parse_clearkey_request, ClearKeyClient, ClearKeyEntry,
    ClearKeyLicense, ClearKeyRequest, ClearKeyResponse, ClearKeyServer,
};
use std::collections::HashMap;

#[test]
fn w3c_clearkey_generate_license_emits_jwk_shape() {
    let kid: [u8; 16] = [
        0x43, 0xba, 0xfe, 0x30, 0x4f, 0x57, 0x43, 0x5e, 0x87, 0x5d, 0x0c, 0x7b, 0xe3, 0x3e, 0x0e,
        0x9d,
    ];
    let key: [u8; 16] = [
        0xeb, 0x67, 0x62, 0xa7, 0x72, 0x7f, 0x4c, 0x41, 0x81, 0x9e, 0xc0, 0x7b, 0x96, 0x10, 0x3c,
        0x91,
    ];

    let json = generate_clearkey_license(&kid, &key).expect("license generation");

    let license = ClearKeyLicense::from_json(&json).expect("license parse");
    assert_eq!(license.key_type, "temporary");
    assert_eq!(license.keys.len(), 1);
    assert_eq!(license.keys[0].kty, "oct");

    let recovered_kid = license.keys[0].key_id_bytes().expect("kid decode");
    let recovered_k = license.keys[0].key_value_bytes().expect("key decode");
    assert_eq!(recovered_kid, kid.to_vec());
    assert_eq!(recovered_k, key.to_vec());
}

#[test]
fn w3c_clearkey_request_two_kids_roundtrip() {
    // ClearKey EME license request from a browser with two content keys.
    let kid1: [u8; 16] = [
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x01,
    ];
    let kid2: [u8; 16] = [
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x02,
    ];

    let request = ClearKeyRequest::new(vec![kid1.to_vec(), kid2.to_vec()]);
    let json = request.to_json().expect("serialize request");

    let parsed = parse_clearkey_request(&json).expect("parse request");
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed[0], kid1);
    assert_eq!(parsed[1], kid2);
}

#[test]
fn w3c_clearkey_response_two_keys_extracted_to_hashmap() {
    let kid1 = vec![0x10u8; 16];
    let key1 = vec![0xAAu8; 16];
    let kid2 = vec![0x20u8; 16];
    let key2 = vec![0xBBu8; 16];

    let mut server = ClearKeyServer::new();
    server.add_key(kid1.clone(), key1.clone());
    server.add_key(kid2.clone(), key2.clone());

    let request = ClearKeyRequest::new(vec![kid1.clone(), kid2.clone()]);
    let response: ClearKeyResponse = server
        .process_request(&request)
        .expect("server respond with both keys");

    let keys_map: HashMap<Vec<u8>, Vec<u8>> = response.get_keys_map().expect("keys map decode");
    assert_eq!(keys_map.len(), 2, "both keys extracted");
    assert_eq!(keys_map.get(&kid1), Some(&key1));
    assert_eq!(keys_map.get(&kid2), Some(&key2));
}

#[test]
fn w3c_clearkey_end_to_end_request_response_key_lookup() {
    // Single content key — full request -> response -> client.get_key roundtrip.
    let kid: [u8; 16] = [
        0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE,
        0xEF,
    ];
    let key: [u8; 16] = [
        0xCA, 0xFE, 0xBA, 0xBE, 0xCA, 0xFE, 0xBA, 0xBE, 0xCA, 0xFE, 0xBA, 0xBE, 0xCA, 0xFE, 0xBA,
        0xBE,
    ];

    let mut server = ClearKeyServer::new();
    server.add_key(kid.to_vec(), key.to_vec());

    let mut client = ClearKeyClient::new();
    client
        .request_keys(vec![kid.to_vec()], &server)
        .expect("client should obtain key");

    let recovered = client.get_key(&kid.to_vec()).expect("key in client map");
    assert_eq!(*recovered, key.to_vec(), "client key matches server key");
    assert_eq!(client.key_count(), 1);
}

#[test]
fn w3c_clearkey_entry_jwk_field_ordering_and_decoding() {
    let kid = [0x11u8; 16];
    let key = [0x22u8; 16];

    let entry = ClearKeyEntry::from_bytes(&kid, &key);
    assert_eq!(entry.kty, "oct");
    // Both kid and k must round-trip exactly
    assert_eq!(entry.key_id_bytes().expect("kid bytes"), kid.to_vec());
    assert_eq!(entry.key_value_bytes().expect("k bytes"), key.to_vec());
}

#[test]
fn w3c_clearkey_request_with_invalid_base64_kid_rejected() {
    // The W3C spec mandates base64url-encoded KIDs.
    let bad_json = r#"{"kids":["!!!!not-valid-base64!!!!"],"type":"temporary"}"#;
    let result = parse_clearkey_request(bad_json);
    assert!(result.is_err(), "invalid base64 must be rejected");
}

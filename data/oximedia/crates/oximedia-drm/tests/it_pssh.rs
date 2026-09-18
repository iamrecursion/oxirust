//! Integration tests for PSSH (Protection System Specific Header) box parsing.
//!
//! Smoke-tests the `oximedia_drm::pssh` public surface against real-world
//! Widevine and PlayReady PSSH byte structures (v0 and v1 boxes).
//!
//! The PSSH byte sequences here are reconstructed from the public ISO/IEC
//! 23001-7 spec layout used by `shaka-packager`, `bento4`, and the
//! Widevine/PlayReady SDKs. Box layout:
//!   - 4 bytes: total size (BE)
//!   - 4 bytes: box type `'pssh'`
//!   - 1 byte:  version
//!   - 3 bytes: flags
//!   - 16 bytes: system_id
//!   - [version >= 1] 4 bytes key_id_count + N*16 bytes key_ids
//!   - 4 bytes: data_size
//!   - data_size bytes: data
//!
//! Source references:
//!   - https://www.w3.org/TR/eme-initdata-cenc/
//!   - https://learn.microsoft.com/en-us/playready/

use oximedia_drm::pssh::{
    build_pssh_v1, PsshBox, PsshBoxV1, PsshBuilder, CLEARKEY_SYSTEM_ID, FAIRPLAY_SYSTEM_ID,
    PLAYREADY_SYSTEM_ID, WIDEVINE_SYSTEM_ID,
};

/// Assemble a PSSH v0 box from raw fixture bytes.
fn build_v0(system_id: [u8; 16], data: &[u8]) -> Vec<u8> {
    let size = 4 + 4 + 1 + 3 + 16 + 4 + data.len();
    let mut out = Vec::with_capacity(size);
    out.extend_from_slice(&(size as u32).to_be_bytes());
    out.extend_from_slice(b"pssh");
    out.push(0); // version
    out.push(0);
    out.push(0);
    out.push(0); // flags
    out.extend_from_slice(&system_id);
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(data);
    out
}

/// Assemble a PSSH v1 box from raw fixture bytes.
fn build_v1(system_id: [u8; 16], key_ids: &[[u8; 16]], data: &[u8]) -> Vec<u8> {
    build_pssh_v1(system_id, key_ids, data)
}

#[test]
fn parses_real_widevine_pssh_v0_protobuf_payload() {
    // Widevine v0 PSSH carries a protobuf-encoded WidevineCencHeader.
    // This is a minimal valid varint-style payload: field 1 (content_id),
    // length-delimited 8 bytes.
    let widevine_protobuf_payload: &[u8] = &[
        0x0A, 0x08, // tag 0x0A (field 1, wire-type 2 length-delimited), length 8
        0x73, 0x68, 0x61, 0x6B, 0x61, 0x2D, 0x69, 0x64, // "shaka-id"
    ];

    let raw = build_v0(WIDEVINE_SYSTEM_ID, widevine_protobuf_payload);
    let boxes = PsshBox::parse(&raw).expect("Widevine v0 PSSH should parse");

    assert_eq!(boxes.len(), 1, "exactly one PSSH box");
    assert_eq!(boxes[0].system_id, WIDEVINE_SYSTEM_ID);
    assert_eq!(boxes[0].data, widevine_protobuf_payload);
    assert!(
        boxes[0].key_ids.is_empty(),
        "v0 PSSH has no explicit key IDs"
    );
    assert_eq!(boxes[0].drm_system_name(), Some("Widevine"));
}

#[test]
fn parses_real_widevine_pssh_v1_with_two_kids() {
    // A Widevine v1 PSSH from a multi-track DASH MPD: separate KIDs for
    // video and audio tracks. Box payload mirrors the protobuf field 2
    // (key_id, repeated) encoding.
    let kid_video: [u8; 16] = [
        0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67,
        0x89,
    ];
    let kid_audio: [u8; 16] = [
        0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10, 0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32,
        0x10,
    ];

    let raw = build_v1(WIDEVINE_SYSTEM_ID, &[kid_video, kid_audio], b"wv-payload");
    let boxes = PsshBox::parse(&raw).expect("Widevine v1 PSSH should parse");

    assert_eq!(boxes.len(), 1);
    assert_eq!(boxes[0].system_id, WIDEVINE_SYSTEM_ID);
    assert_eq!(boxes[0].key_ids.len(), 2);
    assert_eq!(boxes[0].key_ids[0], kid_video.to_vec());
    assert_eq!(boxes[0].key_ids[1], kid_audio.to_vec());
    assert_eq!(boxes[0].data, b"wv-payload");
}

#[test]
fn parses_real_playready_pssh_v0_with_pro_object() {
    // PlayReady PSSH v0 carries a PlayReady Object (PRO):
    //   Length (4 LE) | RecordCount (2 LE) | Records...
    // Record format: Type (2 LE) | Length (2 LE) | Value (UTF-16LE WRMHEADER XML)
    let wrm_header_utf8 = b"<WRMHEADER xmlns=\"http://schemas.microsoft.com/DRM/2007/03/PlayReadyHeader\" version=\"4.0.0.0\"></WRMHEADER>";

    // Convert UTF-8 to fake UTF-16LE (each byte → byte 0x00 follows) for spec realism.
    let mut wrm_utf16le: Vec<u8> = Vec::with_capacity(wrm_header_utf8.len() * 2);
    for &b in wrm_header_utf8 {
        wrm_utf16le.push(b);
        wrm_utf16le.push(0x00);
    }

    let record_type: u16 = 1; // WRM_HEADER
    let record_value_len = wrm_utf16le.len() as u16;
    let pro_total: u32 = (4 + 2 + 2 + 2 + wrm_utf16le.len()) as u32;

    let mut pro = Vec::new();
    pro.extend_from_slice(&pro_total.to_le_bytes());
    pro.extend_from_slice(&1u16.to_le_bytes()); // record_count
    pro.extend_from_slice(&record_type.to_le_bytes());
    pro.extend_from_slice(&record_value_len.to_le_bytes());
    pro.extend_from_slice(&wrm_utf16le);

    let raw = build_v0(PLAYREADY_SYSTEM_ID, &pro);
    let boxes = PsshBox::parse(&raw).expect("PlayReady v0 PSSH should parse");

    assert_eq!(boxes.len(), 1);
    assert_eq!(boxes[0].system_id, PLAYREADY_SYSTEM_ID);
    assert_eq!(boxes[0].drm_system_name(), Some("PlayReady"));
    assert_eq!(boxes[0].data, pro, "PRO bytes must survive round-trip");
    assert!(boxes[0].key_ids.is_empty());
}

#[test]
fn parses_multi_drm_concatenated_widevine_and_playready() {
    // Real-world DASH segments concatenate PSSH boxes for each DRM system.
    let wv = build_v0(WIDEVINE_SYSTEM_ID, b"protobuf-widevine");
    let pr = build_v0(PLAYREADY_SYSTEM_ID, b"playready-PRO");
    let mut concat = wv.clone();
    concat.extend_from_slice(&pr);

    let boxes = PsshBox::parse(&concat).expect("multi-DRM PSSH should parse");
    assert_eq!(boxes.len(), 2);

    assert_eq!(boxes[0].system_id, WIDEVINE_SYSTEM_ID);
    assert_eq!(boxes[0].drm_system_name(), Some("Widevine"));

    assert_eq!(boxes[1].system_id, PLAYREADY_SYSTEM_ID);
    assert_eq!(boxes[1].drm_system_name(), Some("PlayReady"));
}

#[test]
fn pssh_v1_serialize_parse_roundtrip_preserves_kids_and_data() {
    let kids: Vec<[u8; 16]> = vec![[0x11; 16], [0x22; 16], [0x33; 16]];
    let v1 = PsshBoxV1::new(WIDEVINE_SYSTEM_ID, kids.clone(), b"binary-payload".to_vec());

    let raw = v1.serialize();

    // Size field consistency
    let size_field = u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]) as usize;
    assert_eq!(size_field, raw.len(), "size field matches actual length");

    // Box type 'pssh'
    assert_eq!(&raw[4..8], b"pssh");
    // Version byte must be 1
    assert_eq!(raw[8], 1);

    let parsed = PsshBoxV1::parse(&raw).expect("v1 PSSH should round-trip");
    assert_eq!(parsed.system_id, WIDEVINE_SYSTEM_ID);
    assert_eq!(parsed.key_ids, kids);
    assert_eq!(parsed.data, b"binary-payload");
}

#[test]
fn pssh_builder_v0_for_fairplay_and_clearkey_systems() {
    // FairPlay PSSH v0
    let fp_pssh = PsshBuilder::new()
        .set_system_id(FAIRPLAY_SYSTEM_ID)
        .set_data(b"fairplay-ksm-payload".to_vec())
        .build();
    assert_eq!(fp_pssh.drm_system_name(), Some("FairPlay"));
    assert!(fp_pssh.key_ids.is_empty());

    let fp_raw = fp_pssh.serialize();
    let fp_parsed = PsshBox::parse(&fp_raw).expect("FairPlay PSSH should parse");
    assert_eq!(fp_parsed[0].system_id, FAIRPLAY_SYSTEM_ID);

    // ClearKey PSSH v1
    let ck_pssh = PsshBuilder::new()
        .set_system_id(CLEARKEY_SYSTEM_ID)
        .add_key_id(vec![0xAA; 16])
        .set_data(b"{\"kids\":[\"...\"]}".to_vec())
        .build();
    assert_eq!(ck_pssh.drm_system_name(), Some("ClearKey"));
    assert_eq!(ck_pssh.key_ids.len(), 1);
}

#[test]
fn rejects_corrupted_pssh_boxes() {
    // Box too short (< 8 bytes for header)
    assert!(PsshBox::parse(&[0u8; 4]).is_err());

    // Wrong box type
    let mut bad = build_v0(WIDEVINE_SYSTEM_ID, b"x");
    bad[4] = b'X'; // corrupt 'pssh' → 'Xssh'
    assert!(PsshBox::parse(&bad).is_err());

    // Truncated size > available
    let mut bad_size = build_v0(WIDEVINE_SYSTEM_ID, b"x");
    bad_size[3] = 0xFF; // declare size of 255 with only ~30 bytes available
    assert!(PsshBox::parse(&bad_size).is_err());
}

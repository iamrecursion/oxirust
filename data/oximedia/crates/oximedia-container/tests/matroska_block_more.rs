//! Integration tests for Matroska v4 block-level `BlockMore`/`BlockAdditional` parsing
//! and the `BlockAddIdType` registry.
//!
//! These tests validate:
//! 1. Hand-crafted EBML byte fixtures that exercise the new parser path.
//! 2. The `BlockAddIdType` registry: `from_id` / `id` round-trips.
//! 3. Regression: `BlockGroup` without `BlockAdditions` still parses cleanly.

use oximedia_container::demux::matroska::ebml::element_id;
use oximedia_container::demux::matroska::matroska_v4::{parse_block_additions, BlockAddIdType};
use oximedia_container::demux::matroska::parser::parse_block_group;

// ============================================================================
// EBML fixture helpers
// ============================================================================

/// Encodes a minimal EBML element (raw ID bytes + VINT size + content).
///
/// The ID is emitted as the minimum number of big-endian bytes needed to hold
/// the value.  The size is encoded as a VINT (marker bit in MSB).
fn encode_element(id: u32, content: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    // ID bytes
    if id <= 0xFF {
        out.push(id as u8);
    } else if id <= 0xFFFF {
        out.push((id >> 8) as u8);
        out.push(id as u8);
    } else if id <= 0xFF_FFFF {
        out.push((id >> 16) as u8);
        out.push((id >> 8) as u8);
        out.push(id as u8);
    } else {
        out.push((id >> 24) as u8);
        out.push((id >> 16) as u8);
        out.push((id >> 8) as u8);
        out.push(id as u8);
    }
    // VINT-encoded size
    let len = content.len();
    if len < 0x7F {
        out.push((len as u8) | 0x80);
    } else if len < 0x3FFF {
        out.push(0x40 | ((len >> 8) as u8));
        out.push(len as u8);
    } else {
        out.push(0x20 | ((len >> 16) as u8));
        out.push((len >> 8) as u8);
        out.push(len as u8);
    }
    out.extend_from_slice(content);
    out
}

/// Encodes a u64 as a minimal-byte big-endian binary element.
fn encode_uint_element(id: u32, v: u64) -> Vec<u8> {
    let bytes = if v == 0 {
        vec![0u8]
    } else {
        let mut b = v.to_be_bytes().to_vec();
        while b.len() > 1 && b[0] == 0 {
            b.remove(0);
        }
        b
    };
    encode_element(id, &bytes)
}

// ============================================================================
// test_block_more_parse_fixture
// ============================================================================

/// Build a hand-crafted `BlockAdditions` byte sequence and assert that
/// `parse_block_additions` extracts the expected `BlockMore` payload.
///
/// Structure:
///   BlockAdditions (0x75A1) {
///     BlockMore (0xA6) {
///       BlockAddID    (0xEE) = 4
///       BlockAdditional (0xA5) = b"test_itu35_payload"
///     }
///   }
#[test]
fn test_block_more_parse_fixture() {
    let payload = b"test_itu35_payload";

    // Build BlockMore content: BLOCK_ADD_ID(4) + BLOCK_ADDITIONAL(payload)
    let mut block_more_content = Vec::new();
    block_more_content.extend(encode_uint_element(element_id::BLOCK_ADD_ID, 4));
    block_more_content.extend(encode_element(element_id::BLOCK_ADDITIONAL, payload));

    // Wrap in a BlockMore master element
    let block_more_elem = encode_element(element_id::BLOCK_MORE, &block_more_content);

    // Wrap in a BlockAdditions master element
    let block_additions_elem = encode_element(element_id::BLOCK_ADDITIONS, &block_more_elem);

    // The content (after the outer BlockAdditions header) is `block_more_elem`
    // We pass it to parse_block_additions with the correct content size.
    let content_start = {
        // Skip the outer element header bytes to find where content starts
        let mut hdr_len = 0usize;
        if element_id::BLOCK_ADDITIONS <= 0xFF {
            hdr_len += 1;
        } else if element_id::BLOCK_ADDITIONS <= 0xFFFF {
            hdr_len += 2;
        } else {
            hdr_len += 3;
        }
        // VINT for block_more_elem.len()
        let len = block_more_elem.len();
        if len < 0x7F {
            hdr_len += 1;
        } else if len < 0x3FFF {
            hdr_len += 2;
        } else {
            hdr_len += 3;
        }
        hdr_len
    };
    let content = &block_additions_elem[content_start..];
    let content_size = content.len() as u64;

    let additions =
        parse_block_additions(content, content_size).expect("parse_block_additions should succeed");

    assert_eq!(additions.len(), 1, "Expected one BlockMore entry");
    assert_eq!(additions[0].add_id, 4, "BlockAddID should be 4 (ItuT35)");
    assert_eq!(
        additions[0].additional, payload,
        "BlockAdditional payload should match"
    );
}

// ============================================================================
// test_block_more_parse_two_entries
// ============================================================================

/// Build a `BlockAdditions` with two `BlockMore` children and verify both are
/// parsed correctly with distinct IDs and payloads.
#[test]
fn test_block_more_parse_two_entries() {
    let payload_a = b"hdr10plus_meta";
    let payload_b = b"dovi_rpu_data";

    // BlockMore #1: add_id=6 (Hdr10Plus), payload=payload_a
    let mut bm1_content = Vec::new();
    bm1_content.extend(encode_uint_element(element_id::BLOCK_ADD_ID, 6));
    bm1_content.extend(encode_element(element_id::BLOCK_ADDITIONAL, payload_a));
    let bm1_elem = encode_element(element_id::BLOCK_MORE, &bm1_content);

    // BlockMore #2: add_id=5 (DolbyVisionConfig), payload=payload_b
    let mut bm2_content = Vec::new();
    bm2_content.extend(encode_uint_element(element_id::BLOCK_ADD_ID, 5));
    bm2_content.extend(encode_element(element_id::BLOCK_ADDITIONAL, payload_b));
    let bm2_elem = encode_element(element_id::BLOCK_MORE, &bm2_content);

    // Concatenate both into BlockAdditions content
    let mut additions_content = Vec::new();
    additions_content.extend(&bm1_elem);
    additions_content.extend(&bm2_elem);

    let size = additions_content.len() as u64;
    let additions = parse_block_additions(&additions_content, size).expect("parse should succeed");

    assert_eq!(additions.len(), 2);
    assert_eq!(additions[0].add_id, 6);
    assert_eq!(additions[0].additional, payload_a.as_slice());
    assert_eq!(additions[1].add_id, 5);
    assert_eq!(additions[1].additional, payload_b.as_slice());
}

// ============================================================================
// test_block_more_default_add_id
// ============================================================================

/// A `BlockMore` element with no `BlockAddID` child should default to `add_id = 0`.
#[test]
fn test_block_more_default_add_id() {
    let payload = b"opaque_payload";

    // BlockMore content: only BLOCK_ADDITIONAL, no BLOCK_ADD_ID
    let block_more_content = encode_element(element_id::BLOCK_ADDITIONAL, payload);
    let block_more_elem = encode_element(element_id::BLOCK_MORE, &block_more_content);

    let size = block_more_elem.len() as u64;
    // Treat block_more_elem directly as BlockAdditions content (one BLOCK_MORE child)
    let additions = parse_block_additions(&block_more_elem, size).expect("parse should succeed");

    assert_eq!(additions.len(), 1);
    assert_eq!(additions[0].add_id, 0, "default BlockAddID should be 0");
    assert_eq!(additions[0].additional, payload.as_slice());
}

// ============================================================================
// test_block_add_id_registry
// ============================================================================

/// Verify the `BlockAddIdType` registry: `from_id` mappings and round-trip
/// `id()` → `from_id()` consistency.
#[test]
fn test_block_add_id_registry() {
    assert_eq!(BlockAddIdType::from_id(0), BlockAddIdType::Default);
    assert_eq!(BlockAddIdType::from_id(4), BlockAddIdType::ItuT35);
    assert_eq!(
        BlockAddIdType::from_id(5),
        BlockAddIdType::DolbyVisionConfig
    );
    assert_eq!(BlockAddIdType::from_id(6), BlockAddIdType::Hdr10Plus);
    assert_eq!(BlockAddIdType::from_id(12), BlockAddIdType::Iamf);
    assert_eq!(BlockAddIdType::from_id(99), BlockAddIdType::Unknown(99));
    assert_eq!(BlockAddIdType::from_id(1000), BlockAddIdType::Unknown(1000));

    // Round-trip: id() then from_id() should recover the same variant
    for &raw in &[0u64, 4, 5, 6, 12] {
        let t = BlockAddIdType::from_id(raw);
        assert_eq!(t.id(), raw, "round-trip failed for id={raw}");
        assert!(t.is_known(), "id={raw} should be known");
    }

    // Unknown variants round-trip too
    let unk = BlockAddIdType::from_id(42);
    assert_eq!(unk.id(), 42);
    assert!(!unk.is_known());

    // Explicit id() values
    assert_eq!(BlockAddIdType::Default.id(), 0);
    assert_eq!(BlockAddIdType::ItuT35.id(), 4);
    assert_eq!(BlockAddIdType::DolbyVisionConfig.id(), 5);
    assert_eq!(BlockAddIdType::Hdr10Plus.id(), 6);
    assert_eq!(BlockAddIdType::Iamf.id(), 12);
    assert_eq!(BlockAddIdType::Unknown(7).id(), 7);
}

// ============================================================================
// test_block_group_no_additions
// ============================================================================

/// A `BlockGroup` without a `BlockAdditions` element should parse successfully
/// and produce an empty `block_additions` list (regression guard).
///
/// We construct a minimal `BlockGroup` body: one `Block` element containing
/// a valid block header (track 1, timecode 0, keyframe, no lacing) + 4 bytes
/// of frame data.
#[test]
fn test_block_group_no_additions() {
    // Block header: VINT track=1 (0x81), timecode=0x0000, flags=0x80 (keyframe)
    // followed by 4 bytes of frame data
    let block_payload: Vec<u8> = vec![
        0x81, // track number: 1 as VINT
        0x00, 0x00, // timecode: 0
        0x80, // flags: keyframe
        0xDE, 0xAD, 0xBE, 0xEF, // frame data
    ];

    // Wrap in a Block element (0xA1)
    let block_elem = encode_element(element_id::BLOCK, &block_payload);

    let size = block_elem.len() as u64;
    let block =
        parse_block_group(&block_elem, size).expect("BlockGroup with no additions should parse");

    assert!(
        block.block_additions.is_empty(),
        "block_additions should be empty when no BlockAdditions element is present"
    );
    assert_eq!(block.header.track_number, 1);
    assert!(block.header.keyframe);
    assert_eq!(block.frames.len(), 1);
    assert_eq!(block.frames[0], vec![0xDE, 0xAD, 0xBE, 0xEF]);
}

// ============================================================================
// test_block_group_with_additions
// ============================================================================

/// A full `BlockGroup` with both a `Block` and a `BlockAdditions` element:
/// verifies that block data AND additions are both correctly extracted.
#[test]
fn test_block_group_with_additions() {
    let frame_data: &[u8] = &[0x01, 0x02, 0x03, 0x04];
    let addition_payload: &[u8] = b"itu35data";

    // Build Block element
    let mut block_payload = vec![
        0x81, // track number: 1 as VINT
        0x00, 0x00, // timecode: 0
        0x80, // flags: keyframe
    ];
    block_payload.extend_from_slice(frame_data);
    let block_elem = encode_element(element_id::BLOCK, &block_payload);

    // Build BlockMore content
    let mut bm_content = Vec::new();
    bm_content.extend(encode_uint_element(element_id::BLOCK_ADD_ID, 4));
    bm_content.extend(encode_element(
        element_id::BLOCK_ADDITIONAL,
        addition_payload,
    ));
    let bm_elem = encode_element(element_id::BLOCK_MORE, &bm_content);

    // Build BlockAdditions element
    let block_additions_elem = encode_element(element_id::BLOCK_ADDITIONS, &bm_elem);

    // Build BlockGroup content = Block + BlockAdditions
    let mut block_group_content = Vec::new();
    block_group_content.extend(&block_elem);
    block_group_content.extend(&block_additions_elem);

    let size = block_group_content.len() as u64;
    let block = parse_block_group(&block_group_content, size)
        .expect("BlockGroup with additions should parse");

    // Validate block header and frame data
    assert_eq!(block.header.track_number, 1);
    assert!(block.header.keyframe);
    assert_eq!(block.frames.len(), 1);
    assert_eq!(block.frames[0], frame_data);

    // Validate additions
    assert_eq!(block.block_additions.len(), 1);
    assert_eq!(block.block_additions[0].add_id, 4);
    assert_eq!(block.block_additions[0].additional, addition_payload);
}

// ============================================================================
// test_block_more_public_api
// ============================================================================

/// Verify that `BlockMore` and `BlockAddIdType` are accessible from the crate
/// root (the public API re-export path `oximedia_container::BlockMore`, etc.).
#[test]
fn test_block_more_public_api() {
    // These imports resolve through oximedia_container::BlockMore / BlockAddIdType
    // which are re-exported in lib.rs.
    use oximedia_container::{BlockAddIdType, BlockMore};

    let bm = BlockMore {
        add_id: 6,
        additional: b"hdr10plus".to_vec(),
    };
    assert_eq!(bm.add_id, 6);
    assert_eq!(
        BlockAddIdType::from_id(bm.add_id),
        BlockAddIdType::Hdr10Plus
    );
}

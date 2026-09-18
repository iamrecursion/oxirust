//! Matroska `SeekHead` (meta seek information) writing.
//!
//! The `SeekHead` element (EBML ID `0x114D9B74`) is a top-level (Level 1)
//! child of `Segment` that lists the byte offsets of other top-level
//! elements -- relative to the first byte of `Segment`'s *data* (i.e.
//! immediately after `Segment`'s own ID+size header) -- so that a reader can
//! jump directly to `Tracks`, `Cues`, etc. without a linear scan. See the
//! Matroska specification, "Global Elements" / "SeekHead":
//! <https://www.matroska.org/technical/elements.html>.
//!
//! Each `Seek` (`0x4DBB`) entry contains:
//! - `SeekID` (`0x53AB`): the raw EBML ID bytes of the referenced element.
//! - `SeekPosition` (`0x53AC`): its byte offset relative to `Segment` data
//!   start.
//!
//! # Why a placeholder/back-patch scheme
//!
//! [`super::writer::MatroskaMuxer`] writes `Cues` (the seek index) in the
//! trailer, *after* every `Cluster`, exactly like real single-pass/streaming
//! muxers (e.g. `mkvmerge` in streaming mode) do -- cluster byte positions
//! for the cue table aren't known until each cluster has actually been
//! written. That means the `Cues` element's offset isn't known when
//! `SeekHead` is written near the top of the file, right after the segment
//! header.
//!
//! This module solves that the same way real single-pass muxers do: reserve
//! a fixed-size placeholder `Seek` entry for each element of interest up
//! front, then overwrite just the `SeekPosition` bytes in place once the
//! real offset is known (or, if the target element is never written at all,
//! overwrite the whole reserved entry with a `Void` element). Every
//! `SeekPosition` is encoded as a fixed 8-byte-wide unsigned integer --EBML
//! explicitly permits zero-padding `uinteger` elements beyond their minimal
//! width-- so the byte layout never has to grow or shrink, and the
//! placeholder can always be safely overwritten in place. This mirrors the
//! seek-write-seek-back back-patch technique [`super::writer::MatroskaMuxer`]
//! already uses for the segment `Duration` field (see `fixup_duration`).

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]

use crate::demux::matroska::ebml::element_id;

// ============================================================================
// Seek Slot
// ============================================================================

/// A single reserved slot inside a placeholder `SeekHead` element, as
/// returned by [`SeekHeadWriter::build_placeholder`].
///
/// All offsets are relative to the start of the `SeekHead` element buffer
/// (i.e. byte 0 is the first byte of the `SeekHead` element's own ID), so a
/// caller that knows the absolute file offset at which it wrote that buffer
/// can translate these into absolute offsets by simple addition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeekSlot {
    /// EBML ID of the element this slot will eventually point to
    /// (e.g. [`element_id::CUES`]).
    pub target_id: u32,

    /// Byte offset of this slot's whole `Seek` sub-element (ID + size +
    /// content). A same-length `Void` element (see [`build_void`]) can
    /// overwrite this span in place if `target_id`'s element ends up never
    /// being written.
    pub entry_offset: usize,

    /// Total byte length of this slot's `Seek` sub-element.
    pub entry_len: usize,

    /// Byte offset of the 8-byte `SeekPosition` value to be patched once
    /// the target element's real, Segment-relative byte offset is known.
    pub position_field_offset: usize,
}

// ============================================================================
// Seek Head Writer
// ============================================================================

/// Builder for a fixed-size `SeekHead` placeholder, reserving one `Seek`
/// entry per target element ID.
#[derive(Debug, Default, Clone)]
pub struct SeekHeadWriter {
    target_ids: Vec<u32>,
}

impl SeekHeadWriter {
    /// Creates a writer that will reserve one `Seek` entry per ID in
    /// `target_ids`, in order.
    #[must_use]
    pub const fn new(target_ids: Vec<u32>) -> Self {
        Self { target_ids }
    }

    /// Builds the placeholder `SeekHead` element (ID + size + content, with
    /// every `SeekPosition` initialized to zero) and the list of reserved
    /// slots, in the same order as the `target_ids` passed to [`Self::new`].
    #[must_use]
    pub fn build_placeholder(&self) -> (Vec<u8>, Vec<SeekSlot>) {
        let mut content: Vec<u8> = Vec::new();
        let mut slots = Vec::with_capacity(self.target_ids.len());

        for &target_id in &self.target_ids {
            let entry_offset = content.len();

            // SeekID: binary payload is the raw EBML ID bytes of the
            // referenced element. EBML element IDs retain their
            // length-marker bit, so this is exactly the encoding
            // `encode_element_id` already uses to write element IDs
            // themselves (see `writer.rs`, `cues.rs`, `cluster.rs`).
            let id_bytes = encode_element_id(target_id);
            let mut seek_inner = Vec::new();
            seek_inner.extend(encode_element_id(element_id::SEEK_ID));
            seek_inner.extend(encode_vint_size(id_bytes.len() as u64));
            seek_inner.extend(&id_bytes);

            // SeekPosition: fixed 8-byte-wide uint placeholder, patched in
            // place once the real offset is known.
            seek_inner.extend(encode_element_id(element_id::SEEK_POSITION));
            seek_inner.push(0x88); // size = 8 bytes (VINT: 1000_1000)
            let position_in_seek_inner = seek_inner.len();
            seek_inner.extend_from_slice(&[0u8; 8]);

            let seek_id_bytes = encode_element_id(element_id::SEEK);
            let seek_size_bytes = encode_vint_size(seek_inner.len() as u64);

            content.extend(&seek_id_bytes);
            content.extend(&seek_size_bytes);
            content.extend(&seek_inner);

            let entry_len = seek_id_bytes.len() + seek_size_bytes.len() + seek_inner.len();
            let position_field_offset =
                entry_offset + seek_id_bytes.len() + seek_size_bytes.len() + position_in_seek_inner;

            slots.push(SeekSlot {
                target_id,
                entry_offset,
                entry_len,
                position_field_offset,
            });
        }

        let mut element = Vec::new();
        element.extend(encode_element_id(element_id::SEEK_HEAD));
        element.extend(encode_vint_size(content.len() as u64));
        let header_len = element.len();
        element.extend(content);

        // Translate every offset from "relative to `content`" to
        // "relative to the full returned `element` buffer".
        for slot in &mut slots {
            slot.entry_offset += header_len;
            slot.position_field_offset += header_len;
        }

        (element, slots)
    }
}

// ============================================================================
// Void Placeholder
// ============================================================================

/// Builds a `Void` element (EBML ID `0xEC`) of exactly `total_len` bytes.
///
/// Used to neutralize a reserved [`SeekSlot`] whose target element ended up
/// never being written (e.g. `Cues`, when a muxer run collects zero cue
/// points). A `Void` element's content is ignored by any conformant EBML
/// reader, so overwriting a slot with one in place is a safe, spec-legal way
/// to cancel a reservation without shifting any other byte offset in the
/// file.
///
/// Returns a same-length buffer for any `total_len >= 2`, which always holds
/// for slots produced by [`SeekHeadWriter::build_placeholder`] (21 bytes for
/// every 4-byte-ID target, the only kind this muxer reserves).
#[must_use]
pub fn build_void(total_len: usize) -> Vec<u8> {
    // The Void ID (0xEC) is always 1 byte. Find a size-VINT width that
    // makes `1 (id) + size_vint_len + content_len == total_len`
    // self-consistent, then fill the rest with zeroed content.
    for size_vint_len in 1..=8usize {
        if total_len < 1 + size_vint_len {
            break;
        }
        let content_len = total_len - 1 - size_vint_len;
        let size_vint = encode_vint_size(content_len as u64);
        if size_vint.len() == size_vint_len {
            let mut out = Vec::with_capacity(total_len);
            out.push(0xEC);
            out.extend(size_vint);
            out.resize(total_len, 0);
            return out;
        }
    }
    // Unreachable for any `total_len >= 2` (an 8-byte-wide size VINT can
    // always encode a content length up to 2^56-1), but degrade gracefully
    // rather than panic if ever called with a pathologically small length.
    vec![0u8; total_len]
}

// ============================================================================
// Encoding Helpers
// ============================================================================
//
// Deliberately duplicated (rather than shared) to match this crate's
// existing per-file convention in `mux/matroska/` (see `cues.rs`,
// `cluster.rs`, `writer.rs`, each of which carries its own copy).

/// Encodes an EBML element ID to its byte representation.
///
/// EBML element IDs already include their class marker bits, so this
/// simply outputs the minimal big-endian bytes that represent `id`.
fn encode_element_id(id: u32) -> Vec<u8> {
    if id <= 0xFF {
        vec![id as u8]
    } else if id <= 0xFFFF {
        vec![(id >> 8) as u8, id as u8]
    } else if id <= 0xFF_FFFF {
        vec![(id >> 16) as u8, (id >> 8) as u8, id as u8]
    } else {
        vec![
            (id >> 24) as u8,
            (id >> 16) as u8,
            (id >> 8) as u8,
            id as u8,
        ]
    }
}

/// Encodes a VINT size (element content length).
fn encode_vint_size(size: u64) -> Vec<u8> {
    if size < 0x80 {
        vec![0x80 | size as u8]
    } else if size < 0x4000 {
        vec![0x40 | (size >> 8) as u8, size as u8]
    } else if size < 0x1F_FFFF {
        vec![0x20 | (size >> 16) as u8, (size >> 8) as u8, size as u8]
    } else if size < 0x0FFF_FFFF {
        vec![
            0x10 | (size >> 24) as u8,
            (size >> 16) as u8,
            (size >> 8) as u8,
            size as u8,
        ]
    } else {
        vec![
            0x01,
            (size >> 48) as u8,
            (size >> 40) as u8,
            (size >> 32) as u8,
            (size >> 24) as u8,
            (size >> 16) as u8,
            (size >> 8) as u8,
            size as u8,
        ]
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demux::matroska::ebml;
    use crate::demux::matroska::parser::{self, MatroskaParser};

    #[test]
    fn test_build_placeholder_three_slots() {
        let writer =
            SeekHeadWriter::new(vec![element_id::INFO, element_id::TRACKS, element_id::CUES]);
        let (element, slots) = writer.build_placeholder();

        assert_eq!(slots.len(), 3);
        assert_eq!(slots[0].target_id, element_id::INFO);
        assert_eq!(slots[1].target_id, element_id::TRACKS);
        assert_eq!(slots[2].target_id, element_id::CUES);

        // Every reserved entry is 21 bytes for a 4-byte target ID:
        // Seek(2) + size(1) + [SeekID(2)+size(1)+id(4)] + [SeekPosition(2)+size(1)+value(8)]
        for slot in &slots {
            assert_eq!(slot.entry_len, 21);
        }

        // The whole element parses as a well-formed SeekHead via the
        // demuxer's own EBML header parser.
        let (_, header) = ebml::parse_element_header(&element).expect("seek head header parses");
        assert_eq!(header.id, element_id::SEEK_HEAD);
        assert_eq!(header.header_size + header.size as usize, element.len());
    }

    #[test]
    fn test_position_field_initially_zero() {
        let writer = SeekHeadWriter::new(vec![element_id::CUES]);
        let (element, slots) = writer.build_placeholder();
        let slot = slots[0];

        let field = &element[slot.position_field_offset..slot.position_field_offset + 8];
        assert_eq!(field, &[0u8; 8], "placeholder SeekPosition must start at 0");
    }

    #[test]
    fn test_build_void_matches_slot_length() {
        let writer = SeekHeadWriter::new(vec![element_id::CUES]);
        let (_, slots) = writer.build_placeholder();
        let slot = slots[0];

        let void = build_void(slot.entry_len);
        assert_eq!(void.len(), slot.entry_len);

        let (_, header) = ebml::parse_element_header(&void).expect("void element parses");
        assert_eq!(header.id, 0xEC, "Void element ID must be 0xEC");
        assert_eq!(
            header.header_size + header.size as usize,
            void.len(),
            "void element must consume exactly its reserved span"
        );
    }

    /// Proves the writer and the demuxer's *existing, unmodified*
    /// `parse_seek_head`/`parse_seek_entry` agree byte-for-byte on the
    /// `SeekHead` format: patch each reserved slot with a distinct value,
    /// then parse the whole buffer with the demuxer's own parser and check
    /// every `(id, position)` pair round-trips exactly.
    #[test]
    fn test_seek_head_round_trips_through_demuxer_parser() {
        let writer =
            SeekHeadWriter::new(vec![element_id::INFO, element_id::TRACKS, element_id::CUES]);
        let (mut element, slots) = writer.build_placeholder();

        let values: [u64; 3] = [0x21, 0x4141, 0x0099_0000];
        for (slot, &value) in slots.iter().zip(values.iter()) {
            element[slot.position_field_offset..slot.position_field_offset + 8]
                .copy_from_slice(&value.to_be_bytes());
        }

        let mut cursor = MatroskaParser::new(&element);
        let header = cursor.read_element().expect("seek head header parses");
        assert_eq!(header.id, element_id::SEEK_HEAD);
        let content = &element[header.header_size..header.header_size + header.size as usize];

        let entries =
            parser::parse_seek_head(content, header.size).expect("seek head content parses");
        assert_eq!(entries.len(), 3);

        let expected_ids = [element_id::INFO, element_id::TRACKS, element_id::CUES];
        for (entry, (&expected_id, &expected_pos)) in
            entries.iter().zip(expected_ids.iter().zip(values.iter()))
        {
            assert_eq!(entry.id, expected_id);
            assert_eq!(entry.position, expected_pos);
        }
    }

    #[test]
    fn test_two_slots_when_cues_omitted() {
        let writer = SeekHeadWriter::new(vec![element_id::INFO, element_id::TRACKS]);
        let (element, slots) = writer.build_placeholder();

        assert_eq!(slots.len(), 2);
        let (_, header) = ebml::parse_element_header(&element).expect("header parses");
        assert_eq!(header.size as usize, 2 * 21);
    }

    #[test]
    fn test_build_void_too_small_degrades_gracefully() {
        // Not a real EBML element, but must not panic.
        let void = build_void(1);
        assert_eq!(void.len(), 1);
    }
}

//! FLAC metadata writer.
//!
//! Split out of `metadata::writer` (which was approaching the workspace's
//! 2000-line-per-file guideline) into its own sibling module. FLAC's writer
//! is fully self-contained: it shares only the [`MetadataWriter`] trait and
//! [`TagMap`]/[`VorbisComments`] with the Matroska/Ogg writers in the parent
//! module, and depends on nothing they define.

use async_trait::async_trait;
use oximedia_core::{OxiError, OxiResult};
use oximedia_io::MediaSource;
use std::io::SeekFrom;

use crate::demux::flac::metadata::{BlockType, MetadataBlock};
use crate::metadata::tags::TagMap;
use crate::metadata::util::MediaSourceExt;
use crate::metadata::vorbis::VorbisComments;

use super::MetadataWriter;

/// FLAC metadata writer.
///
/// Updates or creates a Vorbis comment block in FLAC files.
///
/// # Strategy
///
/// Like `MatroskaMetadataWriter` and `OggMetadataWriter`
/// (`crate::metadata::writer`), this performs a **full-file rewrite**:
///
/// 1. Read the `fLaC` magic and every metadata block up to (and including)
///    the one carrying the last-metadata-block flag, replacing the Vorbis
///    comment block (or appending a new one if absent) while every other
///    block (STREAMINFO, SEEKTABLE, PICTURE, …) is preserved verbatim.
/// 2. Read every remaining byte (the audio frames) so they can be copied
///    back unmodified.
/// 3. Re-serialize the block list — STREAMINFO first (verified, not just
///    assumed), last-metadata-block flag on exactly the final block — and
///    splice it back in front of the audio frames.
/// 4. If the rewritten file would be shorter than the original, append a
///    `PADDING` metadata block so the output length never shrinks: a
///    [`MediaSource`] has no truncate operation, so writing fewer bytes than
///    the original at offset 0 would leave stale bytes from the old file
///    sitting right after the (now shorter) header, corrupting the stream
///    boundary between metadata and audio.
pub struct FlacMetadataWriter;

#[async_trait]
impl MetadataWriter for FlacMetadataWriter {
    async fn write<R: MediaSource>(source: &mut R, tags: &TagMap) -> OxiResult<()> {
        if !source.is_writable() {
            return Err(OxiError::Unsupported(
                "FLAC metadata writing requires a writable MediaSource".into(),
            ));
        }

        // Read FLAC magic
        source.seek(SeekFrom::Start(0)).await?;
        let mut magic = [0u8; 4];
        source.read_exact(&mut magic).await?;

        if &magic != b"fLaC" {
            return Err(OxiError::UnknownFormat);
        }

        // Read all metadata blocks
        let mut blocks = Vec::new();
        let mut vorbis_comment_found = false;

        loop {
            let mut header = [0u8; 4];

            if source.read_exact(&mut header).await.is_err() {
                break;
            }

            let is_last = header[0] & 0x80 != 0;
            let block_type = BlockType::from(header[0]);
            let length = u32::from_be_bytes([0, header[1], header[2], header[3]]);

            let mut block_data = vec![0u8; length as usize];
            source.read_exact(&mut block_data).await?;

            let block = MetadataBlock {
                is_last,
                block_type,
                length,
                data: block_data,
            };

            // Replace the Vorbis comment block; every other block type
            // (STREAMINFO, SEEKTABLE, PICTURE, APPLICATION, CUESHEET, …) is
            // kept byte-for-byte.
            if block_type == BlockType::VorbisComment {
                vorbis_comment_found = true;
                let new_comments = Self::create_vorbis_comment_block(tags, is_last);
                blocks.push(new_comments);
            } else {
                blocks.push(block);
            }

            if is_last {
                break;
            }
        }

        // The FLAC spec requires STREAMINFO to be present and to be the
        // first metadata block. A source that doesn't start with one is not
        // a conformant FLAC file we can safely rewrite — fail honestly
        // instead of silently producing a non-conformant copy.
        if blocks.first().map(|b| b.block_type) != Some(BlockType::StreamInfo) {
            return Err(OxiError::InvalidData(
                "FLAC file is missing a leading STREAMINFO metadata block".into(),
            ));
        }

        // If no Vorbis comment block was present, add one.
        if !vorbis_comment_found {
            let new_comments = Self::create_vorbis_comment_block(tags, false);
            blocks.push(new_comments);
        }

        // Exactly the last block in the (possibly reordered) list carries
        // the last-metadata-block flag.
        if let Some(last_block) = blocks.last_mut() {
            last_block.is_last = true;
        }
        for block in blocks.iter_mut().rev().skip(1) {
            block.is_last = false;
        }

        // The read loop above left `source` positioned exactly at the first
        // byte of audio frame data (right after the block carrying the
        // last-metadata-block flag). Capture that boundary, then read every
        // remaining byte so it can be copied back verbatim after the
        // rewritten metadata.
        let audio_data_start = MediaSource::position(source);
        let mut audio_data: Vec<u8> = Vec::new();
        let mut chunk = [0u8; 8192];
        loop {
            let n = source.read(&mut chunk).await?;
            if n == 0 {
                break;
            }
            audio_data.extend_from_slice(&chunk[..n]);
        }
        let original_total_len = audio_data_start + audio_data.len() as u64;

        // Serialize the new block list.
        let mut new_blocks_bytes = Vec::new();
        for block in &blocks {
            encode_block(&mut new_blocks_bytes, block);
        }

        // A full-file rewrite onto the same backing store cannot truncate it
        // (`MediaSource` has no `set_len`). If the new content is shorter
        // than the original, append a PADDING block so the output is never
        // shorter than the original — this keeps an in-place rewrite (e.g.
        // onto a `FileSource`) self-consistent instead of leaving stale
        // trailing bytes from the old file mixed into the new audio stream.
        let base_len = 4u64 + new_blocks_bytes.len() as u64 + audio_data.len() as u64;
        if base_len < original_total_len {
            let gap = original_total_len - base_len;
            // A PADDING block has a fixed 4-byte header; the smallest one
            // (0 data bytes) already covers gaps of 1..=4 bytes, growing the
            // output very slightly rather than leaving it short.
            #[allow(clippy::cast_possible_truncation)]
            let padding_len = gap.saturating_sub(4) as u32;

            if let Some(last_block) = blocks.last_mut() {
                last_block.is_last = false;
            }
            blocks.push(MetadataBlock {
                is_last: true,
                block_type: BlockType::Padding,
                length: padding_len,
                data: vec![0u8; padding_len as usize],
            });

            new_blocks_bytes.clear();
            for block in &blocks {
                encode_block(&mut new_blocks_bytes, block);
            }
        }

        // Assemble and write the complete file back.
        let mut output = Vec::with_capacity(4 + new_blocks_bytes.len() + audio_data.len());
        output.extend_from_slice(b"fLaC");
        output.extend_from_slice(&new_blocks_bytes);
        output.extend_from_slice(&audio_data);

        source.seek(SeekFrom::Start(0)).await?;
        source.write_all(&output).await?;

        Ok(())
    }
}

impl FlacMetadataWriter {
    /// Creates a Vorbis comment metadata block from tags.
    fn create_vorbis_comment_block(tags: &TagMap, is_last: bool) -> MetadataBlock {
        let mut comments = VorbisComments::with_vendor("OxiMedia");
        comments.tags = tags.clone();

        let data = comments.encode();
        #[allow(clippy::cast_possible_truncation)]
        let length = data.len() as u32;

        MetadataBlock {
            is_last,
            block_type: BlockType::VorbisComment,
            length,
            data,
        }
    }
}

/// Serializes a single FLAC metadata block (4-byte header + payload) and
/// appends it to `out`.
///
/// Header layout (FLAC spec §"METADATA_BLOCK_HEADER"): bit 7 of the first
/// byte is the last-metadata-block flag, the low 7 bits are the block type;
/// the following 3 bytes are the big-endian block length in bytes. The
/// length is always derived from `block.data.len()` (not the possibly-stale
/// `block.length` field) so a caller can never desync the header from the
/// payload that follows it.
fn encode_block(out: &mut Vec<u8>, block: &MetadataBlock) {
    let mut header_byte = block.block_type.as_u8();
    if block.is_last {
        header_byte |= 0x80;
    }
    out.push(header_byte);
    #[allow(clippy::cast_possible_truncation)]
    let length = block.data.len() as u32;
    out.extend_from_slice(&length.to_be_bytes()[1..]);
    out.extend_from_slice(&block.data);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::reader::{FlacMetadataReader, MetadataReader};
    use oximedia_io::MemorySource;

    #[test]
    fn test_flac_create_vorbis_comment_block() {
        let mut tags = TagMap::new();
        tags.set("TITLE", "Test Title");
        tags.set("ARTIST", "Test Artist");

        let block = FlacMetadataWriter::create_vorbis_comment_block(&tags, false);

        assert_eq!(block.block_type, BlockType::VorbisComment);
        assert!(!block.is_last);
        assert!(block.length > 0);

        // Verify we can parse it back
        let comments = VorbisComments::parse(&block.data).expect("operation should succeed");
        assert_eq!(comments.tags.get_text("TITLE"), Some("Test Title"));
        assert_eq!(comments.tags.get_text("ARTIST"), Some("Test Artist"));
    }

    #[test]
    fn test_flac_create_empty_vorbis_comment_block() {
        let tags = TagMap::new();
        let block = FlacMetadataWriter::create_vorbis_comment_block(&tags, true);

        assert_eq!(block.block_type, BlockType::VorbisComment);
        assert!(block.is_last);

        let comments = VorbisComments::parse(&block.data).expect("operation should succeed");
        assert!(comments.is_empty());
    }

    // ── FLAC writer end-to-end tests ─────────────────────────────────────────

    /// A structurally-valid (but semantically-arbitrary) 34-byte STREAMINFO
    /// payload. `FlacMetadataWriter` never parses STREAMINFO's content — it
    /// only checks the block *type* and copies the bytes through — so varied,
    /// non-constant bytes are enough to make a byte-identity check after
    /// rewriting meaningful.
    fn streaminfo_payload() -> Vec<u8> {
        (0u8..34u8).collect()
    }

    fn flac_block(block_type: BlockType, is_last: bool, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut header = block_type.as_u8();
        if is_last {
            header |= 0x80;
        }
        out.push(header);
        #[allow(clippy::cast_possible_truncation)]
        let len = data.len() as u32;
        out.extend_from_slice(&len.to_be_bytes()[1..]);
        out.extend_from_slice(data);
        out
    }

    /// Builds a minimal but structurally valid FLAC file: `fLaC` magic,
    /// STREAMINFO, an optional Vorbis comment block carrying `tags`, then
    /// `audio_data` standing in for encoded audio frames (opaque to the
    /// writer, must survive byte-for-byte).
    fn build_minimal_flac(tags: Option<&TagMap>, audio_data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"fLaC");

        let comment_data = tags.map(|t| {
            let mut comments = VorbisComments::with_vendor("TestVendor");
            comments.tags = t.clone();
            comments.encode()
        });

        out.extend(flac_block(
            BlockType::StreamInfo,
            comment_data.is_none(),
            &streaminfo_payload(),
        ));
        if let Some(ref data) = comment_data {
            out.extend(flac_block(BlockType::VorbisComment, true, data));
        }
        out.extend_from_slice(audio_data);
        out
    }

    async fn writable_flac_source(bytes: &[u8]) -> MemorySource {
        let mut s = MemorySource::new_writable(bytes.len() + 4096);
        s.write_all(bytes).await.expect("seed write should succeed");
        s.seek(SeekFrom::Start(0))
            .await
            .expect("seek should succeed");
        s
    }

    #[tokio::test]
    async fn test_flac_write_updates_tags_and_preserves_audio() {
        let mut old_tags = TagMap::new();
        old_tags.set("TITLE", "Old Title");
        let audio_data: Vec<u8> = (0..500u32).map(|i| (i % 256) as u8).collect();
        let initial = build_minimal_flac(Some(&old_tags), &audio_data);

        let mut source = writable_flac_source(&initial).await;

        let mut new_tags = TagMap::new();
        new_tags.set("TITLE", "New Title");
        new_tags.set("ARTIST", "New Artist");

        FlacMetadataWriter::write(&mut source, &new_tags)
            .await
            .expect("FlacMetadataWriter::write should succeed");

        let output = source.written_data().to_vec();

        // Structural check: magic + STREAMINFO must still be the first block.
        assert_eq!(&output[0..4], b"fLaC");
        assert_eq!(output[4] & 0x7F, BlockType::StreamInfo.as_u8());
        // STREAMINFO payload itself must be untouched.
        assert_eq!(&output[8..42], streaminfo_payload().as_slice());

        // Read back through the production reader — proves the bytes on
        // "disk" actually decode to the new tags, not just that we built
        // some in-memory struct correctly.
        let read_source = MemorySource::from_vec(output.clone());
        let parsed = FlacMetadataReader::read(read_source)
            .await
            .expect("read back should succeed");
        assert_eq!(parsed.get_text("TITLE"), Some("New Title"));
        assert_eq!(parsed.get_text("ARTIST"), Some("New Artist"));

        // Audio frame bytes must survive the metadata rewrite unmodified.
        assert!(
            output.ends_with(&audio_data[..]),
            "audio frame data must be preserved byte-for-byte"
        );
    }

    #[tokio::test]
    async fn test_flac_write_adds_vorbis_comment_when_absent() {
        // A file with only STREAMINFO (no pre-existing comment block).
        let audio_data = vec![0x99u8; 64];
        let initial = build_minimal_flac(None, &audio_data);

        let mut source = writable_flac_source(&initial).await;

        let mut tags = TagMap::new();
        tags.set("ALBUM", "Brand New");

        FlacMetadataWriter::write(&mut source, &tags)
            .await
            .expect("write should succeed");

        let output = source.written_data().to_vec();
        let read_source = MemorySource::from_vec(output.clone());
        let parsed = FlacMetadataReader::read(read_source)
            .await
            .expect("read back should succeed");
        assert_eq!(parsed.get_text("ALBUM"), Some("Brand New"));
        assert!(output.ends_with(&audio_data[..]));
    }

    #[tokio::test]
    async fn test_flac_write_exactly_one_last_block_flag() {
        // Structural sanity: after the rewrite, exactly one metadata block
        // may carry the last-metadata-block flag, and it must be the one
        // immediately preceding the audio data.
        let mut tags = TagMap::new();
        tags.set("TITLE", "Flag Check");
        let audio_data = vec![0x42u8; 32];
        let initial = build_minimal_flac(Some(&tags), &audio_data);
        let mut source = writable_flac_source(&initial).await;

        FlacMetadataWriter::write(&mut source, &tags)
            .await
            .expect("write should succeed");
        let output = source.written_data().to_vec();

        // Walk metadata blocks counting last-flags, and confirm the walk
        // ends exactly at the start of the (preserved) audio data.
        let mut cursor = 4usize; // past "fLaC"
        let mut last_flag_count = 0usize;
        loop {
            let header = output[cursor];
            let is_last = header & 0x80 != 0;
            let length = u32::from_be_bytes([
                0,
                output[cursor + 1],
                output[cursor + 2],
                output[cursor + 3],
            ]) as usize;
            if is_last {
                last_flag_count += 1;
            }
            cursor += 4 + length;
            if is_last {
                break;
            }
        }
        assert_eq!(
            last_flag_count, 1,
            "exactly one block may carry the last-metadata-block flag"
        );
        assert_eq!(&output[cursor..], audio_data.as_slice());
    }

    #[tokio::test]
    async fn test_flac_write_rejects_non_writable_source() {
        let mut tags = TagMap::new();
        tags.set("TITLE", "x");
        let initial = build_minimal_flac(Some(&tags), &[0u8; 16]);
        // MemorySource::from_vec is NOT writable.
        let mut source = MemorySource::from_vec(initial);
        let new_tags = TagMap::new();
        let result = FlacMetadataWriter::write(&mut source, &new_tags).await;
        assert!(result.is_err(), "non-writable source must be rejected");
    }

    #[tokio::test]
    async fn test_flac_write_rejects_missing_streaminfo() {
        // A "FLAC" file whose first block is a Vorbis comment, not
        // STREAMINFO — not conformant, must not be silently rewritten.
        let mut out = Vec::new();
        out.extend_from_slice(b"fLaC");
        let comments = VorbisComments::with_vendor("Bad");
        out.extend(flac_block(
            BlockType::VorbisComment,
            true,
            &comments.encode(),
        ));

        let mut source = writable_flac_source(&out).await;
        let tags = TagMap::new();
        let result = FlacMetadataWriter::write(&mut source, &tags).await;
        assert!(
            result.is_err(),
            "a FLAC stream missing a leading STREAMINFO block must be rejected"
        );
    }

    #[tokio::test]
    async fn test_flac_write_shrink_no_stale_bytes_on_file() {
        // Critical padding test, mirroring
        // `test_mkv_write_shrink_no_stale_bytes_on_file`: a FileSource is NOT
        // truncated on write, so replacing a large Vorbis comment block with
        // a tiny one must NOT leave stale trailing bytes from the old,
        // longer file. The writer must pad with a PADDING block so the file
        // never shrinks and metadata/audio stay self-consistent.
        use oximedia_io::source::FileSource;

        let mut path = std::env::temp_dir();
        path.push(format!(
            "oximedia_flac_writer_shrink_test_{}.flac",
            std::process::id()
        ));

        let mut big_tags = TagMap::new();
        for i in 0..40 {
            big_tags.set(
                format!("CUSTOM_FIELD_{i:03}"),
                format!("a fairly long stale value for field {i} ........................"),
            );
        }
        let audio_data = vec![0x7Eu8; 256];
        let initial = build_minimal_flac(Some(&big_tags), &audio_data);
        let initial_len = initial.len();
        tokio::fs::write(&path, &initial)
            .await
            .expect("temp file write should succeed");

        {
            let file = tokio::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .await
                .expect("temp file open should succeed");
            let mut source = FileSource::new_writable(file)
                .await
                .expect("writable FileSource should be created");

            let mut tiny_tags = TagMap::new();
            tiny_tags.set("TITLE", "tiny");
            FlacMetadataWriter::write(&mut source, &tiny_tags)
                .await
                .expect("shrink write should succeed");
        }

        let on_disk = tokio::fs::read(&path)
            .await
            .expect("temp file read should succeed");
        assert!(
            on_disk.len() >= initial_len,
            "padded file ({}) must not be shorter than original ({initial_len})",
            on_disk.len()
        );

        // Audio data must still be found byte-for-byte at the tail.
        assert!(
            on_disk.ends_with(&audio_data[..]),
            "audio data must survive a shrinking metadata rewrite"
        );

        // And the tags must read back as exactly the tiny replacement.
        let read_source = MemorySource::from_vec(on_disk.clone());
        let parsed = FlacMetadataReader::read(read_source)
            .await
            .expect("read back should succeed");
        assert_eq!(parsed.get_text("TITLE"), Some("tiny"));
        assert_eq!(parsed.len(), 1, "only the replacement tag must remain");

        let _ = tokio::fs::remove_file(&path).await;
    }
}

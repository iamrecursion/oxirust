//! Compressed MusicXML (`.mxl`) container reading.
//!
//! An `.mxl` file is a ZIP archive. `META-INF/container.xml` names the main
//! score in its first `<rootfile full-path="...">`; archives without a
//! container fall back to the first non-`META-INF/` `.musicxml` / `.xml`
//! entry. The score XML is then handed to the regular MusicXML parser.
//!
//! Everything happens in memory with `oxiarc-archive` / `oxiarc-deflate`
//! (Pure Rust, COOLJAPAN OxiARC), and the archive is treated as hostile:
//!
//! - the archive, each extracted entry, and the total extracted size are
//!   capped ([`MAX_MXL_ARCHIVE_BYTES`], [`MAX_MXL_ENTRY_BYTES`],
//!   [`MAX_MXL_TOTAL_BYTES`]), as is the entry count ([`MAX_MXL_ENTRIES`]);
//! - an entry's declared size must not exceed [`MAX_MXL_COMPRESSION_RATIO`]
//!   times its compressed size;
//! - deflate data is inflated into a buffer of exactly the declared size, so
//!   a stream that decodes to more (a lying header / zip bomb) is rejected
//!   without ever allocating more than that; the CRC-32 is verified;
//! - absolute paths, drive prefixes and `..` segments in entry names or in the
//!   container's rootfile path are rejected, as are encrypted entries.
//!
//! Only the container and the score entry are decompressed.

use oxiarc_archive::ZipReader;
use oxiarc_core::{CompressionMethod, Crc32, Entry, EntryType};
use std::io::Cursor;

/// Largest `.mxl` archive accepted, in bytes.
pub const MAX_MXL_ARCHIVE_BYTES: usize = 32 * 1024 * 1024;
/// Largest uncompressed size of a single extracted entry, in bytes.
pub const MAX_MXL_ENTRY_BYTES: u64 = 16 * 1024 * 1024;
/// Largest total uncompressed size extracted from one archive, in bytes.
pub const MAX_MXL_TOTAL_BYTES: u64 = 24 * 1024 * 1024;
/// Largest number of entries an archive may list.
pub const MAX_MXL_ENTRIES: usize = 1024;
/// Largest accepted ratio of declared uncompressed to compressed size.
/// Real MusicXML compresses roughly 5-30x.
pub const MAX_MXL_COMPRESSION_RATIO: u64 = 200;

const CONTAINER_PATH: &str = "META-INF/container.xml";

fn format_error(message: impl Into<String>) -> crate::Error {
    crate::Error::Format(message.into())
}

/// Whether an archive path is a plain relative path (no absolute path, drive
/// prefix, `..` segment or NUL).
fn is_safe_entry_path(path: &str) -> bool {
    !path.is_empty()
        && !path.starts_with('/')
        && !path.starts_with('\\')
        && !path.contains('\0')
        && !path.split(['/', '\\']).any(|segment| segment == "..")
        && !path
            .split(['/', '\\'])
            .next()
            .is_some_and(|first| first.len() == 2 && first.ends_with(':'))
}

/// Extract the score XML from an `.mxl` archive.
pub(crate) fn extract_score_xml(data: &[u8]) -> crate::Result<String> {
    if data.len() > MAX_MXL_ARCHIVE_BYTES {
        return Err(format_error(format!(
            ".mxl archive is {} bytes, larger than the {MAX_MXL_ARCHIVE_BYTES}-byte limit",
            data.len()
        )));
    }

    let mut reader = ZipReader::new(Cursor::new(data))
        .map_err(|e| format_error(format!("invalid .mxl archive: {e}")))?;
    let entries: Vec<Entry> = reader.entries().to_vec();
    if entries.len() > MAX_MXL_ENTRIES {
        return Err(format_error(format!(
            ".mxl archive lists {} entries, more than the {MAX_MXL_ENTRIES}-entry limit",
            entries.len()
        )));
    }
    if let Some(bad) = entries
        .iter()
        .find(|entry| !is_safe_entry_path(&entry.name))
    {
        return Err(format_error(format!(
            ".mxl archive contains an unsafe entry path {:?}",
            bad.name
        )));
    }

    let mut budget = MAX_MXL_TOTAL_BYTES;
    let score_name = match entries.iter().find(|entry| entry.name == CONTAINER_PATH) {
        Some(container) => {
            let xml = read_entry_text(&mut reader, container, &mut budget)?;
            let path = rootfile_path(&xml).ok_or_else(|| {
                format_error("META-INF/container.xml names no <rootfile full-path=...>")
            })?;
            if !is_safe_entry_path(&path) {
                return Err(format_error(format!(
                    "container.xml rootfile path {path:?} is not a safe relative path"
                )));
            }
            path
        }
        None => entries
            .iter()
            .find(|entry| is_score_candidate(entry))
            .map(|entry| entry.name.clone())
            .ok_or_else(|| {
                format_error(".mxl archive has no container.xml and no .musicxml/.xml score")
            })?,
    };

    let score = entries
        .iter()
        .find(|entry| entry.name == score_name && entry.entry_type != EntryType::Directory)
        .ok_or_else(|| {
            format_error(format!(
                "score {score_name:?} named in META-INF/container.xml is missing from the archive"
            ))
        })?;
    read_entry_text(&mut reader, score, &mut budget)
}

/// A score entry for the no-container fallback: a file outside `META-INF/`
/// with a `.musicxml` or `.xml` extension.
fn is_score_candidate(entry: &Entry) -> bool {
    let lower = entry.name.to_ascii_lowercase();
    entry.entry_type != EntryType::Directory
        && !lower.starts_with("meta-inf/")
        && (lower.ends_with(".musicxml") || lower.ends_with(".xml"))
}

/// The `full-path` of the first `<rootfile>` in a container document.
fn rootfile_path(container_xml: &str) -> Option<String> {
    let mut rest = container_xml;
    while let Some(start) = rest.find("<rootfile") {
        let after = &rest[start + "<rootfile".len()..];
        let end = after.find('>')?;
        let tag = &after[..end];
        // `<rootfiles>` also starts with `<rootfile`; only the exact element counts.
        if tag.starts_with(|c: char| c.is_whitespace() || c == '/') {
            if let Some(path) = attribute(tag, "full-path") {
                return Some(path);
            }
        }
        rest = &after[end..];
    }
    None
}

/// Value of `name="..."` (or single-quoted) inside a start tag.
fn attribute(tag: &str, name: &str) -> Option<String> {
    let mut rest = tag;
    while let Some(position) = rest.find(name) {
        let preceded_ok = rest[..position]
            .chars()
            .last()
            .is_some_and(char::is_whitespace);
        let after = rest[position + name.len()..].trim_start();
        if preceded_ok {
            if let Some(value) = after.strip_prefix('=') {
                let value = value.trim_start();
                let quote = value.chars().next().filter(|c| *c == '"' || *c == '\'')?;
                let inner = &value[1..];
                let close = inner.find(quote)?;
                return Some(inner[..close].to_string());
            }
        }
        rest = &rest[position + name.len()..];
    }
    None
}

/// Decompress one entry within the size budget and decode it as UTF-8.
fn read_entry_text(
    reader: &mut ZipReader<Cursor<&[u8]>>,
    entry: &Entry,
    budget: &mut u64,
) -> crate::Result<String> {
    let bytes = read_entry(reader, entry, budget)?;
    let text = String::from_utf8(bytes)
        .map_err(|_| format_error(format!("{:?} in .mxl archive is not UTF-8", entry.name)))?;
    Ok(text.trim_start_matches('\u{feff}').to_string())
}

fn read_entry(
    reader: &mut ZipReader<Cursor<&[u8]>>,
    entry: &Entry,
    budget: &mut u64,
) -> crate::Result<Vec<u8>> {
    let name = &entry.name;
    if ZipReader::<Cursor<&[u8]>>::is_encrypted(entry) {
        return Err(format_error(format!(
            "{name:?} in .mxl archive is encrypted"
        )));
    }
    if entry.size > MAX_MXL_ENTRY_BYTES {
        return Err(format_error(format!(
            "{name:?} in .mxl archive declares {} bytes, more than the {MAX_MXL_ENTRY_BYTES}-byte limit",
            entry.size
        )));
    }
    if entry.size > *budget {
        return Err(format_error(format!(
            "{name:?} would exceed the {MAX_MXL_TOTAL_BYTES}-byte total extraction limit"
        )));
    }
    if entry.size
        > entry
            .compressed_size
            .saturating_mul(MAX_MXL_COMPRESSION_RATIO)
    {
        return Err(format_error(format!(
            "{name:?} in .mxl archive has a suspicious compression ratio ({} -> {} bytes)",
            entry.compressed_size, entry.size
        )));
    }

    let raw = reader
        .extract_raw(entry)
        .map_err(|e| format_error(format!("cannot read {name:?} from .mxl archive: {e}")))?;
    let declared = usize::try_from(entry.size)
        .map_err(|_| format_error(format!("{name:?} is too large for this platform")))?;

    let data = match entry.method {
        CompressionMethod::Stored => {
            if raw.len() != declared {
                return Err(format_error(format!(
                    "{name:?}: stored size {} does not match declared size {declared}",
                    raw.len()
                )));
            }
            raw
        }
        CompressionMethod::Deflate => {
            // Inflate into exactly the declared size: a stream that decodes
            // to more is rejected without allocating beyond it.
            let mut out = vec![0_u8; declared];
            let written = oxiarc_deflate::inflate_into(&raw, &mut out).map_err(|e| {
                format_error(format!(
                    "{name:?} in .mxl archive does not inflate to its declared {declared} bytes: {e}"
                ))
            })?;
            if written != declared {
                return Err(format_error(format!(
                    "{name:?} inflated to {written} bytes, declared {declared}"
                )));
            }
            out
        }
        other => {
            return Err(format_error(format!(
                "{name:?} uses unsupported compression method {other} (only stored and deflate)"
            )));
        }
    };

    if let Some(expected) = entry.crc32 {
        let actual = Crc32::compute(&data);
        if actual != expected {
            return Err(format_error(format!(
                "{name:?} in .mxl archive fails its CRC-32 check"
            )));
        }
    }

    *budget -= entry.size;
    Ok(data)
}

#[cfg(test)]
#[path = "formats_mxl_tests.rs"]
mod tests;

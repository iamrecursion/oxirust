//! Memory-mapped file I/O for large batch input files.
//!
//! For files larger than [`MMAP_THRESHOLD`] (4 MiB), `MmapReader` uses
//! [`memmap2::Mmap`] for read-only memory mapping.  Smaller files fall back to
//! a standard `BufReader<File>` so the overhead of page-table manipulation is
//! avoided for short-lived reads.
//!
//! # Safety
//!
//! `memmap2::Mmap::map` requires one `unsafe` block.  The safety contract is:
//! the mapped file must not be truncated or otherwise modified by another
//! process while this mapping is live.  For batch input files this is a
//! reasonable assumption: inputs are immutable sources being read by the batch
//! engine; they are never written to during a job run.
//!
//! # Example
//! ```no_run
//! use std::io::Read;
//! use oximedia_batch::operations::mmap_reader::open_smart;
//!
//! let mut reader = open_smart(std::path::Path::new("/path/to/input.mkv")).unwrap();
//! let mut buf = Vec::new();
//! reader.read_to_end(&mut buf).unwrap();
//! ```

use crate::error::{BatchError, Result};
use std::fs::File;
use std::io::{BufReader, Cursor, Read};
use std::path::Path;

use memmap2::Mmap;

/// Files larger than this threshold (bytes) are memory-mapped; smaller files
/// use a buffered reader.
pub const MMAP_THRESHOLD: u64 = 4 * 1024 * 1024; // 4 MiB

// ---------------------------------------------------------------------------
// MmapReader
// ---------------------------------------------------------------------------

/// A reader that transparently uses memory-mapped I/O for large files.
///
/// Created via [`MmapReader::new`] or the convenience [`open_smart`] function.
/// Implements [`std::io::Read`] so it integrates with any existing code that
/// accepts a reader.
pub enum MmapReader {
    /// Memory-mapped reader for large files (>= MMAP_THRESHOLD).
    Mmap(Cursor<Mmap>),
    /// Buffered file reader for small files (< MMAP_THRESHOLD).
    Buf(BufReader<File>),
}

impl MmapReader {
    /// Open *path* and choose the appropriate reader strategy.
    ///
    /// Files whose byte length is ≥ [`MMAP_THRESHOLD`] are memory-mapped;
    /// all others use a `BufReader<File>`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened, stat-ed, or mapped.
    ///
    /// # Safety note
    ///
    /// The `memmap2::Mmap::map` call below requires an `unsafe` block because
    /// the OS could in principle truncate the file while the mapping is live,
    /// leading to a SIGBUS.  For batch input files this is acceptable: the
    /// batch engine never modifies files it is reading.
    pub fn new(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| {
            BatchError::IoError(std::io::Error::new(
                e.kind(),
                format!("cannot open '{}': {e}", path.display()),
            ))
        })?;

        let meta = file.metadata().map_err(|e| {
            BatchError::IoError(std::io::Error::new(
                e.kind(),
                format!("cannot stat '{}': {e}", path.display()),
            ))
        })?;

        if meta.len() >= MMAP_THRESHOLD {
            // SAFETY: The file is opened read-only and is not modified by the
            // batch engine while this mapping is live.  On platforms that
            // report a consistent view through mmap, this is sound.
            #[allow(unsafe_code)]
            let mmap = unsafe {
                Mmap::map(&file).map_err(|e| {
                    BatchError::IoError(std::io::Error::new(
                        e.kind(),
                        format!("cannot mmap '{}': {e}", path.display()),
                    ))
                })?
            };
            Ok(Self::Mmap(Cursor::new(mmap)))
        } else {
            Ok(Self::Buf(BufReader::new(file)))
        }
    }

    /// Returns `true` if this reader is using memory-mapped I/O.
    #[must_use]
    pub fn is_mmap(&self) -> bool {
        matches!(self, Self::Mmap(_))
    }
}

impl Read for MmapReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Self::Mmap(cursor) => cursor.read(buf),
            Self::Buf(reader) => reader.read(buf),
        }
    }
}

// ---------------------------------------------------------------------------
// open_smart — convenience function
// ---------------------------------------------------------------------------

/// Open *path* and return a boxed reader using the optimal I/O strategy.
///
/// This is a convenience wrapper around [`MmapReader::new`] that erases the
/// concrete type behind `Box<dyn Read>`.
///
/// # Errors
///
/// Returns an error if the file cannot be opened or mapped.
pub fn open_smart(path: &Path) -> Result<Box<dyn Read>> {
    let reader = MmapReader::new(path)?;
    Ok(Box::new(reader))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Read;

    fn write_temp(content: &[u8]) -> tempfile::NamedTempFile {
        let tmp = tempfile::NamedTempFile::new().expect("temp file");
        fs::write(tmp.path(), content).expect("write");
        tmp
    }

    #[test]
    fn test_mmap_reader_small_file_fallback() {
        // A file well below 4 MiB must use the BufReader path.
        let tmp = write_temp(b"small content");
        let reader = MmapReader::new(tmp.path()).expect("reader");
        assert!(!reader.is_mmap(), "small file should use BufReader path");
    }

    #[test]
    fn test_mmap_reader_content_matches_fs_read() {
        // Verify that the reader returns byte-identical content to fs::read().
        let payload = vec![0xABu8; 1024];
        let tmp = write_temp(&payload);

        let mut reader = MmapReader::new(tmp.path()).expect("reader");
        let mut got = Vec::new();
        reader.read_to_end(&mut got).expect("read_to_end");
        assert_eq!(got, payload);
    }

    #[test]
    fn test_open_smart_small_file() {
        let payload = b"hello world";
        let tmp = write_temp(payload);

        let mut reader = open_smart(tmp.path()).expect("open_smart");
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).expect("read");
        assert_eq!(buf, payload);
    }

    #[test]
    fn test_mmap_reader_large_file() {
        // Build a >4 MiB file to exercise the mmap path.
        let tmp_dir = std::env::temp_dir();
        let path = tmp_dir.join(format!(
            "oximedia_batch_mmap_test_{}.bin",
            std::process::id()
        ));

        // 5 MiB of deterministic data
        let data: Vec<u8> = (0..5 * 1024 * 1024u64).map(|i| (i % 251) as u8).collect();
        fs::write(&path, &data).expect("write large file");

        let reader_result = MmapReader::new(&path);
        // Attempt cleanup even if reader construction fails.
        let cleanup = || {
            let _ = fs::remove_file(&path);
        };

        let mut reader = match reader_result {
            Ok(r) => r,
            Err(e) => {
                cleanup();
                panic!("failed to open large file reader: {e}");
            }
        };

        assert!(reader.is_mmap(), "large file should use mmap path");

        let mut got = Vec::new();
        if let Err(e) = reader.read_to_end(&mut got) {
            cleanup();
            panic!("read_to_end failed: {e}");
        }

        cleanup();
        assert_eq!(got, data, "content mismatch for large file");
    }
}

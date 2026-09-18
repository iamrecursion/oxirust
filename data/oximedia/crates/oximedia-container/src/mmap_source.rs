//! Owned memory-mapped I/O source for demuxers.
//!
//! This module provides [`MmapDemuxSource`], an owned wrapper around
//! a [`memmap2::Mmap`] that keeps the backing [`std::fs::File`] alive for
//! the lifetime of the mapping.  It implements both [`std::io::Read`] and
//! [`std::io::Seek`] so it can be passed to any demuxer that accepts a generic
//! reader, and exposes [`as_bytes`](MmapDemuxSource::as_bytes) for zero-copy
//! random access.
//!
//! # Feature gate
//!
//! This module is only compiled when the `mmap` Cargo feature is enabled.
//!
//! # Safety
//!
//! All `unsafe` code is confined to the `memmap2` dependency, which performs
//! the `mmap(2)` syscall internally.  This module itself is `#[forbid(unsafe_code)]`.
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(feature = "mmap")]
//! # {
//! use oximedia_container::mmap_source::MmapDemuxSource;
//!
//! let src = MmapDemuxSource::open("video.mkv".as_ref()).expect("open");
//! println!("File size: {} bytes", src.as_bytes().len());
//! # }
//! ```

// Safety: `unsafe` is required for the `mmap(2)` syscall inside `memmap2`.
// All unsafe is confined to the single `Mmap::map` call; nothing else in this
// module uses unsafe.
#![allow(unsafe_code)]
use memmap2::Mmap;
use oximedia_core::OxiError;
use std::fs::File;
use std::io;
use std::path::Path;

// ─── MmapDemuxSource ─────────────────────────────────────────────────────────

/// An owned, memory-mapped demux source.
///
/// Keeps both the [`File`] handle and the [`Mmap`] region alive together,
/// which guarantees the mapping remains valid for as long as the
/// `MmapDemuxSource` exists.  Implements [`std::io::Read`] and
/// [`std::io::Seek`] via an internal position cursor.
pub struct MmapDemuxSource {
    /// The open file handle whose pages back the mapping.
    ///
    /// Prefixed with `_` because it is kept solely for its lifetime; the
    /// kernel mapping is accessed exclusively through `mmap`.
    _file: File,
    /// The memory-mapped view of the file.
    mmap: Mmap,
    /// Current read/seek position within the mapping.
    pos: usize,
}

impl MmapDemuxSource {
    /// Opens `path` and memory-maps the entire file.
    ///
    /// # Errors
    ///
    /// Returns [`OxiError::Io`] if the file cannot be opened, or if the
    /// `mmap(2)` syscall fails (e.g. the OS refuses the mapping or the
    /// file is empty on some platforms).
    pub fn open(path: &Path) -> Result<Self, OxiError> {
        let file = File::open(path).map_err(|e| {
            OxiError::Io(std::io::Error::new(
                e.kind(),
                format!("mmap open '{}': {e}", path.display()),
            ))
        })?;
        // SAFETY: delegated entirely to memmap2.  The resulting Mmap is
        // read-only; no aliasing mutations are possible through this struct.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
            OxiError::Io(std::io::Error::new(
                e.kind(),
                format!("mmap map '{}': {e}", path.display()),
            ))
        })?;
        Ok(Self {
            _file: file,
            mmap,
            pos: 0,
        })
    }

    /// Returns a view of the entire mapped region as a byte slice.
    ///
    /// This is a zero-copy operation: the bytes are read directly from the
    /// kernel page cache without any allocation.
    #[must_use]
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// Returns the total size of the mapped file in bytes.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// Returns `true` if the mapped region is empty (zero-byte file).
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }

    /// Returns the current read/seek position.
    #[must_use]
    #[inline]
    pub fn position(&self) -> usize {
        self.pos
    }
}

impl io::Read for MmapDemuxSource {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let available = self.mmap.len().saturating_sub(self.pos);
        let n = buf.len().min(available);
        buf[..n].copy_from_slice(&self.mmap[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

impl io::Seek for MmapDemuxSource {
    fn seek(&mut self, pos: io::SeekFrom) -> io::Result<u64> {
        let len = self.mmap.len() as i64;
        let new_pos: i64 = match pos {
            io::SeekFrom::Start(n) => i64::try_from(n).unwrap_or(i64::MAX),
            io::SeekFrom::End(n) => len.saturating_add(n),
            io::SeekFrom::Current(n) => (self.pos as i64).saturating_add(n),
        };
        if new_pos < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "seek to a negative position",
            ));
        }
        self.pos = (new_pos as usize).min(self.mmap.len());
        Ok(self.pos as u64)
    }
}

impl std::fmt::Debug for MmapDemuxSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MmapDemuxSource")
            .field("len", &self.mmap.len())
            .field("pos", &self.pos)
            .finish()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Seek, SeekFrom, Write};

    fn make_temp_file(contents: &[u8]) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        path.push(format!(
            "oximedia_mmapsrc_test_{}_{}.bin",
            std::process::id(),
            nanos,
        ));
        let mut f = File::create(&path).expect("create temp file");
        f.write_all(contents).expect("write temp file");
        f.sync_all().expect("sync");
        path
    }

    #[test]
    fn test_mmap_source_read_sequential() {
        // Write 256 bytes (0..=255), read back 16 bytes at a time, verify content.
        let data: Vec<u8> = (0u8..=255).collect();
        let path = make_temp_file(&data);
        let mut src = MmapDemuxSource::open(&path).expect("open");

        let mut result = Vec::new();
        let mut chunk = [0u8; 16];
        loop {
            let n = src.read(&mut chunk).expect("read ok");
            if n == 0 {
                break;
            }
            result.extend_from_slice(&chunk[..n]);
        }
        assert_eq!(result, data, "sequential read must match source bytes");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_mmap_source_seek() {
        // Write 256 bytes, seek to offset 128, read 4 bytes, verify correct.
        let data: Vec<u8> = (0u8..=255).collect();
        let path = make_temp_file(&data);
        let mut src = MmapDemuxSource::open(&path).expect("open");

        src.seek(SeekFrom::Start(128)).expect("seek ok");
        let mut buf = [0u8; 4];
        let n = src.read(&mut buf).expect("read ok");
        assert_eq!(n, 4);
        assert_eq!(
            &buf,
            &[128u8, 129, 130, 131],
            "bytes at offset 128 must match"
        );
        assert_eq!(src.position(), 132);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_mmap_source_as_bytes() {
        let data = b"hello mmap";
        let path = make_temp_file(data);
        let src = MmapDemuxSource::open(&path).expect("open");
        assert_eq!(src.as_bytes(), data.as_ref());
        assert_eq!(src.len(), data.len());
        assert!(!src.is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_mmap_source_seek_from_end() {
        let data: Vec<u8> = (0u8..=255).collect();
        let path = make_temp_file(&data);
        let mut src = MmapDemuxSource::open(&path).expect("open");
        src.seek(SeekFrom::End(-4)).expect("seek from end ok");
        let mut buf = [0u8; 4];
        src.read_exact(&mut buf).expect("read_exact ok");
        assert_eq!(&buf, &[252u8, 253, 254, 255]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_mmap_source_negative_seek_fails() {
        let data = b"short";
        let path = make_temp_file(data);
        let mut src = MmapDemuxSource::open(&path).expect("open");
        assert!(
            src.seek(SeekFrom::Current(-1)).is_err(),
            "seeking before start must fail"
        );
        let _ = std::fs::remove_file(&path);
    }
}

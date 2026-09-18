//! TAR archive creation with checksums

use crate::checksum::{ChecksumAlgorithm, ChecksumGenerator};
use crate::{Error, Result};
use oxiarc_archive::{TarReader, TarWriter};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// TAR archiver with checksum support
pub struct TarArchiver {
    algorithm: ChecksumAlgorithm,
}

impl Default for TarArchiver {
    fn default() -> Self {
        Self::new()
    }
}

impl TarArchiver {
    /// Create a new TAR archiver
    #[must_use]
    pub fn new() -> Self {
        Self {
            algorithm: ChecksumAlgorithm::Sha256,
        }
    }

    /// Set the checksum algorithm
    #[must_use]
    pub fn with_algorithm(mut self, algorithm: ChecksumAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Create a TAR archive from files
    ///
    /// # Errors
    ///
    /// Returns an error if archive creation fails
    pub fn create_archive(
        &self,
        output: &Path,
        files: &[(PathBuf, PathBuf)], // (source, path_in_archive)
    ) -> Result<String> {
        let file = File::create(output)?;
        let mut tar_writer = TarWriter::new(file);

        for (source, dest) in files {
            let dest_str = dest
                .to_str()
                .ok_or_else(|| Error::Archive("Non-UTF8 path in archive".to_string()))?;
            let file_data = std::fs::read(source)
                .map_err(|e| Error::Archive(format!("Failed to read file: {e}")))?;
            tar_writer
                .add_file(dest_str, &file_data)
                .map_err(|e| Error::Archive(format!("Failed to add file: {e}")))?;
        }

        tar_writer
            .finish()
            .map_err(|e| Error::Archive(format!("Failed to finalize archive: {e}")))?;

        // Generate checksum for the archive
        let generator = ChecksumGenerator::new().with_algorithms(vec![self.algorithm]);
        let checksum = generator.generate_file(output)?;

        checksum
            .checksums
            .get(&self.algorithm)
            .cloned()
            .ok_or_else(|| Error::Metadata("Checksum generation failed".to_string()))
    }

    /// Create archive from a directory
    ///
    /// # Errors
    ///
    /// Returns an error if archive creation fails
    pub fn create_from_directory(&self, output: &Path, dir: &Path) -> Result<String> {
        let files: Vec<(PathBuf, PathBuf)> = walkdir::WalkDir::new(dir)
            .follow_links(false)
            .into_iter()
            .filter_map(std::result::Result::ok)
            .filter(|e| e.file_type().is_file())
            .filter_map(|e| {
                let path = e.path().to_path_buf();
                let relative = path.strip_prefix(dir).ok()?.to_path_buf();
                Some((path, relative))
            })
            .collect();

        self.create_archive(output, &files)
    }

    /// Extract a TAR archive
    ///
    /// # Errors
    ///
    /// Returns an error if extraction fails
    pub fn extract_archive(archive_path: &Path, output_dir: &Path) -> Result<()> {
        let file = File::open(archive_path)?;
        // TarReader requires Read+Seek, so read file into memory and use Cursor
        let mut file_data = Vec::new();
        {
            let mut f = file;
            f.read_to_end(&mut file_data)?;
        }
        let cursor = std::io::Cursor::new(file_data);
        let mut tar_reader = TarReader::new(cursor)
            .map_err(|e| Error::Archive(format!("Failed to read archive: {e}")))?;

        let entries = tar_reader.entries().to_vec();
        for entry in &entries {
            let target_path = output_dir.join(&entry.name);
            if entry.is_dir() {
                std::fs::create_dir_all(&target_path)?;
            } else if entry.is_file() {
                if let Some(parent) = target_path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let data = tar_reader
                    .extract_to_vec(entry)
                    .map_err(|e| Error::Archive(format!("Failed to extract entry: {e}")))?;
                std::fs::write(&target_path, &data)?;
            }
        }

        Ok(())
    }

    /// List contents of a TAR archive
    ///
    /// # Errors
    ///
    /// Returns an error if the archive cannot be read
    pub fn list_contents(archive_path: &Path) -> Result<Vec<PathBuf>> {
        let file = File::open(archive_path)?;
        let mut file_data = Vec::new();
        {
            let mut f = file;
            f.read_to_end(&mut file_data)?;
        }
        let cursor = std::io::Cursor::new(file_data);
        let tar_reader = TarReader::new(cursor)
            .map_err(|e| Error::Archive(format!("Failed to read archive: {e}")))?;

        let contents = tar_reader
            .entries()
            .iter()
            .map(|e| PathBuf::from(&e.name))
            .collect();

        Ok(contents)
    }

    /// Create archive with checksum file
    ///
    /// # Errors
    ///
    /// Returns an error if archive creation fails
    pub fn create_with_checksum_file(
        &self,
        output: &Path,
        files: &[(PathBuf, PathBuf)],
    ) -> Result<()> {
        let checksum = self.create_archive(output, files)?;

        let checksum_path = output.with_extension("tar.sha256");
        let mut checksum_file = File::create(checksum_path)?;
        writeln!(
            checksum_file,
            "{}  {}",
            checksum,
            output.file_name().unwrap_or_default().to_string_lossy()
        )?;

        Ok(())
    }
}

// ── Streaming TAR writer ──────────────────────────────────────────────────────

/// Streaming TAR writer that flushes each entry immediately instead of
/// buffering the entire archive in memory.
///
/// This is suitable for large archives where holding all data in RAM is
/// undesirable. The output is a valid POSIX ustar archive.
pub struct StreamingTarWriter<W: Write> {
    inner: W,
    bytes_written: u64,
}

impl<W: Write> StreamingTarWriter<W> {
    /// Creates a new [`StreamingTarWriter`] wrapping `writer`.
    pub fn new(writer: W) -> Self {
        Self {
            inner: writer,
            bytes_written: 0,
        }
    }

    /// Appends a single file entry (header + data) to the archive immediately.
    ///
    /// # Errors
    ///
    /// Returns an error if writing to the underlying writer fails or if the
    /// entry name exceeds 99 bytes (POSIX ustar limit).
    pub fn append_entry(&mut self, name: &str, data: &[u8]) -> Result<()> {
        let header = build_tar_header(name, data.len() as u64)?;
        self.inner.write_all(&header)?;
        self.inner.write_all(data)?;

        // Pad to 512-byte block boundary
        let padding = (512 - (data.len() % 512)) % 512;
        if padding > 0 {
            let zeros = vec![0u8; padding];
            self.inner.write_all(&zeros)?;
        }

        self.bytes_written += 512 + data.len() as u64 + padding as u64;
        Ok(())
    }

    /// Writes the end-of-archive marker (two consecutive 512-byte zero blocks).
    ///
    /// Must be called after all entries have been appended to produce a valid
    /// TAR archive.
    ///
    /// # Errors
    ///
    /// Returns an error if writing to the underlying writer fails.
    pub fn finish(&mut self) -> Result<()> {
        self.inner.write_all(&[0u8; 1024])?;
        self.bytes_written += 1024;
        Ok(())
    }

    /// Returns the total number of bytes written so far (including headers,
    /// data, padding, and the end-of-archive blocks if `finish` was called).
    #[must_use]
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }
}

/// Builds a 512-byte POSIX ustar header for a regular file.
///
/// # Errors
///
/// Returns an error if `name` exceeds 99 bytes (ustar name field limit).
fn build_tar_header(name: &str, size: u64) -> Result<[u8; 512]> {
    let name_bytes = name.as_bytes();
    if name_bytes.len() > 99 {
        return Err(Error::Archive(format!(
            "TAR entry name exceeds 99 bytes: {name}"
        )));
    }

    let mut header = [0u8; 512];

    // name field: bytes 0..100
    header[..name_bytes.len()].copy_from_slice(name_bytes);

    // mode field: bytes 100..108
    header[100..108].copy_from_slice(b"0000644\0");

    // uid field: bytes 108..116
    header[108..116].copy_from_slice(b"0000000\0");

    // gid field: bytes 116..124
    header[116..124].copy_from_slice(b"0000000\0");

    // size field: bytes 124..136 (11 octal digits + NUL)
    let size_str = format!("{size:011o}\0");
    header[124..136].copy_from_slice(size_str.as_bytes());

    // mtime field: bytes 136..148 (11 octal digits + NUL, set to 0)
    header[136..148].copy_from_slice(b"00000000000\0");

    // checksum placeholder: bytes 148..156 (8 spaces, overwritten below)
    header[148..156].copy_from_slice(b"        ");

    // typeflag: byte 156 — '0' = regular file
    header[156] = b'0';

    // magic + version: bytes 257..265 — "ustar  \0" (GNU variant used by most tools)
    header[257..262].copy_from_slice(b"ustar");
    header[262] = b' ';
    header[263] = b' ';
    header[264] = b'\0';

    // Compute unsigned checksum (sum of all bytes, treating checksum field as spaces)
    let sum: u32 = header.iter().map(|&b| b as u32).sum();
    let cksum = format!("{sum:06o}\0 ");
    header[148..156].copy_from_slice(cksum.as_bytes());

    Ok(header)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use tempfile::{NamedTempFile, TempDir};

    #[test]
    fn test_create_tar_archive() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let archive_path = temp_dir.path().join("test.tar");

        let mut file1 = NamedTempFile::new().expect("operation should succeed");
        let mut file2 = NamedTempFile::new().expect("operation should succeed");
        file1
            .write_all(b"File 1 content")
            .expect("operation should succeed");
        file2
            .write_all(b"File 2 content")
            .expect("operation should succeed");
        file1.flush().expect("operation should succeed");
        file2.flush().expect("operation should succeed");

        let files = vec![
            (file1.path().to_path_buf(), PathBuf::from("file1.txt")),
            (file2.path().to_path_buf(), PathBuf::from("file2.txt")),
        ];

        let archiver = TarArchiver::new();
        let checksum = archiver
            .create_archive(&archive_path, &files)
            .expect("operation should succeed");

        assert!(!checksum.is_empty());
        assert!(archive_path.exists());
    }

    #[test]
    fn test_extract_tar_archive() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let archive_path = temp_dir.path().join("test.tar");
        let extract_dir = temp_dir.path().join("extracted");

        let mut test_file = NamedTempFile::new().expect("operation should succeed");
        test_file
            .write_all(b"Extract test")
            .expect("operation should succeed");
        test_file.flush().expect("operation should succeed");

        let files = vec![(test_file.path().to_path_buf(), PathBuf::from("test.txt"))];

        let archiver = TarArchiver::new();
        archiver
            .create_archive(&archive_path, &files)
            .expect("operation should succeed");

        TarArchiver::extract_archive(&archive_path, &extract_dir)
            .expect("operation should succeed");

        assert!(extract_dir.join("test.txt").exists());
    }

    #[test]
    fn test_list_tar_contents() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let archive_path = temp_dir.path().join("test.tar");

        let mut test_file = NamedTempFile::new().expect("operation should succeed");
        test_file
            .write_all(b"List test")
            .expect("operation should succeed");
        test_file.flush().expect("operation should succeed");

        let files = vec![(test_file.path().to_path_buf(), PathBuf::from("listed.txt"))];

        let archiver = TarArchiver::new();
        archiver
            .create_archive(&archive_path, &files)
            .expect("operation should succeed");

        let contents = TarArchiver::list_contents(&archive_path).expect("operation should succeed");
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0], PathBuf::from("listed.txt"));
    }

    // ── StreamingTarWriter tests ──────────────────────────────────────────

    /// Verify that two entries written via [`StreamingTarWriter`] produce a
    /// byte stream that can be parsed back as a valid ustar archive.
    ///
    /// We inspect the raw bytes rather than relying on oxiarc-archive so the
    /// test remains self-contained and deterministic.
    #[test]
    fn test_streaming_tar_round_trip() {
        let entry1_data = b"hello from entry one";
        let entry2_data = b"second entry content here";

        let mut buf = Vec::<u8>::new();
        let mut writer = StreamingTarWriter::new(&mut buf);

        writer
            .append_entry("entry1.txt", entry1_data)
            .expect("append entry1 should succeed");
        writer
            .append_entry("entry2.txt", entry2_data)
            .expect("append entry2 should succeed");
        writer.finish().expect("finish should succeed");

        // ── Verify basic structural invariants ──────────────────────────────

        // Must be a multiple of 512 bytes
        assert_eq!(buf.len() % 512, 0, "TAR output must be 512-byte aligned");

        // Minimum size: 2 headers + 2 data blocks (each rounded up) + 2 EOA blocks
        assert!(buf.len() >= 4 * 512, "TAR output too short");

        // ── Check entry1 header ──────────────────────────────────────────────
        let hdr1 = &buf[..512];
        // Name field should start with "entry1.txt\0"
        assert_eq!(&hdr1[..10], b"entry1.txt", "entry1 name mismatch");
        assert_eq!(hdr1[10], 0, "entry1 name NUL terminator missing");
        // typeflag must be '0' (regular file)
        assert_eq!(hdr1[156], b'0', "entry1 typeflag must be regular file");
        // magic field "ustar"
        assert_eq!(&hdr1[257..262], b"ustar", "entry1 ustar magic mismatch");

        // Size field (octal ASCII, bytes 124..136): must decode to entry1_data.len()
        let size_str = std::str::from_utf8(&hdr1[124..136])
            .expect("size field must be valid UTF-8")
            .trim_matches('\0')
            .trim();
        let decoded_size = u64::from_str_radix(size_str, 8).expect("size must be valid octal");
        assert_eq!(
            decoded_size,
            entry1_data.len() as u64,
            "entry1 size mismatch"
        );

        // ── Check entry2 header ──────────────────────────────────────────────
        // entry1 occupies 1 header + 1 data block (20 bytes → pads to 512)
        let hdr2_offset = 512 + 512; // header + one 512-byte data block
        let hdr2 = &buf[hdr2_offset..hdr2_offset + 512];
        assert_eq!(&hdr2[..10], b"entry2.txt", "entry2 name mismatch");
        assert_eq!(hdr2[156], b'0', "entry2 typeflag must be regular file");

        // ── End-of-archive: last 1024 bytes must be all zeros ────────────────
        let eoa = &buf[buf.len() - 1024..];
        assert!(
            eoa.iter().all(|&b| b == 0),
            "end-of-archive marker must be 1024 zero bytes"
        );
    }

    /// Verify that [`StreamingTarWriter::bytes_written`] tracks the exact number
    /// of bytes flushed to the underlying writer.
    #[test]
    fn test_streaming_tar_bytes_written() {
        let data_a = b"aaaa"; // 4 bytes → pads to 512
        let data_b = b"bbbbbbbb"; // 8 bytes → pads to 512

        let mut buf = Vec::<u8>::new();
        let mut writer = StreamingTarWriter::new(&mut buf);

        assert_eq!(writer.bytes_written(), 0, "initial bytes_written must be 0");

        writer
            .append_entry("a.txt", data_a)
            .expect("append a should succeed");
        // 512 (header) + 512 (data padded) = 1024
        assert_eq!(
            writer.bytes_written(),
            1024,
            "after first entry bytes_written must be 1024"
        );

        writer
            .append_entry("b.txt", data_b)
            .expect("append b should succeed");
        // another 512 (header) + 512 (data padded) = 1024 more → total 2048
        assert_eq!(
            writer.bytes_written(),
            2048,
            "after second entry bytes_written must be 2048"
        );

        writer.finish().expect("finish should succeed");
        // finish writes 1024 EOA bytes → total 3072
        assert_eq!(
            writer.bytes_written(),
            3072,
            "after finish bytes_written must be 3072"
        );

        // Capture expected before the writer goes out of scope
        let expected = writer.bytes_written();
        // writer goes out of scope here, releasing the mutable borrow on buf
        {
            let _release = writer;
        }

        // The buffer length must match what bytes_written reported
        assert_eq!(
            buf.len() as u64,
            expected,
            "buf.len() must equal bytes_written()"
        );
    }

    #[test]
    fn test_create_from_directory() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let source_dir = temp_dir.path().join("source");
        fs::create_dir_all(&source_dir).expect("operation should succeed");

        let test_file = source_dir.join("test.txt");
        fs::write(&test_file, b"Directory archive test").expect("operation should succeed");

        let archive_path = temp_dir.path().join("dir.tar");
        let archiver = TarArchiver::new();
        let checksum = archiver
            .create_from_directory(&archive_path, &source_dir)
            .expect("operation should succeed");

        assert!(!checksum.is_empty());
        assert!(archive_path.exists());
    }
}

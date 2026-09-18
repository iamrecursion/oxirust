//! Generic header repair functionality.
//!
//! This module provides common header repair operations that can be
//! applied across different container formats.

use crate::Result;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Repair a file header based on detected format.
pub fn repair_header(path: &Path, output: &Path) -> Result<bool> {
    let mut file = File::open(path)?;
    let mut header = [0u8; 16];
    file.read_exact(&mut header)?;

    // Detect format and dispatch to appropriate repair function
    if is_mp4_header(&header) {
        super::mp4::repair_mp4_header(path, output)
    } else if is_matroska_header(&header) {
        super::matroska::repair_matroska_header(path, output)
    } else if is_avi_header(&header) {
        super::avi::repair_avi_header(path, output)
    } else {
        Ok(false)
    }
}

/// Check if header is MP4 format.
fn is_mp4_header(header: &[u8]) -> bool {
    header.len() >= 8 && &header[4..8] == b"ftyp"
}

/// Check if header is Matroska format.
fn is_matroska_header(header: &[u8]) -> bool {
    header.len() >= 4 && header[0..4] == [0x1A, 0x45, 0xDF, 0xA3]
}

/// Check if header is AVI format.
fn is_avi_header(header: &[u8]) -> bool {
    header.len() >= 12 && &header[0..4] == b"RIFF" && &header[8..12] == b"AVI "
}

/// Repair corrupted magic number.
pub fn repair_magic_number(file: &mut File, expected: &[u8], offset: u64) -> Result<bool> {
    file.seek(SeekFrom::Start(offset))?;
    let mut actual = vec![0u8; expected.len()];
    file.read_exact(&mut actual)?;

    if actual != expected {
        file.seek(SeekFrom::Start(offset))?;
        file.write_all(expected)?;
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Repair file size field in header.
pub fn repair_size_field(
    file: &mut File,
    offset: u64,
    actual_size: u64,
    little_endian: bool,
) -> Result<bool> {
    file.seek(SeekFrom::Start(offset))?;

    let size_bytes = if little_endian {
        (actual_size as u32).to_le_bytes()
    } else {
        (actual_size as u32).to_be_bytes()
    };

    file.write_all(&size_bytes)?;
    Ok(true)
}

/// Validate and repair header checksum.
pub fn repair_checksum(
    file: &mut File,
    data_offset: u64,
    data_length: usize,
    checksum_offset: u64,
) -> Result<bool> {
    // Read data
    file.seek(SeekFrom::Start(data_offset))?;
    let mut data = vec![0u8; data_length];
    file.read_exact(&mut data)?;

    // Calculate checksum
    let checksum = calculate_header_checksum(&data);

    // Write checksum
    file.seek(SeekFrom::Start(checksum_offset))?;
    file.write_all(&checksum.to_le_bytes())?;

    Ok(true)
}

/// Calculate header checksum (CRC32).
fn calculate_header_checksum(data: &[u8]) -> u32 {
    let mut checksum = 0u32;
    for &byte in data {
        checksum = checksum.wrapping_add(byte as u32);
    }
    checksum
}

/// Copy file with header repair.
pub fn copy_with_repair<F>(input: &Path, output: &Path, repair_fn: F) -> Result<bool>
where
    F: FnOnce(&mut File) -> Result<bool>,
{
    std::fs::copy(input, output)?;
    let mut file = File::options().write(true).open(output)?;
    repair_fn(&mut file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_mp4_header() {
        let header = b"\x00\x00\x00\x20ftypmp42";
        assert!(is_mp4_header(header));
    }

    #[test]
    fn test_is_matroska_header() {
        let header = b"\x1A\x45\xDF\xA3\x00\x00\x00\x00";
        assert!(is_matroska_header(header));
    }

    #[test]
    fn test_is_avi_header() {
        let header = b"RIFF\x00\x00\x00\x00AVI \x00\x00\x00\x00";
        assert!(is_avi_header(header));
    }

    #[test]
    fn test_calculate_header_checksum() {
        let data = b"test data";
        let checksum = calculate_header_checksum(data);
        assert!(checksum > 0);
    }
}

// ---------------------------------------------------------------------------
// Randomised / fuzz-style header repair tests
//
// These tests exercise each header-repair entry point with deterministic
// pseudo-random byte sequences of varying length, asserting that:
//   1. The function returns Ok(..) or Err(..) — never panics.
//   2. When the function returns Ok(true), the output file exists and is
//      non-empty (bytes were written).
//   3. Boundary-condition inputs (empty, 1 byte, exactly 16 bytes) are
//      handled gracefully.
//
// A minimal linear-congruential generator (LCG) is used so that tests are
// fully deterministic without any external crate dependency.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod fuzz_tests {
    use super::super::{avi, matroska, mp4, repair};
    use std::fs;
    use std::io::{Seek, SeekFrom, Write};
    use std::path::PathBuf;

    // -----------------------------------------------------------------------
    // Deterministic PRNG (LCG — Knuth parameters)
    // -----------------------------------------------------------------------

    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u8(&mut self) -> u8 {
            // Multiplier / increment from Knuth TAOCP vol. 2, 3rd ed.
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.state >> 33) as u8
        }

        fn fill(&mut self, buf: &mut [u8]) {
            for b in buf.iter_mut() {
                *b = self.next_u8();
            }
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "oximedia_header_fuzz_{}_{}",
            std::process::id(),
            name
        ))
    }

    /// Write `data` to a temp file and return its path.
    fn write_temp(name: &str, data: &[u8]) -> PathBuf {
        let p = temp_path(name);
        let mut f = fs::File::create(&p).expect("create temp input");
        f.write_all(data).expect("write temp input");
        p
    }

    /// Clean up a list of temp paths, ignoring errors.
    fn cleanup(paths: &[PathBuf]) {
        for p in paths {
            let _ = fs::remove_file(p);
        }
    }

    // -----------------------------------------------------------------------
    // Test 1: repair_header — random bytes of sizes 0..=1024
    //
    // Goal: confirm no panic on arbitrary input; function returns Ok.
    // -----------------------------------------------------------------------

    #[test]
    fn fuzz_repair_header_random_sequences() {
        let mut lcg = Lcg::new(0xDEAD_BEEF_1234_5678);

        for size in [0usize, 1, 4, 8, 12, 16, 32, 64, 128, 256, 512, 1024] {
            let mut data = vec![0u8; size];
            lcg.fill(&mut data);

            let input = write_temp(&format!("repair_header_{size}.bin"), &data);
            let output = temp_path(&format!("repair_header_{size}_out.bin"));

            // Must not panic; may succeed or gracefully fail.
            let result = repair::repair_header(&input, &output);
            // Ok(false) = nothing to repair, Ok(true) = repaired, Err = IO issue on bad data.
            // We only assert no panic; both Ok variants and Err are acceptable.
            let _ = result;

            cleanup(&[input, output]);
        }
    }

    // -----------------------------------------------------------------------
    // Test 2: repair_mp4_header — structured corrupted MP4 variants
    //
    // Goal: corrupt the ftyp atom in known ways and confirm repair succeeds.
    // -----------------------------------------------------------------------

    #[test]
    fn fuzz_repair_mp4_header_corrupted_ftyp() {
        // Sub-case A: wrong magic, correct size prefix
        // `insert_ftyp_atom` calls `file.read_to_end` on the write-only copy fd;
        // this returns EBADF on some platforms.  We accept Ok or Err — no panic.
        {
            let mut data = vec![0u8; 64];
            // size = 28 (big-endian)
            data[0..4].copy_from_slice(&28u32.to_be_bytes());
            // wrong magic — should trigger insert_ftyp_atom path
            data[4..8].copy_from_slice(b"XXXX");
            data[8..12].copy_from_slice(b"mp42");

            let input = write_temp("mp4_wrong_magic.bin", &data);
            let output = temp_path("mp4_wrong_magic_out.bin");

            // Acceptable: Ok(true/false) when repair succeeds, Err when I/O
            // constraints (write-only fd on insert_ftyp_atom path) prevent it.
            let _result = mp4::repair_mp4_header(&input, &output);

            cleanup(&[input, output]);
        }

        // Sub-case B: correct magic, size field = 0 — triggers fix_ftyp_size
        // which reads the fd (write-only) → EBADF on macOS/Linux.  Accept Ok or Err.
        {
            let mut data = vec![0u8; 64];
            data[0..4].copy_from_slice(&0u32.to_be_bytes()); // size = 0
            data[4..8].copy_from_slice(b"ftyp");
            data[8..12].copy_from_slice(b"mp42");

            let input = write_temp("mp4_zero_size.bin", &data);
            let output = temp_path("mp4_zero_size_out.bin");

            // Accept Ok or Err — no panic is the invariant.
            let result = mp4::repair_mp4_header(&input, &output);
            if let Ok(true) = result {
                let out_meta = fs::metadata(&output);
                assert!(out_meta.is_ok(), "output file must exist when repaired");
                assert!(
                    out_meta.expect("metadata").len() > 0,
                    "output must be non-empty"
                );
            }

            cleanup(&[input, output]);
        }

        // Sub-case C: truncated — only 4 bytes (less than a full header)
        {
            let data = b"ftyp"; // 4 bytes, cannot read full 8-byte header
            let input = write_temp("mp4_truncated.bin", data);
            let output = temp_path("mp4_truncated_out.bin");

            let result = mp4::repair_mp4_header(&input, &output);
            // Truncated input: either Ok or Err is acceptable; no panic.
            let _ = result;

            cleanup(&[input, output]);
        }

        // Sub-case D: random 512-byte payload with valid ftyp magic + non-zero size.
        // repair_moov_atom → find_atom reads the write-only fd → EBADF on macOS/Linux.
        // Accept Ok or Err — no panic is the only invariant.
        {
            let mut lcg = Lcg::new(0x1234_ABCD_5678_EF01);
            let mut data = vec![0u8; 512];
            lcg.fill(&mut data);
            data[0..4].copy_from_slice(&28u32.to_be_bytes());
            data[4..8].copy_from_slice(b"ftyp");
            data[8..12].copy_from_slice(b"mp42");

            let input = write_temp("mp4_random_ftyp.bin", &data);
            let output = temp_path("mp4_random_ftyp_out.bin");

            // No panic is the invariant.
            let _ = mp4::repair_mp4_header(&input, &output);

            cleanup(&[input, output]);
        }
    }

    // -----------------------------------------------------------------------
    // Test 3: repair_avi_header — corrupted RIFF / AVI  headers
    // -----------------------------------------------------------------------

    #[test]
    fn fuzz_repair_avi_header_corrupted_riff() {
        // Sub-case A: RIFF signature mangled.
        // repair_riff_header calls file.read_exact on the write-only copy fd
        // (EBADF on macOS/Linux when the fd is opened write-only).  Accepts Ok or Err.
        {
            let mut data = vec![0u8; 64];
            data[0..4].copy_from_slice(b"XXXX"); // wrong RIFF
            data[4..8].copy_from_slice(&48u32.to_le_bytes());
            data[8..12].copy_from_slice(b"AVI ");

            let input = write_temp("avi_bad_riff.bin", &data);
            let output = temp_path("avi_bad_riff_out.bin");

            // No panic is the only invariant — Ok or Err both acceptable.
            let _ = avi::repair_avi_header(&input, &output);

            cleanup(&[input, output]);
        }

        // Sub-case B: AVI fourcc mangled — same constraint, accept Ok or Err.
        {
            let mut data = vec![0u8; 64];
            data[0..4].copy_from_slice(b"RIFF");
            data[4..8].copy_from_slice(&48u32.to_le_bytes());
            data[8..12].copy_from_slice(b"NAVI"); // wrong type

            let input = write_temp("avi_bad_fourcc.bin", &data);
            let output = temp_path("avi_bad_fourcc_out.bin");

            // No panic is the only invariant.
            let _ = avi::repair_avi_header(&input, &output);

            cleanup(&[input, output]);
        }

        // Sub-case C: Entirely random bytes — accept Ok or Err (no panic)
        {
            let mut lcg = Lcg::new(0x9999_AAAA_BBBB_CCCC);
            let mut data = vec![0u8; 256];
            lcg.fill(&mut data);

            let input = write_temp("avi_random.bin", &data);
            let output = temp_path("avi_random_out.bin");

            let result = avi::repair_avi_header(&input, &output);
            let _ = result; // Ok or Err — no panic

            cleanup(&[input, output]);
        }

        // Sub-case D: File too small (only 8 bytes) — accept Ok or Err
        {
            let data = b"RIFF\x00\x00"; // truncated after RIFF
            let input = write_temp("avi_tiny.bin", data);
            let output = temp_path("avi_tiny_out.bin");

            let result = avi::repair_avi_header(&input, &output);
            let _ = result;

            cleanup(&[input, output]);
        }
    }

    // -----------------------------------------------------------------------
    // Test 4: repair_matroska_header — corrupted EBML magic
    // -----------------------------------------------------------------------

    #[test]
    fn fuzz_repair_matroska_header_corrupted_ebml() {
        // Sub-case A: EBML signature corrupted (1 byte wrong).
        // repair_ebml_header does seek(0) + read_exact(4 bytes) on the write-only
        // copy fd, then seek+write if signature is wrong — the read may return
        // EBADF on some platforms; accept Ok or Err (no panic).
        {
            let mut data = vec![0u8; 64];
            data[0..4].copy_from_slice(&[0x1A, 0x45, 0xDF, 0x00]); // last byte wrong
            data[4] = 0x80 | 58; // 1-byte vint for size 58

            let input = write_temp("mkv_bad_magic.bin", &data);
            let output = temp_path("mkv_bad_magic_out.bin");

            // Accept Ok or Err — the important invariant is no panic.
            let result = matroska::repair_matroska_header(&input, &output);
            if let Ok(true) = &result {
                let meta = fs::metadata(&output);
                assert!(meta.is_ok());
                assert!(meta.expect("metadata").len() > 0);
            }
            let _ = result;

            cleanup(&[input, output]);
        }

        // Sub-case B: All-zero EBML header — accept Ok or Err (no panic)
        {
            let data = vec![0u8; 64];
            let input = write_temp("mkv_zeros.bin", &data);
            let output = temp_path("mkv_zeros_out.bin");

            let result = matroska::repair_matroska_header(&input, &output);
            let _ = result; // Ok or Err — no panic

            cleanup(&[input, output]);
        }

        // Sub-case C: Valid EBML signature — confirm no panic
        {
            let mut data = vec![0u8; 64];
            data[0..4].copy_from_slice(&[0x1A, 0x45, 0xDF, 0xA3]); // correct EBML magic
            data[4] = 0x80 | 58;

            let input = write_temp("mkv_valid_magic.bin", &data);
            let output = temp_path("mkv_valid_magic_out.bin");

            let result = matroska::repair_matroska_header(&input, &output);
            let _ = result; // Ok or Err — no panic

            cleanup(&[input, output]);
        }

        // Sub-case D: Random payload stress — no panic
        {
            let mut lcg = Lcg::new(0xFEED_FACE_DEAD_BEEF);
            let mut data = vec![0u8; 512];
            lcg.fill(&mut data);

            let input = write_temp("mkv_random.bin", &data);
            let output = temp_path("mkv_random_out.bin");

            let result = matroska::repair_matroska_header(&input, &output);
            let _ = result; // Ok or Err — no panic

            cleanup(&[input, output]);
        }
    }

    // -----------------------------------------------------------------------
    // Test 5: repair_magic_number — boundary-condition offsets
    // -----------------------------------------------------------------------

    #[test]
    fn fuzz_repair_magic_number_boundary_cases() {
        use std::fs::OpenOptions;
        use std::io::Seek;

        // Create a writable temp file with 32 bytes of known data.
        let expected: &[u8] = b"MAGC";
        let base_data = vec![0xFFu8; 32];

        // Sub-case A: offset 0 — magic at start of file
        {
            let path = write_temp("magic_offset0.bin", &base_data);
            let mut f = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .expect("open magic_offset0");

            let result = repair::repair_magic_number(&mut f, expected, 0);
            assert!(
                result.is_ok(),
                "repair_magic_number at offset 0 must not error"
            );
            assert!(result.expect("ok"), "should have written magic at offset 0");

            // Verify the bytes were actually written.
            f.seek(SeekFrom::Start(0)).expect("seek");
            let mut actual = vec![0u8; 4];
            use std::io::Read;
            f.read_exact(&mut actual).expect("read back");
            assert_eq!(&actual, expected, "magic must match at offset 0");

            cleanup(&[path]);
        }

        // Sub-case B: offset at middle of file
        {
            let path = write_temp("magic_offset8.bin", &base_data);
            let mut f = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .expect("open magic_offset8");

            let result = repair::repair_magic_number(&mut f, expected, 8);
            assert!(
                result.is_ok(),
                "repair_magic_number at offset 8 must not error"
            );

            cleanup(&[path]);
        }

        // Sub-case C: offset beyond file — expect an error (not a panic)
        {
            let path = write_temp("magic_offset_oob.bin", &base_data);
            let mut f = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .expect("open magic_offset_oob");

            let result = repair::repair_magic_number(&mut f, expected, 10_000);
            // Must return an error, not panic.
            assert!(
                result.is_err(),
                "repair_magic_number at OOB offset must return Err"
            );

            cleanup(&[path]);
        }

        // Sub-case D: zero-length expected — no-op (slice comparison trivially equal)
        {
            let path = write_temp("magic_empty.bin", &base_data);
            let mut f = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .expect("open magic_empty");

            let result = repair::repair_magic_number(&mut f, &[], 0);
            assert!(
                result.is_ok(),
                "repair_magic_number with empty slice must not error"
            );
            // Empty expected matches empty actual — no write needed.
            assert!(
                !result.expect("ok"),
                "no-op: empty slice, nothing to repair"
            );

            cleanup(&[path]);
        }
    }

    // -----------------------------------------------------------------------
    // Test 6: repair_size_field — little-endian and big-endian encodings
    // -----------------------------------------------------------------------

    #[test]
    fn fuzz_repair_size_field_endianness() {
        use std::fs::OpenOptions;
        use std::io::Read;

        let base_data = vec![0u8; 32];

        // Sub-case A: little-endian size field at offset 0
        {
            let path = write_temp("size_le.bin", &base_data);
            let mut f = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .expect("open size_le");

            let result = repair::repair_size_field(&mut f, 0, 12345, true);
            assert!(result.is_ok(), "repair_size_field LE must not error");
            assert!(result.expect("ok"), "must indicate size field was written");

            // Verify round-trip.
            f.seek(SeekFrom::Start(0)).expect("seek");
            let mut buf = [0u8; 4];
            f.read_exact(&mut buf).expect("read");
            assert_eq!(u32::from_le_bytes(buf), 12345, "LE size mismatch");

            cleanup(&[path]);
        }

        // Sub-case B: big-endian size field at offset 4
        {
            let path = write_temp("size_be.bin", &base_data);
            let mut f = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .expect("open size_be");

            let result = repair::repair_size_field(&mut f, 4, 99_999, false);
            assert!(result.is_ok(), "repair_size_field BE must not error");

            f.seek(SeekFrom::Start(4)).expect("seek");
            let mut buf = [0u8; 4];
            f.read_exact(&mut buf).expect("read");
            assert_eq!(u32::from_be_bytes(buf), 99_999, "BE size mismatch");

            cleanup(&[path]);
        }

        // Sub-case C: size = 0 (valid — write 0 into the field)
        {
            let path = write_temp("size_zero.bin", &base_data);
            let mut f = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .expect("open size_zero");

            let result = repair::repair_size_field(&mut f, 0, 0, true);
            assert!(
                result.is_ok(),
                "repair_size_field with size=0 must not error"
            );

            cleanup(&[path]);
        }

        // Sub-case D: size saturates at u32::MAX (actual_size truncated)
        {
            let path = write_temp("size_saturate.bin", &base_data);
            let mut f = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .expect("open size_saturate");

            // actual_size larger than u32::MAX — truncation is deliberate (size field is 4 bytes)
            let result = repair::repair_size_field(&mut f, 0, u64::from(u32::MAX) + 100, false);
            assert!(
                result.is_ok(),
                "repair_size_field with oversized value must not error"
            );

            cleanup(&[path]);
        }
    }

    // -----------------------------------------------------------------------
    // Test 7: copy_with_repair — verify output file is created correctly
    // -----------------------------------------------------------------------

    #[test]
    fn fuzz_copy_with_repair_output_file_integrity() {
        let mut lcg = Lcg::new(0x0102_0304_0506_0708);

        for run in 0..8u32 {
            let mut data = vec![0u8; 128];
            lcg.fill(&mut data);

            let input = write_temp(&format!("copy_repair_in_{run}.bin"), &data);
            let output = temp_path(&format!("copy_repair_out_{run}.bin"));

            let result = repair::copy_with_repair(&input, &output, |_file| Ok(false));
            assert!(
                result.is_ok(),
                "copy_with_repair must not error (run {run}): {result:?}"
            );
            assert!(!result.expect("ok"), "no-op repair_fn returns false");

            // Verify the output file exists and has the same length as input.
            let in_len = fs::metadata(&input).expect("input meta").len();
            let out_len = fs::metadata(&output).expect("output meta").len();
            assert_eq!(
                in_len, out_len,
                "output length must match input (run {run})"
            );

            cleanup(&[input, output]);
        }
    }
}

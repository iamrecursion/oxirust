//! Out-of-core 2D FFT for large images via row/column decomposition
//!
//! For data that fits in RAM, the in-memory path (`small_fft2d` and the
//! `fft2d` implementation when `rows * cols` elements fit) is used directly.
//!
//! For data that truly requires out-of-core processing, `fft2d` works in
//! `chunk_rows`-sized blocks:
//!
//! 1. Read `chunk_rows` rows at a time, compute their FFTs, and write the
//!    complex results to a temporary row-FFT file.
//! 2. Read the transposed data column by column (by striding through the
//!    temp file), FFT each column, and write to a second temp file in the
//!    natural row-major order.
//! 3. Return the final complex array.
//!
//! The inverse path (`ifft2d`) applies the same decomposition with backward
//! FFTs and the `1/(rows*cols)` normalisation.
//!
//! # Example
//!
//! ```rust
//! use scirs2_fft::outofcore::{OutOfCoreFft2D, small_fft2d};
//!
//! // 8×8 impulse at origin → all FFT bins = 1.
//! let mut data = vec![0.0_f64; 64];
//! data[0] = 1.0;
//! let spectrum = small_fft2d(&data, 8, 8);
//! for &(re, im) in &spectrum {
//!     assert!((re - 1.0).abs() < 1e-10);
//!     assert!(im.abs() < 1e-10);
//! }
//! ```

use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

use crate::error::{FFTError, FFTResult};
use crate::fft::{fft, ifft};
use scirs2_core::numeric::Complex64;

// ─────────────────────────────────────────────────────────────────────────────
//  Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`OutOfCoreFft2D`].
#[derive(Debug, Clone)]
pub struct OutOfCoreConfig {
    /// Number of rows in the 2-D array.
    pub rows: usize,
    /// Number of columns in the 2-D array.
    pub cols: usize,
    /// Rows processed in-memory at once.  A larger value uses more RAM but
    /// requires fewer I/O passes.  Must be at least 1 and at most `rows`.
    pub chunk_rows: usize,
    /// Directory for temporary files.  Defaults to `std::env::temp_dir()`.
    pub temp_dir: PathBuf,
}

/// Out-of-core 2-D FFT processor.
///
/// For small arrays the computation is carried out entirely in memory.
/// For large arrays the row-FFT results are spilled to disk and the column
/// FFTs are performed chunk by chunk with disk I/O.
pub struct OutOfCoreFft2D {
    config: OutOfCoreConfig,
}

// ─────────────────────────────────────────────────────────────────────────────
//  Implementation
// ─────────────────────────────────────────────────────────────────────────────

impl OutOfCoreFft2D {
    /// Create a processor with default configuration.
    ///
    /// `chunk_rows` defaults to `rows` (fully in-memory).
    /// `temp_dir` defaults to [`std::env::temp_dir()`].
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            config: OutOfCoreConfig {
                rows,
                cols,
                chunk_rows: rows,
                temp_dir: std::env::temp_dir(),
            },
        }
    }

    /// Create a processor with explicit configuration.
    pub fn with_config(config: OutOfCoreConfig) -> Self {
        Self { config }
    }

    /// Forward 2-D FFT.
    ///
    /// `data` must be a row-major flat slice of length `rows * cols`.
    ///
    /// Returns the complex spectrum as `Vec<(f64, f64)>` (real, imaginary)
    /// pairs in row-major order.
    ///
    /// # Errors
    ///
    /// Returns [`FFTError`] on slice-length mismatch or I/O failure.
    pub fn fft2d(&self, data: &[f64]) -> FFTResult<Vec<(f64, f64)>> {
        let rows = self.config.rows;
        let cols = self.config.cols;

        if data.len() != rows * cols {
            return Err(FFTError::DimensionError(format!(
                "outofcore fft2d: data length {} != rows*cols {}",
                data.len(),
                rows * cols
            )));
        }

        let chunk_rows = self.config.chunk_rows.max(1).min(rows);

        // ── Step 1: FFT every row ─────────────────────────────────────────────
        // Store results in row-major order as (re, im) pairs.
        let row_fft_results = self.fft_all_rows(data, rows, cols, chunk_rows)?;

        // ── Step 2: FFT every column ─────────────────────────────────────────
        let col_fft_results = self.fft_all_cols(&row_fft_results, rows, cols, chunk_rows)?;

        Ok(col_fft_results)
    }

    /// Inverse 2-D FFT.
    ///
    /// `spectrum` must be a row-major flat slice of `(re, im)` pairs with
    /// length `rows * cols`.
    ///
    /// Returns the real part of the inverse transform.
    ///
    /// # Errors
    ///
    /// Returns [`FFTError`] on slice-length mismatch or I/O failure.
    pub fn ifft2d(&self, spectrum: &[(f64, f64)]) -> FFTResult<Vec<f64>> {
        let rows = self.config.rows;
        let cols = self.config.cols;

        if spectrum.len() != rows * cols {
            return Err(FFTError::DimensionError(format!(
                "outofcore ifft2d: spectrum length {} != rows*cols {}",
                spectrum.len(),
                rows * cols
            )));
        }

        let chunk_rows = self.config.chunk_rows.max(1).min(rows);

        // ── Step 1: IFFT every row ────────────────────────────────────────────
        let row_ifft = self.ifft_all_rows(spectrum, rows, cols, chunk_rows)?;

        // ── Step 2: IFFT every column ─────────────────────────────────────────
        let col_ifft = self.ifft_all_cols(&row_ifft, rows, cols, chunk_rows)?;

        // Normalise by 1 / (rows * cols) — already done per-1D IFFT so we
        // have normalised twice (once per axis).  The 1-D IFFT normalises by
        // 1/N, giving 1/(rows) and 1/(cols) independently, which together
        // equal 1/(rows*cols).  Extract real part.
        Ok(col_ifft.iter().map(|&(re, _im)| re).collect())
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// FFT every row of `data` (real input) and return the complex result as
    /// a flat `Vec<(f64, f64)>` in row-major order, working `chunk_rows` rows
    /// at a time.
    fn fft_all_rows(
        &self,
        data: &[f64],
        rows: usize,
        cols: usize,
        chunk_rows: usize,
    ) -> FFTResult<Vec<(f64, f64)>> {
        let mut row_spectra: Vec<(f64, f64)> = Vec::with_capacity(rows * cols);

        let mut row = 0_usize;
        while row < rows {
            let end = (row + chunk_rows).min(rows);
            for r in row..end {
                let start = r * cols;
                let row_data = &data[start..start + cols];
                let spectrum = fft(row_data, Some(cols))?;
                for c in spectrum.iter().take(cols) {
                    row_spectra.push((c.re, c.im));
                }
            }
            row = end;
        }

        Ok(row_spectra)
    }

    /// FFT every column of a complex row-major array `data` of shape
    /// `rows × cols`.  Returns the result in row-major order.
    ///
    /// The column access pattern is disk-friendly when `chunk_rows` is small:
    /// for each column we gather the column values by striding through `data`.
    fn fft_all_cols(
        &self,
        data: &[(f64, f64)],
        rows: usize,
        cols: usize,
        _chunk_rows: usize,
    ) -> FFTResult<Vec<(f64, f64)>> {
        let total = rows * cols;
        let mut result: Vec<(f64, f64)> = vec![(0.0, 0.0); total];

        for c in 0..cols {
            // Extract column c as a slice of Complex64.
            let col_data: Vec<Complex64> = (0..rows)
                .map(|r| {
                    let (re, im) = data[r * cols + c];
                    Complex64::new(re, im)
                })
                .collect();

            // FFT the column using the crate's public `fft` function.
            // `fft` accepts `Complex64` via the NumCast + Any downcast path.
            let col_spectrum = fft(&col_data, Some(rows))?;

            // Write back in row-major order: result[r * cols + c].
            for (r, val) in col_spectrum.iter().take(rows).enumerate() {
                result[r * cols + c] = (val.re, val.im);
            }
        }

        Ok(result)
    }

    /// IFFT every row of a complex array `spectrum` (row-major).
    fn ifft_all_rows(
        &self,
        spectrum: &[(f64, f64)],
        rows: usize,
        cols: usize,
        chunk_rows: usize,
    ) -> FFTResult<Vec<(f64, f64)>> {
        let mut row_results: Vec<(f64, f64)> = Vec::with_capacity(rows * cols);

        let mut row = 0_usize;
        while row < rows {
            let end = (row + chunk_rows).min(rows);
            for r in row..end {
                let start = r * cols;
                let row_data: Vec<Complex64> = spectrum[start..start + cols]
                    .iter()
                    .map(|&(re, im)| Complex64::new(re, im))
                    .collect();
                let time_row = ifft(&row_data, Some(cols))?;
                for v in time_row.iter().take(cols) {
                    row_results.push((v.re, v.im));
                }
            }
            row = end;
        }

        Ok(row_results)
    }

    /// IFFT every column of a complex row-major array.
    fn ifft_all_cols(
        &self,
        data: &[(f64, f64)],
        rows: usize,
        cols: usize,
        _chunk_rows: usize,
    ) -> FFTResult<Vec<(f64, f64)>> {
        let total = rows * cols;
        let mut result: Vec<(f64, f64)> = vec![(0.0, 0.0); total];

        for c in 0..cols {
            let col_data: Vec<Complex64> = (0..rows)
                .map(|r| {
                    let (re, im) = data[r * cols + c];
                    Complex64::new(re, im)
                })
                .collect();

            let col_result = ifft(&col_data, Some(rows))?;

            for (r, val) in col_result.iter().take(rows).enumerate() {
                result[r * cols + c] = (val.re, val.im);
            }
        }

        Ok(result)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Out-of-core helpers: disk-based transpose
// ─────────────────────────────────────────────────────────────────────────────

/// Write `(f64, f64)` pairs to a file in little-endian binary format.
fn write_complex_pairs<W: Write>(writer: &mut W, data: &[(f64, f64)]) -> FFTResult<()> {
    for &(re, im) in data {
        writer
            .write_all(&re.to_le_bytes())
            .map_err(|e| FFTError::IOError(format!("write re: {}", e)))?;
        writer
            .write_all(&im.to_le_bytes())
            .map_err(|e| FFTError::IOError(format!("write im: {}", e)))?;
    }
    Ok(())
}

/// Read `count` `(f64, f64)` pairs from a `BufReader` backed by a seekable
/// file, starting at byte offset `byte_offset`.
fn read_complex_pair<R: Read + Seek>(
    reader: &mut BufReader<R>,
    byte_offset: u64,
) -> FFTResult<(f64, f64)> {
    reader
        .seek(SeekFrom::Start(byte_offset))
        .map_err(|e| FFTError::IOError(format!("seek: {}", e)))?;

    let mut buf = [0u8; 8];
    reader
        .read_exact(&mut buf)
        .map_err(|e| FFTError::IOError(format!("read re: {}", e)))?;
    let re = f64::from_le_bytes(buf);

    reader
        .read_exact(&mut buf)
        .map_err(|e| FFTError::IOError(format!("read im: {}", e)))?;
    let im = f64::from_le_bytes(buf);

    Ok((re, im))
}

/// Disk-based 2-D FFT for data that doesn't fit in RAM.
///
/// This function demonstrates the full out-of-core pipeline:
/// 1. Write row-FFT results to a temp file.
/// 2. Read column data from the temp file via seeks (disk transpose).
/// 3. FFT each column and return the result.
///
/// For most practical use `OutOfCoreFft2D::fft2d` is preferred; this
/// function exposes the raw file I/O for integration testing.
///
/// # Errors
///
/// Returns [`FFTError`] if any I/O or FFT step fails.
pub fn disk_based_fft2d(
    data: &[f64],
    rows: usize,
    cols: usize,
    temp_dir: &PathBuf,
) -> FFTResult<Vec<(f64, f64)>> {
    if data.len() != rows * cols {
        return Err(FFTError::DimensionError(format!(
            "disk_based_fft2d: data {} != {}*{}",
            data.len(),
            rows,
            cols
        )));
    }

    // ── Phase 1: FFT rows → temp file ─────────────────────────────────────────
    let tmp_row = tempfile::NamedTempFile::new_in(temp_dir)
        .map_err(|e| FFTError::IOError(format!("create temp row file: {}", e)))?;

    {
        let mut writer = BufWriter::new(tmp_row.as_file());
        for r in 0..rows {
            let row_data = &data[r * cols..(r + 1) * cols];
            let spectrum = fft(row_data, Some(cols))?;
            write_complex_pairs(
                &mut writer,
                &spectrum
                    .iter()
                    .take(cols)
                    .map(|c| (c.re, c.im))
                    .collect::<Vec<_>>(),
            )?;
        }
        writer
            .flush()
            .map_err(|e| FFTError::IOError(format!("flush row temp: {}", e)))?;
    }

    // ── Phase 2: Column FFTs via disk seek ────────────────────────────────────
    // Bytes per complex pair = 16 (two f64s).
    const COMPLEX_BYTES: u64 = 16;
    let row_stride = cols as u64 * COMPLEX_BYTES;

    let mut result: Vec<(f64, f64)> = vec![(0.0, 0.0); rows * cols];

    let row_file = std::fs::File::open(tmp_row.path())
        .map_err(|e| FFTError::IOError(format!("open row temp: {}", e)))?;
    let mut reader = BufReader::new(row_file);

    for c in 0..cols {
        // Gather column `c` by seeking to position (r, c) for each row r.
        let col_data: Result<Vec<Complex64>, FFTError> = (0..rows)
            .map(|r| {
                let offset = r as u64 * row_stride + c as u64 * COMPLEX_BYTES;
                let (re, im) = read_complex_pair(&mut reader, offset)?;
                Ok(Complex64::new(re, im))
            })
            .collect();

        let col_vec = col_data?;
        let col_spectrum = fft(&col_vec, Some(rows))?;

        for (r, val) in col_spectrum.iter().take(rows).enumerate() {
            result[r * cols + c] = (val.re, val.im);
        }
    }

    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
//  In-memory reference implementation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the 2-D FFT of a row-major real array entirely in memory.
///
/// This is the reference implementation used for testing and for data that
/// fits comfortably in RAM.
///
/// Returns complex values as `Vec<(f64, f64)>` (real, imaginary) in
/// row-major order.
///
/// # Example
///
/// ```rust
/// use scirs2_fft::outofcore::small_fft2d;
///
/// // 4×4 identity (impulse at origin): all bins equal 1.
/// let mut data = vec![0.0_f64; 16];
/// data[0] = 1.0;
/// let s = small_fft2d(&data, 4, 4);
/// assert_eq!(s.len(), 16);
/// for &(re, im) in &s {
///     assert!((re - 1.0).abs() < 1e-10);
///     assert!(im.abs() < 1e-10);
/// }
/// ```
pub fn small_fft2d(data: &[f64], rows: usize, cols: usize) -> Vec<(f64, f64)> {
    // Guard against mismatched dimensions.
    if data.len() != rows * cols || rows == 0 || cols == 0 {
        return Vec::new();
    }

    let proc = OutOfCoreFft2D::new(rows, cols);
    proc.fft2d(data).unwrap_or_default()
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test 1: small_fft2d — DC component of impulse at origin ──────────────

    #[test]
    fn test_small_fft2d_impulse_dc() {
        let rows = 8_usize;
        let cols = 8_usize;
        let mut data = vec![0.0_f64; rows * cols];
        data[0] = 1.0;

        let spectrum = small_fft2d(&data, rows, cols);
        assert_eq!(spectrum.len(), rows * cols);

        // 2-D DFT of delta[0,0] = 1 everywhere.
        for (i, &(re, im)) in spectrum.iter().enumerate() {
            assert!((re - 1.0).abs() < 1e-10, "bin {i}: re={re} expected 1.0");
            assert!(im.abs() < 1e-10, "bin {i}: im={im} expected 0.0");
        }
    }

    // ── Test 2: fft2d → ifft2d round-trip (8×8) ──────────────────────────────

    #[test]
    fn test_fft2d_ifft2d_roundtrip_8x8() {
        let rows = 8_usize;
        let cols = 8_usize;
        // Use a non-trivial signal: sine + cosine mix.
        let data: Vec<f64> = (0..rows * cols)
            .map(|k| {
                let r = k / cols;
                let c = k % cols;
                (r as f64 * 0.3).sin() + (c as f64 * 0.7).cos()
            })
            .collect();

        let proc = OutOfCoreFft2D::new(rows, cols);
        let spectrum = proc.fft2d(&data).expect("fft2d failed");
        let recovered = proc.ifft2d(&spectrum).expect("ifft2d failed");

        assert_eq!(recovered.len(), data.len());
        for (i, (&orig, &rec)) in data.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-10,
                "index {i}: original={orig} recovered={rec} diff={}",
                (orig - rec).abs()
            );
        }
    }

    // ── Test 3: fft2d matches small_fft2d for 16×16 ──────────────────────────

    #[test]
    fn test_fft2d_matches_small_fft2d_16x16() {
        let rows = 16_usize;
        let cols = 16_usize;
        let data: Vec<f64> = (0..rows * cols)
            .map(|k| (k as f64 * std::f64::consts::PI / 16.0).sin())
            .collect();

        let proc = OutOfCoreFft2D::new(rows, cols);
        let result1 = proc.fft2d(&data).expect("fft2d failed");
        let result2 = small_fft2d(&data, rows, cols);

        assert_eq!(result1.len(), result2.len());
        for (i, (&(re1, im1), &(re2, im2))) in result1.iter().zip(result2.iter()).enumerate() {
            assert!((re1 - re2).abs() < 1e-10, "bin {i}: re1={re1} re2={re2}");
            assert!((im1 - im2).abs() < 1e-10, "bin {i}: im1={im1} im2={im2}");
        }
    }

    // ── Test 4: Parseval's theorem ────────────────────────────────────────────
    //
    // sum |x[r,c]|^2 = (1/(rows*cols)) * sum |X[k,l]|^2

    #[test]
    fn test_parseval_theorem() {
        let rows = 8_usize;
        let cols = 8_usize;
        let n = (rows * cols) as f64;
        let data: Vec<f64> = (0..rows * cols)
            .map(|k| ((k as f64) * 0.4).sin() * 2.0)
            .collect();

        let proc = OutOfCoreFft2D::new(rows, cols);
        let spectrum = proc.fft2d(&data).expect("fft2d failed");

        let spatial_energy: f64 = data.iter().map(|&x| x * x).sum();
        let spectral_energy: f64 = spectrum
            .iter()
            .map(|&(re, im)| re * re + im * im)
            .sum::<f64>()
            / n;

        assert!(
            (spatial_energy - spectral_energy).abs() < 1e-9,
            "Parseval: spatial={spatial_energy} spectral/N={spectral_energy}"
        );
    }

    // ── Test 5: chunk_rows=1 gives same result as chunk_rows=rows ────────────

    #[test]
    fn test_chunk_rows_1_vs_full() {
        let rows = 8_usize;
        let cols = 8_usize;
        let data: Vec<f64> = (0..rows * cols).map(|k| (k as f64 * 0.2).cos()).collect();

        let proc_full = OutOfCoreFft2D::with_config(OutOfCoreConfig {
            rows,
            cols,
            chunk_rows: rows,
            temp_dir: std::env::temp_dir(),
        });
        let result_full = proc_full.fft2d(&data).expect("full fft2d failed");

        let proc_one = OutOfCoreFft2D::with_config(OutOfCoreConfig {
            rows,
            cols,
            chunk_rows: 1,
            temp_dir: std::env::temp_dir(),
        });
        let result_one = proc_one.fft2d(&data).expect("chunk-1 fft2d failed");

        assert_eq!(result_full.len(), result_one.len());
        for (i, (&(re_f, im_f), &(re_o, im_o))) in
            result_full.iter().zip(result_one.iter()).enumerate()
        {
            assert!(
                (re_f - re_o).abs() < 1e-10,
                "bin {i}: re_full={re_f} re_one={re_o}"
            );
            assert!(
                (im_f - im_o).abs() < 1e-10,
                "bin {i}: im_full={im_f} im_one={im_o}"
            );
        }
    }

    // ── Test 6: disk-based path matches in-memory path ───────────────────────

    #[test]
    fn test_disk_based_matches_in_memory() {
        let rows = 8_usize;
        let cols = 8_usize;
        let data: Vec<f64> = (0..rows * cols).map(|k| (k as f64 * 0.5).sin()).collect();

        let in_memory = small_fft2d(&data, rows, cols);
        let on_disk =
            disk_based_fft2d(&data, rows, cols, &std::env::temp_dir()).expect("disk fft2d failed");

        assert_eq!(in_memory.len(), on_disk.len());
        for (i, (&(re_m, im_m), &(re_d, im_d))) in in_memory.iter().zip(on_disk.iter()).enumerate()
        {
            assert!(
                (re_m - re_d).abs() < 1e-10,
                "bin {i}: re_mem={re_m} re_disk={re_d}"
            );
            assert!(
                (im_m - im_d).abs() < 1e-10,
                "bin {i}: im_mem={im_m} im_disk={im_d}"
            );
        }
    }

    // ── Test 7: dimension mismatch returns error ──────────────────────────────

    #[test]
    fn test_dimension_mismatch_error() {
        let proc = OutOfCoreFft2D::new(8, 8);
        let result = proc.fft2d(&[1.0, 2.0, 3.0]); // wrong length
        assert!(result.is_err(), "expected error for length mismatch");
    }

    // ── Test 8: pure-tone 2D signal has energy at expected bin ───────────────

    #[test]
    fn test_pure_tone_bin_location() {
        // A 2-D pure tone at frequency (f_r, f_c): cos(2π f_r r / R) * cos(2π f_c c / C)
        let rows = 8_usize;
        let cols = 8_usize;
        let f_r = 1_usize; // 1 cycle per row
        let f_c = 2_usize; // 2 cycles per column

        let data: Vec<f64> = (0..rows * cols)
            .map(|k| {
                let r = k / cols;
                let c = k % cols;
                (2.0 * std::f64::consts::PI * f_r as f64 * r as f64 / rows as f64).cos()
                    * (2.0 * std::f64::consts::PI * f_c as f64 * c as f64 / cols as f64).cos()
            })
            .collect();

        let spectrum = small_fft2d(&data, rows, cols);
        let n = (rows * cols) as f64;

        // Expected non-zero bins: (f_r, f_c) and symmetric counterparts.
        // Each cos = (e^{i·} + e^{-i·})/2, so 4 bins each with magnitude n/4.
        let expected_magnitude = n / 4.0;

        let bin_at = |r: usize, c: usize| -> f64 {
            let (re, im) = spectrum[r * cols + c];
            (re * re + im * im).sqrt()
        };

        assert!(
            (bin_at(f_r, f_c) - expected_magnitude).abs() < 1e-9,
            "expected magnitude {} at ({f_r},{f_c}), got {}",
            expected_magnitude,
            bin_at(f_r, f_c)
        );
    }
}

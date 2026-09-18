//! NAIF SPK (`.bsp`) kernels: the DAF container and SPK segment
//! evaluation for data types 2, 3 (Chebyshev) and 13 (Hermite).
//!
//! # Format provenance (clean room)
//!
//! Implemented from NAIF's **public format documentation** only — the
//! "DAF Required Reading" (`daf.req`) and "SPK Required Reading"
//! (`spk.req`) distributed with the SPICE toolkit documentation on
//! `naif.jpl.nasa.gov` — not from the SPICE toolkit source code (which
//! this project has no need to open; the documents specify every byte).
//! Verification: cross-format agreement with the classic-binary DE440
//! reader of this crate on NAIF's `de440.bsp` (same underlying JPL
//! integration), plus Hermite node-reproduction tests on a NAIF
//! asteroid kernel — see `tests/spk.rs`.
//!
//! # Scope
//!
//! * [`SpkFile::parse`] — validates the DAF file record (`DAF/SPK`,
//!   `LTL-IEEE`/`BIG-IEEE`, ND = 2, NI = 6) and exposes the summary
//!   (segment) list. Pre-1996 DAF files without the binary-format word
//!   are not supported.
//! * [`SpkFile::state`] — the state (km, km/s) of `target` relative to
//!   `center` at a two-part TDB Julian date, from the **last** matching
//!   segment in file order (the documented NAIF precedence rule:
//!   later-loaded/later-in-file data wins).
//! * Segment data types: 2 (Chebyshev position, velocity by
//!   differentiation), 3 (Chebyshev position and velocity), 13
//!   (unevenly spaced Hermite interpolation of position/velocity
//!   records). Everything else is [`SpkError::UnsupportedType`].
//!
//! No chaining is performed: `state(target, center, ..)` requires a
//! segment stored for exactly that `(target, center)` pair. Composing
//! states across centers (e.g. asteroid-heliocentric plus Sun-barycentric)
//! is the caller's choice of ephemerides, not this reader's.
//!
//! # Units and time argument
//!
//! SPK stores kilometers and kilometers **per second**, with epochs in
//! TDB seconds past J2000.0 (`ET`); this module keeps those units (note
//! the classic-binary reader's rates are per **day**). The reference
//! frame of a segment is reported verbatim as its NAIF integer code
//! ([`SpkSegment::frame_id`]; `1` = J2000/ICRF equatorial, the value
//! used by all JPL planetary kernels).

use crate::cheby::{chebyshev_rate, chebyshev_value, f64_to_usize_trunc};
use crate::parse::f64_at;
use crate::Endianness;

/// DAF record length in bytes (`daf.req`: 1024 bytes = 128 doubles).
const RECORD_BYTES: usize = 1024;

/// Doubles per summary for SPK's ND = 2, NI = 6:
/// `SS = ND + (NI + 1) / 2 = 5` (`daf.req`, "Summary Records").
const SUMMARY_DOUBLES: usize = 5;

/// Maximum Chebyshev coefficients per component accepted for types 2/3
/// (JPL planetary kernels use ≤ 14; generous headroom, stack-buffer
/// bound).
const MAX_CHEB_COEFFS: usize = 64;

/// Maximum Hermite window (states) accepted for type 13 (NAIF's own
/// maximum window for types 8/9/12/13 era kernels is far below this;
/// `codes_300ast` uses 8). Bounds the divided-difference stack buffers.
const MAX_HERMITE_WINDOW: usize = 16;

/// TDB seconds per day, and the J2000.0 epoch as a Julian date — for the
/// two-part-JD convenience entry point.
const SECONDS_PER_DAY: f64 = 86_400.0;
const J2000_JD: f64 = 2_451_545.0;

/// A two-part `ET` (TDB seconds past J2000.0): `hi + lo`, with `hi` exact.
///
/// A single `f64` of `ET` carries an ulp of ~2 µs at ±300 years from
/// J2000 — about 100 m of Mercury along-track motion — so the epoch is
/// kept split: `hi = trunc(jd_hi − J2000) · 86400` (an exact integer
/// product) and `lo` the small remainder. Record arguments are then
/// formed as `(hi − big) + lo`, where `hi − big` is exact by Sterbenz's
/// lemma once the record/epoch bracket has brought `big` within a factor
/// of two of `hi` (record midpoints and node epochs always are).
#[derive(Debug, Clone, Copy)]
struct Et {
    hi: f64,
    lo: f64,
}

impl Et {
    /// From a two-part TDB Julian date.
    fn from_jd(jd: (f64, f64)) -> Self {
        // Exact for the DE/SPK span: |jd.0 − J2000| < 2^22 days and both
        // operands are large and close (Sterbenz), and an integer count
        // of days times 86400 is an exact integer below 2^53.
        let d = jd.0 - J2000_JD;
        let di = libm::trunc(d);
        Self {
            hi: di * SECONDS_PER_DAY,
            lo: ((d - di) + jd.1) * SECONDS_PER_DAY,
        }
    }

    /// From a single `f64` of ET seconds (public entry points).
    const fn from_seconds(et_s: f64) -> Self {
        Self { hi: et_s, lo: 0.0 }
    }

    /// Collapsed value, for coarse uses (span checks, record selection,
    /// epoch bracketing) where a microsecond cannot change the outcome
    /// by more than one record — and the evaluation below is continuous
    /// across record boundaries at fit-noise level.
    fn approx(self) -> f64 {
        self.hi + self.lo
    }

    /// `(self − big)`, keeping the split's precision: exact `hi − big`
    /// (Sterbenz once `big` is within 2× of `hi`) plus the small `lo`.
    fn minus(self, big: f64) -> f64 {
        (self.hi - big) + self.lo
    }
}

/// Errors while parsing or evaluating an SPK/DAF file.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpkError {
    /// Not a supported DAF/SPK file: bad `DAF/SPK` id word, missing
    /// `LTL-IEEE`/`BIG-IEEE` binary-format word (pre-1996 file), or
    /// ND/NI not the SPK-documented 2/6.
    InvalidDaf,
    /// The byte slice ends before an address the record structure
    /// requires.
    Truncated,
    /// No segment stores `target` relative to `center` covering the
    /// requested epoch.
    NoSegment,
    /// The matching segment's data type is not implemented (payload:
    /// the NAIF type code).
    UnsupportedType(i32),
    /// A segment's internal structure fails a consistency check
    /// (record counts, directory sizes, coefficient counts, window
    /// size, non-finite control values).
    Corrupt,
}

impl core::fmt::Display for SpkError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidDaf => f.write_str("not a supported DAF/SPK file"),
            Self::Truncated => f.write_str("SPK byte slice is truncated"),
            Self::NoSegment => f.write_str("no SPK segment for this target/center/epoch"),
            Self::UnsupportedType(t) => write!(f, "SPK data type {t} not implemented"),
            Self::Corrupt => f.write_str("SPK segment failed a consistency check"),
        }
    }
}

impl core::error::Error for SpkError {}

/// One SPK segment (a DAF array with SPK summary semantics).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpkSegment {
    /// Segment start epoch, TDB seconds past J2000.0.
    pub et_start_s: f64,
    /// Segment end epoch, TDB seconds past J2000.0.
    pub et_end_s: f64,
    /// NAIF id of the body whose motion the segment stores.
    pub target: i32,
    /// NAIF id of the center of motion.
    pub center: i32,
    /// NAIF integer frame code of the stored states (`1` = J2000/ICRF).
    pub frame_id: i32,
    /// SPK data type (2, 3, 13, …).
    pub data_type: i32,
    /// First DAF word address (1-based) of the segment data.
    start_word: usize,
    /// Last DAF word address (1-based, inclusive).
    end_word: usize,
}

impl SpkSegment {
    /// The segment's DAF word-address range (1-based, inclusive) — for
    /// tooling and tests that read raw segment content through
    /// [`SpkFile::read_words`], independent of the evaluation paths.
    #[must_use]
    pub const fn word_range(&self) -> (usize, usize) {
        (self.start_word, self.end_word)
    }
}

/// A state evaluated from an SPK segment.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpkState {
    /// Position of `target` relative to `center`, km, in the segment's
    /// frame.
    pub pos_km: [f64; 3],
    /// Velocity, km per **second** (SPK convention).
    pub vel_km_s: [f64; 3],
    /// The segment's NAIF frame code (`1` = J2000/ICRF).
    pub frame_id: i32,
    /// The segment's SPK data type the state came from.
    pub data_type: i32,
}

/// A parsed SPK (`.bsp`) file borrowing the caller's bytes.
#[derive(Debug, Clone)]
pub struct SpkFile<'a> {
    data: &'a [u8],
    endian: Endianness,
    /// Record number (1-based) of the first summary record.
    fward: usize,
}

impl<'a> SpkFile<'a> {
    /// Parses and validates the DAF file record.
    ///
    /// # Errors
    ///
    /// [`SpkError::InvalidDaf`] for a non-SPK or pre-binary-format-word
    /// DAF; [`SpkError::Truncated`] if even the 1024-byte file record is
    /// missing.
    pub fn parse(data: &'a [u8]) -> Result<Self, SpkError> {
        if data.len() < RECORD_BYTES {
            return Err(SpkError::Truncated);
        }
        // LOCIDW: 8 characters. `daf.req` specifies "DAF/SPK " for SPK.
        if &data[0..8] != b"DAF/SPK " {
            return Err(SpkError::InvalidDaf);
        }
        // LOCFMT at byte 88: the binary file format word (added 1996;
        // files older than that are not supported here).
        let endian = match &data[88..96] {
            b"LTL-IEEE" => Endianness::Little,
            b"BIG-IEEE" => Endianness::Big,
            _ => return Err(SpkError::InvalidDaf),
        };
        let nd = i32_le_be(data, 8, endian)?;
        let ni = i32_le_be(data, 12, endian)?;
        if nd != 2 || ni != 6 {
            return Err(SpkError::InvalidDaf);
        }
        let fward = i32_le_be(data, 76, endian)?;
        if fward < 2 {
            return Err(SpkError::InvalidDaf);
        }
        #[allow(clippy::cast_sign_loss)] // fward >= 2 checked above
        Ok(Self {
            data,
            endian,
            fward: fward as usize,
        })
    }

    /// The DAF word (1-based double-precision address) at `w`.
    fn word(&self, w: usize) -> Result<f64, SpkError> {
        if w == 0 {
            return Err(SpkError::Corrupt);
        }
        f64_at(self.data, (w - 1) * 8, self.endian).map_err(|_| SpkError::Truncated)
    }

    /// A little-endian/big-endian i32 embedded in the packed integer
    /// part of a summary, at absolute byte offset `off`.
    fn summary_i32(&self, off: usize) -> Result<i32, SpkError> {
        i32_le_be(self.data, off, self.endian)
    }

    /// Calls `visit` for every segment, in file order. Returning
    /// `Some(..)` from `visit` stops the walk early with that value.
    fn walk_segments<T>(
        &self,
        mut visit: impl FnMut(&SpkSegment) -> Option<T>,
    ) -> Result<Option<T>, SpkError> {
        let mut rec = self.fward;
        // A DAF summary chain cannot plausibly exceed the record count of
        // the file; use that as a loop bound against malformed cycles.
        let max_records = self.data.len() / RECORD_BYTES + 1;
        let mut visited = 0_usize;
        while rec != 0 {
            visited += 1;
            if visited > max_records {
                return Err(SpkError::Corrupt);
            }
            let base = (rec - 1) * RECORD_BYTES;
            if base + RECORD_BYTES > self.data.len() {
                return Err(SpkError::Truncated);
            }
            let next = control_word(self.word(word_of_byte(base))?)?;
            let nsum = control_word(self.word(word_of_byte(base) + 2)?)?;
            if nsum > (128 - 3) / SUMMARY_DOUBLES {
                return Err(SpkError::Corrupt);
            }
            for i in 0..nsum {
                let s = base + 24 + i * SUMMARY_DOUBLES * 8;
                let et_start_s =
                    f64_at(self.data, s, self.endian).map_err(|_| SpkError::Truncated)?;
                let et_end_s =
                    f64_at(self.data, s + 8, self.endian).map_err(|_| SpkError::Truncated)?;
                let target = self.summary_i32(s + 16)?;
                let center = self.summary_i32(s + 20)?;
                let frame_id = self.summary_i32(s + 24)?;
                let data_type = self.summary_i32(s + 28)?;
                let start_word = self.summary_i32(s + 32)?;
                let end_word = self.summary_i32(s + 36)?;
                if start_word < 1 || end_word < start_word {
                    return Err(SpkError::Corrupt);
                }
                #[allow(clippy::cast_sign_loss)] // bounds checked above
                let seg = SpkSegment {
                    et_start_s,
                    et_end_s,
                    target,
                    center,
                    frame_id,
                    data_type,
                    start_word: start_word as usize,
                    end_word: end_word as usize,
                };
                if let Some(v) = visit(&seg) {
                    return Ok(Some(v));
                }
            }
            rec = next;
        }
        Ok(None)
    }

    /// Calls `visit` for every segment summary in file order (for
    /// inspection/tooling; evaluation goes through [`SpkFile::state`]).
    ///
    /// # Errors
    ///
    /// [`SpkError::Truncated`] / [`SpkError::Corrupt`] if the summary
    /// chain is malformed.
    pub fn for_each_segment(&self, mut visit: impl FnMut(&SpkSegment)) -> Result<(), SpkError> {
        self.walk_segments(|seg| {
            visit(seg);
            None::<()>
        })
        .map(|_| ())
    }

    /// The segment that NAIF precedence selects for `(target, center)`
    /// at `et_s`: the **last** covering segment in file order.
    ///
    /// # Errors
    ///
    /// [`SpkError::NoSegment`] if none covers the epoch; the walk's
    /// [`SpkError::Truncated`] / [`SpkError::Corrupt`] otherwise.
    pub fn segment_for(&self, target: i32, center: i32, et_s: f64) -> Result<SpkSegment, SpkError> {
        let mut found: Option<SpkSegment> = None;
        self.walk_segments(|seg| {
            if seg.target == target
                && seg.center == center
                && et_s >= seg.et_start_s
                && et_s <= seg.et_end_s
            {
                found = Some(*seg);
            }
            None::<()>
        })?;
        found.ok_or(SpkError::NoSegment)
    }

    /// Raw double-precision words `first_word ..` of the file (1-based
    /// DAF addresses) copied into `out` — DAF is a generic container and
    /// exposing addressed reads keeps external verification (tests,
    /// tooling) independent of the evaluation code paths.
    ///
    /// # Errors
    ///
    /// [`SpkError::Truncated`] if any requested word is past the end.
    pub fn read_words(&self, first_word: usize, out: &mut [f64]) -> Result<(), SpkError> {
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = self.word(first_word + i)?;
        }
        Ok(())
    }

    /// State of `target` relative to `center` at a two-part TDB Julian
    /// date (same argument convention as
    /// [`crate::DeFile::series_state`]): `ET = ((hi − 2451545.0) + lo) ·
    /// 86400 s`.
    ///
    /// # Errors
    ///
    /// See [`SpkFile::state_at_et`].
    pub fn state(
        &self,
        target: i32,
        center: i32,
        jd_tdb: (f64, f64),
    ) -> Result<SpkState, SpkError> {
        self.state_at(target, center, Et::from_jd(jd_tdb))
    }

    /// State of `target` relative to `center` at `et_s` TDB seconds past
    /// J2000.0.
    ///
    /// # Errors
    ///
    /// [`SpkError::NoSegment`] (no covering segment),
    /// [`SpkError::UnsupportedType`] (segment type other than 2/3/13),
    /// [`SpkError::Truncated`] / [`SpkError::Corrupt`] for malformed
    /// files.
    pub fn state_at_et(&self, target: i32, center: i32, et_s: f64) -> Result<SpkState, SpkError> {
        self.state_at(target, center, Et::from_seconds(et_s))
    }

    /// Shared implementation of [`SpkFile::state`] / [`SpkFile::state_at_et`].
    fn state_at(&self, target: i32, center: i32, et: Et) -> Result<SpkState, SpkError> {
        let seg = self.segment_for(target, center, et.approx())?;
        let (pos_km, vel_km_s) = match seg.data_type {
            2 => self.eval_chebyshev(&seg, et, false)?,
            3 => self.eval_chebyshev(&seg, et, true)?,
            13 => self.eval_hermite13(&seg, et)?,
            other => return Err(SpkError::UnsupportedType(other)),
        };
        Ok(SpkState {
            pos_km,
            vel_km_s,
            frame_id: seg.frame_id,
            data_type: seg.data_type,
        })
    }

    /// SPK types 2 and 3 (`spk.req`): fixed-length Chebyshev records.
    ///
    /// Segment layout: `N` records of `RSIZE` doubles — `MID`, `RADIUS`
    /// (seconds), then the coefficient sets (3 components for type 2;
    /// 6 for type 3) — followed by the four-double directory
    /// `INIT, INTLEN, RSIZE, N`. Record selection is
    /// `floor((et − INIT)/INTLEN)` clamped to `[0, N-1]`; the normalized
    /// argument is `(et − MID)/RADIUS`. Type 2 velocity is the
    /// differentiated position series (chain rule `1/RADIUS`, giving
    /// km/s); type 3 stores velocity coefficients directly.
    fn eval_chebyshev(
        &self,
        seg: &SpkSegment,
        et: Et,
        has_velocity_sets: bool,
    ) -> Result<([f64; 3], [f64; 3]), SpkError> {
        let nwords = seg.end_word - seg.start_word + 1;
        if nwords < 4 {
            return Err(SpkError::Corrupt);
        }
        let dir = seg.end_word - 3;
        let init = self.word(dir)?;
        let intlen = self.word(dir + 1)?;
        let rsize = f64_usize(self.word(dir + 2)?)?;
        let n = f64_usize(self.word(dir + 3)?)?;
        // `is_finite` guards: a NaN `INIT`/`INTLEN` would sail through
        // the `<= 0.0` comparison (NaN compares false) and yield silent
        // NaN states; `checked_mul` guards the size identity against
        // overflow from huge (corrupt) `RSIZE`/`N` values.
        if !init.is_finite() || !intlen.is_finite() || intlen <= 0.0 || rsize < 3 {
            return Err(SpkError::Corrupt);
        }
        let expected = n.checked_mul(rsize).and_then(|w| w.checked_add(4));
        if expected != Some(nwords) {
            return Err(SpkError::Corrupt);
        }
        let sets = if has_velocity_sets { 6 } else { 3 };
        if (rsize - 2) % sets != 0 {
            return Err(SpkError::Corrupt);
        }
        let ncf = (rsize - 2) / sets;
        if ncf == 0 || ncf > MAX_CHEB_COEFFS {
            return Err(SpkError::Corrupt);
        }

        let idx = f64_to_usize_trunc((et.approx() - init) / intlen).min(n - 1);
        let rec = seg.start_word + idx * rsize;
        let mid = self.word(rec)?;
        let radius = self.word(rec + 1)?;
        if !mid.is_finite() || radius <= 0.0 || !radius.is_finite() {
            return Err(SpkError::Corrupt);
        }
        // Two-part argument: `et.hi − mid` is Sterbenz-exact (the record
        // midpoint is within one interval of the epoch).
        let x = et.minus(mid) / radius;

        let mut coeffs = [0.0_f64; MAX_CHEB_COEFFS];
        let mut pos = [0.0_f64; 3];
        let mut vel = [0.0_f64; 3];
        for c in 0..3 {
            self.read_words(rec + 2 + c * ncf, &mut coeffs[..ncf])?;
            pos[c] = chebyshev_value(&coeffs[..ncf], x);
            if has_velocity_sets {
                self.read_words(rec + 2 + (c + 3) * ncf, &mut coeffs[..ncf])?;
                vel[c] = chebyshev_value(&coeffs[..ncf], x);
            } else {
                vel[c] = chebyshev_rate(&coeffs[..ncf], x) / radius;
            }
        }
        Ok((pos, vel))
    }

    /// SPK type 13 (`spk.req`): unevenly spaced position/velocity
    /// records, Hermite interpolation.
    ///
    /// Segment layout: `N` states of 6 doubles (km, km/s), then the `N`
    /// epochs, then an epoch directory of `⌊(N−1)/100⌋` entries (every
    /// 100th epoch — a search accelerator this reader does not need;
    /// per `spk.req`, an exact multiple of 100 states stores one *fewer*
    /// directory entry than `N/100`), then the two-double trailer
    /// `WINDOW − 1, N` (the penultimate word stores the window size
    /// **minus one** — `spk.req`'s "Window size - 1"). Evaluation picks
    /// the `WINDOW` consecutive states centered on the epoch bracket
    /// (clamped at the segment ends) and Hermite-interpolates each
    /// position component with its velocity as the derivative data — a
    /// polynomial of degree `2·WINDOW − 1` (`codes_300ast`: trailer
    /// value 7 → window 8, "degree 15"); the velocity output is the
    /// interpolant's derivative.
    fn eval_hermite13(&self, seg: &SpkSegment, et: Et) -> Result<([f64; 3], [f64; 3]), SpkError> {
        let nwords = seg.end_word - seg.start_word + 1;
        if nwords < 2 {
            return Err(SpkError::Corrupt);
        }
        let window = f64_usize(self.word(seg.end_word - 1)?)? + 1;
        let n = f64_usize(self.word(seg.end_word)?)?;
        if !(2..=MAX_HERMITE_WINDOW).contains(&window) || window > n {
            return Err(SpkError::Corrupt);
        }
        if nwords != 7 * n + (n - 1) / 100 + 2 {
            return Err(SpkError::Corrupt);
        }
        let states = seg.start_word; // n * 6 words
        let epochs = seg.start_word + 6 * n; // n words

        // Last epoch index with epoch[i] <= et (binary search; the
        // collapsed value is fine here, see `Et::approx`).
        let et_s = et.approx();
        let mut lo = 0_usize;
        let mut hi = n - 1;
        if et_s < self.word(epochs)? {
            hi = 0;
        }
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            if self.word(epochs + mid)? <= et_s {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        // Window of `window` consecutive states around the bracket.
        let first = lo
            .saturating_add(1)
            .saturating_sub(window.div_ceil(2))
            .min(n - window);

        let mut t = [0.0_f64; MAX_HERMITE_WINDOW];
        self.read_words(epochs + first, &mut t[..window])?;
        // Shift epochs so the evaluation point is x = 0 (conditioning);
        // `epoch − et.hi` is Sterbenz-exact (window epochs bracket et).
        for tj in &mut t[..window] {
            *tj = -et.minus(*tj);
        }
        // Strictly increasing AND finite: a NaN epoch would pass a bare
        // `<=` comparison (NaN compares false) and corrupt the divided
        // differences silently.
        if !t[0].is_finite() {
            return Err(SpkError::Corrupt);
        }
        for j in 1..window {
            if !t[j].is_finite() || t[j] <= t[j - 1] {
                return Err(SpkError::Corrupt);
            }
        }

        let mut state_rows = [[0.0_f64; 6]; MAX_HERMITE_WINDOW];
        for (j, row) in state_rows.iter_mut().enumerate().take(window) {
            self.read_words(states + (first + j) * 6, row)?;
        }

        let mut pos = [0.0_f64; 3];
        let mut vel = [0.0_f64; 3];
        for c in 0..3 {
            let mut p = [0.0_f64; MAX_HERMITE_WINDOW];
            let mut v = [0.0_f64; MAX_HERMITE_WINDOW];
            for j in 0..window {
                p[j] = state_rows[j][c];
                v[j] = state_rows[j][c + 3];
            }
            let (value, rate) = hermite_value_rate(&t[..window], &p[..window], &v[..window])?;
            pos[c] = value;
            vel[c] = rate;
        }
        Ok((pos, vel))
    }
}

/// Hermite interpolation through `(t_j, p_j)` with derivatives `v_j`,
/// evaluated (value and derivative) at `x = 0` — the caller pre-shifts
/// the abscissas. Newton form on doubled nodes with the standard
/// divided-difference table (e.g. Burden & Faires, *Numerical Analysis*,
/// §3.4 "Hermite Interpolation"); the confluent first difference
/// `f[z_{2j}, z_{2j+1}] = v_j`.
#[allow(clippy::many_single_char_names)] // t/p/v/z/q: the textbook symbols of the algorithm
fn hermite_value_rate(t: &[f64], p: &[f64], v: &[f64]) -> Result<(f64, f64), SpkError> {
    let w = t.len();
    let m = 2 * w;
    let mut z = [0.0_f64; 2 * MAX_HERMITE_WINDOW];
    let mut q = [0.0_f64; 2 * MAX_HERMITE_WINDOW];
    // Table row 0: values on doubled nodes.
    for j in 0..w {
        z[2 * j] = t[j];
        z[2 * j + 1] = t[j];
        q[2 * j] = p[j];
        q[2 * j + 1] = p[j];
    }
    // In-place divided differences: after pass k, q[i] holds
    // f[z_{i-k}, .., z_i]; Newton coefficients are the diagonal q[i]
    // captured as each pass fixes it.
    let mut coeff = [0.0_f64; 2 * MAX_HERMITE_WINDOW];
    coeff[0] = q[0];
    for k in 1..m {
        for i in (k..m).rev() {
            let denom = z[i] - z[i - k];
            if denom == 0.0 {
                if k == 1 {
                    // Confluent pair: the derivative datum.
                    q[i] = v[(i - 1) / 2];
                    continue;
                }
                return Err(SpkError::Corrupt);
            }
            q[i] = (q[i] - q[i - 1]) / denom;
        }
        coeff[k] = q[k];
    }
    // Evaluate the Newton form and its derivative at x = 0.
    let mut value = coeff[m - 1];
    let mut rate = 0.0_f64;
    for i in (0..m - 1).rev() {
        let dx = 0.0 - z[i];
        rate = value + dx * rate;
        value = coeff[i] + dx * value;
    }
    Ok((value, rate))
}

/// An i32 at byte offset `off` under `endian`.
fn i32_le_be(data: &[u8], off: usize, endian: Endianness) -> Result<i32, SpkError> {
    let bytes: [u8; 4] = data
        .get(off..off + 4)
        .and_then(|s| s.try_into().ok())
        .ok_or(SpkError::Truncated)?;
    Ok(match endian {
        Endianness::Little => i32::from_le_bytes(bytes),
        Endianness::Big => i32::from_be_bytes(bytes),
    })
}

/// DAF word address (1-based) of the double at byte offset `base`.
const fn word_of_byte(base: usize) -> usize {
    base / 8 + 1
}

/// A DAF control word (`NEXT`/`PREV`/`NSUM` are stored as doubles) as a
/// non-negative integer.
fn control_word(x: f64) -> Result<usize, SpkError> {
    if !x.is_finite() || !(0.0..=1e12).contains(&x) {
        return Err(SpkError::Corrupt);
    }
    Ok(f64_to_usize_trunc(x + 0.5))
}

/// A positive integral f64 (record sizes/counts) as usize.
fn f64_usize(x: f64) -> Result<usize, SpkError> {
    if !x.is_finite() || !(0.5..=1e12).contains(&x) {
        return Err(SpkError::Corrupt);
    }
    Ok(f64_to_usize_trunc(x + 0.5))
}

#[cfg(test)]
mod tests {
    use super::hermite_value_rate;

    /// Hermite through a cubic's samples reproduces the cubic exactly
    /// (2 nodes with derivatives determine degree 3).
    #[test]
    #[allow(clippy::many_single_char_names)] // textbook symbols
    fn hermite_reproduces_a_cubic() {
        // f(x) = x^3 - 2x + 1, f'(x) = 3x^2 - 2, nodes shifted so the
        // evaluation point x0 = 0.3 becomes 0.
        let x0 = 0.3_f64;
        let nodes = [-1.0_f64, 2.0];
        let f = |x: f64| x * x * x - 2.0 * x + 1.0;
        let fp = |x: f64| 3.0 * x * x - 2.0;
        let t = [nodes[0] - x0, nodes[1] - x0];
        let p = [f(nodes[0]), f(nodes[1])];
        let v = [fp(nodes[0]), fp(nodes[1])];
        let (value, rate) = match hermite_value_rate(&t, &p, &v) {
            Ok(vr) => vr,
            Err(e) => panic!("hermite failed: {e}"),
        };
        assert!((value - f(x0)).abs() < 1e-14, "value {value}");
        assert!((rate - fp(x0)).abs() < 1e-14, "rate {rate}");
    }

    /// Node reproduction: at a (shifted) node the interpolant returns
    /// the node's own value and derivative.
    #[test]
    #[allow(clippy::many_single_char_names)] // textbook symbols
    fn hermite_passes_through_nodes() {
        let t = [0.0_f64, 1.5, 4.0, 7.0]; // node 0 == evaluation point
        let p = [2.0_f64, -1.0, 0.5, 3.0];
        let v = [0.25_f64, 1.0, -0.5, 0.125];
        let (value, rate) = match hermite_value_rate(&t, &p, &v) {
            Ok(vr) => vr,
            Err(e) => panic!("hermite failed: {e}"),
        };
        assert!((value - p[0]).abs() < 1e-12, "value {value}");
        assert!((rate - v[0]).abs() < 1e-12, "rate {rate}");
    }
}

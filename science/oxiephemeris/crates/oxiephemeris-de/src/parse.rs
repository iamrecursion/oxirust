//! Classic binary DE header parsing.
//!
//! Byte layout of record 1, from the `WRITE(12,REC=1)` statements of the
//! public-domain JPL `asc2eph.f` (2013-08-15 revision):
//!
//! ```text
//! offset  size        field
//! 0       3*84        TTL    three 84-character title lines
//! 252     400*6       CNAM   first 400 constant names, 6 chars each
//! 2652    3*8         SS     start JD, stop JD, record span (days)
//! 2676    4           NCON   number of constants (i32)
//! 2680    8           AU     km per astronomical unit
//! 2688    8           EMRAT  Earth/Moon mass ratio
//! 2696    12*3*4      IPT    12 series x (start, ncf, na) i32 triplets
//! 2840    4           NUMDE  ephemeris number (i32)
//! 2844    3*4         LPT    libration pointer triplet (series 13)
//! 2856    (NCON-400)*6  extra constant names, only if NCON > 400
//! ...     3*4         RPT    lunar Euler-angle-rate triplet (2013+ files)
//! ...     3*4         TPT    TT-TDB triplet (2013+ files)
//! ```
//!
//! `start` values are 1-based indices into the record's doubles. Record 2
//! holds the `NCON` constant values (`CVAL`); data records start at record 3
//! (byte offset `2 * reclen`).  The record length is `8 * NCOEFF` bytes where
//! `NCOEFF = max over present series of (start - 1 + ncomp*ncf*na)`
//! (`FSIZER2` of `testeph.f`, generalized to the optional RPT/TPT series).
//!
//! There is no magic number: endianness is detected by sanity-checking
//! `NCON`, `SS`, and `AU` under both byte orders.

use crate::{DeError, Endianness, Series};

/// Offset of the three 84-char title lines.
const OFF_TTL: usize = 0;
/// Offset of the first 400 six-char constant names.
const OFF_CNAM: usize = 252;
/// Offset of `SS` (start JD, stop JD, span days).
const OFF_SS: usize = 2652;
/// Offset of `NCON`.
const OFF_NCON: usize = 2676;
/// Offset of `AU`.
const OFF_AU: usize = 2680;
/// Offset of `EMRAT`.
const OFF_EMRAT: usize = 2688;
/// Offset of the 12 `IPT` triplets.
const OFF_IPT: usize = 2696;
/// Offset of `NUMDE`.
const OFF_NUMDE: usize = 2840;
/// Offset of the `LPT` triplet.
const OFF_LPT: usize = 2844;
/// Offset of the variable tail (extra names, then `RPT`, `TPT`).
const OFF_TAIL: usize = 2856;
/// `OLDMAX` of `asc2eph.f`: names beyond 400 go to the tail.
const OLDMAX: usize = 400;
/// Chars per constant name (`CHARACTER*6 CNAM`).
const CNAM_LEN: usize = 6;
/// Largest supported coefficient count per component (`testeph.f` uses
/// `PC(18)`; real files stay well below this).
pub(crate) const MAX_NCF: usize = 32;

/// Layout of one Chebyshev series inside every data record.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SeriesLayout {
    /// Zero-based index of the first coefficient double.
    pub start: usize,
    /// Coefficients per component.
    pub ncf: usize,
    /// Sub-intervals (granules) per record.
    pub na: usize,
    /// `na` as f64 (`DNA` in `testeph.f`), cached losslessly.
    pub na_f: f64,
    /// Components (3 bodies/angles, 2 nutation, 1 TT-TDB).
    pub ncomp: usize,
    /// Whether the series has coefficients in this file.
    pub present: bool,
}

/// A parsed classic binary JPL DE file borrowing the caller's bytes.
#[derive(Debug, Clone)]
pub struct DeFile<'a> {
    pub(crate) data: &'a [u8],
    pub(crate) endian: Endianness,
    pub(crate) ss: [f64; 3],
    pub(crate) ncon: usize,
    pub(crate) au_km: f64,
    pub(crate) emrat: f64,
    pub(crate) numde: i32,
    pub(crate) ncoeff: usize,
    /// Record length in bytes (`8 * ncoeff`).
    pub(crate) reclen: usize,
    /// Number of data records.
    pub(crate) nrec: usize,
    pub(crate) layouts: [SeriesLayout; 15],
}

/// Reads an f64 at `off` with the given byte order.
pub(crate) fn f64_at(data: &[u8], off: usize, endian: Endianness) -> Result<f64, DeError> {
    let bytes: [u8; 8] = data
        .get(off..off.wrapping_add(8))
        .and_then(|s| s.try_into().ok())
        .ok_or(DeError::Truncated)?;
    Ok(match endian {
        Endianness::Little => f64::from_le_bytes(bytes),
        Endianness::Big => f64::from_be_bytes(bytes),
    })
}

/// Reads an i32 at `off` with the given byte order.
fn i32_at(data: &[u8], off: usize, endian: Endianness) -> Result<i32, DeError> {
    let bytes: [u8; 4] = data
        .get(off..off.wrapping_add(4))
        .and_then(|s| s.try_into().ok())
        .ok_or(DeError::Truncated)?;
    Ok(match endian {
        Endianness::Little => i32::from_le_bytes(bytes),
        Endianness::Big => i32::from_be_bytes(bytes),
    })
}

/// Sanity predicate for endianness detection: `NCON`, `SS`, and `AU` must
/// all be plausible under the candidate byte order.
fn header_plausible(data: &[u8], endian: Endianness) -> bool {
    let Ok(ncon) = i32_at(data, OFF_NCON, endian) else {
        return false;
    };
    let Ok(ss0) = f64_at(data, OFF_SS, endian) else {
        return false;
    };
    let Ok(ss1) = f64_at(data, OFF_SS + 8, endian) else {
        return false;
    };
    let Ok(ss2) = f64_at(data, OFF_SS + 16, endian) else {
        return false;
    };
    let Ok(au) = f64_at(data, OFF_AU, endian) else {
        return false;
    };
    ncon > 0
        && ncon < 3000
        && ss0.is_finite()
        && ss1.is_finite()
        && ss2.is_finite()
        && ss0 < ss1
        && ss2 > 0.0
        && ss2 < 400.0
        && au.is_finite()
        && au > 1.4e8
        && au < 1.6e8
}

/// Reads one `(start, ncf, na)` i32 triplet at `off` into a validated
/// [`SeriesLayout`] for a series with `ncomp` components.
fn triplet_at(
    data: &[u8],
    off: usize,
    endian: Endianness,
    ncomp: usize,
) -> Result<SeriesLayout, DeError> {
    let start = i32_at(data, off, endian)?;
    let ncf = i32_at(data, off + 4, endian)?;
    let na = i32_at(data, off + 8, endian)?;
    if start < 0 || ncf < 0 || na < 0 {
        return Err(DeError::InvalidHeader);
    }
    let present = ncf > 0 && na > 0;
    if present && start < 1 {
        return Err(DeError::InvalidHeader);
    }
    let to_usize = |v: i32| usize::try_from(v).map_err(|_| DeError::InvalidHeader);
    let ncf = to_usize(ncf)?;
    if present && ncf > MAX_NCF {
        return Err(DeError::InvalidHeader);
    }
    let na_u32 = u32::try_from(na).map_err(|_| DeError::InvalidHeader)?;
    Ok(SeriesLayout {
        // Convert the 1-based Fortran pointer to a 0-based double index.
        start: to_usize(start)?.saturating_sub(1),
        ncf,
        na: to_usize(na)?,
        na_f: f64::from(na_u32),
        ncomp,
        present,
    })
}

/// Loose plausibility bound for the optional `RPT`/`TPT` triplets: files
/// written before the 2013 `asc2eph.f` revision have arbitrary (usually
/// zero) bytes there, so implausible values mean "absent", not "error".
fn tail_triplet_plausible(data: &[u8], off: usize, endian: Endianness) -> bool {
    for i in 0..3_usize {
        let Ok(v) = i32_at(data, off + 4 * i, endian) else {
            return false;
        };
        if !(0..=1_000_000).contains(&v) {
            return false;
        }
    }
    true
}

impl<'a> DeFile<'a> {
    /// Parses a classic binary JPL DE file (header + coefficient records).
    ///
    /// Endianness is detected automatically; the optional post-2013
    /// `RPT`/`TPT` pointer triplets (lunar Euler-angle rates, TT-TDB) are
    /// recognized when present and plausible.
    ///
    /// # Errors
    /// * [`DeError::Truncated`] — `data` is too short for the header or for
    ///   the record count implied by the header span.
    /// * [`DeError::InvalidHeader`] — no byte order yields a sane header, or
    ///   the pointer table is inconsistent.
    pub fn parse(data: &'a [u8]) -> Result<Self, DeError> {
        if data.len() < OFF_TAIL {
            return Err(DeError::Truncated);
        }
        let little = header_plausible(data, Endianness::Little);
        let big = header_plausible(data, Endianness::Big);
        let endian = match (little, big) {
            (true, false) => Endianness::Little,
            (false, true) => Endianness::Big,
            _ => return Err(DeError::InvalidHeader),
        };

        let ss = [
            f64_at(data, OFF_SS, endian)?,
            f64_at(data, OFF_SS + 8, endian)?,
            f64_at(data, OFF_SS + 16, endian)?,
        ];
        let ncon =
            usize::try_from(i32_at(data, OFF_NCON, endian)?).map_err(|_| DeError::InvalidHeader)?;
        let au_km = f64_at(data, OFF_AU, endian)?;
        let emrat = f64_at(data, OFF_EMRAT, endian)?;
        let numde = i32_at(data, OFF_NUMDE, endian)?;
        if !emrat.is_finite() || emrat <= 0.0 {
            return Err(DeError::InvalidHeader);
        }

        let mut layouts = [SeriesLayout::default(); 15];
        for &series in Series::ALL {
            let slot = series.slot();
            match slot {
                0..=11 => {
                    layouts[slot] = triplet_at(data, OFF_IPT + 12 * slot, endian, series.ncomp())?;
                }
                12 => {
                    layouts[slot] = triplet_at(data, OFF_LPT, endian, series.ncomp())?;
                }
                _ => {} // RPT/TPT handled below (position depends on NCON).
            }
        }

        // Extra constant names occupy the tail only when NCON > 400.
        let extra_names = ncon.saturating_sub(OLDMAX) * CNAM_LEN;
        let rpt_off = OFF_TAIL + extra_names;
        let tpt_off = rpt_off + 12;
        if tail_triplet_plausible(data, rpt_off, endian)
            && tail_triplet_plausible(data, tpt_off, endian)
        {
            layouts[Series::LunarEulerRates.slot()] =
                triplet_at(data, rpt_off, endian, Series::LunarEulerRates.ncomp())?;
            layouts[Series::TtTdb.slot()] =
                triplet_at(data, tpt_off, endian, Series::TtTdb.ncomp())?;
        }

        // NCOEFF = max over present series of (start + ncomp*ncf*na);
        // `start` is already 0-based here (FSIZER2 of testeph.f uses the
        // 1-based pointer and subtracts 1).
        let mut ncoeff = 0_usize;
        for lay in &layouts {
            if lay.present {
                let end = lay
                    .start
                    .checked_add(lay.ncomp * lay.ncf * lay.na)
                    .ok_or(DeError::InvalidHeader)?;
                ncoeff = ncoeff.max(end);
            }
        }
        let reclen = ncoeff.checked_mul(8).ok_or(DeError::InvalidHeader)?;
        // The header record itself must fit in one record, and record 2 must
        // hold all NCON constant values.
        if reclen < OFF_TAIL || reclen < ncon.saturating_mul(8) {
            return Err(DeError::InvalidHeader);
        }
        if ncon > OLDMAX && tpt_off + 12 > reclen {
            return Err(DeError::InvalidHeader);
        }

        // Number of data records from the span; must be a whole number.
        let nrec_f = (ss[1] - ss[0]) / ss[2];
        let nrec = crate::cheby::f64_to_usize_trunc(libm::round(nrec_f));
        if nrec == 0 || libm::fabs(nrec_f - crate::cheby::usize_to_f64(nrec)) > 1e-6 {
            return Err(DeError::InvalidHeader);
        }
        let total = reclen.checked_mul(nrec + 2).ok_or(DeError::InvalidHeader)?;
        if data.len() < total {
            return Err(DeError::Truncated);
        }

        Ok(Self {
            data,
            endian,
            ss,
            ncon,
            au_km,
            emrat,
            numde,
            ncoeff,
            reclen,
            nrec,
            layouts,
        })
    }

    /// Detected byte order of the file.
    #[must_use]
    pub const fn endianness(&self) -> Endianness {
        self.endian
    }

    /// `(start JD, stop JD)` of the file span (TDB).
    #[must_use]
    pub const fn span(&self) -> (f64, f64) {
        (self.ss[0], self.ss[1])
    }

    /// Days covered by one data record (`SS(3)`; 32 for DE440).
    #[must_use]
    pub const fn step_days(&self) -> f64 {
        self.ss[2]
    }

    /// Kilometers per astronomical unit (the header `AU` constant).
    #[must_use]
    pub const fn au_km(&self) -> f64 {
        self.au_km
    }

    /// Earth/Moon mass ratio (the header `EMRAT` constant).
    #[must_use]
    pub const fn emrat(&self) -> f64 {
        self.emrat
    }

    /// Ephemeris number (`NUMDE`; 440 for DE440).
    #[must_use]
    pub const fn numde(&self) -> i32 {
        self.numde
    }

    /// Number of header constants (`NCON`).
    #[must_use]
    pub const fn constant_count(&self) -> usize {
        self.ncon
    }

    /// Doubles per data record (`NCOEFF`; `KSIZE/2`; 1018 for DE440).
    #[must_use]
    pub const fn ncoeff(&self) -> usize {
        self.ncoeff
    }

    /// Number of data records in the file.
    #[must_use]
    pub const fn record_count(&self) -> usize {
        self.nrec
    }

    /// The DE number from the header's `NUMDE` field (e.g. `440` for
    /// DE440, `441` for DE441) — the ephemeris' own identity, useful for
    /// recording provenance.
    #[must_use]
    pub const fn de_number(&self) -> i32 {
        self.numde
    }

    /// Pointer triplet `(start_1based, ncf, na)` for a series, or `None`
    /// if the series has no coefficients in this file.
    #[must_use]
    pub fn pointer_triplet(&self, series: Series) -> Option<(usize, usize, usize)> {
        let lay = self.layouts.get(series.slot())?;
        lay.present.then_some((lay.start + 1, lay.ncf, lay.na))
    }

    /// One of the three 84-character title lines (0..=2), trimmed.
    #[must_use]
    pub fn title(&self, line: usize) -> Option<&'a str> {
        if line >= 3 {
            return None;
        }
        let off = OFF_TTL + 84 * line;
        let bytes = self.data.get(off..off + 84)?;
        core::str::from_utf8(bytes).ok().map(str::trim)
    }

    /// Name of constant `index` (0-based, `index < constant_count()`).
    #[must_use]
    pub fn constant_name(&self, index: usize) -> Option<&'a str> {
        if index >= self.ncon {
            return None;
        }
        let off = if index < OLDMAX {
            OFF_CNAM + CNAM_LEN * index
        } else {
            OFF_TAIL + CNAM_LEN * (index - OLDMAX)
        };
        let bytes = self.data.get(off..off + CNAM_LEN)?;
        core::str::from_utf8(bytes).ok().map(str::trim)
    }

    /// Value of constant `index` (0-based), from record 2 (`CVAL`).
    #[must_use]
    pub fn constant_value(&self, index: usize) -> Option<f64> {
        if index >= self.ncon {
            return None;
        }
        f64_at(self.data, self.reclen + 8 * index, self.endian).ok()
    }

    /// Looks up a header constant by (trimmed) name, e.g. `"AU"`, `"GM1"`.
    #[must_use]
    pub fn constant(&self, name: &str) -> Option<f64> {
        let wanted = name.trim();
        (0..self.ncon).find_map(|i| {
            (self.constant_name(i)? == wanted)
                .then(|| self.constant_value(i))
                .flatten()
        })
    }

    /// Iterates over all `(name, value)` header constants in file order.
    pub fn constants(&self) -> impl Iterator<Item = (&'a str, f64)> + '_ {
        (0..self.ncon).filter_map(|i| Some((self.constant_name(i)?, self.constant_value(i)?)))
    }

    /// Byte slice of data record `index` (0-based; record 0 starts at byte
    /// offset `2 * reclen`).
    pub(crate) fn record_bytes(&self, index: usize) -> Result<&'a [u8], DeError> {
        let off = self
            .reclen
            .checked_mul(index.checked_add(2).ok_or(DeError::Truncated)?)
            .ok_or(DeError::Truncated)?;
        let end = off.checked_add(self.reclen).ok_or(DeError::Truncated)?;
        self.data.get(off..end).ok_or(DeError::Truncated)
    }

    /// Layout for a series, if present.
    pub(crate) fn layout(&self, series: Series) -> Result<SeriesLayout, DeError> {
        let lay = self
            .layouts
            .get(series.slot())
            .ok_or(DeError::SeriesUnavailable)?;
        if lay.present {
            Ok(*lay)
        } else {
            Err(DeError::SeriesUnavailable)
        }
    }
}

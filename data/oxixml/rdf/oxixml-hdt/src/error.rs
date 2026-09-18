//! The [`HdtError`] type returned by every fallible operation in this crate.

use std::error::Error;
use std::fmt;
use std::io;

/// Everything that can go wrong while reading, writing, or querying an HDT file.
///
/// Every variant carries enough context (section name, expected vs. actual values) to
/// diagnose a malformed file without re-reading it in a hex editor.
#[derive(Debug)]
#[non_exhaustive]
pub enum HdtError {
    /// An underlying I/O operation failed.
    Io(io::Error),

    /// A [`ControlInformation`](crate::control_info::ControlInformation) block did not start
    /// with the four-byte `$HDT` cookie.
    InvalidCookie {
        /// The four bytes that were found instead of `$HDT`.
        found: [u8; 4],
    },

    /// A control-information block declared a
    /// [`ControlInfoType`](crate::control_info::ControlInfoType) that did not match what the
    /// reader expected at that point in the file.
    UnexpectedControlType {
        /// The section being parsed, for diagnostics (e.g. `"dictionary"`).
        section: &'static str,
        /// The raw type byte that was actually read.
        found: u8,
    },

    /// A CRC checksum embedded in the file did not match the checksum computed over the bytes
    /// it covers, meaning the file is corrupted (or was tampered with).
    CrcMismatch {
        /// Which checksum failed (`"CRC8"`, `"CRC16"`, or `"CRC32C"`).
        kind: &'static str,
        /// The section being parsed, for diagnostics.
        section: &'static str,
        /// The checksum stored in the file.
        expected: u32,
        /// The checksum computed while reading.
        actual: u32,
    },

    /// The byte stream ended before all the data a section declared was available could be
    /// read.
    UnexpectedEof {
        /// The section being parsed, for diagnostics.
        section: &'static str,
    },

    /// A section used a data-structure `format` identifier (dictionary type, triples order,
    /// array encoding, ...) that this crate does not implement.
    UnsupportedFormat {
        /// The section being parsed, for diagnostics.
        section: &'static str,
        /// A human-readable description of the offending value.
        found: String,
    },

    /// A dictionary or triples ID pointed outside the range that was declared for its section.
    IdOutOfRange {
        /// The section being parsed, for diagnostics.
        section: &'static str,
        /// The offending ID.
        id: u64,
        /// The number of valid entries in the section (IDs are `1..=len`).
        len: u64,
    },

    /// A term string stored in the dictionary could not be parsed back into an RDF term.
    TermParse {
        /// The malformed string, for diagnostics.
        value: String,
        /// The underlying parse error's message.
        message: String,
    },

    /// A structural invariant of the file format was violated in a way not covered by a more
    /// specific variant above (e.g. inconsistent section lengths).
    Malformed(String),

    /// [`BitmapTriples::write`](crate::triples::BitmapTriples::write) was asked to serialize a
    /// structure [`built`](crate::triples::BitmapTriples::build) from a subject ID space that is
    /// not dense and `1`-based.
    ///
    /// The `BitmapTriples` wire format has no field for an explicit subject ID list: a
    /// conforming reader always treats the `k`-th `BitmapY` group as subject `k`. Writing a
    /// gapped-subject structure out and reading it back would therefore silently renumber its
    /// subjects, so `write` refuses instead. Densify subject IDs before writing -- any ID set
    /// produced through [`FourSectionDictionary`](crate::dictionary::FourSectionDictionary)
    /// already is.
    NonDenseSubjectIds,
}

impl fmt::Display for HdtError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::InvalidCookie { found } => write!(
                f,
                "invalid HDT cookie: expected \"$HDT\", found {:?}",
                String::from_utf8_lossy(found)
            ),
            Self::UnexpectedControlType { section, found } => write!(
                f,
                "unexpected control-information type {found} while reading the {section} section"
            ),
            Self::CrcMismatch {
                kind,
                section,
                expected,
                actual,
            } => write!(
                f,
                "{kind} mismatch in the {section} section: expected {expected:#x}, computed {actual:#x}"
            ),
            Self::UnexpectedEof { section } => {
                write!(
                    f,
                    "unexpected end of input while reading the {section} section"
                )
            }
            Self::UnsupportedFormat { section, found } => {
                write!(f, "unsupported format in the {section} section: {found}")
            }
            Self::IdOutOfRange { section, id, len } => write!(
                f,
                "ID {id} is out of range for the {section} section (valid range is 1..={len})"
            ),
            Self::TermParse { value, message } => {
                write!(f, "could not parse dictionary term {value:?}: {message}")
            }
            Self::Malformed(message) => write!(f, "malformed HDT file: {message}"),
            Self::NonDenseSubjectIds => write!(
                f,
                "cannot write BitmapTriples: subject IDs are not dense and 1-based, and the HDT wire format cannot represent gaps in the subject ID space"
            ),
        }
    }
}

impl Error for HdtError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for HdtError {
    #[inline]
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// A convenience alias for `Result<T, HdtError>`.
pub type Result<T> = std::result::Result<T, HdtError>;

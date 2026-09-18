// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Core types of the in-memory HDF5 model: error, dtype, storage, layout,
//! attributes and links.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// 1.  Error type
// ---------------------------------------------------------------------------

/// Errors that can occur when operating on an in-memory HDF5 store or when
/// saving it to / opening it from a `.h5` file.
///
/// The enum is `#[non_exhaustive]`: new failure modes may be added in a minor
/// release, so downstream `match` expressions need a wildcard arm.  It
/// implements [`std::error::Error`], so `?` converts it into
/// `Box<dyn std::error::Error>` and `anyhow::Error`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Hdf5Error {
    /// A group, dataset or attribute was not found under the given path.
    NotFound(String),
    /// An item with the given name already exists.
    AlreadyExists(String),
    /// The requested hyperslab indices are out of range.
    HyperslabOutOfRange {
        /// The dimension index where the error occurred.
        dim: usize,
        /// The requested start offset.
        start: usize,
        /// The requested length.
        count: usize,
        /// The actual dimension size.
        size: usize,
    },
    /// The shapes or types are incompatible for the attempted operation.
    ShapeMismatch {
        /// Expected shape.
        expected: Vec<usize>,
        /// Provided shape.
        got: Vec<usize>,
    },
    /// The file is locked and cannot be written.
    FileLocked,
    /// A link target was not found.
    LinkTargetNotFound(String),
    /// Generic error carrying a human-readable message.
    Generic(String),
    /// An operating-system I/O error while reading or writing a `.h5` file.
    ///
    /// Carries the rendered [`std::io::Error`] so the enum stays `Clone` and
    /// `PartialEq`.
    Io(String),
    /// The bytes are not a well-formed HDF5 file (bad signature, truncated or
    /// corrupt metadata), or the in-memory tree cannot be laid out as one.
    InvalidFile(String),
    /// The HDF5 file is well formed but uses a feature the in-memory model has
    /// no representation for (for example an external link, an enumeration
    /// type or a compound member holding a string).
    ///
    /// Nothing is dropped silently: opening such a file fails with this error
    /// rather than returning a tree that is missing the object.
    Unsupported(String),
}

impl std::fmt::Display for Hdf5Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Hdf5Error::NotFound(p) => write!(f, "HDF5: not found: {p}"),
            Hdf5Error::AlreadyExists(p) => write!(f, "HDF5: already exists: {p}"),
            Hdf5Error::HyperslabOutOfRange {
                dim,
                start,
                count,
                size,
            } => write!(
                f,
                "HDF5: hyperslab out of range: dim={dim} start={start} count={count} size={size}"
            ),
            Hdf5Error::ShapeMismatch { expected, got } => {
                write!(
                    f,
                    "HDF5: shape mismatch: expected {expected:?}, got {got:?}"
                )
            }
            Hdf5Error::FileLocked => write!(f, "HDF5: file is locked"),
            Hdf5Error::LinkTargetNotFound(t) => {
                write!(f, "HDF5: link target not found: {t}")
            }
            Hdf5Error::Generic(msg) => write!(f, "HDF5: {msg}"),
            Hdf5Error::Io(msg) => write!(f, "HDF5: I/O error: {msg}"),
            Hdf5Error::InvalidFile(msg) => write!(f, "HDF5: invalid file: {msg}"),
            Hdf5Error::Unsupported(msg) => write!(f, "HDF5: unsupported: {msg}"),
        }
    }
}

impl std::error::Error for Hdf5Error {}

/// Convenient alias for results from this module.
pub type Hdf5Result<T> = Result<T, Hdf5Error>;

/// Check that a buffer of `got` elements has the `expected` length.
///
/// The writers in this module validate their caller-supplied slices with this
/// helper instead of `assert_eq!`, so a length mismatch is an ordinary
/// [`Hdf5Error::ShapeMismatch`] (`expected: [expected]`, `got: [got]`) rather
/// than a panic inside a function that already returns [`Hdf5Result`].
pub(crate) fn ensure_len(got: usize, expected: usize) -> Hdf5Result<()> {
    if got == expected {
        Ok(())
    } else {
        Err(Hdf5Error::ShapeMismatch {
            expected: vec![expected],
            got: vec![got],
        })
    }
}

/// Check that a flat buffer of `got` elements exactly fills an array of
/// extents `dims`.
///
/// The product of the extents is formed with overflow checking, so absurd
/// extents produce a [`Hdf5Error::ShapeMismatch`] (`expected: dims`,
/// `got: [got]`) instead of an arithmetic panic.  An empty `dims` (a scalar)
/// holds exactly one element.
pub(crate) fn ensure_volume(got: usize, dims: &[usize]) -> Hdf5Result<()> {
    match dims
        .iter()
        .try_fold(1_usize, |acc, &extent| acc.checked_mul(extent))
    {
        Some(volume) if volume == got => Ok(()),
        _ => Err(Hdf5Error::ShapeMismatch {
            expected: dims.to_vec(),
            got: vec![got],
        }),
    }
}

// ---------------------------------------------------------------------------
// 2.  Datatypes
// ---------------------------------------------------------------------------

/// Scalar element type of an HDF5 dataset.
#[derive(Debug, Clone, PartialEq)]
pub enum Hdf5Dtype {
    /// 32-bit IEEE float.
    Float32,
    /// 64-bit IEEE float.
    Float64,
    /// 32-bit signed integer.
    Int32,
    /// 64-bit signed integer (the default NumPy integer, used by most
    /// trajectory formats for step counters and particle ids).
    Int64,
    /// 64-bit unsigned integer.
    Uint64,
    /// 8-bit unsigned integer (byte).
    Uint8,
    /// Variable-length UTF-8 string.
    VlenString,
    /// Compound (struct-like) type described by a list of (name, dtype) fields.
    Compound(Vec<(String, Hdf5Dtype)>),
    /// A user-defined named type that aliases another dtype.
    Named {
        /// The user-supplied type name.
        name: String,
        /// The underlying base dtype.
        base: Box<Hdf5Dtype>,
    },
}

impl Hdf5Dtype {
    /// Return the byte-size of a single element (0 for variable-length types).
    pub fn element_size(&self) -> usize {
        match self {
            Hdf5Dtype::Float32 => 4,
            Hdf5Dtype::Float64 => 8,
            Hdf5Dtype::Int32 => 4,
            Hdf5Dtype::Int64 => 8,
            Hdf5Dtype::Uint64 => 8,
            Hdf5Dtype::Uint8 => 1,
            Hdf5Dtype::VlenString => 0,
            Hdf5Dtype::Compound(fields) => fields.iter().map(|(_, dt)| dt.element_size()).sum(),
            Hdf5Dtype::Named { base, .. } => base.element_size(),
        }
    }
}

// ---------------------------------------------------------------------------
// 3.  Storage layout helpers
// ---------------------------------------------------------------------------

/// Chunked storage descriptor for a dataset.
///
/// [`Hdf5File::save`](super::file::Hdf5File::save) turns it into a real HDF5
/// chunked layout and, when [`gzip_level`](Self::gzip_level) is non-zero, into
/// a DEFLATE filter pipeline, so the chunk shape and compression level are
/// honoured in the `.h5` file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkLayout {
    /// Chunk dimensions (same rank as dataset shape, every extent non-zero and
    /// no larger than the dataset's own extent).
    pub chunk_shape: Vec<usize>,
    /// DEFLATE (gzip) compression level: 0 stores the chunks uncompressed,
    /// 1 is fastest and 9 compresses best.
    pub gzip_level: u8,
}

impl ChunkLayout {
    /// Create a new chunk descriptor.
    pub fn new(chunk_shape: Vec<usize>, gzip_level: u8) -> Self {
        Self {
            chunk_shape,
            gzip_level: gzip_level.min(9),
        }
    }

    /// Return the number of elements per chunk.
    pub fn chunk_volume(&self) -> usize {
        self.chunk_shape.iter().product()
    }
}

/// Hyperslab selection: a start offset + count (length) per dimension.
#[derive(Debug, Clone)]
pub struct Hyperslab {
    /// Start index per dimension.
    pub start: Vec<usize>,
    /// Number of elements to select per dimension.
    pub count: Vec<usize>,
}

impl Hyperslab {
    /// Create a new hyperslab.
    ///
    /// Fails with [`Hdf5Error::ShapeMismatch`] (`expected` = the rank of
    /// `start`, `got` = the rank of `count`) when the two vectors have
    /// different lengths.
    pub fn new(start: Vec<usize>, count: Vec<usize>) -> Hdf5Result<Self> {
        ensure_len(count.len(), start.len())?;
        Ok(Self { start, count })
    }

    /// Total number of elements in the selection.
    pub fn volume(&self) -> usize {
        self.count.iter().product()
    }

    /// Validate this selection against a dataset shape.
    pub fn validate(&self, shape: &[usize]) -> Hdf5Result<()> {
        if self.start.len() != shape.len() {
            return Err(Hdf5Error::Generic(format!(
                "Hyperslab rank {} != dataset rank {}",
                self.start.len(),
                shape.len()
            )));
        }
        for (dim, (&s, (&c, &sz))) in self
            .start
            .iter()
            .zip(self.count.iter().zip(shape.iter()))
            .enumerate()
        {
            if s.checked_add(c).is_none_or(|end| end > sz) {
                return Err(Hdf5Error::HyperslabOutOfRange {
                    dim,
                    start: s,
                    count: c,
                    size: sz,
                });
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 4.  Attribute value
// ---------------------------------------------------------------------------

/// The value stored in an HDF5 attribute.
#[derive(Debug, Clone, PartialEq)]
pub enum AttrValue {
    /// Scalar 64-bit float.
    Float64(f64),
    /// Scalar 32-bit float.
    Float32(f32),
    /// Scalar 32-bit integer.
    Int32(i32),
    /// Scalar 64-bit signed integer.
    Int64(i64),
    /// Scalar 64-bit unsigned integer (step counters, atom counts, sizes).
    UInt64(u64),
    /// UTF-8 string.
    String(String),
    /// Array of 64-bit floats.
    ArrayF64(Vec<f64>),
    /// Array of 32-bit floats.
    ArrayF32(Vec<f32>),
    /// Array of 32-bit integers.
    ArrayI32(Vec<i32>),
    /// Array of 64-bit signed integers.
    ArrayI64(Vec<i64>),
    /// Array of UTF-8 strings.
    ArrayString(Vec<String>),
}

impl AttrValue {
    /// Store a count or size losslessly as a 64-bit unsigned attribute.
    ///
    /// `usize` never exceeds 64 bits on a supported target, so this is exact
    /// (unlike an `as i32` cast, which wraps above `i32::MAX`).
    pub fn from_count(count: usize) -> Self {
        AttrValue::UInt64(count as u64)
    }

    /// Interpret the attribute as a non-negative integer.
    ///
    /// Accepts [`AttrValue::UInt64`], [`AttrValue::Int64`] and the legacy
    /// [`AttrValue::Int32`] encoding that older files used for counters.
    /// Returns `None` for any other variant and for negative values.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            AttrValue::UInt64(v) => Some(*v),
            AttrValue::Int64(v) => u64::try_from(*v).ok(),
            AttrValue::Int32(v) => u64::try_from(*v).ok(),
            _ => None,
        }
    }

    /// Interpret the attribute as a `usize` count (see [`AttrValue::as_u64`]);
    /// `None` if the value does not fit.
    pub fn as_usize(&self) -> Option<usize> {
        self.as_u64().and_then(|v| usize::try_from(v).ok())
    }
}

// ---------------------------------------------------------------------------
// 5.  Dataset storage
// ---------------------------------------------------------------------------

/// In-memory storage for a single dataset element type.
#[derive(Debug, Clone)]
pub enum DataStorage {
    /// Float32 array.
    Float32(Vec<f32>),
    /// Float64 array.
    Float64(Vec<f64>),
    /// Int32 array.
    Int32(Vec<i32>),
    /// Int64 array.
    Int64(Vec<i64>),
    /// Uint64 array.
    Uint64(Vec<u64>),
    /// Uint8 array.
    Uint8(Vec<u8>),
    /// Variable-length strings (one per element).
    VlenString(Vec<String>),
    /// Compound: each element is a map from field-name to float64 value.
    Compound(Vec<HashMap<String, f64>>),
}

impl DataStorage {
    /// Return the number of elements stored.
    pub fn len(&self) -> usize {
        match self {
            DataStorage::Float32(v) => v.len(),
            DataStorage::Float64(v) => v.len(),
            DataStorage::Int32(v) => v.len(),
            DataStorage::Int64(v) => v.len(),
            DataStorage::Uint64(v) => v.len(),
            DataStorage::Uint8(v) => v.len(),
            DataStorage::VlenString(v) => v.len(),
            DataStorage::Compound(v) => v.len(),
        }
    }

    /// Return `true` if the storage is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ---------------------------------------------------------------------------
// 6.  Link types
// ---------------------------------------------------------------------------

/// An HDF5 link (soft or hard) stored in a group.
#[derive(Debug, Clone)]
pub enum Hdf5Link {
    /// Soft (symbolic) link: stores the target path as a string.
    Soft(String),
    /// Hard link: stores the target path (both point to same object).
    Hard(String),
}

// ---------------------------------------------------------------------------
// 7.  External dataset reference
// ---------------------------------------------------------------------------

/// A reference to a dataset living in an external file.
#[derive(Debug, Clone)]
pub struct ExternalRef {
    /// Simulated external filename.
    pub filename: String,
    /// Path inside the external file.
    pub dataset_path: String,
    /// Byte offset (simulated, 64-bit for large-file support).
    pub byte_offset: u64,
}

// ---------------------------------------------------------------------------
// 8.  Dimension scale descriptor
// ---------------------------------------------------------------------------

/// A dimension-scale association following the HDF5 DimensionScales convention.
#[derive(Debug, Clone)]
pub struct DimScale {
    /// Path of the dataset that serves as a scale.
    pub scale_dataset: String,
    /// Axis index this scale is attached to.
    pub axis: usize,
    /// Human-readable label for this axis.
    pub label: String,
}

// ---------------------------------------------------------------------------
// 9.  Collective I/O metadata
// ---------------------------------------------------------------------------

/// Mock metadata for a collective-I/O operation (Parallel HDF5 style).
#[derive(Debug, Clone)]
pub struct CollectiveIoMeta {
    /// Number of MPI ranks participating.
    pub n_ranks: usize,
    /// Rank that initiated the write.
    pub root_rank: usize,
    /// Total bytes transferred across all ranks.
    pub total_bytes: u64,
    /// Simulated wall-clock time in seconds.
    pub wall_time_s: f64,
}

// ---------------------------------------------------------------------------
// File lock simulation
// ---------------------------------------------------------------------------

/// File-level lock state for exclusive write access simulation.
#[derive(Debug, Clone, PartialEq)]
pub enum LockState {
    /// No lock held.
    Unlocked,
    /// An exclusive write lock is held by `owner_id`.
    WriteLocked {
        /// Simulated process/thread identifier.
        owner_id: u64,
    },
    /// A shared read lock is held by `n_readers`.
    ReadLocked {
        /// Number of concurrent readers.
        n_readers: usize,
    },
}

// ---------------------------------------------------------------------------
// Parallel HDF5 metadata
// ---------------------------------------------------------------------------

/// Simulated parallel HDF5 (PHDF5) metadata for a multi-rank write.
#[derive(Debug, Clone)]
pub struct ParallelHdf5Meta {
    /// Total number of MPI ranks.
    pub n_ranks: usize,
    /// Per-rank byte counts.
    pub rank_byte_counts: Vec<u64>,
    /// Whether collective metadata writes are enabled.
    pub collective_metadata: bool,
    /// POSIX advisory lock held during the collective operation.
    pub lock_held: bool,
}

impl ParallelHdf5Meta {
    /// Create metadata for a `n_ranks`-rank job.
    pub fn new(n_ranks: usize) -> Self {
        Self {
            n_ranks,
            rank_byte_counts: vec![0; n_ranks],
            collective_metadata: true,
            lock_held: false,
        }
    }

    /// Record the number of bytes written by rank `rank`.
    pub fn record_rank_bytes(&mut self, rank: usize, bytes: u64) {
        if rank < self.n_ranks {
            self.rank_byte_counts[rank] = bytes;
        }
    }

    /// Total bytes across all ranks.
    pub fn total_bytes(&self) -> u64 {
        self.rank_byte_counts.iter().sum()
    }
}

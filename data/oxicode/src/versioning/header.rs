//! Versioned data header format.

use super::version::Version;
use crate::{Error, Result};

/// Magic bytes for versioned data: "OXIV" (OXIcode Versioned)
pub const VERSIONED_MAGIC: [u8; 4] = [0x4F, 0x58, 0x49, 0x56];

/// Header format version
const HEADER_VERSION: u8 = 1;

/// Minimum header size: magic (4) + header_version (1) + version (6) = 11 bytes
const MIN_HEADER_SIZE: usize = 11;

/// Header for versioned data.
///
/// Format:
/// - Bytes 0-3: Magic "OXIV"
/// - Byte 4: Header format version
/// - Bytes 5-10: Data version (major:u16 + minor:u16 + patch:u16, little-endian)
/// - Bytes 11+: Optional metadata (reserved for future use)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VersionedHeader {
    header_version: u8,
    version: Version,
}

impl VersionedHeader {
    /// Create a new header with the given data version.
    #[inline]
    pub const fn new(version: Version) -> Self {
        Self {
            header_version: HEADER_VERSION,
            version,
        }
    }

    /// Get the data version.
    #[inline]
    pub const fn version(&self) -> Version {
        self.version
    }

    /// Get the header format version.
    #[inline]
    pub const fn header_version(&self) -> u8 {
        self.header_version
    }

    /// Get the total header size in bytes.
    #[inline]
    pub const fn header_size(&self) -> usize {
        MIN_HEADER_SIZE
    }

    /// Convert to bytes.
    pub fn to_bytes(&self) -> [u8; MIN_HEADER_SIZE] {
        let mut bytes = [0u8; MIN_HEADER_SIZE];

        // Magic
        bytes[0..4].copy_from_slice(&VERSIONED_MAGIC);

        // Header version
        bytes[4] = self.header_version;

        // Data version
        let version_bytes = self.version.to_bytes();
        bytes[5..11].copy_from_slice(&version_bytes);

        bytes
    }

    /// Parse from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < MIN_HEADER_SIZE {
            return Err(Error::UnexpectedEnd {
                additional: MIN_HEADER_SIZE - data.len(),
            });
        }

        // Check magic
        if data[0..4] != VERSIONED_MAGIC {
            return Err(Error::InvalidData {
                message: "invalid version header magic",
            });
        }

        // Header version
        let header_version = data[4];
        if header_version > HEADER_VERSION {
            return Err(Error::InvalidData {
                message: "unsupported header version",
            });
        }

        // Data version
        let version = Version::from_bytes(&data[5..11]).ok_or(Error::InvalidData {
            message: "invalid version bytes",
        })?;

        Ok(Self {
            header_version,
            version,
        })
    }
}

impl Default for VersionedHeader {
    fn default() -> Self {
        Self::new(Version::zero())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let header = VersionedHeader::new(Version::new(1, 2, 3));
        let bytes = header.to_bytes();
        let parsed = VersionedHeader::from_bytes(&bytes).expect("parse failed");

        assert_eq!(header, parsed);
    }

    #[test]
    fn test_header_magic() {
        let header = VersionedHeader::new(Version::new(1, 0, 0));
        let bytes = header.to_bytes();

        assert_eq!(&bytes[0..4], &VERSIONED_MAGIC);
    }

    #[test]
    fn test_invalid_magic() {
        let mut bytes = VersionedHeader::new(Version::new(1, 0, 0)).to_bytes();
        bytes[0] = 0xFF; // Corrupt magic

        let result = VersionedHeader::from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_header_size() {
        let header = VersionedHeader::new(Version::new(1, 0, 0));
        assert_eq!(header.header_size(), MIN_HEADER_SIZE);
    }

    #[test]
    fn test_insufficient_data() {
        let data = [0u8; 5]; // Too short
        let result = VersionedHeader::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_version_extraction() {
        let version = Version::new(5, 10, 15);
        let header = VersionedHeader::new(version);
        let bytes = header.to_bytes();
        let parsed = VersionedHeader::from_bytes(&bytes).expect("parse failed");

        assert_eq!(parsed.version(), version);
    }
}

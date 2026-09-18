//! The plain HDT header section: an opaque, `ntriples`-formatted metadata payload.
//!
//! Per the specification, the header carries dataset metadata (authorship, statistics, a VoID
//! description, ...) as plain RDF, conventionally serialized as N-Triples. This crate treats
//! that payload as an opaque byte blob — it neither generates a default VoID header nor parses
//! the one it reads back — leaving header content entirely up to the caller (see
//! [`crate::writer::HdtWriter::with_header`]).
//!
//! # Wire format
//!
//! ```text
//! ControlInformation   type=Header, format="ntriples", properties: length=<byte count>
//! bytes                 the header payload, verbatim, `length` bytes long
//! ```
//!
//! Unlike the dictionary and triples sections, the specification does not describe the header
//! payload itself as CRC-protected (only its [`ControlInformation`] preamble is, via CRC16).

use crate::control_info::{ControlInfoType, ControlInformation};
use crate::error::{HdtError, Result};

/// The section name used in diagnostics for this module.
const SECTION: &str = "header";

/// The header's `format` value: a plain-text N-Triples payload.
const HEADER_FORMAT: &str = "ntriples";

/// An HDT header: an opaque `ntriples`-formatted metadata payload.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Header {
    bytes: Vec<u8>,
}

impl Header {
    /// Wraps `bytes` as a header payload, verbatim.
    #[must_use]
    pub fn new(bytes: impl Into<Vec<u8>>) -> Self {
        Self {
            bytes: bytes.into(),
        }
    }

    /// An empty header (`0` bytes of payload).
    #[must_use]
    pub const fn empty() -> Self {
        Self { bytes: Vec::new() }
    }

    /// The raw header payload bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Serializes this header, appending to `out`.
    pub fn write(&self, out: &mut Vec<u8>) {
        let info = ControlInformation::new(ControlInfoType::Header, HEADER_FORMAT)
            .with_property("length", self.bytes.len().to_string());
        info.write(out);
        out.extend_from_slice(&self.bytes);
    }

    /// Deserializes a header previously written by [`write`](Self::write), starting at `*pos`
    /// and advancing it past it.
    ///
    /// # Errors
    ///
    /// Returns an [`HdtError`] if the control information is malformed or the declared payload
    /// length runs past the end of `data`.
    pub fn read(data: &[u8], pos: &mut usize) -> Result<Self> {
        let info = ControlInformation::read(data, pos, SECTION)?;
        if info.control_type != Some(ControlInfoType::Header) {
            return Err(HdtError::UnexpectedControlType {
                section: SECTION,
                found: info.control_type.map_or(0, |type_| type_ as u8),
            });
        }
        // Unlike the dictionary, triples, and global sections, `info.format` is deliberately not
        // checked against `HEADER_FORMAT` here: this type treats the payload as an opaque blob
        // regardless of its declared format (see the module docs), so an unexpected `format`
        // value changes nothing about how the following bytes are read. Rejecting it would only
        // add a way for an otherwise-harmless file to fail to load, with no matching decoder-side
        // benefit — unlike the sibling sections, where an unrecognized format really does mean
        // the following bytes cannot be decoded correctly.
        //
        // `length` comes straight from the file's free-form control-information properties, so it
        // is fully attacker-controlled: `usize::try_from` and `checked_add` keep an absurd value
        // (`u64::MAX`, say) an `HdtError` instead of an arithmetic overflow panic in debug builds.
        let length = usize::try_from(info.property_u64("length"))
            .map_err(|_| HdtError::UnexpectedEof { section: SECTION })?;
        let end = pos
            .checked_add(length)
            .ok_or(HdtError::UnexpectedEof { section: SECTION })?;
        let bytes = data
            .get(*pos..end)
            .ok_or(HdtError::UnexpectedEof { section: SECTION })?
            .to_vec();
        *pos = end;
        Ok(Self { bytes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_payload() -> Result<()> {
        let header = Header::new(&b"<urn:a> <urn:b> <urn:c> .\n"[..]);
        let mut buffer = Vec::new();
        header.write(&mut buffer);
        let mut pos = 0;
        let decoded = Header::read(&buffer, &mut pos)?;
        assert_eq!(pos, buffer.len());
        assert_eq!(decoded, header);
        Ok(())
    }

    #[test]
    fn roundtrips_empty() -> Result<()> {
        let header = Header::empty();
        let mut buffer = Vec::new();
        header.write(&mut buffer);
        let mut pos = 0;
        let decoded = Header::read(&buffer, &mut pos)?;
        assert!(decoded.as_bytes().is_empty());
        Ok(())
    }

    /// Regression test: a crafted `length` property must not overflow the position arithmetic.
    #[test]
    fn rejects_absurd_declared_length() {
        for declared in [u64::MAX, u64::MAX - 1, u64::from(u32::MAX) << 32] {
            let mut buffer = Vec::new();
            ControlInformation::new(ControlInfoType::Header, HEADER_FORMAT)
                .with_property("length", declared.to_string())
                .write(&mut buffer);
            buffer.extend_from_slice(b"hello");
            let mut pos = 0;
            let error = Header::read(&buffer, &mut pos).expect_err("expected an error");
            assert!(matches!(error, HdtError::UnexpectedEof { .. }));
        }
    }

    #[test]
    fn rejects_truncated_payload() {
        let header = Header::new(&b"hello"[..]);
        let mut buffer = Vec::new();
        header.write(&mut buffer);
        buffer.truncate(buffer.len() - 2);
        let mut pos = 0;
        let error = Header::read(&buffer, &mut pos).expect_err("expected an error");
        assert!(matches!(error, HdtError::UnexpectedEof { .. }));
    }
}

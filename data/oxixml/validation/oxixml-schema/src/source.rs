//! Turning the *bytes* of a schema document into the text the parser wants.
//!
//! A schema document is an XML document like any other, so its encoding is
//! whatever [XML 1.0 Appendix F][sniff] says it is: a byte-order mark, then the
//! four-byte prologue pattern, then the `encoding="…"` pseudo-attribute of the
//! XML declaration. Reading the file with [`std::fs::read_to_string`] instead
//! assumes UTF-8, which rejects every UTF-16 schema document out of hand — and
//! the W3C test suite ships a good number of them (`ibmData/valid/S3_3_4`,
//! `ibmData/valid/S3_2_3`, `ibmData/instance_invalid/D4_3_15`, …).
//!
//! [sniff]: https://www.w3.org/TR/xml/#sec-guessing

use oxixml_encoding::sniff_xml_encoding;

/// Decodes the bytes of a schema document, honouring its byte-order mark and
/// its `encoding` pseudo-attribute.
///
/// # Errors
///
/// Returns a human-readable description when the bytes are not valid in the
/// encoding they announce.
pub(crate) fn decode(bytes: &[u8]) -> Result<String, String> {
    let sniffed = sniff_xml_encoding(bytes);
    let encoding = sniffed.best_effort_encoding();
    let body = bytes.get(sniffed.bom_len..).unwrap_or(&[]);
    let mut text = String::with_capacity(body.len());
    let mut decoder = encoding.new_decoder();
    decoder
        .decode_to_string(body, &mut text, true)
        .map_err(|error| format!("cannot decode as {encoding}: {error}"))?;
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::decode;

    #[test]
    fn plain_utf8_is_returned_unchanged() {
        assert_eq!(decode(b"<root/>").as_deref(), Ok("<root/>"));
    }

    #[test]
    fn a_utf16_document_with_a_byte_order_mark_is_decoded() {
        let mut bytes = vec![0xFF, 0xFE];
        for unit in "<root/>".encode_utf16() {
            bytes.extend_from_slice(&unit.to_le_bytes());
        }
        assert_eq!(decode(&bytes).as_deref(), Ok("<root/>"));
    }

    #[test]
    fn a_utf16_document_without_a_byte_order_mark_is_decoded() {
        let mut bytes = Vec::new();
        for unit in "<?xml version=\"1.0\"?><root/>".encode_utf16() {
            bytes.extend_from_slice(&unit.to_be_bytes());
        }
        let text = decode(&bytes).expect("a UTF-16BE prologue is recognized");
        assert!(text.ends_with("<root/>"), "decoded `{text}`");
    }
}

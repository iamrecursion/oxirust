//! The object identifiers a certificate path validator has to recognise, and
//! the dotted-decimal rendering its diagnostics are written in.
//!
//! Each constant is the *contents* of a DER `OBJECT IDENTIFIER` — the octets
//! after the tag and length, which is exactly what
//! [`Der::read`](crate::crypto::der::Der::read) hands back — so comparing an
//! identifier read out of a certificate against one of these is a slice
//! comparison and nothing more.

/// Renders the contents of a DER `OBJECT IDENTIFIER` as dotted decimal.
///
/// This exists for error messages: an extension this crate cannot process has
/// to be reported *by name*, and its name is its OID. The rendering is total —
/// a truncated or absurdly large subidentifier becomes `?` rather than a
/// failure, because a diagnostic that cannot be produced is worse than an
/// imprecise one.
pub(crate) fn dotted(contents: &[u8]) -> String {
    let mut rendered = String::new();
    let mut value = 0_u128;
    let mut overflowed = false;
    let mut incomplete = false;
    let mut first = true;
    for byte in contents {
        incomplete = true;
        match value.checked_mul(128) {
            Some(shifted) => value = shifted | u128::from(byte & 0x7f),
            None => overflowed = true,
        }
        if byte & 0x80 != 0 {
            continue;
        }
        if !first {
            rendered.push('.');
        }
        if overflowed {
            rendered.push('?');
        } else if first {
            // X.690 §8.19.4 packs the first two arcs into one subidentifier:
            // the first is 0, 1 or 2, and the second is what is left over.
            let leading = if value < 80 { value / 40 } else { 2 };
            rendered.push_str(&leading.to_string());
            rendered.push('.');
            rendered.push_str(&(value - leading * 40).to_string());
        } else {
            rendered.push_str(&value.to_string());
        }
        first = false;
        value = 0;
        overflowed = false;
        incomplete = false;
    }
    if incomplete {
        if !first {
            rendered.push('.');
        }
        rendered.push('?');
    }
    rendered
}

/// `1.3.14.3.2.26` — `id-sha1`.
pub(crate) const SHA1: &[u8] = &[0x2b, 0x0e, 0x03, 0x02, 0x1a];
/// `2.16.840.1.101.3.4.2.4` — `id-sha224`.
pub(crate) const SHA224: &[u8] = &[0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x04];
/// `2.16.840.1.101.3.4.2.1` — `id-sha256`.
pub(crate) const SHA256: &[u8] = &[0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01];
/// `2.16.840.1.101.3.4.2.2` — `id-sha384`.
pub(crate) const SHA384: &[u8] = &[0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x02];
/// `2.16.840.1.101.3.4.2.3` — `id-sha512`.
pub(crate) const SHA512: &[u8] = &[0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x03];

/// `1.2.840.113549.1.1.5` — `sha1WithRSAEncryption`.
pub(crate) const RSA_SHA1: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x05];
/// `1.2.840.113549.1.1.14` — `sha224WithRSAEncryption`.
pub(crate) const RSA_SHA224: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0e];
/// `1.2.840.113549.1.1.11` — `sha256WithRSAEncryption`.
pub(crate) const RSA_SHA256: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0b];
/// `1.2.840.113549.1.1.12` — `sha384WithRSAEncryption`.
pub(crate) const RSA_SHA384: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0c];
/// `1.2.840.113549.1.1.13` — `sha512WithRSAEncryption`.
pub(crate) const RSA_SHA512: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0d];
/// `1.2.840.113549.1.1.10` — `id-RSASSA-PSS`.
pub(crate) const RSASSA_PSS: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0a];
/// `1.2.840.113549.1.1.8` — `id-mgf1`.
pub(crate) const MGF1: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x08];

/// `1.2.840.10040.4.3` — `id-dsa-with-sha1`.
pub(crate) const DSA_SHA1: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x38, 0x04, 0x03];
/// `2.16.840.1.101.3.4.3.1` — `id-dsa-with-sha224`.
pub(crate) const DSA_SHA224: &[u8] = &[0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x03, 0x01];
/// `2.16.840.1.101.3.4.3.2` — `id-dsa-with-sha256`.
pub(crate) const DSA_SHA256: &[u8] = &[0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x03, 0x02];

/// `1.2.840.10045.4.1` — `ecdsa-with-SHA1`.
pub(crate) const ECDSA_SHA1: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x04, 0x01];
/// `1.2.840.10045.4.3.1` — `ecdsa-with-SHA224`.
pub(crate) const ECDSA_SHA224: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x04, 0x03, 0x01];
/// `1.2.840.10045.4.3.2` — `ecdsa-with-SHA256`.
pub(crate) const ECDSA_SHA256: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x04, 0x03, 0x02];
/// `1.2.840.10045.4.3.3` — `ecdsa-with-SHA384`.
pub(crate) const ECDSA_SHA384: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x04, 0x03, 0x03];
/// `1.2.840.10045.4.3.4` — `ecdsa-with-SHA512`.
pub(crate) const ECDSA_SHA512: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x04, 0x03, 0x04];

/// `2.5.29.14` — `id-ce-subjectKeyIdentifier`.
pub(crate) const SUBJECT_KEY_IDENTIFIER: &[u8] = &[0x55, 0x1d, 0x0e];
/// `2.5.29.15` — `id-ce-keyUsage`.
pub(crate) const KEY_USAGE: &[u8] = &[0x55, 0x1d, 0x0f];
/// `2.5.29.19` — `id-ce-basicConstraints`.
pub(crate) const BASIC_CONSTRAINTS: &[u8] = &[0x55, 0x1d, 0x13];
/// `2.5.29.35` — `id-ce-authorityKeyIdentifier`.
pub(crate) const AUTHORITY_KEY_IDENTIFIER: &[u8] = &[0x55, 0x1d, 0x23];

/// The RFC 4514 short name of a directory attribute type, for the compact
/// rendering [`DistinguishedName`](super::DistinguishedName) writes.
///
/// [RFC 4514 §3](https://www.rfc-editor.org/rfc/rfc4514#section-3) fixes the
/// first nine; `UID` and `emailAddress` are the two additions common enough in
/// certificates that printing them as a dotted OID would only obscure the
/// message.
pub(crate) fn attribute_short_name(oid: &[u8]) -> Option<&'static str> {
    match oid {
        [0x55, 0x04, 0x03] => Some("CN"),
        [0x55, 0x04, 0x06] => Some("C"),
        [0x55, 0x04, 0x07] => Some("L"),
        [0x55, 0x04, 0x08] => Some("ST"),
        [0x55, 0x04, 0x09] => Some("STREET"),
        [0x55, 0x04, 0x0a] => Some("O"),
        [0x55, 0x04, 0x0b] => Some("OU"),
        [0x09, 0x92, 0x26, 0x89, 0x93, 0xf2, 0x2c, 0x64, 0x01, 0x19] => Some("DC"),
        [0x09, 0x92, 0x26, 0x89, 0x93, 0xf2, 0x2c, 0x64, 0x01, 0x01] => Some("UID"),
        [0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x09, 0x01] => Some("emailAddress"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_two_arcs_are_unpacked() {
        assert_eq!(dotted(SHA256), "2.16.840.1.101.3.4.2.1");
        assert_eq!(dotted(RSA_SHA256), "1.2.840.113549.1.1.11");
        assert_eq!(dotted(ECDSA_SHA384), "1.2.840.10045.4.3.3");
        assert_eq!(dotted(BASIC_CONSTRAINTS), "2.5.29.19");
        assert_eq!(dotted(SHA1), "1.3.14.3.2.26");
        // The one place the packing is not simply `first / 40`: an arc of 0 or 1
        // can never exceed 39, so anything from 80 up belongs to arc 2.
        assert_eq!(dotted(&[0x00]), "0.0");
        assert_eq!(dotted(&[0x28]), "1.0");
        assert_eq!(dotted(&[0x50]), "2.0");
    }

    #[test]
    fn the_directory_attribute_names_round_trip() {
        assert_eq!(attribute_short_name(&[0x55, 0x04, 0x03]), Some("CN"));
        assert_eq!(dotted(&[0x55, 0x04, 0x03]), "2.5.4.3");
        assert_eq!(
            attribute_short_name(&[0x09, 0x92, 0x26, 0x89, 0x93, 0xf2, 0x2c, 0x64, 0x01, 0x19]),
            Some("DC")
        );
        assert_eq!(
            dotted(&[0x09, 0x92, 0x26, 0x89, 0x93, 0xf2, 0x2c, 0x64, 0x01, 0x19]),
            "0.9.2342.19200300.100.1.25"
        );
        assert_eq!(attribute_short_name(&[0x55, 0x04, 0x2a]), None);
    }

    /// A diagnostic must always be printable, however broken the identifier it
    /// is describing.
    #[test]
    fn a_malformed_identifier_still_renders() {
        // A final subidentifier whose continuation bit is set but which the
        // encoding never finishes.
        assert_eq!(dotted(&[0x55, 0x1d, 0x80]), "2.5.29.?");
        assert_eq!(dotted(&[0x80]), "?");
        assert_eq!(dotted(&[]), "");
        // Twenty continuation octets overflow any fixed-width accumulator and
        // never finish the subidentifier they start.
        let huge = [0xff_u8; 20];
        assert_eq!(dotted(&huge), "?");
        // The same overflow, but terminated: the arc itself is unprintable.
        let mut terminated = huge.to_vec();
        terminated.push(0x01);
        assert_eq!(dotted(&terminated), "?");
    }
}

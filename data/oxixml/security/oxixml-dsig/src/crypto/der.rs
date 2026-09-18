//! A minimal DER reader, just enough to get a key out of the encodings that turn
//! up inside a `<ds:KeyInfo>`.
//!
//! These shapes are recognised, all of them
//! [X.690](https://www.itu.int/rec/T-REC-X.690) Distinguished Encoding Rules:
//!
//! | shape | where it appears |
//! |---|---|
//! | `SubjectPublicKeyInfo` ([RFC 5280 §4.1](https://www.rfc-editor.org/rfc/rfc5280#section-4.1)) | `<dsig11:DEREncodedKeyValue>` |
//! | `Certificate` ([RFC 5280 §4.1](https://www.rfc-editor.org/rfc/rfc5280#section-4.1)) | `<ds:X509Certificate>` |
//! | `RSAPublicKey` ([RFC 8017 App. A.1.1](https://www.rfc-editor.org/rfc/rfc8017#appendix-A.1.1)) | inside an RSA `SubjectPublicKeyInfo` |
//! | an uncompressed SEC 1 point, `Dss-Parms` and a `DSAPublicKey` INTEGER | inside an EC or DSA `SubjectPublicKeyInfo` |
//! | `RSAPrivateKey`, SEC 1 `ECPrivateKey`, PKCS#8 `PrivateKeyInfo` | keys handed to [`crate::sign`] |
//!
//! Nothing *here* validates a certificate: the functions in this module walk
//! straight down to `subjectPublicKeyInfo` and treat the certificate as a
//! *container* for a public key. Reading the rest of it — issuer and subject
//! names, validity dates, extensions — and checking a certification path is
//! [`crate::x509`]'s job, built on the same reader.
//!
//! The reader is strict in the ways that matter for security: lengths must be
//! minimally encoded, an INTEGER must not have a redundant leading octet, and
//! every structure must be consumed exactly.

use oxixml_support::ec::CurveId;

/// A cursor over DER octets.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Der<'a> {
    input: &'a [u8],
    position: usize,
}

/// A DER tag byte.
pub(crate) mod tag {
    /// `BOOLEAN`.
    pub(crate) const BOOLEAN: u8 = 0x01;
    /// `INTEGER`.
    pub(crate) const INTEGER: u8 = 0x02;
    /// `BIT STRING`.
    pub(crate) const BIT_STRING: u8 = 0x03;
    /// `OCTET STRING`.
    pub(crate) const OCTET_STRING: u8 = 0x04;
    /// `NULL`.
    pub(crate) const NULL: u8 = 0x05;
    /// `OBJECT IDENTIFIER`.
    pub(crate) const OID: u8 = 0x06;
    /// `UTCTime` — the two-digit-year form of an X.509 `Validity`.
    pub(crate) const UTC_TIME: u8 = 0x17;
    /// `GeneralizedTime` — the four-digit-year form.
    pub(crate) const GENERALIZED_TIME: u8 = 0x18;
    /// `SEQUENCE` (constructed).
    pub(crate) const SEQUENCE: u8 = 0x30;
    /// `SET` (constructed) — an X.500 `RelativeDistinguishedName`.
    pub(crate) const SET: u8 = 0x31;
    /// `[0]` primitive, context-specific — the `keyIdentifier` of an
    /// `AuthorityKeyIdentifier`, which is IMPLICIT and so keeps the primitive
    /// form of the `OCTET STRING` it replaces.
    pub(crate) const IMPLICIT_0: u8 = 0x80;
    /// `[1]` primitive, context-specific — the X.509 `issuerUniqueID`.
    pub(crate) const IMPLICIT_1: u8 = 0x81;
    /// `[2]` primitive, context-specific — the X.509 `subjectUniqueID`, and the
    /// `authorityCertSerialNumber` of an `AuthorityKeyIdentifier`.
    pub(crate) const IMPLICIT_2: u8 = 0x82;
    /// `[0]` constructed, context-specific — the X.509 `version` wrapper and the
    /// SEC 1 `parameters` field.
    pub(crate) const CONTEXT_0: u8 = 0xa0;
    /// `[1]` constructed, context-specific — the SEC 1 `publicKey` field.
    pub(crate) const CONTEXT_1: u8 = 0xa1;
    /// `[2]` constructed, context-specific — the `saltLength` of an
    /// `RSASSA-PSS-params`.
    pub(crate) const CONTEXT_2: u8 = 0xa2;
    /// `[3]` constructed, context-specific — the X.509 `extensions` wrapper.
    pub(crate) const CONTEXT_3: u8 = 0xa3;
}

/// The OID `1.2.840.113549.1.1.1` (`rsaEncryption`), DER-encoded contents.
pub(crate) const RSA_ENCRYPTION_OID: &[u8] =
    &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x01];

/// The OID `1.2.840.10045.2.1` (`id-ecPublicKey`), DER-encoded contents.
pub(crate) const EC_PUBLIC_KEY_OID: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02, 0x01];

/// The OID `1.2.840.10040.4.1` (`id-dsa`), DER-encoded contents.
pub(crate) const DSA_OID: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x38, 0x04, 0x01];

/// The OID `1.2.840.10045.3.1.7` (`prime256v1`, NIST P-256), DER-encoded contents.
pub(crate) const P256_OID: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07];

/// The OID `1.3.132.0.34` (`secp384r1`, NIST P-384), DER-encoded contents.
pub(crate) const P384_OID: &[u8] = &[0x2b, 0x81, 0x04, 0x00, 0x22];

/// The named curve an OID selects, or [`None`] for a curve this crate does not
/// implement.
#[must_use]
pub(crate) fn curve_from_oid(oid: &[u8]) -> Option<CurveId> {
    match oid {
        P256_OID => Some(CurveId::P256),
        P384_OID => Some(CurveId::P384),
        _ => None,
    }
}

/// What went wrong while reading DER.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DerError {
    /// The input ended in the middle of a value.
    Truncated,
    /// A tag other than the one the grammar requires.
    UnexpectedTag {
        /// The tag the grammar wanted.
        expected: u8,
        /// The tag that was there.
        found: u8,
    },
    /// A length that is not the shortest encoding, or one that does not fit in
    /// a `usize`.
    BadLength,
    /// An INTEGER with a redundant leading `0x00`/`0xff`, or a negative one
    /// where the grammar requires an unsigned value.
    BadInteger,
    /// A BIT STRING whose unused-bit count is not zero where the grammar needs
    /// whole octets, is above seven, or leaves a non-zero bit unused.
    UnalignedBitString,
    /// A BOOLEAN whose contents are not the single octet `0x00` or `0xff`.
    BadBoolean,
    /// Octets left over after the outermost structure.
    TrailingData,
    /// The key is not RSA.
    NotRsa,
    /// The key is not of the algorithm the caller asked for.
    WrongKeyAlgorithm,
    /// The `AlgorithmIdentifier` names an algorithm this crate has no key type
    /// for.
    UnknownKeyAlgorithm,
    /// An elliptic-curve key names a curve other than P-256 or P-384.
    UnsupportedCurve,
    /// A PKCS#8 `PrivateKeyInfo` wraps another key structure more times than
    /// [`MAX_PRIVATE_KEY_NESTING`] allows.
    TooDeeplyNested,
}

impl std::fmt::Display for DerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated => f.write_str("the DER encoding ends in the middle of a value"),
            Self::UnexpectedTag { expected, found } => {
                write!(f, "expected DER tag {expected:#04x}, found {found:#04x}")
            }
            Self::BadLength => f.write_str("a DER length is not minimally encoded"),
            Self::BadInteger => f.write_str("a DER INTEGER is negative or not minimally encoded"),
            Self::UnalignedBitString => {
                f.write_str("a DER BIT STRING has a malformed unused-bit count")
            }
            Self::BadBoolean => f.write_str("a DER BOOLEAN is not the single octet 0x00 or 0xff"),
            Self::TrailingData => f.write_str("octets remain after the DER structure"),
            Self::NotRsa => f.write_str("the key is not an RSA key"),
            Self::WrongKeyAlgorithm => {
                f.write_str("the key is not of the algorithm the signature method needs")
            }
            Self::UnknownKeyAlgorithm => {
                f.write_str("the key names an algorithm this crate does not implement")
            }
            Self::UnsupportedCurve => f.write_str(
                "the elliptic-curve key names a curve other than P-256 or P-384, the two \
                 XML Signature 1.1 requires",
            ),
            Self::TooDeeplyNested => write!(
                f,
                "the key wraps another key structure more than {MAX_PRIVATE_KEY_NESTING} times"
            ),
        }
    }
}

type DerResult<T> = Result<T, DerError>;

impl<'a> Der<'a> {
    /// Wraps a slice.
    pub(crate) const fn new(input: &'a [u8]) -> Self {
        Self { input, position: 0 }
    }

    /// Whether every octet has been consumed.
    pub(crate) const fn is_empty(&self) -> bool {
        self.position >= self.input.len()
    }

    /// Fails unless every octet has been consumed.
    pub(crate) fn finish(&self) -> DerResult<()> {
        if self.is_empty() {
            Ok(())
        } else {
            Err(DerError::TrailingData)
        }
    }

    fn take(&mut self, len: usize) -> DerResult<&'a [u8]> {
        let end = self.position.checked_add(len).ok_or(DerError::BadLength)?;
        let slice = self
            .input
            .get(self.position..end)
            .ok_or(DerError::Truncated)?;
        self.position = end;
        Ok(slice)
    }

    /// The tag of the next TLV, without consuming anything.
    pub(crate) fn peek_tag(&self) -> DerResult<u8> {
        self.input
            .get(self.position)
            .copied()
            .ok_or(DerError::Truncated)
    }

    /// Reads one TLV with the given tag and returns its contents.
    pub(crate) fn read(&mut self, expected: u8) -> DerResult<&'a [u8]> {
        let found = self.peek_tag()?;
        if found != expected {
            return Err(DerError::UnexpectedTag { expected, found });
        }
        self.position += 1;
        let len = self.read_length()?;
        self.take(len)
    }

    /// Reads one TLV with the given tag, returning [`None`] (and consuming
    /// nothing) if the next tag is different.
    pub(crate) fn read_optional(&mut self, expected: u8) -> DerResult<Option<&'a [u8]>> {
        if self.is_empty() || self.peek_tag()? != expected {
            return Ok(None);
        }
        self.read(expected).map(Some)
    }

    /// Reads a `SEQUENCE` and returns a reader over its contents.
    pub(crate) fn read_sequence(&mut self) -> DerResult<Der<'a>> {
        self.read(tag::SEQUENCE).map(Der::new)
    }

    /// Reads one TLV with the given tag and returns the *whole* encoding — tag
    /// and length octets included — alongside its contents.
    ///
    /// An X.509 signature is computed over the `tbsCertificate` exactly as it
    /// was encoded, and two structures are compared by their encodings rather
    /// than by their meaning, so both need the octets a plain
    /// [`read`](Der::read) throws away.
    pub(crate) fn read_tagged(&mut self, expected: u8) -> DerResult<(&'a [u8], &'a [u8])> {
        let start = self.position;
        let contents = self.read(expected)?;
        let whole = self
            .input
            .get(start..self.position)
            .ok_or(DerError::Truncated)?;
        Ok((whole, contents))
    }

    /// Reads one TLV of whatever tag comes next, returning the tag, the whole
    /// encoding and the contents.
    pub(crate) fn read_any(&mut self) -> DerResult<(u8, &'a [u8], &'a [u8])> {
        let tag = self.peek_tag()?;
        let (whole, contents) = self.read_tagged(tag)?;
        Ok((tag, whole, contents))
    }

    /// Reads a `BOOLEAN`, which DER pins to exactly `0x00` or `0xff`.
    pub(crate) fn read_boolean(&mut self) -> DerResult<bool> {
        match self.read(tag::BOOLEAN)? {
            [0x00] => Ok(false),
            [0xff] => Ok(true),
            _ => Err(DerError::BadBoolean),
        }
    }

    /// Reads a `BIT STRING` that may legitimately end mid-octet — the shape the
    /// X.509 `KeyUsage` extension uses — and returns the unused-bit count with
    /// the octets that carry the bits.
    ///
    /// DER requires the unused bits themselves to be zero, and that is checked
    /// here: a `KeyUsage` with rubbish in the padding is not a `KeyUsage` this
    /// crate will read a permission out of.
    pub(crate) fn read_bit_string_padded(&mut self) -> DerResult<(u8, &'a [u8])> {
        let contents = self.read(tag::BIT_STRING)?;
        let (unused, bits) = contents.split_first().ok_or(DerError::Truncated)?;
        if *unused > 7 || (*unused > 0 && bits.is_empty()) {
            return Err(DerError::UnalignedBitString);
        }
        if *unused > 0 {
            let mask = 0xff_u8 >> (8 - *unused);
            if bits.last().is_some_and(|byte| byte & mask != 0) {
                return Err(DerError::UnalignedBitString);
            }
        }
        Ok((*unused, bits))
    }

    fn read_length(&mut self) -> DerResult<usize> {
        let first = *self.input.get(self.position).ok_or(DerError::Truncated)?;
        self.position += 1;
        if first < 0x80 {
            return Ok(usize::from(first));
        }
        if first == 0x80 || first == 0xff {
            // Indefinite length and the reserved form are both illegal in DER.
            return Err(DerError::BadLength);
        }
        let count = usize::from(first & 0x7f);
        if count > core::mem::size_of::<usize>() {
            return Err(DerError::BadLength);
        }
        let octets = self.take(count)?;
        if octets.first().copied() == Some(0) {
            // A leading zero means the length was not minimally encoded.
            return Err(DerError::BadLength);
        }
        let mut value = 0_usize;
        for byte in octets {
            value = value.checked_shl(8).ok_or(DerError::BadLength)? | usize::from(*byte);
        }
        if value < 0x80 {
            return Err(DerError::BadLength);
        }
        Ok(value)
    }

    /// Reads an `INTEGER` and returns its magnitude, big-endian, with the DER
    /// sign octet removed.
    pub(crate) fn read_unsigned_integer(&mut self) -> DerResult<&'a [u8]> {
        let contents = self.read(tag::INTEGER)?;
        let first = contents.first().copied().ok_or(DerError::BadInteger)?;
        if first & 0x80 != 0 {
            // Negative: not an unsigned value.
            return Err(DerError::BadInteger);
        }
        if contents.len() > 1 {
            let second = contents.get(1).copied().ok_or(DerError::BadInteger)?;
            if first == 0x00 && second & 0x80 == 0 {
                return Err(DerError::BadInteger);
            }
        }
        Ok(if first == 0x00 && contents.len() > 1 {
            contents.get(1..).unwrap_or_default()
        } else {
            contents
        })
    }

    /// Reads a `BIT STRING` whose unused-bit count must be zero.
    pub(crate) fn read_bit_string(&mut self) -> DerResult<&'a [u8]> {
        let contents = self.read(tag::BIT_STRING)?;
        match contents.split_first() {
            Some((0, rest)) => Ok(rest),
            Some(_) => Err(DerError::UnalignedBitString),
            None => Err(DerError::Truncated),
        }
    }

    /// Skips one TLV of any tag.
    pub(crate) fn skip_any(&mut self) -> DerResult<()> {
        let _tag = self.peek_tag()?;
        self.position += 1;
        let len = self.read_length()?;
        self.take(len)?;
        Ok(())
    }
}

/// An RSA public key recovered from DER: `(modulus, exponent)`, big-endian.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RsaPublicComponents {
    /// The modulus `n`.
    pub(crate) modulus: Vec<u8>,
    /// The public exponent `e`.
    pub(crate) exponent: Vec<u8>,
}

/// An RSA private key recovered from DER.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RsaPrivateComponents {
    /// The modulus `n`.
    pub(crate) modulus: Vec<u8>,
    /// The public exponent `e`.
    pub(crate) public_exponent: Vec<u8>,
    /// The private exponent `d`.
    pub(crate) private_exponent: Vec<u8>,
}

/// An elliptic-curve public key recovered from DER.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EcPublicComponents {
    /// Which curve the `AlgorithmIdentifier` named.
    pub(crate) curve: CurveId,
    /// The SEC 1 encoded point, verbatim.
    pub(crate) point: Vec<u8>,
}

/// A DSA public key recovered from DER.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DsaPublicComponents {
    /// The prime modulus `p`.
    pub(crate) p: Vec<u8>,
    /// The subgroup order `q`.
    pub(crate) q: Vec<u8>,
    /// The generator `g`.
    pub(crate) g: Vec<u8>,
    /// The public value `y`.
    pub(crate) y: Vec<u8>,
}

/// An elliptic-curve private key recovered from DER.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EcPrivateComponents {
    /// The curve, which SEC 1 makes optional and PKCS#8 puts in the
    /// `AlgorithmIdentifier`.
    pub(crate) curve: Option<CurveId>,
    /// The private scalar, big-endian.
    pub(crate) private_key: Vec<u8>,
    /// The SEC 1 encoded public point, when the encoding carried one.
    pub(crate) public_key: Option<Vec<u8>>,
}

/// A DSA private key recovered from DER.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DsaPrivateComponents {
    /// The prime modulus `p`.
    pub(crate) p: Vec<u8>,
    /// The subgroup order `q`.
    pub(crate) q: Vec<u8>,
    /// The generator `g`.
    pub(crate) g: Vec<u8>,
    /// The private value `x`.
    pub(crate) x: Vec<u8>,
}

/// Whichever kind of public key a `SubjectPublicKeyInfo` turned out to hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PublicKeyComponents {
    /// An RSA key.
    Rsa(RsaPublicComponents),
    /// An elliptic-curve key on one of the two supported curves.
    Ec(EcPublicComponents),
    /// A DSA key, with its domain parameters.
    Dsa(DsaPublicComponents),
}

/// The key algorithm an `AlgorithmIdentifier` names, with the parameters that
/// belong to it.
enum KeyAlgorithm {
    /// `rsaEncryption`.
    Rsa,
    /// `id-ecPublicKey` with a named curve.
    Ec(CurveId),
    /// `id-dsa` with `Dss-Parms`.
    Dsa {
        /// The prime modulus `p`.
        p: Vec<u8>,
        /// The subgroup order `q`.
        q: Vec<u8>,
        /// The generator `g`.
        g: Vec<u8>,
    },
}

/// Parses a PKCS#1 `RSAPublicKey ::= SEQUENCE { modulus INTEGER, publicExponent INTEGER }`.
pub(crate) fn parse_rsa_public_key(input: &[u8]) -> DerResult<RsaPublicComponents> {
    let mut outer = Der::new(input);
    let mut sequence = outer.read_sequence()?;
    outer.finish()?;
    let modulus = sequence.read_unsigned_integer()?.to_vec();
    let exponent = sequence.read_unsigned_integer()?.to_vec();
    sequence.finish()?;
    Ok(RsaPublicComponents { modulus, exponent })
}

/// Parses a `SubjectPublicKeyInfo` of any of the three supported algorithms.
pub(crate) fn parse_public_key_info(input: &[u8]) -> DerResult<PublicKeyComponents> {
    let mut outer = Der::new(input);
    let mut spki = outer.read_sequence()?;
    outer.finish()?;
    let components = read_public_key_info(&mut spki)?;
    spki.finish()?;
    Ok(components)
}

/// Parses a `SubjectPublicKeyInfo` and returns the RSA key inside it.
pub(crate) fn parse_subject_public_key_info(input: &[u8]) -> DerResult<RsaPublicComponents> {
    match parse_public_key_info(input)? {
        PublicKeyComponents::Rsa(components) => Ok(components),
        _ => Err(DerError::NotRsa),
    }
}

/// Parses an X.509 `Certificate` and returns its `subjectPublicKey`.
///
/// Only the path down to `tbsCertificate.subjectPublicKeyInfo` is walked; the
/// signature and every extension are skipped. See the module documentation for
/// why that is the right amount of parsing here.
pub(crate) fn parse_certificate_key(input: &[u8]) -> DerResult<PublicKeyComponents> {
    let mut outer = Der::new(input);
    let mut certificate = outer.read_sequence()?;
    outer.finish()?;
    let tbs = certificate.read(tag::SEQUENCE)?;
    let mut tbs = Der::new(tbs);
    // version [0] EXPLICIT Version DEFAULT v1
    tbs.read_optional(tag::CONTEXT_0)?;
    tbs.read_unsigned_integer()?; // serialNumber
    tbs.skip_any()?; // signature (AlgorithmIdentifier)
    tbs.skip_any()?; // issuer
    tbs.skip_any()?; // validity
    tbs.skip_any()?; // subject
    let mut spki = tbs.read_sequence()?;
    let components = read_public_key_info(&mut spki)?;
    spki.finish()?;
    Ok(components)
}

/// Parses an X.509 `Certificate` and returns the RSA `subjectPublicKey`.
pub(crate) fn parse_certificate_public_key(input: &[u8]) -> DerResult<RsaPublicComponents> {
    match parse_certificate_key(input)? {
        PublicKeyComponents::Rsa(components) => Ok(components),
        _ => Err(DerError::NotRsa),
    }
}

/// Reads the body of a `SubjectPublicKeyInfo`: an `AlgorithmIdentifier` and the
/// `subjectPublicKey` BIT STRING, interpreted according to it.
fn read_public_key_info(spki: &mut Der<'_>) -> DerResult<PublicKeyComponents> {
    let algorithm = read_algorithm_identifier(spki)?;
    let key = spki.read_bit_string()?;
    match algorithm {
        KeyAlgorithm::Rsa => parse_rsa_public_key(key).map(PublicKeyComponents::Rsa),
        // RFC 5480 §2.2: the subjectPublicKey of an EC key *is* the SEC 1 point,
        // with no DER wrapper of its own.
        KeyAlgorithm::Ec(curve) => Ok(PublicKeyComponents::Ec(EcPublicComponents {
            curve,
            point: key.to_vec(),
        })),
        KeyAlgorithm::Dsa { p, q, g } => {
            let mut reader = Der::new(key);
            let y = reader.read_unsigned_integer()?.to_vec();
            reader.finish()?;
            Ok(PublicKeyComponents::Dsa(DsaPublicComponents { p, q, g, y }))
        }
    }
}

/// Reads an `AlgorithmIdentifier` and its parameters.
fn read_algorithm_identifier(reader: &mut Der<'_>) -> DerResult<KeyAlgorithm> {
    let mut algorithm = reader.read_sequence()?;
    let oid = algorithm.read(tag::OID)?;
    match oid {
        RSA_ENCRYPTION_OID => {
            // parameters: NULL for rsaEncryption, absent in some encoders.
            algorithm.read_optional(tag::NULL)?;
            Ok(KeyAlgorithm::Rsa)
        }
        EC_PUBLIC_KEY_OID => {
            let curve = algorithm.read(tag::OID)?;
            curve_from_oid(curve)
                .map(KeyAlgorithm::Ec)
                .ok_or(DerError::UnsupportedCurve)
        }
        DSA_OID => {
            // Dss-Parms ::= SEQUENCE { p INTEGER, q INTEGER, g INTEGER }
            let mut parameters = algorithm.read_sequence()?;
            let p = parameters.read_unsigned_integer()?.to_vec();
            let q = parameters.read_unsigned_integer()?.to_vec();
            let g = parameters.read_unsigned_integer()?.to_vec();
            parameters.finish()?;
            Ok(KeyAlgorithm::Dsa { p, q, g })
        }
        _ => Err(DerError::UnknownKeyAlgorithm),
    }
}

/// How many times a PKCS#8 `PrivateKeyInfo` may wrap another key structure
/// before [`parse_rsa_private_key`] gives up.
///
/// A real PKCS#8 key wraps a PKCS#1 `RSAPrivateKey` in an `OCTET STRING`
/// exactly once, so this ceiling is already generous; it exists to keep the
/// cost of a hostile blob linear in its size rather than letting it grow
/// without limit. See [`parse_rsa_private_key`] for why this is a loop rather
/// than a recursion depth to check.
pub(crate) const MAX_PRIVATE_KEY_NESTING: usize = 256;

/// Parses a PKCS#1 `RSAPrivateKey`, or a PKCS#8 `PrivateKeyInfo` wrapping one.
///
/// This is written as a loop, not as self-recursion: unwrapping one PKCS#8
/// layer used to end in a call back into this same function, one stack frame
/// per layer, with no limit on how many layers a crafted blob could nest. A
/// stack overflow aborts the process outright — it cannot be caught, unlike an
/// ordinary panic — so [`RsaPrivateKey::from_der`](crate::RsaPrivateKey::from_der)
/// being a public entry point made every layer an attacker chose to add a free
/// stack frame in a signing service. The loop below tracks nesting as an
/// ordinary counter instead, bounded by [`MAX_PRIVATE_KEY_NESTING`], so the
/// depth is a matter of iteration count rather than of the stack.
pub(crate) fn parse_rsa_private_key(input: &[u8]) -> DerResult<RsaPrivateComponents> {
    let mut current = input;
    for _ in 0..MAX_PRIVATE_KEY_NESTING {
        let mut outer = Der::new(current);
        let mut sequence = outer.read_sequence()?;
        outer.finish()?;
        let version = sequence.read_unsigned_integer()?;
        // PKCS#8 has an AlgorithmIdentifier where PKCS#1 has the modulus.
        if sequence.peek_tag() != Ok(tag::SEQUENCE) {
            let modulus = sequence.read_unsigned_integer()?.to_vec();
            let public_exponent = sequence.read_unsigned_integer()?.to_vec();
            let private_exponent = sequence.read_unsigned_integer()?.to_vec();
            // primes, exponents and the coefficient follow; the plain (non-CRT)
            // exponentiation this crate performs does not need them.
            return Ok(RsaPrivateComponents {
                modulus,
                public_exponent,
                private_exponent,
            });
        }
        if version != [0] {
            return Err(DerError::BadInteger);
        }
        let mut algorithm = sequence.read_sequence()?;
        expect_rsa_oid(&mut algorithm)?;
        current = sequence.read(tag::OCTET_STRING)?;
    }
    Err(DerError::TooDeeplyNested)
}

/// Parses a SEC 1 `ECPrivateKey`, or a PKCS#8 `PrivateKeyInfo` wrapping one.
///
/// ```text
/// ECPrivateKey ::= SEQUENCE {
///   version        INTEGER { ecPrivkeyVer1(1) },
///   privateKey     OCTET STRING,
///   parameters [0] ECParameters OPTIONAL,
///   publicKey  [1] BIT STRING OPTIONAL }
/// ```
///
/// The curve may therefore come from either layer; PKCS#8's
/// `AlgorithmIdentifier` wins when both are present, and the two must agree.
/// The nesting is bounded exactly as [`parse_rsa_private_key`]'s is, and for the
/// same reason.
pub(crate) fn parse_ec_private_key(input: &[u8]) -> DerResult<EcPrivateComponents> {
    let mut current = input;
    let mut curve = None;
    for _ in 0..MAX_PRIVATE_KEY_NESTING {
        let mut outer = Der::new(current);
        let mut sequence = outer.read_sequence()?;
        outer.finish()?;
        let version = sequence.read_unsigned_integer()?;
        // PKCS#8 has an AlgorithmIdentifier where SEC 1 has the private scalar.
        if sequence.peek_tag() == Ok(tag::SEQUENCE) {
            if version != [0] {
                return Err(DerError::BadInteger);
            }
            match read_algorithm_identifier(&mut sequence)? {
                KeyAlgorithm::Ec(named) => curve = Some(named),
                _ => return Err(DerError::WrongKeyAlgorithm),
            }
            current = sequence.read(tag::OCTET_STRING)?;
            continue;
        }
        if version != [1] {
            return Err(DerError::BadInteger);
        }
        let private_key = sequence.read(tag::OCTET_STRING)?.to_vec();
        if let Some(parameters) = sequence.read_optional(tag::CONTEXT_0)? {
            let mut parameters = Der::new(parameters);
            let named =
                curve_from_oid(parameters.read(tag::OID)?).ok_or(DerError::UnsupportedCurve)?;
            parameters.finish()?;
            if curve.is_some_and(|outer| outer != named) {
                return Err(DerError::WrongKeyAlgorithm);
            }
            curve = Some(named);
        }
        let public_key = match sequence.read_optional(tag::CONTEXT_1)? {
            Some(wrapped) => {
                let mut wrapped = Der::new(wrapped);
                let point = wrapped.read_bit_string()?.to_vec();
                wrapped.finish()?;
                Some(point)
            }
            None => None,
        };
        return Ok(EcPrivateComponents {
            curve,
            private_key,
            public_key,
        });
    }
    Err(DerError::TooDeeplyNested)
}

/// Parses a PKCS#8 `PrivateKeyInfo` holding a DSA key, whose `privateKey` octet
/// string is the bare INTEGER `x` and whose parameters carry `p`, `q` and `g`.
pub(crate) fn parse_dsa_private_key(input: &[u8]) -> DerResult<DsaPrivateComponents> {
    let mut current = input;
    for _ in 0..MAX_PRIVATE_KEY_NESTING {
        let mut outer = Der::new(current);
        let mut sequence = outer.read_sequence()?;
        outer.finish()?;
        let version = sequence.read_unsigned_integer()?;
        if version != [0] {
            return Err(DerError::BadInteger);
        }
        let KeyAlgorithm::Dsa { p, q, g } = read_algorithm_identifier(&mut sequence)? else {
            return Err(DerError::WrongKeyAlgorithm);
        };
        let wrapped = sequence.read(tag::OCTET_STRING)?;
        // A nested PrivateKeyInfo is not a shape any encoder produces, but the
        // loop keeps a crafted one from costing a stack frame per layer.
        if Der::new(wrapped).peek_tag() == Ok(tag::SEQUENCE) {
            current = wrapped;
            continue;
        }
        let mut reader = Der::new(wrapped);
        let x = reader.read_unsigned_integer()?.to_vec();
        reader.finish()?;
        return Ok(DsaPrivateComponents { p, q, g, x });
    }
    Err(DerError::TooDeeplyNested)
}

fn expect_rsa_oid(algorithm: &mut Der<'_>) -> DerResult<()> {
    let oid = algorithm.read(tag::OID)?;
    if oid != RSA_ENCRYPTION_OID {
        return Err(DerError::NotRsa);
    }
    // parameters: NULL for rsaEncryption, absent in some encoders.
    algorithm.read_optional(tag::NULL)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkey;

    #[test]
    fn spki_and_bare_public_key_agree() -> Result<(), DerError> {
        let from_spki = parse_subject_public_key_info(&testkey::bytes(testkey::SPKI_BASE64))?;
        let bare = parse_rsa_public_key(&testkey::bytes(testkey::RSA_PUBLIC_KEY_BASE64))?;
        assert_eq!(from_spki, bare);
        assert_eq!(from_spki.modulus, testkey::bytes(testkey::MODULUS_BASE64));
        assert_eq!(from_spki.exponent, [0x01, 0x00, 0x01]);
        Ok(())
    }

    #[test]
    fn a_certificate_yields_the_same_key() -> Result<(), DerError> {
        let from_certificate =
            parse_certificate_public_key(&testkey::bytes(testkey::CERTIFICATE_BASE64))?;
        assert_eq!(
            from_certificate.modulus,
            testkey::bytes(testkey::MODULUS_BASE64)
        );
        assert_eq!(from_certificate.exponent, [0x01, 0x00, 0x01]);
        Ok(())
    }

    #[test]
    fn pkcs1_and_pkcs8_private_keys_agree() -> Result<(), DerError> {
        let pkcs1 = parse_rsa_private_key(&testkey::bytes(testkey::RSA_PRIVATE_KEY_BASE64))?;
        let pkcs8 = parse_rsa_private_key(&testkey::bytes(testkey::PKCS8_PRIVATE_KEY_BASE64))?;
        assert_eq!(pkcs1, pkcs8);
        assert_eq!(
            pkcs1.private_exponent,
            testkey::bytes(testkey::PRIVATE_EXPONENT_BASE64)
        );
        Ok(())
    }

    /// Encodes a DER length in the shortest form, long or short.
    fn der_length(len: usize) -> Vec<u8> {
        if len < 0x80 {
            return vec![len as u8];
        }
        let bytes = len.to_be_bytes();
        // `len >= 0x80` here, so at least one byte is always non-zero; the
        // fallback only matters for the `len == 0` case this branch never
        // reaches, and picks the single least-significant byte for it.
        let first_nonzero = bytes
            .iter()
            .position(|&byte| byte != 0)
            .unwrap_or(bytes.len() - 1);
        let significant = &bytes[first_nonzero..];
        let mut out = vec![0x80 | u8::try_from(significant.len()).unwrap_or(8)];
        out.extend_from_slice(significant);
        out
    }

    /// Builds one DER TLV.
    fn der_tlv(tag: u8, contents: &[u8]) -> Vec<u8> {
        let mut out = vec![tag];
        out.extend(der_length(contents.len()));
        out.extend_from_slice(contents);
        out
    }

    /// Wraps `inner` in one more layer of the PKCS#8-wraps-PKCS#1 shape:
    /// `SEQUENCE { INTEGER 0, SEQUENCE { OID rsaEncryption, NULL }, OCTET STRING { inner } }`.
    fn wrap_pkcs8_layer(inner: &[u8]) -> Vec<u8> {
        let version = der_tlv(tag::INTEGER, &[0x00]);
        let mut algorithm_contents = der_tlv(tag::OID, RSA_ENCRYPTION_OID);
        algorithm_contents.extend(der_tlv(tag::NULL, &[]));
        let algorithm = der_tlv(tag::SEQUENCE, &algorithm_contents);
        let octet_string = der_tlv(tag::OCTET_STRING, inner);

        let mut contents = version;
        contents.extend(algorithm);
        contents.extend(octet_string);
        der_tlv(tag::SEQUENCE, &contents)
    }

    /// Nests `innermost` inside `layers` PKCS#8 wrappers.
    fn nested_pkcs8(layers: usize, innermost: &[u8]) -> Vec<u8> {
        let mut current = innermost.to_vec();
        for _ in 0..layers {
            current = wrap_pkcs8_layer(&current);
        }
        current
    }

    /// `RsaPrivateKey::from_der` — and this parser underneath it — used to
    /// unwrap a PKCS#8 layer by calling itself, one stack frame per layer, with
    /// no limit. A blob that nests the wrapper hundreds of times must now come
    /// back as a bounded `DerError` instead of recursing; see
    /// [`MAX_PRIVATE_KEY_NESTING`].
    #[test]
    fn nesting_far_past_the_limit_is_rejected_instead_of_recursing() {
        for layers in [MAX_PRIVATE_KEY_NESTING + 1, 500, 2_000] {
            let blob = nested_pkcs8(layers, &[]);
            assert_eq!(
                parse_rsa_private_key(&blob),
                Err(DerError::TooDeeplyNested),
                "{layers} layers should be rejected"
            );
        }
    }

    #[test]
    fn nesting_just_inside_the_limit_still_parses() -> Result<(), DerError> {
        let innermost = testkey::bytes(testkey::RSA_PRIVATE_KEY_BASE64);
        let expected = parse_rsa_private_key(&innermost)?;
        let layers = MAX_PRIVATE_KEY_NESTING - 8;
        let blob = nested_pkcs8(layers, &innermost);
        assert_eq!(parse_rsa_private_key(&blob)?, expected);
        Ok(())
    }

    #[test]
    fn a_non_minimal_length_is_rejected() {
        // 0x81 0x01 encodes the length 1 in the long form, which DER forbids.
        let input = [tag::SEQUENCE, 0x81, 0x01, 0x05];
        assert_eq!(
            Der::new(&input).read_sequence().err(),
            Some(DerError::BadLength)
        );
    }

    #[test]
    fn a_non_minimal_integer_is_rejected() {
        let input = [tag::INTEGER, 0x02, 0x00, 0x01];
        assert_eq!(
            Der::new(&input).read_unsigned_integer(),
            Err(DerError::BadInteger)
        );
    }

    #[test]
    fn a_negative_integer_is_rejected() {
        let input = [tag::INTEGER, 0x01, 0xff];
        assert_eq!(
            Der::new(&input).read_unsigned_integer(),
            Err(DerError::BadInteger)
        );
    }

    #[test]
    fn a_truncated_value_is_rejected() {
        let input = [tag::SEQUENCE, 0x05, 0x00];
        assert_eq!(
            Der::new(&input).read_sequence().err(),
            Some(DerError::Truncated)
        );
    }

    /// A P-256 `SubjectPublicKeyInfo`, the payload of the
    /// `<dsig11:DEREncodedKeyValue>` in the W3C XML Signature 1.1 interoperability
    /// document `signature-enveloping-derencoded-ec.xml`.
    const EC_P256_SPKI_BASE64: &str = "MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEn/Jpc2WrgVE5vIkIGFvmMDPwZXOKcrdsEYuNIN+NsnA1/J22COeVLgSwObFJGFbIlaroYirLnC+dqIBErTi4Hg==";

    /// A DER INTEGER holding an unsigned magnitude, with the sign octet the
    /// encoding needs.
    fn der_integer(magnitude: &[u8]) -> Vec<u8> {
        let mut contents = Vec::with_capacity(magnitude.len() + 1);
        if magnitude.first().is_some_and(|byte| byte & 0x80 != 0) {
            contents.push(0x00);
        }
        contents.extend_from_slice(magnitude);
        der_tlv(tag::INTEGER, &contents)
    }

    /// Assembles a DSA `SubjectPublicKeyInfo` from its components.
    fn dsa_spki(p: &[u8], q: &[u8], g: &[u8], y: &[u8]) -> Vec<u8> {
        let mut parameters = der_integer(p);
        parameters.extend(der_integer(q));
        parameters.extend(der_integer(g));
        let mut algorithm = der_tlv(tag::OID, DSA_OID);
        algorithm.extend(der_tlv(tag::SEQUENCE, &parameters));

        let mut key = vec![0x00];
        key.extend(der_integer(y));

        let mut contents = der_tlv(tag::SEQUENCE, &algorithm);
        contents.extend(der_tlv(tag::BIT_STRING, &key));
        der_tlv(tag::SEQUENCE, &contents)
    }

    #[test]
    fn an_unknown_algorithm_is_rejected() {
        // 1.2.840.113549.1.1.10 (id-RSASSA-PSS) is not a key algorithm.
        let algorithm = der_tlv(
            tag::OID,
            &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0a],
        );
        let mut contents = der_tlv(tag::SEQUENCE, &algorithm);
        contents.extend(der_tlv(tag::BIT_STRING, &[0x00]));
        let spki = der_tlv(tag::SEQUENCE, &contents);
        assert_eq!(
            parse_public_key_info(&spki),
            Err(DerError::UnknownKeyAlgorithm)
        );
        assert_eq!(
            parse_subject_public_key_info(&spki),
            Err(DerError::UnknownKeyAlgorithm)
        );
    }

    #[test]
    fn an_elliptic_curve_key_is_not_read_as_rsa() {
        let spki = testkey::bytes(EC_P256_SPKI_BASE64);
        assert_eq!(parse_subject_public_key_info(&spki), Err(DerError::NotRsa));
    }

    #[test]
    fn an_elliptic_curve_subject_public_key_info_parses() -> Result<(), DerError> {
        let spki = testkey::bytes(EC_P256_SPKI_BASE64);
        let PublicKeyComponents::Ec(components) = parse_public_key_info(&spki)? else {
            panic!("a P-256 SubjectPublicKeyInfo holds an elliptic-curve key");
        };
        assert_eq!(components.curve, CurveId::P256);
        // An uncompressed P-256 point: the 0x04 tag and two 32-octet coordinates.
        assert_eq!(components.point.len(), 65);
        assert_eq!(components.point.first().copied(), Some(0x04));
        Ok(())
    }

    #[test]
    fn an_unsupported_curve_is_reported_as_such() {
        // secp521r1 (1.3.132.0.35), which this crate does not implement.
        let mut algorithm = der_tlv(tag::OID, EC_PUBLIC_KEY_OID);
        algorithm.extend(der_tlv(tag::OID, &[0x2b, 0x81, 0x04, 0x00, 0x23]));
        let mut contents = der_tlv(tag::SEQUENCE, &algorithm);
        contents.extend(der_tlv(tag::BIT_STRING, &[0x00, 0x04]));
        let spki = der_tlv(tag::SEQUENCE, &contents);
        assert_eq!(
            parse_public_key_info(&spki),
            Err(DerError::UnsupportedCurve)
        );
    }

    #[test]
    fn a_dsa_subject_public_key_info_parses() -> Result<(), DerError> {
        let spki = dsa_spki(&[0x8d, 0xf2], &[0xc7, 0x73], &[0x62, 0x6d], &[0x19, 0x13]);
        let PublicKeyComponents::Dsa(components) = parse_public_key_info(&spki)? else {
            panic!("an id-dsa SubjectPublicKeyInfo holds a DSA key");
        };
        assert_eq!(components.p, [0x8d, 0xf2]);
        assert_eq!(components.q, [0xc7, 0x73]);
        assert_eq!(components.g, [0x62, 0x6d]);
        assert_eq!(components.y, [0x19, 0x13]);
        assert_eq!(parse_subject_public_key_info(&spki), Err(DerError::NotRsa));
        Ok(())
    }

    /// A SEC 1 `ECPrivateKey` carries the curve in an optional `[0]` field and
    /// the public point in an optional `[1]` field; a PKCS#8 wrapper moves the
    /// curve into the `AlgorithmIdentifier`. Both must reach the same scalar.
    #[test]
    fn sec1_and_pkcs8_elliptic_curve_private_keys_agree() -> Result<(), DerError> {
        let scalar = [0x11_u8; 32];
        let point = {
            let mut point = vec![0x04_u8];
            point.extend_from_slice(&[0x22; 64]);
            point
        };

        let mut sec1_contents = der_tlv(tag::INTEGER, &[0x01]);
        sec1_contents.extend(der_tlv(tag::OCTET_STRING, &scalar));
        sec1_contents.extend(der_tlv(tag::CONTEXT_0, &der_tlv(tag::OID, P256_OID)));
        let mut bit_string = vec![0x00];
        bit_string.extend_from_slice(&point);
        sec1_contents.extend(der_tlv(
            tag::CONTEXT_1,
            &der_tlv(tag::BIT_STRING, &bit_string),
        ));
        let sec1 = der_tlv(tag::SEQUENCE, &sec1_contents);

        let parsed = parse_ec_private_key(&sec1)?;
        assert_eq!(parsed.curve, Some(CurveId::P256));
        assert_eq!(parsed.private_key, scalar);
        assert_eq!(parsed.public_key, Some(point.clone()));

        let mut algorithm = der_tlv(tag::OID, EC_PUBLIC_KEY_OID);
        algorithm.extend(der_tlv(tag::OID, P256_OID));
        let mut pkcs8_contents = der_tlv(tag::INTEGER, &[0x00]);
        pkcs8_contents.extend(der_tlv(tag::SEQUENCE, &algorithm));
        pkcs8_contents.extend(der_tlv(tag::OCTET_STRING, &sec1));
        let pkcs8 = der_tlv(tag::SEQUENCE, &pkcs8_contents);
        assert_eq!(parse_ec_private_key(&pkcs8)?, parsed);
        Ok(())
    }

    #[test]
    fn a_dsa_private_key_parses() -> Result<(), DerError> {
        let mut parameters = der_integer(&[0x8d, 0xf2]);
        parameters.extend(der_integer(&[0xc7, 0x73]));
        parameters.extend(der_integer(&[0x62, 0x6d]));
        let mut algorithm = der_tlv(tag::OID, DSA_OID);
        algorithm.extend(der_tlv(tag::SEQUENCE, &parameters));
        let mut contents = der_tlv(tag::INTEGER, &[0x00]);
        contents.extend(der_tlv(tag::SEQUENCE, &algorithm));
        contents.extend(der_tlv(tag::OCTET_STRING, &der_integer(&[0x20, 0x70])));
        let pkcs8 = der_tlv(tag::SEQUENCE, &contents);

        let parsed = parse_dsa_private_key(&pkcs8)?;
        assert_eq!(parsed.p, [0x8d, 0xf2]);
        assert_eq!(parsed.x, [0x20, 0x70]);
        Ok(())
    }

    /// The same unbounded-recursion guard the RSA path has, on the two new ones.
    #[test]
    fn nested_private_keys_are_bounded() {
        let mut algorithm = der_tlv(tag::OID, EC_PUBLIC_KEY_OID);
        algorithm.extend(der_tlv(tag::OID, P256_OID));
        let mut blob = Vec::new();
        for _ in 0..MAX_PRIVATE_KEY_NESTING + 1 {
            let mut contents = der_tlv(tag::INTEGER, &[0x00]);
            contents.extend(der_tlv(tag::SEQUENCE, &algorithm));
            contents.extend(der_tlv(tag::OCTET_STRING, &blob));
            blob = der_tlv(tag::SEQUENCE, &contents);
        }
        assert_eq!(parse_ec_private_key(&blob), Err(DerError::TooDeeplyNested));
    }

    #[test]
    fn trailing_data_after_the_structure_is_rejected() {
        let mut input = testkey::bytes(testkey::SPKI_BASE64);
        input.push(0x00);
        assert_eq!(
            parse_subject_public_key_info(&input),
            Err(DerError::TrailingData)
        );
    }
}

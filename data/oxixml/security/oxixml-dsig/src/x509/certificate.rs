//! The X.509 certificate, read far enough to validate a path through it.
//!
//! [`crate::crypto::der`] already walks a certificate down to its
//! `subjectPublicKeyInfo`, which is all core validation of an XML signature
//! needs. Path validation needs the rest: who issued it, who it is for, when it
//! is valid, whether it may issue certificates of its own, and the signature
//! that ties it to its issuer.
//!
//! ```text
//! Certificate ::= SEQUENCE {
//!   tbsCertificate       TBSCertificate,
//!   signatureAlgorithm   AlgorithmIdentifier,
//!   signatureValue       BIT STRING }
//!
//! TBSCertificate ::= SEQUENCE {
//!   version         [0] EXPLICIT Version DEFAULT v1,
//!   serialNumber        CertificateSerialNumber,
//!   signature           AlgorithmIdentifier,
//!   issuer              Name,
//!   validity            Validity,
//!   subject             Name,
//!   subjectPublicKeyInfo SubjectPublicKeyInfo,
//!   issuerUniqueID  [1] IMPLICIT UniqueIdentifier OPTIONAL,
//!   subjectUniqueID [2] IMPLICIT UniqueIdentifier OPTIONAL,
//!   extensions      [3] EXPLICIT Extensions OPTIONAL }
//! ```
//!
//! Four extensions are processed — `basicConstraints`, `keyUsage`,
//! `subjectKeyIdentifier` and `authorityKeyIdentifier`. Any *other* extension
//! marked critical is recorded and makes the certificate unusable, because that
//! is what marking an extension critical means.

use std::fmt;

use crate::crypto::der::{self, Der, DerError, tag};
use crate::key::{DsaPublicKey, EcPublicKey, RsaPublicKey, VerificationKey};
use crate::x509::error::CertificateError;
use crate::x509::name::DistinguishedName;
use crate::x509::oid;
use crate::x509::time::{Timestamp, parse_generalized_time, parse_utc_time};

/// A parsed X.509 certificate.
///
/// Parsing succeeds for any structurally valid certificate, including one whose
/// key or signature algorithm this crate cannot use: those show up as
/// [`public_key`](Certificate::public_key) returning [`None`] and as a failure
/// at the point the certificate is actually relied on, which keeps one
/// unusable certificate in a `<ds:X509Data>` from making the whole chain
/// unreadable.
///
/// # Example
///
/// ```
/// use oxixml_dsig::Certificate;
///
/// # const DER_BASE64: &str = "MIIBpDCCAQ2gAwIBAgIFASNFZ4kwDQYJKoZIhvcNAQELBQAwFjEUMBIGA1UEAwwLT3hpWE1MIFRlc3QwHhcNMjYwMTAxMDAwMDAwWhcNMzYwMTAxMDAwMDAwWjAWMRQwEgYDVQQDDAtPeGlYTUwgVGVzdDCBnzANBgkqhkiG9w0BAQEFAAOBjQAwgYkCgYEAjkcoCslxEus5u8QIPMAYjDIpNgX12fbWalSPmhbKjlNeR5ukRjB0xBJqYvncXrPl7lHzDemgFsdEbJCvTgF1imtaMkc6yfaHkJ4+RgzPR2SBU4KEaVq69fNR5Dco0vl/VUZD54drPqBBvXhiqwpAK3ZLHcPSLxzRQi1a20l+tlUCAwEAATANBgkqhkiG9w0BAQsFAAOBgQAbgJ1S/Y0svcUZpowjNkA5BxrXfJNopeDC6Rxz77AHzTfqnwkVWkjAYxY8mfbeCQkEucKK2FxB8bv1WzDE/rlW5h8jtyxmvYA3d/imKCjC7SPcibgHh55nIkmwA9oXpLL5C4rM0fcApbudoveWWwRHWAlxDiusBrr8vbzaDOByHA==";
/// let der = oxixml_text::base64::base64_decode(DER_BASE64)?;
/// let certificate = Certificate::parse(&der)?;
/// assert_eq!(certificate.subject().to_string(), "CN=OxiXML Test");
/// assert_eq!(certificate.not_before().to_string(), "2026-01-01T00:00:00Z");
/// assert!(certificate.is_self_issued());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone)]
pub struct Certificate {
    encoding: Vec<u8>,
    tbs_encoding: Vec<u8>,
    version: u8,
    serial_number: Vec<u8>,
    signature_algorithm: Vec<u8>,
    signature: Vec<u8>,
    issuer: DistinguishedName,
    subject: DistinguishedName,
    not_before: Timestamp,
    not_after: Timestamp,
    public_key_info: Vec<u8>,
    public_key: Option<VerificationKey>,
    certificate_authority: bool,
    path_length_constraint: Option<usize>,
    key_usage: Option<KeyUsage>,
    subject_key_identifier: Option<Vec<u8>>,
    authority_key_identifier: Option<Vec<u8>>,
    authority_certificate_serial: Option<Vec<u8>>,
    unhandled_critical_extensions: Vec<String>,
}

impl Certificate {
    /// Parses a DER `Certificate` — the payload of a `<ds:X509Certificate>`,
    /// base64 decoded.
    ///
    /// # Errors
    ///
    /// [`CertificateError::Malformed`] if the encoding is not an X.509
    /// certificate, if it uses a field this crate refuses (a negative serial
    /// number, an extension repeated twice, extensions on a certificate that
    /// does not declare itself v3), or if a validity date is not the
    /// `YYMMDDHHMMSSZ`/`YYYYMMDDHHMMSSZ` form
    /// [RFC 5280 §4.1.2.5](https://www.rfc-editor.org/rfc/rfc5280#section-4.1.2.5)
    /// requires.
    pub fn parse(encoding: &[u8]) -> Result<Self, CertificateError> {
        let mut outer = Der::new(encoding);
        let mut certificate = outer.read_sequence().map_err(malformed)?;
        outer.finish().map_err(malformed)?;

        let (tbs_encoding, tbs) = certificate.read_tagged(tag::SEQUENCE).map_err(malformed)?;
        let (signature_algorithm, _) = certificate.read_tagged(tag::SEQUENCE).map_err(malformed)?;
        let signature = certificate.read_bit_string().map_err(malformed)?.to_vec();
        certificate.finish().map_err(malformed)?;

        let mut parsed = Self::read_tbs(&mut Der::new(tbs))?;
        // RFC 5280 §4.1.1.2: the algorithm identifier outside the signed part
        // must repeat the one inside it. They are compared as encoded, so a
        // certificate cannot be re-labelled with a weaker algorithm on the way
        // past a verifier that only reads one of the two.
        if parsed.signature_algorithm != signature_algorithm {
            return Err(CertificateError::Malformed(
                "the signatureAlgorithm outside the tbsCertificate differs from the one inside it"
                    .to_owned(),
            ));
        }
        parsed.encoding = encoding.to_vec();
        parsed.tbs_encoding = tbs_encoding.to_vec();
        parsed.signature = signature;
        Ok(parsed)
    }

    /// Reads a `TBSCertificate`, leaving the fields that live outside it empty
    /// for [`parse`](Certificate::parse) to fill in.
    fn read_tbs(tbs: &mut Der<'_>) -> Result<Self, CertificateError> {
        let version = match tbs.read_optional(tag::CONTEXT_0).map_err(malformed)? {
            Some(wrapper) => {
                let mut wrapper = Der::new(wrapper);
                let value = wrapper.read_unsigned_integer().map_err(malformed)?;
                wrapper.finish().map_err(malformed)?;
                match value {
                    [0] => 1,
                    [1] => 2,
                    [2] => 3,
                    _ => {
                        return Err(CertificateError::Malformed(
                            "the certificate declares a version other than v1, v2 or v3".to_owned(),
                        ));
                    }
                }
            }
            None => 1,
        };
        // A serial number is an INTEGER that RFC 5280 §4.1.2.2 requires to be
        // positive. `read_unsigned_integer` therefore refuses the negative ones
        // some older authorities emitted; such a certificate is not usable here.
        let serial_number = tbs.read_unsigned_integer().map_err(malformed)?.to_vec();
        let (signature_algorithm, _) = tbs.read_tagged(tag::SEQUENCE).map_err(malformed)?;
        let (issuer_encoding, _) = tbs.read_tagged(tag::SEQUENCE).map_err(malformed)?;
        let issuer = DistinguishedName::parse(issuer_encoding)?;
        let (not_before, not_after) = read_validity(tbs)?;
        let (subject_encoding, _) = tbs.read_tagged(tag::SEQUENCE).map_err(malformed)?;
        let subject = DistinguishedName::parse(subject_encoding)?;
        let (public_key_info, _) = tbs.read_tagged(tag::SEQUENCE).map_err(malformed)?;
        let public_key = read_public_key(public_key_info);

        // The two unique identifiers of a v2 certificate sit between the public
        // key and the extensions. They are IMPLICIT, so they keep the primitive
        // tag of the BIT STRING they replace rather than becoming constructed
        // context tags — and skipping them is what keeps the extensions of such
        // a certificate from being read as absent.
        tbs.read_optional(tag::IMPLICIT_1).map_err(malformed)?;
        tbs.read_optional(tag::IMPLICIT_2).map_err(malformed)?;

        let mut certificate = Self {
            encoding: Vec::new(),
            tbs_encoding: Vec::new(),
            version,
            serial_number,
            signature_algorithm: signature_algorithm.to_vec(),
            signature: Vec::new(),
            issuer,
            subject,
            not_before,
            not_after,
            public_key_info: public_key_info.to_vec(),
            public_key,
            certificate_authority: false,
            path_length_constraint: None,
            key_usage: None,
            subject_key_identifier: None,
            authority_key_identifier: None,
            authority_certificate_serial: None,
            unhandled_critical_extensions: Vec::new(),
        };
        if let Some(wrapper) = tbs.read_optional(tag::CONTEXT_3).map_err(malformed)? {
            if version != 3 {
                return Err(CertificateError::Malformed(
                    "only a v3 certificate may carry extensions".to_owned(),
                ));
            }
            certificate.read_extensions(wrapper)?;
        }
        tbs.finish().map_err(malformed)?;
        Ok(certificate)
    }

    /// Reads the `[3] EXPLICIT Extensions` field.
    fn read_extensions(&mut self, wrapper: &[u8]) -> Result<(), CertificateError> {
        let mut wrapper = Der::new(wrapper);
        let mut extensions = wrapper.read_sequence().map_err(malformed)?;
        wrapper.finish().map_err(malformed)?;
        let mut seen: Vec<Vec<u8>> = Vec::new();
        while !extensions.is_empty() {
            let mut extension = extensions.read_sequence().map_err(malformed)?;
            let identifier = extension.read(tag::OID).map_err(malformed)?;
            let critical = if extension.peek_tag() == Ok(tag::BOOLEAN) {
                extension.read_boolean().map_err(malformed)?
            } else {
                false
            };
            let value = extension.read(tag::OCTET_STRING).map_err(malformed)?;
            extension.finish().map_err(malformed)?;

            // RFC 5280 §4.2: "A certificate MUST NOT include more than one
            // instance of a particular extension." Two instances of the same
            // extension disagreeing is a parser-differential waiting to happen.
            if seen.iter().any(|previous| previous == identifier) {
                return Err(CertificateError::Malformed(format!(
                    "the extension `{}` appears more than once",
                    oid::dotted(identifier)
                )));
            }
            seen.push(identifier.to_vec());

            match identifier {
                oid::BASIC_CONSTRAINTS => self.read_basic_constraints(value)?,
                oid::KEY_USAGE => self.key_usage = Some(read_key_usage(value)?),
                oid::SUBJECT_KEY_IDENTIFIER => {
                    self.subject_key_identifier = Some(read_key_identifier(value)?);
                }
                oid::AUTHORITY_KEY_IDENTIFIER => self.read_authority_key_identifier(value)?,
                _ if critical => self
                    .unhandled_critical_extensions
                    .push(oid::dotted(identifier)),
                _ => {}
            }
        }
        Ok(())
    }

    /// Reads `BasicConstraints ::= SEQUENCE { cA BOOLEAN DEFAULT FALSE,
    /// pathLenConstraint INTEGER (0..MAX) OPTIONAL }`.
    fn read_basic_constraints(&mut self, value: &[u8]) -> Result<(), CertificateError> {
        let mut wrapper = Der::new(value);
        let mut constraints = wrapper.read_sequence().map_err(malformed)?;
        wrapper.finish().map_err(malformed)?;
        if constraints.peek_tag() == Ok(tag::BOOLEAN) {
            self.certificate_authority = constraints.read_boolean().map_err(malformed)?;
        }
        if constraints.peek_tag() == Ok(tag::INTEGER) {
            let magnitude = constraints.read_unsigned_integer().map_err(malformed)?;
            // A constraint wider than a `usize` cannot be reached by any path
            // this crate will build, so saturating at the maximum is exact.
            let mut limit = 0_usize;
            for byte in magnitude {
                limit = limit.saturating_mul(256).saturating_add(usize::from(*byte));
            }
            self.path_length_constraint = Some(limit);
        }
        constraints.finish().map_err(malformed)?;
        Ok(())
    }

    /// Reads `AuthorityKeyIdentifier ::= SEQUENCE { keyIdentifier [0] IMPLICIT
    /// OCTET STRING OPTIONAL, authorityCertIssuer [1] IMPLICIT GeneralNames
    /// OPTIONAL, authorityCertSerialNumber [2] IMPLICIT INTEGER OPTIONAL }`.
    fn read_authority_key_identifier(&mut self, value: &[u8]) -> Result<(), CertificateError> {
        let mut wrapper = Der::new(value);
        let mut identifier = wrapper.read_sequence().map_err(malformed)?;
        wrapper.finish().map_err(malformed)?;
        self.authority_key_identifier = identifier
            .read_optional(tag::IMPLICIT_0)
            .map_err(malformed)?
            .map(<[u8]>::to_vec);
        // authorityCertIssuer, which is a name form this crate does not use.
        identifier
            .read_optional(tag::CONTEXT_1)
            .map_err(malformed)?;
        // The serial number is an IMPLICIT INTEGER, so what is here is the
        // INTEGER's contents — sign octet and all. It is compared against a
        // serial read with `read_unsigned_integer`, which strips that octet, so
        // it is stripped here too; otherwise every issuer whose serial happens
        // to have its top bit set would be ruled out.
        self.authority_certificate_serial = identifier
            .read_optional(tag::IMPLICIT_2)
            .map_err(malformed)?
            .map(unsigned_magnitude);
        Ok(())
    }

    /// The DER encoding the certificate was parsed from.
    #[must_use]
    pub fn encoding(&self) -> &[u8] {
        &self.encoding
    }

    /// The `tbsCertificate`, exactly as encoded — the octets the signature
    /// covers.
    #[must_use]
    pub fn signed_octets(&self) -> &[u8] {
        &self.tbs_encoding
    }

    /// The certificate version: 1, 2 or 3.
    #[must_use]
    pub const fn version(&self) -> u8 {
        self.version
    }

    /// The serial number, big-endian and without the DER sign octet.
    #[must_use]
    pub fn serial_number(&self) -> &[u8] {
        &self.serial_number
    }

    /// Who issued the certificate.
    #[must_use]
    pub const fn issuer(&self) -> &DistinguishedName {
        &self.issuer
    }

    /// Who the certificate is for.
    #[must_use]
    pub const fn subject(&self) -> &DistinguishedName {
        &self.subject
    }

    /// The start of the validity period, inclusive.
    #[must_use]
    pub const fn not_before(&self) -> Timestamp {
        self.not_before
    }

    /// The end of the validity period, inclusive.
    #[must_use]
    pub const fn not_after(&self) -> Timestamp {
        self.not_after
    }

    /// Whether the certificate is valid at `at`, both ends inclusive.
    #[must_use]
    pub fn is_valid_at(&self, at: Timestamp) -> bool {
        self.not_before <= at && at <= self.not_after
    }

    /// The certified public key, or [`None`] when it is of an algorithm, size
    /// or curve this crate cannot use.
    #[must_use]
    pub const fn public_key(&self) -> Option<&VerificationKey> {
        self.public_key.as_ref()
    }

    /// The `subjectPublicKeyInfo`, exactly as encoded.
    #[must_use]
    pub fn public_key_info(&self) -> &[u8] {
        &self.public_key_info
    }

    /// The `signatureAlgorithm` identifier as dotted decimal.
    #[must_use]
    pub fn signature_algorithm(&self) -> String {
        Der::new(&self.signature_algorithm)
            .read_sequence()
            .and_then(|mut algorithm| algorithm.read(tag::OID))
            .map_or_else(|_| String::new(), oid::dotted)
    }

    /// The `AlgorithmIdentifier` the certificate was signed with, as encoded.
    pub(crate) fn signature_algorithm_encoding(&self) -> &[u8] {
        &self.signature_algorithm
    }

    /// The `signatureValue`.
    #[must_use]
    pub fn signature(&self) -> &[u8] {
        &self.signature
    }

    /// Whether a `basicConstraints` extension says the certificate may issue
    /// other certificates.
    ///
    /// A certificate with no `basicConstraints` at all is **not** a certificate
    /// authority: [RFC 5280 §4.2.1.9](https://www.rfc-editor.org/rfc/rfc5280#section-4.2.1.9)
    /// makes `cA` default to false, and treating a silent certificate as a CA
    /// is the flaw that let any leaf certificate mint others.
    #[must_use]
    pub const fn is_certificate_authority(&self) -> bool {
        self.certificate_authority
    }

    /// The `pathLenConstraint` of the `basicConstraints` extension: how many
    /// further CA certificates may appear below this one.
    #[must_use]
    pub const fn path_length_constraint(&self) -> Option<usize> {
        self.path_length_constraint
    }

    /// The `keyUsage` extension, if the certificate has one.
    #[must_use]
    pub const fn key_usage(&self) -> Option<KeyUsage> {
        self.key_usage
    }

    /// The `subjectKeyIdentifier` extension, if the certificate has one.
    #[must_use]
    pub fn subject_key_identifier(&self) -> Option<&[u8]> {
        self.subject_key_identifier.as_deref()
    }

    /// The `keyIdentifier` of the `authorityKeyIdentifier` extension, if the
    /// certificate has one.
    #[must_use]
    pub fn authority_key_identifier(&self) -> Option<&[u8]> {
        self.authority_key_identifier.as_deref()
    }

    /// The identifiers of the critical extensions this crate does not process.
    ///
    /// A certificate with any of these cannot be validated — see
    /// [`CertificateError::UnknownCriticalExtension`].
    pub fn unhandled_critical_extensions(&self) -> impl ExactSizeIterator<Item = &str> {
        self.unhandled_critical_extensions
            .iter()
            .map(String::as_str)
    }

    /// Whether the subject and issuer name the same entry, which is what
    /// [RFC 5280 §6.1](https://www.rfc-editor.org/rfc/rfc5280#section-6.1)
    /// calls a self-issued certificate.
    ///
    /// This says nothing about the signature: a self-issued certificate is
    /// usually self-*signed* as well, but the two are different claims.
    #[must_use]
    pub fn is_self_issued(&self) -> bool {
        self.subject.matches(&self.issuer)
    }

    /// Whether this certificate's `authorityKeyIdentifier` rules `issuer` out.
    ///
    /// The identifier is a *hint* for finding the right issuer among several
    /// that share a name, which is how a certificate authority rolls its key
    /// over. It is never evidence on its own — only the signature is — so a
    /// candidate is rejected here only when both certificates carry an
    /// identifier and the two disagree.
    #[must_use]
    pub fn is_plausible_issuer(&self, issuer: &Self) -> bool {
        let identifier_conflicts = matches!(
            (
                self.authority_key_identifier.as_deref(),
                issuer.subject_key_identifier.as_deref(),
            ),
            (Some(wanted), Some(offered)) if wanted != offered
        );
        let serial_conflicts = self
            .authority_certificate_serial
            .as_deref()
            .is_some_and(|wanted| wanted != issuer.serial_number.as_slice());
        !identifier_conflicts && !serial_conflicts
    }

    /// Picks the end-entity certificate out of an unordered bag of them.
    ///
    /// A `<ds:X509Data>` may carry a whole chain, and
    /// [XML Signature §4.5.4](https://www.w3.org/TR/xmldsig-core1/#sec-X509Data)
    /// does not require any particular order — the signer's certificate is
    /// conventionally first, and that is a convention, not a rule. The one that
    /// no other certificate in the bag was issued by is the leaf; if every
    /// certificate issued another (or the bag holds exactly one), the first is
    /// taken.
    #[must_use]
    pub fn select_end_entity(certificates: &[Self]) -> Option<&Self> {
        certificates
            .iter()
            .find(|candidate| {
                !certificates.iter().any(|other| {
                    !std::ptr::eq(*candidate, other) && other.issuer.matches(&candidate.subject)
                })
            })
            .or_else(|| certificates.first())
    }
}

impl PartialEq for Certificate {
    /// Encoding equality: two certificates are the same certificate when they
    /// are the same octets.
    fn eq(&self, other: &Self) -> bool {
        self.encoding == other.encoding
    }
}

impl Eq for Certificate {}

impl fmt::Debug for Certificate {
    /// Prints what identifies the certificate rather than the kilobyte of DER
    /// it was read from.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Certificate")
            .field("subject", &self.subject.to_string())
            .field("issuer", &self.issuer.to_string())
            .field("not_before", &self.not_before.to_string())
            .field("not_after", &self.not_after.to_string())
            .field("certificate_authority", &self.certificate_authority)
            .finish_non_exhaustive()
    }
}

/// The permissions a `keyUsage` extension grants —
/// [RFC 5280 §4.2.1.3](https://www.rfc-editor.org/rfc/rfc5280#section-4.2.1.3).
///
/// A certificate without the extension grants everything; one *with* it grants
/// only what it names, and using the key for anything else is forbidden.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyUsage {
    bits: u16,
}

impl KeyUsage {
    /// Whether the bit at `position` in the RFC's numbering is set. Bit 0 is
    /// the most significant bit of the first octet.
    const fn has(self, position: u8) -> bool {
        self.bits & (0x8000_u16 >> position) != 0
    }

    /// `digitalSignature` (bit 0) — the usage an XML signature needs.
    #[must_use]
    pub const fn digital_signature(self) -> bool {
        self.has(0)
    }

    /// `contentCommitment` (bit 1), which X.509 used to call `nonRepudiation`.
    #[must_use]
    pub const fn content_commitment(self) -> bool {
        self.has(1)
    }

    /// `keyEncipherment` (bit 2).
    #[must_use]
    pub const fn key_encipherment(self) -> bool {
        self.has(2)
    }

    /// `dataEncipherment` (bit 3).
    #[must_use]
    pub const fn data_encipherment(self) -> bool {
        self.has(3)
    }

    /// `keyAgreement` (bit 4).
    #[must_use]
    pub const fn key_agreement(self) -> bool {
        self.has(4)
    }

    /// `keyCertSign` (bit 5) — the usage every certificate authority in a path
    /// must have.
    #[must_use]
    pub const fn key_certificate_sign(self) -> bool {
        self.has(5)
    }

    /// `cRLSign` (bit 6).
    #[must_use]
    pub const fn crl_sign(self) -> bool {
        self.has(6)
    }

    /// `encipherOnly` (bit 7).
    #[must_use]
    pub const fn encipher_only(self) -> bool {
        self.has(7)
    }

    /// `decipherOnly` (bit 8).
    #[must_use]
    pub const fn decipher_only(self) -> bool {
        self.has(8)
    }
}

/// Wraps a DER failure as a certificate failure.
fn malformed(error: DerError) -> CertificateError {
    CertificateError::Malformed(error.to_string())
}

/// The magnitude of a DER `INTEGER`'s contents, without the sign octet an
/// unsigned value carries when its top bit is set.
fn unsigned_magnitude(contents: &[u8]) -> Vec<u8> {
    match contents {
        [0x00, rest @ ..] if !rest.is_empty() => rest.to_vec(),
        _ => contents.to_vec(),
    }
}

/// Reads `Validity ::= SEQUENCE { notBefore Time, notAfter Time }`.
fn read_validity(tbs: &mut Der<'_>) -> Result<(Timestamp, Timestamp), CertificateError> {
    let mut validity = tbs.read_sequence().map_err(malformed)?;
    let not_before = read_time(&mut validity)?;
    let not_after = read_time(&mut validity)?;
    validity.finish().map_err(malformed)?;
    Ok((not_before, not_after))
}

/// Reads one `Time ::= CHOICE { utcTime UTCTime, generalTime GeneralizedTime }`.
fn read_time(validity: &mut Der<'_>) -> Result<Timestamp, CertificateError> {
    let (value_tag, _, contents) = validity.read_any().map_err(malformed)?;
    let parsed = match value_tag {
        tag::UTC_TIME => parse_utc_time(contents),
        tag::GENERALIZED_TIME => parse_generalized_time(contents),
        _ => None,
    };
    parsed.ok_or_else(|| {
        CertificateError::Malformed(
            "a validity date is not a UTCTime or GeneralizedTime in the form RFC 5280 requires"
                .to_owned(),
        )
    })
}

/// Turns a `subjectPublicKeyInfo` into a usable verification key, or [`None`]
/// when this crate cannot represent the key it holds.
fn read_public_key(public_key_info: &[u8]) -> Option<VerificationKey> {
    match der::parse_public_key_info(public_key_info).ok()? {
        der::PublicKeyComponents::Rsa(components) => {
            RsaPublicKey::from_components(&components.modulus, &components.exponent)
                .ok()
                .map(VerificationKey::Rsa)
        }
        der::PublicKeyComponents::Ec(components) => {
            EcPublicKey::from_sec1_point(components.curve, &components.point)
                .ok()
                .map(VerificationKey::Ec)
        }
        der::PublicKeyComponents::Dsa(components) => DsaPublicKey::from_components(
            &components.p,
            &components.q,
            &components.g,
            &components.y,
        )
        .ok()
        .map(VerificationKey::Dsa),
    }
}

/// Reads a `KeyUsage ::= BIT STRING` extension value.
fn read_key_usage(value: &[u8]) -> Result<KeyUsage, CertificateError> {
    let mut wrapper = Der::new(value);
    let (_, bits) = wrapper.read_bit_string_padded().map_err(malformed)?;
    wrapper.finish().map_err(malformed)?;
    let high = bits.first().copied().unwrap_or(0);
    let low = bits.get(1).copied().unwrap_or(0);
    Ok(KeyUsage {
        bits: u16::from_be_bytes([high, low]),
    })
}

/// Reads a `KeyIdentifier ::= OCTET STRING` extension value.
fn read_key_identifier(value: &[u8]) -> Result<Vec<u8>, CertificateError> {
    let mut wrapper = Der::new(value);
    let identifier = wrapper.read(tag::OCTET_STRING).map_err(malformed)?.to_vec();
    wrapper.finish().map_err(malformed)?;
    Ok(identifier)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkey;
    use crate::x509::testcert::{self, CertificateSpec};

    fn pinned_certificate() -> Result<Certificate, CertificateError> {
        Certificate::parse(&testkey::bytes(testkey::CERTIFICATE_BASE64))
    }

    #[test]
    fn the_pinned_certificate_reads_field_for_field() -> Result<(), CertificateError> {
        let certificate = pinned_certificate()?;
        assert_eq!(certificate.version(), 3);
        assert_eq!(certificate.serial_number(), [0x01, 0x23, 0x45, 0x67, 0x89]);
        assert_eq!(certificate.subject().to_string(), "CN=OxiXML Test");
        assert_eq!(certificate.issuer().to_string(), "CN=OxiXML Test");
        assert!(certificate.is_self_issued());
        assert_eq!(certificate.signature_algorithm(), "1.2.840.113549.1.1.11");
        assert_eq!(certificate.not_before().to_string(), "2026-01-01T00:00:00Z");
        assert_eq!(certificate.not_after().to_string(), "2036-01-01T00:00:00Z");
        assert!(certificate.is_valid_at(Timestamp::from_unix_seconds(1_800_000_000)));
        assert!(!certificate.is_valid_at(Timestamp::from_unix_seconds(0)));
        // No extensions at all: not a certificate authority, no key usage
        // restriction, nothing critical left unprocessed.
        assert!(!certificate.is_certificate_authority());
        assert_eq!(certificate.key_usage(), None);
        assert_eq!(certificate.unhandled_critical_extensions().len(), 0);
        assert!(matches!(
            certificate.public_key(),
            Some(VerificationKey::Rsa(_))
        ));
        assert_eq!(
            certificate.signed_octets().first().copied(),
            Some(tag::SEQUENCE)
        );
        Ok(())
    }

    /// The signature covers the `tbsCertificate` *as encoded*, tag and length
    /// included — not its contents. Getting this wrong makes every chain fail
    /// in a way that looks like a cryptography bug.
    #[test]
    fn the_signed_octets_are_a_whole_element() -> Result<(), CertificateError> {
        let certificate = pinned_certificate()?;
        let signed = certificate.signed_octets();
        assert!(
            certificate
                .encoding()
                .windows(signed.len())
                .any(|window| window == signed)
        );
        let mut reader = Der::new(signed);
        assert!(reader.read_sequence().is_ok());
        assert!(reader.finish().is_ok());
        Ok(())
    }

    #[test]
    fn extensions_are_read_and_unknown_critical_ones_recorded() -> Result<(), CertificateError> {
        let spec = CertificateSpec::new("Authority")
            .certificate_authority(Some(2))
            .key_usage(0b0000_0110)
            .subject_key_identifier(vec![0xaa, 0xbb])
            .authority_key_identifier(vec![0xcc, 0xdd])
            // 2.5.29.30, nameConstraints: recognised by the RFC, not processed
            // here, and marked critical.
            .critical_extension(vec![0x55, 0x1d, 0x1e], vec![0x30, 0x00]);
        let certificate = Certificate::parse(&testcert::self_signed(&spec))?;
        assert!(certificate.is_certificate_authority());
        assert_eq!(certificate.path_length_constraint(), Some(2));
        let usage = certificate.key_usage().expect("a keyUsage extension");
        assert!(usage.key_certificate_sign());
        assert!(usage.crl_sign());
        assert!(!usage.digital_signature());
        assert_eq!(
            certificate.subject_key_identifier(),
            Some([0xaa, 0xbb].as_slice())
        );
        assert_eq!(
            certificate.authority_key_identifier(),
            Some([0xcc, 0xdd].as_slice())
        );
        assert_eq!(
            certificate
                .unhandled_critical_extensions()
                .collect::<Vec<_>>(),
            ["2.5.29.30"]
        );
        Ok(())
    }

    /// A non-critical extension nobody processes is exactly what the
    /// "non-critical" marking is for, and must not be recorded.
    #[test]
    fn an_unknown_non_critical_extension_is_ignored() -> Result<(), CertificateError> {
        let spec =
            CertificateSpec::new("Leaf").extension(vec![0x55, 0x1d, 0x1f], vec![0x30, 0x00], false);
        let certificate = Certificate::parse(&testcert::self_signed(&spec))?;
        assert_eq!(certificate.unhandled_critical_extensions().len(), 0);
        Ok(())
    }

    /// `issuerUniqueID` and `subjectUniqueID` sit between the public key and
    /// the extensions. A parser that does not skip them reads the extensions of
    /// such a certificate as absent — which silently turns off every extension
    /// check there is.
    #[test]
    fn unique_identifiers_do_not_hide_the_extensions() -> Result<(), CertificateError> {
        let spec = CertificateSpec::new("Authority")
            .certificate_authority(None)
            .unique_identifiers(true);
        let certificate = Certificate::parse(&testcert::self_signed(&spec))?;
        assert!(
            certificate.is_certificate_authority(),
            "the basicConstraints extension must still be found"
        );
        Ok(())
    }

    #[test]
    fn a_repeated_extension_is_refused() {
        let spec = CertificateSpec::new("Leaf")
            .extension(vec![0x55, 0x1d, 0x0f], vec![0x03, 0x02, 0x07, 0x80], false)
            .extension(vec![0x55, 0x1d, 0x0f], vec![0x03, 0x02, 0x07, 0x80], false);
        let error = Certificate::parse(&testcert::self_signed(&spec))
            .expect_err("a repeated extension is malformed");
        assert!(error.to_string().contains("more than once"), "{error}");
    }

    /// RFC 5280 §4.1.1.2: the algorithm identifier repeated outside the signed
    /// part must be the one inside it, or a certificate could be re-labelled
    /// in transit.
    #[test]
    fn a_mismatched_outer_signature_algorithm_is_refused() {
        let mut encoding = testkey::bytes(testkey::CERTIFICATE_BASE64);
        // The outer AlgorithmIdentifier's OID ends in the digest selector; 0x0b
        // is sha256WithRSAEncryption and 0x05 is sha1WithRSAEncryption. The
        // second occurrence in the encoding is the outer one.
        let needle = [0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0b];
        let positions: Vec<usize> = encoding
            .windows(needle.len())
            .enumerate()
            .filter(|(_, window)| *window == needle)
            .map(|(index, _)| index)
            .collect();
        assert_eq!(positions.len(), 2, "the identifier appears twice");
        let last = positions.last().copied().unwrap_or(0) + needle.len() - 1;
        if let Some(byte) = encoding.get_mut(last) {
            *byte = 0x05;
        }
        let error = Certificate::parse(&encoding).expect_err("the two identifiers now differ");
        assert!(error.to_string().contains("signatureAlgorithm"), "{error}");
    }

    #[test]
    fn a_truncated_certificate_is_refused() {
        let encoding = testkey::bytes(testkey::CERTIFICATE_BASE64);
        let truncated = encoding.get(..encoding.len() - 1).unwrap_or_default();
        assert!(Certificate::parse(truncated).is_err());
        assert!(Certificate::parse(&[]).is_err());
    }

    #[test]
    fn trailing_octets_after_the_certificate_are_refused() {
        let mut encoding = testkey::bytes(testkey::CERTIFICATE_BASE64);
        encoding.push(0x00);
        assert!(Certificate::parse(&encoding).is_err());
    }

    /// The end entity is the certificate nothing else was issued by.
    #[test]
    fn the_end_entity_is_picked_out_of_an_unordered_bag() -> Result<(), CertificateError> {
        let chain = testcert::chain()?;
        let bag = vec![
            chain.root.clone(),
            chain.leaf.clone(),
            chain.intermediate.clone(),
        ];
        assert_eq!(
            Certificate::select_end_entity(&bag).map(Certificate::subject),
            Some(chain.leaf.subject())
        );
        assert_eq!(
            Certificate::select_end_entity(&[]).map(Certificate::subject),
            None
        );
        Ok(())
    }

    #[test]
    fn a_key_usage_bit_string_reads_the_documented_positions() -> Result<(), CertificateError> {
        // 0x80 with seven unused bits is digitalSignature alone.
        let usage = read_key_usage(&[0x03, 0x02, 0x07, 0x80])?;
        assert!(usage.digital_signature());
        assert!(!usage.content_commitment());
        assert!(!usage.key_certificate_sign());
        // 0x04 in the second octet, seven unused bits: decipherOnly.
        let wide = read_key_usage(&[0x03, 0x03, 0x07, 0x00, 0x80])?;
        assert!(wide.decipher_only());
        assert!(!wide.digital_signature());
        // Padding bits that are not zero are not a DER BIT STRING.
        assert!(read_key_usage(&[0x03, 0x02, 0x07, 0x81]).is_err());
        Ok(())
    }
}

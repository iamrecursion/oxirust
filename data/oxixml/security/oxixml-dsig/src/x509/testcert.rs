//! A certificate factory for the unit tests.
//!
//! Path validation cannot be tested against a key alone: it needs certificates
//! that really are signed by one another, that really do expire, and that carry
//! the extensions the algorithm branches on. Rather than pinning a wall of
//! base64 whose contents nobody can check, the tests build the certificates
//! here — from the same DER shapes the parser reads and the same signing
//! primitives the crate ships — so that a fixture and the code under test can
//! never drift apart.
//!
//! The keys are pinned: the RSA pair of [`crate::testkey`] and elliptic-curve
//! scalars derived from a one-octet seed, so every certificate built here is
//! byte-for-byte reproducible.

use crate::algorithm::DigestAlgorithm;
use crate::crypto::der::{self, tag};
use crate::crypto::{ecdsa, pkcs1};
use crate::key::{EcPrivateKey, NamedCurve, RsaPrivateKey};
use crate::testkey;
use crate::x509::certificate::Certificate;
use crate::x509::error::CertificateError;
use crate::x509::oid;
use crate::x509::time::Timestamp;

/// Encodes a DER length in the shortest form.
fn length(value: usize) -> Vec<u8> {
    if value < 0x80 {
        return vec![u8::try_from(value).unwrap_or(0)];
    }
    let octets = value.to_be_bytes();
    let significant = octets
        .iter()
        .position(|byte| *byte != 0)
        .map_or(&octets[..], |first| &octets[first..]);
    let mut encoded = vec![0x80 | u8::try_from(significant.len()).unwrap_or(1)];
    encoded.extend_from_slice(significant);
    encoded
}

/// Builds one DER TLV.
pub(crate) fn tlv(value_tag: u8, contents: &[u8]) -> Vec<u8> {
    let mut encoded = vec![value_tag];
    encoded.extend(length(contents.len()));
    encoded.extend_from_slice(contents);
    encoded
}

/// Builds a DER `INTEGER` from an unsigned big-endian magnitude, adding the
/// sign octet exactly when the encoding needs one and never otherwise — a
/// non-minimal INTEGER is refused by this crate's own reader, and a fixture
/// that produced one would read as a validation failure.
pub(crate) fn integer(magnitude: &[u8]) -> Vec<u8> {
    let trimmed = magnitude
        .iter()
        .position(|byte| *byte != 0)
        .map_or(&magnitude[magnitude.len()..], |first| &magnitude[first..]);
    let mut contents = Vec::with_capacity(trimmed.len() + 1);
    if trimmed.first().is_some_and(|byte| byte & 0x80 != 0) {
        contents.push(0x00);
    }
    if trimmed.is_empty() {
        contents.push(0x00);
    } else {
        contents.extend_from_slice(trimmed);
    }
    tlv(tag::INTEGER, &contents)
}

/// A `Name` holding a single `CN` as a `PrintableString`.
fn common_name(name: &str) -> Vec<u8> {
    let mut pair = tlv(tag::OID, &[0x55, 0x04, 0x03]);
    pair.extend(tlv(0x13, name.as_bytes()));
    tlv(tag::SEQUENCE, &tlv(tag::SET, &tlv(tag::SEQUENCE, &pair)))
}

/// A `UTCTime`, which is the encoding every date these fixtures use falls in.
fn utc_time(at: Timestamp) -> Vec<u8> {
    let (year, month, day, hour, minute, second) = at.to_utc();
    let text = format!(
        "{:02}{month:02}{day:02}{hour:02}{minute:02}{second:02}Z",
        year.rem_euclid(100)
    );
    tlv(tag::UTC_TIME, text.as_bytes())
}

/// The key a fixture certificate certifies, and that signs the ones below it.
pub(crate) enum Signer {
    /// The pinned 1024-bit RSA pair, signing with `sha256WithRSAEncryption`.
    Rsa(Box<RsaPrivateKey>),
    /// A P-256 key, signing with `ecdsa-with-SHA256`.
    Ec(Box<EcPrivateKey>),
}

impl Signer {
    /// The pinned RSA key of [`crate::testkey`].
    pub(crate) fn rsa() -> Self {
        Self::Rsa(Box::new(
            RsaPrivateKey::from_der(&testkey::bytes(testkey::RSA_PRIVATE_KEY_BASE64))
                .expect("the pinned test key parses"),
        ))
    }

    /// A P-256 key whose scalar is `seed` repeated, which is in range for every
    /// seed from 1 to 254.
    pub(crate) fn ec(seed: u8) -> Self {
        Self::Ec(Box::new(
            EcPrivateKey::from_scalar(NamedCurve::P256, &[seed; 32])
                .expect("a repeated octet is a valid P-256 scalar"),
        ))
    }

    /// The `SubjectPublicKeyInfo` of the matching public key.
    fn public_key_info(&self) -> Vec<u8> {
        match self {
            Self::Rsa(_) => testkey::bytes(testkey::SPKI_BASE64),
            Self::Ec(key) => {
                let mut algorithm = tlv(tag::OID, der::EC_PUBLIC_KEY_OID);
                algorithm.extend(tlv(tag::OID, der::P256_OID));
                let mut key_bits = vec![0x00];
                key_bits.extend_from_slice(key.public_key().encoded_point());
                let mut contents = tlv(tag::SEQUENCE, &algorithm);
                contents.extend(tlv(tag::BIT_STRING, &key_bits));
                tlv(tag::SEQUENCE, &contents)
            }
        }
    }

    /// The `AlgorithmIdentifier` of the signature this key produces.
    fn algorithm_identifier(&self) -> Vec<u8> {
        match self {
            Self::Rsa(_) => {
                let mut contents = tlv(tag::OID, oid::RSA_SHA256);
                contents.extend(tlv(tag::NULL, &[]));
                tlv(tag::SEQUENCE, &contents)
            }
            // RFC 5758 §3.2: the parameters of an ECDSA identifier are absent,
            // not NULL.
            Self::Ec(_) => tlv(tag::SEQUENCE, &tlv(tag::OID, oid::ECDSA_SHA256)),
        }
    }

    /// Signs `message`, producing the octets an X.509 `signatureValue` carries:
    /// the RSA signature itself, or the ASN.1 `SEQUENCE { r, s }` that ECDSA
    /// uses everywhere except inside an XML signature.
    fn sign(&self, message: &[u8]) -> Vec<u8> {
        match self {
            Self::Rsa(key) => pkcs1::sign(key.n(), key.d(), DigestAlgorithm::Sha256, message)
                .expect("the pinned modulus is wide enough for SHA-256"),
            Self::Ec(key) => {
                let raw = sign_p1363(key, message);
                let half = raw.len() / 2;
                let mut contents = integer(raw.get(..half).unwrap_or_default());
                contents.extend(integer(raw.get(half..).unwrap_or_default()));
                tlv(tag::SEQUENCE, &contents)
            }
        }
    }
}

/// An ECDSA signature in the fixed-width `r ‖ s` form XML Signature uses —
/// which an X.509 certificate must *not* carry, and which the tests feed to the
/// validator to prove it does not accept one.
fn sign_p1363(key: &EcPrivateKey, message: &[u8]) -> Vec<u8> {
    ecdsa::sign(
        key.parameters(),
        key.scalar(),
        DigestAlgorithm::Sha256,
        message,
    )
    .expect("the pinned scalar is in range")
}

/// What a fixture certificate should say.
pub(crate) struct CertificateSpec {
    common_name: String,
    serial: Vec<u8>,
    not_before: Timestamp,
    not_after: Timestamp,
    certificate_authority: bool,
    path_length: Option<usize>,
    key_usage: Option<u8>,
    subject_key_identifier: Option<Vec<u8>>,
    authority_key_identifier: Option<Vec<u8>>,
    extensions: Vec<(Vec<u8>, Vec<u8>, bool)>,
    unique_identifiers: bool,
    p1363_signature: bool,
    public_key_info: Option<Vec<u8>>,
}

impl CertificateSpec {
    /// A certificate for `CN=<common_name>`, valid for the whole of the 2020s
    /// and with no extensions.
    pub(crate) fn new(common_name: &str) -> Self {
        Self {
            common_name: common_name.to_owned(),
            serial: vec![0x01],
            not_before: Timestamp::from_unix_seconds(1_577_836_800), // 2020-01-01
            not_after: Timestamp::from_unix_seconds(2_208_988_800),  // 2040-01-01
            certificate_authority: false,
            path_length: None,
            key_usage: None,
            subject_key_identifier: None,
            authority_key_identifier: None,
            extensions: Vec::new(),
            unique_identifiers: false,
            p1363_signature: false,
            public_key_info: None,
        }
    }

    /// Certifies the given `SubjectPublicKeyInfo` instead of the one belonging
    /// to the signing key, which is how a certificate for a key this crate
    /// cannot represent is produced.
    pub(crate) fn public_key_info(mut self, encoding: Vec<u8>) -> Self {
        self.public_key_info = Some(encoding);
        self
    }

    /// Sets the serial number.
    pub(crate) fn serial(mut self, serial: u8) -> Self {
        self.serial = vec![serial];
        self
    }

    /// Sets the validity period.
    pub(crate) const fn validity(mut self, not_before: Timestamp, not_after: Timestamp) -> Self {
        self.not_before = not_before;
        self.not_after = not_after;
        self
    }

    /// Adds a critical `basicConstraints` extension with `cA` true, and the
    /// given `pathLenConstraint` if there is one.
    pub(crate) const fn certificate_authority(mut self, path_length: Option<usize>) -> Self {
        self.certificate_authority = true;
        self.path_length = path_length;
        self
    }

    /// Adds a `keyUsage` extension whose first octet is `bits`.
    pub(crate) const fn key_usage(mut self, bits: u8) -> Self {
        self.key_usage = Some(bits);
        self
    }

    /// Adds a `subjectKeyIdentifier` extension.
    pub(crate) fn subject_key_identifier(mut self, identifier: Vec<u8>) -> Self {
        self.subject_key_identifier = Some(identifier);
        self
    }

    /// Adds an `authorityKeyIdentifier` extension.
    pub(crate) fn authority_key_identifier(mut self, identifier: Vec<u8>) -> Self {
        self.authority_key_identifier = Some(identifier);
        self
    }

    /// Adds an extension with the given identifier, value and criticality.
    pub(crate) fn extension(mut self, identifier: Vec<u8>, value: Vec<u8>, critical: bool) -> Self {
        self.extensions.push((identifier, value, critical));
        self
    }

    /// Adds an extension marked critical.
    pub(crate) fn critical_extension(self, identifier: Vec<u8>, value: Vec<u8>) -> Self {
        self.extension(identifier, value, true)
    }

    /// Adds the two v2 unique identifiers, which sit between the public key and
    /// the extensions.
    pub(crate) const fn unique_identifiers(mut self, present: bool) -> Self {
        self.unique_identifiers = present;
        self
    }

    /// Writes the ECDSA signature in the fixed-width form of XML Signature
    /// instead of the ASN.1 form X.509 requires.
    pub(crate) const fn p1363_signature(mut self, raw: bool) -> Self {
        self.p1363_signature = raw;
        self
    }

    /// The encoded extensions, or [`None`] when there are none.
    fn encoded_extensions(&self) -> Option<Vec<u8>> {
        let mut extensions = Vec::new();
        if self.certificate_authority {
            let mut constraints = tlv(tag::BOOLEAN, &[0xff]);
            if let Some(limit) = self.path_length {
                constraints.extend(integer(&limit.to_be_bytes()));
            }
            extensions.extend(extension(
                oid::BASIC_CONSTRAINTS,
                &tlv(tag::SEQUENCE, &constraints),
                true,
            ));
        }
        if let Some(bits) = self.key_usage {
            let value = if bits == 0 {
                tlv(tag::BIT_STRING, &[0x00])
            } else {
                let unused = u8::try_from(bits.trailing_zeros()).unwrap_or(0);
                tlv(tag::BIT_STRING, &[unused, bits])
            };
            extensions.extend(extension(oid::KEY_USAGE, &value, true));
        }
        if let Some(identifier) = &self.subject_key_identifier {
            extensions.extend(extension(
                oid::SUBJECT_KEY_IDENTIFIER,
                &tlv(tag::OCTET_STRING, identifier),
                false,
            ));
        }
        if let Some(identifier) = &self.authority_key_identifier {
            let inner = tlv(tag::IMPLICIT_0, identifier);
            extensions.extend(extension(
                oid::AUTHORITY_KEY_IDENTIFIER,
                &tlv(tag::SEQUENCE, &inner),
                false,
            ));
        }
        for (identifier, value, critical) in &self.extensions {
            extensions.extend(extension(identifier, value, *critical));
        }
        (!extensions.is_empty()).then(|| tlv(tag::CONTEXT_3, &tlv(tag::SEQUENCE, &extensions)))
    }
}

/// One `Extension`, with the `critical` flag omitted when it is false as DER
/// requires of a field at its default.
fn extension(identifier: &[u8], value: &[u8], critical: bool) -> Vec<u8> {
    let mut contents = tlv(tag::OID, identifier);
    if critical {
        contents.extend(tlv(tag::BOOLEAN, &[0xff]));
    }
    contents.extend(tlv(tag::OCTET_STRING, value));
    tlv(tag::SEQUENCE, &contents)
}

/// Builds a certificate for `spec`, certifying `subject_key` and signed by
/// `issuer_key` in the name of `issuer_name`.
pub(crate) fn build(
    spec: &CertificateSpec,
    subject_key: &Signer,
    issuer_name: &str,
    issuer_key: &Signer,
) -> Vec<u8> {
    let algorithm = issuer_key.algorithm_identifier();
    let mut tbs = tlv(tag::CONTEXT_0, &integer(&[0x02]));
    tbs.extend(integer(&spec.serial));
    tbs.extend_from_slice(&algorithm);
    tbs.extend(common_name(issuer_name));
    let mut validity = utc_time(spec.not_before);
    validity.extend(utc_time(spec.not_after));
    tbs.extend(tlv(tag::SEQUENCE, &validity));
    tbs.extend(common_name(&spec.common_name));
    tbs.extend(
        spec.public_key_info
            .clone()
            .unwrap_or_else(|| subject_key.public_key_info()),
    );
    if spec.unique_identifiers {
        tbs.extend(tlv(tag::IMPLICIT_1, &[0x00, 0xa5]));
        tbs.extend(tlv(tag::IMPLICIT_2, &[0x00, 0x5a]));
    }
    if let Some(extensions) = spec.encoded_extensions() {
        tbs.extend(extensions);
    }
    let tbs = tlv(tag::SEQUENCE, &tbs);

    let signature = match (spec.p1363_signature, issuer_key) {
        (true, Signer::Ec(key)) => sign_p1363(key, &tbs),
        _ => issuer_key.sign(&tbs),
    };
    let mut signature_bits = vec![0x00];
    signature_bits.extend_from_slice(&signature);

    let mut certificate = tbs;
    certificate.extend_from_slice(&algorithm);
    certificate.extend(tlv(tag::BIT_STRING, &signature_bits));
    tlv(tag::SEQUENCE, &certificate)
}

/// Builds a certificate that signs itself with the pinned RSA key.
pub(crate) fn self_signed(spec: &CertificateSpec) -> Vec<u8> {
    let key = Signer::rsa();
    let name = spec.common_name.clone();
    build(spec, &key, &name, &key)
}

/// Flips one bit of the `signatureValue`, leaving everything else — including
/// every length — exactly as it was.
pub(crate) fn break_signature(encoding: &[u8]) -> Vec<u8> {
    let mut broken = encoding.to_vec();
    if let Some(last) = broken.last_mut() {
        *last ^= 0x01;
    }
    broken
}

/// The three certificates of an ordinary chain, and the keys behind them.
pub(crate) struct TestChain {
    /// The self-signed trust anchor.
    pub(crate) root: Certificate,
    /// The certificate authority the root issued.
    pub(crate) intermediate: Certificate,
    /// The end-entity certificate the intermediate issued.
    pub(crate) leaf: Certificate,
}

/// The seed of the root key.
pub(crate) const ROOT_SEED: u8 = 0x11;
/// The seed of the intermediate key.
pub(crate) const INTERMEDIATE_SEED: u8 = 0x22;
/// The seed of the leaf key.
pub(crate) const LEAF_SEED: u8 = 0x33;

/// The subject common name of the root.
pub(crate) const ROOT_NAME: &str = "OxiXML Test Root";
/// The subject common name of the intermediate.
pub(crate) const INTERMEDIATE_NAME: &str = "OxiXML Test Intermediate";
/// The subject common name of the leaf.
pub(crate) const LEAF_NAME: &str = "OxiXML Test Leaf";

/// The specification of the root as [`chain`] builds it.
pub(crate) fn root_spec() -> CertificateSpec {
    CertificateSpec::new(ROOT_NAME)
        .serial(1)
        .certificate_authority(None)
        .key_usage(0b0000_0100)
}

/// The specification of the intermediate as [`chain`] builds it.
pub(crate) fn intermediate_spec() -> CertificateSpec {
    CertificateSpec::new(INTERMEDIATE_NAME)
        .serial(2)
        .certificate_authority(None)
        .key_usage(0b0000_0100)
}

/// The specification of the leaf as [`chain`] builds it.
pub(crate) fn leaf_spec() -> CertificateSpec {
    CertificateSpec::new(LEAF_NAME)
        .serial(3)
        .key_usage(0b1000_0000)
}

/// Builds `root → intermediate → leaf`, all on P-256 with SHA-256.
pub(crate) fn chain() -> Result<TestChain, CertificateError> {
    let root_key = Signer::ec(ROOT_SEED);
    let intermediate_key = Signer::ec(INTERMEDIATE_SEED);
    let leaf_key = Signer::ec(LEAF_SEED);
    let root = build(&root_spec(), &root_key, ROOT_NAME, &root_key);
    let intermediate = build(
        &intermediate_spec(),
        &intermediate_key,
        ROOT_NAME,
        &root_key,
    );
    let leaf = build(
        &leaf_spec(),
        &leaf_key,
        INTERMEDIATE_NAME,
        &intermediate_key,
    );
    Ok(TestChain {
        root: Certificate::parse(&root)?,
        intermediate: Certificate::parse(&intermediate)?,
        leaf: Certificate::parse(&leaf)?,
    })
}

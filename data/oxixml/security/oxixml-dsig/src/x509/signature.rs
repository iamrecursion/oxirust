//! Checking the signature on a certificate.
//!
//! The arithmetic is the same as for an XML signature and reuses the same
//! verifiers, but the *encodings* around it are not, and the difference is
//! exactly where a naive reuse goes wrong:
//!
//! * **the scheme is named by an OID**, in the `AlgorithmIdentifier` of the
//!   certificate, rather than by an `Algorithm` URI;
//! * **an ECDSA or DSA signature is ASN.1**, `SEQUENCE { r INTEGER, s INTEGER }`
//!   ([RFC 3279 §2.2.2](https://www.rfc-editor.org/rfc/rfc3279#section-2.2.2)),
//!   where [XML Signature §6.4.3](https://www.w3.org/TR/xmldsig-core1/#sec-ECDSA)
//!   uses the plain concatenation of two fixed-width integers. The two are not
//!   interchangeable and neither form may be accepted where the other is
//!   required: this module decodes the ASN.1 form and pads each component out
//!   to the width the crate's verifiers take, and the decoding inherits the DER
//!   reader's strictness, so a component with a redundant leading octet or a
//!   trailing byte after the `SEQUENCE` is refused rather than silently
//!   tolerated.
//!
//! RSA needs no such conversion — a PKCS#1 signature is a bare integer of the
//! modulus's width in both worlds.

use crate::algorithm::DigestAlgorithm;
use crate::crypto::der::{Der, tag};
use crate::crypto::{dsa, ecdsa, pkcs1, pss};
use crate::key::VerificationKey;
use crate::model::PssParameters;
use crate::x509::certificate::Certificate;
use crate::x509::error::CertificateError;
use crate::x509::oid;

/// The signature scheme an X.509 `AlgorithmIdentifier` names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CertificateSignature {
    /// RSASSA-PKCS1-v1_5 over the named digest.
    RsaPkcs1(DigestAlgorithm),
    /// RSASSA-PSS with the parameters the identifier carried.
    RsaPss(PssParameters),
    /// ECDSA over the named digest.
    Ecdsa(DigestAlgorithm),
    /// DSA over the named digest.
    Dsa(DigestAlgorithm),
}

/// Reads an `AlgorithmIdentifier` and returns the scheme it names, or [`None`]
/// for one this crate cannot check.
pub(crate) fn parse_algorithm(encoding: &[u8]) -> Option<CertificateSignature> {
    let mut outer = Der::new(encoding);
    let mut algorithm = outer.read_sequence().ok()?;
    outer.finish().ok()?;
    let identifier = algorithm.read(tag::OID).ok()?;
    let scheme = match identifier {
        oid::RSA_SHA1 => CertificateSignature::RsaPkcs1(DigestAlgorithm::Sha1),
        oid::RSA_SHA224 => CertificateSignature::RsaPkcs1(DigestAlgorithm::Sha224),
        oid::RSA_SHA256 => CertificateSignature::RsaPkcs1(DigestAlgorithm::Sha256),
        oid::RSA_SHA384 => CertificateSignature::RsaPkcs1(DigestAlgorithm::Sha384),
        oid::RSA_SHA512 => CertificateSignature::RsaPkcs1(DigestAlgorithm::Sha512),
        oid::DSA_SHA1 => CertificateSignature::Dsa(DigestAlgorithm::Sha1),
        oid::DSA_SHA224 => CertificateSignature::Dsa(DigestAlgorithm::Sha224),
        oid::DSA_SHA256 => CertificateSignature::Dsa(DigestAlgorithm::Sha256),
        oid::ECDSA_SHA1 => CertificateSignature::Ecdsa(DigestAlgorithm::Sha1),
        oid::ECDSA_SHA224 => CertificateSignature::Ecdsa(DigestAlgorithm::Sha224),
        oid::ECDSA_SHA256 => CertificateSignature::Ecdsa(DigestAlgorithm::Sha256),
        oid::ECDSA_SHA384 => CertificateSignature::Ecdsa(DigestAlgorithm::Sha384),
        oid::ECDSA_SHA512 => CertificateSignature::Ecdsa(DigestAlgorithm::Sha512),
        oid::RSASSA_PSS => CertificateSignature::RsaPss(read_pss_parameters(&mut algorithm)?),
        _ => return None,
    };
    // The parameters of an RSA identifier are NULL (RFC 4055 §2.1) and those of
    // an ECDSA or DSA one are absent (RFC 5758 §3.2, RFC 3279 §2.2.2). Neither
    // says anything the signature check needs — the key carries every domain
    // parameter there is — so one field of whatever shape is skipped and
    // anything beyond it is a malformed identifier. PSS has already consumed
    // its own, which it does need.
    if !matches!(scheme, CertificateSignature::RsaPss(_)) {
        if !algorithm.is_empty() {
            algorithm.skip_any().ok()?;
        }
        algorithm.finish().ok()?;
    }
    Some(scheme)
}

/// Reads an `RSASSA-PSS-params`, whose four fields are all EXPLICIT and all
/// defaulted — to SHA-1, MGF1 over SHA-1, a 20-octet salt and the only trailer
/// field RFC 8017 defines.
fn read_pss_parameters(algorithm: &mut Der<'_>) -> Option<PssParameters> {
    let mut digest = DigestAlgorithm::Sha1;
    let mut mask_digest = DigestAlgorithm::Sha1;
    let mut salt_length = 20_usize;
    if let Some(contents) = algorithm.read_optional(tag::SEQUENCE).ok()? {
        let mut parameters = Der::new(contents);
        if let Some(wrapper) = parameters.read_optional(tag::CONTEXT_0).ok()? {
            digest = read_digest_algorithm(wrapper)?;
        }
        if let Some(wrapper) = parameters.read_optional(tag::CONTEXT_1).ok()? {
            mask_digest = read_mask_generation_algorithm(wrapper)?;
        }
        if let Some(wrapper) = parameters.read_optional(tag::CONTEXT_2).ok()? {
            let mut wrapper = Der::new(wrapper);
            let magnitude = wrapper.read_unsigned_integer().ok()?;
            wrapper.finish().ok()?;
            salt_length = 0;
            for byte in magnitude {
                salt_length = salt_length
                    .checked_mul(256)?
                    .checked_add(usize::from(*byte))?;
            }
        }
        if let Some(wrapper) = parameters.read_optional(tag::CONTEXT_3).ok()? {
            let mut wrapper = Der::new(wrapper);
            let trailer = wrapper.read_unsigned_integer().ok()?;
            wrapper.finish().ok()?;
            // RFC 8017 defines exactly one trailer field, and a certificate
            // asking for another cannot be verified by anyone.
            if trailer != [1] {
                return None;
            }
        }
        parameters.finish().ok()?;
    }
    algorithm.finish().ok()?;
    Some(
        PssParameters::new(digest)
            .mask_generation_digest(mask_digest)
            .salt_length(salt_length),
    )
}

/// Reads a digest `AlgorithmIdentifier`, whose parameters are NULL or absent.
fn read_digest_algorithm(encoding: &[u8]) -> Option<DigestAlgorithm> {
    let mut outer = Der::new(encoding);
    let mut algorithm = outer.read_sequence().ok()?;
    outer.finish().ok()?;
    let identifier = algorithm.read(tag::OID).ok()?;
    algorithm.read_optional(tag::NULL).ok()?;
    algorithm.finish().ok()?;
    match identifier {
        oid::SHA1 => Some(DigestAlgorithm::Sha1),
        oid::SHA224 => Some(DigestAlgorithm::Sha224),
        oid::SHA256 => Some(DigestAlgorithm::Sha256),
        oid::SHA384 => Some(DigestAlgorithm::Sha384),
        oid::SHA512 => Some(DigestAlgorithm::Sha512),
        _ => None,
    }
}

/// Reads a mask generation `AlgorithmIdentifier`: `id-mgf1` with the digest it
/// is built on as its parameters.
fn read_mask_generation_algorithm(encoding: &[u8]) -> Option<DigestAlgorithm> {
    let mut outer = Der::new(encoding);
    let mut algorithm = outer.read_sequence().ok()?;
    outer.finish().ok()?;
    if algorithm.read(tag::OID).ok()? != oid::MGF1 {
        return None;
    }
    let (_, inner, _) = algorithm.read_any().ok()?;
    algorithm.finish().ok()?;
    read_digest_algorithm(inner)
}

/// Whether `signature` over `message` checks out under `issuer_key`.
#[must_use]
pub(crate) fn verify(
    issuer_key: &VerificationKey,
    scheme: CertificateSignature,
    message: &[u8],
    signature: &[u8],
) -> bool {
    match (scheme, issuer_key) {
        (CertificateSignature::RsaPkcs1(digest), VerificationKey::Rsa(key)) => {
            pkcs1::verify(key.n(), key.e(), digest, message, signature)
        }
        (CertificateSignature::RsaPss(parameters), VerificationKey::Rsa(key)) => {
            pss::verify(key.n(), key.e(), parameters, message, signature)
        }
        (CertificateSignature::Ecdsa(digest), VerificationKey::Ec(key)) => {
            // Both supported curves have an order exactly as wide as the field,
            // which is what `Curve::encode_scalar` — and so the crate's ECDSA
            // verifier — expects each half of the signature to be.
            let width = key.parameters().field_bytes();
            decode_asn1_signature(signature, width).is_some_and(|raw| {
                ecdsa::verify(key.parameters(), key.point(), digest, message, &raw)
            })
        }
        (CertificateSignature::Dsa(digest), VerificationKey::Dsa(key)) => {
            let width = key.subgroup_bits().div_ceil(8);
            decode_asn1_signature(signature, width)
                .is_some_and(|raw| dsa::verify(key.parameters(), digest, message, &raw, width))
        }
        _ => false,
    }
}

/// Turns the ASN.1 `SEQUENCE { r INTEGER, s INTEGER }` of an X.509 signature
/// into the fixed-width `r ‖ s` the crate's verifiers take.
///
/// Returns [`None`] for anything that is not exactly that structure, including
/// the fixed-width form itself: a certificate carrying an XML Signature style
/// signature value is malformed, and accepting it would mean accepting two
/// encodings of the same signature.
fn decode_asn1_signature(signature: &[u8], width: usize) -> Option<Vec<u8>> {
    let mut outer = Der::new(signature);
    let mut sequence = outer.read_sequence().ok()?;
    outer.finish().ok()?;
    let r = sequence.read_unsigned_integer().ok()?;
    let s = sequence.read_unsigned_integer().ok()?;
    sequence.finish().ok()?;
    let mut raw = Vec::with_capacity(2 * width);
    for component in [r, s] {
        if component.len() > width {
            return None;
        }
        raw.resize(raw.len() + width - component.len(), 0);
        raw.extend_from_slice(component);
    }
    Some(raw)
}

/// Whether `certificate` was signed by the holder of `issuer_key`.
///
/// # Errors
///
/// [`CertificateError::UnsupportedSignatureAlgorithm`] when the certificate
/// names a scheme this crate cannot check — which is not the same answer as
/// "the signature is wrong", and must not be reported as one.
pub(crate) fn is_signed_by(
    certificate: &Certificate,
    issuer_key: &VerificationKey,
) -> Result<bool, CertificateError> {
    let scheme = parse_algorithm(certificate.signature_algorithm_encoding()).ok_or_else(|| {
        CertificateError::UnsupportedSignatureAlgorithm {
            subject: certificate.subject().to_string(),
            oid: certificate.signature_algorithm(),
        }
    })?;
    Ok(verify(
        issuer_key,
        scheme,
        certificate.signed_octets(),
        certificate.signature(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkey;
    use crate::x509::testcert::{self, CertificateSpec, Signer};

    /// Builds an `AlgorithmIdentifier` from an identifier and its parameters.
    fn algorithm_identifier(identifier: &[u8], parameters: &[u8]) -> Vec<u8> {
        let mut contents = testcert::tlv(tag::OID, identifier);
        contents.extend_from_slice(parameters);
        testcert::tlv(tag::SEQUENCE, &contents)
    }

    #[test]
    fn the_signature_algorithm_identifiers_are_recognised() {
        assert_eq!(
            parse_algorithm(&algorithm_identifier(
                oid::RSA_SHA256,
                &testcert::tlv(tag::NULL, &[])
            )),
            Some(CertificateSignature::RsaPkcs1(DigestAlgorithm::Sha256))
        );
        // An ECDSA identifier carries no parameters at all.
        assert_eq!(
            parse_algorithm(&algorithm_identifier(oid::ECDSA_SHA384, &[])),
            Some(CertificateSignature::Ecdsa(DigestAlgorithm::Sha384))
        );
        assert_eq!(
            parse_algorithm(&algorithm_identifier(oid::DSA_SHA1, &[])),
            Some(CertificateSignature::Dsa(DigestAlgorithm::Sha1))
        );
        // rsaEncryption is a key algorithm, not a signature algorithm.
        assert_eq!(
            parse_algorithm(&algorithm_identifier(
                &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x01],
                &testcert::tlv(tag::NULL, &[])
            )),
            None
        );
        // One parameter field is skipped whatever shape it has; a second is not
        // a field of an `AlgorithmIdentifier` at all.
        assert_eq!(
            parse_algorithm(&algorithm_identifier(oid::ECDSA_SHA256, &[0x05, 0x00])),
            Some(CertificateSignature::Ecdsa(DigestAlgorithm::Sha256))
        );
        assert_eq!(
            parse_algorithm(&algorithm_identifier(
                oid::ECDSA_SHA256,
                &[0x05, 0x00, 0x05, 0x00]
            )),
            None
        );
    }

    /// The defaults of `RSASSA-PSS-params` are SHA-1 throughout, and every
    /// field that is present overrides one of them.
    #[test]
    fn the_pss_parameters_are_read_with_their_defaults() {
        assert_eq!(
            parse_algorithm(&algorithm_identifier(oid::RSASSA_PSS, &[])),
            Some(CertificateSignature::RsaPss(
                PssParameters::new(DigestAlgorithm::Sha1).salt_length(20)
            ))
        );

        let hash = testcert::tlv(
            tag::CONTEXT_0,
            &algorithm_identifier(oid::SHA256, &testcert::tlv(tag::NULL, &[])),
        );
        let mask = testcert::tlv(
            tag::CONTEXT_1,
            &algorithm_identifier(
                oid::MGF1,
                &algorithm_identifier(oid::SHA256, &testcert::tlv(tag::NULL, &[])),
            ),
        );
        let salt = testcert::tlv(tag::CONTEXT_2, &testcert::integer(&[32]));
        let mut parameters = hash;
        parameters.extend(mask);
        parameters.extend(salt);
        assert_eq!(
            parse_algorithm(&algorithm_identifier(
                oid::RSASSA_PSS,
                &testcert::tlv(tag::SEQUENCE, &parameters)
            )),
            Some(CertificateSignature::RsaPss(PssParameters::new(
                DigestAlgorithm::Sha256
            )))
        );

        // A trailer field other than 1 is one RFC 8017 does not define.
        let trailer = testcert::tlv(tag::CONTEXT_3, &testcert::integer(&[2]));
        assert_eq!(
            parse_algorithm(&algorithm_identifier(
                oid::RSASSA_PSS,
                &testcert::tlv(tag::SEQUENCE, &trailer)
            )),
            None
        );
    }

    /// The whole point of this module: a certificate signature is ASN.1, and
    /// the fixed-width form XML Signature uses must not verify in its place.
    #[test]
    fn an_ecdsa_certificate_signature_is_asn1_and_not_p1363()
    -> Result<(), Box<dyn std::error::Error>> {
        let key = Signer::ec(0x44);
        let spec = CertificateSpec::new("Encoding");
        let asn1 = Certificate::parse(&testcert::build(&spec, &key, "Encoding", &key))?;
        let issuer_key = asn1.public_key().ok_or("a P-256 key")?;
        assert!(is_signed_by(&asn1, issuer_key)?);

        let raw = Certificate::parse(&testcert::build(
            &spec.p1363_signature(true),
            &key,
            "Encoding",
            &key,
        ))?;
        assert!(
            !is_signed_by(&raw, raw.public_key().ok_or("a P-256 key")?)?,
            "a fixed-width r ‖ s is not an X.509 signature value"
        );
        Ok(())
    }

    #[test]
    fn an_rsa_certificate_signature_verifies() -> Result<(), Box<dyn std::error::Error>> {
        let certificate = Certificate::parse(&testkey::bytes(testkey::CERTIFICATE_BASE64))?;
        let key = certificate.public_key().ok_or("an RSA key")?;
        assert!(is_signed_by(&certificate, key)?);
        let broken = Certificate::parse(&testcert::break_signature(certificate.encoding()))?;
        assert!(!is_signed_by(&broken, key)?);
        Ok(())
    }

    /// A key of the wrong kind never verifies, whatever the scheme says.
    #[test]
    fn a_key_of_the_wrong_kind_is_refused() -> Result<(), Box<dyn std::error::Error>> {
        let certificate = Certificate::parse(&testkey::bytes(testkey::CERTIFICATE_BASE64))?;
        let elliptic = Signer::ec(0x55);
        let elliptic = Certificate::parse(&testcert::build(
            &CertificateSpec::new("Elliptic"),
            &elliptic,
            "Elliptic",
            &elliptic,
        ))?;
        let ec_key = elliptic.public_key().ok_or("a P-256 key")?;
        assert!(!is_signed_by(&certificate, ec_key)?);
        Ok(())
    }

    #[test]
    fn a_malformed_asn1_signature_is_refused() {
        // A `SEQUENCE` of two INTEGERs is the only shape accepted, and each of
        // them must be minimally encoded.
        let mut contents = testcert::integer(&[0x01]);
        contents.extend(testcert::integer(&[0x02]));
        let well_formed = testcert::tlv(tag::SEQUENCE, &contents);
        assert_eq!(
            decode_asn1_signature(&well_formed, 4),
            Some(vec![0, 0, 0, 1, 0, 0, 0, 2])
        );
        // A component wider than the curve.
        assert_eq!(decode_asn1_signature(&well_formed, 0), None);
        // A redundant leading octet, which DER forbids.
        let mut padded = testcert::tlv(tag::INTEGER, &[0x00, 0x01]);
        padded.extend(testcert::integer(&[0x02]));
        assert_eq!(
            decode_asn1_signature(&testcert::tlv(tag::SEQUENCE, &padded), 4),
            None
        );
        // A third component.
        let mut three = contents.clone();
        three.extend(testcert::integer(&[0x03]));
        assert_eq!(
            decode_asn1_signature(&testcert::tlv(tag::SEQUENCE, &three), 4),
            None
        );
        // Octets after the structure.
        let mut trailing = well_formed.clone();
        trailing.push(0x00);
        assert_eq!(decode_asn1_signature(&trailing, 4), None);
        // The fixed-width form.
        assert_eq!(decode_asn1_signature(&[0x01_u8; 64], 32), None);
    }

    #[test]
    fn an_unimplemented_scheme_is_reported_rather_than_failing_the_signature()
    -> Result<(), Box<dyn std::error::Error>> {
        let certificate = Certificate::parse(&testkey::bytes(testkey::CERTIFICATE_BASE64))?;
        let key = certificate.public_key().ok_or("an RSA key")?.clone();
        // md5WithRSAEncryption, 1.2.840.113549.1.1.4, in both copies of the
        // identifier so that they still agree with one another.
        let mut encoding = testkey::bytes(testkey::CERTIFICATE_BASE64);
        let needle = [0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0b];
        let positions: Vec<usize> = encoding
            .windows(needle.len())
            .enumerate()
            .filter(|(_, window)| *window == needle)
            .map(|(index, _)| index)
            .collect();
        for start in positions {
            if let Some(byte) = encoding.get_mut(start + needle.len() - 1) {
                *byte = 0x04;
            }
        }
        let certificate = Certificate::parse(&encoding)?;
        assert_eq!(certificate.signature_algorithm(), "1.2.840.113549.1.1.4");
        assert!(matches!(
            is_signed_by(&certificate, &key),
            Err(CertificateError::UnsupportedSignatureAlgorithm { .. })
        ));
        Ok(())
    }
}

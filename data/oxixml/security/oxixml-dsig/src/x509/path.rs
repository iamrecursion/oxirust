//! Certification path validation — [RFC 5280 §6.1](https://www.rfc-editor.org/rfc/rfc5280#section-6.1).
//!
//! A certificate says who signed it, not whether that signer should be
//! believed. Turning the first into the second means building a *path*: a
//! sequence of certificates that starts at the one in the document and ends at
//! a key the caller decided in advance to trust, with every link checked.
//!
//! # The algorithm
//!
//! Building runs from the leaf upwards. At each step the current certificate's
//! `issuer` is looked for among the trust anchors first and the other supplied
//! certificates second, matching by name, then by
//! `authorityKeyIdentifier`/`subjectKeyIdentifier` where both are present, and
//! finally — the only evidence that counts — by whether the candidate's key
//! actually verifies the signature. A certificate already on the path is never
//! reused, so a cycle terminates rather than spinning, and the path may not
//! grow past [`PathValidator::max_path_length`].
//!
//! Validation then runs back down, from the anchor to the leaf, applying
//! §6.1.3 and §6.1.4 to every certificate:
//!
//! * the validity period must contain the caller's validation time, both ends
//!   inclusive;
//! * the issuer name must be the subject name of the certificate above;
//! * no critical extension may be one this crate does not process;
//! * every certificate that issued another must say so — `basicConstraints`
//!   with `cA` true, and `keyCertSign` if it restricts its key usage at all —
//!   and must not be deeper than a `pathLenConstraint` above it allows;
//! * the leaf, if it restricts its key usage, must allow `digitalSignature` or
//!   `contentCommitment`, since signing is what it is being used for.
//!
//! The trust anchor itself is *not* validated. [RFC 5280 §6.1.1](https://www.rfc-editor.org/rfc/rfc5280#section-6.1.1)
//! defines an anchor as a name and a public key rather than as a certificate:
//! it is trusted because the caller said so, and re-deriving that from its own
//! self-signed certificate would add nothing. A certificate that is *both* an
//! anchor and the leaf — a pinned self-signed certificate — is a path of length
//! one and is checked in full, because there it is being relied on as a leaf.
//!
//! # What is deliberately not done
//!
//! These are gaps, not oversights, and none of them is silently assumed to
//! pass:
//!
//! * **Revocation.** No CRL is fetched or consulted and no OCSP responder is
//!   asked. A revoked certificate whose dates are still good validates here.
//!   `<ds:X509CRL>` is parsed into [`X509Data`](crate::X509Data) and left to
//!   the caller.
//! * **Name constraints** (§6.1.4 g). A `nameConstraints` extension is not
//!   processed — and, being critical in every real certificate that has one,
//!   makes the certificate unusable rather than under-checked.
//! * **Certificate policies** (§6.1.4 a–f), policy mapping, policy constraints
//!   and `inhibitAnyPolicy`. Same treatment: not processed, and refused when
//!   marked critical.
//! * **Fetching anything.** No `authorityInfoAccess` is followed; every
//!   certificate that takes part must be supplied by the caller or carried in
//!   the `<ds:KeyInfo>`. This crate never issues a network request on behalf of
//!   a document it is checking.

use crate::key::VerificationKey;
use crate::x509::certificate::Certificate;
use crate::x509::error::CertificateError;
use crate::x509::signature;
use crate::x509::time::Timestamp;

/// Builds and validates certification paths.
///
/// # Example
///
/// ```
/// use oxixml_dsig::{Certificate, PathValidator, Timestamp};
///
/// # const DER_BASE64: &str = "MIIBpDCCAQ2gAwIBAgIFASNFZ4kwDQYJKoZIhvcNAQELBQAwFjEUMBIGA1UEAwwLT3hpWE1MIFRlc3QwHhcNMjYwMTAxMDAwMDAwWhcNMzYwMTAxMDAwMDAwWjAWMRQwEgYDVQQDDAtPeGlYTUwgVGVzdDCBnzANBgkqhkiG9w0BAQEFAAOBjQAwgYkCgYEAjkcoCslxEus5u8QIPMAYjDIpNgX12fbWalSPmhbKjlNeR5ukRjB0xBJqYvncXrPl7lHzDemgFsdEbJCvTgF1imtaMkc6yfaHkJ4+RgzPR2SBU4KEaVq69fNR5Dco0vl/VUZD54drPqBBvXhiqwpAK3ZLHcPSLxzRQi1a20l+tlUCAwEAATANBgkqhkiG9w0BAQsFAAOBgQAbgJ1S/Y0svcUZpowjNkA5BxrXfJNopeDC6Rxz77AHzTfqnwkVWkjAYxY8mfbeCQkEucKK2FxB8bv1WzDE/rlW5h8jtyxmvYA3d/imKCjC7SPcibgHh55nIkmwA9oXpLL5C4rM0fcApbudoveWWwRHWAlxDiusBrr8vbzaDOByHA==";
/// let der = oxixml_text::base64::base64_decode(DER_BASE64)?;
/// let certificate = Certificate::parse(&der)?;
///
/// // A self-signed certificate that is itself the anchor: a path of length one.
/// let chain = [certificate.clone()];
/// let anchors = [certificate];
/// let validator = PathValidator::new(Timestamp::from_unix_seconds(1_800_000_000));
/// let path = validator.validate(&chain, &anchors)?;
/// assert_eq!(path.len(), 1);
/// assert_eq!(path.leaf().subject().to_string(), "CN=OxiXML Test");
///
/// // …and outside its validity period it is refused, not merely noted.
/// let too_early = PathValidator::new(Timestamp::from_unix_seconds(0));
/// assert!(too_early.validate(&chain, &anchors).is_err());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PathValidator {
    at: Timestamp,
    max_path_length: usize,
    max_certificates: usize,
}

impl PathValidator {
    /// How many certificates a path may hold before
    /// [`validate`](PathValidator::validate) gives up, counting the leaf and
    /// every intermediate but not the anchor.
    ///
    /// Real paths are two or three certificates deep and the longest published
    /// hierarchies are five; eight leaves room for every one of them.
    ///
    /// Together with [`DEFAULT_MAX_CERTIFICATES`](PathValidator::DEFAULT_MAX_CERTIFICATES)
    /// this is what bounds the work a hostile `<ds:X509Data>` can ask for.
    /// Building a path tries each candidate issuer in turn and checks a
    /// signature against every one whose name matches, so a document carrying
    /// `n` certificates that all claim the same subject name would cost
    /// `max_path_length × n` public-key operations — and each of those is
    /// already permitted to be as expensive as
    /// [`MAXIMUM_RSA_MODULUS_BITS`](crate::MAXIMUM_RSA_MODULUS_BITS) allows.
    /// Bounding `n` as well turns that product into a constant.
    pub const DEFAULT_MAX_PATH_LENGTH: usize = 8;

    /// How many certificates may be offered to build a path from.
    ///
    /// A `<ds:X509Data>` holds a chain — two or three certificates, occasionally
    /// a few more while a certificate authority rolls a key over. Sixteen is
    /// generous for that and small enough that the signature checks path
    /// building can be made to perform stay bounded by a constant rather than
    /// by the size of an attacker-supplied document; see
    /// [`DEFAULT_MAX_PATH_LENGTH`](PathValidator::DEFAULT_MAX_PATH_LENGTH).
    ///
    /// The limit applies to the certificates *offered by the document*. The
    /// trust anchors are not counted: they are the caller's own configuration,
    /// and a store holding every public root there is is a legitimate one.
    pub const DEFAULT_MAX_CERTIFICATES: usize = 16;

    /// A validator that checks validity periods against `at`.
    ///
    /// The instant is a parameter rather than a clock reading because a library
    /// that reads the clock cannot be tested, cannot answer "was this valid
    /// when it was signed?", and gives two callers in the same request
    /// different answers. Supplying it is the caller's job.
    #[must_use]
    pub const fn new(at: Timestamp) -> Self {
        Self {
            at,
            max_path_length: Self::DEFAULT_MAX_PATH_LENGTH,
            max_certificates: Self::DEFAULT_MAX_CERTIFICATES,
        }
    }

    /// Sets the ceiling on the path length —
    /// see [`DEFAULT_MAX_PATH_LENGTH`](PathValidator::DEFAULT_MAX_PATH_LENGTH).
    #[must_use]
    pub const fn max_path_length(mut self, certificates: usize) -> Self {
        self.max_path_length = certificates;
        self
    }

    /// Sets the ceiling on how many certificates may be offered —
    /// see [`DEFAULT_MAX_CERTIFICATES`](PathValidator::DEFAULT_MAX_CERTIFICATES).
    #[must_use]
    pub const fn max_certificates(mut self, certificates: usize) -> Self {
        self.max_certificates = certificates;
        self
    }

    /// The ceiling [`max_certificates`](PathValidator::max_certificates) set,
    /// so that a caller assembling the certificates can refuse an oversized
    /// input before paying to parse it.
    #[must_use]
    pub const fn certificate_limit(&self) -> usize {
        self.max_certificates
    }

    /// The time validity periods are checked against.
    #[must_use]
    pub const fn validation_time(&self) -> Timestamp {
        self.at
    }

    /// Refuses an oversized set of candidate certificates before any work is
    /// done on it.
    const fn check_size(&self, certificates: usize) -> Result<(), CertificateError> {
        if certificates > self.max_certificates {
            return Err(CertificateError::TooManyCertificates {
                certificates,
                limit: self.max_certificates,
            });
        }
        Ok(())
    }

    /// Validates a path from the end-entity certificate of `chain` to one of
    /// `anchors`.
    ///
    /// The end-entity certificate is chosen by
    /// [`Certificate::select_end_entity`], because
    /// [XML Signature §4.5.4](https://www.w3.org/TR/xmldsig-core1/#sec-X509Data)
    /// fixes no order for the certificates in a `<ds:X509Data>`.
    ///
    /// # Errors
    ///
    /// A [`CertificateError`] naming the rule that was broken and the
    /// certificate that broke it, or
    /// [`TooManyCertificates`](CertificateError::TooManyCertificates) when
    /// `chain` is longer than
    /// [`max_certificates`](PathValidator::max_certificates) allows. There is no
    /// "valid but untrusted" outcome: a path that does not reach an anchor is an
    /// error.
    pub fn validate(
        &self,
        chain: &[Certificate],
        anchors: &[Certificate],
    ) -> Result<CertificatePath, CertificateError> {
        // Before `select_end_entity`, whose search is quadratic in the number of
        // certificates it is handed.
        self.check_size(chain.len())?;
        let leaf = Certificate::select_end_entity(chain).ok_or(CertificateError::NoCertificates)?;
        self.validate_leaf(leaf, chain, anchors)
    }

    /// Validates a path from `leaf` to one of `anchors`, using `intermediates`
    /// for the certificates in between.
    ///
    /// `intermediates` may contain `leaf` itself — a certificate already on the
    /// path is never used again — so the whole of a `<ds:X509Data>` can be
    /// passed unchanged.
    ///
    /// # Errors
    ///
    /// As [`validate`](PathValidator::validate).
    pub fn validate_leaf(
        &self,
        leaf: &Certificate,
        intermediates: &[Certificate],
        anchors: &[Certificate],
    ) -> Result<CertificatePath, CertificateError> {
        self.check_size(intermediates.len())?;
        if anchors.is_empty() {
            return Err(CertificateError::NoTrustAnchors);
        }
        let public_key = leaf.public_key().cloned().ok_or_else(|| {
            CertificateError::UnsupportedKeyAlgorithm {
                subject: leaf.subject().to_string(),
            }
        })?;
        let (path, anchor) = self.build(leaf, intermediates, anchors)?;
        self.check(&path, &anchor)?;
        let mut certificates = path.into_iter();
        let leaf = certificates.next().unwrap_or_else(|| leaf.clone());
        Ok(CertificatePath {
            leaf,
            issuers: certificates.collect(),
            anchor,
            public_key,
        })
    }

    /// Builds the path, leaf first, and returns it with the anchor it reached.
    fn build(
        &self,
        leaf: &Certificate,
        intermediates: &[Certificate],
        anchors: &[Certificate],
    ) -> Result<(Vec<Certificate>, Certificate), CertificateError> {
        let mut path: Vec<Certificate> = Vec::new();
        let mut current = leaf.clone();
        loop {
            if path.len() >= self.max_path_length {
                return Err(CertificateError::PathTooLong {
                    limit: self.max_path_length,
                });
            }
            path.push(current.clone());

            let from_anchors = find_issuer(&current, anchors, &[])?;
            if let Some(anchor) = from_anchors.issuer {
                return Ok((path, anchor.clone()));
            }
            let from_chain = find_issuer(&current, intermediates, &path)?;
            if let Some(issuer) = from_chain.issuer {
                current = issuer.clone();
                continue;
            }
            let subject = current.subject().to_string();
            return Err(match from_anchors.name_match.or(from_chain.name_match) {
                Some(issuer) => CertificateError::BadSignature { subject, issuer },
                None if current.is_self_issued() => CertificateError::UntrustedRoot { subject },
                None => CertificateError::IssuerNotFound {
                    subject,
                    issuer: current.issuer().to_string(),
                },
            });
        }
    }

    /// Applies §6.1.3 and §6.1.4 to `path`, working down from `anchor`.
    fn check(&self, path: &[Certificate], anchor: &Certificate) -> Result<(), CertificateError> {
        let length = path.len();
        // RFC 5280 §6.1.2 (k): the counter starts at the path length, so on its
        // own it never bites; it is `pathLenConstraint` that lowers it.
        let mut max_path_length = length;
        let mut working_issuer = anchor.subject();
        for (index, certificate) in path.iter().rev().enumerate() {
            let subject = certificate.subject().to_string();
            if let Some(identifier) = certificate.unhandled_critical_extensions().next() {
                return Err(CertificateError::UnknownCriticalExtension {
                    subject,
                    oid: identifier.to_owned(),
                });
            }
            if self.at < certificate.not_before() {
                return Err(CertificateError::NotYetValid {
                    subject,
                    not_before: certificate.not_before(),
                    at: self.at,
                });
            }
            if self.at > certificate.not_after() {
                return Err(CertificateError::Expired {
                    subject,
                    not_after: certificate.not_after(),
                    at: self.at,
                });
            }
            if !certificate.issuer().matches(working_issuer) {
                return Err(CertificateError::NameMismatch {
                    subject,
                    issuer: certificate.issuer().to_string(),
                    expected: working_issuer.to_string(),
                });
            }

            if index + 1 == length {
                // The leaf. Its key is about to be used to check a signature,
                // so a keyUsage extension that forbids that forbids the whole
                // operation.
                if certificate
                    .key_usage()
                    .is_some_and(|usage| !usage.digital_signature() && !usage.content_commitment())
                {
                    return Err(CertificateError::KeyUsageForbidden {
                        subject,
                        usage: "digitalSignature",
                    });
                }
                continue;
            }

            if !certificate.is_certificate_authority() {
                return Err(CertificateError::NotACertificateAuthority { subject });
            }
            if certificate
                .key_usage()
                .is_some_and(|usage| !usage.key_certificate_sign())
            {
                return Err(CertificateError::KeyUsageForbidden {
                    subject,
                    usage: "keyCertSign",
                });
            }
            // §6.1.4 (l) and (m). A self-issued certificate does not lengthen
            // the path: it is the same authority re-certifying itself, which is
            // how a key rollover is expressed.
            if !certificate.is_self_issued() {
                if max_path_length == 0 {
                    return Err(CertificateError::PathLengthConstraintExceeded { subject });
                }
                max_path_length -= 1;
            }
            if let Some(limit) = certificate.path_length_constraint() {
                max_path_length = max_path_length.min(limit);
            }
            working_issuer = certificate.subject();
        }
        Ok(())
    }
}

/// What looking for the issuer of a certificate turned up.
struct IssuerSearch<'a> {
    /// The candidate whose key verifies the signature.
    issuer: Option<&'a Certificate>,
    /// The subject of a candidate that matched by name but not by signature,
    /// which is the difference between "no issuer" and "a forged certificate".
    name_match: Option<String>,
}

/// Looks for the issuer of `child` among `candidates`, skipping anything
/// already on `path`.
fn find_issuer<'a>(
    child: &Certificate,
    candidates: &'a [Certificate],
    path: &[Certificate],
) -> Result<IssuerSearch<'a>, CertificateError> {
    let mut name_match = None;
    for candidate in candidates {
        if path.contains(candidate) {
            continue;
        }
        if !candidate.subject().matches(child.issuer()) || !child.is_plausible_issuer(candidate) {
            continue;
        }
        let Some(key) = candidate.public_key() else {
            continue;
        };
        if signature::is_signed_by(child, key)? {
            return Ok(IssuerSearch {
                issuer: Some(candidate),
                name_match,
            });
        }
        name_match.get_or_insert_with(|| candidate.subject().to_string());
    }
    Ok(IssuerSearch {
        issuer: None,
        name_match,
    })
}

/// A validated certification path: the leaf, the certificates between it and
/// the trust anchor, and the anchor itself.
///
/// Holding one of these is the evidence that the leaf's key may be used — every
/// check [`PathValidator`] performs has passed, at the validation time it was
/// given.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CertificatePath {
    leaf: Certificate,
    issuers: Vec<Certificate>,
    anchor: Certificate,
    public_key: VerificationKey,
}

impl CertificatePath {
    /// The end-entity certificate the path was built from.
    #[must_use]
    pub const fn leaf(&self) -> &Certificate {
        &self.leaf
    }

    /// The trust anchor the path terminates at.
    #[must_use]
    pub const fn anchor(&self) -> &Certificate {
        &self.anchor
    }

    /// The validated public key of the leaf — the one an XML signature made
    /// with this certificate must be checked against.
    #[must_use]
    pub const fn public_key(&self) -> &VerificationKey {
        &self.public_key
    }

    /// Every certificate on the path, leaf first, anchor excluded.
    pub fn certificates(&self) -> impl Iterator<Item = &Certificate> {
        std::iter::once(&self.leaf).chain(self.issuers.iter())
    }

    /// How many certificates the path holds, anchor excluded. Never zero.
    #[must_use]
    pub fn len(&self) -> usize {
        self.issuers.len() + 1
    }

    /// Always `false`: a path always holds at least the leaf.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkey;
    use crate::x509::testcert::{self, CertificateSpec, Signer};

    /// 2030-01-01T00:00:00Z, inside the validity of every fixture certificate.
    const AT: Timestamp = Timestamp::from_unix_seconds(1_893_456_000);

    fn validator() -> PathValidator {
        PathValidator::new(AT)
    }

    #[test]
    fn a_three_certificate_path_validates() -> Result<(), Box<dyn std::error::Error>> {
        let chain = testcert::chain()?;
        let path = validator().validate(
            &[chain.leaf.clone(), chain.intermediate.clone()],
            std::slice::from_ref(&chain.root),
        )?;
        assert_eq!(path.len(), 2);
        assert_eq!(path.leaf(), &chain.leaf);
        assert_eq!(path.anchor(), &chain.root);
        assert_eq!(
            path.certificates()
                .map(|certificate| certificate.subject().to_string())
                .collect::<Vec<_>>(),
            [
                format!("CN={}", testcert::LEAF_NAME),
                format!("CN={}", testcert::INTERMEDIATE_NAME),
            ]
        );
        assert_eq!(Some(path.public_key()), chain.leaf.public_key());
        // The root may also be carried in the chain, which is what a signer
        // that ships the whole hierarchy does.
        assert!(
            validator()
                .validate(
                    &[
                        chain.root.clone(),
                        chain.leaf.clone(),
                        chain.intermediate.clone()
                    ],
                    std::slice::from_ref(&chain.root),
                )
                .is_ok()
        );
        Ok(())
    }

    /// A pinned self-signed certificate is a path of length one — and is still
    /// checked, unlike the anchor of a longer path.
    #[test]
    fn a_self_signed_certificate_is_its_own_path() -> Result<(), Box<dyn std::error::Error>> {
        let certificate = Certificate::parse(&testkey::bytes(testkey::CERTIFICATE_BASE64))?;
        let anchors = [certificate.clone()];
        let path = validator().validate(std::slice::from_ref(&certificate), &anchors)?;
        assert_eq!(path.len(), 1);
        assert_eq!(path.leaf(), &certificate);

        // The pinned certificate is valid from 2026 to 2036, and validating in
        // 2020 must fail rather than defer to the pin.
        let error = PathValidator::new(Timestamp::from_unix_seconds(1_577_836_800))
            .validate(std::slice::from_ref(&certificate), &anchors)
            .expect_err("not yet valid in 2020");
        assert!(
            matches!(error, CertificateError::NotYetValid { .. }),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn an_expired_certificate_is_refused() -> Result<(), Box<dyn std::error::Error>> {
        let root_key = Signer::ec(testcert::ROOT_SEED);
        let intermediate_key = Signer::ec(testcert::INTERMEDIATE_SEED);
        let leaf_key = Signer::ec(testcert::LEAF_SEED);
        let root = Certificate::parse(&testcert::build(
            &testcert::root_spec(),
            &root_key,
            testcert::ROOT_NAME,
            &root_key,
        ))?;
        let intermediate = Certificate::parse(&testcert::build(
            &testcert::intermediate_spec().validity(
                Timestamp::from_unix_seconds(1_577_836_800),
                // 2025-01-01, five years before the validation time.
                Timestamp::from_unix_seconds(1_735_689_600),
            ),
            &intermediate_key,
            testcert::ROOT_NAME,
            &root_key,
        ))?;
        let leaf = Certificate::parse(&testcert::build(
            &testcert::leaf_spec(),
            &leaf_key,
            testcert::INTERMEDIATE_NAME,
            &intermediate_key,
        ))?;

        let error = validator()
            .validate(&[leaf, intermediate], &[root])
            .expect_err("the intermediate expired in 2025");
        let CertificateError::Expired {
            subject, not_after, ..
        } = &error
        else {
            panic!("{error}");
        };
        assert_eq!(subject, &format!("CN={}", testcert::INTERMEDIATE_NAME));
        assert_eq!(not_after.to_string(), "2025-01-01T00:00:00Z");
        Ok(())
    }

    /// A certificate whose signature has been tampered with is not merely
    /// "unfindable": the error has to say the signature is wrong, because that
    /// is the difference between a misconfiguration and an attack.
    #[test]
    fn a_broken_intermediate_signature_is_refused() -> Result<(), Box<dyn std::error::Error>> {
        let chain = testcert::chain()?;
        let broken = Certificate::parse(&testcert::break_signature(chain.intermediate.encoding()))?;
        let error = validator()
            .validate(
                &[chain.leaf.clone(), broken],
                std::slice::from_ref(&chain.root),
            )
            .expect_err("the intermediate is no longer signed by the root");
        assert!(
            matches!(error, CertificateError::BadSignature { .. }),
            "{error}"
        );
        assert!(
            error.to_string().contains(testcert::INTERMEDIATE_NAME),
            "{error}"
        );
        Ok(())
    }

    /// The encoding difference between X.509 and XML Signature, at the level
    /// the validator sees it.
    #[test]
    fn a_fixed_width_signature_in_a_certificate_is_refused()
    -> Result<(), Box<dyn std::error::Error>> {
        let root_key = Signer::ec(testcert::ROOT_SEED);
        let leaf_key = Signer::ec(testcert::LEAF_SEED);
        let root = Certificate::parse(&testcert::build(
            &testcert::root_spec(),
            &root_key,
            testcert::ROOT_NAME,
            &root_key,
        ))?;
        let leaf = Certificate::parse(&testcert::build(
            &testcert::leaf_spec().p1363_signature(true),
            &leaf_key,
            testcert::ROOT_NAME,
            &root_key,
        ))?;
        let error = validator()
            .validate(&[leaf], &[root])
            .expect_err("an X.509 signature value is ASN.1, never r ‖ s");
        assert!(
            matches!(error, CertificateError::BadSignature { .. }),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn an_untrusted_root_is_refused() -> Result<(), Box<dyn std::error::Error>> {
        let chain = testcert::chain()?;
        let other_key = Signer::ec(0x77);
        let other_root = Certificate::parse(&testcert::build(
            &CertificateSpec::new("Somebody Else").certificate_authority(None),
            &other_key,
            "Somebody Else",
            &other_key,
        ))?;

        // The chain is internally consistent, and its root is simply not one of
        // the anchors.
        let error = validator()
            .validate(
                &[
                    chain.leaf.clone(),
                    chain.intermediate.clone(),
                    chain.root.clone(),
                ],
                std::slice::from_ref(&other_root),
            )
            .expect_err("the chain's own root is not an anchor");
        assert!(
            matches!(error, CertificateError::UntrustedRoot { .. }),
            "{error}"
        );

        // Without the root in the bag there is not even a candidate issuer.
        let error = validator()
            .validate(
                &[chain.leaf.clone(), chain.intermediate.clone()],
                &[other_root],
            )
            .expect_err("no issuer for the intermediate");
        assert!(
            matches!(error, CertificateError::IssuerNotFound { .. }),
            "{error}"
        );

        // And with no anchors at all, nothing can be trusted.
        assert_eq!(
            validator().validate(&[chain.leaf], &[]),
            Err(CertificateError::NoTrustAnchors)
        );
        Ok(())
    }

    #[test]
    fn an_unknown_critical_extension_makes_a_certificate_unusable()
    -> Result<(), Box<dyn std::error::Error>> {
        let root_key = Signer::ec(testcert::ROOT_SEED);
        let leaf_key = Signer::ec(testcert::LEAF_SEED);
        let root = Certificate::parse(&testcert::build(
            &testcert::root_spec(),
            &root_key,
            testcert::ROOT_NAME,
            &root_key,
        ))?;
        // 2.5.29.32, certificatePolicies: recognised by RFC 5280, not processed
        // here.
        let leaf = Certificate::parse(&testcert::build(
            &testcert::leaf_spec().critical_extension(vec![0x55, 0x1d, 0x20], vec![0x30, 0x00]),
            &leaf_key,
            testcert::ROOT_NAME,
            &root_key,
        ))?;
        let error = validator()
            .validate(&[leaf], &[root])
            .expect_err("a critical extension nobody processes is fatal");
        let CertificateError::UnknownCriticalExtension { oid, .. } = &error else {
            panic!("{error}");
        };
        assert_eq!(oid, "2.5.29.32");
        Ok(())
    }

    /// Both length limits: the validator's own ceiling, and a
    /// `pathLenConstraint` set by a certificate authority.
    #[test]
    fn a_path_that_is_too_long_is_refused() -> Result<(), Box<dyn std::error::Error>> {
        let chain = testcert::chain()?;
        let error = validator()
            .max_path_length(1)
            .validate(
                &[chain.leaf.clone(), chain.intermediate.clone()],
                std::slice::from_ref(&chain.root),
            )
            .expect_err("two certificates do not fit under a ceiling of one");
        assert_eq!(error, CertificateError::PathTooLong { limit: 1 });

        // `pathLenConstraint` counts the certificate authorities that may
        // follow, not the certificates: a constraint of zero still permits an
        // end-entity certificate directly below, and only bites once a second
        // authority appears. So the case needs a four-deep hierarchy.
        let root_key = Signer::ec(testcert::ROOT_SEED);
        let first_key = Signer::ec(0x61);
        let second_key = Signer::ec(0x62);
        let leaf_key = Signer::ec(testcert::LEAF_SEED);
        let root = Certificate::parse(&testcert::build(
            &testcert::root_spec(),
            &root_key,
            testcert::ROOT_NAME,
            &root_key,
        ))?;
        let authority = |name: &str,
                         path_length: Option<usize>,
                         key: &Signer,
                         issuer: &str,
                         issuer_key: &Signer| {
            testcert::build(
                &CertificateSpec::new(name)
                    .serial(2)
                    .certificate_authority(path_length)
                    .key_usage(0b0000_0100),
                key,
                issuer,
                issuer_key,
            )
        };
        let permissive = Certificate::parse(&authority(
            "First Authority",
            Some(1),
            &first_key,
            testcert::ROOT_NAME,
            &root_key,
        ))?;
        let constrained = Certificate::parse(&authority(
            "First Authority",
            Some(0),
            &first_key,
            testcert::ROOT_NAME,
            &root_key,
        ))?;
        let second = Certificate::parse(&authority(
            "Second Authority",
            None,
            &second_key,
            "First Authority",
            &first_key,
        ))?;
        let leaf = Certificate::parse(&testcert::build(
            &testcert::leaf_spec(),
            &leaf_key,
            "Second Authority",
            &second_key,
        ))?;

        assert!(
            validator()
                .validate(
                    &[leaf.clone(), second.clone(), permissive],
                    std::slice::from_ref(&root)
                )
                .is_ok(),
            "a constraint of one permits the second authority"
        );
        let error = validator()
            .validate(&[leaf, second, constrained], &[root])
            .expect_err("a constraint of zero permits no further authority");
        assert!(
            matches!(error, CertificateError::PathLengthConstraintExceeded { .. }),
            "{error}"
        );
        Ok(())
    }

    /// Two ways a certificate can turn out not to be allowed to issue the one
    /// below it.
    #[test]
    fn an_intermediate_must_be_allowed_to_issue() -> Result<(), Box<dyn std::error::Error>> {
        let root_key = Signer::ec(testcert::ROOT_SEED);
        let intermediate_key = Signer::ec(testcert::INTERMEDIATE_SEED);
        let leaf_key = Signer::ec(testcert::LEAF_SEED);
        let root = Certificate::parse(&testcert::build(
            &testcert::root_spec(),
            &root_key,
            testcert::ROOT_NAME,
            &root_key,
        ))?;
        let leaf = |issuer: &str| {
            testcert::build(&testcert::leaf_spec(), &leaf_key, issuer, &intermediate_key)
        };

        // No basicConstraints at all: not a certificate authority.
        let plain = Certificate::parse(&testcert::build(
            &CertificateSpec::new(testcert::INTERMEDIATE_NAME).serial(2),
            &intermediate_key,
            testcert::ROOT_NAME,
            &root_key,
        ))?;
        let error = validator()
            .validate(
                &[
                    Certificate::parse(&leaf(testcert::INTERMEDIATE_NAME))?,
                    plain,
                ],
                std::slice::from_ref(&root),
            )
            .expect_err("a leaf certificate may not issue certificates");
        assert!(
            matches!(error, CertificateError::NotACertificateAuthority { .. }),
            "{error}"
        );

        // basicConstraints says it may, keyUsage says it may not.
        let restricted = Certificate::parse(&testcert::build(
            &CertificateSpec::new(testcert::INTERMEDIATE_NAME)
                .serial(2)
                .certificate_authority(None)
                .key_usage(0b1000_0000),
            &intermediate_key,
            testcert::ROOT_NAME,
            &root_key,
        ))?;
        let error = validator()
            .validate(
                &[
                    Certificate::parse(&leaf(testcert::INTERMEDIATE_NAME))?,
                    restricted,
                ],
                &[root],
            )
            .expect_err("keyCertSign is missing");
        assert!(
            matches!(
                error,
                CertificateError::KeyUsageForbidden {
                    usage: "keyCertSign",
                    ..
                }
            ),
            "{error}"
        );
        Ok(())
    }

    /// A leaf that says its key is only good for encipherment may not be used
    /// to check a signature.
    #[test]
    fn a_leaf_must_be_allowed_to_sign() -> Result<(), Box<dyn std::error::Error>> {
        let root_key = Signer::ec(testcert::ROOT_SEED);
        let leaf_key = Signer::ec(testcert::LEAF_SEED);
        let root = Certificate::parse(&testcert::build(
            &testcert::root_spec(),
            &root_key,
            testcert::ROOT_NAME,
            &root_key,
        ))?;
        let leaf = Certificate::parse(&testcert::build(
            // keyEncipherment alone.
            &testcert::leaf_spec().key_usage(0b0010_0000),
            &leaf_key,
            testcert::ROOT_NAME,
            &root_key,
        ))?;
        let error = validator()
            .validate(&[leaf], &[root])
            .expect_err("digitalSignature is missing");
        assert!(
            matches!(
                error,
                CertificateError::KeyUsageForbidden {
                    usage: "digitalSignature",
                    ..
                }
            ),
            "{error}"
        );
        Ok(())
    }

    /// The key identifiers only ever *narrow* the candidates: a certificate
    /// whose `authorityKeyIdentifier` names a different key is skipped, and one
    /// that names no key at all is still tried.
    #[test]
    fn key_identifiers_disambiguate_but_never_decide() -> Result<(), Box<dyn std::error::Error>> {
        let root_key = Signer::ec(testcert::ROOT_SEED);
        let leaf_key = Signer::ec(testcert::LEAF_SEED);
        let root = Certificate::parse(&testcert::build(
            &testcert::root_spec().subject_key_identifier(vec![0x01, 0x02]),
            &root_key,
            testcert::ROOT_NAME,
            &root_key,
        ))?;
        let agreeing = Certificate::parse(&testcert::build(
            &testcert::leaf_spec().authority_key_identifier(vec![0x01, 0x02]),
            &leaf_key,
            testcert::ROOT_NAME,
            &root_key,
        ))?;
        assert!(
            validator()
                .validate(&[agreeing], std::slice::from_ref(&root))
                .is_ok()
        );

        let disagreeing = Certificate::parse(&testcert::build(
            &testcert::leaf_spec().authority_key_identifier(vec![0xff, 0xfe]),
            &leaf_key,
            testcert::ROOT_NAME,
            &root_key,
        ))?;
        let error = validator()
            .validate(&[disagreeing], &[root])
            .expect_err("the identifier rules the only candidate out");
        assert!(
            matches!(error, CertificateError::IssuerNotFound { .. }),
            "{error}"
        );
        Ok(())
    }

    /// A cycle in the supplied certificates must terminate rather than spin:
    /// two certificates that name each other as issuer, neither reaching an
    /// anchor.
    #[test]
    fn a_cycle_terminates() -> Result<(), Box<dyn std::error::Error>> {
        let first_key = Signer::ec(0x51);
        let second_key = Signer::ec(0x52);
        let first = Certificate::parse(&testcert::build(
            &CertificateSpec::new("First").certificate_authority(None),
            &first_key,
            "Second",
            &second_key,
        ))?;
        let second = Certificate::parse(&testcert::build(
            &CertificateSpec::new("Second").certificate_authority(None),
            &second_key,
            "First",
            &first_key,
        ))?;
        let anchor_key = Signer::ec(0x53);
        let anchor = Certificate::parse(&testcert::build(
            &CertificateSpec::new("Anchor").certificate_authority(None),
            &anchor_key,
            "Anchor",
            &anchor_key,
        ))?;
        let error = validator()
            .validate_leaf(&first, &[first.clone(), second], &[anchor])
            .expect_err("the cycle reaches no anchor");
        assert!(
            matches!(error, CertificateError::IssuerNotFound { .. }),
            "{error}"
        );
        Ok(())
    }

    /// Path building tries every candidate whose subject name matches, so a
    /// document that repeats one name across a great many certificates would
    /// otherwise buy `max_path_length` signature checks per certificate it
    /// carries — each of which this crate already permits to be as expensive as
    /// a 16384-bit modulus. The number of candidates is therefore bounded, and
    /// an oversized set is refused *before* any signature is checked.
    #[test]
    fn an_oversized_certificate_set_is_refused_before_any_work()
    -> Result<(), Box<dyn std::error::Error>> {
        let chain = testcert::chain()?;
        let limit = PathValidator::DEFAULT_MAX_CERTIFICATES;
        let offered: Vec<Certificate> = std::iter::repeat_n(chain.intermediate.clone(), limit)
            .chain(std::iter::once(chain.leaf.clone()))
            .collect();
        assert_eq!(offered.len(), limit + 1);

        let error = validator()
            .validate(&offered, std::slice::from_ref(&chain.root))
            .expect_err("one more certificate than the limit allows");
        assert_eq!(
            error,
            CertificateError::TooManyCertificates {
                certificates: limit + 1,
                limit,
            }
        );
        // The same set through the other entry point, which a caller reaches
        // when it has picked the leaf itself.
        assert!(matches!(
            validator().validate_leaf(&chain.leaf, &offered, std::slice::from_ref(&chain.root)),
            Err(CertificateError::TooManyCertificates { .. })
        ));

        // Exactly at the limit, the set is considered — and here it validates,
        // duplicates and all.
        let at_limit = offered.get(1..).unwrap_or_default();
        assert_eq!(at_limit.len(), limit);
        assert!(
            validator()
                .validate(at_limit, std::slice::from_ref(&chain.root))
                .is_ok()
        );
        // …and the ceiling is configurable.
        assert!(
            validator()
                .max_certificates(limit + 1)
                .validate(&offered, std::slice::from_ref(&chain.root))
                .is_ok()
        );
        assert_eq!(validator().certificate_limit(), limit);
        Ok(())
    }

    /// A certificate whose key this crate cannot represent cannot be a leaf,
    /// and says so rather than failing somewhere less obvious.
    #[test]
    fn a_leaf_with_an_unusable_key_is_reported() -> Result<(), Box<dyn std::error::Error>> {
        // secp521r1, 1.3.132.0.35 — a curve XML Signature 1.1 does not make
        // REQUIRED and this crate does not implement.
        let mut algorithm = testcert::tlv(0x06, &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02, 0x01]);
        algorithm.extend(testcert::tlv(0x06, &[0x2b, 0x81, 0x04, 0x00, 0x23]));
        let mut contents = testcert::tlv(0x30, &algorithm);
        let mut point = vec![0x00, 0x04];
        point.extend_from_slice(&[0x01_u8; 132]);
        contents.extend(testcert::tlv(0x03, &point));
        let public_key_info = testcert::tlv(0x30, &contents);

        let root_key = Signer::ec(testcert::ROOT_SEED);
        let root = Certificate::parse(&testcert::build(
            &testcert::root_spec(),
            &root_key,
            testcert::ROOT_NAME,
            &root_key,
        ))?;
        let leaf = Certificate::parse(&testcert::build(
            &testcert::leaf_spec().public_key_info(public_key_info),
            &root_key,
            testcert::ROOT_NAME,
            &root_key,
        ))?;
        assert_eq!(leaf.public_key(), None);
        let error = validator()
            .validate(&[leaf], &[root])
            .expect_err("a P-521 key cannot be used");
        assert!(
            matches!(error, CertificateError::UnsupportedKeyAlgorithm { .. }),
            "{error}"
        );
        Ok(())
    }
}

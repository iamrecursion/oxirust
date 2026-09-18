//! Why a certification path was refused.
//!
//! Every variant names one rule of
//! [RFC 5280 §6.1](https://www.rfc-editor.org/rfc/rfc5280#section-6.1) and the
//! certificate that broke it. That specificity is the point: "the chain did not
//! validate" sends an operator looking everywhere at once, while "the
//! intermediate expired on 2024-03-01" sends them to the one thing that has to
//! change.
//!
//! There is no variant for "valid but not trusted". A path either terminates at
//! an anchor the caller listed or it does not, and when it does not the reason
//! is [`UntrustedRoot`](CertificateError::UntrustedRoot) or
//! [`IssuerNotFound`](CertificateError::IssuerNotFound) — never a success with a
//! caveat attached.

use std::error::Error;
use std::fmt;

use crate::error::DsigError;
use crate::x509::time::Timestamp;

/// Why a certificate could not be read, or a certification path could not be
/// validated.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CertificateError {
    /// The DER is not a well-formed X.509 `Certificate`, or one of its fields
    /// is not what the grammar requires.
    Malformed(String),

    /// The certificate certifies a key of an algorithm, size or curve this
    /// crate cannot use — see [the gap list](crate#gaps).
    UnsupportedKeyAlgorithm {
        /// The subject of the certificate.
        subject: String,
    },

    /// The certificate is signed with an algorithm this crate does not
    /// implement, so the signature cannot be checked and the certificate
    /// cannot be relied on.
    UnsupportedSignatureAlgorithm {
        /// The subject of the certificate.
        subject: String,
        /// The `signatureAlgorithm` identifier, as dotted decimal.
        oid: String,
    },

    /// There was no certificate to validate: the `<ds:X509Data>` carried none,
    /// or none of the ones it carried certifies a key of the kind the signature
    /// method needs.
    NoCertificates,

    /// No trust anchor was configured, so nothing could ever be trusted.
    NoTrustAnchors,

    /// No certificate in the chain or the anchor set has the subject name this
    /// certificate names as its issuer.
    IssuerNotFound {
        /// The subject of the certificate whose issuer is missing.
        subject: String,
        /// The issuer it names.
        issuer: String,
    },

    /// The path terminates at a self-issued certificate that is not one of the
    /// trust anchors. The chain is internally consistent and proves nothing.
    UntrustedRoot {
        /// The subject of the self-issued certificate the path ended at.
        subject: String,
    },

    /// A certificate's signature does not verify under the public key of the
    /// certificate that issued it.
    BadSignature {
        /// The subject of the certificate whose signature failed.
        subject: String,
        /// The issuer whose key it was checked against.
        issuer: String,
    },

    /// The validation time is before a certificate's `notBefore`.
    NotYetValid {
        /// The subject of the certificate.
        subject: String,
        /// When the certificate becomes valid.
        not_before: Timestamp,
        /// The time validation was performed at.
        at: Timestamp,
    },

    /// The validation time is after a certificate's `notAfter`.
    Expired {
        /// The subject of the certificate.
        subject: String,
        /// When the certificate stopped being valid.
        not_after: Timestamp,
        /// The time validation was performed at.
        at: Timestamp,
    },

    /// A certificate's issuer name is not the subject name of the certificate
    /// above it in the path.
    NameMismatch {
        /// The subject of the certificate.
        subject: String,
        /// The issuer it names.
        issuer: String,
        /// The subject of the certificate that actually issued it.
        expected: String,
    },

    /// A certificate carries an extension marked critical that this crate does
    /// not process. [RFC 5280 §6.1.3](https://www.rfc-editor.org/rfc/rfc5280#section-6.1.3)
    /// requires such a certificate to be rejected: "critical" is the issuer
    /// saying the certificate must not be used by anyone who cannot honour it.
    UnknownCriticalExtension {
        /// The subject of the certificate.
        subject: String,
        /// The extension identifier, as dotted decimal.
        oid: String,
    },

    /// A certificate in the middle of the path is not marked as a certificate
    /// authority by a `basicConstraints` extension, so it may not issue
    /// certificates.
    NotACertificateAuthority {
        /// The subject of the certificate.
        subject: String,
    },

    /// A certificate's `keyUsage` extension forbids the use the path needs it
    /// for.
    KeyUsageForbidden {
        /// The subject of the certificate.
        subject: String,
        /// The key usage that is missing.
        usage: &'static str,
    },

    /// The path is longer than the validator's ceiling — see
    /// [`PathValidator::max_path_length`](super::PathValidator::max_path_length).
    PathTooLong {
        /// The ceiling, in certificates.
        limit: usize,
    },

    /// A `pathLenConstraint` somewhere above forbids a path that reaches this
    /// far down.
    PathLengthConstraintExceeded {
        /// The subject of the certificate the constraint stopped at.
        subject: String,
    },

    /// More certificates were offered to build a path from than the validator
    /// will consider — see
    /// [`PathValidator::max_certificates`](super::PathValidator::max_certificates).
    TooManyCertificates {
        /// How many were offered.
        certificates: usize,
        /// The limit.
        limit: usize,
    },
}

impl fmt::Display for CertificateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Malformed(message) => write!(f, "malformed X.509 certificate: {message}"),
            Self::UnsupportedKeyAlgorithm { subject } => write!(
                f,
                "the certificate of {subject} certifies a key this crate cannot use"
            ),
            Self::UnsupportedSignatureAlgorithm { subject, oid } => write!(
                f,
                "the certificate of {subject} is signed with `{oid}`, which this crate does not \
                 implement"
            ),
            Self::NoCertificates => {
                f.write_str("there is no certificate to build a certification path from")
            }
            Self::NoTrustAnchors => {
                f.write_str("no trust anchor is configured, so no path can be trusted")
            }
            Self::IssuerNotFound { subject, issuer } => write!(
                f,
                "no certificate for the issuer of {subject}, which is {issuer}"
            ),
            Self::UntrustedRoot { subject } => write!(
                f,
                "the path ends at the self-issued certificate of {subject}, which is not a trust \
                 anchor"
            ),
            Self::BadSignature { subject, issuer } => write!(
                f,
                "the certificate of {subject} is not signed by the key of {issuer}"
            ),
            Self::NotYetValid {
                subject,
                not_before,
                at,
            } => write!(
                f,
                "the certificate of {subject} is not valid until {not_before}, and validation was \
                 performed at {at}"
            ),
            Self::Expired {
                subject,
                not_after,
                at,
            } => write!(
                f,
                "the certificate of {subject} expired at {not_after}, and validation was performed \
                 at {at}"
            ),
            Self::NameMismatch {
                subject,
                issuer,
                expected,
            } => write!(
                f,
                "the certificate of {subject} names {issuer} as its issuer, but it was issued by \
                 {expected}"
            ),
            Self::UnknownCriticalExtension { subject, oid } => write!(
                f,
                "the certificate of {subject} marks the extension `{oid}` critical, and this crate \
                 does not process it"
            ),
            Self::NotACertificateAuthority { subject } => write!(
                f,
                "the certificate of {subject} issued another certificate without a \
                 basicConstraints extension saying it may"
            ),
            Self::KeyUsageForbidden { subject, usage } => write!(
                f,
                "the keyUsage extension of the certificate of {subject} does not allow {usage}"
            ),
            Self::PathTooLong { limit } => write!(
                f,
                "the certification path is longer than the configured limit of {limit} \
                 certificates"
            ),
            Self::PathLengthConstraintExceeded { subject } => write!(
                f,
                "a pathLenConstraint in the certification path forbids issuing through the \
                 certificate of {subject}"
            ),
            Self::TooManyCertificates {
                certificates,
                limit,
            } => write!(
                f,
                "{certificates} certificates were offered to build a certification path from, and \
                 the configurable limit is {limit}"
            ),
        }
    }
}

impl Error for CertificateError {}

impl From<CertificateError> for DsigError {
    fn from(error: CertificateError) -> Self {
        Self::Certificate(Box::new(error))
    }
}

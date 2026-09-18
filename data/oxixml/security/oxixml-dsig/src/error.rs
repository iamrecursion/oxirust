//! The error type shared by every operation in this crate.
//!
//! XML Signature processing fails in two very different ways, and telling them
//! apart matters: a *processing* failure means the verifier could not carry the
//! computation out at all (a malformed `<ds:Signature>`, an algorithm nobody
//! implements, a reference that points nowhere), while a *cryptographic*
//! failure means the computation ran to completion and the answer was "no".
//!
//! Only the first kind is a [`DsigError`]. The second kind is reported as a
//! [`VerificationResult`](crate::VerificationResult) whose
//! [`core_valid`](crate::VerificationResult::core_valid) is `false`, because
//! "this signature is forged" is an answer, not an error — and because turning
//! it into an error makes it far too easy for a caller to treat a forgery as a
//! transient failure and retry.

use std::error::Error;
use std::fmt;

use crate::x509::CertificateError;

/// The result type used throughout `oxixml-dsig`.
pub type DsigResult<T> = Result<T, DsigError>;

/// Something that stopped XML Signature processing from producing an answer.
///
/// A *cryptographic* failure — a signature that simply does not verify — is
/// not one of these; it is a [`VerificationResult`](crate::VerificationResult)
/// whose [`core_valid`](crate::VerificationResult::core_valid) is `false`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DsigError {
    /// The `<ds:Signature>` subtree does not match the schema of
    /// [XML Signature §4](https://www.w3.org/TR/xmldsig-core1/#sec-Signature):
    /// a required child is missing, a required attribute is absent, or an
    /// element appears where the content model forbids it.
    Structure(String),

    /// An `Algorithm` attribute names a URI this crate does not recognise at
    /// all.
    UnknownAlgorithm(String),

    /// An `Algorithm` attribute names a URI this crate recognises but does not
    /// implement. See the [crate-level gap list](crate#gaps).
    UnsupportedAlgorithm(String),

    /// Base64 character data (a `DigestValue`, a `SignatureValue`, an RSA
    /// `Modulus`, …) is not valid base64.
    Base64 {
        /// Which element the bad text came from.
        context: String,
        /// The decoder's complaint.
        message: String,
    },

    /// A `Reference` `URI` could not be dereferenced: the fragment names no
    /// element, the XPointer is not one of the forms XML Signature requires, or
    /// the reference is external and no [resolver](crate::UriResolver) accepted
    /// it.
    Dereference(String),

    /// A canonicalization algorithm failed; the message comes from
    /// [`oxixml_c14n`].
    Canonicalization(String),

    /// An XPath expression could not be parsed or evaluated.
    XPath(String),

    /// A transform failed for a reason of its own: `base64` fed non-base64
    /// octets, an XSLT transform with no stylesheet, an XML parse of an octet
    /// stream that is not well formed.
    Transform(String),

    /// No key could be produced for the signature: the
    /// [`KeySelector`](crate::KeySelector) declined, or the `KeyInfo` carries
    /// no key material this crate can use.
    NoKey(String),

    /// Key material was found but is malformed: a truncated DER encoding, an
    /// RSA modulus of zero, a `KeyInfoReference` cycle.
    Key(String),

    /// An X.509 certificate could not be read, or a certification path from it
    /// to a trust anchor could not be validated.
    ///
    /// The [`CertificateError`] inside names the rule that was broken and the
    /// certificate that broke it; it is boxed because it is much larger than
    /// the other variants and far rarer.
    Certificate(Box<CertificateError>),

    /// A DOM operation failed; the message comes from [`oxixml_dom`].
    Dom(String),

    /// The signature is well formed and could have been checked, but a policy
    /// configured on the [`Verifier`](crate::Verifier) forbids it — most often
    /// an `HMACOutputLength` below the floor that
    /// [XML Signature §4.4.2](https://www.w3.org/TR/xmldsig-core1/#sec-SignatureMethod)
    /// makes mandatory.
    Policy(String),
}

impl fmt::Display for DsigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Structure(message) => write!(f, "malformed XML Signature: {message}"),
            Self::UnknownAlgorithm(uri) => write!(f, "unknown algorithm identifier `{uri}`"),
            Self::UnsupportedAlgorithm(uri) => write!(f, "unsupported algorithm `{uri}`"),
            Self::Base64 { context, message } => {
                write!(f, "invalid base64 in {context}: {message}")
            }
            Self::Dereference(message) => write!(f, "cannot dereference reference: {message}"),
            Self::Canonicalization(message) => write!(f, "canonicalization failed: {message}"),
            Self::XPath(message) => write!(f, "XPath error: {message}"),
            Self::Transform(message) => write!(f, "transform failed: {message}"),
            Self::NoKey(message) => write!(f, "no verification key: {message}"),
            Self::Key(message) => write!(f, "malformed key material: {message}"),
            Self::Certificate(error) => write!(f, "certificate rejected: {error}"),
            Self::Dom(message) => write!(f, "DOM error: {message}"),
            Self::Policy(message) => write!(f, "rejected by policy: {message}"),
        }
    }
}

impl Error for DsigError {}

impl From<oxixml_c14n::C14nError> for DsigError {
    fn from(error: oxixml_c14n::C14nError) -> Self {
        Self::Canonicalization(error.to_string())
    }
}

impl From<oxixml_dom::DomError> for DsigError {
    fn from(error: oxixml_dom::DomError) -> Self {
        Self::Dom(error.to_string())
    }
}

impl From<oxixml_dom::BuildError> for DsigError {
    fn from(error: oxixml_dom::BuildError) -> Self {
        Self::Transform(error.to_string())
    }
}

/// Decodes base64 character data, tagging a failure with the element it came
/// from.
pub(crate) fn decode_base64(context: &str, text: &str) -> DsigResult<Vec<u8>> {
    oxixml_text::base64::base64_decode_forgiving(text).map_err(|error| DsigError::Base64 {
        context: context.to_owned(),
        message: error.to_string(),
    })
}

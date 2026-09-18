//! The error type shared by every operation in this crate.
//!
//! One flat enum is used throughout rather than a per-layer hierarchy: an XML Encryption
//! operation crosses the DOM, the base64 codec and the cipher primitives in a single call, and
//! a caller almost always wants to report "decryption failed" with one message.
//!
//! # Deliberately vague decryption failures
//!
//! [XML Encryption 1.1 §6.7](https://www.w3.org/TR/xmlenc-core1/#sec-Error-Messages) warns that
//! a decryptor which distinguishes *padding is wrong* from *the key was wrong* hands an attacker
//! a padding oracle (Vaudenay). Accordingly, every CBC unpadding failure and every AEAD tag
//! mismatch surfaces as the single opaque [`EncError::DecryptionFailed`] variant, whose
//! `Display` output carries no detail about which check failed.

use core::fmt;
use std::io;
use std::string::FromUtf8Error;

use oxixml_dom::{BuildError, DomError, SerializeError};
use oxixml_support::ec::EcError;
use oxixml_text::base64::Base64Error;

/// The result type used throughout `oxixml-enc`.
pub type Result<T> = core::result::Result<T, EncError>;

/// Anything that can go wrong while encrypting, decrypting, parsing or building XML Encryption
/// structures.
#[derive(Debug)]
#[non_exhaustive]
pub enum EncError {
    /// An algorithm URI was syntactically fine but is not implemented by this crate.
    ///
    /// The payload is the offending URI. See the crate-level documentation for the list of
    /// deliberately unimplemented algorithms.
    UnsupportedAlgorithm(String),

    /// A key of the wrong length was supplied for an algorithm.
    InvalidKeyLength {
        /// A human-readable name for the algorithm, e.g. `"AES"`.
        algorithm: &'static str,
        /// The number of key octets that were supplied.
        length: usize,
    },

    /// An input whose length must be a whole number of cipher blocks was not.
    InvalidBlockLength {
        /// The block size the algorithm requires, in octets.
        block_size: usize,
        /// The number of octets that were supplied.
        length: usize,
    },

    /// Decryption failed.
    ///
    /// This single variant covers an AEAD authentication-tag mismatch, malformed CBC padding,
    /// a malformed RSA-OAEP or PKCS#1 v1.5 encoding and a failed key-wrap integrity check, so
    /// that the error carries no oracle. See the [crate documentation](crate).
    DecryptionFailed,

    /// A required element or attribute was missing from an `EncryptedData`/`EncryptedKey`
    /// structure, or one was present where the schema forbids it.
    InvalidStructure(String),

    /// The `CipherData` used a `CipherReference`, which this crate cannot dereference on its
    /// own because doing so requires an application-supplied retrieval mechanism.
    ///
    /// The payload is the `URI` attribute of the reference.
    CipherReferenceNotResolved(String),

    /// No key could be found for an `EncryptedData` or `EncryptedKey`.
    ///
    /// The payload describes what was looked for (a `KeyName`, a `Recipient`, …).
    KeyNotFound(String),

    /// An RSA key or an RSA operation was rejected before any secret was used.
    ///
    /// The payload is a short static reason; it never depends on secret data.
    InvalidRsaKey(&'static str),

    /// A key-agreement public value, private value or domain parameter was rejected.
    ///
    /// The payload is a short static reason. It never depends on secret data: every check that
    /// produces this variant runs on values the document already published, before the private
    /// key is used, so reporting the reason tells an attacker nothing they did not send.
    InvalidAgreementKey(&'static str),

    /// A key-derivation function was asked for something it will not do.
    ///
    /// The payload is a short static reason; it never depends on secret data.
    InvalidKeyDerivation(&'static str),

    /// The message handed to a public-key primitive was too long for the modulus.
    MessageTooLong {
        /// The number of octets that were supplied.
        length: usize,
        /// The largest number of octets the primitive accepts.
        maximum: usize,
    },

    /// The random source failed.
    Entropy(io::Error),

    /// A base64 `CipherValue` (or `OAEPparams`) could not be decoded.
    Base64(Base64Error),

    /// Decrypted octets were not well-formed UTF-8, so they cannot be XML.
    NotUtf8(FromUtf8Error),

    /// A DOM operation failed.
    Dom(DomError),

    /// Re-parsing decrypted XML failed.
    Build(BuildError),

    /// Serializing the plaintext to octets failed.
    Serialize(SerializeError),
}

impl fmt::Display for EncError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedAlgorithm(uri) => write!(f, "unsupported algorithm `{uri}`"),
            Self::InvalidKeyLength { algorithm, length } => {
                write!(f, "invalid {algorithm} key length: {length} octets")
            }
            Self::InvalidBlockLength { block_size, length } => write!(
                f,
                "input of {length} octets is not a multiple of the {block_size}-octet block size"
            ),
            Self::DecryptionFailed => f.write_str("decryption failed"),
            Self::InvalidStructure(what) => write!(f, "invalid XML Encryption structure: {what}"),
            Self::CipherReferenceNotResolved(uri) => {
                write!(f, "CipherReference `{uri}` was not resolved")
            }
            Self::KeyNotFound(what) => write!(f, "no key available for {what}"),
            Self::InvalidRsaKey(reason) => write!(f, "invalid RSA key: {reason}"),
            Self::InvalidAgreementKey(reason) => {
                write!(f, "invalid key-agreement value: {reason}")
            }
            Self::InvalidKeyDerivation(reason) => write!(f, "key derivation failed: {reason}"),
            Self::MessageTooLong { length, maximum } => write!(
                f,
                "message of {length} octets exceeds the {maximum}-octet capacity"
            ),
            Self::Entropy(error) => write!(f, "the random source failed: {error}"),
            Self::Base64(error) => write!(f, "invalid base64: {error}"),
            Self::NotUtf8(error) => write!(f, "decrypted octets are not UTF-8: {error}"),
            Self::Dom(error) => write!(f, "{error}"),
            Self::Build(error) => write!(f, "{error}"),
            Self::Serialize(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for EncError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Entropy(error) => Some(error),
            Self::Base64(error) => Some(error),
            Self::NotUtf8(error) => Some(error),
            Self::Dom(error) => Some(error),
            Self::Build(error) => Some(error),
            Self::Serialize(error) => Some(error),
            _ => None,
        }
    }
}

/// Every elliptic-curve rejection folds into [`EncError::InvalidAgreementKey`].
///
/// Nothing is lost by flattening: an [`EcError`] only ever describes a value the *sender* chose
/// and published, so its detail is already known to whoever sent it. The reason is kept as a
/// short static string so that it can never carry a secret.
impl From<EcError> for EncError {
    fn from(error: EcError) -> Self {
        Self::InvalidAgreementKey(match error {
            EcError::InvalidPointLength { .. } => {
                "the encoded curve point does not have the curve's length"
            }
            EcError::UnsupportedPointEncoding(_) => {
                "only uncompressed curve points (04 ‖ X ‖ Y) are supported"
            }
            EcError::CoordinateOutOfRange => {
                "a curve point coordinate is not a canonical field element"
            }
            EcError::NotOnCurve => "the curve point does not satisfy the curve equation",
            EcError::IdentityPoint => "the point at infinity is not a valid public key",
            EcError::InvalidScalarLength { .. } => {
                "the scalar does not have the curve's scalar width"
            }
            EcError::ScalarOutOfRange => "the scalar is zero or not less than the group order",
            _ => "the elliptic-curve value was rejected",
        })
    }
}

impl From<Base64Error> for EncError {
    fn from(error: Base64Error) -> Self {
        Self::Base64(error)
    }
}

impl From<FromUtf8Error> for EncError {
    fn from(error: FromUtf8Error) -> Self {
        Self::NotUtf8(error)
    }
}

impl From<DomError> for EncError {
    fn from(error: DomError) -> Self {
        Self::Dom(error)
    }
}

impl From<BuildError> for EncError {
    fn from(error: BuildError) -> Self {
        Self::Build(error)
    }
}

impl From<SerializeError> for EncError {
    fn from(error: SerializeError) -> Self {
        Self::Serialize(error)
    }
}

impl From<io::Error> for EncError {
    fn from(error: io::Error) -> Self {
        Self::Entropy(error)
    }
}

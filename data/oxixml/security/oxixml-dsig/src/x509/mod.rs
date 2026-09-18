//! X.509 certificates and certification path validation —
//! [RFC 5280](https://www.rfc-editor.org/rfc/rfc5280).
//!
//! A `<ds:KeyInfo>` may carry a whole certificate chain, and reading the
//! signer's public key out of the first certificate it finds proves nothing at
//! all: the document under examination supplied that certificate. This module
//! turns the chain into evidence, by building a certification path from the
//! signer's certificate to a trust anchor the *caller* chose and checking every
//! link of it — signature, validity period, name, basic constraints, key usage
//! and critical extensions.
//!
//! # Using it
//!
//! The path validator is wired into [`TrustedCertificates`](crate::TrustedCertificates),
//! which is the [`KeySelector`](crate::KeySelector) an application configures
//! with the certificates it trusts:
//!
//! ```
//! use oxixml_dsig::{PathValidator, Timestamp, TrustedCertificates};
//!
//! # let root_der: Vec<u8> = oxixml_text::base64::base64_decode("MIIBpDCCAQ2gAwIBAgIFASNFZ4kwDQYJKoZIhvcNAQELBQAwFjEUMBIGA1UEAwwLT3hpWE1MIFRlc3QwHhcNMjYwMTAxMDAwMDAwWhcNMzYwMTAxMDAwMDAwWjAWMRQwEgYDVQQDDAtPeGlYTUwgVGVzdDCBnzANBgkqhkiG9w0BAQEFAAOBjQAwgYkCgYEAjkcoCslxEus5u8QIPMAYjDIpNgX12fbWalSPmhbKjlNeR5ukRjB0xBJqYvncXrPl7lHzDemgFsdEbJCvTgF1imtaMkc6yfaHkJ4+RgzPR2SBU4KEaVq69fNR5Dco0vl/VUZD54drPqBBvXhiqwpAK3ZLHcPSLxzRQi1a20l+tlUCAwEAATANBgkqhkiG9w0BAQsFAAOBgQAbgJ1S/Y0svcUZpowjNkA5BxrXfJNopeDC6Rxz77AHzTfqnwkVWkjAYxY8mfbeCQkEucKK2FxB8bv1WzDE/rlW5h8jtyxmvYA3d/imKCjC7SPcibgHh55nIkmwA9oXpLL5C4rM0fcApbudoveWWwRHWAlxDiusBrr8vbzaDOByHA==")?;
//! // Without a validator, a certificate is trusted only when its encoding is
//! // one the caller listed. With one, any certificate that chains to a listed
//! // anchor is trusted — and every certificate on the way is checked.
//! let trust = TrustedCertificates::new()
//!     .with_certificate(root_der)
//!     .with_path_validation(PathValidator::new(Timestamp::from_unix_seconds(1_800_000_000)));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! The validation time is a parameter, never a clock reading —
//! see [`Timestamp`].
//!
//! # What is checked, and what is not
//!
//! [`PathValidator`] documents the algorithm and the four things it
//! deliberately leaves out: revocation, name constraints, certificate policies,
//! and fetching anything over the network. None of them is silently assumed to
//! have passed — a critical extension this crate does not process makes the
//! certificate unusable.

mod certificate;
mod error;
mod name;
mod oid;
mod path;
mod signature;
#[cfg(test)]
pub(crate) mod testcert;
mod time;

pub use certificate::{Certificate, KeyUsage};
pub use error::CertificateError;
pub use name::DistinguishedName;
pub use path::{CertificatePath, PathValidator};
pub use time::Timestamp;

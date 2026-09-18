//! The cryptography XML Signature needs, built from `oxixml-support` primitives.
//!
//! Nothing in here is exported: the crate's public surface talks about
//! [`DigestAlgorithm`](crate::DigestAlgorithm) and
//! [`VerificationKey`](crate::VerificationKey), never about padding schemes.
//! The submodules are:
//!
//! * [`digest`] — SHA-1/224/256/384/512 and HMAC dispatch, plus the
//!   constant-time comparison every digest check goes through;
//! * [`pkcs1`] — EMSA-PKCS1-v1_5 and the RSA signature primitives;
//! * [`pss`] — EMSA-PSS and MGF1, on the same RSA primitives;
//! * [`ecdsa`] and [`dsa`] — the two discrete-logarithm signature schemes, over
//!   [`oxixml_support::ec`] and [`oxixml_support::bigint`] respectively;
//! * [`rfc6979`] — the deterministic nonce both of them sign with;
//! * [`der`] — the small DER reader that pulls a key out of a
//!   `SubjectPublicKeyInfo`, an X.509 certificate or a PKCS#1/PKCS#8/SEC 1
//!   private key.

pub(crate) mod der;
pub(crate) mod digest;
pub(crate) mod dsa;
pub(crate) mod ecdsa;
pub(crate) mod pkcs1;
pub(crate) mod pss;
pub(crate) mod rfc6979;

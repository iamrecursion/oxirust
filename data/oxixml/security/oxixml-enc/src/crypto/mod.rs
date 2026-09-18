//! The symmetric and public-key primitives XML Encryption is built out of, implemented from the
//! standards with no external crates.
//!
//! Nothing here knows anything about XML. Each module implements one standard and is testable
//! against that standard's own vectors:
//!
//! | Module | Standard |
//! |---|---|
//! | [`aes`] | FIPS 197, the AES block cipher |
//! | [`cbc`] | NIST SP 800-38A §6.2 plus the padding of XMLEnc §5.2.1 |
//! | [`ctr`] | NIST SP 800-38A §6.5 and the `inc32` of SP 800-38D |
//! | [`gcm`] | NIST SP 800-38D, `GHASH` and `GCM-AE`/`GCM-AD` |
//! | [`keywrap`] | RFC 3394 and RFC 5649 |
//! | [`digest`] | FIPS 180-4, via [`oxixml_support::hash`] |
//! | [`mgf1`] | RFC 8017 Appendix B.2.1 |
//! | [`rsa`] | RFC 8017 §5, §7.1, §7.2 and, for key generation, FIPS 186-5 Appendix A.1.1 |
//! | [`primes`] | FIPS 186-5 Appendix B.3, the Miller–Rabin test and prime generation |
//! | [`kdf`] | NIST SP 800-56A §5.8.1 and the XMLEnc 1.0 digest construction |
//! | [`ecdh`] | NIST SP 800-56A §5.7.1.2, elliptic-curve Diffie–Hellman |
//! | [`dh`] | NIST SP 800-56A §5.7.1.1, finite-field Diffie–Hellman |
//! | [`ct`] | data-independent comparison and selection |
//!
//! # Reading the security notes
//!
//! Every module documents what it does *not* protect against. The short version: the AES tables
//! and the big-integer exponentiation are not constant-time, so this crate is appropriate for
//! processing documents, and inappropriate as the crypto core of a service that an attacker can
//! co-locate with and time. Those limits are stated in each module rather than buried here.

pub mod aes;
pub mod cbc;
pub mod ct;
pub mod ctr;
pub mod dh;
pub mod digest;
pub mod ecdh;
pub mod gcm;
pub mod kdf;
pub mod keywrap;
pub mod mgf1;
pub mod primes;
pub mod rsa;

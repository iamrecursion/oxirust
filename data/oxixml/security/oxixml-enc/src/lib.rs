//! `oxixml-enc` — [XML Encryption 1.1](https://www.w3.org/TR/xmlenc-core1/) in pure Rust, with
//! its ciphers implemented from the standards rather than pulled in.
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the COOLJAPAN
//! Pure-Rust foundation for all XML and RDF processing.
//!
//! # What this crate does
//!
//! * **Models** `xenc:EncryptedData` and `xenc:EncryptedKey` completely and losslessly
//!   ([`model`]), including the schema's extension wildcards.
//! * **Encrypts and decrypts nodes** of an [`oxixml_dom`] tree ([`encrypt_element`],
//!   [`encrypt_content`], [`decrypt`]), with the namespace and entity handling of XMLEnc §4.5
//!   that makes a decrypted fragment mean the same thing it did before it was encrypted.
//! * **Implements the primitives**: AES from FIPS 197, CBC and CTR from SP 800-38A, GCM from
//!   SP 800-38D, AES Key Wrap from RFC 3394 and RFC 5649, and RSA with OAEP and PKCS#1 v1.5
//!   from RFC 8017 ([`crypto`]). No external crates, no `unsafe`, no C.
//! * **Agrees keys**: `ECDH-ES` over P-256 and P-384, finite-field Diffie–Hellman, and the
//!   ConcatKDF of NIST SP 800-56A that turns the shared secret into a key-encryption key
//!   ([`agreement`]) — including reading and writing the `xenc:AgreementMethod` that carries an
//!   agreement's public values through a document ([`model::AgreementMethod`]) and resolving a
//!   registered EC or Diffie–Hellman private key against one
//!   ([`key::KeyStore::open_encrypted_key`]).
//!
//! # Example: encrypt an element, then get it back
//!
//! ```
//! use oxixml_dom::build;
//! use oxixml_enc::algorithm::{BlockEncryption, KeyWrap};
//! use oxixml_enc::key::{KeyStore, SymmetricKey};
//! use oxixml_enc::model::KeyInfo;
//! use oxixml_enc::random::SystemRandom;
//! use oxixml_enc::{EncryptOptions, decrypt, encrypt_element, wrap_key_for_key_info};
//!
//! let mut doc = build::parse_str(
//!     r#"<Order xmlns="urn:example:orders">
//!          <Customer>A. Nonymous</Customer>
//!          <Payment><Card>4019 2445 0277 5567</Card></Payment>
//!        </Order>"#,
//! )?;
//! let root = doc.document_element().expect("a document element");
//! let payment = doc
//!     .get_elements_by_tag_name_ns(root, Some("urn:example:orders"), "Payment")
//!     .next()
//!     .expect("a Payment element");
//!
//! let mut rng = SystemRandom::new();
//!
//! // A fresh content-encryption key, wrapped under a long-lived key-encryption key so that it
//! // can travel inside the document.
//! let key_encryption_key = SymmetricKey::new(vec![0x5c; 32]);
//! let content_key = SymmetricKey::generate(32, &mut rng)?;
//! let carried =
//!     wrap_key_for_key_info(KeyWrap::Aes256, &key_encryption_key, &content_key, Some("kek"))?;
//!
//! let options = EncryptOptions::new()
//!     .with_algorithm(BlockEncryption::Aes256Gcm)
//!     .with_key_info(KeyInfo::with_encrypted_key(carried));
//! let encrypted = encrypt_element(&mut doc, payment, &content_key, &options, &mut rng)?;
//!
//! let ciphered = doc.to_xml();
//! assert!(!ciphered.contains("4019"));
//! assert!(!ciphered.contains("Payment"));
//! assert!(ciphered.contains("A. Nonymous"), "the rest of the document is untouched");
//!
//! // The recipient holds only the key-encryption key.
//! let store = KeyStore::new().with_named_key_encryption_key("kek", key_encryption_key);
//! decrypt(&mut doc, encrypted, &store)?;
//! assert!(doc.to_xml().contains("4019 2445 0277 5567"));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Security posture, stated plainly
//!
//! Cryptographic code that does not say what it fails to protect is worse than useless, so:
//!
//! * **Prefer AES-GCM.** It is the default. CBC-mode XML Encryption is subject to the
//!   Jager–Somorovsky adaptive chosen-ciphertext attack (CCS 2011), which recovers plaintext
//!   from any decryptor whose behaviour differs between "padding was wrong" and "key was
//!   wrong". This crate reports every decryption failure as one opaque
//!   [`EncError::DecryptionFailed`], which is necessary but cannot compensate for a difference
//!   the *application* introduces downstream.
//! * **`EncryptedData` is not authenticated, even under GCM.** XML Encryption defines no
//!   associated data, so `Type`, `Id`, `EncryptionMethod` and `KeyInfo` are unprotected.
//!   Combine with XML Signature when the document's structure matters.
//! * **The primitives are not constant-time.** AES uses table lookups indexed by secret octets,
//!   and RSA private-key operations, finite-field Diffie–Hellman and the curve arithmetic all
//!   rest on the variable-time big integers of [`oxixml_support::bigint`]. This crate is built
//!   for processing documents, not for serving an attacker who can co-locate and measure it.
//!   Everything that *can* be data-independent — tag comparison, padding validation, OAEP
//!   decoding — is; see [`crypto::ct`].
//! * **A key agreement authenticates nobody.** `ECDH-ES` is ephemeral-static, so anyone at all
//!   can produce a document that decrypts correctly. Combine it with XML Signature if the
//!   sender's identity matters; see [`agreement`].
//! * **Encryption hides content, not shape.** The position and approximate size of every
//!   encrypted element remain visible (XMLEnc §6.3).
//!
//! # Deliberately not implemented
//!
//! These are gaps by decision, not oversights, and each one fails loudly with
//! [`EncError::UnsupportedAlgorithm`] rather than silently:
//!
//! * **Triple DES** (`tripledes-cbc`, `kw-tripledes`) — disallowed for encryption by
//!   NIST SP 800-131A; shipping it would invite new documents to use it.
//! * **PBKDF2** (`pbkdf2`) — it derives a key from a passphrase, not from a shared secret, and
//!   is OPTIONAL in XMLEnc 1.1. Choosing an iteration count and a passphrase policy is the
//!   application's decision, not this crate's.
//! * **Curves other than P-256 and P-384**, and compressed point encodings. A `NamedCurve`
//!   naming anything else — P-521, a Brainpool curve — is refused rather than approximated.
//! * **Dereferencing `CipherReference`** — fetching a URI named by an untrusted document is
//!   server-side request forgery; the reference is modelled faithfully and
//!   [`EncError::CipherReferenceNotResolved`] hands the decision back to the application.
//! * **`ds:Transform` evaluation** — transforms are modelled and preserved, never run.
//! * **Building an `xenc:AgreementMethod` for `dh` or `dh-es`.** [`KeyStore::open_encrypted_key`]
//!   *decrypts* an `EncryptedKey` agreed via any of the three key-agreement methods, and
//!   [`ecdh_es_wrap_key_for_key_info`] builds one for `ECDH-ES`, but there is no equivalent
//!   builder for the two finite-field methods yet; an application that wants `dh`/`dh-es` builds
//!   the `AgreementMethod` by hand from [`model::AgreementMethod`]'s public fields.
//!
//! [`KeyStore::open_encrypted_key`]: key::KeyStore::open_encrypted_key
//!
//! # Module map
//!
//! | Module | What is in it |
//! |---|---|
//! | [`algorithm`] | the algorithm URIs of XMLEnc §5, as exhaustive enumerations |
//! | [`model`] | `EncryptedData`, `EncryptedKey` and everything under them |
//! | [`key`] | key material, and the [`KeyResolver`] hook that finds the right one |
//! | [`cipher`] | the IV/tag framing of XMLEnc §5.2, and key wrapping and transport |
//! | [`agreement`] | the key agreement of XMLEnc §5.6 and the key derivation of §5.4 |
//! | [`crypto`] | AES, CBC, CTR, GCM, key wrap, MGF1, RSA, ECDH, DH, KDFs, primes |
//! | [`fragment`] | the namespace and entity handling of XMLEnc §4.5.3 and §4.5.4 |
//! | [`random`] | where initialization vectors and OAEP seeds come from |
//! | [`api`] | [`encrypt_element`], [`encrypt_content`] and [`decrypt`] |

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod agreement;
pub mod algorithm;
pub mod api;
pub mod cipher;
pub mod crypto;
mod error;
pub mod fragment;
pub mod key;
pub mod model;
pub mod random;

pub use algorithm::{BlockEncryption, KeyAgreement, KeyDerivation, KeyTransport, KeyWrap};
pub use api::{
    Decrypted, EncryptOptions, decrypt, ecdh_es_wrap_key_for_key_info, encrypt_content,
    encrypt_element, transport_key_for_key_info, wrap_key_for_key_info,
};
pub use error::{EncError, Result};
pub use key::{KeyResolver, KeyStore, SymmetricKey};
pub use model::{ContentType, EncryptedData, EncryptedKey};

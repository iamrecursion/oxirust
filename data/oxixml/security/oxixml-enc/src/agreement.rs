//! Key agreement: [XML Encryption §5.6](https://www.w3.org/TR/xmlenc-core1/#sec-Alg-KeyAgreement),
//! where the two parties *derive* a key-encryption key instead of one of them encrypting one for
//! the other.
//!
//! An `xenc:AgreementMethod` inside a `ds:KeyInfo` says three things: which
//! [`KeyAgreement`](crate::algorithm::KeyAgreement) produces the shared secret, whose public
//! values go into it (`OriginatorKeyInfo` and `RecipientKeyInfo`), and — for the XMLEnc 1.1
//! methods — which [`KeyDerivation`](crate::algorithm::KeyDerivation) turns that secret into
//! key material. This module is the second and third of those: given the keys and the
//! parameters, it produces the key-encryption key, and for `ECDH-ES` it goes one step further
//! and wraps a content-encryption key under it, which is what the agreement exists to do.
//!
//! ```text
//! ECDH-ES:  Z = x(d_originator · Q_recipient)   →  ConcatKDF  →  KEK  →  AES-KW(CEK)
//! dh-es:    ZZ = y_recipient ^ x_originator     →  ConcatKDF  →  KEK  →  AES-KW(CEK)
//! dh:       ZZ = y_recipient ^ x_originator     →  §5.5.2 KDF →  KEK  →  AES-KW(CEK)
//! ```
//!
//! # What an agreement does and does not establish
//!
//! `ECDH-ES` and `dh-es` are **ephemeral-static**: the originator makes a fresh key pair for
//! every message and publishes its public half in the document. That gives forward secrecy
//! against a later compromise of the *originator's* key — there is nothing to compromise — and
//! none against a compromise of the recipient's, which is the long-lived one. It authenticates
//! nothing at all: anyone can make an ephemeral key pair, so anyone can produce a document that
//! decrypts. If the recipient needs to know who sent it, that is XML Signature's job.
//!
//! The derivation parameters are what bind the key to its context, and they are only as good as
//! the caller makes them. Two agreements whose `AlgorithmID`, `PartyUInfo` and `PartyVInfo`
//! coincide derive the *same* key from the same secret; see [`ConcatKdfParams`].
//!
//! # Example: `ECDH-ES` with the ConcatKDF and AES Key Wrap
//!
//! ```
//! use oxixml_enc::agreement::{ecdh_es_unwrap_key, ecdh_es_wrap_key};
//! use oxixml_enc::algorithm::KeyWrap;
//! use oxixml_enc::crypto::digest::Digest;
//! use oxixml_enc::crypto::ecdh::EcPrivateKey;
//! use oxixml_enc::crypto::kdf::ConcatKdfParams;
//! use oxixml_enc::key::SymmetricKey;
//! use oxixml_enc::random::PseudoRandom;
//! use oxixml_support::ec::CurveId;
//!
//! // In real use this is `SystemRandom`; a deterministic source keeps the example reproducible.
//! let mut rng = PseudoRandom::new(2026);
//!
//! let recipient = EcPrivateKey::generate(CurveId::P256, &mut rng)?;
//! let content_key = SymmetricKey::new(vec![0x4b; 32]);
//! let params = ConcatKdfParams::new(Digest::Sha256)
//!     .with_algorithm_id(KeyWrap::Aes256.uri().as_bytes().to_vec());
//!
//! let agreed = ecdh_es_wrap_key(
//!     &recipient.public_key(),
//!     KeyWrap::Aes256,
//!     &params,
//!     &content_key,
//!     &mut rng,
//! )?;
//!
//! // The document carries `agreed.wrapped_key` and the originator's public point; the
//! // recipient's private key is all that is needed to get the content key back.
//! let recovered = ecdh_es_unwrap_key(
//!     &recipient,
//!     &agreed.originator_public_key,
//!     KeyWrap::Aes256,
//!     &params,
//!     &agreed.wrapped_key,
//! )?;
//! assert_eq!(recovered.as_bytes(), content_key.as_bytes());
//! # Ok::<(), oxixml_enc::EncError>(())
//! ```

use oxixml_support::ec::CurveId;

use crate::algorithm::KeyWrap;
use crate::cipher;
use crate::crypto::dh::{DhPrivateKey, DhPublicKey};
use crate::crypto::digest::Digest;
use crate::crypto::ecdh::{EcPrivateKey, EcPublicKey};
use crate::crypto::kdf::{ConcatKdfParams, concat_kdf, dh_legacy_kdf};
use crate::error::Result;
use crate::key::SymmetricKey;
use crate::random::RandomSource;

/// Wraps a derived key and overwrites the intermediate shared secret, whether or not the
/// derivation succeeded.
///
/// Taking the derivation's `Result` rather than its value is what makes the failure path wipe
/// the secret too; an early `?` at the call site would leave it in the allocation. Zeroing is
/// hygiene, not a guarantee — a `Vec` that has been reallocated leaves its old contents behind —
/// but it costs nothing.
fn finish(mut shared_secret: Vec<u8>, derived: Result<Vec<u8>>) -> Result<SymmetricKey> {
    shared_secret.fill(0);
    derived.map(SymmetricKey::new)
}

/// Derives the key-encryption key of an `ECDH-ES` agreement: ECDH, then the ConcatKDF.
///
/// `key_len` is the length the derived key must have, which is the
/// [`KeyWrap::key_len`](crate::algorithm::KeyWrap::key_len) of the wrap algorithm the
/// `EncryptedKey` declares — or the
/// [`BlockEncryption::key_len`](crate::algorithm::BlockEncryption::key_len) when the agreement
/// derives a content-encryption key directly.
///
/// # Errors
///
/// Returns [`EncError::InvalidAgreementKey`](crate::EncError::InvalidAgreementKey) when the two
/// keys are on different curves or the peer's point is not a valid public key, and
/// [`EncError::InvalidKeyDerivation`](crate::EncError::InvalidKeyDerivation) when `key_len` is
/// out of range.
pub fn ecdh_es_key_encryption_key(
    private_key: &EcPrivateKey,
    peer: &EcPublicKey,
    params: &ConcatKdfParams,
    key_len: usize,
) -> Result<SymmetricKey> {
    let shared_secret = private_key.shared_secret(peer)?;
    let derived = concat_kdf(params, &shared_secret, key_len);
    finish(shared_secret, derived)
}

/// Derives the key-encryption key of a `dh-es` agreement: finite-field Diffie–Hellman, then the
/// ConcatKDF.
///
/// # Errors
///
/// Returns [`EncError::InvalidAgreementKey`](crate::EncError::InvalidAgreementKey) when the two
/// values belong to different domain parameters or the shared secret is degenerate, and
/// [`EncError::InvalidKeyDerivation`](crate::EncError::InvalidKeyDerivation) when `key_len` is
/// out of range.
pub fn dh_es_key_encryption_key(
    private_key: &DhPrivateKey,
    peer: &DhPublicKey,
    params: &ConcatKdfParams,
    key_len: usize,
) -> Result<SymmetricKey> {
    let shared_secret = private_key.shared_secret(peer)?;
    let derived = concat_kdf(params, &shared_secret, key_len);
    finish(shared_secret, derived)
}

/// Derives the key-encryption key of the legacy `dh` agreement, whose key derivation function is
/// fixed by XMLEnc 1.0 §5.5.2 rather than named by a `KeyDerivationMethod`.
///
/// `digest` comes from the `ds:DigestMethod` child of the `AgreementMethod`, `ka_nonce` from its
/// `KA-Nonce` child (empty when absent), and `encryption_algorithm` is the URI of the algorithm
/// the derived key will be **used for** — the key wrap or the block cipher — never the URI of
/// the agreement method itself.
///
/// # Errors
///
/// The errors of [`DhPrivateKey::shared_secret`] and
/// [`dh_legacy_kdf`].
pub fn dh_key_encryption_key(
    private_key: &DhPrivateKey,
    peer: &DhPublicKey,
    digest: Digest,
    encryption_algorithm: &str,
    ka_nonce: &[u8],
    key_len: usize,
) -> Result<SymmetricKey> {
    let shared_secret = private_key.shared_secret(peer)?;
    let derived = dh_legacy_kdf(
        digest,
        encryption_algorithm,
        ka_nonce,
        &shared_secret,
        key_len,
    );
    finish(shared_secret, derived)
}

/// What an `ECDH-ES` agreement produces for the sender: the wrapped content-encryption key, and
/// the ephemeral public key the recipient needs in order to reach the same key-encryption key.
///
/// The document carries the first as the `EncryptedKey`'s `CipherValue` and the second inside
/// the `AgreementMethod`'s `OriginatorKeyInfo`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgreedKey {
    /// The originator's ephemeral public key. It is *public*: publishing it is the protocol.
    pub originator_public_key: EcPublicKey,
    /// The content-encryption key, wrapped under the derived key-encryption key.
    pub wrapped_key: Vec<u8>,
}

/// Performs a complete `ECDH-ES` key agreement as a sender: makes a fresh ephemeral key pair on
/// the recipient's curve, derives the key-encryption key with the ConcatKDF, and wraps
/// `content_key` under it.
///
/// The ephemeral private key exists only inside this call and is dropped before it returns,
/// which is the whole point of the *ephemeral*-static mode: there is no originator key left to
/// compromise.
///
/// # Errors
///
/// Returns [`EncError::Entropy`](crate::EncError::Entropy) when the random source fails, the
/// errors of [`ecdh_es_key_encryption_key`], and
/// [`EncError::InvalidBlockLength`](crate::EncError::InvalidBlockLength) when `wrap` is one of
/// the RFC 3394 algorithms and `content_key` is not a positive multiple of eight octets — use a
/// `-pad` algorithm for that.
pub fn ecdh_es_wrap_key(
    recipient: &EcPublicKey,
    wrap: KeyWrap,
    params: &ConcatKdfParams,
    content_key: &SymmetricKey,
    rng: &mut dyn RandomSource,
) -> Result<AgreedKey> {
    let originator = EcPrivateKey::generate(recipient.curve(), rng)?;
    let key_encryption_key =
        ecdh_es_key_encryption_key(&originator, recipient, params, wrap.key_len())?;
    let wrapped_key = cipher::wrap_key(wrap, &key_encryption_key, content_key)?;
    Ok(AgreedKey {
        originator_public_key: originator.public_key(),
        wrapped_key,
    })
}

/// Recovers a content-encryption key wrapped by [`ecdh_es_wrap_key`].
///
/// `originator` is the public key the document published in the `AgreementMethod`'s
/// `OriginatorKeyInfo`; it is attacker-controlled, and is validated as such — see
/// [`crate::crypto::ecdh`].
///
/// # Errors
///
/// The errors of [`ecdh_es_key_encryption_key`], and
/// [`EncError::DecryptionFailed`](crate::EncError::DecryptionFailed) when the key wrap's
/// integrity check fails — which is what a wrong derivation parameter looks like from here.
pub fn ecdh_es_unwrap_key(
    recipient: &EcPrivateKey,
    originator: &EcPublicKey,
    wrap: KeyWrap,
    params: &ConcatKdfParams,
    wrapped_key: &[u8],
) -> Result<SymmetricKey> {
    let key_encryption_key =
        ecdh_es_key_encryption_key(recipient, originator, params, wrap.key_len())?;
    cipher::unwrap_key(wrap, &key_encryption_key, wrapped_key)
}

/// Returns the curve of a `NamedCurve` URI, refusing the ones this crate does not implement.
///
/// A convenience over [`crate::crypto::ecdh::curve_from_named_curve_uri`] that reports the
/// refusal as an error rather than as `None`, for callers reading an `xenc11:ECKeyValue`.
///
/// # Errors
///
/// Returns [`EncError::UnsupportedAlgorithm`](crate::EncError::UnsupportedAlgorithm) naming the
/// URI, for every curve other than P-256 and P-384.
pub fn require_named_curve(uri: &str) -> Result<CurveId> {
    crate::crypto::ecdh::curve_from_named_curve_uri(uri)
        .ok_or_else(|| crate::EncError::UnsupportedAlgorithm(uri.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::EncError;
    use crate::random::PseudoRandom;

    fn hex(text: &str) -> Vec<u8> {
        oxixml_text::hex::decode(text).expect("hexadecimal")
    }

    /// The worked example of RFC 7518 Appendix C, end to end: the receiver's private key and the
    /// originator's ephemeral public point are the ones printed there, and the derived key is
    /// the one the appendix arrives at.
    ///
    /// It is a JOSE example rather than an XML one, but the agreement and the derivation are the
    /// same two standards XMLEnc 1.1 §5.6.4 and §5.4.1 cite, which makes it the only published
    /// end-to-end vector for this path.
    #[test]
    fn the_rfc_7518_appendix_c_agreement_is_reproduced() -> Result<()> {
        let recipient = EcPrivateKey::from_be_bytes(
            CurveId::P256,
            &hex("5449836690D75CAF29F0DD029DDB31B3DDB8ABA9D2D515C5012465E817D4A9DC"),
        )?;
        let mut originator_point = vec![0x04];
        originator_point.extend_from_slice(&hex(
            "808D060082C176EED3E776A4AC598CC8672C1779F974EECC9B03411CA5B9495D",
        ));
        originator_point.extend_from_slice(&hex(
            "48B5BFC527DFCE53D6AC7115237D031CCFF87A0570B77350A9E503EE7305A69B",
        ));
        let originator = EcPublicKey::from_sec1(CurveId::P256, &originator_point)?;

        let params = ConcatKdfParams::new(Digest::Sha256)
            .with_algorithm_id(b"\x00\x00\x00\x07A128GCM".to_vec())
            .with_party_u_info(b"\x00\x00\x00\x05Alice".to_vec())
            .with_party_v_info(b"\x00\x00\x00\x03Bob".to_vec())
            .with_supp_pub_info(vec![0x00, 0x00, 0x00, 0x80]);

        let derived = ecdh_es_key_encryption_key(&recipient, &originator, &params, 16)?;
        assert_eq!(
            oxixml_text::hex::encode_upper(derived.as_bytes()),
            "56AA8DEAF8236D205C2228CD71A7101A"
        );
        Ok(())
    }

    #[test]
    fn an_ecdh_es_agreement_wraps_and_unwraps_a_content_key() -> Result<()> {
        let mut rng = PseudoRandom::new(31);
        let recipient = EcPrivateKey::generate(CurveId::P256, &mut rng)?;
        let recipient_public = recipient.public_key();
        let content_key = SymmetricKey::new(vec![0x77; 32]);
        let params = ConcatKdfParams::new(Digest::Sha256)
            .with_algorithm_id(KeyWrap::Aes256.uri().as_bytes().to_vec())
            .with_party_v_info(b"recipient".to_vec());

        let agreed = ecdh_es_wrap_key(
            &recipient_public,
            KeyWrap::Aes256,
            &params,
            &content_key,
            &mut rng,
        )?;
        assert_eq!(agreed.wrapped_key.len(), 40, "RFC 3394 adds one block");
        assert_ne!(
            agreed.originator_public_key, recipient_public,
            "the ephemeral key is a fresh one"
        );

        let recovered = ecdh_es_unwrap_key(
            &recipient,
            &agreed.originator_public_key,
            KeyWrap::Aes256,
            &params,
            &agreed.wrapped_key,
        )?;
        assert_eq!(recovered.as_bytes(), content_key.as_bytes());
        Ok(())
    }

    /// Every input to the derivation is bound to the key: change one and the wrap's integrity
    /// check fails rather than yielding a subtly different key.
    #[test]
    fn a_changed_derivation_parameter_breaks_the_unwrap() -> Result<()> {
        let mut rng = PseudoRandom::new(32);
        let recipient = EcPrivateKey::generate(CurveId::P256, &mut rng)?;
        let content_key = SymmetricKey::new(vec![0x11; 16]);
        let params = ConcatKdfParams::new(Digest::Sha256).with_algorithm_id(b"a".to_vec());

        let agreed = ecdh_es_wrap_key(
            &recipient.public_key(),
            KeyWrap::Aes128,
            &params,
            &content_key,
            &mut rng,
        )?;

        let tampered = ConcatKdfParams::new(Digest::Sha256).with_algorithm_id(b"b".to_vec());
        assert!(matches!(
            ecdh_es_unwrap_key(
                &recipient,
                &agreed.originator_public_key,
                KeyWrap::Aes128,
                &tampered,
                &agreed.wrapped_key,
            ),
            Err(EncError::DecryptionFailed)
        ));

        // As does a different recipient.
        let stranger = EcPrivateKey::generate(CurveId::P256, &mut rng)?;
        assert!(matches!(
            ecdh_es_unwrap_key(
                &stranger,
                &agreed.originator_public_key,
                KeyWrap::Aes128,
                &params,
                &agreed.wrapped_key,
            ),
            Err(EncError::DecryptionFailed)
        ));
        Ok(())
    }

    #[test]
    fn only_the_implemented_named_curves_are_accepted() -> Result<()> {
        assert_eq!(
            require_named_curve(crate::crypto::ecdh::P256_NAMED_CURVE_URI)?,
            CurveId::P256
        );
        assert_eq!(
            require_named_curve(crate::crypto::ecdh::P384_NAMED_CURVE_URI)?,
            CurveId::P384
        );
        assert!(matches!(
            require_named_curve(crate::crypto::ecdh::P521_NAMED_CURVE_URI),
            Err(EncError::UnsupportedAlgorithm(_))
        ));
        Ok(())
    }
}

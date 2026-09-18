//! Key agreement end to end, and proof that everything adjacent to it fails closed.
//!
//! The unit tests inside the crate check each standard against its own vectors. These check the
//! path a *document* takes: agree a key-encryption key, wrap a content-encryption key under it,
//! encrypt with that, and get the plaintext back — and then that every algorithm the crate does
//! not implement is refused rather than approximated.

use oxixml_enc::agreement::{
    dh_es_key_encryption_key, dh_key_encryption_key, ecdh_es_unwrap_key, ecdh_es_wrap_key,
    require_named_curve,
};
use oxixml_enc::algorithm::{BlockEncryption, KeyAgreement, KeyDerivation, KeyTransport, KeyWrap};
use oxixml_enc::cipher;
use oxixml_enc::crypto::dh::{DhParameters, DhPrivateKey};
use oxixml_enc::crypto::digest::Digest;
use oxixml_enc::crypto::ecdh::{
    EcPrivateKey, EcPublicKey, P521_NAMED_CURVE_URI, curve_from_named_curve_uri,
};
use oxixml_enc::crypto::kdf::{ConcatKdfParams, PBKDF2_URI};
use oxixml_enc::key::{KeyStore, SymmetricKey};
use oxixml_enc::model::{EncryptedKey, EncryptionMethod};
use oxixml_enc::random::PseudoRandom;
use oxixml_enc::{EncError, Result};
use oxixml_support::ec::CurveId;

/// A 512-bit safe prime, small enough to keep the test quick. A real deployment uses 2048 bits
/// or more, and should prefer `ECDH-ES` to finite-field Diffie–Hellman entirely.
const DH_PRIME: &str = concat!(
    "FEF3B78C842B8406334F4AAF28C8077743921557B93BC3E50E1F1A1688562AC0",
    "242F191536AC32BBE8C3E858D86D6DDCA8B383C6DDCA0BAFC57355BBA8C035B7",
);

fn hex(text: &str) -> Vec<u8> {
    oxixml_text::hex::decode(text).expect("hexadecimal")
}

/// `ECDH-ES` protects a real payload: the recipient publishes a public point, the sender agrees
/// a key-encryption key against it, wraps a fresh content key, and encrypts with that.
#[test]
fn ecdh_es_protects_a_payload_end_to_end() -> Result<()> {
    let mut rng = PseudoRandom::new(0xc0ffee);
    let recipient = EcPrivateKey::generate(CurveId::P256, &mut rng)?;
    let recipient_public = recipient.public_key();

    let content_key = SymmetricKey::generate(32, &mut rng)?;
    let params = ConcatKdfParams::new(Digest::Sha256)
        .with_algorithm_id(KeyWrap::Aes256.uri().as_bytes().to_vec())
        .with_party_u_info(b"originator".to_vec())
        .with_party_v_info(b"recipient".to_vec());

    let agreed = ecdh_es_wrap_key(
        &recipient_public,
        KeyWrap::Aes256,
        &params,
        &content_key,
        &mut rng,
    )?;
    let ciphertext = cipher::encrypt(
        BlockEncryption::Aes256Gcm,
        &content_key,
        b"the payment instruction",
        &mut rng,
    )?;

    // What travels is the wrapped key, the originator's public point, and the ciphertext.
    let published_point = agreed.originator_public_key.to_sec1()?;
    assert_eq!(published_point.len(), 65);

    let originator = EcPublicKey::from_sec1(CurveId::P256, &published_point)?;
    let recovered_key = ecdh_es_unwrap_key(
        &recipient,
        &originator,
        KeyWrap::Aes256,
        &params,
        &agreed.wrapped_key,
    )?;
    assert_eq!(
        cipher::decrypt(BlockEncryption::Aes256Gcm, &recovered_key, &ciphertext)?,
        b"the payment instruction"
    );
    Ok(())
}

/// Both finite-field methods: `dh-es`, which derives with the ConcatKDF, and the legacy `dh`,
/// which derives with the fixed construction of XMLEnc 1.0 §5.5.2. The same shared secret must
/// give the same key on both sides — and a *different* key under each method, because the two
/// derivations are not interchangeable.
#[test]
fn finite_field_agreement_derives_the_same_key_on_both_sides() -> Result<()> {
    let parameters = DhParameters::new(&hex(DH_PRIME), &[0x02], None)?;
    let mut rng = PseudoRandom::new(0xfeed);
    let originator = DhPrivateKey::generate(&parameters, &mut rng)?;
    let recipient = DhPrivateKey::generate(&parameters, &mut rng)?;
    let originator_public = originator.public_key()?;
    let recipient_public = recipient.public_key()?;

    let params = ConcatKdfParams::new(Digest::Sha256)
        .with_algorithm_id(KeyWrap::Aes128.uri().as_bytes().to_vec());
    let sender = dh_es_key_encryption_key(&originator, &recipient_public, &params, 16)?;
    let receiver = dh_es_key_encryption_key(&recipient, &originator_public, &params, 16)?;
    assert_eq!(sender.as_bytes(), receiver.as_bytes());
    assert_eq!(sender.len(), 16);

    let legacy_sender = dh_key_encryption_key(
        &originator,
        &recipient_public,
        Digest::Sha1,
        KeyWrap::Aes128.uri(),
        &[],
        16,
    )?;
    let legacy_receiver = dh_key_encryption_key(
        &recipient,
        &originator_public,
        Digest::Sha1,
        KeyWrap::Aes128.uri(),
        &[],
        16,
    )?;
    assert_eq!(legacy_sender.as_bytes(), legacy_receiver.as_bytes());
    assert_ne!(
        legacy_sender.as_bytes(),
        sender.as_bytes(),
        "the two derivations are different functions of the same secret"
    );

    // The derived key is usable as a key-encryption key, which is the point of deriving it.
    let content_key = SymmetricKey::new(vec![0x5e; 16]);
    let wrapped = cipher::wrap_key(KeyWrap::Aes128, &sender, &content_key)?;
    assert_eq!(
        cipher::unwrap_key(KeyWrap::Aes128, &receiver, &wrapped)?.as_bytes(),
        content_key.as_bytes()
    );
    Ok(())
}

/// Every algorithm URI the crate deliberately does not implement is unrecognized, in the family
/// it would belong to. A URI that parsed would eventually be *used*.
#[test]
fn unimplemented_algorithm_uris_are_not_recognized() {
    // Triple DES: withdrawn by NIST SP 800-131A.
    assert_eq!(
        BlockEncryption::from_uri("http://www.w3.org/2001/04/xmlenc#tripledes-cbc"),
        None
    );
    assert_eq!(
        KeyWrap::from_uri("http://www.w3.org/2001/04/xmlenc#kw-tripledes"),
        None
    );
    // PBKDF2 derives from a passphrase, not from a shared secret.
    assert_eq!(KeyDerivation::from_uri(PBKDF2_URI), None);
    // Curves other than the two implemented ones.
    assert_eq!(curve_from_named_curve_uri(P521_NAMED_CURVE_URI), None);
    assert_eq!(
        curve_from_named_curve_uri("urn:oid:1.3.36.3.3.2.8.1.1.7"), // brainpoolP256r1
        None
    );
    // And no family answers for another family's URI.
    for uri in [
        KeyAgreement::EcdhEs.uri(),
        KeyAgreement::Dh.uri(),
        KeyDerivation::ConcatKdf.uri(),
    ] {
        assert_eq!(BlockEncryption::from_uri(uri), None, "{uri}");
        assert_eq!(KeyWrap::from_uri(uri), None, "{uri}");
        assert_eq!(KeyTransport::from_uri(uri), None, "{uri}");
    }
}

/// An `EncryptedKey` that names an unimplemented algorithm is refused when it is opened, rather
/// than falling back to something the crate does implement.
#[test]
fn an_encrypted_key_naming_an_unimplemented_algorithm_fails_closed() {
    let store = KeyStore::new().with_default_key_encryption_key(SymmetricKey::new(vec![0x31; 32]));
    for uri in [
        "http://www.w3.org/2001/04/xmlenc#kw-tripledes",
        "http://www.w3.org/2009/xmlenc11#pbkdf2",
        // A key-agreement URI belongs on an `AgreementMethod`, never on an `EncryptionMethod`.
        KeyAgreement::EcdhEs.uri(),
        KeyAgreement::Dh.uri(),
        KeyDerivation::ConcatKdf.uri(),
    ] {
        let mut encrypted = EncryptedKey::new(vec![0u8; 40]);
        encrypted.common.encryption_method = Some(EncryptionMethod::new(uri));
        assert!(
            matches!(
                store.open_encrypted_key(&encrypted),
                Err(EncError::UnsupportedAlgorithm(_))
            ),
            "`{uri}` was not refused"
        );
    }
}

/// A curve this crate does not implement is refused by name, and a compressed point is refused
/// by encoding. Neither is approximated with the nearest thing that does work.
#[test]
fn unsupported_curves_and_encodings_fail_closed() -> Result<()> {
    assert!(matches!(
        require_named_curve(P521_NAMED_CURVE_URI),
        Err(EncError::UnsupportedAlgorithm(_))
    ));

    let mut rng = PseudoRandom::new(5);
    let key = EcPrivateKey::generate(CurveId::P256, &mut rng)?;
    let mut compressed = key.public_key().to_sec1()?;
    compressed[0] = 0x02;
    compressed.truncate(33);
    assert!(matches!(
        EcPublicKey::from_sec1(CurveId::P256, &compressed),
        Err(EncError::InvalidAgreementKey(_))
    ));
    Ok(())
}

/// A `ds:KeyInfo` whose only child is an `xenc:AgreementMethod` resolves to no key at all. The
/// element round-trips verbatim, but nothing in the resolver guesses at what it means.
#[test]
fn an_agreement_method_in_a_key_info_does_not_resolve_by_itself() -> Result<()> {
    let document = oxixml_dom::build::parse_str(
        r#"<xenc:EncryptedData xmlns:xenc="http://www.w3.org/2001/04/xmlenc#"
                                xmlns:ds="http://www.w3.org/2000/09/xmldsig#">
             <xenc:EncryptionMethod Algorithm="http://www.w3.org/2009/xmlenc11#aes256-gcm"/>
             <ds:KeyInfo>
               <xenc:AgreementMethod Algorithm="http://www.w3.org/2009/xmlenc11#ECDH-ES">
                 <xenc:OriginatorKeyInfo/>
                 <xenc:RecipientKeyInfo/>
               </xenc:AgreementMethod>
             </ds:KeyInfo>
             <xenc:CipherData><xenc:CipherValue>AAECAw==</xenc:CipherValue></xenc:CipherData>
           </xenc:EncryptedData>"#,
    )?;
    let root = document
        .document_element()
        .ok_or_else(|| EncError::InvalidStructure("no document element".to_owned()))?;

    let mut document = document;
    assert!(matches!(
        oxixml_enc::decrypt(&mut document, root, &KeyStore::new()),
        Err(EncError::KeyNotFound(_))
    ));
    // The element survived the attempt untouched.
    assert!(document.to_xml().contains("AgreementMethod"));
    Ok(())
}

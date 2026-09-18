//! Known-answer tests for RSA key transport, MGF1 and the OAEP encoding.
//!
//! # Why these vectors look different from the others
//!
//! RSAES-OAEP and RSAES-PKCS1-v1_5 are *randomized*: the same key and message give a different
//! ciphertext every time, so there is no published "encrypt this and compare" vector the way
//! there is for AES. The check that actually proves interoperability is the other direction —
//! **decrypt a ciphertext this crate did not produce**. The ciphertexts below were produced by
//! an independent, widely deployed implementation for a freshly generated 1024-bit key, so a
//! successful decryption exercises the whole path (`RSADP`, the `EM` layout, MGF1, the label
//! hash, the separator scan) against a foreign encoder.
//!
//! The MGF1 vectors *are* deterministic, and are checked directly.
//!
//! A 1024-bit key is used because [`oxixml_support::bigint`] is schoolbook arithmetic and a
//! 2048-bit private-key operation would make the suite slow for no extra coverage; the code
//! path does not depend on the modulus size.

use oxixml_enc::EncError;
use oxixml_enc::crypto::digest::Digest;
use oxixml_enc::crypto::mgf1::{self, mgf1};
use oxixml_enc::crypto::rsa::{OaepParams, RsaPrivateKey};
use oxixml_enc::random::{FixedRandom, PseudoRandom};
use oxixml_text::hex;

/// Decodes a hex string from a test vector.
fn unhex(text: &str) -> Vec<u8> {
    hex::decode(text).expect("the embedded test vectors are valid hex")
}

/// The modulus of the 1024-bit test key.
const MODULUS: &str = concat!(
    "B81C0417F43CDFB4E4975A5A4DBA570A153A435BE3AD3136F27C1F0A9F21C8CA",
    "5222F013827F1772BA91F97F8A2AA9C6C063336E875B91FBBE5731E0E72A1175",
    "02FAE4E912FBB0400B0D3C79F7265DE81AF04761486DDDCCC5572A6816692180",
    "E8E3DB2A1EAD6F75B1CFDA98A7F2A53AA2C71594E6FD83B9CBF8151DABF06B75",
);

/// The private exponent of the 1024-bit test key.
const PRIVATE_EXPONENT: &str = concat!(
    "9680A0800C726ED56B40A6E5C7409AC26D53F1987E68282128DC7A6BAE6669C3",
    "AA9D50F041EF2EF5E4BA4BE10CCB8F7A1FEF782DD48CF0AB808FBB0085380887",
    "E0AEC90D2D5608B5F469E170D0C5A8A9A04C53A2C867E39BFFECFAFDB134192D",
    "A043814CCC4745AFF78F2A008A3784963133986CACABDB9555598B2C4F94B409",
);

/// The 256-bit content-encryption key every foreign ciphertext below carries.
const TRANSPORTED_KEY: &str = "000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F";

/// The test key pair.
fn test_key() -> Result<RsaPrivateKey, EncError> {
    RsaPrivateKey::new(
        &unhex(MODULUS),
        &[0x01, 0x00, 0x01],
        &unhex(PRIVATE_EXPONENT),
    )
}

#[test]
fn a_foreign_oaep_sha1_ciphertext_decrypts() -> Result<(), EncError> {
    let ciphertext = unhex(concat!(
        "8684C5E3358B119C5DD539CC8B79DBC3EECFEE3B511104B81FF7D2176CC18F7A",
        "6758382A2CE701340B00A4BBCB8BA11AA3212AE66EB05DFB1F24D9B7889EFE14",
        "73AF0D2933A2730A01DEB8749D44F914614D77B4B04D026388A1366C5302BF3C",
        "9BCB06EDC98CE4A4DCCB184332AE9B589C90DE486B171E6E151B48AB961A27AA",
    ));
    let recovered = test_key()?.decrypt_oaep(&ciphertext, &OaepParams::sha1())?;
    assert_eq!(recovered, unhex(TRANSPORTED_KEY));
    Ok(())
}

#[test]
fn a_foreign_oaep_sha256_ciphertext_decrypts() -> Result<(), EncError> {
    let ciphertext = unhex(concat!(
        "79FBDAEB980FA18A4BE04C2217C4821B1748B337F59B162604E0BA2EF9B90550",
        "B5CA05479DAD1970E49C9C2D0664D7580D5737B5E9210AAF318E31DD3EF31A20",
        "DC6C2FE0FF2DAF6E672632351CFD01A2D9547A30039E8392F58FD8DA2B19FE38",
        "723FE05C68DC2D2ADD3E36625AC4CA94471A84E9EA3BEA8C60473585CB20F68C",
    ));
    let recovered = test_key()?.decrypt_oaep(&ciphertext, &OaepParams::new(Digest::Sha256))?;
    assert_eq!(recovered, unhex(TRANSPORTED_KEY));
    Ok(())
}

#[test]
fn a_foreign_pkcs1_v1_5_ciphertext_decrypts() -> Result<(), EncError> {
    let ciphertext = unhex(concat!(
        "6F93D0B6B4AAF38E89CE0C21025555CA9171E81882F28758CB4958CE7AEDC4C1",
        "9638D5BE47D18C4227DFE638537A3478678D3B4806830654F9465E6065034A99",
        "582D8FEDA9A99F0A38AB046240AC238ED46DD57644C6EFD96DACC290FDE943AC",
        "C8BD66FCAED8428F57452477E6CE0B471A7C32C2525E41AEDC1762D480A05BBE",
    ));
    let recovered = test_key()?.decrypt_pkcs1v15(&ciphertext)?;
    assert_eq!(recovered, unhex(TRANSPORTED_KEY));
    Ok(())
}

#[test]
fn the_foreign_ciphertexts_are_bound_to_their_parameters() -> Result<(), EncError> {
    // The SHA-1 ciphertext must not decrypt under SHA-256 parameters, or the digest is being
    // ignored somewhere.
    let ciphertext = unhex(concat!(
        "8684C5E3358B119C5DD539CC8B79DBC3EECFEE3B511104B81FF7D2176CC18F7A",
        "6758382A2CE701340B00A4BBCB8BA11AA3212AE66EB05DFB1F24D9B7889EFE14",
        "73AF0D2933A2730A01DEB8749D44F914614D77B4B04D026388A1366C5302BF3C",
        "9BCB06EDC98CE4A4DCCB184332AE9B589C90DE486B171E6E151B48AB961A27AA",
    ));
    let key = test_key()?;
    assert!(matches!(
        key.decrypt_oaep(&ciphertext, &OaepParams::new(Digest::Sha256)),
        Err(EncError::DecryptionFailed)
    ));
    assert!(matches!(
        key.decrypt_oaep(&ciphertext, &OaepParams::sha1().with_label(b"x".to_vec())),
        Err(EncError::DecryptionFailed)
    ));
    assert!(matches!(
        key.decrypt_pkcs1v15(&ciphertext),
        Err(EncError::DecryptionFailed)
    ));
    Ok(())
}

#[test]
fn mgf1_matches_independently_computed_masks() {
    // MGF1(Z, l) = Hash(Z ‖ 00000000) ‖ Hash(Z ‖ 00000001) ‖ …, truncated.
    assert_eq!(
        mgf1(Digest::Sha1, b"bar", 50),
        unhex(concat!(
            "BC0C655E016BC2931D85A2E675181ADCEF7F581F76DF2739DA74FAAC41627BE2",
            "F7F415C89E983FD0CE80CED9878641CB4876",
        )),
    );
    assert_eq!(
        mgf1(Digest::Sha256, b"bar", 50),
        unhex(concat!(
            "382576A7841021CC28FC4C0948753FB8312090CEA942EA4C4E735D10DC724B15",
            "5F9F6069F289D61DACA0CB814502EF04EAE1",
        )),
    );
}

#[test]
fn mgf1_uris_select_the_right_hash() {
    assert_eq!(
        mgf1::mgf1_digest_from_uri("http://www.w3.org/2009/xmlenc11#mgf1sha1"),
        Some(Digest::Sha1)
    );
    assert_eq!(
        mgf1::mgf1_digest_from_uri("http://www.w3.org/2009/xmlenc11#mgf1sha512"),
        Some(Digest::Sha512)
    );
    // The XMLEnc 1.0 spelling that Example 33 of the specification uses.
    assert_eq!(
        mgf1::mgf1_digest_from_uri("http://www.w3.org/2001/04/xmlenc#MGF1withSHA1"),
        Some(Digest::Sha1)
    );
}

#[test]
fn this_crates_own_oaep_ciphertexts_are_well_formed() -> Result<(), EncError> {
    // Encrypting the same key twice with a *fresh* seed must give different ciphertexts, and
    // both must decrypt: the seed has to reach the encoding.
    let key = test_key()?;
    let public = key.public_key();
    let params = OaepParams::new(Digest::Sha256);
    let content_key = unhex(TRANSPORTED_KEY);

    let mut rng = PseudoRandom::new(2026);
    let first = public.encrypt_oaep(&content_key, &params, &mut rng)?;
    let second = public.encrypt_oaep(&content_key, &params, &mut rng)?;
    assert_ne!(first, second);
    assert_eq!(first.len(), 128);
    assert_eq!(key.decrypt_oaep(&first, &params)?, content_key);
    assert_eq!(key.decrypt_oaep(&second, &params)?, content_key);
    Ok(())
}

#[test]
fn a_pinned_seed_gives_a_reproducible_oaep_ciphertext() -> Result<(), EncError> {
    // With the seed pinned, OAEP is a deterministic function; two runs must agree, which is
    // what lets a regression in the encoding show up as a changed ciphertext.
    let public = test_key()?.public_key();
    let params = OaepParams::sha1();
    let content_key = unhex(TRANSPORTED_KEY);
    let first = public.encrypt_oaep(&content_key, &params, &mut FixedRandom::new(&[0xa7]))?;
    let second = public.encrypt_oaep(&content_key, &params, &mut FixedRandom::new(&[0xa7]))?;
    assert_eq!(first, second);
    assert_eq!(test_key()?.decrypt_oaep(&first, &params)?, content_key);
    Ok(())
}

#[test]
fn an_all_zero_ciphertext_and_an_over_large_one_fail_opaquely() -> Result<(), EncError> {
    let key = test_key()?;
    assert!(matches!(
        key.decrypt_oaep(&[0u8; 128], &OaepParams::sha1()),
        Err(EncError::DecryptionFailed)
    ));
    // An integer that is not less than the modulus is rejected by `RSADP`.
    assert!(matches!(
        key.decrypt_oaep(&[0xffu8; 128], &OaepParams::sha1()),
        Err(EncError::DecryptionFailed)
    ));
    Ok(())
}

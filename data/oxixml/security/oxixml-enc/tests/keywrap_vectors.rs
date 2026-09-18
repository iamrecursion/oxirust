//! Known-answer tests for AES Key Wrap.
//!
//! All six vectors of [RFC 3394 §4](https://www.rfc-editor.org/rfc/rfc3394#section-4) are here —
//! every combination of key-encryption-key size and key-data size the document publishes — plus
//! the two vectors of [RFC 5649 §6](https://www.rfc-editor.org/rfc/rfc5649#section-6), which
//! cover the padded variant's two branches: the ordinary six-pass case and the degenerate
//! single-block case that is one AES encryption.

use oxixml_enc::EncError;
use oxixml_enc::crypto::aes::Aes;
use oxixml_enc::crypto::keywrap;
use oxixml_text::hex;

/// Decodes a hex string from a test vector.
fn unhex(text: &str) -> Vec<u8> {
    hex::decode(text).expect("the embedded test vectors are valid hex")
}

/// The 128-, 192- and 256-bit key-encryption keys RFC 3394 uses.
const KEK_128: &str = "000102030405060708090A0B0C0D0E0F";
const KEK_192: &str = "000102030405060708090A0B0C0D0E0F1011121314151617";
const KEK_256: &str = "000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F";

/// The 128-, 192- and 256-bit key data RFC 3394 wraps.
const KEY_128: &str = "00112233445566778899AABBCCDDEEFF";
const KEY_192: &str = "00112233445566778899AABBCCDDEEFF0001020304050607";
const KEY_256: &str = "00112233445566778899AABBCCDDEEFF000102030405060708090A0B0C0D0E0F";

#[test]
fn rfc_3394_section_4_vectors() -> Result<(), EncError> {
    let cases = [
        (
            "§4.1 wrap 128 bits with a 128-bit KEK",
            KEK_128,
            KEY_128,
            "1FA68B0A8112B447AEF34BD8FB5A7B829D3E862371D2CFE5",
        ),
        (
            "§4.2 wrap 128 bits with a 192-bit KEK",
            KEK_192,
            KEY_128,
            "96778B25AE6CA435F92B5B97C050AED2468AB8A17AD84E5D",
        ),
        (
            "§4.3 wrap 128 bits with a 256-bit KEK",
            KEK_256,
            KEY_128,
            "64E8C3F9CE0F5BA263E9777905818A2A93C8191E7D6E8AE7",
        ),
        (
            "§4.4 wrap 192 bits with a 192-bit KEK",
            KEK_192,
            KEY_192,
            "031D33264E15D33268F24EC260743EDCE1C6C7DDEE725A936BA814915C6762D2",
        ),
        (
            "§4.5 wrap 192 bits with a 256-bit KEK",
            KEK_256,
            KEY_192,
            "A8F9BC1612C68B3FF6E6F4FBE30E71E4769C8B80A32CB8958CD5D17D6B254DA1",
        ),
        (
            "§4.6 wrap 256 bits with a 256-bit KEK",
            KEK_256,
            KEY_256,
            "28C9F404C4B810F4CBCCB35CFB87F8263F5786E2D80ED326CBC7F0E71A99F43BFB988B9B7A02DD21",
        ),
    ];

    for (name, kek, key_data, expected) in cases {
        let cipher = Aes::new(&unhex(kek))?;
        let wrapped = keywrap::wrap(&cipher, &unhex(key_data))?;
        assert_eq!(wrapped, unhex(expected), "{name}");
        assert_eq!(
            keywrap::unwrap(&cipher, &wrapped)?,
            unhex(key_data),
            "{name} (unwrap)"
        );
    }
    Ok(())
}

#[test]
fn rfc_5649_section_6_vectors() -> Result<(), EncError> {
    // The RFC 5649 examples use a 192-bit key-encryption key.
    let kek = Aes::new(&unhex("5840df6e29b02af1ab493b705bf16ea1ae8338f4dcc176a8"))?;

    // §6, first example: 20 octets of key data, which pad to 24 and take the six-pass path.
    let key_data = unhex("c37b7e6492584340bed12207808941155068f738");
    let wrapped = keywrap::wrap_padded(&kek, &key_data)?;
    assert_eq!(
        wrapped,
        unhex("138bdeaa9b8fa7fc61f97742e72248ee5ae6ae5360d1ae6a5f54f373fa543b6a"),
        "RFC 5649 §6 first example"
    );
    assert_eq!(keywrap::unwrap_padded(&kek, &wrapped)?, key_data);

    // §6, second example: 7 octets, which pad to 8 and take the single-block path.
    let key_data = unhex("466f7250617369");
    let wrapped = keywrap::wrap_padded(&kek, &key_data)?;
    assert_eq!(
        wrapped,
        unhex("afbeb0f07dfbf5419200f2ccb50bb24f"),
        "RFC 5649 §6 second example"
    );
    assert_eq!(keywrap::unwrap_padded(&kek, &wrapped)?, key_data);
    Ok(())
}

#[test]
fn the_integrity_check_rejects_every_single_bit_flip() -> Result<(), EncError> {
    let kek = Aes::new(&unhex(KEK_256))?;
    let wrapped = keywrap::wrap(&kek, &unhex(KEY_256))?;
    for index in 0..wrapped.len() {
        for bit in 0..8u32 {
            let mut broken = wrapped.clone();
            broken[index] ^= 1 << bit;
            assert!(
                matches!(
                    keywrap::unwrap(&kek, &broken),
                    Err(EncError::DecryptionFailed)
                ),
                "flipping bit {bit} of octet {index} was not caught"
            );
        }
    }
    Ok(())
}

#[test]
fn wrapping_grows_the_key_material_by_exactly_one_semiblock() -> Result<(), EncError> {
    let kek = Aes::new(&unhex(KEK_128))?;
    for blocks in 2..12usize {
        let key_data = vec![0x5au8; blocks * 8];
        let wrapped = keywrap::wrap(&kek, &key_data)?;
        assert_eq!(wrapped.len(), key_data.len() + 8);
        assert_eq!(keywrap::unwrap(&kek, &wrapped)?, key_data);
    }
    Ok(())
}

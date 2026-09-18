//! Known-answer tests for AES-GCM.
//!
//! The vectors are the ones from McGrew and Viega's *The Galois/Counter Mode of Operation*,
//! which NIST SP 800-38D adopted as its own validation set and which every GCM implementation
//! is checked against. They are numbered here as in that document.
//!
//! The set is chosen to exercise each structurally different path:
//!
//! * cases 1, 7 and 13 have an empty plaintext, so the tag is `GHASH` over the length block
//!   alone — the case that catches a `GHASH` that mishandles the empty input;
//! * cases 2, 8 and 14 have exactly one block and no associated data;
//! * case 3 has four full blocks;
//! * case 4 has a *partial* final block plus associated data, so it exercises the zero-padding
//!   of both `A` and `C` inside `GHASH`;
//! * case 16 repeats case 4 with a 256-bit key.

use oxixml_enc::EncError;
use oxixml_enc::crypto::aes::Aes;
use oxixml_enc::crypto::gcm;
use oxixml_text::hex;

/// Decodes a hex string from a test vector.
fn unhex(text: &str) -> Vec<u8> {
    hex::decode(text).expect("the embedded test vectors are valid hex")
}

/// One published GCM vector.
struct Vector {
    name: &'static str,
    key: &'static str,
    iv: &'static str,
    plaintext: &'static str,
    aad: &'static str,
    ciphertext: &'static str,
    tag: &'static str,
}

/// The plaintext shared by cases 3, 4 and 16, minus its final four octets in the 60-octet cases.
const LONG_PLAINTEXT: &str = concat!(
    "d9313225f88406e5a55909c5aff5269a",
    "86a7a9531534f7da2e4c303d8a318a72",
    "1c3c0c95956809532fcf0e2449a6b525",
    "b16aedf5aa0de657ba637b39",
);

const VECTORS: &[Vector] = &[
    Vector {
        name: "test case 1 (AES-128, empty)",
        key: "00000000000000000000000000000000",
        iv: "000000000000000000000000",
        plaintext: "",
        aad: "",
        ciphertext: "",
        tag: "58e2fccefa7e3061367f1d57a4e7455a",
    },
    Vector {
        name: "test case 2 (AES-128, one block)",
        key: "00000000000000000000000000000000",
        iv: "000000000000000000000000",
        plaintext: "00000000000000000000000000000000",
        aad: "",
        ciphertext: "0388dace60b6a392f328c2b971b2fe78",
        tag: "ab6e47d42cec13bdf53a67b21257bddf",
    },
    Vector {
        name: "test case 3 (AES-128, four blocks)",
        key: "feffe9928665731c6d6a8f9467308308",
        iv: "cafebabefacedbaddecaf888",
        plaintext: concat!(
            "d9313225f88406e5a55909c5aff5269a",
            "86a7a9531534f7da2e4c303d8a318a72",
            "1c3c0c95956809532fcf0e2449a6b525",
            "b16aedf5aa0de657ba637b391aafd255",
        ),
        aad: "",
        ciphertext: concat!(
            "42831ec2217774244b7221b784d0d49c",
            "e3aa212f2c02a4e035c17e2329aca12e",
            "21d514b25466931c7d8f6a5aac84aa05",
            "1ba30b396a0aac973d58e091473f5985",
        ),
        tag: "4d5c2af327cd64a62cf35abd2ba6fab4",
    },
    Vector {
        name: "test case 4 (AES-128, partial block plus associated data)",
        key: "feffe9928665731c6d6a8f9467308308",
        iv: "cafebabefacedbaddecaf888",
        plaintext: LONG_PLAINTEXT,
        aad: "feedfacedeadbeeffeedfacedeadbeefabaddad2",
        ciphertext: concat!(
            "42831ec2217774244b7221b784d0d49c",
            "e3aa212f2c02a4e035c17e2329aca12e",
            "21d514b25466931c7d8f6a5aac84aa05",
            "1ba30b396a0aac973d58e091",
        ),
        tag: "5bc94fbc3221a5db94fae95ae7121a47",
    },
    Vector {
        name: "test case 7 (AES-192, empty)",
        key: "000000000000000000000000000000000000000000000000",
        iv: "000000000000000000000000",
        plaintext: "",
        aad: "",
        ciphertext: "",
        tag: "cd33b28ac773f74ba00ed1f312572435",
    },
    Vector {
        name: "test case 8 (AES-192, one block)",
        key: "000000000000000000000000000000000000000000000000",
        iv: "000000000000000000000000",
        plaintext: "00000000000000000000000000000000",
        aad: "",
        ciphertext: "98e7247c07f0fe411c267e4384b0f600",
        tag: "2ff58d80033927ab8ef4d4587514f0fb",
    },
    Vector {
        name: "test case 13 (AES-256, empty)",
        key: "0000000000000000000000000000000000000000000000000000000000000000",
        iv: "000000000000000000000000",
        plaintext: "",
        aad: "",
        ciphertext: "",
        tag: "530f8afbc74536b9a963b4f1c4cb738b",
    },
    Vector {
        name: "test case 14 (AES-256, one block)",
        key: "0000000000000000000000000000000000000000000000000000000000000000",
        iv: "000000000000000000000000",
        plaintext: "00000000000000000000000000000000",
        aad: "",
        ciphertext: "cea7403d4d606b6e074ec5d3baf39d18",
        tag: "d0d1c8a799996bf0265b98b5d48ab919",
    },
    Vector {
        name: "test case 16 (AES-256, partial block plus associated data)",
        key: "feffe9928665731c6d6a8f9467308308feffe9928665731c6d6a8f9467308308",
        iv: "cafebabefacedbaddecaf888",
        plaintext: LONG_PLAINTEXT,
        aad: "feedfacedeadbeeffeedfacedeadbeefabaddad2",
        ciphertext: concat!(
            "522dc1f099567d07f47f37a32a84427d",
            "643a8cdcbfe5c0c97598a2bd2555d1aa",
            "8cb08e48590dbb3da7b08b1056828838",
            "c5f61e6393ba7a0abcc9f662",
        ),
        tag: "76fc6ece0f4e1768cddf8853bb2d551b",
    },
];

#[test]
fn published_gcm_vectors_encrypt_correctly() -> Result<(), EncError> {
    for vector in VECTORS {
        let aes = Aes::new(&unhex(vector.key))?;
        let (ciphertext, tag) = gcm::encrypt(
            &aes,
            &unhex(vector.iv),
            &unhex(vector.plaintext),
            &unhex(vector.aad),
        )?;
        assert_eq!(
            ciphertext,
            unhex(vector.ciphertext),
            "ciphertext of {}",
            vector.name
        );
        assert_eq!(tag.to_vec(), unhex(vector.tag), "tag of {}", vector.name);
    }
    Ok(())
}

#[test]
fn published_gcm_vectors_decrypt_correctly() -> Result<(), EncError> {
    for vector in VECTORS {
        let aes = Aes::new(&unhex(vector.key))?;
        let plaintext = gcm::decrypt(
            &aes,
            &unhex(vector.iv),
            &unhex(vector.ciphertext),
            &unhex(vector.aad),
            &unhex(vector.tag),
        )?;
        assert_eq!(plaintext, unhex(vector.plaintext), "{}", vector.name);
    }
    Ok(())
}

#[test]
fn every_single_bit_flip_is_caught() -> Result<(), EncError> {
    // Authenticity is the whole point of GCM, so check it exhaustively on one vector rather
    // than at a handful of positions.
    let vector = &VECTORS[3];
    let aes = Aes::new(&unhex(vector.key))?;
    let iv = unhex(vector.iv);
    let aad = unhex(vector.aad);
    let ciphertext = unhex(vector.ciphertext);
    let tag = unhex(vector.tag);

    for index in 0..ciphertext.len() {
        for bit in 0..8u32 {
            let mut broken = ciphertext.clone();
            broken[index] ^= 1 << bit;
            assert!(
                matches!(
                    gcm::decrypt(&aes, &iv, &broken, &aad, &tag),
                    Err(EncError::DecryptionFailed)
                ),
                "flipping bit {bit} of ciphertext octet {index} was not caught"
            );
        }
    }
    for index in 0..tag.len() {
        let mut broken = tag.clone();
        broken[index] ^= 0x80;
        assert!(matches!(
            gcm::decrypt(&aes, &iv, &ciphertext, &aad, &broken),
            Err(EncError::DecryptionFailed)
        ));
    }
    for index in 0..aad.len() {
        let mut broken = aad.clone();
        broken[index] ^= 0x01;
        assert!(matches!(
            gcm::decrypt(&aes, &iv, &ciphertext, &broken, &tag),
            Err(EncError::DecryptionFailed)
        ));
    }
    Ok(())
}

#[test]
fn a_different_iv_gives_a_different_ciphertext() -> Result<(), EncError> {
    // The catastrophic GCM failure is nonce reuse; this at least pins that the nonce reaches
    // the keystream.
    let aes = Aes::new(&unhex(VECTORS[2].key))?;
    let plaintext = unhex(LONG_PLAINTEXT);
    let (first, first_tag) =
        gcm::encrypt(&aes, &unhex("cafebabefacedbaddecaf888"), &plaintext, &[])?;
    let (second, second_tag) =
        gcm::encrypt(&aes, &unhex("cafebabefacedbaddecaf889"), &plaintext, &[])?;
    assert_ne!(first, second);
    assert_ne!(first_tag, second_tag);
    Ok(())
}

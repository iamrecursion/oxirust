//! A pinned 1024-bit RSA key pair used by the unit tests.
//!
//! Generated once with a throw-away script and committed verbatim so that the
//! tests are deterministic and need no key generation at run time. It is a
//! **test** key: the private exponent is right here in the source, and nothing
//! outside `cfg(test)` can reach it.

use oxixml_support::bigint::BigUint;

/// `n`, base64 as it would appear in `<ds:Modulus>`.
pub(crate) const MODULUS_BASE64: &str = "jkcoCslxEus5u8QIPMAYjDIpNgX12fbWalSPmhbKjlNeR5ukRjB0xBJqYvncXrPl7lHzDemgFsdEbJCvTgF1imtaMkc6yfaHkJ4+RgzPR2SBU4KEaVq69fNR5Dco0vl/VUZD54drPqBBvXhiqwpAK3ZLHcPSLxzRQi1a20l+tlU=";

/// `e = 65537`, base64 as it would appear in `<ds:Exponent>`.
pub(crate) const PUBLIC_EXPONENT_BASE64: &str = "AQAB";

/// `d`, base64.
pub(crate) const PRIVATE_EXPONENT_BASE64: &str = "JQjKmHw07arC/WgaiM2r1hpcdc55X0IML+NQ+dgC46c5vEP8L7T/P+ZXQOou/YuwRyeXr8BnRmaaHfRPXWutz3Ipph9dTRcvUmLxeSPEBHRpnqGt+pSoAn3bKaAz6Yj2dt/HAqU1zzt57XRNabcAHisCkA9OCvlwqeTZ+fcv+OE=";

/// The same public key as a DER `SubjectPublicKeyInfo`, base64 — the payload of
/// a `<dsig11:DEREncodedKeyValue>`.
pub(crate) const SPKI_BASE64: &str = "MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQCORygKyXES6zm7xAg8wBiMMik2BfXZ9tZqVI+aFsqOU15Hm6RGMHTEEmpi+dxes+XuUfMN6aAWx0RskK9OAXWKa1oyRzrJ9oeQnj5GDM9HZIFTgoRpWrr181HkNyjS+X9VRkPnh2s+oEG9eGKrCkArdksdw9IvHNFCLVrbSX62VQIDAQAB";

/// The same public key as a bare PKCS#1 `RSAPublicKey`, base64.
pub(crate) const RSA_PUBLIC_KEY_BASE64: &str = "MIGJAoGBAI5HKArJcRLrObvECDzAGIwyKTYF9dn21mpUj5oWyo5TXkebpEYwdMQSamL53F6z5e5R8w3poBbHRGyQr04BdYprWjJHOsn2h5CePkYMz0dkgVOChGlauvXzUeQ3KNL5f1VGQ+eHaz6gQb14YqsKQCt2Sx3D0i8c0UItWttJfrZVAgMBAAE=";

/// A PKCS#1 `RSAPrivateKey`, base64.
pub(crate) const RSA_PRIVATE_KEY_BASE64: &str = "MIICXQIBAAKBgQCORygKyXES6zm7xAg8wBiMMik2BfXZ9tZqVI+aFsqOU15Hm6RGMHTEEmpi+dxes+XuUfMN6aAWx0RskK9OAXWKa1oyRzrJ9oeQnj5GDM9HZIFTgoRpWrr181HkNyjS+X9VRkPnh2s+oEG9eGKrCkArdksdw9IvHNFCLVrbSX62VQIDAQABAoGAJQjKmHw07arC/WgaiM2r1hpcdc55X0IML+NQ+dgC46c5vEP8L7T/P+ZXQOou/YuwRyeXr8BnRmaaHfRPXWutz3Ipph9dTRcvUmLxeSPEBHRpnqGt+pSoAn3bKaAz6Yj2dt/HAqU1zzt57XRNabcAHisCkA9OCvlwqeTZ+fcv+OECQQD4ge8oqmftl71XHon1dv/tzC4Eze4NI9zmyuIqNB3dkBlGjbwFSH7HrebRJnGUoFP2b4T3vgbSZ1s35gAmvFTZAkEAkpFOWB+djzIpWj6h4sYcNTO6R0Yygse/Www+4eiSKQkQPE+Ao2Ybpyhr2Moc6RLhhaOyGbyxo9/R9Oc1yf7P3QJBAOeZgOBvYAhoQIeu6Hgh/YcLoV/M5CzO3ZMiRgF7LNJO959vfGmYFL/X7KPDplsTyOyFJnEVYpo4xSSlcEKphpECQEjyfRWrFpYwlxhNV/EwPNXV2v6iV31zwS+llp6FTEmUMzpxL0skBNtbaOHLJjxyo0AKSxg4LAmt2ieT89eHNRkCQQDKgurVmoAiN0hlXcoymZmIU2g1GbggfWb5nSmldq3TJIbq1dXikhd8Q/Dit2XW9Ncd+JadjU1tka+Rzo6y20pg";

/// A PKCS#8 `PrivateKeyInfo` wrapping the same key, base64.
pub(crate) const PKCS8_PRIVATE_KEY_BASE64: &str = "MIICdwIBADANBgkqhkiG9w0BAQEFAASCAmEwggJdAgEAAoGBAI5HKArJcRLrObvECDzAGIwyKTYF9dn21mpUj5oWyo5TXkebpEYwdMQSamL53F6z5e5R8w3poBbHRGyQr04BdYprWjJHOsn2h5CePkYMz0dkgVOChGlauvXzUeQ3KNL5f1VGQ+eHaz6gQb14YqsKQCt2Sx3D0i8c0UItWttJfrZVAgMBAAECgYAlCMqYfDTtqsL9aBqIzavWGlx1znlfQgwv41D52ALjpzm8Q/wvtP8/5ldA6i79i7BHJ5evwGdGZpod9E9da63PcimmH11NFy9SYvF5I8QEdGmeoa36lKgCfdspoDPpiPZ238cCpTXPO3ntdE1ptwAeKwKQD04K+XCp5Nn59y/44QJBAPiB7yiqZ+2XvVceifV2/+3MLgTN7g0j3ObK4io0Hd2QGUaNvAVIfset5tEmcZSgU/ZvhPe+BtJnWzfmACa8VNkCQQCSkU5YH52PMilaPqHixhw1M7pHRjKCx79bDD7h6JIpCRA8T4CjZhunKGvYyhzpEuGFo7IZvLGj39H05zXJ/s/dAkEA55mA4G9gCGhAh67oeCH9hwuhX8zkLM7dkyJGAXss0k73n298aZgUv9fso8OmWxPI7IUmcRVimjjFJKVwQqmGkQJASPJ9FasWljCXGE1X8TA81dXa/qJXfXPBL6WWnoVMSZQzOnEvSyQE21to4csmPHKjQApLGDgsCa3aJ5Pz14c1GQJBAMqC6tWagCI3SGVdyjKZmYhTaDUZuCB9ZvmdKaV2rdMkhurV1eKSF3xD8OK3Zdb01x34lp2NTW2Rr5HOjrLbSmA=";

/// A self-signed X.509 v3 certificate carrying the same public key, base64 —
/// the payload of a `<ds:X509Certificate>`.
pub(crate) const CERTIFICATE_BASE64: &str = "MIIBpDCCAQ2gAwIBAgIFASNFZ4kwDQYJKoZIhvcNAQELBQAwFjEUMBIGA1UEAwwLT3hpWE1MIFRlc3QwHhcNMjYwMTAxMDAwMDAwWhcNMzYwMTAxMDAwMDAwWjAWMRQwEgYDVQQDDAtPeGlYTUwgVGVzdDCBnzANBgkqhkiG9w0BAQEFAAOBjQAwgYkCgYEAjkcoCslxEus5u8QIPMAYjDIpNgX12fbWalSPmhbKjlNeR5ukRjB0xBJqYvncXrPl7lHzDemgFsdEbJCvTgF1imtaMkc6yfaHkJ4+RgzPR2SBU4KEaVq69fNR5Dco0vl/VUZD54drPqBBvXhiqwpAK3ZLHcPSLxzRQi1a20l+tlUCAwEAATANBgkqhkiG9w0BAQsFAAOBgQAbgJ1S/Y0svcUZpowjNkA5BxrXfJNopeDC6Rxz77AHzTfqnwkVWkjAYxY8mfbeCQkEucKK2FxB8bv1WzDE/rlW5h8jtyxmvYA3d/imKCjC7SPcibgHh55nIkmwA9oXpLL5C4rM0fcApbudoveWWwRHWAlxDiusBrr8vbzaDOByHA==";

/// Decodes one of the base64 constants above.
pub(crate) fn bytes(base64: &str) -> Vec<u8> {
    oxixml_text::base64::base64_decode(base64).expect("the pinned test key is valid base64")
}

/// Decodes a hexadecimal string, ignoring ASCII whitespace.
///
/// The published test vectors for ECDSA, DSA and RSASSA-PSS are all printed in
/// hexadecimal, so they are transcribed that way rather than converted to
/// base64, which would make them impossible to check against their source.
pub(crate) fn hex(text: &str) -> Vec<u8> {
    let digits: Vec<u32> = text
        .chars()
        .filter(|character| !character.is_ascii_whitespace())
        .map(|character| {
            character
                .to_digit(16)
                .expect("the transcribed vector is hexadecimal")
        })
        .collect();
    assert!(
        digits.len() % 2 == 0,
        "a hexadecimal vector has an even number of digits"
    );
    digits
        .chunks(2)
        .map(|pair| {
            let high = pair.first().copied().unwrap_or(0);
            let low = pair.get(1).copied().unwrap_or(0);
            u8::try_from((high << 4) | low).unwrap_or(0)
        })
        .collect()
}

/// `n` as a big integer.
pub(crate) fn modulus() -> BigUint {
    BigUint::from_be_bytes(&bytes(MODULUS_BASE64))
}

/// `e` as a big integer.
pub(crate) fn public_exponent() -> BigUint {
    BigUint::from_be_bytes(&bytes(PUBLIC_EXPONENT_BASE64))
}

/// `d` as a big integer.
pub(crate) fn private_exponent() -> BigUint {
    BigUint::from_be_bytes(&bytes(PRIVATE_EXPONENT_BASE64))
}

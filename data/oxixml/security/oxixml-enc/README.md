# oxixml-enc

XML Encryption 1.1, in Pure Rust, with its ciphers implemented from the standards rather than
pulled in.

`oxixml-enc` models `xenc:EncryptedData` and `xenc:EncryptedKey` completely and losslessly,
encrypts and decrypts nodes of an `oxixml-dom` tree with the namespace and entity handling that
makes a decrypted fragment mean what it did before encryption, and implements the primitives
itself: AES (FIPS 197), CBC/CTR (SP 800-38A), GCM (SP 800-38D), AES Key Wrap (RFC 3394/5649), and
RSA with OAEP and PKCS#1 v1.5 (RFC 8017). No external crates, no `unsafe`, no C. AES-GCM is the
default; CBC-mode decryption failures are reported as one opaque error to resist the
Jager–Somorovsky adaptive chosen-ciphertext attack.

Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace, the Pure-Rust foundation
for XML and RDF processing in the COOLJAPAN ecosystem.

## Install

```toml
[dependencies]
oxixml-enc = "0.1.2"
```

## Example

```rust
use oxixml_dom::build;
use oxixml_enc::algorithm::{BlockEncryption, KeyWrap};
use oxixml_enc::key::{KeyStore, SymmetricKey};
use oxixml_enc::model::KeyInfo;
use oxixml_enc::random::SystemRandom;
use oxixml_enc::{EncryptOptions, decrypt, encrypt_element, wrap_key_for_key_info};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut doc = build::parse_str(
        r#"<Order xmlns="urn:example:orders">
             <Customer>A. Nonymous</Customer>
             <Payment><Card>4019 2445 0277 5567</Card></Payment>
           </Order>"#,
    )?;
    let root = doc.document_element().expect("a document element");
    let payment = doc
        .get_elements_by_tag_name_ns(root, Some("urn:example:orders"), "Payment")
        .next()
        .expect("a Payment element");

    let mut rng = SystemRandom::new();

    // A fresh content-encryption key, wrapped under a long-lived key-encryption key.
    let key_encryption_key = SymmetricKey::new(vec![0x5c; 32]);
    let content_key = SymmetricKey::generate(32, &mut rng)?;
    let carried =
        wrap_key_for_key_info(KeyWrap::Aes256, &key_encryption_key, &content_key, Some("kek"))?;

    let options = EncryptOptions::new()
        .with_algorithm(BlockEncryption::Aes256Gcm)
        .with_key_info(KeyInfo::with_encrypted_key(carried));
    let encrypted = encrypt_element(&mut doc, payment, &content_key, &options, &mut rng)?;
    assert!(!doc.to_xml().contains("4019"));

    // The recipient holds only the key-encryption key.
    let store = KeyStore::new().with_named_key_encryption_key("kek", key_encryption_key);
    decrypt(&mut doc, encrypted, &store)?;
    assert!(doc.to_xml().contains("4019 2445 0277 5567"));
    Ok(())
}
```

## What it replaces

`oxixml-enc` replaces xmlsec1 (C) or Apache Santuario (Java) for XML Encryption.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](../../LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).

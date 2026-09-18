//! End-to-end XML Encryption: encrypt a node, serialize the document, parse it back, decrypt,
//! and compare the tree with the one we started from.
//!
//! The serialize/re-parse step in the middle matters. It is easy to write a round trip that
//! only ever passes DOM nodes around and so never exercises the base64 encoding, the element
//! ordering, the namespace prefixes or the `Type` attribute. Every test here goes through XML
//! text, exactly as a document that travelled between two parties would.

use oxixml_dom::{Document, NodeId, build};
use oxixml_enc::algorithm::{BlockEncryption, KeyTransport, KeyWrap};
use oxixml_enc::crypto::digest::Digest;
use oxixml_enc::crypto::rsa::{OaepParams, RsaPrivateKey};
use oxixml_enc::key::{KeyStore, StaticKey, SymmetricKey};
use oxixml_enc::model::{EncryptedData, KeyInfo};
use oxixml_enc::random::{PseudoRandom, SystemRandom};
use oxixml_enc::{
    ContentType, EncError, EncryptOptions, decrypt, encrypt_content, encrypt_element,
    transport_key_for_key_info, wrap_key_for_key_info,
};
use oxixml_text::hex;

/// A document with several namespaces, mixed content and an entity, so that the fragment
/// handling of XMLEnc §4.5.3 is actually exercised rather than trivially satisfied.
const SAMPLE: &str = concat!(
    "<!DOCTYPE Order [<!ENTITY currency \"JPY\">]>",
    "<Order xmlns=\"urn:example:orders\" xmlns:m=\"urn:example:meta\" m:rev=\"3\">",
    "<Customer m:id=\"c1\">A. Nonymous</Customer>",
    "<Payment><Card issuer=\"Example Bank\">4019 2445 0277 5567</Card>",
    "<Amount>1200 &currency;</Amount><!-- reviewed --></Payment>",
    "<Note xmlns=\"\">plain, in no namespace</Note>",
    "</Order>",
);

/// Parses `xml`, failing the test with a readable message rather than an opaque unwrap.
fn parse(xml: &str) -> Result<Document, EncError> {
    Ok(build::parse_str(xml)?)
}

/// Returns the first element of the *live* tree with the given local name.
///
/// `nodes_in_document_order` deliberately also lists detached nodes — the arena keeps a removed
/// subtree alive so its identifiers stay valid — so the search has to check that the node is
/// still attached. Decryption detaches the `EncryptedData` it replaces, and without this filter
/// a second search would find that corpse.
fn find(doc: &Document, local: &str) -> Option<NodeId> {
    doc.nodes_in_document_order().into_iter().find(|node| {
        doc.tree_root(*node) == Some(doc.root())
            && doc
                .element(*node)
                .is_some_and(|data| data.local_name() == local)
    })
}

/// The content-encryption key used where the key itself is not what is being tested.
fn content_key(algorithm: BlockEncryption) -> SymmetricKey {
    SymmetricKey::new(vec![0x2b; algorithm.key_len()])
}

/// The 1024-bit RSA key shared with `rsa_vectors.rs`.
fn rsa_key() -> Result<RsaPrivateKey, EncError> {
    let modulus = hex::decode(concat!(
        "B81C0417F43CDFB4E4975A5A4DBA570A153A435BE3AD3136F27C1F0A9F21C8CA",
        "5222F013827F1772BA91F97F8A2AA9C6C063336E875B91FBBE5731E0E72A1175",
        "02FAE4E912FBB0400B0D3C79F7265DE81AF04761486DDDCCC5572A6816692180",
        "E8E3DB2A1EAD6F75B1CFDA98A7F2A53AA2C71594E6FD83B9CBF8151DABF06B75",
    ))
    .expect("valid hex");
    let private = hex::decode(concat!(
        "9680A0800C726ED56B40A6E5C7409AC26D53F1987E68282128DC7A6BAE6669C3",
        "AA9D50F041EF2EF5E4BA4BE10CCB8F7A1FEF782DD48CF0AB808FBB0085380887",
        "E0AEC90D2D5608B5F469E170D0C5A8A9A04C53A2C867E39BFFECFAFDB134192D",
        "A043814CCC4745AFF78F2A008A3784963133986CACABDB9555598B2C4F94B409",
    ))
    .expect("valid hex");
    RsaPrivateKey::new(&modulus, &[0x01, 0x00, 0x01], &private)
}

#[test]
fn element_encryption_round_trips_through_serialized_xml() -> Result<(), EncError> {
    for algorithm in BlockEncryption::ALL {
        let original = parse(SAMPLE)?;
        let mut doc = parse(SAMPLE)?;
        let payment = find(&doc, "Payment").expect("a Payment element");

        let key = content_key(algorithm);
        let options = EncryptOptions::new().with_algorithm(algorithm);
        encrypt_element(&mut doc, payment, &key, &options, &mut SystemRandom::new())?;

        // Everything sensitive is gone, and the rest of the document is untouched.
        let ciphered = doc.to_xml();
        assert!(!ciphered.contains("4019"), "{algorithm:?}: {ciphered}");
        assert!(!ciphered.contains("Payment"), "{algorithm:?}: {ciphered}");
        assert!(!ciphered.contains("Example Bank"), "{algorithm:?}");
        assert!(ciphered.contains("A. Nonymous"), "{algorithm:?}");

        // Travel: the document is serialized and parsed by the recipient.
        let mut received = parse(&ciphered)?;
        let encrypted = find(&received, "EncryptedData").expect("an EncryptedData element");
        let outcome = decrypt(&mut received, encrypted, &StaticKey::new(key))?;
        assert_eq!(outcome.nodes().len(), 1, "{algorithm:?}");

        assert!(
            original.structural_eq(&received),
            "{algorithm:?}: {:?}",
            original.structural_diff(&received)
        );
    }
    Ok(())
}

#[test]
fn content_encryption_round_trips_through_serialized_xml() -> Result<(), EncError> {
    for algorithm in [BlockEncryption::Aes128Cbc, BlockEncryption::Aes256Gcm] {
        let original = parse(SAMPLE)?;
        let mut doc = parse(SAMPLE)?;
        let payment = find(&doc, "Payment").expect("a Payment element");

        let key = content_key(algorithm);
        let options = EncryptOptions::new().with_algorithm(algorithm);
        encrypt_content(&mut doc, payment, &key, &options, &mut SystemRandom::new())?;

        let ciphered = doc.to_xml();
        assert!(!ciphered.contains("4019"), "{algorithm:?}: {ciphered}");
        // Content encryption deliberately leaves the element itself visible.
        assert!(ciphered.contains("Payment"), "{algorithm:?}: {ciphered}");
        assert!(
            ciphered.contains("http://www.w3.org/2001/04/xmlenc#Content"),
            "{algorithm:?}: {ciphered}"
        );

        let mut received = parse(&ciphered)?;
        let encrypted = find(&received, "EncryptedData").expect("an EncryptedData element");
        decrypt(&mut received, encrypted, &StaticKey::new(key))?;

        assert!(
            original.structural_eq(&received),
            "{algorithm:?}: {:?}",
            original.structural_diff(&received)
        );
    }
    Ok(())
}

#[test]
fn an_element_in_no_namespace_stays_in_no_namespace() -> Result<(), EncError> {
    // XMLEnc §4.5.3.1's worked example: without the `xmlns=""` augmentation this element comes
    // back in `urn:example:orders`, silently corrupting the document.
    let original = parse(SAMPLE)?;
    let mut doc = parse(SAMPLE)?;
    let note = find(&doc, "Note").expect("a Note element");
    assert_eq!(
        doc.element(note).and_then(|data| data.namespace_uri()),
        None,
        "the fixture must really put Note in no namespace"
    );

    let key = content_key(BlockEncryption::Aes256Gcm);
    encrypt_element(
        &mut doc,
        note,
        &key,
        &EncryptOptions::new(),
        &mut SystemRandom::new(),
    )?;

    let mut received = parse(&doc.to_xml())?;
    let encrypted = find(&received, "EncryptedData").expect("an EncryptedData element");
    decrypt(&mut received, encrypted, &StaticKey::new(key))?;

    let restored = find(&received, "Note").expect("the Note element is back");
    assert_eq!(
        received
            .element(restored)
            .and_then(|data| data.namespace_uri()),
        None,
        "the element must not be pulled into the default namespace"
    );
    assert!(
        original.structural_eq(&received),
        "{:?}",
        original.structural_diff(&received)
    );
    Ok(())
}

#[test]
fn an_entity_reference_survives_the_round_trip() -> Result<(), EncError> {
    // The `&currency;` entity is expanded before encryption and must reappear as its
    // replacement text, not as an unresolved reference, thanks to the text wrapping of §4.5.4.
    let mut doc = parse(SAMPLE)?;
    let amount = find(&doc, "Amount").expect("an Amount element");
    assert_eq!(doc.text_content(amount).as_deref(), Some("1200 JPY"));

    let key = content_key(BlockEncryption::Aes256Gcm);
    encrypt_element(
        &mut doc,
        amount,
        &key,
        &EncryptOptions::new(),
        &mut SystemRandom::new(),
    )?;

    let mut received = parse(&doc.to_xml())?;
    let encrypted = find(&received, "EncryptedData").expect("an EncryptedData element");
    decrypt(&mut received, encrypted, &StaticKey::new(key))?;

    let restored = find(&received, "Amount").expect("the Amount element is back");
    assert_eq!(received.text_content(restored).as_deref(), Some("1200 JPY"));
    Ok(())
}

#[test]
fn a_wrapped_key_travels_inside_the_document() -> Result<(), EncError> {
    let original = parse(SAMPLE)?;
    let mut doc = parse(SAMPLE)?;
    let payment = find(&doc, "Payment").expect("a Payment element");

    let mut rng = SystemRandom::new();
    let key_encryption_key = SymmetricKey::new(vec![0x91; 32]);
    let cek = SymmetricKey::generate(32, &mut rng)?;
    let carried = wrap_key_for_key_info(KeyWrap::Aes256, &key_encryption_key, &cek, Some("kek"))?;

    let options = EncryptOptions::new()
        .with_algorithm(BlockEncryption::Aes256Gcm)
        .with_key_info(KeyInfo::with_encrypted_key(carried));
    encrypt_element(&mut doc, payment, &cek, &options, &mut rng)?;

    let ciphered = doc.to_xml();
    assert!(ciphered.contains("xenc:EncryptedKey"), "{ciphered}");
    assert!(ciphered.contains("kw-aes256"), "{ciphered}");
    assert!(ciphered.contains(">kek</ds:KeyName>"), "{ciphered}");

    // The recipient holds only the key-encryption key; the content key is in the document.
    let mut received = parse(&ciphered)?;
    let encrypted = find(&received, "EncryptedData").expect("an EncryptedData element");
    let store = KeyStore::new().with_named_key_encryption_key("kek", key_encryption_key);
    decrypt(&mut received, encrypted, &store)?;

    assert!(
        original.structural_eq(&received),
        "{:?}",
        original.structural_diff(&received)
    );
    Ok(())
}

#[test]
fn every_key_wrap_algorithm_carries_a_key() -> Result<(), EncError> {
    for algorithm in KeyWrap::ALL {
        let mut doc = parse("<a><b>payload</b></a>")?;
        let target = find(&doc, "b").expect("a b element");

        let mut rng = PseudoRandom::new(11);
        let kek = SymmetricKey::new(vec![0x33; algorithm.key_len()]);
        let cek = SymmetricKey::generate(16, &mut rng)?;
        let carried = wrap_key_for_key_info(algorithm, &kek, &cek, None)?;

        let options = EncryptOptions::new()
            .with_algorithm(BlockEncryption::Aes128Gcm)
            .with_key_info(KeyInfo::with_encrypted_key(carried));
        encrypt_element(&mut doc, target, &cek, &options, &mut rng)?;

        let mut received = parse(&doc.to_xml())?;
        let encrypted = find(&received, "EncryptedData").expect("an EncryptedData element");
        let store = KeyStore::new().with_default_key_encryption_key(kek);
        decrypt(&mut received, encrypted, &store)?;
        assert_eq!(received.to_xml(), "<a><b>payload</b></a>", "{algorithm:?}");
    }
    Ok(())
}

#[test]
fn rsa_key_transport_carries_a_key() -> Result<(), EncError> {
    let private = rsa_key()?;
    let public = private.public_key();

    for (algorithm, params) in [
        (KeyTransport::RsaOaepMgf1p, OaepParams::sha1()),
        (KeyTransport::RsaOaep, OaepParams::new(Digest::Sha256)),
        (KeyTransport::Rsa15, OaepParams::sha1()),
    ] {
        let mut doc = parse("<a><b>payload</b></a>")?;
        let target = find(&doc, "b").expect("a b element");

        let mut rng = PseudoRandom::new(12);
        let cek = SymmetricKey::generate(32, &mut rng)?;
        let carried = transport_key_for_key_info(
            algorithm,
            &public,
            &cek,
            &params,
            Some("recipient"),
            &mut rng,
        )?;

        let options = EncryptOptions::new()
            .with_algorithm(BlockEncryption::Aes256Gcm)
            .with_key_info(KeyInfo::with_encrypted_key(carried));
        encrypt_element(&mut doc, target, &cek, &options, &mut rng)?;

        let ciphered = doc.to_xml();
        assert!(
            ciphered.contains(algorithm.uri()),
            "{algorithm:?}: {ciphered}"
        );
        if algorithm.allows_mgf_element() {
            assert!(ciphered.contains("xenc11:MGF"), "{algorithm:?}: {ciphered}");
        } else {
            assert!(
                !ciphered.contains("xenc11:MGF"),
                "{algorithm:?}: {ciphered}"
            );
        }

        let mut received = parse(&ciphered)?;
        let encrypted = find(&received, "EncryptedData").expect("an EncryptedData element");
        let store = KeyStore::new().with_named_key_encryption_key("recipient", private.clone());
        decrypt(&mut received, encrypted, &store)?;
        assert_eq!(received.to_xml(), "<a><b>payload</b></a>", "{algorithm:?}");
    }
    Ok(())
}

#[test]
fn a_retrieval_method_finds_a_detached_encrypted_key() -> Result<(), EncError> {
    // The `EncryptedKey` sits elsewhere in the document and the `EncryptedData` points at it,
    // which is the layout WS-Security uses.
    let mut doc = parse("<Envelope><Keys/><Body><Secret>hidden</Secret></Body></Envelope>")?;

    let mut rng = PseudoRandom::new(13);
    let kek = SymmetricKey::new(vec![0x44; 32]);
    let cek = SymmetricKey::generate(32, &mut rng)?;

    let mut carried = wrap_key_for_key_info(KeyWrap::Aes256, &kek, &cek, Some("kek"))?;
    carried.common.id = Some("ek-1".to_owned());
    let keys = find(&doc, "Keys").expect("a Keys element");
    carried.write_into(&mut doc, Some(keys))?;

    let secret = find(&doc, "Secret").expect("a Secret element");
    let key_info = KeyInfo {
        id: None,
        items: vec![oxixml_enc::model::KeyInfoItem::RetrievalMethod {
            uri: "#ek-1".to_owned(),
            type_uri: Some("http://www.w3.org/2001/04/xmlenc#EncryptedKey".to_owned()),
        }],
    };
    let options = EncryptOptions::new().with_key_info(key_info);
    encrypt_element(&mut doc, secret, &cek, &options, &mut rng)?;

    let ciphered = doc.to_xml();
    assert!(!ciphered.contains("hidden"), "{ciphered}");
    assert!(ciphered.contains("ds:RetrievalMethod"), "{ciphered}");

    let mut received = parse(&ciphered)?;
    let encrypted = find(&received, "EncryptedData").expect("an EncryptedData element");
    let store = KeyStore::new().with_named_key_encryption_key("kek", kek);
    decrypt(&mut received, encrypted, &store)?;
    assert!(received.to_xml().contains("<Secret>hidden</Secret>"));
    Ok(())
}

#[test]
fn the_produced_structure_matches_the_schema_ordering() -> Result<(), EncError> {
    let mut doc = parse(SAMPLE)?;
    let payment = find(&doc, "Payment").expect("a Payment element");
    let key = content_key(BlockEncryption::Aes128Cbc);
    let options = EncryptOptions::new()
        .with_algorithm(BlockEncryption::Aes128Cbc)
        .with_id("ed-1")
        .with_key_info(KeyInfo::with_key_name("session"));
    let encrypted = encrypt_element(
        &mut doc,
        payment,
        &key,
        &options,
        &mut PseudoRandom::new(14),
    )?;

    // Re-parse the produced element through the model, which is the strictest check available.
    let model = EncryptedData::parse(&doc, encrypted)?;
    assert_eq!(model.common.id.as_deref(), Some("ed-1"));
    assert_eq!(model.common.content_type, Some(ContentType::Element));
    let method = model.common.encryption_method.as_ref().expect("a method");
    assert_eq!(method.block_encryption(), Some(BlockEncryption::Aes128Cbc));
    assert_eq!(method.key_size, Some(128));
    assert_eq!(
        model.common.key_info.as_ref().and_then(KeyInfo::key_name),
        Some("session")
    );

    // CBC framing: a 16-octet initialization vector plus whole blocks.
    let ciphertext = model.ciphertext()?;
    assert!(ciphertext.len() > 16);
    assert_eq!((ciphertext.len() - 16) % 16, 0);
    Ok(())
}

#[test]
fn a_tampered_gcm_ciphertext_is_rejected_after_travelling() -> Result<(), EncError> {
    let mut doc = parse(SAMPLE)?;
    let payment = find(&doc, "Payment").expect("a Payment element");
    let key = content_key(BlockEncryption::Aes256Gcm);
    encrypt_element(
        &mut doc,
        payment,
        &key,
        &EncryptOptions::new(),
        &mut SystemRandom::new(),
    )?;

    // Flip one base64 character in the CipherValue, which flips real ciphertext bits.
    let ciphered = doc.to_xml();
    let start = ciphered.find("<xenc:CipherValue").expect("a CipherValue");
    let value_start = start + ciphered[start..].find('>').expect("the tag end") + 1;
    let mut tampered = ciphered.clone();
    let victim = tampered.as_bytes()[value_start + 24];
    let replacement = if victim == b'A' { b'B' } else { b'A' };
    // Safe because base64 characters are all single-byte ASCII.
    tampered.replace_range(
        value_start + 24..value_start + 25,
        core::str::from_utf8(&[replacement]).expect("ASCII"),
    );

    let mut received = parse(&tampered)?;
    let encrypted = find(&received, "EncryptedData").expect("an EncryptedData element");
    assert!(matches!(
        decrypt(&mut received, encrypted, &StaticKey::new(key)),
        Err(EncError::DecryptionFailed)
    ));
    Ok(())
}

#[test]
fn nested_encryption_peels_off_in_order() -> Result<(), EncError> {
    // Encrypt an element, then encrypt the resulting EncryptedData. Two decryptions must
    // restore the original exactly.
    let original = parse("<a><b><c>deep</c></b></a>")?;
    let mut doc = parse("<a><b><c>deep</c></b></a>")?;

    let inner_key = SymmetricKey::new(vec![0x01; 32]);
    let outer_key = SymmetricKey::new(vec![0x02; 32]);
    let mut rng = SystemRandom::new();

    let target = find(&doc, "b").expect("a b element");
    let inner = encrypt_element(
        &mut doc,
        target,
        &inner_key,
        &EncryptOptions::new(),
        &mut rng,
    )?;
    let outer = encrypt_element(
        &mut doc,
        inner,
        &outer_key,
        &EncryptOptions::new(),
        &mut rng,
    )?;
    assert_eq!(outer, outer);

    let mut received = parse(&doc.to_xml())?;
    let node = find(&received, "EncryptedData").expect("the outer EncryptedData");
    let revealed = decrypt(&mut received, node, &StaticKey::new(outer_key))?;
    let inner_node = revealed
        .nodes()
        .first()
        .copied()
        .expect("the inner EncryptedData");
    decrypt(&mut received, inner_node, &StaticKey::new(inner_key))?;

    assert!(
        original.structural_eq(&received),
        "{:?}",
        original.structural_diff(&received)
    );
    Ok(())
}

#[test]
fn many_elements_can_be_encrypted_under_one_key() -> Result<(), EncError> {
    let original = parse(SAMPLE)?;
    let mut doc = parse(SAMPLE)?;
    let key = content_key(BlockEncryption::Aes256Gcm);
    let mut rng = SystemRandom::new();

    for local in ["Customer", "Payment", "Note"] {
        let target = find(&doc, local).expect("the element");
        encrypt_element(&mut doc, target, &key, &EncryptOptions::new(), &mut rng)?;
    }

    let ciphered = doc.to_xml();
    assert!(!ciphered.contains("Nonymous"), "{ciphered}");
    assert!(!ciphered.contains("4019"), "{ciphered}");

    let mut received = parse(&ciphered)?;
    // Decrypt repeatedly until none is left; each decryption invalidates the node list.
    while let Some(node) = find(&received, "EncryptedData") {
        decrypt(&mut received, node, &StaticKey::new(key.clone()))?;
    }
    assert!(
        original.structural_eq(&received),
        "{:?}",
        original.structural_diff(&received)
    );
    Ok(())
}

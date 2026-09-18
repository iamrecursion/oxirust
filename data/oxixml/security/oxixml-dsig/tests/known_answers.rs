//! Known-answer tests that need no external suite.
//!
//! Several of the W3C XML Signature 1.1 interoperability documents are embedded
//! here verbatim, so that the cryptographic path has independent coverage even
//! when `OXIXML_XML_SUITES_DIR` is not configured. All were produced by
//! Oracle's JSR-105 implementation for the
//! [XML Security 1.1 interop round](https://www.w3.org/TR/xmldsig-core1/); they
//! are reproduced for testing only.
//!
//! Together they pin down every step of verification against an implementation
//! that shares no code with this one:
//!
//! * Canonical XML 1.0 over a `<ds:SignedInfo>` subset, including the namespace
//!   declaration inherited from the enclosing `<ds:Signature>` and the
//!   attribute reordering (`Type` before `URI`);
//! * shortname-XPointer dereferencing of `URI="#DSig.Object_…"` against an `Id`
//!   attribute that no DTD declares;
//! * SHA-1 and SHA-256 digests over the canonical form of that subtree;
//! * HMAC-SHA-256, and RSASSA-PKCS1-v1_5 with its `DigestInfo` prefix and
//!   `0x00 0x01 ff… 0x00` padding, against a 1024-bit key given as
//!   `<ds:Modulus>`/`<ds:Exponent>`;
//! * ECDSA on P-256 and on P-384, with the key given as a
//!   `<dsig11:ECKeyValue>` and as a `<dsig11:DEREncodedKeyValue>`, and with the
//!   `r ‖ s` signature value of XML Signature §6.4.3 rather than ASN.1. The
//!   P-384 document uses SHA-1, whose digest is *narrower* than the group order,
//!   which is the case FIPS 186-4's truncation rule is easiest to get wrong.

mod common;

use oxixml_dsig::{
    DigestAlgorithm, DsigError, NamedCurve, SharedSecret, Signature, SignatureAlgorithm,
    UntrustedKeyInfo, Verifier,
};

use common::{INTEROP_HMAC_SECRET, parse, tamper_text};

/// `signature-enveloping-hmac-sha256.xml` from the Oracle interop set.
const HMAC_SHA256: &str = concat!(
    r##"<dsig:Signature xmlns:dsig="http://www.w3.org/2000/09/xmldsig#">"##,
    r##"<dsig:SignedInfo>"##,
    r##"<dsig:CanonicalizationMethod Algorithm="http://www.w3.org/TR/2001/REC-xml-c14n-20010315"/>"##,
    r##"<dsig:SignatureMethod Algorithm="http://www.w3.org/2001/04/xmldsig-more#hmac-sha256"/>"##,
    r##"<dsig:Reference URI="#DSig.Object_I08V3cMJvHneFuSSVRb87A22" Type="http://www.w3.org/2000/09/xmldsig#Object">"##,
    r##"<dsig:DigestMethod Algorithm="http://www.w3.org/2000/09/xmldsig#sha1"/>"##,
    r##"<dsig:DigestValue>myrT5qEfA7Wemy2WONCZG66c5QE=</dsig:DigestValue>"##,
    r##"</dsig:Reference></dsig:SignedInfo>"##,
    r##"<dsig:SignatureValue>s8ntBS/35iYGZYg16NrU4vwxdUufDXw/YVN5E9AIUK0=</dsig:SignatureValue>"##,
    r##"<dsig:Object Id="DSig.Object_I08V3cMJvHneFuSSVRb87A22" MimeType="text/xml">"##,
    r##"<Web>up up and away</Web></dsig:Object></dsig:Signature>"##,
);

/// `signature-enveloping-rsa-sha256.xml` from the Oracle interop set.
const RSA_SHA256: &str = concat!(
    r##"<dsig:Signature xmlns:dsig="http://www.w3.org/2000/09/xmldsig#">"##,
    r##"<dsig:SignedInfo>"##,
    r##"<dsig:CanonicalizationMethod Algorithm="http://www.w3.org/TR/2001/REC-xml-c14n-20010315"/>"##,
    r##"<dsig:SignatureMethod Algorithm="http://www.w3.org/2001/04/xmldsig-more#rsa-sha256"/>"##,
    r##"<dsig:Reference URI="#DSig.Object_gdHd5sa901sX14P1Fv8QJA22" Type="http://www.w3.org/2000/09/xmldsig#Object">"##,
    r##"<dsig:DigestMethod Algorithm="http://www.w3.org/2000/09/xmldsig#sha1"/>"##,
    r##"<dsig:DigestValue>a8uS43VzNNwzLOM6wHczXPq906w=</dsig:DigestValue>"##,
    r##"</dsig:Reference></dsig:SignedInfo>"##,
    r##"<dsig:SignatureValue>a1MU1N6mcju2/bp2sp7017z4fUumuz0cprBLhCL/aZDYnryDAX+ztBQjtPxky"##,
    r##"CsS92wgd3ractNxosoU3M0XljZqs1/x9B8bWOYwhh4+B0FOTwHSzLApsPqes8SJie1N71UyQN+QT"##,
    r##"vsW2SnLhDIF4J0ZQ0bSVsNKDpRLNg0tYCQ=</dsig:SignatureValue>"##,
    r##"<dsig:KeyInfo><dsig:KeyValue><dsig:RSAKeyValue><dsig:Modulus>"##,
    r##"gIb6nAB9oS/AI5jIj6WymvQhRxiMlE07G4abmMliYi5zWzvaFE2tnU+RZIBgtoXcgDEIU/vsLQut"##,
    r##"7nzCn9mHxC8JEaV4D4U91j64AyZakShqJw7qjJfqUxxPL0yJv2oFiouPDjGuJ9JPi0NrsZq+yfWf"##,
    r##"M54s4b9SNkcOIVMybZU=</dsig:Modulus><dsig:Exponent>AQAB</dsig:Exponent>"##,
    r##"</dsig:RSAKeyValue></dsig:KeyValue></dsig:KeyInfo>"##,
    r##"<dsig:Object Id="DSig.Object_gdHd5sa901sX14P1Fv8QJA22" MimeType="text/xml">"##,
    r##"<Web>up up and away</Web></dsig:Object></dsig:Signature>"##,
);

/// `signature-enveloping-p256_sha256.xml` from the Oracle interop set: ECDSA on
/// P-256 with SHA-256, keyed by a `<dsig11:ECKeyValue>`.
const ECDSA_P256_SHA256: &str = concat!(
    r##"<dsig:Signature xmlns:dsig="http://www.w3.org/2000/09/xmldsig#"><dsig:SignedInfo>"##,
    r##"<dsig:CanonicalizationMethod Algorithm="http://www.w3.org/TR/2001/REC-xml-c14n-20010315"/>"##,
    r##"<dsig:SignatureMethod Algorithm="http://www.w3.org/2001/04/xmldsig-more#ecdsa-sha256"/>"##,
    r##"<dsig:Reference URI="#DSig.Object_1" Type="http://www.w3.org/2000/09/xmldsig#Object">"##,
    r##"<dsig:DigestMethod Algorithm="http://www.w3.org/2001/04/xmlenc#sha256"/>"##,
    r##"<dsig:DigestValue>vIgv7JtPOh3hpedKK0rm8XHtYCSoBX4eEF0YwnB26Es=</dsig:DigestValue>"##,
    r##"</dsig:Reference></dsig:SignedInfo>"##,
    r##"<dsig:SignatureValue>eYx4ImirtPG/eJLWgJHoMS30voH+tozerMftKbYz27vtYNgsHfAvV4M+"##,
    r##"oEkNgoibq5qnwsO2Z8nn+ndKxhVqFg==</dsig:SignatureValue>"##,
    r##"<dsig:KeyInfo><dsig:KeyValue>"##,
    r##"<ECKeyValue xmlns="http://www.w3.org/2009/xmldsig11#">"##,
    r##"<NamedCurve URI="urn:oid:1.2.840.10045.3.1.7"/>"##,
    r##"<PublicKey>BJ/yaXNlq4FRObyJCBhb5jAz8GVzinK3bBGLjSDfjbJwNfydtgjnlS4EsDmxSRhW"##,
    r##"yJWq6GIqy5wvnaiARK04uB4=</PublicKey></ECKeyValue>"##,
    r##"</dsig:KeyValue></dsig:KeyInfo>"##,
    r##"<dsig:Object Id="DSig.Object_1" MimeType="text/xml">"##,
    r##"<Web>up up and away</Web></dsig:Object></dsig:Signature>"##,
);

/// `signature-enveloping-p384_sha1.xml` from the Oracle interop set: ECDSA on
/// P-384 with SHA-1, so the digest is narrower than the group order and no
/// truncation may happen.
const ECDSA_P384_SHA1: &str = concat!(
    r##"<dsig:Signature xmlns:dsig="http://www.w3.org/2000/09/xmldsig#"><dsig:SignedInfo>"##,
    r##"<dsig:CanonicalizationMethod Algorithm="http://www.w3.org/TR/2001/REC-xml-c14n-20010315"/>"##,
    r##"<dsig:SignatureMethod Algorithm="http://www.w3.org/2001/04/xmldsig-more#ecdsa-sha1"/>"##,
    r##"<dsig:Reference URI="#DSig.Object_1" Type="http://www.w3.org/2000/09/xmldsig#Object">"##,
    r##"<dsig:DigestMethod Algorithm="http://www.w3.org/2000/09/xmldsig#sha1"/>"##,
    r##"<dsig:DigestValue>QyX9FBWyHn4T5eTnSMtylMw+tNQ=</dsig:DigestValue>"##,
    r##"</dsig:Reference></dsig:SignedInfo>"##,
    r##"<dsig:SignatureValue>f384icRqZaSpxvuo3JMKgJ7fsn4KEAA31h+b5bPAef7KfKFc37jL3inA"##,
    r##"GW6e6KasFD/mBheWvM3iRXhBwAA9zajh8i6k9gvQrh1tLKjsMCVOskK9cE1v/Vevz1T2p0lP"##,
    r##"</dsig:SignatureValue>"##,
    r##"<dsig:KeyInfo><dsig:KeyValue>"##,
    r##"<ECKeyValue xmlns="http://www.w3.org/2009/xmldsig11#">"##,
    r##"<NamedCurve URI="urn:oid:1.3.132.0.34"/>"##,
    r##"<PublicKey>BO/yd/OZzDfjX4qivDY/vsUIuh6KWAxoxW5P4ukvwd+T6pVljWsX2UBJNNy5MdhT"##,
    r##"wB8e2YwB8kUbJwdsAS/XGi/fz8unFrs+lVlAgIs6s/xBYFbfUoRiAacD2SpVDe6XBA=="##,
    r##"</PublicKey></ECKeyValue>"##,
    r##"</dsig:KeyValue></dsig:KeyInfo>"##,
    r##"<dsig:Object Id="DSig.Object_1" MimeType="text/xml">"##,
    r##"<Web>up up and away</Web></dsig:Object></dsig:Signature>"##,
);

/// `signature-enveloping-derencoded-ec.xml` from the Oracle interop set: the
/// same P-256 key, given as a DER `SubjectPublicKeyInfo` instead.
const ECDSA_DER_ENCODED: &str = concat!(
    r##"<dsig:Signature xmlns:dsig="http://www.w3.org/2000/09/xmldsig#"><dsig:SignedInfo>"##,
    r##"<dsig:CanonicalizationMethod Algorithm="http://www.w3.org/TR/2001/REC-xml-c14n-20010315"/>"##,
    r##"<dsig:SignatureMethod Algorithm="http://www.w3.org/2001/04/xmldsig-more#ecdsa-sha256"/>"##,
    r##"<dsig:Reference URI="#DSig.Object_zv1ejyt3CTdWWFZEI3SgsQ22" "##,
    r##"Type="http://www.w3.org/2000/09/xmldsig#Object">"##,
    r##"<dsig:DigestMethod Algorithm="http://www.w3.org/2001/04/xmlenc#sha256"/>"##,
    r##"<dsig:DigestValue>Rw1PkDehlYLIVVATQVi2Z41XEBCUNaV7z63AteOr6P0=</dsig:DigestValue>"##,
    r##"</dsig:Reference></dsig:SignedInfo>"##,
    r##"<dsig:SignatureValue>56WG05ONca26cxpmLaRc+6RHfk1wTimQ4rguHq+ogAk08DvCB+bs3WLg"##,
    r##"dMJQDWOxMOjVMuxuF+Bue7X382iB9Q==</dsig:SignatureValue>"##,
    r##"<dsig:KeyInfo>"##,
    r##"<dsig11:DEREncodedKeyValue xmlns:dsig11="http://www.w3.org/2009/xmldsig11#">"##,
    r##"MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEn/Jpc2WrgVE5vIkIGFvmMDPwZXOKcrdsEYuNIN+N"##,
    r##"snA1/J22COeVLgSwObFJGFbIlaroYirLnC+dqIBErTi4Hg=="##,
    r##"</dsig11:DEREncodedKeyValue></dsig:KeyInfo>"##,
    r##"<dsig:Object Id="DSig.Object_zv1ejyt3CTdWWFZEI3SgsQ22" MimeType="text/xml">"##,
    r##"<Web>up up and away</Web></dsig:Object></dsig:Signature>"##,
);

/// The canonical form of the HMAC document's `<ds:SignedInfo>`, which is what
/// the MAC is computed over. Pinning it makes a canonicalization regression
/// point at canonicalization instead of at the cryptography.
const HMAC_CANONICAL_SIGNED_INFO: &str = concat!(
    r##"<dsig:SignedInfo xmlns:dsig="http://www.w3.org/2000/09/xmldsig#">"##,
    r##"<dsig:CanonicalizationMethod Algorithm="http://www.w3.org/TR/2001/REC-xml-c14n-20010315">"##,
    r##"</dsig:CanonicalizationMethod>"##,
    r##"<dsig:SignatureMethod Algorithm="http://www.w3.org/2001/04/xmldsig-more#hmac-sha256">"##,
    r##"</dsig:SignatureMethod>"##,
    r##"<dsig:Reference Type="http://www.w3.org/2000/09/xmldsig#Object" URI="#DSig.Object_I08V3cMJvHneFuSSVRb87A22">"##,
    r##"<dsig:DigestMethod Algorithm="http://www.w3.org/2000/09/xmldsig#sha1">"##,
    r##"</dsig:DigestMethod>"##,
    r##"<dsig:DigestValue>myrT5qEfA7Wemy2WONCZG66c5QE=</dsig:DigestValue>"##,
    r##"</dsig:Reference></dsig:SignedInfo>"##,
);

#[test]
fn the_hmac_sha256_interop_document_verifies() -> Result<(), Box<dyn std::error::Error>> {
    let document = parse(HMAC_SHA256);
    let signature = Signature::find(&document)?;
    assert_eq!(
        signature.signed_info().signature_method().algorithm(),
        SignatureAlgorithm::HmacSha256
    );
    assert_eq!(
        signature
            .signed_info()
            .references()
            .next()
            .map(|reference| reference.digest_algorithm()),
        Some(DigestAlgorithm::Sha1)
    );

    let result = Verifier::new().verify(
        &document,
        &signature,
        &SharedSecret::new(INTEROP_HMAC_SECRET.to_vec()),
    )?;
    assert!(result.core_valid());
    assert!(result.signature_valid());
    assert_eq!(result.references().len(), 1);

    // The canonical SignedInfo is byte-for-byte what the Recommendation asks
    // for: the inherited `xmlns:dsig`, expanded empty elements, and attributes
    // sorted so that `Type` precedes `URI`.
    assert_eq!(
        String::from_utf8(result.canonical_signed_info().to_vec())?,
        HMAC_CANONICAL_SIGNED_INFO
    );
    Ok(())
}

#[test]
fn the_rsa_sha256_interop_document_verifies() -> Result<(), Box<dyn std::error::Error>> {
    let document = parse(RSA_SHA256);
    let signature = Signature::find(&document)?;
    assert_eq!(
        signature.signed_info().signature_method().algorithm(),
        SignatureAlgorithm::RsaSha256
    );

    let key = signature
        .key_info()
        .and_then(oxixml_dsig::KeyInfo::rsa_public_key)
        .expect("the KeyInfo carries an RSAKeyValue");
    assert_eq!(key.modulus_bits(), 1024);
    assert_eq!(key.exponent(), [0x01, 0x00, 0x01]);

    let result = Verifier::new().verify(&document, &signature, &UntrustedKeyInfo)?;
    assert!(result.core_valid());
    Ok(())
}

#[test]
fn the_ecdsa_interop_documents_verify() -> Result<(), Box<dyn std::error::Error>> {
    for (name, source, curve, method) in [
        (
            "p256 ECKeyValue",
            ECDSA_P256_SHA256,
            NamedCurve::P256,
            SignatureAlgorithm::EcdsaSha256,
        ),
        (
            "p384 ECKeyValue",
            ECDSA_P384_SHA1,
            NamedCurve::P384,
            SignatureAlgorithm::EcdsaSha1,
        ),
        (
            "p256 DEREncodedKeyValue",
            ECDSA_DER_ENCODED,
            NamedCurve::P256,
            SignatureAlgorithm::EcdsaSha256,
        ),
    ] {
        let document = parse(source);
        let signature = Signature::find(&document)?;
        assert_eq!(
            signature.signed_info().signature_method().algorithm(),
            method,
            "{name}"
        );

        let key = signature
            .key_info()
            .and_then(oxixml_dsig::KeyInfo::ec_public_key)
            .unwrap_or_else(|| panic!("{name}: the KeyInfo carries an EC key"));
        assert_eq!(key.curve(), curve, "{name}");
        // An uncompressed SEC 1 point: the 0x04 tag and two coordinates.
        assert_eq!(
            key.encoded_point().len(),
            1 + 2 * curve.field_bytes(),
            "{name}"
        );

        let result = Verifier::new().verify(&document, &signature, &UntrustedKeyInfo)?;
        assert!(result.core_valid(), "{name}: {result:?}");

        // The signature value is the raw `r ‖ s` of §6.4.3, two fixed-width
        // halves — not an ASN.1 SEQUENCE, which would be a different length.
        assert_eq!(
            signature.signature_value().len(),
            2 * curve.field_bytes(),
            "{name}"
        );
    }
    Ok(())
}

/// The interop suite also ships documents on P-521, which XML Signature 1.1
/// does not require and this crate does not implement. Naming that curve must
/// leave the verifier with no key at all rather than with a key it mishandles.
#[test]
fn an_unsupported_curve_stays_fail_closed() {
    let source = ECDSA_P256_SHA256.replace(
        "urn:oid:1.2.840.10045.3.1.7",
        // secp521r1.
        "urn:oid:1.3.132.0.35",
    );
    let document = parse(&source);
    let signature = Signature::find(&document).expect("the signature still parses");
    // The hint is still readable — a caller can report the curve by name …
    assert!(matches!(
        signature.key_info().and_then(|info| info.items().next()),
        Some(oxixml_dsig::KeyInfoItem::EcKeyValue { named_curve, .. })
            if named_curve.as_deref() == Some("urn:oid:1.3.132.0.35")
    ));
    // … but no key comes out of it.
    assert!(
        signature
            .key_info()
            .and_then(oxixml_dsig::KeyInfo::ec_public_key)
            .is_none()
    );
    assert!(matches!(
        Verifier::new().verify(&document, &signature, &UntrustedKeyInfo),
        Err(DsigError::NoKey(_))
    ));
}

#[test]
fn a_tampered_payload_breaks_the_reference_not_the_signature()
-> Result<(), Box<dyn std::error::Error>> {
    for (name, source) in [("hmac", HMAC_SHA256), ("rsa", RSA_SHA256)] {
        let mut document = parse(source);
        let web = common::element_named(&document, "Web");
        tamper_text(&mut document, web);

        let signature = Signature::find(&document)?;
        let result = if name == "hmac" {
            Verifier::new().verify(
                &document,
                &signature,
                &SharedSecret::new(INTEROP_HMAC_SECRET.to_vec()),
            )?
        } else {
            Verifier::new().verify(&document, &signature, &UntrustedKeyInfo)?
        };
        assert!(!result.core_valid(), "{name}");
        assert!(result.signature_valid(), "{name}: SignedInfo was untouched");
        assert!(
            result.references().all(|reference| !reference.valid()),
            "{name}"
        );
    }
    Ok(())
}

#[test]
fn a_tampered_digest_value_breaks_the_signature_not_the_reference()
-> Result<(), Box<dyn std::error::Error>> {
    // Rewriting the DigestValue to match the tampered payload is the obvious
    // forgery, and it is exactly what the signature over SignedInfo prevents.
    let mut document = parse(HMAC_SHA256);
    let web = common::element_named(&document, "Web");
    tamper_text(&mut document, web);

    let digest_node = Signature::find(&document)?
        .signed_info()
        .references()
        .next()
        .map(oxixml_dsig::Reference::digest_value_node)
        .expect("a Reference");

    // Recompute what the DigestValue would have to be, by asking the verifier.
    let signature = Signature::find(&document)?;
    let result = Verifier::new().verify(
        &document,
        &signature,
        &SharedSecret::new(INTEROP_HMAC_SECRET.to_vec()),
    )?;
    let forged = result
        .references()
        .next()
        .and_then(|reference| reference.computed_digest())
        .expect("a computed digest")
        .to_vec();
    document.set_text_content(digest_node, oxixml_text::base64::base64_encode(&forged))?;

    let signature = Signature::find(&document)?;
    let result = Verifier::new().verify(
        &document,
        &signature,
        &SharedSecret::new(INTEROP_HMAC_SECRET.to_vec()),
    )?;
    assert!(
        result.references().all(|reference| reference.valid()),
        "the forged digest matches the tampered payload"
    );
    assert!(!result.signature_valid(), "but SignedInfo no longer MACs");
    assert!(!result.core_valid());
    Ok(())
}

#[test]
fn the_rsa_document_does_not_verify_under_a_different_key() -> Result<(), Box<dyn std::error::Error>>
{
    // The 1024-bit key of the interop document, with one bit of the modulus
    // flipped: still a plausible RSA key, but not the signer's.
    let document = parse(RSA_SHA256);
    let signature = Signature::find(&document)?;
    let genuine = signature
        .key_info()
        .and_then(oxixml_dsig::KeyInfo::rsa_public_key)
        .expect("an RSAKeyValue");
    let mut modulus = genuine.modulus().to_vec();
    if let Some(byte) = modulus.get_mut(1) {
        *byte ^= 0x01;
    }
    let impostor = oxixml_dsig::RsaPublicKey::from_components(&modulus, genuine.exponent())?;

    let result =
        Verifier::new().verify(&document, &signature, &oxixml_dsig::FixedKey::rsa(impostor))?;
    assert!(!result.signature_valid());
    assert!(result.references().all(|reference| reference.valid()));
    Ok(())
}

/// A forged `SignatureValue` must fail for the ECDSA documents too, and must
/// fail *as a signature*: the references still digest correctly, so a verifier
/// that reported the wrong half of the failure would be misleading.
#[test]
fn a_forged_ecdsa_signature_value_is_rejected() -> Result<(), Box<dyn std::error::Error>> {
    for (name, source) in [("p256", ECDSA_P256_SHA256), ("p384", ECDSA_P384_SHA1)] {
        let mut document = parse(source);
        let value_node = Signature::find(&document)?.signature_value_node();
        let original = document.text_content(value_node).unwrap_or_default();
        let mut octets = oxixml_text::base64::base64_decode_forgiving(&original)?;
        if let Some(first) = octets.first_mut() {
            *first ^= 0x80;
        }
        document.set_text_content(value_node, oxixml_text::base64::base64_encode(&octets))?;

        let signature = Signature::find(&document)?;
        let result = Verifier::new().verify(&document, &signature, &UntrustedKeyInfo)?;
        assert!(!result.signature_valid(), "{name}");
        assert!(
            result.references().all(|reference| reference.valid()),
            "{name}"
        );
    }
    Ok(())
}

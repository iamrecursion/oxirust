//! Generation followed by verification, across every algorithm and transform
//! this crate implements.
//!
//! A round-trip test proves less than an interop vector — a bug shared by the
//! signer and the verifier cancels out — so these tests are the complement of
//! [`known_answers`](../known_answers/index.html), not a replacement for it.
//! What they *do* cover is everything the interop suite does not exercise at
//! all: enveloped signatures, detached signatures with a resolver, the
//! transform chain, multiple references, manifests, and the signing direction
//! generally.

mod common;

use oxixml_c14n::{CanonicalizationAlgorithm, InclusiveNamespaces};
use oxixml_dom::{Document, NodeId};
use oxixml_dsig::{
    DigestAlgorithm, DsaPrivateKey, DsigError, EcPrivateKey, Filter2Spec, FilterOperation,
    FixedKey, KeyInfoSpec, NamedCurve, PathValidator, PssParameters, ReferenceSpec, RsaPrivateKey,
    SharedSecret, Signature, SignatureAlgorithm, SignatureBuilder, SigningKey, StaticResolver,
    Timestamp, TransformSpec, TrustedCertificates, UntrustedKeyInfo, Verifier, sign, sign_with,
};

use common::{hex, parse};

type Failure = Box<dyn std::error::Error>;

/// The pinned 1024-bit RSA test key, as PKCS#1 DER.
const RSA_PRIVATE_KEY_BASE64: &str = concat!(
    "MIICXQIBAAKBgQCORygKyXES6zm7xAg8wBiMMik2BfXZ9tZqVI+aFsqOU15Hm6RGMHTEEmpi+dxe",
    "s+XuUfMN6aAWx0RskK9OAXWKa1oyRzrJ9oeQnj5GDM9HZIFTgoRpWrr181HkNyjS+X9VRkPnh2s+",
    "oEG9eGKrCkArdksdw9IvHNFCLVrbSX62VQIDAQABAoGAJQjKmHw07arC/WgaiM2r1hpcdc55X0IM",
    "L+NQ+dgC46c5vEP8L7T/P+ZXQOou/YuwRyeXr8BnRmaaHfRPXWutz3Ipph9dTRcvUmLxeSPEBHRp",
    "nqGt+pSoAn3bKaAz6Yj2dt/HAqU1zzt57XRNabcAHisCkA9OCvlwqeTZ+fcv+OECQQD4ge8oqmft",
    "l71XHon1dv/tzC4Eze4NI9zmyuIqNB3dkBlGjbwFSH7HrebRJnGUoFP2b4T3vgbSZ1s35gAmvFTZ",
    "AkEAkpFOWB+djzIpWj6h4sYcNTO6R0Yygse/Www+4eiSKQkQPE+Ao2Ybpyhr2Moc6RLhhaOyGbyx",
    "o9/R9Oc1yf7P3QJBAOeZgOBvYAhoQIeu6Hgh/YcLoV/M5CzO3ZMiRgF7LNJO959vfGmYFL/X7KPD",
    "plsTyOyFJnEVYpo4xSSlcEKphpECQEjyfRWrFpYwlxhNV/EwPNXV2v6iV31zwS+llp6FTEmUMzpx",
    "L0skBNtbaOHLJjxyo0AKSxg4LAmt2ieT89eHNRkCQQDKgurVmoAiN0hlXcoymZmIU2g1GbggfWb5",
    "nSmldq3TJIbq1dXikhd8Q/Dit2XW9Ncd+JadjU1tka+Rzo6y20pg",
);

fn rsa_key() -> Result<RsaPrivateKey, Failure> {
    Ok(RsaPrivateKey::from_der(
        &oxixml_text::base64::base64_decode(RSA_PRIVATE_KEY_BASE64)?,
    )?)
}

fn secret() -> SharedSecret {
    SharedSecret::new(b"round-trip secret".to_vec())
}

fn hmac_key() -> SigningKey {
    SigningKey::hmac(*b"round-trip secret")
}

fn verify(document: &Document, node: NodeId) -> Result<bool, Failure> {
    let signature = Signature::parse(document, node)?;
    Ok(Verifier::new()
        .verify(document, &signature, &secret())?
        .core_valid())
}

#[test]
fn every_hmac_algorithm_round_trips() -> Result<(), Failure> {
    for algorithm in [
        SignatureAlgorithm::HmacSha1,
        SignatureAlgorithm::HmacSha224,
        SignatureAlgorithm::HmacSha256,
        SignatureAlgorithm::HmacSha384,
        SignatureAlgorithm::HmacSha512,
    ] {
        for digest in [
            DigestAlgorithm::Sha1,
            DigestAlgorithm::Sha224,
            DigestAlgorithm::Sha256,
            DigestAlgorithm::Sha384,
            DigestAlgorithm::Sha512,
        ] {
            let mut document = parse("<doc><payload>content</payload></doc>");
            let root = document.document_element().expect("a document element");
            let node = SignatureBuilder::new(algorithm)
                .prefix("ds")
                .reference(
                    ReferenceSpec::new("")
                        .digest(digest)
                        .transform(TransformSpec::EnvelopedSignature),
                )
                .build_into(&mut document, root)?;
            sign(&mut document, node, &hmac_key())?;
            assert!(verify(&document, node)?, "{algorithm} over {digest}");
        }
    }
    Ok(())
}

#[test]
fn every_rsa_algorithm_round_trips() -> Result<(), Failure> {
    let key = rsa_key()?;
    for algorithm in [
        SignatureAlgorithm::RsaSha1,
        SignatureAlgorithm::RsaSha224,
        SignatureAlgorithm::RsaSha256,
        SignatureAlgorithm::RsaSha384,
        SignatureAlgorithm::RsaSha512,
    ] {
        let mut document = parse(r##"<doc><payload Id="p">content</payload></doc>"##);
        let root = document.document_element().expect("a document element");
        let node = SignatureBuilder::new(algorithm)
            .reference(ReferenceSpec::new("#p"))
            .key_info(KeyInfoSpec::RsaKeyValue(key.public_key()))
            .build_into(&mut document, root)?;
        sign(&mut document, node, &SigningKey::rsa(key.clone()))?;

        let signature = Signature::parse(&document, node)?;
        let result = Verifier::new().verify(&document, &signature, &UntrustedKeyInfo)?;
        assert!(result.core_valid(), "{algorithm}");

        // A pinned key must give the same answer.
        let pinned = FixedKey::rsa(key.public_key());
        assert!(
            Verifier::new()
                .verify(&document, &signature, &pinned)?
                .core_valid(),
            "{algorithm} with a pinned key"
        );
    }
    Ok(())
}

/// The P-256 and P-384 private keys of RFC 6979 Appendix A.2, which are the two
/// curves XML Signature 1.1 requires.
const P256_PRIVATE_KEY: &str = "C9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721";
const P384_PRIVATE_KEY: &str = concat!(
    "6B9D3DAD2E1B8C1C05B19875B6659F4DE23C3B667BF297BA9AA47740787137D8",
    "96D5724E4C70A825F872C9EA60D2EDF5",
);

#[test]
fn every_ecdsa_algorithm_round_trips_on_both_curves() -> Result<(), Failure> {
    for (curve, scalar) in [
        (NamedCurve::P256, P256_PRIVATE_KEY),
        (NamedCurve::P384, P384_PRIVATE_KEY),
    ] {
        let key = EcPrivateKey::from_scalar(curve, &hex(scalar))?;
        for algorithm in [
            SignatureAlgorithm::EcdsaSha1,
            SignatureAlgorithm::EcdsaSha224,
            SignatureAlgorithm::EcdsaSha256,
            SignatureAlgorithm::EcdsaSha384,
            SignatureAlgorithm::EcdsaSha512,
        ] {
            let mut document = parse(r##"<doc><payload Id="p">content</payload></doc>"##);
            let root = document.document_element().expect("a document element");
            let node = SignatureBuilder::new(algorithm)
                .reference(ReferenceSpec::new("#p"))
                .key_info(KeyInfoSpec::EcKeyValue(key.public_key()))
                .build_into(&mut document, root)?;
            sign(&mut document, node, &SigningKey::ec(key.clone()))?;

            let signature = Signature::parse(&document, node)?;
            // XML Signature §6.4.3: `r ‖ s`, each as wide as the group order.
            assert_eq!(
                signature.signature_value().len(),
                2 * curve.field_bytes(),
                "{curve} with {algorithm}"
            );
            assert!(
                Verifier::new()
                    .verify(&document, &signature, &UntrustedKeyInfo)?
                    .core_valid(),
                "{curve} with {algorithm}"
            );
            assert!(
                Verifier::new()
                    .verify(&document, &signature, &FixedKey::ec(key.public_key()))?
                    .core_valid(),
                "{curve} with {algorithm}, pinned"
            );
        }
    }
    Ok(())
}

/// A key on the other curve must not verify, even though the family matches.
#[test]
fn an_ecdsa_signature_does_not_verify_under_the_other_curve() -> Result<(), Failure> {
    let key = EcPrivateKey::from_scalar(NamedCurve::P256, &hex(P256_PRIVATE_KEY))?;
    let other = EcPrivateKey::from_scalar(NamedCurve::P384, &hex(P384_PRIVATE_KEY))?;

    let mut document = parse(r##"<doc><payload Id="p">content</payload></doc>"##);
    let root = document.document_element().expect("a document element");
    let node = SignatureBuilder::new(SignatureAlgorithm::EcdsaSha256)
        .reference(ReferenceSpec::new("#p"))
        .build_into(&mut document, root)?;
    sign(&mut document, node, &SigningKey::ec(key))?;

    let signature = Signature::parse(&document, node)?;
    let result =
        Verifier::new().verify(&document, &signature, &FixedKey::ec(other.public_key()))?;
    assert!(!result.signature_valid());
    Ok(())
}

#[test]
fn both_dsa_algorithms_round_trip() -> Result<(), Failure> {
    // The FIPS 186-2 Appendix 5 parameters, which XML Signature §6.4.1 uses as
    // its own worked example.
    let p = hex(concat!(
        "8DF2A494492276AA3D25759BB06869CBEAC0D83AFB8D0CF7CBB8324F0D7882E5",
        "D0762FC5B7210EAFC2E9ADAC32AB7AAC49693DFBF83724C2EC0736EE31C80291",
    ));
    let q = hex("C773218C737EC8EE993B4F2DED30F48EDACE915F");
    let g = hex(concat!(
        "626D027839EA0A13413163A55B4CB500299D5522956CEFCB3BFF10F399CE2C2E",
        "71CB9DE5FA24BABF58E5B79521925C9CC42E9F6F464B088CC572AF53E6D78802",
    ));
    let x = hex("2070B3223DBA372FDE1C0FFC7B2E3B498B260614");
    let key = DsaPrivateKey::from_components(&p, &q, &g, &x)?;

    for (algorithm, half_width) in [
        (SignatureAlgorithm::DsaSha1, 20),
        (SignatureAlgorithm::DsaSha256, 32),
    ] {
        let mut document = parse(r##"<doc><payload Id="p">content</payload></doc>"##);
        let root = document.document_element().expect("a document element");
        let node = SignatureBuilder::new(algorithm)
            .reference(ReferenceSpec::new("#p"))
            .key_info(KeyInfoSpec::DsaKeyValue(Box::new(key.public_key())))
            .build_into(&mut document, root)?;
        sign(&mut document, node, &SigningKey::dsa(key.clone()))?;

        let signature = Signature::parse(&document, node)?;
        // §6.4.1 and §6.4.2 fix the width by identifier, not by key: the
        // 160-bit `q` here is left-padded into 32-octet halves for SHA-256.
        assert_eq!(
            signature.signature_value().len(),
            2 * half_width,
            "{algorithm}"
        );
        assert!(
            Verifier::new()
                .verify(&document, &signature, &UntrustedKeyInfo)?
                .core_valid(),
            "{algorithm}"
        );
        assert!(
            Verifier::new()
                .verify(&document, &signature, &FixedKey::dsa(key.public_key()))?
                .core_valid(),
            "{algorithm}, pinned"
        );
    }
    Ok(())
}

#[test]
fn rsa_pss_round_trips_with_the_default_and_declared_parameters() -> Result<(), Failure> {
    let key = rsa_key()?;

    // No <pss:RSAPSSParams>: the RFC 9231 defaults apply on both sides.
    let mut document = parse(r##"<doc><payload Id="p">content</payload></doc>"##);
    let root = document.document_element().expect("a document element");
    let node = SignatureBuilder::new(SignatureAlgorithm::RsaPss)
        .reference(ReferenceSpec::new("#p"))
        .key_info(KeyInfoSpec::RsaKeyValue(key.public_key()))
        .build_into(&mut document, root)?;
    sign(&mut document, node, &SigningKey::rsa(key.clone()))?;
    let signature = Signature::parse(&document, node)?;
    assert!(
        signature
            .signed_info()
            .signature_method()
            .pss_parameters()
            .is_none()
    );
    assert_eq!(
        signature
            .signed_info()
            .signature_method()
            .effective_pss_parameters(),
        PssParameters::default()
    );
    assert!(
        Verifier::new()
            .verify(&document, &signature, &UntrustedKeyInfo)?
            .core_valid()
    );

    // Declared parameters must be honoured rather than replaced by the
    // defaults — a SHA-384 digest with MGF1 over SHA-1 and a short salt is
    // nothing like the default set.
    let declared = PssParameters::new(DigestAlgorithm::Sha384)
        .mask_generation_digest(DigestAlgorithm::Sha1)
        .salt_length(16);
    let mut document = parse(r##"<doc><payload Id="p">content</payload></doc>"##);
    let root = document.document_element().expect("a document element");
    let node = SignatureBuilder::new(SignatureAlgorithm::RsaPss)
        .pss_parameters(declared)
        .reference(ReferenceSpec::new("#p"))
        .key_info(KeyInfoSpec::RsaKeyValue(key.public_key()))
        .build_into(&mut document, root)?;
    sign(&mut document, node, &SigningKey::rsa(key.clone()))?;

    let signature = Signature::parse(&document, node)?;
    assert_eq!(
        signature
            .signed_info()
            .signature_method()
            .pss_parameters()
            .copied(),
        Some(declared)
    );
    assert!(
        Verifier::new()
            .verify(&document, &signature, &UntrustedKeyInfo)?
            .core_valid()
    );

    // Rewriting the declared salt length must break verification: the
    // parameters are part of what the signature commits to.
    let salt = document
        .get_elements_by_tag_name(document.root(), "pss:SaltLength")
        .next()
        .expect("a <pss:SaltLength>");
    document.set_text_content(salt, "20")?;
    let signature = Signature::parse(&document, node)?;
    assert!(
        !Verifier::new()
            .verify(&document, &signature, &UntrustedKeyInfo)?
            .signature_valid()
    );
    Ok(())
}

/// A `<pss:SaltLength>` is attacker-supplied text with no bound of its own, and
/// it reaches the arithmetic as an arbitrary `usize`. Both directions must
/// refuse an impossible one in constant time rather than trying to produce or
/// consume a salt of that size — a signature template of a few hundred octets
/// would otherwise ask for gigabytes of mask generation output.
#[test]
fn an_absurd_declared_salt_length_is_refused_rather_than_generated() -> Result<(), Failure> {
    let key = rsa_key()?;

    // First a genuine signature, so that verification has something to chew on.
    let mut document = parse(r##"<doc><payload Id="p">content</payload></doc>"##);
    let root = document.document_element().expect("a document element");
    let node = SignatureBuilder::new(SignatureAlgorithm::RsaPss)
        .pss_parameters(PssParameters::default())
        .reference(ReferenceSpec::new("#p"))
        .key_info(KeyInfoSpec::RsaKeyValue(key.public_key()))
        .build_into(&mut document, root)?;
    sign(&mut document, node, &SigningKey::rsa(key.clone()))?;

    let salt = document
        .get_elements_by_tag_name(document.root(), "pss:SaltLength")
        .next()
        .expect("a <pss:SaltLength>");
    for declared in [usize::MAX.to_string(), (usize::MAX / 2).to_string()] {
        document.set_text_content(salt, &declared)?;

        // Verification: a bounded "not valid", not a panic and not a hang.
        let signature = Signature::parse(&document, node)?;
        let result = Verifier::new().verify(&document, &signature, &UntrustedKeyInfo)?;
        assert!(!result.signature_valid(), "salt length {declared}");
        assert!(!result.core_valid(), "salt length {declared}");

        // Signing: a bounded error, before any salt is generated.
        assert!(
            matches!(
                sign(&mut document, node, &SigningKey::rsa(key.clone())),
                Err(oxixml_dsig::DsigError::Key(_))
            ),
            "salt length {declared}"
        );
    }
    Ok(())
}

#[test]
fn every_canonicalization_method_round_trips() -> Result<(), Failure> {
    for algorithm in [
        CanonicalizationAlgorithm::Canonical1_0,
        CanonicalizationAlgorithm::Canonical1_0WithComments,
        CanonicalizationAlgorithm::Canonical1_1,
        CanonicalizationAlgorithm::Canonical1_1WithComments,
        CanonicalizationAlgorithm::Exclusive1_0,
        CanonicalizationAlgorithm::Exclusive1_0WithComments,
    ] {
        let mut document = parse(
            r##"<e:doc xmlns:e="urn:e" xmlns:unused="urn:unused"><!-- note --><p>content</p></e:doc>"##,
        );
        let root = document.document_element().expect("a document element");
        let node = SignatureBuilder::new(SignatureAlgorithm::HmacSha256)
            .prefix("ds")
            .canonicalization(algorithm)
            .inclusive_prefixes(InclusiveNamespaces::parse("e"))
            .reference(ReferenceSpec::new("").transform(TransformSpec::EnvelopedSignature))
            .build_into(&mut document, root)?;
        sign(&mut document, node, &hmac_key())?;
        assert!(verify(&document, node)?, "{algorithm}");
    }
    Ok(())
}

#[test]
fn an_exclusive_c14n_transform_survives_re_enveloping() -> Result<(), Failure> {
    // Sign a payload with exclusive canonicalization inside one envelope …
    let mut document = parse(
        r##"<a:env xmlns:a="urn:a" xmlns:junk="urn:junk"><p:load xmlns:p="urn:p" Id="x">v</p:load></a:env>"##,
    );
    let root = document.document_element().expect("a document element");
    let node = SignatureBuilder::new(SignatureAlgorithm::HmacSha256)
        .prefix("ds")
        .canonicalization(CanonicalizationAlgorithm::Exclusive1_0)
        .reference(
            ReferenceSpec::new("#x").transform(TransformSpec::Canonicalization(
                CanonicalizationAlgorithm::Exclusive1_0,
                InclusiveNamespaces::none(),
            )),
        )
        .build_into(&mut document, root)?;
    sign(&mut document, node, &hmac_key())?;

    let reference_digest = {
        let signature = Signature::parse(&document, node)?;
        signature
            .signed_info()
            .references()
            .next()
            .map(|reference| reference.digest_value().to_vec())
            .expect("a reference")
    };

    // … then move the same payload into a differently-decorated envelope and
    // check the digest is unchanged, which is the entire point of exclusive
    // canonicalization.
    let mut other =
        parse(r##"<b:other xmlns:b="urn:b"><p:load xmlns:p="urn:p" Id="x">v</p:load></b:other>"##);
    let other_root = other.document_element().expect("a document element");
    let other_node = SignatureBuilder::new(SignatureAlgorithm::HmacSha256)
        .prefix("ds")
        .canonicalization(CanonicalizationAlgorithm::Exclusive1_0)
        .reference(
            ReferenceSpec::new("#x").transform(TransformSpec::Canonicalization(
                CanonicalizationAlgorithm::Exclusive1_0,
                InclusiveNamespaces::none(),
            )),
        )
        .build_into(&mut other, other_root)?;
    sign(&mut other, other_node, &hmac_key())?;
    let other_digest = {
        let signature = Signature::parse(&other, other_node)?;
        signature
            .signed_info()
            .references()
            .next()
            .map(|reference| reference.digest_value().to_vec())
            .expect("a reference")
    };
    assert_eq!(reference_digest, other_digest);
    Ok(())
}

#[test]
fn a_detached_signature_round_trips_through_a_resolver() -> Result<(), Failure> {
    let payload = b"the detached payload".to_vec();
    let resolver = StaticResolver::new().with("payload.bin", payload.clone());

    let mut document = parse("<manifest/>");
    let root = document.document_element().expect("a document element");
    let node = SignatureBuilder::new(SignatureAlgorithm::HmacSha256)
        .prefix("ds")
        .reference(ReferenceSpec::new("payload.bin").digest(DigestAlgorithm::Sha512))
        .build_into(&mut document, root)?;
    sign_with(&mut document, node, &hmac_key(), Some(&resolver))?;

    let signature = Signature::parse(&document, node)?;
    let result = Verifier::new()
        .resolver(&resolver)
        .verify(&document, &signature, &secret())?;
    assert!(result.core_valid());

    // Without the resolver the reference cannot even be dereferenced.
    let result = Verifier::new().verify(&document, &signature, &secret())?;
    assert!(!result.core_valid());
    assert!(
        result
            .references()
            .all(|reference| reference.error().is_some())
    );

    // With different data it must not verify.
    let other = StaticResolver::new().with("payload.bin", b"something else".to_vec());
    let result = Verifier::new()
        .resolver(&other)
        .verify(&document, &signature, &secret())?;
    assert!(!result.core_valid());
    Ok(())
}

#[test]
fn a_reference_without_a_uri_round_trips() -> Result<(), Failure> {
    let resolver = StaticResolver::new().with_implicit(b"implicit data".to_vec());
    let mut document = parse("<doc/>");
    let root = document.document_element().expect("a document element");
    let node = SignatureBuilder::new(SignatureAlgorithm::HmacSha256)
        .reference(ReferenceSpec::implicit())
        .build_into(&mut document, root)?;
    sign_with(&mut document, node, &hmac_key(), Some(&resolver))?;

    let signature = Signature::parse(&document, node)?;
    assert_eq!(
        signature
            .signed_info()
            .references()
            .next()
            .and_then(oxixml_dsig::Reference::uri),
        None
    );
    assert!(
        Verifier::new()
            .resolver(&resolver)
            .verify(&document, &signature, &secret())?
            .core_valid()
    );
    Ok(())
}

#[test]
fn the_base64_transform_round_trips() -> Result<(), Failure> {
    let mut document = parse(r##"<doc><blob Id="b">T3hpWE1M</blob></doc>"##);
    let root = document.document_element().expect("a document element");
    let node = SignatureBuilder::new(SignatureAlgorithm::HmacSha256)
        .reference(ReferenceSpec::new("#b").transform(TransformSpec::Base64))
        .build_into(&mut document, root)?;
    sign(&mut document, node, &hmac_key())?;
    assert!(verify(&document, node)?);

    // The digest is over the *decoded* octets, so it must equal the digest of
    // "OxiXML" itself.
    let signature = Signature::parse(&document, node)?;
    let expected = {
        let mut hasher = Vec::new();
        hasher.extend_from_slice(b"OxiXML");
        DigestAlgorithm::Sha256.digest(&hasher)
    };
    assert_eq!(
        signature
            .signed_info()
            .references()
            .next()
            .map(|reference| reference.digest_value().to_vec()),
        Some(expected)
    );
    Ok(())
}

#[test]
fn the_xpath_transform_round_trips() -> Result<(), Failure> {
    let mut document = parse("<doc><keep>signed</keep><skip>volatile</skip></doc>");
    let root = document.document_element().expect("a document element");
    let node = SignatureBuilder::new(SignatureAlgorithm::HmacSha256)
        .prefix("ds")
        .reference(ReferenceSpec::new("").transform(TransformSpec::xpath(
            "not(ancestor-or-self::skip) and not(ancestor-or-self::ds:Signature)",
        )))
        .build_into(&mut document, root)?;
    // The expression needs the `ds` prefix in scope on the <ds:XPath> element;
    // it already is, because the whole signature is written with that prefix.
    sign(&mut document, node, &hmac_key())?;
    assert!(verify(&document, node)?);

    // The excluded subtree may change freely.
    let skip = common::element_named(&document, "skip");
    document.set_text_content(skip, "changed")?;
    assert!(verify(&document, node)?);

    // The included one may not.
    let keep = common::element_named(&document, "keep");
    document.set_text_content(keep, "tampered")?;
    assert!(!verify(&document, node)?);
    Ok(())
}

#[test]
fn an_xpath_filter_2_union_and_intersect_round_trip() -> Result<(), Failure> {
    let mut document = parse(
        r##"<doc><chapter Id="c1"><t>one</t></chapter><chapter Id="c2"><t>two</t></chapter></doc>"##,
    );
    let root = document.document_element().expect("a document element");
    let node = SignatureBuilder::new(SignatureAlgorithm::HmacSha256)
        .prefix("ds")
        .reference(
            ReferenceSpec::new("").transform(TransformSpec::Filter2(vec![Filter2Spec::new(
                FilterOperation::Intersect,
                "id('c1')",
            )])),
        )
        .build_into(&mut document, root)?;
    sign(&mut document, node, &hmac_key())?;
    assert!(verify(&document, node)?);

    // Only the intersected chapter is covered.
    let second = document
        .get_elements_by_tag_name(root, "chapter")
        .nth(1)
        .expect("a second chapter");
    document.set_text_content(second, "rewritten")?;
    assert!(verify(&document, node)?);

    let first = document
        .get_elements_by_tag_name(root, "chapter")
        .next()
        .expect("a first chapter");
    document.set_text_content(first, "rewritten")?;
    assert!(!verify(&document, node)?);
    Ok(())
}

#[test]
fn several_references_are_all_checked() -> Result<(), Failure> {
    let mut document = parse(r##"<doc><a Id="a">1</a><b Id="b">2</b><c Id="c">3</c></doc>"##);
    let root = document.document_element().expect("a document element");
    let node = SignatureBuilder::new(SignatureAlgorithm::HmacSha384)
        .prefix("ds")
        .reference(ReferenceSpec::new("#a").digest(DigestAlgorithm::Sha1))
        .reference(ReferenceSpec::new("#b").digest(DigestAlgorithm::Sha256))
        .reference(ReferenceSpec::new("#c").digest(DigestAlgorithm::Sha512))
        .build_into(&mut document, root)?;
    sign(&mut document, node, &hmac_key())?;

    let signature = Signature::parse(&document, node)?;
    let result = Verifier::new().verify(&document, &signature, &secret())?;
    assert!(result.core_valid());
    assert_eq!(result.references().len(), 3);

    // Break the middle one only.
    let b = common::element_named(&document, "b");
    document.set_text_content(b, "changed")?;
    let signature = Signature::parse(&document, node)?;
    let result = Verifier::new().verify(&document, &signature, &secret())?;
    assert!(!result.core_valid());
    let validity: Vec<bool> = result.references().map(|r| r.valid()).collect();
    assert_eq!(validity, [true, false, true]);
    Ok(())
}

/// A self-signed certificate carrying the pinned test key, valid from
/// 2026-01-01 to 2036-01-01.
const CERTIFICATE_BASE64: &str = concat!(
    "MIIBpDCCAQ2gAwIBAgIFASNFZ4kwDQYJKoZIhvcNAQELBQAwFjEUMBIGA1UEAwwLT3hpWE1MIFRl",
    "c3QwHhcNMjYwMTAxMDAwMDAwWhcNMzYwMTAxMDAwMDAwWjAWMRQwEgYDVQQDDAtPeGlYTUwgVGVz",
    "dDCBnzANBgkqhkiG9w0BAQEFAAOBjQAwgYkCgYEAjkcoCslxEus5u8QIPMAYjDIpNgX12fbWalSP",
    "mhbKjlNeR5ukRjB0xBJqYvncXrPl7lHzDemgFsdEbJCvTgF1imtaMkc6yfaHkJ4+RgzPR2SBU4KE",
    "aVq69fNR5Dco0vl/VUZD54drPqBBvXhiqwpAK3ZLHcPSLxzRQi1a20l+tlUCAwEAATANBgkqhkiG",
    "9w0BAQsFAAOBgQAbgJ1S/Y0svcUZpowjNkA5BxrXfJNopeDC6Rxz77AHzTfqnwkVWkjAYxY8mfbe",
    "CQkEucKK2FxB8bv1WzDE/rlW5h8jtyxmvYA3d/imKCjC7SPcibgHh55nIkmwA9oXpLL5C4rM0fcA",
    "pbudoveWWwRHWAlxDiusBrr8vbzaDOByHA==",
);

/// Builds and signs a document whose `<ds:KeyInfo>` carries `der`.
fn certified_document(der: &[u8]) -> Result<(Document, NodeId), Failure> {
    let mut document = parse(r##"<doc><payload Id="p">certified</payload></doc>"##);
    let root = document.document_element().expect("a document element");
    let node = SignatureBuilder::new(SignatureAlgorithm::RsaSha256)
        .prefix("ds")
        .reference(ReferenceSpec::new("#p"))
        .key_info(KeyInfoSpec::X509Certificate(der.to_vec()))
        .build_into(&mut document, root)?;
    sign(&mut document, node, &SigningKey::rsa(rsa_key()?))?;
    Ok((document, node))
}

#[test]
fn a_signature_with_an_x509_certificate_key_info_round_trips() -> Result<(), Failure> {
    let der = oxixml_text::base64::base64_decode(CERTIFICATE_BASE64)?;

    let mut document = parse(r##"<doc><payload Id="p">certified</payload></doc>"##);
    let root = document.document_element().expect("a document element");
    let node = SignatureBuilder::new(SignatureAlgorithm::RsaSha256)
        .prefix("ds")
        .reference(ReferenceSpec::new("#p"))
        .key_info(KeyInfoSpec::X509Certificate(der.clone()))
        .build_into(&mut document, root)?;
    sign(&mut document, node, &SigningKey::rsa(rsa_key()?))?;

    let signature = Signature::parse(&document, node)?;
    // A store that trusts this exact certificate accepts it …
    let trusted = oxixml_dsig::TrustedCertificates::new().with_certificate(der.clone());
    assert!(
        Verifier::new()
            .verify(&document, &signature, &trusted)?
            .core_valid()
    );
    // … an empty store does not.
    assert!(
        Verifier::new()
            .verify(
                &document,
                &signature,
                &oxixml_dsig::TrustedCertificates::new()
            )
            .is_err()
    );
    Ok(())
}

/// The same certificate used as a *trust anchor* rather than as a pinned
/// encoding. The store now builds a certification path — one certificate long,
/// since this one signs itself — and checks it before the key is used at all.
#[test]
fn a_certificate_is_path_validated_before_its_key_is_used() -> Result<(), Failure> {
    let der = oxixml_text::base64::base64_decode(CERTIFICATE_BASE64)?;
    let (document, node) = certified_document(&der)?;
    let signature = Signature::parse(&document, node)?;

    // 2027, inside the certificate's validity period.
    let trusted = TrustedCertificates::new()
        .with_certificate(der.clone())
        .with_path_validation(PathValidator::new(Timestamp::from_unix_seconds(
            1_800_000_000,
        )));
    assert!(
        Verifier::new()
            .verify(&document, &signature, &trusted)?
            .core_valid()
    );

    // Outside it, verification does not merely report a caveat: it fails, with
    // an error naming the certificate and the dates.
    let lapsed = TrustedCertificates::new()
        .with_certificate(der)
        .with_path_validation(PathValidator::new(Timestamp::from_unix_seconds(0)));
    let error = Verifier::new()
        .verify(&document, &signature, &lapsed)
        .expect_err("the certificate is not valid until 2026");
    assert!(matches!(error, DsigError::Certificate(_)), "{error}");
    assert!(error.to_string().contains("CN=OxiXML Test"), "{error}");
    assert!(
        error.to_string().contains("2026-01-01T00:00:00Z"),
        "{error}"
    );
    Ok(())
}

/// A chain the store cannot reach an anchor from is refused outright — the leaf
/// is never trusted "anyway" because it happens to carry a usable key.
#[test]
fn a_certificate_that_reaches_no_anchor_is_refused() -> Result<(), Failure> {
    let der = oxixml_text::base64::base64_decode(CERTIFICATE_BASE64)?;
    let (document, node) = certified_document(&der)?;
    let signature = Signature::parse(&document, node)?;

    let at = PathValidator::new(Timestamp::from_unix_seconds(1_800_000_000));

    // A store with no anchors at all can trust nothing.
    let error = Verifier::new()
        .verify(
            &document,
            &signature,
            &TrustedCertificates::new().with_path_validation(at),
        )
        .expect_err("there is no anchor to reach");
    assert!(matches!(error, DsigError::Certificate(_)), "{error}");

    // An anchor for somebody else: the same encoding with one letter of its
    // *subject* changed, so the certificate in the document no longer names it
    // as its issuer. (Altering the anchor's serial or signature would change
    // nothing — RFC 5280 §6.1.1 takes only a name and a key from an anchor.)
    let mut other = der.clone();
    let subject = der
        .windows(b"OxiXML Test".len())
        .enumerate()
        .filter(|(_, window)| *window == b"OxiXML Test")
        .map(|(index, _)| index)
        .next_back()
        .expect("the subject common name");
    if let Some(byte) = other.get_mut(subject) {
        *byte = b'0';
    }
    assert_ne!(other, der);
    let elsewhere = TrustedCertificates::new()
        .with_certificate(other)
        .with_path_validation(at);
    let error = Verifier::new()
        .verify(&document, &signature, &elsewhere)
        .expect_err("the certificate chains to nothing in this store");
    assert!(matches!(error, DsigError::Certificate(_)), "{error}");
    assert!(error.to_string().contains("CN=OxiXML Test"), "{error}");
    Ok(())
}

#[test]
fn a_manifest_is_only_checked_when_asked() -> Result<(), Failure> {
    // A <ds:Manifest> inside a <ds:Object>, referenced by the SignedInfo.
    let mut document = parse(concat!(
        r##"<doc><payload Id="p">data</payload>"##,
        r##"<ds:Object xmlns:ds="http://www.w3.org/2000/09/xmldsig#" Id="m">"##,
        r##"<ds:Manifest Id="man"><ds:Reference URI="#p">"##,
        r##"<ds:DigestMethod Algorithm="http://www.w3.org/2001/04/xmlenc#sha256"/>"##,
        r##"<ds:DigestValue>47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU=</ds:DigestValue>"##,
        r##"</ds:Reference></ds:Manifest></ds:Object></doc>"##,
    ));
    let root = document.document_element().expect("a document element");
    let node = SignatureBuilder::new(SignatureAlgorithm::HmacSha256)
        .prefix("ds")
        .reference(ReferenceSpec::new("#man").reference_type(oxixml_dsig::MANIFEST_TYPE))
        .build_into(&mut document, root)?;
    sign(&mut document, node, &hmac_key())?;

    let signature = Signature::parse(&document, node)?;

    // Core validation covers the manifest's own digest and stops there, so the
    // deliberately wrong DigestValue inside it does not make the signature
    // invalid.
    let result = Verifier::new().verify(&document, &signature, &secret())?;
    assert!(result.core_valid());
    assert_eq!(result.manifests().len(), 0);

    // Asking for manifest validation surfaces the bad reference without
    // changing core validity.
    let result =
        Verifier::new()
            .validate_manifests(true)
            .verify(&document, &signature, &secret())?;
    assert!(result.core_valid());
    assert_eq!(result.manifests().len(), 1);
    assert!(result.manifests().all(|manifest| !manifest.valid()));
    Ok(())
}

#[test]
fn a_signature_survives_a_serialize_and_reparse() -> Result<(), Failure> {
    let mut document = parse(r##"<doc xmlns="urn:d"><p a="1" b="2">text</p></doc>"##);
    let root = document.document_element().expect("a document element");
    let node = SignatureBuilder::new(SignatureAlgorithm::HmacSha512)
        .prefix("ds")
        .reference(ReferenceSpec::new("").transform(TransformSpec::EnvelopedSignature))
        .build_into(&mut document, root)?;
    sign(&mut document, node, &hmac_key())?;

    let serialized = document.to_xml();
    let reparsed = parse(&serialized);
    let signature = Signature::find(&reparsed)?;
    assert!(
        Verifier::new()
            .verify(&reparsed, &signature, &secret())?
            .core_valid(),
        "{serialized}"
    );
    Ok(())
}

//! The cases where the answer must be "no".
//!
//! Verifiers are judged by what they reject, and XML Signature has a long list
//! of ways to be talked into accepting something it should not: signature
//! wrapping through duplicate identifiers, MAC truncation
//! ([CVE-2009-0217](https://nvd.nist.gov/vuln/detail/CVE-2009-0217)), transforms
//! that fetch remote resources, key material taken on trust from the document
//! being checked. Each of those has a test here.

mod common;

use oxixml_dsig::{
    DsigError, ReferenceSpec, SharedSecret, Signature, SignatureAlgorithm, SignatureBuilder,
    SigningKey, TransformSpec, UntrustedKeyInfo, Verifier, sign,
};

use common::{parse, parse_nested};

type Failure = Box<dyn std::error::Error>;

fn secret() -> SharedSecret {
    SharedSecret::new(b"k".to_vec())
}

#[test]
fn a_duplicated_identifier_is_refused_rather_than_guessed() -> Result<(), Failure> {
    // Sign a payload identified by `p` …
    let mut document = parse(r##"<doc><payload Id="p">genuine</payload></doc>"##);
    let root = document.document_element().expect("a document element");
    let node = SignatureBuilder::new(SignatureAlgorithm::HmacSha256)
        .prefix("ds")
        .reference(ReferenceSpec::new("#p"))
        .build_into(&mut document, root)?;
    sign(&mut document, node, &SigningKey::hmac(*b"k"))?;
    let signature = Signature::parse(&document, node)?;
    assert!(
        Verifier::new()
            .verify(&document, &signature, &secret())?
            .core_valid()
    );

    // … then add a second element with the same identifier, the classic
    // signature-wrapping move. The verifier must refuse to pick one.
    let wrapper = document.create_element("payload")?;
    document.set_attribute(wrapper, "Id", "p")?;
    document.set_text_content(wrapper, "forged")?;
    document.append_child(root, wrapper)?;

    let signature = Signature::parse(&document, node)?;
    let result = Verifier::new().verify(&document, &signature, &secret())?;
    assert!(!result.core_valid());
    let message = result
        .references()
        .next()
        .and_then(oxixml_dsig::ReferenceValidity::error)
        .unwrap_or_default()
        .to_owned();
    assert!(message.contains('2'), "{message}");

    // The escape hatch exists, and is documented as dangerous.
    let lenient =
        Verifier::new()
            .allow_duplicate_ids(true)
            .verify(&document, &signature, &secret())?;
    assert!(lenient.core_valid(), "the first element still digests");
    Ok(())
}

#[test]
fn a_short_hmac_output_length_is_deemed_invalid() -> Result<(), Failure> {
    // The signature below is genuine; only its truncation length is illegal.
    // Build it by hand, because `sign` refuses to produce one.
    let document = parse(concat!(
        r##"<ds:Signature xmlns:ds="http://www.w3.org/2000/09/xmldsig#"><ds:SignedInfo>"##,
        r##"<ds:CanonicalizationMethod Algorithm="http://www.w3.org/TR/2001/REC-xml-c14n-20010315"/>"##,
        r##"<ds:SignatureMethod Algorithm="http://www.w3.org/2000/09/xmldsig#hmac-sha1">"##,
        r##"<ds:HMACOutputLength>40</ds:HMACOutputLength></ds:SignatureMethod>"##,
        r##"<ds:Reference URI="#o">"##,
        r##"<ds:DigestMethod Algorithm="http://www.w3.org/2000/09/xmldsig#sha1"/>"##,
        r##"<ds:DigestValue>2jmj7l5rSw0yVb/vlWAYkK/YBwk=</ds:DigestValue></ds:Reference>"##,
        r##"</ds:SignedInfo><ds:SignatureValue>AAAAAAA=</ds:SignatureValue>"##,
        r##"<ds:Object Id="o"/></ds:Signature>"##,
    ));
    let signature = Signature::find(&document)?;
    assert_eq!(
        signature
            .signed_info()
            .signature_method()
            .hmac_output_length(),
        Some(40)
    );

    let result = Verifier::new().verify(&document, &signature, &secret())?;
    assert!(!result.core_valid());
    assert!(!result.signature_valid());
    let reason = result.signature_error().unwrap_or_default();
    assert!(reason.contains("80"), "{reason}");
    assert!(reason.contains("CVE-2009-0217"), "{reason}");
    Ok(())
}

#[test]
fn an_over_long_hmac_output_length_is_refused() -> Result<(), Failure> {
    let document = parse(concat!(
        r##"<ds:Signature xmlns:ds="http://www.w3.org/2000/09/xmldsig#"><ds:SignedInfo>"##,
        r##"<ds:CanonicalizationMethod Algorithm="http://www.w3.org/TR/2001/REC-xml-c14n-20010315"/>"##,
        r##"<ds:SignatureMethod Algorithm="http://www.w3.org/2000/09/xmldsig#hmac-sha1">"##,
        r##"<ds:HMACOutputLength>4096</ds:HMACOutputLength></ds:SignatureMethod>"##,
        r##"<ds:Reference URI="#o">"##,
        r##"<ds:DigestMethod Algorithm="http://www.w3.org/2000/09/xmldsig#sha1"/>"##,
        r##"<ds:DigestValue>2jmj7l5rSw0yVb/vlWAYkK/YBwk=</ds:DigestValue></ds:Reference>"##,
        r##"</ds:SignedInfo><ds:SignatureValue>AAAAAAA=</ds:SignatureValue>"##,
        r##"<ds:Object Id="o"/></ds:Signature>"##,
    ));
    let signature = Signature::find(&document)?;
    let result = Verifier::new().verify(&document, &signature, &secret())?;
    assert!(!result.signature_valid());
    assert!(
        result
            .signature_error()
            .unwrap_or_default()
            .contains("4096")
    );
    Ok(())
}

/// Builds a `<ds:Signature>` whose `SignatureMethod` is `method`, with an empty
/// signature value and one reference to an empty `<ds:Object>`.
fn signature_with_method(method: &str) -> String {
    format!(
        concat!(
            r##"<ds:Signature xmlns:ds="http://www.w3.org/2000/09/xmldsig#"><ds:SignedInfo>"##,
            r##"<ds:CanonicalizationMethod Algorithm="http://www.w3.org/TR/2001/REC-xml-c14n-20010315"/>"##,
            r##"<ds:SignatureMethod Algorithm="{}"/>"##,
            r##"<ds:Reference URI="#o">"##,
            r##"<ds:DigestMethod Algorithm="http://www.w3.org/2000/09/xmldsig#sha1"/>"##,
            r##"<ds:DigestValue>2jmj7l5rSw0yVb/vlWAYkK/YBwk=</ds:DigestValue></ds:Reference>"##,
            r##"</ds:SignedInfo><ds:SignatureValue>AA==</ds:SignatureValue>"##,
            r##"<ds:Object Id="o"/></ds:Signature>"##,
        ),
        method
    )
}

/// Every `SignatureMethod` XML Signature 1.1 defines is implemented, so an
/// unusable one is now a *keying* failure rather than an algorithm failure —
/// and it must still be an error, never a "not valid" verdict a caller could
/// mistake for a checked signature.
#[test]
fn a_method_with_no_key_of_the_right_kind_is_an_error_not_a_verdict() {
    for method in [
        "http://www.w3.org/2001/04/xmldsig-more#ecdsa-sha256",
        "http://www.w3.org/2000/09/xmldsig#dsa-sha1",
        "http://www.w3.org/2007/05/xmldsig-more#rsa-pss",
    ] {
        let source = signature_with_method(method);
        let document = parse(&source);
        let signature = Signature::find(&document).expect("the signature parses");
        assert!(
            signature
                .signed_info()
                .signature_method()
                .algorithm()
                .is_supported(),
            "{method}"
        );
        // A shared secret cannot key any of them.
        assert!(
            matches!(
                Verifier::new().verify(&document, &signature, &secret()),
                Err(DsigError::NoKey(_))
            ),
            "{method}"
        );
        // Nor can the document's own KeyInfo, which it does not have.
        assert!(
            matches!(
                Verifier::new().verify(&document, &signature, &UntrustedKeyInfo),
                Err(DsigError::NoKey(_))
            ),
            "{method}"
        );
    }
}

#[test]
fn an_unknown_algorithm_is_reported_at_parse_time() {
    let document = parse(concat!(
        r##"<ds:Signature xmlns:ds="http://www.w3.org/2000/09/xmldsig#"><ds:SignedInfo>"##,
        r##"<ds:CanonicalizationMethod Algorithm="http://www.w3.org/TR/2001/REC-xml-c14n-20010315"/>"##,
        r##"<ds:SignatureMethod Algorithm="urn:made:up"/>"##,
        r##"<ds:Reference URI="#o">"##,
        r##"<ds:DigestMethod Algorithm="http://www.w3.org/2000/09/xmldsig#sha1"/>"##,
        r##"<ds:DigestValue>2jmj7l5rSw0yVb/vlWAYkK/YBwk=</ds:DigestValue></ds:Reference>"##,
        r##"</ds:SignedInfo><ds:SignatureValue>AA==</ds:SignatureValue>"##,
        r##"<ds:Object Id="o"/></ds:Signature>"##,
    ));
    assert!(matches!(
        Signature::find(&document),
        Err(DsigError::UnknownAlgorithm(_))
    ));
}

#[test]
fn an_xslt_transform_is_refused() -> Result<(), Failure> {
    let document = parse(concat!(
        r##"<ds:Signature xmlns:ds="http://www.w3.org/2000/09/xmldsig#"><ds:SignedInfo>"##,
        r##"<ds:CanonicalizationMethod Algorithm="http://www.w3.org/TR/2001/REC-xml-c14n-20010315"/>"##,
        r##"<ds:SignatureMethod Algorithm="http://www.w3.org/2001/04/xmldsig-more#hmac-sha256"/>"##,
        r##"<ds:Reference URI="#o"><ds:Transforms>"##,
        r##"<ds:Transform Algorithm="http://www.w3.org/TR/1999/REC-xslt-19991116"/>"##,
        r##"</ds:Transforms>"##,
        r##"<ds:DigestMethod Algorithm="http://www.w3.org/2000/09/xmldsig#sha1"/>"##,
        r##"<ds:DigestValue>2jmj7l5rSw0yVb/vlWAYkK/YBwk=</ds:DigestValue></ds:Reference>"##,
        r##"</ds:SignedInfo><ds:SignatureValue>AA==</ds:SignatureValue>"##,
        r##"<ds:Object Id="o"/></ds:Signature>"##,
    ));
    let signature = Signature::find(&document)?;
    let result = Verifier::new().verify(&document, &signature, &secret())?;
    let error = result
        .references()
        .next()
        .and_then(oxixml_dsig::ReferenceValidity::error)
        .unwrap_or_default();
    assert!(error.contains("xslt"), "{error}");
    assert!(!result.core_valid());
    Ok(())
}

#[test]
fn an_external_reference_is_never_fetched_by_default() -> Result<(), Failure> {
    let document = parse(concat!(
        r##"<ds:Signature xmlns:ds="http://www.w3.org/2000/09/xmldsig#"><ds:SignedInfo>"##,
        r##"<ds:CanonicalizationMethod Algorithm="http://www.w3.org/TR/2001/REC-xml-c14n-20010315"/>"##,
        r##"<ds:SignatureMethod Algorithm="http://www.w3.org/2001/04/xmldsig-more#hmac-sha256"/>"##,
        r##"<ds:Reference URI="http://127.0.0.1:1/secret">"##,
        r##"<ds:DigestMethod Algorithm="http://www.w3.org/2000/09/xmldsig#sha1"/>"##,
        r##"<ds:DigestValue>2jmj7l5rSw0yVb/vlWAYkK/YBwk=</ds:DigestValue></ds:Reference>"##,
        r##"</ds:SignedInfo><ds:SignatureValue>AA==</ds:SignatureValue></ds:Signature>"##,
    ));
    let signature = Signature::find(&document)?;
    let result = Verifier::new().verify(&document, &signature, &secret())?;
    assert!(!result.core_valid());
    let error = result
        .references()
        .next()
        .and_then(oxixml_dsig::ReferenceValidity::error)
        .unwrap_or_default();
    assert!(error.contains("UriResolver"), "{error}");
    Ok(())
}

#[test]
fn a_shared_secret_never_stands_in_for_a_public_key() {
    let document = parse(concat!(
        r##"<ds:Signature xmlns:ds="http://www.w3.org/2000/09/xmldsig#"><ds:SignedInfo>"##,
        r##"<ds:CanonicalizationMethod Algorithm="http://www.w3.org/TR/2001/REC-xml-c14n-20010315"/>"##,
        r##"<ds:SignatureMethod Algorithm="http://www.w3.org/2001/04/xmldsig-more#rsa-sha256"/>"##,
        r##"<ds:Reference URI="#o">"##,
        r##"<ds:DigestMethod Algorithm="http://www.w3.org/2000/09/xmldsig#sha1"/>"##,
        r##"<ds:DigestValue>2jmj7l5rSw0yVb/vlWAYkK/YBwk=</ds:DigestValue></ds:Reference>"##,
        r##"</ds:SignedInfo><ds:SignatureValue>AA==</ds:SignatureValue>"##,
        r##"<ds:Object Id="o"/></ds:Signature>"##,
    ));
    let signature = Signature::find(&document).expect("the signature parses");
    assert!(matches!(
        Verifier::new().verify(&document, &signature, &secret()),
        Err(DsigError::NoKey(_))
    ));
}

#[test]
fn a_key_info_with_no_usable_key_is_reported_as_such() {
    let document = parse(concat!(
        r##"<ds:Signature xmlns:ds="http://www.w3.org/2000/09/xmldsig#"><ds:SignedInfo>"##,
        r##"<ds:CanonicalizationMethod Algorithm="http://www.w3.org/TR/2001/REC-xml-c14n-20010315"/>"##,
        r##"<ds:SignatureMethod Algorithm="http://www.w3.org/2001/04/xmldsig-more#rsa-sha256"/>"##,
        r##"<ds:Reference URI="#o">"##,
        r##"<ds:DigestMethod Algorithm="http://www.w3.org/2000/09/xmldsig#sha1"/>"##,
        r##"<ds:DigestValue>2jmj7l5rSw0yVb/vlWAYkK/YBwk=</ds:DigestValue></ds:Reference>"##,
        r##"</ds:SignedInfo><ds:SignatureValue>AA==</ds:SignatureValue>"##,
        r##"<ds:KeyInfo><ds:KeyValue><ds:DSAKeyValue><ds:Y>AA==</ds:Y></ds:DSAKeyValue>"##,
        r##"</ds:KeyValue></ds:KeyInfo><ds:Object Id="o"/></ds:Signature>"##,
    ));
    let signature = Signature::find(&document).expect("the signature parses");
    assert!(matches!(
        Verifier::new().verify(&document, &signature, &UntrustedKeyInfo),
        Err(DsigError::NoKey(_))
    ));
}

#[test]
fn a_degenerate_rsa_key_is_refused() {
    // n = 3, e = 3: arithmetic that would make everything "verify".
    let document = parse(concat!(
        r##"<ds:Signature xmlns:ds="http://www.w3.org/2000/09/xmldsig#"><ds:SignedInfo>"##,
        r##"<ds:CanonicalizationMethod Algorithm="http://www.w3.org/TR/2001/REC-xml-c14n-20010315"/>"##,
        r##"<ds:SignatureMethod Algorithm="http://www.w3.org/2001/04/xmldsig-more#rsa-sha256"/>"##,
        r##"<ds:Reference URI="#o">"##,
        r##"<ds:DigestMethod Algorithm="http://www.w3.org/2000/09/xmldsig#sha1"/>"##,
        r##"<ds:DigestValue>2jmj7l5rSw0yVb/vlWAYkK/YBwk=</ds:DigestValue></ds:Reference>"##,
        r##"</ds:SignedInfo><ds:SignatureValue>Aw==</ds:SignatureValue>"##,
        r##"<ds:KeyInfo><ds:KeyValue><ds:RSAKeyValue><ds:Modulus>Aw==</ds:Modulus>"##,
        r##"<ds:Exponent>Aw==</ds:Exponent></ds:RSAKeyValue></ds:KeyValue></ds:KeyInfo>"##,
        r##"<ds:Object Id="o"/></ds:Signature>"##,
    ));
    let signature = Signature::find(&document).expect("the signature parses");
    assert!(matches!(
        Verifier::new().verify(&document, &signature, &UntrustedKeyInfo),
        Err(DsigError::NoKey(_))
    ));
}

#[test]
fn malformed_base64_is_reported_with_the_element_that_carried_it() {
    let document = parse(concat!(
        r##"<ds:Signature xmlns:ds="http://www.w3.org/2000/09/xmldsig#"><ds:SignedInfo>"##,
        r##"<ds:CanonicalizationMethod Algorithm="http://www.w3.org/TR/2001/REC-xml-c14n-20010315"/>"##,
        r##"<ds:SignatureMethod Algorithm="http://www.w3.org/2001/04/xmldsig-more#hmac-sha256"/>"##,
        r##"<ds:Reference URI="#o">"##,
        r##"<ds:DigestMethod Algorithm="http://www.w3.org/2000/09/xmldsig#sha1"/>"##,
        r##"<ds:DigestValue>not base64 at all!</ds:DigestValue></ds:Reference>"##,
        r##"</ds:SignedInfo><ds:SignatureValue>AA==</ds:SignatureValue>"##,
        r##"<ds:Object Id="o"/></ds:Signature>"##,
    ));
    match Signature::find(&document) {
        Err(DsigError::Base64 { context, .. }) => assert_eq!(context, "<DigestValue>"),
        other => panic!("expected a base64 error, got {other:?}"),
    }
}

#[test]
fn a_variable_reference_in_an_xpath_transform_is_an_error() -> Result<(), Failure> {
    let document = parse(concat!(
        r##"<doc><ds:Signature xmlns:ds="http://www.w3.org/2000/09/xmldsig#"><ds:SignedInfo>"##,
        r##"<ds:CanonicalizationMethod Algorithm="http://www.w3.org/TR/2001/REC-xml-c14n-20010315"/>"##,
        r##"<ds:SignatureMethod Algorithm="http://www.w3.org/2001/04/xmldsig-more#hmac-sha256"/>"##,
        r##"<ds:Reference URI=""><ds:Transforms>"##,
        r##"<ds:Transform Algorithm="http://www.w3.org/TR/1999/REC-xpath-19991116">"##,
        r##"<ds:XPath>$forbidden</ds:XPath></ds:Transform></ds:Transforms>"##,
        r##"<ds:DigestMethod Algorithm="http://www.w3.org/2000/09/xmldsig#sha1"/>"##,
        r##"<ds:DigestValue>2jmj7l5rSw0yVb/vlWAYkK/YBwk=</ds:DigestValue></ds:Reference>"##,
        r##"</ds:SignedInfo><ds:SignatureValue>AA==</ds:SignatureValue></ds:Signature></doc>"##,
    ));
    let signature = Signature::find(&document)?;
    let result = Verifier::new().verify(&document, &signature, &secret())?;
    let error = result
        .references()
        .next()
        .and_then(oxixml_dsig::ReferenceValidity::error)
        .unwrap_or_default();
    assert!(error.contains("variable"), "{error}");
    Ok(())
}

#[test]
fn an_enveloped_transform_after_a_reparse_is_refused() -> Result<(), Failure> {
    // Canonicalization turns the node-set into octets; the following enveloped
    // transform would have to re-parse them, and the <ds:Signature> it is
    // supposed to remove would no longer be the same node.
    let document = parse(concat!(
        r##"<doc><ds:Signature xmlns:ds="http://www.w3.org/2000/09/xmldsig#"><ds:SignedInfo>"##,
        r##"<ds:CanonicalizationMethod Algorithm="http://www.w3.org/TR/2001/REC-xml-c14n-20010315"/>"##,
        r##"<ds:SignatureMethod Algorithm="http://www.w3.org/2001/04/xmldsig-more#hmac-sha256"/>"##,
        r##"<ds:Reference URI=""><ds:Transforms>"##,
        r##"<ds:Transform Algorithm="http://www.w3.org/TR/2001/REC-xml-c14n-20010315"/>"##,
        r##"<ds:Transform Algorithm="http://www.w3.org/2000/09/xmldsig#enveloped-signature"/>"##,
        r##"</ds:Transforms>"##,
        r##"<ds:DigestMethod Algorithm="http://www.w3.org/2000/09/xmldsig#sha1"/>"##,
        r##"<ds:DigestValue>2jmj7l5rSw0yVb/vlWAYkK/YBwk=</ds:DigestValue></ds:Reference>"##,
        r##"</ds:SignedInfo><ds:SignatureValue>AA==</ds:SignatureValue></ds:Signature></doc>"##,
    ));
    let signature = Signature::find(&document)?;
    let result = Verifier::new().verify(&document, &signature, &secret())?;
    let error = result
        .references()
        .next()
        .and_then(oxixml_dsig::ReferenceValidity::error)
        .unwrap_or_default();
    assert!(error.contains("enveloped"), "{error}");
    Ok(())
}

#[test]
fn moving_the_signature_breaks_an_enveloped_signature() -> Result<(), Failure> {
    let mut document = parse("<doc><a>one</a><b>two</b></doc>");
    let root = document.document_element().expect("a document element");
    let node = SignatureBuilder::new(SignatureAlgorithm::HmacSha256)
        .prefix("ds")
        .reference(ReferenceSpec::new("").transform(TransformSpec::EnvelopedSignature))
        .build_into(&mut document, root)?;
    sign(&mut document, node, &SigningKey::hmac(*b"k"))?;

    // The signature covers document order, so relocating it inside the
    // document must not change the digest — it is removed either way.
    let a = common::element_named(&document, "a");
    document.remove_child(root, node)?;
    document.append_child(a, node)?;
    let signature = Signature::parse(&document, node)?;
    assert!(
        Verifier::new()
            .verify(&document, &signature, &secret())?
            .core_valid()
    );

    // Reordering the *signed* content, though, must break it.
    let b = common::element_named(&document, "b");
    document.remove_child(root, b)?;
    document.insert_before(root, b, Some(a))?;
    let signature = Signature::parse(&document, node)?;
    assert!(
        !Verifier::new()
            .verify(&document, &signature, &secret())?
            .core_valid()
    );
    Ok(())
}

#[test]
fn a_reference_to_a_missing_element_is_reported_per_reference() -> Result<(), Failure> {
    let document = parse(concat!(
        r##"<ds:Signature xmlns:ds="http://www.w3.org/2000/09/xmldsig#"><ds:SignedInfo>"##,
        r##"<ds:CanonicalizationMethod Algorithm="http://www.w3.org/TR/2001/REC-xml-c14n-20010315"/>"##,
        r##"<ds:SignatureMethod Algorithm="http://www.w3.org/2001/04/xmldsig-more#hmac-sha256"/>"##,
        r##"<ds:Reference URI="#nowhere">"##,
        r##"<ds:DigestMethod Algorithm="http://www.w3.org/2000/09/xmldsig#sha1"/>"##,
        r##"<ds:DigestValue>2jmj7l5rSw0yVb/vlWAYkK/YBwk=</ds:DigestValue></ds:Reference>"##,
        r##"</ds:SignedInfo><ds:SignatureValue>AA==</ds:SignatureValue></ds:Signature>"##,
    ));
    let signature = Signature::find(&document)?;
    let result = Verifier::new().verify(&document, &signature, &secret())?;
    assert!(!result.core_valid());
    assert_eq!(result.references().len(), 1);
    let error = result
        .references()
        .next()
        .and_then(oxixml_dsig::ReferenceValidity::error)
        .unwrap_or_default();
    assert!(error.contains("nowhere"), "{error}");
    Ok(())
}

#[test]
fn comments_matter_exactly_where_the_recommendation_says_they_do() -> Result<(), Failure> {
    // `#x` strips comments, `#xpointer(id('x'))` keeps them — so adding a
    // comment inside the signed element breaks only the second one.
    for (uri, breaks) in [("#x", false), ("#xpointer(id('x'))", true)] {
        let mut document = parse(r##"<doc><p Id="x">text</p></doc>"##);
        let root = document.document_element().expect("a document element");
        let node = SignatureBuilder::new(SignatureAlgorithm::HmacSha256)
            .prefix("ds")
            .reference(ReferenceSpec::new(uri))
            .build_into(&mut document, root)?;
        sign(&mut document, node, &SigningKey::hmac(*b"k"))?;

        let paragraph = common::element_named(&document, "p");
        let comment = document.create_comment(" added later ")?;
        document.append_child(paragraph, comment)?;

        let signature = Signature::parse(&document, node)?;
        let valid = Verifier::new()
            .verify(&document, &signature, &secret())?
            .core_valid();
        assert_eq!(valid, !breaks, "{uri}");
    }
    Ok(())
}

/// A `<ds:XPath>` full of parentheses must fail the reference, not the process.
///
/// `Verifier::verify` validates references — which runs every transform, and so
/// the XPath parser — *before* it selects a key or checks the signature value,
/// so an attacker needs no key material at all to reach the parser. An
/// unbounded recursive descent turns a sub-2 kB `<ds:Signature>` into a stack
/// overflow, and a stack overflow aborts the process: it is not a `DsigError`
/// and it cannot even be caught with `catch_unwind`, so a verifying service has
/// no way to defend itself. The parser therefore bounds its own nesting.
#[test]
fn a_deeply_nested_xpath_transform_fails_the_reference_rather_than_the_process()
-> Result<(), Failure> {
    for levels in [700_usize, 10_000] {
        let mut document = parse("<doc><a>signed</a></doc>");
        let root = document.document_element().expect("a document element");
        let node = SignatureBuilder::new(SignatureAlgorithm::HmacSha256)
            .prefix("ds")
            .reference(ReferenceSpec::new("").transform(TransformSpec::xpath("true()")))
            .build_into(&mut document, root)?;
        sign(&mut document, node, &SigningKey::hmac(*b"k"))?;

        // The hostile edit: replace the benign expression with a parenthesis
        // chain. The document is no longer correctly signed, which is the
        // point — no key of ours matches it either.
        let expression = common::element_named(&document, "ds:XPath");
        document.set_text_content(
            expression,
            format!("{}true(){}", "(".repeat(levels), ")".repeat(levels)),
        )?;

        let signature = Signature::parse(&document, node)?;
        let wrong_key = SharedSecret::new(b"not the signing key".to_vec());
        let result = Verifier::new().verify(&document, &signature, &wrong_key)?;
        assert!(!result.core_valid(), "{levels} levels");
        let error = result
            .references()
            .next()
            .and_then(oxixml_dsig::ReferenceValidity::error)
            .unwrap_or_default()
            .to_owned();
        assert!(error.contains("nests more than"), "{levels}: {error}");
    }
    Ok(())
}

/// A deeply nested *document* must fail the same way.
///
/// Every reference is canonicalized before it is digested, so an ordinary
/// enveloped signature over a document that merely happens to nest deeply used
/// to abort the verifier — no unusual transform required.
#[test]
fn a_deeply_nested_document_verifies_without_overflowing_the_stack() -> Result<(), Failure> {
    let depth = 4_000;
    let mut document = parse_nested(
        &format!("<doc>{}{}</doc>", "<a>".repeat(depth), "</a>".repeat(depth)),
        depth + 2,
    );
    let root = document.document_element().expect("a document element");
    let node = SignatureBuilder::new(SignatureAlgorithm::HmacSha256)
        .prefix("ds")
        .reference(ReferenceSpec::new("").transform(TransformSpec::EnvelopedSignature))
        .build_into(&mut document, root)?;
    sign(&mut document, node, &SigningKey::hmac(*b"k"))?;

    let signature = Signature::parse(&document, node)?;
    assert!(
        Verifier::new()
            .verify(&document, &signature, &secret())?
            .core_valid()
    );

    // …and the same document under a key that does not match must simply come
    // back as "not valid", again without aborting.
    let wrong_key = SharedSecret::new(b"not the signing key".to_vec());
    assert!(
        !Verifier::new()
            .verify(&document, &signature, &wrong_key)?
            .core_valid()
    );
    Ok(())
}

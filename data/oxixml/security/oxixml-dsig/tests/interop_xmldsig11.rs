//! Verification against the W3C XML Signature 1.1 interoperability documents.
//!
//! The 45 documents under `xmldsig-core1-interop/interop/xmldsig11/oracle` were
//! produced by Oracle's JSR-105 implementation for the
//! [XML Security 1.1 interop round](https://www.w3.org/TR/xmldsig-core1/) and
//! are the best independent check available: nothing in this crate had any hand
//! in creating them, so a signature that verifies here agrees with a completely
//! separate implementation about canonicalization, reference processing,
//! `DigestInfo` encoding and PKCS#1 padding all at once.
//!
//! The suite is found through `OXIXML_XML_SUITES_DIR` and every test skips
//! silently when that is not set.
//!
//! # What is covered
//!
//! | family | expectation |
//! |---|---|
//! | RSASSA-PKCS1-v1_5, SHA-224/256/384/512 | core valid |
//! | HMAC-SHA-1/224/256/384/512 | core valid, except the deliberately truncated one |
//! | ECDSA over P-256 and P-384, keyed by `<dsig11:ECKeyValue>` or `<dsig11:DEREncodedKeyValue>` | core valid |
//! | ECDSA over P-521 | no key: the curve is a documented gap |
//! | ECDSA keyed by the withdrawn RFC 4050 `<ECDSAKeyValue>` | no key: the form is a documented gap |
//! | `<dsig11:X509Digest>` keying | no key: resolving it needs a certificate store |
//!
//! Every document that verifies is then *tampered with* in two independent
//! ways and re-checked, so a passing run also proves the verifier says no when
//! it should.

mod common;

use std::path::Path;

use oxixml_dsig::{
    DsigError, SharedSecret, Signature, SignatureFamily, UntrustedKeyInfo, VerificationResult,
    Verifier,
};

use common::{INTEROP_HMAC_SECRET, interop_directory, parse, tamper_text, xml_files};

/// What the suite expects of one document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Expectation {
    /// The signature must be core valid.
    Valid,
    /// The signature is well formed but must be *deemed invalid*, because its
    /// `HMACOutputLength` is below the floor of XML Signature §4.4.2.
    TruncatedHmac,
    /// The signature is well formed and its algorithm is implemented, but its
    /// keying material is a documented gap: an unsupported curve, a `KeyInfo`
    /// form this crate does not read, or a certificate reference that needs a
    /// store. Every one of those must fail closed, with no key at all rather
    /// than a key the verifier might mishandle.
    NoKey,
}

/// Classifies a document by its file name, which encodes the curve, the digest
/// and the `KeyInfo` form the suite used.
fn expectation(name: &str) -> Expectation {
    if name.contains("truncated40") {
        // The one document whose HMACOutputLength is below the mandatory floor.
        Expectation::TruncatedHmac
    } else if name.contains("x509digest")
        // The withdrawn RFC 4050 `<ECDSAKeyValue>`, with decimal coordinates.
        || name.contains("_4050")
        // P-521, which XML Signature 1.1 does not require.
        || name.contains("p521")
    {
        Expectation::NoKey
    } else {
        Expectation::Valid
    }
}

/// Verifies one document with whichever selector its `SignatureMethod` calls
/// for.
fn verify(path: &Path) -> Result<VerificationResult, DsigError> {
    let text = std::fs::read_to_string(path).expect("the interop document is readable");
    let document = parse(&text);
    let signature = Signature::find(&document)?;
    let algorithm = signature.signed_info().signature_method().algorithm();
    if matches!(algorithm.family(), SignatureFamily::Hmac) {
        Verifier::new().verify(
            &document,
            &signature,
            &SharedSecret::new(INTEROP_HMAC_SECRET.to_vec()),
        )
    } else {
        Verifier::new().verify(&document, &signature, &UntrustedKeyInfo)
    }
}

#[test]
fn every_interop_document_behaves_as_expected() {
    let Some(directory) = interop_directory() else {
        eprintln!("SKIP: OXIXML_XML_SUITES_DIR is not set or has no xmldsig-core1-interop");
        return;
    };

    let mut valid = 0_usize;
    let mut truncated = 0_usize;
    let mut no_key = 0_usize;
    let mut failures: Vec<String> = Vec::new();

    let files = xml_files(&directory);
    assert!(
        !files.is_empty(),
        "the interop directory holds no documents"
    );

    for path in &files {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_owned();
        let outcome = verify(path);
        match (expectation(&name), outcome) {
            (Expectation::Valid, Ok(result)) => {
                if result.core_valid() {
                    valid += 1;
                } else {
                    failures.push(format!(
                        "{name}: expected core validity, got signature={} references={:?}",
                        result.signature_valid(),
                        result
                            .references()
                            .map(|reference| (
                                reference.uri().map(str::to_owned),
                                reference.valid()
                            ))
                            .collect::<Vec<_>>()
                    ));
                }
            }
            (Expectation::TruncatedHmac, Ok(result)) => {
                if result.core_valid() {
                    failures.push(format!("{name}: a 40-bit HMAC truncation must be rejected"));
                } else if result.signature_error().is_some() {
                    truncated += 1;
                } else {
                    failures.push(format!(
                        "{name}: rejected, but not for the truncation-length reason"
                    ));
                }
            }
            (Expectation::NoKey, Err(DsigError::NoKey(_))) => no_key += 1,
            (expected, actual) => {
                failures.push(format!("{name}: expected {expected:?}, got {actual:?}"));
            }
        }
    }

    assert!(failures.is_empty(), "{}", failures.join("\n"));
    eprintln!(
        "xmldsig11 interop: {} documents — {valid} core valid, {truncated} rejected for a short \
         HMAC truncation, {no_key} with keying material this crate documents as a gap",
        files.len()
    );
    // HMAC and RSA, plus ECDSA on P-256 and P-384 through both KeyInfo forms.
    assert_eq!(valid, 26, "documents that must verify");
    assert_eq!(truncated, 1);
    // P-521, the RFC 4050 key form, and the X509Digest reference.
    assert_eq!(no_key, 18, "documents whose keying material is a gap");
}

#[test]
fn tampering_with_the_signed_object_breaks_every_valid_document() {
    let Some(directory) = interop_directory() else {
        eprintln!("SKIP: OXIXML_XML_SUITES_DIR is not set or has no xmldsig-core1-interop");
        return;
    };

    let mut checked = 0_usize;
    let mut failures: Vec<String> = Vec::new();
    for path in xml_files(&directory) {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_owned();
        if expectation(&name) != Expectation::Valid {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("readable");
        let mut document = parse(&text);
        // Every one of these is an enveloping signature over a <dsig:Object>
        // containing a <Web> element; change its text and the digest must move.
        let Some(web) = document
            .get_elements_by_tag_name(document.root(), "Web")
            .next()
        else {
            continue;
        };
        tamper_text(&mut document, web);

        let signature = Signature::find(&document).expect("the signature still parses");
        let algorithm = signature.signed_info().signature_method().algorithm();
        let result = if matches!(algorithm.family(), SignatureFamily::Hmac) {
            Verifier::new().verify(
                &document,
                &signature,
                &SharedSecret::new(INTEROP_HMAC_SECRET.to_vec()),
            )
        } else {
            Verifier::new().verify(&document, &signature, &UntrustedKeyInfo)
        }
        .expect("verification still runs");

        if result.core_valid() {
            failures.push(format!("{name}: tampered content still verified"));
        } else if result.signature_valid()
            && result.references().any(|reference| !reference.valid())
        {
            // Exactly the expected shape: SignedInfo is untouched, the
            // reference no longer digests.
            checked += 1;
        } else {
            failures.push(format!(
                "{name}: rejected, but not because the reference digest moved"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    eprintln!("xmldsig11 interop: {checked} documents correctly reject a tampered payload");
    // Every document the suite expects to verify, ECDSA included.
    assert_eq!(checked, 26);
}

#[test]
fn tampering_with_the_signature_value_breaks_every_valid_document() {
    let Some(directory) = interop_directory() else {
        eprintln!("SKIP: OXIXML_XML_SUITES_DIR is not set or has no xmldsig-core1-interop");
        return;
    };

    let mut checked = 0_usize;
    let mut failures: Vec<String> = Vec::new();
    for path in xml_files(&directory) {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_owned();
        if expectation(&name) != Expectation::Valid {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("readable");
        let mut document = parse(&text);
        let value = Signature::find(&document)
            .expect("the signature parses")
            .signature_value_node();
        // Re-encode the signature value with its first octet flipped.
        let original = document.text_content(value).unwrap_or_default();
        let mut octets = oxixml_text::base64::base64_decode_forgiving(&original)
            .expect("the interop signature value is base64");
        if let Some(first) = octets.first_mut() {
            *first ^= 0x80;
        }
        document
            .set_text_content(value, oxixml_text::base64::base64_encode(&octets))
            .expect("text content can be replaced");

        let signature = Signature::find(&document).expect("the signature still parses");
        let algorithm = signature.signed_info().signature_method().algorithm();
        let result = if matches!(algorithm.family(), SignatureFamily::Hmac) {
            Verifier::new().verify(
                &document,
                &signature,
                &SharedSecret::new(INTEROP_HMAC_SECRET.to_vec()),
            )
        } else {
            Verifier::new().verify(&document, &signature, &UntrustedKeyInfo)
        }
        .expect("verification still runs");

        if result.signature_valid() {
            failures.push(format!("{name}: a flipped SignatureValue still verified"));
        } else if result.references().all(|reference| reference.valid()) {
            checked += 1;
        } else {
            failures.push(format!(
                "{name}: the references broke too, which they should not"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    eprintln!("xmldsig11 interop: {checked} documents correctly reject a forged SignatureValue");
    // Every document the suite expects to verify, ECDSA included.
    assert_eq!(checked, 26);
}

#[test]
fn the_wrong_hmac_secret_never_verifies() {
    let Some(directory) = interop_directory() else {
        eprintln!("SKIP: OXIXML_XML_SUITES_DIR is not set or has no xmldsig-core1-interop");
        return;
    };

    let mut checked = 0_usize;
    for path in xml_files(&directory) {
        let text = std::fs::read_to_string(&path).expect("readable");
        let document = parse(&text);
        let Ok(signature) = Signature::find(&document) else {
            continue;
        };
        if !matches!(
            signature
                .signed_info()
                .signature_method()
                .algorithm()
                .family(),
            SignatureFamily::Hmac
        ) {
            continue;
        }
        let result = Verifier::new()
            .verify(
                &document,
                &signature,
                &SharedSecret::new(b"not the secret".to_vec()),
            )
            .expect("verification runs");
        assert!(!result.core_valid(), "{path:?} verified under a wrong key");
        checked += 1;
    }
    eprintln!("xmldsig11 interop: {checked} HMAC documents reject a wrong secret");
    assert!(checked >= 6);
}

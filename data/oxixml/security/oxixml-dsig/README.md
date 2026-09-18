# oxixml-dsig

XML Signature (XMLDSig Core 1.1), verification and generation, in Pure Rust.

An XML signature is not a signature over a file — it is a signature over the *meaning* of a
document, reached through a pipeline: dereference a `URI` to a node-set or an octet stream, push
it through a chain of transforms, canonicalize whatever comes out, digest that. Only then does
any cryptography happen, over `<ds:SignedInfo>` rather than over the content itself.
`oxixml-dsig` implements that pipeline exactly, on top of `oxixml-dom` for the tree and
`oxixml-c14n` for canonicalization. HMAC, RSASSA-PKCS1-v1_5 and RSASSA-PSS over
SHA-1/224/256/384/512, DSA over SHA-1/256, and ECDSA over P-256/P-384 (with all five digests) are
all implemented for both signing and verification, alongside X.509 parsing and RFC 5280 §6.1
certification-path validation; the XSLT transform is recognised and reported by name rather than
silently accepted.

Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace, the Pure-Rust foundation
for XML and RDF processing in the COOLJAPAN ecosystem.

## Install

```toml
[dependencies]
oxixml-dsig = "0.1.2"
```

## Example

```rust
use oxixml_dsig::{
    ReferenceSpec, SharedSecret, Signature, SignatureAlgorithm, SignatureBuilder,
    SigningKey, TransformSpec, Verifier, sign,
};
use oxixml_dom::build;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Sign a document in place — an *enveloped* signature, the common case.
    let mut document = build::parse_str("<order id=\"1\"><item>tea</item></order>")?;
    let root = document.document_element().expect("a document element");
    let node = SignatureBuilder::new(SignatureAlgorithm::HmacSha256)
        .prefix("ds")
        .reference(ReferenceSpec::new("").transform(TransformSpec::EnvelopedSignature))
        .build_into(&mut document, root)?;
    sign(&mut document, node, &SigningKey::hmac(*b"shared secret"))?;

    // Verify it. `SharedSecret` is the trust decision: this crate never picks a key
    // out of the document being checked unless you ask it to.
    let signature = Signature::find(&document)?;
    let result = Verifier::new().verify(
        &document,
        &signature,
        &SharedSecret::new(b"shared secret".to_vec()),
    )?;
    assert!(result.core_valid());

    // Change one character of the signed content and it stops verifying.
    let item = document.get_elements_by_tag_name(root, "item").next().expect("an item");
    document.set_text_content(item, "coffee")?;
    let signature = Signature::find(&document)?;
    let result = Verifier::new().verify(&document, &signature, &SharedSecret::new(b"shared secret".to_vec()))?;
    assert!(!result.core_valid());
    Ok(())
}
```

## What it replaces

`oxixml-dsig` replaces xmlsec1 (C) or Apache Santuario (Java) for signing and verifying XML
Signatures.

## Conformance

Verified against the 45 W3C XML Signature 1.1 interoperability documents (`tests/known_answers.rs`,
`tests/interop_xmldsig11.rs`) whenever `OXIXML_XML_SUITES_DIR` is set — produced by an
implementation sharing no code with this one. Last measured: of the 45 documents, **26 are core
valid** under the algorithms this crate implements, and each of those 26 correctly **stops**
verifying once its payload or its `SignatureValue` is tampered with; 1 further document is
correctly rejected for a truncated HMAC output length; the remaining 18 need keying material this
crate cannot resolve from the document alone (a documented gap — see the crate's rustdoc). The
signing direction is covered by round-trips across every algorithm, canonicalization method and
transform (`tests/roundtrip.rs`).

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](../../LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).

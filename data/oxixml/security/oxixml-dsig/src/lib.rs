//! `oxixml-dsig` — XML Signature (XMLDSig Core 1.1), verification and generation.
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace — the
//! COOLJAPAN Pure-Rust foundation for all XML and RDF processing.
//!
//! An XML signature is not a signature over a file. It is a signature over the
//! *meaning* of a document, reached through a pipeline that
//! [XML Signature §4.4.3.2](https://www.w3.org/TR/xmldsig-core1/#sec-ReferenceProcessingModel)
//! spells out: dereference a `URI` to a node-set or an octet stream, push it
//! through a chain of transforms, canonicalize whatever comes out, digest that.
//! Only then does any cryptography happen, and it happens over
//! `<ds:SignedInfo>` — a few hundred bytes listing the digests — rather than
//! over the content itself.
//!
//! That indirection is the whole design, and it is also where implementations
//! go wrong. This crate implements the pipeline exactly, on top of
//! [`oxixml_dom`] for the tree, [`oxixml_c14n`] for canonicalization, and
//! `oxixml_support`'s hashes and big integers for the arithmetic. There is no
//! `unsafe`, no C, and no dependency outside the workspace.
//!
//! # Example
//!
//! ```
//! use oxixml_dsig::{
//!     ReferenceSpec, SharedSecret, Signature, SignatureAlgorithm, SignatureBuilder,
//!     SigningKey, TransformSpec, Verifier, sign,
//! };
//! use oxixml_dom::build;
//!
//! // Sign a document in place — an *enveloped* signature, the common case.
//! let mut document = build::parse_str("<order id=\"1\"><item>tea</item></order>")?;
//! let root = document.document_element().expect("a document element");
//! let node = SignatureBuilder::new(SignatureAlgorithm::HmacSha256)
//!     .prefix("ds")
//!     .reference(ReferenceSpec::new("").transform(TransformSpec::EnvelopedSignature))
//!     .build_into(&mut document, root)?;
//! sign(&mut document, node, &SigningKey::hmac(*b"shared secret"))?;
//!
//! // Verify it. `SharedSecret` is the trust decision: this crate never picks a
//! // key out of the document being checked unless you ask it to.
//! let signature = Signature::find(&document)?;
//! let result = Verifier::new().verify(
//!     &document,
//!     &signature,
//!     &SharedSecret::new(b"shared secret".to_vec()),
//! )?;
//! assert!(result.core_valid());
//!
//! // Change one character of the signed content and it stops verifying.
//! let item = document.get_elements_by_tag_name(root, "item").next().expect("an item");
//! document.set_text_content(item, "coffee")?;
//! let signature = Signature::find(&document)?;
//! let result = Verifier::new().verify(
//!     &document,
//!     &signature,
//!     &SharedSecret::new(b"shared secret".to_vec()),
//! )?;
//! assert!(!result.core_valid());
//! assert!(result.signature_valid(), "SignedInfo itself was not touched");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # What is implemented
//!
//! | | |
//! |---|---|
//! | **digests** | SHA-1, SHA-224, SHA-256, SHA-384, SHA-512 |
//! | **signatures** | every `SignatureMethod` XML Signature 1.1 defines — HMAC-SHA-1/224/256/384/512, RSASSA-PKCS1-v1_5 over all five digests, RSASSA-PSS, DSA with SHA-1 and SHA-256, and ECDSA over P-256 and P-384 with all five digests — sign **and** verify |
//! | **canonicalization** | everything [`oxixml_c14n`] offers: Canonical XML 1.0, 1.1 and Exclusive 1.0, each with and without comments, plus 2.0 |
//! | **transforms** | enveloped signature, base64, canonicalization, XPath filtering, [XPath Filter 2.0](https://www.w3.org/TR/xmldsig-filter2/) |
//! | **references** | the null URI, shortname XPointers, `#xpointer(…)`, and external URIs through a caller-supplied [`UriResolver`] |
//! | **`KeyInfo`** | `KeyName`, `RSAKeyValue`, `DSAKeyValue`, `ECKeyValue`, `DEREncodedKeyValue`, `KeyInfoReference`, `X509Data` (certificate, subject name, issuer/serial, SKI, CRL, `X509Digest`), `RetrievalMethod`, and the elements it can only report |
//! | **certificates** | X.509 parsing and [RFC 5280 §6.1](https://www.rfc-editor.org/rfc/rfc5280#section-6.1) certification path validation — signatures, validity dates, names, basic constraints, key usage and critical extensions — through [`PathValidator`] |
//! | **manifests** | parsed always, validated on request |
//!
//! The XPath transforms are backed by a complete XPath 1.0 engine written for
//! this crate — thirteen axes, every node test, the whole core function library,
//! plus XML Signature's `here()`.
//!
//! # Gaps
//!
//! These are recognised and reported by name, never silently ignored and never
//! silently accepted:
//!
//! * **Curves other than P-256 and P-384** — the two XML Signature 1.1 makes
//!   REQUIRED are implemented; P-521 and the rest are not. A
//!   `<dsig11:ECKeyValue>` naming one still parses to
//!   [`KeyInfoItem::EcKeyValue`], so a caller can report the curve by name, but
//!   no key comes out of it. Only the uncompressed SEC 1 point encoding is read,
//!   which is the only one the Recommendation permits.
//! * **Explicit `<dsig11:ECParameters>`** — an EC key must name its curve; a key
//!   that spells the domain parameters out instead is parsed but not
//!   interpreted.
//! * **The RFC 4050 `<ECDSAKeyValue>`** — the withdrawn predecessor of
//!   `<dsig11:ECKeyValue>`, with decimal `X` and `Y` attributes, is not read.
//! * **XSLT transform** — recognised, not implemented. Running a stylesheet as
//!   part of verification means running attacker-supplied code, and it needs an
//!   XSLT processor this crate does not have.
//! * **`<dsig11:X509Digest>`, `<ds:X509SubjectName>`, `<ds:X509IssuerSerial>`,
//!   `<ds:X509SKI>`** — parsed and exposed, but resolving one to a key needs a
//!   certificate store. Supply the certificate through a [`KeySelector`]
//!   instead.
//! * **`<ds:RetrievalMethod>`** — parsed and exposed, never followed, for the
//!   same reason external `URI`s are never fetched.
//! * **Certificate revocation** — no CRL is fetched and no OCSP responder is
//!   asked, so a revoked certificate whose dates are still good passes path
//!   validation. A `<ds:X509CRL>` is parsed and handed to the caller in
//!   [`X509Data`] rather than acted on.
//! * **Name constraints and certificate policies** — `nameConstraints`,
//!   `certificatePolicies`, `policyMappings`, `policyConstraints` and
//!   `inhibitAnyPolicy` are not processed. A certificate that marks one of them
//!   critical is *refused* rather than under-checked, which is the fail-closed
//!   direction; see [`PathValidator`].
//! * **`authorityInfoAccess`** — never followed. Every certificate that takes
//!   part in a path is one the caller supplied or one the `<ds:KeyInfo>`
//!   carried.
//! * **`xmlns()` XPointer scheme** — a prefix inside `#xpointer(…)` resolves
//!   against the declarations in scope on the `<ds:Reference>` element rather
//!   than through an `xmlns()` part.
//!
//! # Security
//!
//! Five things are worth knowing before this crate is put in front of untrusted
//! input.
//!
//! **Core validity is not trust.** [`VerificationResult::core_valid`] says the
//! digests match and the `SignatureValue` checks out under *the key the
//! [`KeySelector`] returned*. Which key that is decides everything.
//! [`FixedKey`], [`SharedSecret`] and [`TrustedCertificates`] each encode a real
//! trust decision; [`UntrustedKeyInfo`] deliberately encodes none, and is named
//! to make that hard to miss.
//!
//! **A certificate chain is checked or it is refused.** Given a
//! [`PathValidator`], [`TrustedCertificates`] builds a certification path from
//! the signer's certificate to one of the anchors it was configured with and
//! applies RFC 5280 §6.1 to every link. A chain that fails any of those checks
//! yields a [`DsigError::Certificate`] naming the reason — it never falls back
//! to reading the key out of the leaf anyway. Without a validator the selector
//! keeps its original behaviour, which is stricter still: only a certificate
//! whose encoding the caller listed is accepted at all.
//!
//! **Nothing is fetched.** An external `URI` is dereferenced only when the
//! caller supplies a [`UriResolver`], so a hostile document cannot make the
//! verifier issue requests on its behalf. The XSLT transform is refused for the
//! same reason.
//!
//! **Truncated MACs are refused.**
//! [§4.4.2](https://www.w3.org/TR/xmldsig-core1/#sec-SignatureMethod) requires a
//! signature to be deemed invalid when its `HMACOutputLength` is below the
//! larger of half the hash length and 80 bits
//! ([CVE-2009-0217](https://nvd.nist.gov/vuln/detail/CVE-2009-0217)). It is, and
//! [`sign`] refuses to produce one.
//!
//! **Ambiguous identifiers are refused.** An `Id` that names two elements is the
//! raw material of an
//! [XML Signature Wrapping](https://www.ws-attacks.org/XML_Signature_Wrapping)
//! attack, so a reference to one fails rather than picking a winner. See
//! [Identifiers](#identifiers).
//!
//! **Signing is not constant time.** Verification runs entirely on public
//! values, so its timing reveals nothing. Signing does not: the big-integer and
//! elliptic-curve arithmetic underneath is variable time, and an attacker who can
//! measure a signing operation closely enough may learn about the key. What *is*
//! ruled out is the far more common catastrophe — DSA and ECDSA derive their
//! per-signature nonce deterministically per
//! [RFC 6979](https://www.rfc-editor.org/rfc/rfc6979) rather than from a random
//! source, so no repeated or biased nonce can publish the private key, and every
//! signature this crate produces is reproducible.
//!
//! SHA-1, DSA and 1024-bit RSA are implemented because existing documents use
//! them. None of the three is recommended for anything new.
//!
//! # Identifiers
//!
//! `URI="#foo"` is a shortname XPointer, which resolves against attributes of
//! type `ID`. A validating parser learns those from the DTD and `xml:id` is one
//! by definition — but essentially every real XML signature is produced without
//! a DTD and points at an attribute spelled `Id`, which is exactly how the XML
//! Signature schema itself declares `<ds:Signature Id>` and `<ds:Object Id>`.
//! So, like every interoperating implementation, this one also honours a list of
//! conventional names, [`DEFAULT_ID_ATTRIBUTES`] (`Id`, `ID`, `id`), overridable
//! through [`Verifier::id_attributes`].
//!
//! # Testing
//!
//! The verification direction is checked against the 45 W3C XML Signature 1.1
//! interoperability documents (discovered through `OXIXML_XML_SUITES_DIR`),
//! produced by an implementation that shares no code with this one; two of them
//! are embedded in the test suite so the coverage survives without it. Every
//! document that verifies is then tampered with — in the payload and in the
//! `SignatureValue`, separately — and must stop verifying. The signing direction
//! is covered by round-trips across every algorithm, canonicalization method and
//! transform.
//!
//! The cryptography underneath is checked against the published vectors for each
//! scheme rather than against itself: the deterministic ECDSA signatures of
//! [RFC 6979](https://www.rfc-editor.org/rfc/rfc6979#appendix-A.2) appendices
//! A.2.5 and A.2.6, the RSASSA-PSS signatures of the PKCS#1 v2.1 vector set, the
//! DSA example of FIPS 186-2 Appendix 5 — which XML Signature §6.4.1 reproduces
//! as its own worked example of a `SignatureValue` — and the NIST CAVP
//! `SigVer` vectors for DSA, whose deliberately corrupted rows must be rejected.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

mod algorithm;
mod crypto;
mod deref;
mod error;
mod idref;
mod key;
mod model;
mod sign;
#[cfg(test)]
mod testkey;
mod transform;
mod verify;
mod x509;
mod xpath;

pub use algorithm::{
    BASE64_TRANSFORM_URI, DSA_SHA1_URI, DSA_SHA256_URI, DSIG_NAMESPACE, DSIG11_NAMESPACE,
    DigestAlgorithm, ECDSA_SHA1_URI, ECDSA_SHA224_URI, ECDSA_SHA256_URI, ECDSA_SHA384_URI,
    ECDSA_SHA512_URI, ENVELOPED_SIGNATURE_URI, EXC_C14N_NAMESPACE, FILTER2_NAMESPACE,
    HMAC_SHA1_URI, HMAC_SHA224_URI, HMAC_SHA256_URI, HMAC_SHA384_URI, HMAC_SHA512_URI,
    MANIFEST_TYPE, MGF1_URI, OBJECT_TYPE, RSA_PSS_URI, RSA_SHA1_URI, RSA_SHA224_URI,
    RSA_SHA256_URI, RSA_SHA384_URI, RSA_SHA512_URI, SHA1_URI, SHA224_URI, SHA256_URI, SHA384_URI,
    SHA512_URI, SIGNATURE_PROPERTIES_TYPE, SignatureAlgorithm, SignatureFamily, TransformAlgorithm,
    XMLDSIG_MORE_NAMESPACE, XPATH_FILTER2_URI, XPATH_TRANSFORM_URI, XSLT_TRANSFORM_URI,
};
pub use deref::{StaticResolver, UriResolver};
pub use error::{DsigError, DsigResult};
pub use idref::DEFAULT_ID_ATTRIBUTES;
pub use key::{
    DsaPrivateKey, DsaPublicKey, EcPrivateKey, EcPublicKey, FixedKey, IssuerSerial, KeyContext,
    KeyInfo, KeyInfoItem, KeySelector, MAXIMUM_DSA_PRIME_BITS, MAXIMUM_DSA_SUBGROUP_BITS,
    MAXIMUM_RSA_MODULUS_BITS, MAXIMUM_RSA_PUBLIC_EXPONENT_BITS, MINIMUM_DSA_PRIME_BITS,
    MINIMUM_DSA_SUBGROUP_BITS, MINIMUM_RSA_MODULUS_BITS, NamedCurve, P256_NAMED_CURVE_URI,
    P384_NAMED_CURVE_URI, RsaPrivateKey, RsaPublicKey, SharedSecret, SigningKey,
    TrustedCertificates, UntrustedKeyInfo, VerificationKey, X509Data, named_curve_from_uri,
    named_curve_uri,
};
pub use model::{
    CanonicalizationMethod, Filter2Step, FilterOperation, Manifest, PssParameters, Reference,
    Signature, SignatureMethod, SignedInfo, Transform, TransformParameters, XPathParameter,
};
pub use sign::{
    Filter2Spec, KeyInfoSpec, ReferenceSpec, SignatureBuilder, TransformSpec, sign, sign_with,
};
pub use verify::{
    ManifestValidity, ReferenceValidity, VerificationResult, Verifier, verify_document,
    verify_signature,
};
pub use x509::{
    Certificate, CertificateError, CertificatePath, DistinguishedName, KeyUsage, PathValidator,
    Timestamp,
};

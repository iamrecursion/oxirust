//! Core validation — [XML Signature §3.2](https://www.w3.org/TR/xmldsig-core1/#sec-CoreValidation).
//!
//! The Recommendation splits validation cleanly in two, and this module keeps
//! the split visible in the result type:
//!
//! > **Reference Validation** … for each `Reference` in `SignedInfo`: apply the
//! > `Transforms`, calculate the digest, compare against `DigestValue`. If they
//! > do not match, validation fails.
//! >
//! > **Signature Validation** … canonicalize the `SignedInfo` element, obtain
//! > the keying information, and use it to confirm the `SignatureValue` over the
//! > `SignedInfo` element.
//!
//! Both must pass for the signature to be *core valid*. Keeping the two answers
//! apart matters in practice: a failed reference means the signed content
//! changed, a failed signature means the `SignedInfo` itself was tampered with
//! or the key is wrong, and telling a user "signature invalid" for the first
//! case sends them looking in the wrong place.
//!
//! # What core validity does *not* mean
//!
//! Core validity is a statement about arithmetic, not about trust. It says the
//! bytes hash to what the `SignedInfo` claims and that the `SignatureValue`
//! matches the key the [`KeySelector`] handed over. Whether that key belongs to
//! anyone you care about is decided entirely by the selector — see
//! [`KeySelector`] — and it is where every real deployment either gets
//! security or fails to.
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
//! // Produce an enveloped HMAC-SHA-256 signature …
//! let mut document = build::parse_str("<order><item>tea</item></order>")?;
//! let root = document.document_element().expect("a document element");
//! let node = SignatureBuilder::new(SignatureAlgorithm::HmacSha256)
//!     .prefix("ds")
//!     .reference(ReferenceSpec::new("").transform(TransformSpec::EnvelopedSignature))
//!     .build_into(&mut document, root)?;
//! sign(&mut document, node, &SigningKey::hmac(*b"secret"))?;
//!
//! // … and check it.
//! let signature = Signature::find(&document)?;
//! let result = Verifier::new().verify(
//!     &document,
//!     &signature,
//!     &SharedSecret::new(b"secret".to_vec()),
//! )?;
//! assert!(result.core_valid());
//! assert!(result.signature_valid());
//! assert!(result.references().all(|reference| reference.valid()));
//!
//! // Any edit to the signed content is caught.
//! let item = document.get_elements_by_tag_name(root, "item").next().expect("an item");
//! document.set_text_content(item, "coffee")?;
//! let signature = Signature::find(&document)?;
//! let result = Verifier::new().verify(
//!     &document,
//!     &signature,
//!     &SharedSecret::new(b"secret".to_vec()),
//! )?;
//! assert!(!result.core_valid());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use oxixml_dom::{Document, NodeId};

use crate::algorithm::{DigestAlgorithm, MANIFEST_TYPE, SignatureAlgorithm, SignatureFamily};
use crate::crypto::{digest, dsa, ecdsa, pkcs1, pss};
use crate::deref::{UriResolver, dereference};
use crate::error::{DsigError, DsigResult};
use crate::idref::{DEFAULT_ID_ATTRIBUTES, IdIndex};
use crate::key::{KeyContext, KeyInfo, KeyInfoItem, KeySelector, VerificationKey};
use crate::model::{Manifest, PssParameters, Reference, Signature, SignatureMethod};
use crate::transform::{TransformContext, apply, to_c14n_node_set};

/// The outcome of core validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationResult {
    core_valid: bool,
    signature_valid: bool,
    signature_error: Option<String>,
    signature_algorithm: SignatureAlgorithm,
    canonical_signed_info: Vec<u8>,
    references: Vec<ReferenceValidity>,
    manifests: Vec<ManifestValidity>,
}

impl VerificationResult {
    /// Whether the signature is *core valid*: every reference digests to what
    /// it claims **and** the `SignatureValue` checks out.
    ///
    /// This is the only method most callers need, and the only one it is safe
    /// to branch on.
    #[must_use]
    pub const fn core_valid(&self) -> bool {
        self.core_valid
    }

    /// Whether the `SignatureValue` matches the canonical `SignedInfo` under
    /// the selected key.
    #[must_use]
    pub const fn signature_valid(&self) -> bool {
        self.signature_valid
    }

    /// Why signature validation failed, when it failed for a reason other than
    /// the arithmetic simply not matching — an `HMACOutputLength` below the
    /// mandatory floor, for instance.
    #[must_use]
    pub fn signature_error(&self) -> Option<&str> {
        self.signature_error.as_deref()
    }

    /// The algorithm the `SignedInfo` named.
    #[must_use]
    pub const fn signature_algorithm(&self) -> SignatureAlgorithm {
        self.signature_algorithm
    }

    /// The canonical `SignedInfo` octets the signature was checked over.
    ///
    /// Useful when diagnosing an interoperability failure: two implementations
    /// that disagree about a signature almost always disagree about these
    /// bytes, not about the cryptography.
    #[must_use]
    pub fn canonical_signed_info(&self) -> &[u8] {
        &self.canonical_signed_info
    }

    /// The per-reference outcomes, in document order.
    pub fn references(&self) -> impl ExactSizeIterator<Item = &ReferenceValidity> {
        self.references.iter()
    }

    /// The per-manifest outcomes, empty unless
    /// [`Verifier::validate_manifests`] was turned on.
    pub fn manifests(&self) -> impl ExactSizeIterator<Item = &ManifestValidity> {
        self.manifests.iter()
    }
}

/// The outcome for one `<ds:Reference>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceValidity {
    uri: Option<String>,
    id: Option<String>,
    reference_type: Option<String>,
    digest_algorithm: DigestAlgorithm,
    expected_digest: Vec<u8>,
    computed_digest: Option<Vec<u8>>,
    error: Option<String>,
}

impl ReferenceValidity {
    /// Whether the computed digest equals the `<ds:DigestValue>`.
    #[must_use]
    pub fn valid(&self) -> bool {
        self.error.is_none()
            && self
                .computed_digest
                .as_ref()
                .is_some_and(|computed| digest::constant_time_eq(computed, &self.expected_digest))
    }

    /// The reference's `URI`.
    #[must_use]
    pub fn uri(&self) -> Option<&str> {
        self.uri.as_deref()
    }

    /// The reference's `Id`.
    #[must_use]
    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    /// The reference's `Type`.
    #[must_use]
    pub fn reference_type(&self) -> Option<&str> {
        self.reference_type.as_deref()
    }

    /// The digest algorithm the reference named.
    #[must_use]
    pub const fn digest_algorithm(&self) -> DigestAlgorithm {
        self.digest_algorithm
    }

    /// The `<ds:DigestValue>` the signature carries.
    #[must_use]
    pub fn expected_digest(&self) -> &[u8] {
        &self.expected_digest
    }

    /// The digest actually computed, or [`None`] if processing never got that
    /// far.
    #[must_use]
    pub fn computed_digest(&self) -> Option<&[u8]> {
        self.computed_digest.as_deref()
    }

    /// Why the reference could not be processed, if it could not.
    #[must_use]
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }
}

/// The outcome for one `<ds:Manifest>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestValidity {
    id: Option<String>,
    references: Vec<ReferenceValidity>,
}

impl ManifestValidity {
    /// The manifest's `Id`.
    #[must_use]
    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    /// Whether every reference in the manifest checks out.
    #[must_use]
    pub fn valid(&self) -> bool {
        self.references.iter().all(ReferenceValidity::valid)
    }

    /// The per-reference outcomes.
    pub fn references(&self) -> impl ExactSizeIterator<Item = &ReferenceValidity> {
        self.references.iter()
    }
}

/// Core validation, with its policy knobs.
///
/// [`Verifier::new`] gives the strict configuration, which is the one to use
/// unless a specific document forces otherwise. Every method that loosens it
/// documents what it gives up.
pub struct Verifier<'a> {
    id_attributes: Vec<String>,
    allow_duplicate_ids: bool,
    allow_short_hmac_output_length: bool,
    validate_manifests: bool,
    resolver: Option<&'a dyn UriResolver>,
}

impl Default for Verifier<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Verifier<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Verifier")
            .field("id_attributes", &self.id_attributes)
            .field("allow_duplicate_ids", &self.allow_duplicate_ids)
            .field(
                "allow_short_hmac_output_length",
                &self.allow_short_hmac_output_length,
            )
            .field("validate_manifests", &self.validate_manifests)
            .field("resolver", &self.resolver.is_some())
            .finish()
    }
}

impl<'a> Verifier<'a> {
    /// The strict configuration: conventional identifier attributes honoured,
    /// duplicate identifiers refused, short HMAC truncation refused, manifests
    /// not validated, no external dereferencing.
    #[must_use]
    pub fn new() -> Self {
        Self {
            id_attributes: DEFAULT_ID_ATTRIBUTES
                .iter()
                .map(|name| (*name).to_owned())
                .collect(),
            allow_duplicate_ids: false,
            allow_short_hmac_output_length: false,
            validate_manifests: false,
            resolver: None,
        }
    }

    /// Replaces the list of attribute names treated as identifiers on top of
    /// what the document declares. See [`crate::idref`](crate#identifiers) for
    /// why the list exists at all.
    #[must_use]
    pub fn id_attributes(mut self, names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.id_attributes = names.into_iter().map(Into::into).collect();
        self
    }

    /// Resolves an identifier that names several elements to the first one
    /// instead of refusing it.
    ///
    /// **This re-enables the classic XML Signature Wrapping attack.** A
    /// document with two elements sharing an `Id` lets an attacker make the
    /// verifier digest one and the application read the other. Turn this on
    /// only for a corpus you control and already know to be ambiguous.
    #[must_use]
    pub const fn allow_duplicate_ids(mut self, allow: bool) -> Self {
        self.allow_duplicate_ids = allow;
        self
    }

    /// Accepts an `HMACOutputLength` below the floor of
    /// [§4.4.2](https://www.w3.org/TR/xmldsig-core1/#sec-SignatureMethod).
    ///
    /// **This re-enables [CVE-2009-0217](https://nvd.nist.gov/vuln/detail/CVE-2009-0217).**
    /// The Recommendation requires a signature with a truncation length below
    /// the larger of half the hash length and 80 bits to be deemed invalid,
    /// because a truncated MAC can be brute-forced. There is no good reason to
    /// turn this on outside a test that wants to observe the old behaviour.
    #[must_use]
    pub const fn allow_short_hmac_output_length(mut self, allow: bool) -> Self {
        self.allow_short_hmac_output_length = allow;
        self
    }

    /// Also validates the references inside every `<ds:Manifest>` a reference
    /// points at.
    ///
    /// Core validation deliberately stops at the manifest's own digest —
    /// [§5.1](https://www.w3.org/TR/xmldsig-core1/#sec-Manifest) leaves what to
    /// do with the references inside it to the application. The results appear
    /// in [`VerificationResult::manifests`] and never affect
    /// [`VerificationResult::core_valid`].
    #[must_use]
    pub const fn validate_manifests(mut self, validate: bool) -> Self {
        self.validate_manifests = validate;
        self
    }

    /// Supplies the octets for references that are not same-document.
    #[must_use]
    pub const fn resolver(mut self, resolver: &'a dyn UriResolver) -> Self {
        self.resolver = Some(resolver);
        self
    }

    /// Runs core validation.
    ///
    /// # Errors
    ///
    /// [`DsigError`] when validation cannot be *performed*: an unsupported
    /// signature algorithm, a `SignedInfo` that will not canonicalize, a
    /// [`KeySelector`] that declines. A signature that simply does not verify
    /// is **not** an error — it is a [`VerificationResult`] whose
    /// [`core_valid`](VerificationResult::core_valid) is `false`.
    pub fn verify(
        &self,
        document: &Document,
        signature: &Signature,
        keys: &dyn KeySelector,
    ) -> DsigResult<VerificationResult> {
        let ids = IdIndex::build(document, &self.id_attributes, self.allow_duplicate_ids);
        let signed_info = signature.signed_info();
        let method = signed_info.signature_method();
        let algorithm = method.algorithm();

        if !algorithm.is_supported() {
            return Err(DsigError::UnsupportedAlgorithm(algorithm.uri().to_owned()));
        }

        // --- reference validation ---------------------------------------------------------
        let context = TransformContext {
            document,
            signature: signature.node(),
            ids: &ids,
            id_attributes: &self.id_attributes,
        };
        let mut references = Vec::new();
        let mut manifests = Vec::new();
        for reference in signed_info.references() {
            let validity = self.check_reference(document, reference, &context);
            if self.validate_manifests
                && validity.valid()
                && reference.reference_type() == Some(MANIFEST_TYPE)
            {
                if let Some(manifest) = self.locate_manifest(document, reference, &ids) {
                    manifests.push(ManifestValidity {
                        id: manifest.id().map(str::to_owned),
                        references: manifest
                            .references()
                            .map(|inner| self.check_reference(document, inner, &context))
                            .collect(),
                    });
                }
            }
            references.push(validity);
        }

        // --- signature validation ---------------------------------------------------------
        let canonical_signed_info = self.canonicalize_signed_info(document, signature)?;
        let key = keys.select(&KeyContext::new(
            algorithm,
            resolve_key_info(document, signature, &ids).as_ref(),
        ))?;
        if !key.matches(algorithm) {
            return Err(DsigError::NoKey(format!(
                "the selected key cannot be used with {algorithm}"
            )));
        }

        let (signature_valid, signature_error) = self.check_signature_value(
            method,
            &key,
            &canonical_signed_info,
            signature.signature_value(),
        );

        let core_valid = signature_valid && references.iter().all(ReferenceValidity::valid);
        Ok(VerificationResult {
            core_valid,
            signature_valid,
            signature_error,
            signature_algorithm: algorithm,
            canonical_signed_info,
            references,
            manifests,
        })
    }

    /// Canonicalizes `<ds:SignedInfo>` with the algorithm it names, as a subset
    /// of the document it is in — which is what pulls in the namespace
    /// declarations made on the enclosing `<ds:Signature>`.
    fn canonicalize_signed_info(
        &self,
        document: &Document,
        signature: &Signature,
    ) -> DsigResult<Vec<u8>> {
        let method = signature.signed_info().canonicalization_method();
        let subset = crate::transform::subtree_node_set(
            document,
            signature.signed_info().node(),
            method.algorithm().includes_comments(),
        );
        let subset = to_c14n_node_set(document, &subset);
        Ok(method
            .algorithm()
            .apply_with(document, &subset, method.inclusive_prefixes())?)
    }

    /// Dereferences, transforms and digests one reference.
    fn check_reference(
        &self,
        document: &Document,
        reference: &Reference,
        context: &TransformContext<'_>,
    ) -> ReferenceValidity {
        let mut validity = ReferenceValidity {
            uri: reference.uri().map(str::to_owned),
            id: reference.id().map(str::to_owned),
            reference_type: reference.reference_type().map(str::to_owned),
            digest_algorithm: reference.digest_algorithm(),
            expected_digest: reference.digest_value().to_vec(),
            computed_digest: None,
            error: None,
        };
        match self.compute_digest(document, reference, context) {
            Ok(computed) => validity.computed_digest = Some(computed),
            Err(error) => validity.error = Some(error.to_string()),
        }
        validity
    }

    fn compute_digest(
        &self,
        document: &Document,
        reference: &Reference,
        context: &TransformContext<'_>,
    ) -> DsigResult<Vec<u8>> {
        let mut data = dereference(document, reference, context.ids, self.resolver)?;
        for transform in reference.transforms() {
            data = apply(transform, data, context)?;
        }
        let octets = data.into_octets()?;
        Ok(digest::digest(reference.digest_algorithm(), &octets))
    }

    /// Finds the `<ds:Manifest>` a same-document reference points at.
    fn locate_manifest(
        &self,
        document: &Document,
        reference: &Reference,
        ids: &IdIndex,
    ) -> Option<Manifest> {
        let fragment = reference.uri()?.strip_prefix('#')?;
        let node = ids.get(fragment)?;
        Manifest::parse(document, node).ok()
    }

    /// Checks the `SignatureValue` against the canonical `SignedInfo`.
    fn check_signature_value(
        &self,
        method: &SignatureMethod,
        key: &VerificationKey,
        signed_info: &[u8],
        signature_value: &[u8],
    ) -> (bool, Option<String>) {
        let algorithm = method.algorithm();
        match (algorithm.family(), key) {
            (SignatureFamily::Hmac, VerificationKey::Hmac(secret)) => {
                let mut mac = digest::hmac(algorithm.digest(), secret, signed_info);
                if let Some(bits) = method.hmac_output_length() {
                    let floor = algorithm.minimum_hmac_output_length().unwrap_or(80);
                    if bits < floor && !self.allow_short_hmac_output_length {
                        return (
                            false,
                            Some(format!(
                                "HMACOutputLength is {bits} bits, below the mandatory floor of \
                                 {floor} bits (XML Signature §4.4.2, CVE-2009-0217)"
                            )),
                        );
                    }
                    if bits > mac.len() * 8 {
                        return (
                            false,
                            Some(format!(
                                "HMACOutputLength is {bits} bits, more than the {} bits {algorithm} \
                                 produces",
                                mac.len() * 8
                            )),
                        );
                    }
                    truncate_to_bits(&mut mac, bits);
                }
                (digest::constant_time_eq(&mac, signature_value), None)
            }
            (SignatureFamily::RsaPkcs1V15, VerificationKey::Rsa(public)) => (
                pkcs1::verify(
                    public.n(),
                    public.e(),
                    algorithm.digest(),
                    signed_info,
                    signature_value,
                ),
                None,
            ),
            (SignatureFamily::RsaPss, VerificationKey::Rsa(public)) => {
                let parameters = method.effective_pss_parameters();
                if parameters.trailer_field() != PssParameters::DEFAULT_TRAILER_FIELD {
                    return (
                        false,
                        Some(format!(
                            "the RSASSA-PSS TrailerField is {}, and RFC 8017 defines only 1",
                            parameters.trailer_field()
                        )),
                    );
                }
                (
                    pss::verify(
                        public.n(),
                        public.e(),
                        parameters,
                        signed_info,
                        signature_value,
                    ),
                    None,
                )
            }
            (SignatureFamily::Ecdsa, VerificationKey::Ec(public)) => (
                ecdsa::verify(
                    public.parameters(),
                    public.point(),
                    algorithm.digest(),
                    signed_info,
                    signature_value,
                ),
                None,
            ),
            (SignatureFamily::Dsa, VerificationKey::Dsa(public)) => {
                let Some(half_width) = algorithm.dsa_signature_half_width() else {
                    return (
                        false,
                        Some(format!("{algorithm} fixes no DSA signature width")),
                    );
                };
                if !public.fits_signature_width(half_width) {
                    return (
                        false,
                        Some(format!(
                            "the key's {}-bit subgroup order does not fit the {half_width}-octet \
                             halves {algorithm} fixes",
                            public.subgroup_bits()
                        )),
                    );
                }
                (
                    dsa::verify(
                        public.parameters(),
                        algorithm.digest(),
                        signed_info,
                        signature_value,
                        half_width,
                    ),
                    None,
                )
            }
            _ => (
                false,
                Some(format!("no key of the right kind for {algorithm}")),
            ),
        }
    }
}

/// Truncates a MAC to `bits` bits, keeping the leftmost ones and zeroing the
/// unused low bits of the final octet — RFC 2104 §5 truncation, which
/// XML Signature parameterises in bits rather than octets.
pub(crate) fn truncate_to_bits(mac: &mut Vec<u8>, bits: usize) {
    let full_octets = bits / 8;
    let spare_bits = bits % 8;
    let length = full_octets + usize::from(spare_bits != 0);
    mac.truncate(length);
    if spare_bits != 0 {
        if let Some(last) = mac.last_mut() {
            let keep = u8::try_from(spare_bits).unwrap_or(8);
            *last &= 0xff_u8 << (8 - keep);
        }
    }
}

/// Follows a same-document `<dsig11:KeyInfoReference>` so that a
/// [`KeySelector`] sees the `KeyInfo` that actually carries the key.
///
/// Only one hop is followed. A chain is not useful and a cycle is an obvious
/// denial of service, so the reference is resolved exactly once and the result
/// is used as-is.
fn resolve_key_info(document: &Document, signature: &Signature, ids: &IdIndex) -> Option<KeyInfo> {
    let key_info = signature.key_info()?;
    let Some(uri) = key_info.key_info_reference() else {
        return Some(key_info.clone());
    };
    let Some(fragment) = uri.strip_prefix('#') else {
        return Some(key_info.clone());
    };
    let Some(node) = ids.get(fragment) else {
        return Some(key_info.clone());
    };
    let referenced = crate::model::parse_key_info_at(document, node).ok()?;
    // Keep any hints the outer KeyInfo carried besides the reference itself.
    let mut merged = referenced;
    merged.extend_items(
        key_info
            .items()
            .filter(|item| !matches!(item, KeyInfoItem::KeyInfoReference { .. }))
            .cloned(),
    );
    Some(merged)
}

/// Verifies the first `<ds:Signature>` in `document` with the default
/// [`Verifier`].
///
/// # Errors
///
/// See [`Verifier::verify`], plus [`DsigError::Structure`] when the document
/// contains no `<ds:Signature>`.
pub fn verify_document(
    document: &Document,
    keys: &dyn KeySelector,
) -> DsigResult<VerificationResult> {
    let signature = Signature::find(document)?;
    Verifier::new().verify(document, &signature, keys)
}

/// Verifies one `<ds:Signature>` element with the default [`Verifier`].
///
/// # Errors
///
/// See [`Verifier::verify`].
pub fn verify_signature(
    document: &Document,
    signature: NodeId,
    keys: &dyn KeySelector,
) -> DsigResult<VerificationResult> {
    let signature = Signature::parse(document, signature)?;
    Verifier::new().verify(document, &signature, keys)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncation_keeps_the_leftmost_bits() {
        let mut mac = vec![0xff, 0xff, 0xff];
        truncate_to_bits(&mut mac, 8);
        assert_eq!(mac, [0xff]);

        let mut mac = vec![0xff, 0xff, 0xff];
        truncate_to_bits(&mut mac, 12);
        assert_eq!(mac, [0xff, 0xf0]);

        let mut mac = vec![0xff, 0xff, 0xff];
        truncate_to_bits(&mut mac, 24);
        assert_eq!(mac, [0xff, 0xff, 0xff]);
    }
}

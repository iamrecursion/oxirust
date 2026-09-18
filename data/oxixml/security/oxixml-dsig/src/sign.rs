//! Generating a signature — [XML Signature §3.1](https://www.w3.org/TR/xmldsig-core1/#sec-CoreGeneration).
//!
//! Core generation is core validation run backwards, and the Recommendation
//! describes it in the same two halves:
//!
//! > **Reference Generation** … apply the `Transforms` to the data object,
//! > calculate the digest, create a `Reference` element including the
//! > `DigestValue`.
//! >
//! > **Signature Generation** … canonicalize the `SignedInfo` element,
//! > calculate the `SignatureValue` over it, construct the `Signature` element.
//!
//! Because the digests go *inside* the `SignedInfo` that is then signed, the
//! order matters and the whole thing is naturally two passes over a template.
//! That is exactly the shape of the API here: [`SignatureBuilder`] writes the
//! template (with empty `<ds:DigestValue>` and `<ds:SignatureValue>`) into a
//! document, and [`sign`] fills it in.
//!
//! Splitting it that way is not just an implementation convenience. The
//! template is a real `<ds:Signature>` element in a real tree, so the enveloped
//! signature transform, the `here()` function and every `#id` reference resolve
//! against exactly the tree that will be published — which is the only way an
//! enveloped signature can be made to verify.
//!
//! # What can be signed
//!
//! Every `SignatureMethod` XML Signature 1.1 defines: HMAC, RSASSA-PKCS1-v1_5,
//! RSASSA-PSS, DSA and ECDSA over P-256 and P-384.
//!
//! All of them are **deterministic**. HMAC and RSASSA-PKCS1-v1_5 are so by
//! construction; DSA and ECDSA derive their nonce from the key and the message
//! per [RFC 6979](https://www.rfc-editor.org/rfc/rfc6979), and RSASSA-PSS derives
//! its salt the same way. Signing the same `SignedInfo` with the same key twice
//! therefore yields the same octets, which makes signatures reproducible and
//! means this crate needs no random number generator. See
//! [the security notes](crate#security) for what that does and does not buy.
//!
//! # Example
//!
//! ```
//! use oxixml_dsig::{
//!     DigestAlgorithm, ReferenceSpec, SharedSecret, Signature, SignatureAlgorithm,
//!     SignatureBuilder, SigningKey, TransformSpec, Verifier, sign,
//! };
//! use oxixml_dom::build;
//!
//! // An enveloped signature over a whole document.
//! let mut document = build::parse_str("<invoice id=\"i1\"><total>42</total></invoice>")?;
//! let root = document.document_element().expect("a document element");
//!
//! let node = SignatureBuilder::new(SignatureAlgorithm::HmacSha256)
//!     .prefix("ds")
//!     .reference(
//!         ReferenceSpec::new("")
//!             .digest(DigestAlgorithm::Sha256)
//!             .transform(TransformSpec::EnvelopedSignature)
//!             .transform(TransformSpec::canonicalization(
//!                 oxixml_c14n::CanonicalizationAlgorithm::Canonical1_0,
//!             )),
//!     )
//!     .build_into(&mut document, root)?;
//!
//! sign(&mut document, node, &SigningKey::hmac(*b"secret"))?;
//!
//! // …and it verifies.
//! let parsed = Signature::parse(&document, node)?;
//! let result = Verifier::new().verify(
//!     &document,
//!     &parsed,
//!     &SharedSecret::new(b"secret".to_vec()),
//! )?;
//! assert!(result.core_valid());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use oxixml_c14n::{CanonicalizationAlgorithm, InclusiveNamespaces};
use oxixml_dom::{Document, NodeId};

use crate::algorithm::{
    DSIG_NAMESPACE, DSIG11_NAMESPACE, DigestAlgorithm, EXC_C14N_NAMESPACE, FILTER2_NAMESPACE,
    MGF1_URI, SignatureAlgorithm, SignatureFamily, TransformAlgorithm, XMLDSIG_MORE_NAMESPACE,
};
use crate::crypto::{digest, dsa, ecdsa, pkcs1, pss};
use crate::deref::{UriResolver, dereference};
use crate::error::{DsigError, DsigResult};
use crate::idref::{DEFAULT_ID_ATTRIBUTES, IdIndex};
use crate::key::{DsaPublicKey, EcPublicKey, RsaPublicKey, SigningKey};
use crate::model::{FilterOperation, PssParameters, Signature, SignatureMethod};
use crate::transform::{TransformContext, apply, to_c14n_node_set};

/// One transform of a [`ReferenceSpec`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TransformSpec {
    /// The enveloped signature transform.
    EnvelopedSignature,
    /// The base64 decoding transform.
    Base64,
    /// A canonicalization algorithm, with the `InclusiveNamespaces` prefix list
    /// the exclusive variants take.
    Canonicalization(CanonicalizationAlgorithm, InclusiveNamespaces),
    /// The XPath filtering transform: the expression, plus the prefix bindings
    /// it needs, which are written onto the `<ds:XPath>` element.
    XPath {
        /// The expression.
        expression: String,
        /// `(prefix, namespace)` pairs declared on the `<ds:XPath>` element.
        namespaces: Vec<(String, String)>,
    },
    /// The XPath Filter 2.0 transform.
    Filter2(Vec<Filter2Spec>),
}

impl TransformSpec {
    /// A canonicalization transform with no prefix list.
    #[must_use]
    pub fn canonicalization(algorithm: CanonicalizationAlgorithm) -> Self {
        Self::Canonicalization(algorithm, InclusiveNamespaces::none())
    }

    /// An XPath filtering transform with no extra namespace bindings.
    #[must_use]
    pub fn xpath(expression: impl Into<String>) -> Self {
        Self::XPath {
            expression: expression.into(),
            namespaces: Vec::new(),
        }
    }

    fn algorithm(&self) -> TransformAlgorithm {
        match self {
            Self::EnvelopedSignature => TransformAlgorithm::EnvelopedSignature,
            Self::Base64 => TransformAlgorithm::Base64,
            Self::Canonicalization(algorithm, _) => {
                TransformAlgorithm::Canonicalization(*algorithm)
            }
            Self::XPath { .. } => TransformAlgorithm::XPath,
            Self::Filter2(_) => TransformAlgorithm::XPathFilter2,
        }
    }
}

/// One `<XPath Filter="…">` of an XPath Filter 2.0 transform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Filter2Spec {
    /// The set operation.
    pub filter: FilterOperation,
    /// The expression.
    pub expression: String,
    /// `(prefix, namespace)` pairs declared on the `<XPath>` element.
    pub namespaces: Vec<(String, String)>,
}

impl Filter2Spec {
    /// A step with no extra namespace bindings.
    #[must_use]
    pub fn new(filter: FilterOperation, expression: impl Into<String>) -> Self {
        Self {
            filter,
            expression: expression.into(),
            namespaces: Vec::new(),
        }
    }

    /// Declares a prefix on the `<XPath>` element.
    #[must_use]
    pub fn namespace(mut self, prefix: impl Into<String>, uri: impl Into<String>) -> Self {
        self.namespaces.push((prefix.into(), uri.into()));
        self
    }
}

/// One `<ds:Reference>` to generate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceSpec {
    uri: Option<String>,
    id: Option<String>,
    reference_type: Option<String>,
    transforms: Vec<TransformSpec>,
    digest: DigestAlgorithm,
}

impl ReferenceSpec {
    /// A reference with the given `URI`. Pass `""` for the whole containing
    /// document.
    #[must_use]
    pub fn new(uri: impl Into<String>) -> Self {
        Self {
            uri: Some(uri.into()),
            id: None,
            reference_type: None,
            transforms: Vec::new(),
            digest: DigestAlgorithm::Sha256,
        }
    }

    /// A reference with *no* `URI` attribute, whose data the application
    /// supplies through a [`UriResolver`].
    #[must_use]
    pub fn implicit() -> Self {
        Self {
            uri: None,
            id: None,
            reference_type: None,
            transforms: Vec::new(),
            digest: DigestAlgorithm::Sha256,
        }
    }

    /// Sets the `Id` attribute.
    #[must_use]
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Sets the `Type` attribute.
    #[must_use]
    pub fn reference_type(mut self, reference_type: impl Into<String>) -> Self {
        self.reference_type = Some(reference_type.into());
        self
    }

    /// Sets the digest algorithm. The default is SHA-256, which
    /// [§6.1](https://www.w3.org/TR/xmldsig-core1/#sec-AlgID) makes REQUIRED.
    #[must_use]
    pub const fn digest(mut self, digest: DigestAlgorithm) -> Self {
        self.digest = digest;
        self
    }

    /// Appends a transform. Order is significant.
    #[must_use]
    pub fn transform(mut self, transform: TransformSpec) -> Self {
        self.transforms.push(transform);
        self
    }
}

/// What to put in the `<ds:KeyInfo>`, if anything.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum KeyInfoSpec {
    /// `<ds:KeyName>`.
    KeyName(String),
    /// `<ds:KeyValue><ds:RSAKeyValue>`.
    RsaKeyValue(RsaPublicKey),
    /// `<ds:KeyValue><dsig11:ECKeyValue>`, with a `<dsig11:NamedCurve>` and the
    /// uncompressed point.
    EcKeyValue(EcPublicKey),
    /// `<ds:KeyValue><ds:DSAKeyValue>`, with the full domain parameters — a
    /// verifier that has to be told the key cannot be assumed to know them.
    DsaKeyValue(Box<DsaPublicKey>),
    /// `<ds:X509Data><ds:X509Certificate>`, from DER.
    X509Certificate(Vec<u8>),
}

/// Writes a `<ds:Signature>` template into a document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignatureBuilder {
    signature_method: SignatureAlgorithm,
    canonicalization: CanonicalizationAlgorithm,
    inclusive_prefixes: InclusiveNamespaces,
    hmac_output_length: Option<usize>,
    pss_parameters: Option<PssParameters>,
    prefix: Option<String>,
    id: Option<String>,
    references: Vec<ReferenceSpec>,
    key_info: Vec<KeyInfoSpec>,
}

impl SignatureBuilder {
    /// A builder for `algorithm`, canonicalizing `SignedInfo` with Canonical
    /// XML 1.0 and writing the signature in the default namespace.
    #[must_use]
    pub fn new(signature_method: SignatureAlgorithm) -> Self {
        Self {
            signature_method,
            canonicalization: CanonicalizationAlgorithm::Canonical1_0,
            inclusive_prefixes: InclusiveNamespaces::none(),
            hmac_output_length: None,
            pss_parameters: None,
            prefix: None,
            id: None,
            references: Vec::new(),
            key_info: Vec::new(),
        }
    }

    /// Sets the `<ds:CanonicalizationMethod>`.
    #[must_use]
    pub const fn canonicalization(mut self, algorithm: CanonicalizationAlgorithm) -> Self {
        self.canonicalization = algorithm;
        self
    }

    /// Sets the `InclusiveNamespaces` `PrefixList` for an exclusive
    /// `<ds:CanonicalizationMethod>`.
    #[must_use]
    pub fn inclusive_prefixes(mut self, prefixes: InclusiveNamespaces) -> Self {
        self.inclusive_prefixes = prefixes;
        self
    }

    /// Emits a `<ds:HMACOutputLength>` and truncates the MAC to that many bits.
    ///
    /// # Errors
    ///
    /// Nothing here, but [`sign`] refuses a length below the floor of
    /// [§4.4.2](https://www.w3.org/TR/xmldsig-core1/#sec-SignatureMethod) —
    /// producing a signature that every conforming verifier must reject is not
    /// a useful thing to be able to do.
    #[must_use]
    pub const fn hmac_output_length(mut self, bits: usize) -> Self {
        self.hmac_output_length = Some(bits);
        self
    }

    /// Emits a `<pss:RSAPSSParams>` and signs with those parameters.
    ///
    /// Only meaningful for [`SignatureAlgorithm::RsaPss`]. Leaving it unset
    /// signs with the defaults of
    /// [RFC 9231 §2.3.9](https://www.rfc-editor.org/rfc/rfc9231#section-2.3.9) —
    /// SHA-256, MGF1 over SHA-256 and a 32-octet salt — and writes no parameter
    /// element, which is what those defaults are for.
    #[must_use]
    pub const fn pss_parameters(mut self, parameters: PssParameters) -> Self {
        self.pss_parameters = Some(parameters);
        self
    }

    /// Writes the signature with a namespace prefix (`<ds:Signature>`) instead
    /// of a default declaration (`<Signature xmlns="…">`).
    #[must_use]
    pub fn prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = Some(prefix.into());
        self
    }

    /// Sets the `Id` of the `<ds:Signature>`.
    #[must_use]
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Appends a reference. At least one is required.
    #[must_use]
    pub fn reference(mut self, reference: ReferenceSpec) -> Self {
        self.references.push(reference);
        self
    }

    /// Appends a `<ds:KeyInfo>` child.
    #[must_use]
    pub fn key_info(mut self, item: KeyInfoSpec) -> Self {
        self.key_info.push(item);
        self
    }

    /// Builds the template and appends it to `parent`.
    ///
    /// The `<ds:DigestValue>` and `<ds:SignatureValue>` elements are created
    /// empty; [`sign`] fills them.
    ///
    /// # Errors
    ///
    /// [`DsigError::Structure`] when no reference was added,
    /// [`DsigError::Dom`] when the tree refuses the new nodes.
    pub fn build_into(&self, document: &mut Document, parent: NodeId) -> DsigResult<NodeId> {
        if self.references.is_empty() {
            return Err(DsigError::Structure(
                "a <ds:SignedInfo> needs at least one <ds:Reference>".to_owned(),
            ));
        }
        let writer = Writer {
            prefix: self.prefix.clone(),
        };
        let signature = writer.element(document, "Signature")?;
        document.declare_namespace(signature, self.prefix.as_deref(), DSIG_NAMESPACE)?;
        if let Some(id) = &self.id {
            document.set_attribute(signature, "Id", id.clone())?;
        }
        document.append_child(parent, signature)?;

        let signed_info = writer.child(document, signature, "SignedInfo")?;
        let c14n = writer.child(document, signed_info, "CanonicalizationMethod")?;
        document.set_attribute(c14n, "Algorithm", self.canonicalization.uri())?;
        if self.canonicalization.is_exclusive() && !self.inclusive_prefixes.is_empty() {
            let parameters =
                document.create_element_ns(Some(EXC_C14N_NAMESPACE), "ec:InclusiveNamespaces")?;
            document.declare_namespace(parameters, Some("ec"), EXC_C14N_NAMESPACE)?;
            document.set_attribute(
                parameters,
                "PrefixList",
                self.inclusive_prefixes.to_prefix_list(),
            )?;
            document.append_child(c14n, parameters)?;
        }

        let method = writer.child(document, signed_info, "SignatureMethod")?;
        document.set_attribute(method, "Algorithm", self.signature_method.uri())?;
        if let Some(bits) = self.hmac_output_length {
            let length = writer.child(document, method, "HMACOutputLength")?;
            writer.text(document, length, &bits.to_string())?;
        }
        if let Some(parameters) = self.pss_parameters {
            writer.pss_parameters(document, method, parameters)?;
        }

        for reference in &self.references {
            writer.reference(document, signed_info, reference)?;
        }

        writer.child(document, signature, "SignatureValue")?;

        if !self.key_info.is_empty() {
            let key_info = writer.child(document, signature, "KeyInfo")?;
            for item in &self.key_info {
                writer.key_info_item(document, key_info, item)?;
            }
        }
        Ok(signature)
    }
}

/// Emits `<ds:…>` elements with a consistent prefix.
struct Writer {
    prefix: Option<String>,
}

impl Writer {
    fn element(&self, document: &mut Document, local: &str) -> DsigResult<NodeId> {
        let qualified = match &self.prefix {
            Some(prefix) => format!("{prefix}:{local}"),
            None => local.to_owned(),
        };
        Ok(document.create_element_ns(Some(DSIG_NAMESPACE), &qualified)?)
    }

    fn child(&self, document: &mut Document, parent: NodeId, local: &str) -> DsigResult<NodeId> {
        let node = self.element(document, local)?;
        document.append_child(parent, node)?;
        Ok(node)
    }

    /// Appends a child in a namespace other than the signature's own, with the
    /// prefix that namespace conventionally uses.
    fn child_ns(
        &self,
        document: &mut Document,
        parent: NodeId,
        namespace: &str,
        prefix: &str,
        local: &str,
    ) -> DsigResult<NodeId> {
        let node = document.create_element_ns(Some(namespace), &format!("{prefix}:{local}"))?;
        document.append_child(parent, node)?;
        Ok(node)
    }

    fn text(&self, document: &mut Document, parent: NodeId, text: &str) -> DsigResult<()> {
        let node = document.create_text_node(text)?;
        document.append_child(parent, node)?;
        Ok(())
    }

    fn reference(
        &self,
        document: &mut Document,
        signed_info: NodeId,
        spec: &ReferenceSpec,
    ) -> DsigResult<NodeId> {
        let reference = self.child(document, signed_info, "Reference")?;
        if let Some(uri) = &spec.uri {
            document.set_attribute(reference, "URI", uri.clone())?;
        }
        if let Some(id) = &spec.id {
            document.set_attribute(reference, "Id", id.clone())?;
        }
        if let Some(reference_type) = &spec.reference_type {
            document.set_attribute(reference, "Type", reference_type.clone())?;
        }
        if !spec.transforms.is_empty() {
            let transforms = self.child(document, reference, "Transforms")?;
            for transform in &spec.transforms {
                self.transform(document, transforms, transform)?;
            }
        }
        let method = self.child(document, reference, "DigestMethod")?;
        document.set_attribute(method, "Algorithm", spec.digest.uri())?;
        self.child(document, reference, "DigestValue")?;
        Ok(reference)
    }

    fn transform(
        &self,
        document: &mut Document,
        parent: NodeId,
        spec: &TransformSpec,
    ) -> DsigResult<()> {
        let transform = self.child(document, parent, "Transform")?;
        document.set_attribute(transform, "Algorithm", spec.algorithm().uri())?;
        match spec {
            TransformSpec::EnvelopedSignature | TransformSpec::Base64 => {}
            TransformSpec::Canonicalization(algorithm, prefixes) => {
                if algorithm.is_exclusive() && !prefixes.is_empty() {
                    let parameters = document
                        .create_element_ns(Some(EXC_C14N_NAMESPACE), "ec:InclusiveNamespaces")?;
                    document.declare_namespace(parameters, Some("ec"), EXC_C14N_NAMESPACE)?;
                    document.set_attribute(parameters, "PrefixList", prefixes.to_prefix_list())?;
                    document.append_child(transform, parameters)?;
                }
            }
            TransformSpec::XPath {
                expression,
                namespaces,
            } => {
                let xpath = self.child(document, transform, "XPath")?;
                for (prefix, uri) in namespaces {
                    document.declare_namespace(xpath, Some(prefix), uri)?;
                }
                self.text(document, xpath, expression)?;
            }
            TransformSpec::Filter2(steps) => {
                for step in steps {
                    let xpath = document.create_element_ns(Some(FILTER2_NAMESPACE), "XPath")?;
                    document.declare_namespace(xpath, None, FILTER2_NAMESPACE)?;
                    document.set_attribute(
                        xpath,
                        "Filter",
                        match step.filter {
                            FilterOperation::Intersect => "intersect",
                            FilterOperation::Subtract => "subtract",
                            FilterOperation::Union => "union",
                        },
                    )?;
                    for (prefix, uri) in &step.namespaces {
                        document.declare_namespace(xpath, Some(prefix), uri)?;
                    }
                    document.append_child(transform, xpath)?;
                    self.text(document, xpath, &step.expression)?;
                }
            }
        }
        Ok(())
    }

    /// Writes the `<pss:RSAPSSParams>` of RFC 9231 §2.3.9.
    ///
    /// Every field is written, defaults included: the point of emitting the
    /// element at all is to say what was used, and a verifier that disagrees
    /// about a default would otherwise fail with nothing to look at.
    fn pss_parameters(
        &self,
        document: &mut Document,
        method: NodeId,
        parameters: PssParameters,
    ) -> DsigResult<()> {
        let element = self.child_ns(
            document,
            method,
            XMLDSIG_MORE_NAMESPACE,
            "pss",
            "RSAPSSParams",
        )?;
        document.declare_namespace(element, Some("pss"), XMLDSIG_MORE_NAMESPACE)?;

        let digest = self.child(document, element, "DigestMethod")?;
        document.set_attribute(digest, "Algorithm", parameters.digest().uri())?;

        let mask = self.child_ns(
            document,
            element,
            XMLDSIG_MORE_NAMESPACE,
            "pss",
            "MaskGenerationFunction",
        )?;
        document.set_attribute(mask, "Algorithm", MGF1_URI)?;
        let mask_digest = self.child(document, mask, "DigestMethod")?;
        document.set_attribute(mask_digest, "Algorithm", parameters.mask_digest().uri())?;

        let salt = self.child_ns(
            document,
            element,
            XMLDSIG_MORE_NAMESPACE,
            "pss",
            "SaltLength",
        )?;
        self.text(document, salt, &parameters.salt_octets().to_string())?;

        let trailer = self.child_ns(
            document,
            element,
            XMLDSIG_MORE_NAMESPACE,
            "pss",
            "TrailerField",
        )?;
        self.text(document, trailer, &parameters.trailer_field().to_string())?;
        Ok(())
    }

    fn key_info_item(
        &self,
        document: &mut Document,
        key_info: NodeId,
        spec: &KeyInfoSpec,
    ) -> DsigResult<()> {
        match spec {
            KeyInfoSpec::KeyName(name) => {
                let node = self.child(document, key_info, "KeyName")?;
                self.text(document, node, name)?;
            }
            KeyInfoSpec::RsaKeyValue(key) => {
                let value = self.child(document, key_info, "KeyValue")?;
                let rsa = self.child(document, value, "RSAKeyValue")?;
                let modulus = self.child(document, rsa, "Modulus")?;
                self.text(
                    document,
                    modulus,
                    &oxixml_text::base64::base64_encode(key.modulus()),
                )?;
                let exponent = self.child(document, rsa, "Exponent")?;
                self.text(
                    document,
                    exponent,
                    &oxixml_text::base64::base64_encode(key.exponent()),
                )?;
            }
            KeyInfoSpec::EcKeyValue(key) => {
                let value = self.child(document, key_info, "KeyValue")?;
                let ec =
                    self.child_ns(document, value, DSIG11_NAMESPACE, "dsig11", "ECKeyValue")?;
                document.declare_namespace(ec, Some("dsig11"), DSIG11_NAMESPACE)?;
                let named_curve =
                    self.child_ns(document, ec, DSIG11_NAMESPACE, "dsig11", "NamedCurve")?;
                document.set_attribute(named_curve, "URI", key.named_curve_uri())?;
                let public_key =
                    self.child_ns(document, ec, DSIG11_NAMESPACE, "dsig11", "PublicKey")?;
                self.text(
                    document,
                    public_key,
                    &oxixml_text::base64::base64_encode(key.encoded_point()),
                )?;
            }
            KeyInfoSpec::DsaKeyValue(key) => {
                let value = self.child(document, key_info, "KeyValue")?;
                let dsa = self.child(document, value, "DSAKeyValue")?;
                for (local, octets) in [
                    ("P", key.prime()),
                    ("Q", key.subgroup_order()),
                    ("G", key.generator()),
                    ("Y", key.public_value()),
                ] {
                    let element = self.child(document, dsa, local)?;
                    self.text(
                        document,
                        element,
                        &oxixml_text::base64::base64_encode(&octets),
                    )?;
                }
            }
            KeyInfoSpec::X509Certificate(der) => {
                let data = self.child(document, key_info, "X509Data")?;
                let certificate = self.child(document, data, "X509Certificate")?;
                self.text(
                    document,
                    certificate,
                    &oxixml_text::base64::base64_encode(der),
                )?;
            }
        }
        Ok(())
    }
}

/// Fills in a `<ds:Signature>` template: every `<ds:DigestValue>`, then the
/// `<ds:SignatureValue>`.
///
/// The template must already be in the tree in its final position: the
/// enveloped signature transform, `here()` and every `#id` reference resolve
/// against the tree as it stands, so signing a detached template and grafting
/// it in afterwards produces a signature that does not verify.
///
/// # Errors
///
/// [`DsigError::UnsupportedAlgorithm`] for a method this crate cannot compute,
/// [`DsigError::NoKey`] when `key` does not match the declared method,
/// [`DsigError::Policy`] for an `HMACOutputLength` below the mandatory floor,
/// and whatever reference processing reports.
pub fn sign(document: &mut Document, signature: NodeId, key: &SigningKey) -> DsigResult<()> {
    sign_with(document, signature, key, None)
}

/// [`sign`], with a resolver for references that are not same-document.
///
/// # Errors
///
/// See [`sign`].
pub fn sign_with(
    document: &mut Document,
    signature: NodeId,
    key: &SigningKey,
    resolver: Option<&dyn UriResolver>,
) -> DsigResult<()> {
    let id_attributes: Vec<String> = DEFAULT_ID_ATTRIBUTES
        .iter()
        .map(|name| (*name).to_owned())
        .collect();

    // Pass 1: every reference digest, computed against the tree as it stands.
    let digests = {
        let parsed = Signature::parse(document, signature)?;
        let algorithm = parsed.signed_info().signature_method().algorithm();
        if !algorithm.is_supported() {
            return Err(DsigError::UnsupportedAlgorithm(algorithm.uri().to_owned()));
        }
        if !key.matches(algorithm) {
            return Err(DsigError::NoKey(format!(
                "the signing key cannot be used with {algorithm}"
            )));
        }
        let ids = IdIndex::build(document, &id_attributes, false);
        let context = TransformContext {
            document,
            signature,
            ids: &ids,
            id_attributes: &id_attributes,
        };
        let mut digests = Vec::new();
        for reference in parsed.signed_info().references() {
            let mut data = dereference(document, reference, &ids, resolver)?;
            for transform in reference.transforms() {
                data = apply(transform, data, &context)?;
            }
            let octets = data.into_octets()?;
            digests.push((
                reference.digest_value_node(),
                digest::digest(reference.digest_algorithm(), &octets),
            ));
        }
        digests
    };

    for (node, value) in digests {
        set_text(document, node, &oxixml_text::base64::base64_encode(&value))?;
    }

    // Pass 2: the signature value over the now-complete SignedInfo.
    let (signature_value_node, octets) = {
        let parsed = Signature::parse(document, signature)?;
        let method = parsed.signed_info().canonicalization_method();
        let subset = crate::transform::subtree_node_set(
            document,
            parsed.signed_info().node(),
            method.algorithm().includes_comments(),
        );
        let subset = to_c14n_node_set(document, &subset);
        let canonical =
            method
                .algorithm()
                .apply_with(document, &subset, method.inclusive_prefixes())?;
        let value =
            compute_signature_value(parsed.signed_info().signature_method(), key, &canonical)?;
        (parsed.signature_value_node(), value)
    };
    set_text(
        document,
        signature_value_node,
        &oxixml_text::base64::base64_encode(&octets),
    )?;
    Ok(())
}

/// Computes the raw `SignatureValue` octets.
fn compute_signature_value(
    method: &SignatureMethod,
    key: &SigningKey,
    signed_info: &[u8],
) -> DsigResult<Vec<u8>> {
    let algorithm = method.algorithm();
    match (algorithm.family(), key) {
        (SignatureFamily::Hmac, SigningKey::Hmac(secret)) => {
            let mut mac = digest::hmac(algorithm.digest(), secret, signed_info);
            if let Some(bits) = method.hmac_output_length() {
                let floor = algorithm.minimum_hmac_output_length().unwrap_or(80);
                if bits < floor {
                    return Err(DsigError::Policy(format!(
                        "refusing to emit an HMACOutputLength of {bits} bits: XML Signature \
                         §4.4.2 requires at least {floor} and every conforming verifier must \
                         reject it"
                    )));
                }
                if bits > mac.len() * 8 {
                    return Err(DsigError::Policy(format!(
                        "HMACOutputLength is {bits} bits, more than the {} bits {algorithm} \
                         produces",
                        mac.len() * 8
                    )));
                }
                crate::verify::truncate_to_bits(&mut mac, bits);
            }
            Ok(mac)
        }
        (SignatureFamily::RsaPkcs1V15, SigningKey::Rsa(private)) => {
            pkcs1::sign(private.n(), private.d(), algorithm.digest(), signed_info).ok_or_else(
                || {
                    DsigError::Key(format!(
                        "the RSA modulus is too small to hold a {} DigestInfo",
                        algorithm.digest()
                    ))
                },
            )
        }
        (SignatureFamily::RsaPss, SigningKey::Rsa(private)) => {
            let parameters = method.effective_pss_parameters();
            if parameters.trailer_field() != PssParameters::DEFAULT_TRAILER_FIELD {
                return Err(DsigError::Policy(format!(
                    "refusing to emit an RSASSA-PSS TrailerField of {}: RFC 8017 defines only 1, \
                     and no verifier could check the result",
                    parameters.trailer_field()
                )));
            }
            pss::sign(private.n(), private.d(), parameters, signed_info).ok_or_else(|| {
                DsigError::Key(format!(
                    "the RSA modulus is too small to hold a {} digest and a {}-octet salt",
                    parameters.digest(),
                    parameters.salt_octets()
                ))
            })
        }
        (SignatureFamily::Ecdsa, SigningKey::Ec(private)) => ecdsa::sign(
            private.parameters(),
            private.scalar(),
            algorithm.digest(),
            signed_info,
        )
        .ok_or_else(|| DsigError::Key("the elliptic-curve private key is out of range".to_owned())),
        (SignatureFamily::Dsa, SigningKey::Dsa(private)) => {
            let half_width = algorithm
                .dsa_signature_half_width()
                .ok_or_else(|| DsigError::UnsupportedAlgorithm(algorithm.uri().to_owned()))?;
            if !private.fits_signature_width(half_width) {
                return Err(DsigError::Key(format!(
                    "the key's subgroup order does not fit the {half_width}-octet halves \
                     {algorithm} fixes"
                )));
            }
            dsa::sign(
                private.p(),
                private.q(),
                private.g(),
                private.x(),
                algorithm.digest(),
                signed_info,
                half_width,
            )
            .ok_or_else(|| DsigError::Key("the DSA private value is out of range".to_owned()))
        }
        _ => Err(DsigError::NoKey(format!(
            "the signing key cannot be used with {algorithm}"
        ))),
    }
}

/// Replaces the character content of an element with `text`.
fn set_text(document: &mut Document, element: NodeId, text: &str) -> DsigResult<()> {
    document.set_text_content(element, text)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::key::{RsaPrivateKey, SharedSecret, UntrustedKeyInfo};
    use crate::testkey;
    use crate::verify::Verifier;
    use oxixml_dom::build;

    fn document() -> Document {
        build::parse_str(r##"<invoice Id="i1"><total>42</total></invoice>"##)
            .expect("well-formed test input")
    }

    #[test]
    fn an_enveloped_hmac_signature_round_trips() -> DsigResult<()> {
        let mut document = document();
        let root = document.document_element().expect("a document element");
        let node = SignatureBuilder::new(SignatureAlgorithm::HmacSha256)
            .prefix("ds")
            .reference(
                ReferenceSpec::new("")
                    .transform(TransformSpec::EnvelopedSignature)
                    .transform(TransformSpec::canonicalization(
                        CanonicalizationAlgorithm::Canonical1_0,
                    )),
            )
            .build_into(&mut document, root)?;
        sign(&mut document, node, &SigningKey::hmac(*b"secret"))?;

        let parsed = Signature::parse(&document, node)?;
        let result =
            Verifier::new().verify(&document, &parsed, &SharedSecret::new(b"secret".to_vec()))?;
        assert!(result.core_valid(), "{result:?}");

        // The wrong secret must not verify.
        let result =
            Verifier::new().verify(&document, &parsed, &SharedSecret::new(b"guess".to_vec()))?;
        assert!(!result.signature_valid());
        assert!(result.references().all(crate::ReferenceValidity::valid));
        Ok(())
    }

    #[test]
    fn an_enveloping_rsa_signature_round_trips() -> DsigResult<()> {
        let mut document =
            build::parse_str(r##"<doc><payload Id="p">secret data</payload></doc>"##)
                .expect("well-formed test input");
        let root = document.document_element().expect("a document element");
        let key = RsaPrivateKey::from_der(&testkey::bytes(testkey::RSA_PRIVATE_KEY_BASE64))?;
        let node = SignatureBuilder::new(SignatureAlgorithm::RsaSha256)
            .reference(ReferenceSpec::new("#p").digest(DigestAlgorithm::Sha512))
            .key_info(KeyInfoSpec::RsaKeyValue(key.public_key()))
            .build_into(&mut document, root)?;
        sign(&mut document, node, &SigningKey::rsa(key))?;

        let parsed = Signature::parse(&document, node)?;
        let result = Verifier::new().verify(&document, &parsed, &UntrustedKeyInfo)?;
        assert!(result.core_valid(), "{result:?}");
        Ok(())
    }

    #[test]
    fn tampering_with_the_signed_content_breaks_the_reference() -> DsigResult<()> {
        let mut document = document();
        let root = document.document_element().expect("a document element");
        let node = SignatureBuilder::new(SignatureAlgorithm::HmacSha256)
            .reference(ReferenceSpec::new("").transform(TransformSpec::EnvelopedSignature))
            .build_into(&mut document, root)?;
        sign(&mut document, node, &SigningKey::hmac(*b"secret"))?;

        let total = document
            .get_elements_by_tag_name(root, "total")
            .next()
            .expect("a <total>");
        document.set_text_content(total, "43")?;

        let parsed = Signature::parse(&document, node)?;
        let result =
            Verifier::new().verify(&document, &parsed, &SharedSecret::new(b"secret".to_vec()))?;
        assert!(!result.core_valid());
        assert!(result.signature_valid(), "SignedInfo itself was untouched");
        assert!(result.references().all(|r| !r.valid()));
        Ok(())
    }

    #[test]
    fn an_xpath_filter_2_signature_round_trips() -> DsigResult<()> {
        let mut document =
            build::parse_str(r##"<doc><a>signed</a><b Id="volatile">changes</b></doc>"##)
                .expect("well-formed test input");
        let root = document.document_element().expect("a document element");
        let node = SignatureBuilder::new(SignatureAlgorithm::HmacSha256)
            .prefix("ds")
            .reference(
                ReferenceSpec::new("").transform(TransformSpec::Filter2(vec![
                    Filter2Spec::new(FilterOperation::Subtract, "id('volatile')"),
                    Filter2Spec::new(
                        FilterOperation::Subtract,
                        "here()/ancestor::ds:Signature[1]",
                    )
                    .namespace("ds", DSIG_NAMESPACE),
                ])),
            )
            .build_into(&mut document, root)?;
        sign(&mut document, node, &SigningKey::hmac(*b"k"))?;

        let parsed = Signature::parse(&document, node)?;
        let before =
            Verifier::new().verify(&document, &parsed, &SharedSecret::new(b"k".to_vec()))?;
        assert!(before.core_valid(), "{before:?}");

        // Changing the subtracted subtree must not break the signature …
        let volatile = document
            .get_elements_by_tag_name(root, "b")
            .next()
            .expect("a <b>");
        document.set_text_content(volatile, "changed")?;
        let parsed = Signature::parse(&document, node)?;
        let after =
            Verifier::new().verify(&document, &parsed, &SharedSecret::new(b"k".to_vec()))?;
        assert!(after.core_valid(), "{after:?}");

        // … but changing the signed part must.
        let signed = document
            .get_elements_by_tag_name(root, "a")
            .next()
            .expect("an <a>");
        document.set_text_content(signed, "tampered")?;
        let parsed = Signature::parse(&document, node)?;
        let tampered =
            Verifier::new().verify(&document, &parsed, &SharedSecret::new(b"k".to_vec()))?;
        assert!(!tampered.core_valid());
        Ok(())
    }

    #[test]
    fn a_short_hmac_output_length_is_refused_when_signing() {
        let mut document = document();
        let root = document.document_element().expect("a document element");
        let node = SignatureBuilder::new(SignatureAlgorithm::HmacSha256)
            .hmac_output_length(40)
            .reference(ReferenceSpec::new("").transform(TransformSpec::EnvelopedSignature))
            .build_into(&mut document, root)
            .expect("the template builds");
        assert!(matches!(
            sign(&mut document, node, &SigningKey::hmac(*b"secret")),
            Err(DsigError::Policy(_))
        ));
    }

    #[test]
    fn a_key_of_the_wrong_kind_is_refused() {
        let mut document = document();
        let root = document.document_element().expect("a document element");
        let node = SignatureBuilder::new(SignatureAlgorithm::RsaSha256)
            .reference(ReferenceSpec::new(""))
            .build_into(&mut document, root)
            .expect("the template builds");
        assert!(matches!(
            sign(&mut document, node, &SigningKey::hmac(*b"secret")),
            Err(DsigError::NoKey(_))
        ));
    }

    #[test]
    fn a_builder_with_no_reference_is_refused() {
        let mut document = document();
        let root = document.document_element().expect("a document element");
        assert!(matches!(
            SignatureBuilder::new(SignatureAlgorithm::HmacSha256).build_into(&mut document, root),
            Err(DsigError::Structure(_))
        ));
    }
}

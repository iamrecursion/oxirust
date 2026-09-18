//! The `<ds:Signature>` data model — [XML Signature §4](https://www.w3.org/TR/xmldsig-core1/#sec-CoreSyntax).
//!
//! ```text
//! <Signature Id?>
//!   <SignedInfo Id?>
//!     <CanonicalizationMethod Algorithm/>
//!     <SignatureMethod Algorithm>(<HMACOutputLength>?)</SignatureMethod>
//!     (<Reference Id? URI? Type?>
//!        (<Transforms><Transform Algorithm>…</Transform>+</Transforms>)?
//!        <DigestMethod Algorithm/>
//!        <DigestValue/>
//!      </Reference>)+
//!   </SignedInfo>
//!   <SignatureValue Id?/>
//!   (<KeyInfo Id?>…</KeyInfo>)?
//!   (<Object …/>)*
//! </Signature>
//! ```
//!
//! Every struct here keeps the [`NodeId`] it was parsed from, because
//! verification is not a function of the parsed values alone: `SignedInfo` has
//! to be canonicalized *as a subtree of the original document*, and a transform
//! may need the namespace declarations in scope at its own element. Parsing is
//! therefore a view over the tree, never a copy of it.
//!
//! # Example
//!
//! ```
//! use oxixml_dsig::{DigestAlgorithm, Signature, SignatureAlgorithm};
//! use oxixml_dom::build;
//!
//! let doc = build::parse_str(
//!     r##"<Signature xmlns="http://www.w3.org/2000/09/xmldsig#">
//!          <SignedInfo>
//!            <CanonicalizationMethod Algorithm="http://www.w3.org/2001/10/xml-exc-c14n#">
//!              <InclusiveNamespaces xmlns="http://www.w3.org/2001/10/xml-exc-c14n#"
//!                                   PrefixList="soap #default"/>
//!            </CanonicalizationMethod>
//!            <SignatureMethod Algorithm="http://www.w3.org/2000/09/xmldsig#hmac-sha1">
//!              <HMACOutputLength>160</HMACOutputLength>
//!            </SignatureMethod>
//!            <Reference URI="#payload">
//!              <Transforms>
//!                <Transform Algorithm="http://www.w3.org/2000/09/xmldsig#enveloped-signature"/>
//!              </Transforms>
//!              <DigestMethod Algorithm="http://www.w3.org/2001/04/xmlenc#sha256"/>
//!              <DigestValue>47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU=</DigestValue>
//!            </Reference>
//!          </SignedInfo>
//!          <SignatureValue>AAAA</SignatureValue>
//!        </Signature>"##,
//! )?;
//! let root = doc.document_element().expect("a document element");
//! let signature = Signature::parse(&doc, root)?;
//!
//! let method = signature.signed_info().signature_method();
//! assert_eq!(method.algorithm(), SignatureAlgorithm::HmacSha1);
//! assert_eq!(method.hmac_output_length(), Some(160));
//!
//! let prefixes = signature.signed_info().canonicalization_method().inclusive_prefixes();
//! assert!(prefixes.contains("soap") && prefixes.contains_default());
//!
//! let reference = signature.signed_info().references().next().expect("a Reference");
//! assert_eq!(reference.uri(), Some("#payload"));
//! assert_eq!(reference.digest_algorithm(), DigestAlgorithm::Sha256);
//! assert_eq!(reference.transforms().count(), 1);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod parse;

pub(crate) use parse::parse_key_info_at;

use oxixml_c14n::{CanonicalizationAlgorithm, InclusiveNamespaces};
use oxixml_dom::NodeId;

use crate::algorithm::{DigestAlgorithm, SignatureAlgorithm, TransformAlgorithm};
use crate::key::KeyInfo;

/// A parsed `<ds:Signature>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signature {
    pub(crate) node: NodeId,
    pub(crate) id: Option<String>,
    pub(crate) signed_info: SignedInfo,
    pub(crate) signature_value: Vec<u8>,
    pub(crate) signature_value_node: NodeId,
    pub(crate) key_info: Option<KeyInfo>,
    pub(crate) objects: Vec<NodeId>,
}

impl Signature {
    /// The `<ds:Signature>` element itself.
    #[must_use]
    pub const fn node(&self) -> NodeId {
        self.node
    }

    /// The `Id` attribute, if any.
    #[must_use]
    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    /// The `<ds:SignedInfo>` — the only part of the document the signature
    /// value actually covers.
    #[must_use]
    pub const fn signed_info(&self) -> &SignedInfo {
        &self.signed_info
    }

    /// The `<ds:SignatureValue>`, base64 decoded.
    #[must_use]
    pub fn signature_value(&self) -> &[u8] {
        &self.signature_value
    }

    /// The `<ds:SignatureValue>` element.
    #[must_use]
    pub const fn signature_value_node(&self) -> NodeId {
        self.signature_value_node
    }

    /// The `<ds:KeyInfo>`, if the signature carries one.
    #[must_use]
    pub const fn key_info(&self) -> Option<&KeyInfo> {
        self.key_info.as_ref()
    }

    /// The `<ds:Object>` children, in document order.
    #[must_use]
    pub fn objects(&self) -> &[NodeId] {
        &self.objects
    }
}

/// A parsed `<ds:SignedInfo>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignedInfo {
    pub(crate) node: NodeId,
    pub(crate) id: Option<String>,
    pub(crate) canonicalization_method: CanonicalizationMethod,
    pub(crate) signature_method: SignatureMethod,
    pub(crate) references: Vec<Reference>,
}

impl SignedInfo {
    /// The `<ds:SignedInfo>` element.
    #[must_use]
    pub const fn node(&self) -> NodeId {
        self.node
    }

    /// The `Id` attribute, if any.
    #[must_use]
    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    /// The `<ds:CanonicalizationMethod>`.
    #[must_use]
    pub const fn canonicalization_method(&self) -> &CanonicalizationMethod {
        &self.canonicalization_method
    }

    /// The `<ds:SignatureMethod>`.
    #[must_use]
    pub const fn signature_method(&self) -> &SignatureMethod {
        &self.signature_method
    }

    /// The `<ds:Reference>` children, in document order. There is always at
    /// least one.
    pub fn references(&self) -> impl ExactSizeIterator<Item = &Reference> {
        self.references.iter()
    }
}

/// A parsed `<ds:CanonicalizationMethod>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalizationMethod {
    pub(crate) node: NodeId,
    pub(crate) algorithm: CanonicalizationAlgorithm,
    pub(crate) inclusive_prefixes: InclusiveNamespaces,
}

impl CanonicalizationMethod {
    /// The `<ds:CanonicalizationMethod>` element.
    #[must_use]
    pub const fn node(&self) -> NodeId {
        self.node
    }

    /// The algorithm the `Algorithm` attribute names.
    #[must_use]
    pub const fn algorithm(&self) -> CanonicalizationAlgorithm {
        self.algorithm
    }

    /// The `<ec:InclusiveNamespaces PrefixList="…">` parameter, empty unless
    /// the algorithm is one of the exclusive ones and the parameter is present.
    #[must_use]
    pub const fn inclusive_prefixes(&self) -> &InclusiveNamespaces {
        &self.inclusive_prefixes
    }
}

/// The parameters of an RSASSA-PSS `<ds:SignatureMethod>`.
///
/// [RFC 9231 §2.3.9](https://www.rfc-editor.org/rfc/rfc9231#section-2.3.9) makes
/// the digest, the mask generation function, the salt length and the trailer
/// field explicit parameters, carried in a `<pss:RSAPSSParams>` child of the
/// `<ds:SignatureMethod>`, and gives every one of them a default:
///
/// | parameter | default |
/// |---|---|
/// | `ds:DigestMethod` | SHA-256 — *not* RFC 8017's SHA-1 |
/// | `MaskGenerationFunction` | MGF1 over the same hash as the digest method |
/// | `SaltLength` | the digest's output length in octets |
/// | `TrailerField` | 1, the only value RFC 8017 defines |
///
/// [`PssParameters::default`] is exactly that set, which the RFC notes is
/// equivalent to the parameterless `#sha256-rsa-MGF1` identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PssParameters {
    pub(crate) digest: DigestAlgorithm,
    pub(crate) mask_generation_digest: DigestAlgorithm,
    pub(crate) salt_length: usize,
    pub(crate) trailer_field: u32,
}

impl PssParameters {
    /// The only trailer field RFC 8017 defines, encoded as the octet `0xbc`.
    pub const DEFAULT_TRAILER_FIELD: u32 = 1;

    /// The parameter set RFC 9231 implies for a digest: MGF1 over the same
    /// hash, a salt as long as the digest, and trailer field 1.
    #[must_use]
    pub const fn new(digest: DigestAlgorithm) -> Self {
        Self {
            digest,
            mask_generation_digest: digest,
            salt_length: digest.output_length(),
            trailer_field: Self::DEFAULT_TRAILER_FIELD,
        }
    }

    /// Uses a different hash inside MGF1 than for the message digest, which
    /// RFC 8017 permits and RFC 9231 carries as
    /// `<MaskGenerationFunction><ds:DigestMethod/></MaskGenerationFunction>`.
    #[must_use]
    pub const fn mask_generation_digest(mut self, digest: DigestAlgorithm) -> Self {
        self.mask_generation_digest = digest;
        self
    }

    /// Sets the salt length in octets.
    #[must_use]
    pub const fn salt_length(mut self, octets: usize) -> Self {
        self.salt_length = octets;
        self
    }

    /// The digest the message is hashed with.
    #[must_use]
    pub const fn digest(self) -> DigestAlgorithm {
        self.digest
    }

    /// The hash MGF1 is built on.
    #[must_use]
    pub const fn mask_digest(self) -> DigestAlgorithm {
        self.mask_generation_digest
    }

    /// The salt length in octets.
    #[must_use]
    pub const fn salt_octets(self) -> usize {
        self.salt_length
    }

    /// The trailer field. Anything other than
    /// [`DEFAULT_TRAILER_FIELD`](PssParameters::DEFAULT_TRAILER_FIELD) makes the
    /// signature unverifiable, because RFC 8017 defines no other value.
    #[must_use]
    pub const fn trailer_field(self) -> u32 {
        self.trailer_field
    }
}

impl Default for PssParameters {
    fn default() -> Self {
        Self::new(DigestAlgorithm::Sha256)
    }
}

/// A parsed `<ds:SignatureMethod>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignatureMethod {
    pub(crate) node: NodeId,
    pub(crate) algorithm: SignatureAlgorithm,
    pub(crate) hmac_output_length: Option<usize>,
    pub(crate) pss_parameters: Option<PssParameters>,
}

impl SignatureMethod {
    /// The `<ds:SignatureMethod>` element.
    #[must_use]
    pub const fn node(&self) -> NodeId {
        self.node
    }

    /// The algorithm the `Algorithm` attribute names.
    #[must_use]
    pub const fn algorithm(&self) -> SignatureAlgorithm {
        self.algorithm
    }

    /// The `<ds:HMACOutputLength>` truncation length **in bits**, if present.
    ///
    /// [XML Signature §4.4.2](https://www.w3.org/TR/xmldsig-core1/#sec-SignatureMethod)
    /// requires a signature to be deemed invalid when this is below the larger
    /// of half the hash length and 80 bits; the verifier enforces that, see
    /// [`SignatureAlgorithm::minimum_hmac_output_length`].
    #[must_use]
    pub const fn hmac_output_length(&self) -> Option<usize> {
        self.hmac_output_length
    }

    /// The `<pss:RSAPSSParams>` child, if the method declared one.
    ///
    /// Only RSASSA-PSS uses it; [`effective_pss_parameters`](SignatureMethod::effective_pss_parameters)
    /// is what the verifier actually applies, since every parameter has a
    /// default.
    #[must_use]
    pub const fn pss_parameters(&self) -> Option<&PssParameters> {
        self.pss_parameters.as_ref()
    }

    /// The RSASSA-PSS parameters to apply: the declared ones, or the defaults of
    /// [RFC 9231 §2.3.9](https://www.rfc-editor.org/rfc/rfc9231#section-2.3.9).
    #[must_use]
    pub fn effective_pss_parameters(&self) -> PssParameters {
        self.pss_parameters.unwrap_or_default()
    }
}

/// A parsed `<ds:Reference>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reference {
    pub(crate) node: NodeId,
    pub(crate) id: Option<String>,
    pub(crate) uri: Option<String>,
    pub(crate) reference_type: Option<String>,
    pub(crate) transforms: Vec<Transform>,
    pub(crate) digest_algorithm: DigestAlgorithm,
    pub(crate) digest_value: Vec<u8>,
    pub(crate) digest_value_node: NodeId,
}

impl Reference {
    /// The `<ds:Reference>` element.
    #[must_use]
    pub const fn node(&self) -> NodeId {
        self.node
    }

    /// The `Id` attribute, if any.
    #[must_use]
    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    /// The `URI` attribute. [`None`] means the attribute is *absent*, which the
    /// Recommendation leaves to the application — as opposed to `Some("")`,
    /// which means the whole containing document.
    #[must_use]
    pub fn uri(&self) -> Option<&str> {
        self.uri.as_deref()
    }

    /// The `Type` attribute, an informative hint at what is referenced
    /// ([`OBJECT_TYPE`](crate::OBJECT_TYPE), [`MANIFEST_TYPE`](crate::MANIFEST_TYPE), …).
    #[must_use]
    pub fn reference_type(&self) -> Option<&str> {
        self.reference_type.as_deref()
    }

    /// The `<ds:Transform>` children, in the order they must be applied.
    pub fn transforms(&self) -> impl ExactSizeIterator<Item = &Transform> {
        self.transforms.iter()
    }

    /// The digest the `<ds:DigestMethod>` names.
    #[must_use]
    pub const fn digest_algorithm(&self) -> DigestAlgorithm {
        self.digest_algorithm
    }

    /// The `<ds:DigestValue>`, base64 decoded.
    #[must_use]
    pub fn digest_value(&self) -> &[u8] {
        &self.digest_value
    }

    /// The `<ds:DigestValue>` element, which [`crate::sign`] overwrites.
    #[must_use]
    pub const fn digest_value_node(&self) -> NodeId {
        self.digest_value_node
    }
}

/// A parsed `<ds:Transform>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transform {
    pub(crate) node: NodeId,
    pub(crate) algorithm: TransformAlgorithm,
    pub(crate) parameters: TransformParameters,
}

impl Transform {
    /// The `<ds:Transform>` element.
    #[must_use]
    pub const fn node(&self) -> NodeId {
        self.node
    }

    /// The algorithm the `Algorithm` attribute names.
    #[must_use]
    pub const fn algorithm(&self) -> TransformAlgorithm {
        self.algorithm
    }

    /// The transform's own parameters, parsed according to its algorithm.
    #[must_use]
    pub const fn parameters(&self) -> &TransformParameters {
        &self.parameters
    }
}

/// The content of a `<ds:Transform>`, interpreted per algorithm.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TransformParameters {
    /// No parameters — the enveloped-signature and base64 transforms, and the
    /// inclusive canonicalization algorithms.
    None,
    /// `<ec:InclusiveNamespaces PrefixList="…"/>` for exclusive canonicalization.
    InclusiveNamespaces(InclusiveNamespaces),
    /// `<ds:XPath>…</ds:XPath>` for the XPath filtering transform.
    XPath(XPathParameter),
    /// One or more `<XPath Filter="…">…</XPath>` for
    /// [XPath Filter 2.0](https://www.w3.org/TR/xmldsig-filter2/).
    Filter2(Vec<Filter2Step>),
}

/// The `<ds:XPath>` parameter of the XPath filtering transform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XPathParameter {
    pub(crate) node: NodeId,
    pub(crate) expression: String,
}

impl XPathParameter {
    /// The `<ds:XPath>` element, which supplies both the namespace bindings the
    /// expression is resolved against and the value of the `here()` function.
    #[must_use]
    pub const fn node(&self) -> NodeId {
        self.node
    }

    /// The expression, verbatim.
    #[must_use]
    pub fn expression(&self) -> &str {
        &self.expression
    }
}

/// One `<XPath Filter="…">` of an XPath Filter 2.0 transform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Filter2Step {
    pub(crate) node: NodeId,
    pub(crate) filter: FilterOperation,
    pub(crate) expression: String,
}

impl Filter2Step {
    /// The `<XPath>` element.
    #[must_use]
    pub const fn node(&self) -> NodeId {
        self.node
    }

    /// The set operation the `Filter` attribute names.
    #[must_use]
    pub const fn filter(&self) -> FilterOperation {
        self.filter
    }

    /// The expression, verbatim.
    #[must_use]
    pub fn expression(&self) -> &str {
        &self.expression
    }
}

/// The `Filter` attribute of an XPath Filter 2.0 `<XPath>` element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FilterOperation {
    /// `Filter="intersect"` — `F' = F ∩ S'`.
    Intersect,
    /// `Filter="subtract"` — `F' = F − S'`.
    Subtract,
    /// `Filter="union"` — `F' = F ∪ S'`.
    Union,
}

/// A parsed `<ds:Manifest>` — [XML Signature §5.1](https://www.w3.org/TR/xmldsig-core1/#sec-Manifest).
///
/// A manifest is a list of references that the *core* validation deliberately
/// does not check: the signature covers the manifest's digest, and what the
/// application does with the references inside it is the application's
/// business. [`Verifier::validate_manifests`](crate::Verifier::validate_manifests)
/// turns the check on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    pub(crate) node: NodeId,
    pub(crate) id: Option<String>,
    pub(crate) references: Vec<Reference>,
}

impl Manifest {
    /// The `<ds:Manifest>` element.
    #[must_use]
    pub const fn node(&self) -> NodeId {
        self.node
    }

    /// The `Id` attribute, if any.
    #[must_use]
    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    /// The references the manifest lists.
    pub fn references(&self) -> impl ExactSizeIterator<Item = &Reference> {
        self.references.iter()
    }
}

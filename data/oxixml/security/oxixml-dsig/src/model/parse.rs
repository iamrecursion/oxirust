//! Turning a `<ds:Signature>` subtree into the [model](super).
//!
//! The parser is strict about the content model of
//! [XML Signature §4](https://www.w3.org/TR/xmldsig-core1/#sec-CoreSyntax) and
//! tolerant about everything else. Required children must be present, in the
//! right namespace, and in the right order; unknown children are ignored,
//! because the schema admits extension elements in several places and a
//! verifier that chokes on them is useless in practice.
//!
//! One rule is worth calling out. Whitespace *inside* `<ds:DigestValue>` and
//! `<ds:SignatureValue>` is skipped rather than rejected: those are
//! `xsd:base64Binary`, whose lexical space explicitly allows the XML whitespace
//! characters between any two base64 characters, and real signers wrap long
//! values across lines.

use oxixml_c14n::{CanonicalizationAlgorithm, InclusiveNamespaces};
use oxixml_dom::{Document, NodeId, NodeKind};

use crate::algorithm::{
    DSIG_NAMESPACE, DSIG11_NAMESPACE, DigestAlgorithm, EXC_C14N_NAMESPACE, FILTER2_NAMESPACE,
    MGF1_URI, SignatureAlgorithm, TransformAlgorithm, XMLDSIG_MORE_NAMESPACE,
};
use crate::error::{DsigError, DsigResult, decode_base64};
use crate::key::{IssuerSerial, KeyInfo, KeyInfoItem, X509Data};
use crate::model::{
    CanonicalizationMethod, Filter2Step, FilterOperation, Manifest, PssParameters, Reference,
    Signature, SignatureMethod, SignedInfo, Transform, TransformParameters, XPathParameter,
};

// --- small DOM helpers ------------------------------------------------------------------------

/// The element children of `node`, in document order.
pub(crate) fn child_elements(document: &Document, node: NodeId) -> impl Iterator<Item = NodeId> {
    document
        .children(node)
        .filter(move |child| document.kind(*child) == Some(NodeKind::Element))
}

/// Whether `node` is an element with the given expanded name.
pub(crate) fn is_element(
    document: &Document,
    node: NodeId,
    namespace: &str,
    local_name: &str,
) -> bool {
    document.element(node).is_some_and(|element| {
        element.namespace_uri() == Some(namespace) && element.local_name() == local_name
    })
}

/// The first element child with the given expanded name.
pub(crate) fn first_child(
    document: &Document,
    node: NodeId,
    namespace: &str,
    local_name: &str,
) -> Option<NodeId> {
    child_elements(document, node).find(|child| is_element(document, *child, namespace, local_name))
}

/// Every element child with the given expanded name.
fn children_named<'a>(
    document: &'a Document,
    node: NodeId,
    namespace: &'a str,
    local_name: &'a str,
) -> impl Iterator<Item = NodeId> + 'a {
    child_elements(document, node)
        .filter(move |child| is_element(document, *child, namespace, local_name))
}

/// The expanded name of an element, rendered for an error message.
fn describe(document: &Document, node: NodeId) -> String {
    document.element(node).map_or_else(
        || "a non-element node".to_owned(),
        |element| match element.namespace_uri() {
            Some(namespace) => format!("{{{namespace}}}{}", element.local_name()),
            None => element.local_name().to_owned(),
        },
    )
}

/// The required `Algorithm` attribute of `node`.
fn algorithm_attribute(document: &Document, node: NodeId) -> DsigResult<&str> {
    document.get_attribute(node, "Algorithm").ok_or_else(|| {
        DsigError::Structure(format!(
            "<{}> has no Algorithm attribute",
            describe(document, node)
        ))
    })
}

/// The string value of an element: every descendant text node, concatenated.
fn text_of(document: &Document, node: NodeId) -> String {
    document.text_content(node).unwrap_or_default()
}

// --- the model ---------------------------------------------------------------------------------

impl Signature {
    /// Parses a `<ds:Signature>` element.
    ///
    /// # Errors
    ///
    /// [`DsigError::Structure`] when `node` is not a `<ds:Signature>` or the
    /// content model is violated; [`DsigError::UnknownAlgorithm`] for an
    /// unrecognised `Algorithm`; [`DsigError::Base64`] for a malformed
    /// `<ds:SignatureValue>` or `<ds:DigestValue>`.
    pub fn parse(document: &Document, node: NodeId) -> DsigResult<Self> {
        if !is_element(document, node, DSIG_NAMESPACE, "Signature") {
            return Err(DsigError::Structure(format!(
                "expected {{{DSIG_NAMESPACE}}}Signature, found {}",
                describe(document, node)
            )));
        }
        let signed_info_node = first_child(document, node, DSIG_NAMESPACE, "SignedInfo")
            .ok_or_else(|| DsigError::Structure("<Signature> has no <SignedInfo>".to_owned()))?;
        let signed_info = SignedInfo::parse(document, signed_info_node)?;

        let signature_value_node = first_child(document, node, DSIG_NAMESPACE, "SignatureValue")
            .ok_or_else(|| {
                DsigError::Structure("<Signature> has no <SignatureValue>".to_owned())
            })?;
        let signature_value =
            decode_base64("<SignatureValue>", &text_of(document, signature_value_node))?;

        let key_info = first_child(document, node, DSIG_NAMESPACE, "KeyInfo")
            .map(|key_info_node| parse_key_info(document, key_info_node))
            .transpose()?;

        let objects = children_named(document, node, DSIG_NAMESPACE, "Object").collect();

        Ok(Self {
            node,
            id: document.get_attribute(node, "Id").map(str::to_owned),
            signed_info,
            signature_value,
            signature_value_node,
            key_info,
            objects,
        })
    }

    /// Finds and parses the first `<ds:Signature>` anywhere in `document`, in
    /// document order.
    ///
    /// # Errors
    ///
    /// [`DsigError::Structure`] if there is no `<ds:Signature>` at all, or if
    /// the first one is malformed.
    pub fn find(document: &Document) -> DsigResult<Self> {
        let node = document
            .get_elements_by_tag_name_ns(document.root(), Some(DSIG_NAMESPACE), "Signature")
            .next()
            .ok_or_else(|| {
                DsigError::Structure("the document contains no <ds:Signature>".to_owned())
            })?;
        Self::parse(document, node)
    }
}

impl SignedInfo {
    /// Parses a `<ds:SignedInfo>` element.
    ///
    /// # Errors
    ///
    /// See [`Signature::parse`].
    pub fn parse(document: &Document, node: NodeId) -> DsigResult<Self> {
        let canonicalization_node =
            first_child(document, node, DSIG_NAMESPACE, "CanonicalizationMethod").ok_or_else(
                || DsigError::Structure("<SignedInfo> has no <CanonicalizationMethod>".to_owned()),
            )?;
        let canonicalization_method =
            parse_canonicalization_method(document, canonicalization_node)?;

        let method_node = first_child(document, node, DSIG_NAMESPACE, "SignatureMethod")
            .ok_or_else(|| {
                DsigError::Structure("<SignedInfo> has no <SignatureMethod>".to_owned())
            })?;
        let signature_method = parse_signature_method(document, method_node)?;

        let references = children_named(document, node, DSIG_NAMESPACE, "Reference")
            .map(|reference| Reference::parse(document, reference))
            .collect::<DsigResult<Vec<_>>>()?;
        if references.is_empty() {
            return Err(DsigError::Structure(
                "<SignedInfo> has no <Reference>".to_owned(),
            ));
        }

        Ok(Self {
            node,
            id: document.get_attribute(node, "Id").map(str::to_owned),
            canonicalization_method,
            signature_method,
            references,
        })
    }
}

/// Parses a `<ds:CanonicalizationMethod>` or the equivalent transform element.
fn parse_canonicalization_method(
    document: &Document,
    node: NodeId,
) -> DsigResult<CanonicalizationMethod> {
    let uri = algorithm_attribute(document, node)?;
    let algorithm = CanonicalizationAlgorithm::from_uri(uri)
        .map_err(|_| DsigError::UnknownAlgorithm(uri.to_owned()))?;
    Ok(CanonicalizationMethod {
        node,
        algorithm,
        inclusive_prefixes: parse_inclusive_namespaces(document, node),
    })
}

/// Reads an `<ec:InclusiveNamespaces PrefixList="…"/>` child, if there is one.
pub(crate) fn parse_inclusive_namespaces(document: &Document, node: NodeId) -> InclusiveNamespaces {
    first_child(document, node, EXC_C14N_NAMESPACE, "InclusiveNamespaces")
        .and_then(|child| document.get_attribute(child, "PrefixList"))
        .map_or_else(InclusiveNamespaces::none, InclusiveNamespaces::parse)
}

/// Parses a `<ds:SignatureMethod>`.
fn parse_signature_method(document: &Document, node: NodeId) -> DsigResult<SignatureMethod> {
    let uri = algorithm_attribute(document, node)?;
    let algorithm = SignatureAlgorithm::from_uri(uri)?;
    let hmac_output_length = first_child(document, node, DSIG_NAMESPACE, "HMACOutputLength")
        .map(|child| {
            let text = text_of(document, child);
            text.trim().parse::<usize>().map_err(|_| {
                DsigError::Structure(format!("<HMACOutputLength> is not an integer: {text:?}"))
            })
        })
        .transpose()?;
    Ok(SignatureMethod {
        node,
        algorithm,
        hmac_output_length,
        pss_parameters: parse_pss_parameters(document, node)?,
    })
}

/// Parses the `<pss:RSAPSSParams>` child of a `<ds:SignatureMethod>` —
/// [RFC 9231 §2.3.9](https://www.rfc-editor.org/rfc/rfc9231#section-2.3.9).
///
/// Every field is optional and every default is derived from the one before it,
/// so an empty `<RSAPSSParams/>` means exactly the same thing as no element at
/// all.
fn parse_pss_parameters(document: &Document, node: NodeId) -> DsigResult<Option<PssParameters>> {
    let Some(element) = first_child(document, node, XMLDSIG_MORE_NAMESPACE, "RSAPSSParams") else {
        return Ok(None);
    };
    let digest = first_child(document, element, DSIG_NAMESPACE, "DigestMethod")
        .map(|child| DigestAlgorithm::from_uri(algorithm_attribute(document, child)?))
        .transpose()?
        .unwrap_or(DigestAlgorithm::Sha256);
    let mut parameters = PssParameters::new(digest);

    if let Some(mask) = first_child(
        document,
        element,
        XMLDSIG_MORE_NAMESPACE,
        "MaskGenerationFunction",
    ) {
        // The Algorithm attribute defaults to MGF1, the only function RFC 8017
        // defines; anything else is a method this crate cannot compute.
        if let Some(uri) = document.get_attribute(mask, "Algorithm")
            && uri != MGF1_URI
        {
            return Err(DsigError::UnsupportedAlgorithm(uri.to_owned()));
        }
        if let Some(child) = first_child(document, mask, DSIG_NAMESPACE, "DigestMethod") {
            parameters = parameters.mask_generation_digest(DigestAlgorithm::from_uri(
                algorithm_attribute(document, child)?,
            )?);
        }
    }

    if let Some(child) = first_child(document, element, XMLDSIG_MORE_NAMESPACE, "SaltLength") {
        let text = text_of(document, child);
        let octets = text.trim().parse::<usize>().map_err(|_| {
            DsigError::Structure(format!(
                "<SaltLength> is not a non-negative integer: {text:?}"
            ))
        })?;
        parameters = parameters.salt_length(octets);
    }

    if let Some(child) = first_child(document, element, XMLDSIG_MORE_NAMESPACE, "TrailerField") {
        let text = text_of(document, child);
        parameters.trailer_field = text.trim().parse::<u32>().map_err(|_| {
            DsigError::Structure(format!(
                "<TrailerField> is not a non-negative integer: {text:?}"
            ))
        })?;
    }

    Ok(Some(parameters))
}

impl Reference {
    /// Parses a `<ds:Reference>` element.
    ///
    /// # Errors
    ///
    /// See [`Signature::parse`].
    pub fn parse(document: &Document, node: NodeId) -> DsigResult<Self> {
        let transforms = first_child(document, node, DSIG_NAMESPACE, "Transforms")
            .map(|list| {
                children_named(document, list, DSIG_NAMESPACE, "Transform")
                    .map(|transform| Transform::parse(document, transform))
                    .collect::<DsigResult<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();

        let digest_method = first_child(document, node, DSIG_NAMESPACE, "DigestMethod")
            .ok_or_else(|| DsigError::Structure("<Reference> has no <DigestMethod>".to_owned()))?;
        let digest_algorithm =
            DigestAlgorithm::from_uri(algorithm_attribute(document, digest_method)?)?;

        let digest_value_node = first_child(document, node, DSIG_NAMESPACE, "DigestValue")
            .ok_or_else(|| DsigError::Structure("<Reference> has no <DigestValue>".to_owned()))?;
        let digest_value = decode_base64("<DigestValue>", &text_of(document, digest_value_node))?;

        Ok(Self {
            node,
            id: document.get_attribute(node, "Id").map(str::to_owned),
            uri: document.get_attribute(node, "URI").map(str::to_owned),
            reference_type: document.get_attribute(node, "Type").map(str::to_owned),
            transforms,
            digest_algorithm,
            digest_value,
            digest_value_node,
        })
    }
}

impl Transform {
    /// Parses a `<ds:Transform>` element together with its parameters.
    ///
    /// # Errors
    ///
    /// [`DsigError::UnknownAlgorithm`] for an unrecognised `Algorithm`, and
    /// [`DsigError::Structure`] for parameters that do not match the algorithm
    /// (an XPath transform with no `<ds:XPath>`, an XPath Filter 2.0 transform
    /// with an unknown `Filter` value).
    pub fn parse(document: &Document, node: NodeId) -> DsigResult<Self> {
        let algorithm = TransformAlgorithm::from_uri(algorithm_attribute(document, node)?)?;
        let parameters = match algorithm {
            TransformAlgorithm::Canonicalization(c14n) if c14n.is_exclusive() => {
                TransformParameters::InclusiveNamespaces(parse_inclusive_namespaces(document, node))
            }
            TransformAlgorithm::XPath => {
                let child =
                    first_child(document, node, DSIG_NAMESPACE, "XPath").ok_or_else(|| {
                        DsigError::Structure("the XPath transform has no <ds:XPath>".to_owned())
                    })?;
                TransformParameters::XPath(XPathParameter {
                    node: child,
                    expression: text_of(document, child),
                })
            }
            TransformAlgorithm::XPathFilter2 => {
                let steps = children_named(document, node, FILTER2_NAMESPACE, "XPath")
                    .map(|child| parse_filter2_step(document, child))
                    .collect::<DsigResult<Vec<_>>>()?;
                if steps.is_empty() {
                    return Err(DsigError::Structure(
                        "the XPath Filter 2.0 transform has no <XPath> children".to_owned(),
                    ));
                }
                TransformParameters::Filter2(steps)
            }
            _ => TransformParameters::None,
        };
        Ok(Self {
            node,
            algorithm,
            parameters,
        })
    }
}

fn parse_filter2_step(document: &Document, node: NodeId) -> DsigResult<Filter2Step> {
    let filter = match document.get_attribute(node, "Filter") {
        Some("intersect") => FilterOperation::Intersect,
        Some("subtract") => FilterOperation::Subtract,
        Some("union") => FilterOperation::Union,
        Some(other) => {
            return Err(DsigError::Structure(format!(
                "unknown XPath Filter 2.0 Filter value {other:?}"
            )));
        }
        None => {
            return Err(DsigError::Structure(
                "an XPath Filter 2.0 <XPath> has no Filter attribute".to_owned(),
            ));
        }
    };
    Ok(Filter2Step {
        node,
        filter,
        expression: text_of(document, node),
    })
}

impl Manifest {
    /// Parses a `<ds:Manifest>` element.
    ///
    /// # Errors
    ///
    /// [`DsigError::Structure`] when `node` is not a `<ds:Manifest>` or one of
    /// its references is malformed.
    pub fn parse(document: &Document, node: NodeId) -> DsigResult<Self> {
        if !is_element(document, node, DSIG_NAMESPACE, "Manifest") {
            return Err(DsigError::Structure(format!(
                "expected {{{DSIG_NAMESPACE}}}Manifest, found {}",
                describe(document, node)
            )));
        }
        let references = children_named(document, node, DSIG_NAMESPACE, "Reference")
            .map(|reference| Reference::parse(document, reference))
            .collect::<DsigResult<Vec<_>>>()?;
        Ok(Self {
            node,
            id: document.get_attribute(node, "Id").map(str::to_owned),
            references,
        })
    }
}

// --- KeyInfo ------------------------------------------------------------------------------------

/// Parses the `<ds:KeyInfo>` element at `node`, checking its name first.
///
/// # Errors
///
/// [`DsigError::Structure`] when `node` is not a `<ds:KeyInfo>`.
pub(crate) fn parse_key_info_at(document: &Document, node: NodeId) -> DsigResult<KeyInfo> {
    if !is_element(document, node, DSIG_NAMESPACE, "KeyInfo") {
        return Err(DsigError::Structure(format!(
            "expected {{{DSIG_NAMESPACE}}}KeyInfo, found {}",
            describe(document, node)
        )));
    }
    parse_key_info(document, node)
}

/// Parses a `<ds:KeyInfo>` element.
pub(crate) fn parse_key_info(document: &Document, node: NodeId) -> DsigResult<KeyInfo> {
    let mut items = Vec::new();
    for child in child_elements(document, node) {
        items.push(parse_key_info_item(document, child)?);
    }
    Ok(KeyInfo {
        id: document.get_attribute(node, "Id").map(str::to_owned),
        items,
    })
}

fn parse_key_info_item(document: &Document, node: NodeId) -> DsigResult<KeyInfoItem> {
    let Some(element) = document.element(node) else {
        return Err(DsigError::Structure(
            "a <KeyInfo> child is not an element".to_owned(),
        ));
    };
    let namespace = element.namespace_uri().map(str::to_owned);
    let local_name = element.local_name().to_owned();
    match (namespace.as_deref(), local_name.as_str()) {
        (Some(DSIG_NAMESPACE), "KeyName") => Ok(KeyInfoItem::KeyName(text_of(document, node))),
        (Some(DSIG_NAMESPACE), "KeyValue") => parse_key_value(document, node),
        (Some(DSIG_NAMESPACE), "RetrievalMethod") => Ok(KeyInfoItem::RetrievalMethod {
            uri: document
                .get_attribute(node, "URI")
                .unwrap_or_default()
                .to_owned(),
            key_type: document.get_attribute(node, "Type").map(str::to_owned),
        }),
        (Some(DSIG_NAMESPACE), "X509Data") => {
            parse_x509_data(document, node).map(KeyInfoItem::X509Data)
        }
        (Some(DSIG_NAMESPACE), "PGPData") => Ok(KeyInfoItem::PgpData),
        (Some(DSIG_NAMESPACE), "SPKIData") => Ok(KeyInfoItem::SpkiData),
        (Some(DSIG_NAMESPACE), "MgmtData") => Ok(KeyInfoItem::MgmtData(text_of(document, node))),
        (Some(DSIG11_NAMESPACE), "DEREncodedKeyValue") => Ok(KeyInfoItem::DerEncodedKeyValue(
            decode_base64("<DEREncodedKeyValue>", &text_of(document, node))?,
        )),
        (Some(DSIG11_NAMESPACE), "KeyInfoReference") => Ok(KeyInfoItem::KeyInfoReference {
            uri: document
                .get_attribute(node, "URI")
                .unwrap_or_default()
                .to_owned(),
        }),
        _ => Ok(KeyInfoItem::Other {
            namespace,
            local_name,
        }),
    }
}

fn parse_key_value(document: &Document, node: NodeId) -> DsigResult<KeyInfoItem> {
    if let Some(rsa) = first_child(document, node, DSIG_NAMESPACE, "RSAKeyValue") {
        let modulus = first_child(document, rsa, DSIG_NAMESPACE, "Modulus")
            .ok_or_else(|| DsigError::Structure("<RSAKeyValue> has no <Modulus>".to_owned()))?;
        let exponent = first_child(document, rsa, DSIG_NAMESPACE, "Exponent")
            .ok_or_else(|| DsigError::Structure("<RSAKeyValue> has no <Exponent>".to_owned()))?;
        return Ok(KeyInfoItem::RsaKeyValue {
            modulus: decode_base64("<Modulus>", &text_of(document, modulus))?,
            exponent: decode_base64("<Exponent>", &text_of(document, exponent))?,
        });
    }
    if let Some(dsa) = first_child(document, node, DSIG_NAMESPACE, "DSAKeyValue") {
        let value = |local_name| {
            first_child(document, dsa, DSIG_NAMESPACE, local_name)
                .map(|child| decode_base64("<DSAKeyValue>", &text_of(document, child)))
                .transpose()
        };
        let y = value("Y")?
            .ok_or_else(|| DsigError::Structure("<DSAKeyValue> has no <Y>".to_owned()))?;
        return Ok(KeyInfoItem::DsaKeyValue {
            p: value("P")?,
            q: value("Q")?,
            g: value("G")?,
            y,
        });
    }
    if let Some(ec) = first_child(document, node, DSIG11_NAMESPACE, "ECKeyValue") {
        let public_key = first_child(document, ec, DSIG11_NAMESPACE, "PublicKey")
            .ok_or_else(|| DsigError::Structure("<ECKeyValue> has no <PublicKey>".to_owned()))?;
        return Ok(KeyInfoItem::EcKeyValue {
            named_curve: first_child(document, ec, DSIG11_NAMESPACE, "NamedCurve")
                .and_then(|child| document.get_attribute(child, "URI"))
                .map(str::to_owned),
            public_key: decode_base64("<PublicKey>", &text_of(document, public_key))?,
        });
    }
    Ok(KeyInfoItem::Other {
        namespace: Some(DSIG_NAMESPACE.to_owned()),
        local_name: "KeyValue".to_owned(),
    })
}

fn parse_x509_data(document: &Document, node: NodeId) -> DsigResult<X509Data> {
    let mut data = X509Data::default();
    for child in child_elements(document, node) {
        let Some(element) = document.element(child) else {
            continue;
        };
        match (element.namespace_uri(), element.local_name()) {
            (Some(DSIG_NAMESPACE), "X509Certificate") => data.certificates.push(decode_base64(
                "<X509Certificate>",
                &text_of(document, child),
            )?),
            (Some(DSIG_NAMESPACE), "X509SubjectName") => {
                data.subject_names.push(text_of(document, child));
            }
            (Some(DSIG_NAMESPACE), "X509SKI") => data
                .subject_key_identifiers
                .push(decode_base64("<X509SKI>", &text_of(document, child))?),
            (Some(DSIG_NAMESPACE), "X509CRL") => data
                .crls
                .push(decode_base64("<X509CRL>", &text_of(document, child))?),
            (Some(DSIG_NAMESPACE), "X509IssuerSerial") => {
                let issuer_name = first_child(document, child, DSIG_NAMESPACE, "X509IssuerName")
                    .map(|issuer| text_of(document, issuer))
                    .unwrap_or_default();
                let serial_number =
                    first_child(document, child, DSIG_NAMESPACE, "X509SerialNumber")
                        .map(|serial| text_of(document, serial))
                        .unwrap_or_default();
                data.issuer_serials.push(IssuerSerial {
                    issuer_name,
                    serial_number,
                });
            }
            (Some(DSIG11_NAMESPACE), "X509Digest") => {
                let algorithm = DigestAlgorithm::from_uri(algorithm_attribute(document, child)?)?;
                data.digests.push((
                    algorithm,
                    decode_base64("<X509Digest>", &text_of(document, child))?,
                ));
            }
            _ => {}
        }
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_dom::build;

    const ENVELOPING: &str = r##"<dsig:Signature xmlns:dsig="http://www.w3.org/2000/09/xmldsig#" Id="sig">
        <dsig:SignedInfo>
          <dsig:CanonicalizationMethod Algorithm="http://www.w3.org/TR/2001/REC-xml-c14n-20010315"/>
          <dsig:SignatureMethod Algorithm="http://www.w3.org/2001/04/xmldsig-more#hmac-sha256"/>
          <dsig:Reference URI="#obj" Type="http://www.w3.org/2000/09/xmldsig#Object">
            <dsig:DigestMethod Algorithm="http://www.w3.org/2000/09/xmldsig#sha1"/>
            <dsig:DigestValue>myrT5qEfA7Wemy2WONCZG66c5QE=</dsig:DigestValue>
          </dsig:Reference>
        </dsig:SignedInfo>
        <dsig:SignatureValue>
          s8ntBS/35iYGZYg16NrU4vwxdUufDXw/YVN5E9AIUK0=
        </dsig:SignatureValue>
        <dsig:Object Id="obj"><Web>up up and away</Web></dsig:Object>
      </dsig:Signature>"##;

    #[test]
    fn an_enveloping_signature_parses() -> DsigResult<()> {
        let document = build::parse_str(ENVELOPING).expect("well-formed test input");
        let signature = Signature::find(&document)?;
        assert_eq!(signature.id(), Some("sig"));
        assert_eq!(signature.objects().len(), 1);
        assert_eq!(signature.signature_value().len(), 32);
        assert_eq!(
            signature.signed_info().signature_method().algorithm(),
            SignatureAlgorithm::HmacSha256
        );
        assert_eq!(
            signature
                .signed_info()
                .canonicalization_method()
                .algorithm(),
            CanonicalizationAlgorithm::Canonical1_0
        );
        let reference = signature
            .signed_info()
            .references()
            .next()
            .expect("a Reference");
        assert_eq!(reference.uri(), Some("#obj"));
        assert_eq!(reference.digest_algorithm(), DigestAlgorithm::Sha1);
        assert_eq!(reference.digest_value().len(), 20);
        assert_eq!(reference.transforms().count(), 0);
        Ok(())
    }

    #[test]
    fn whitespace_inside_a_base64_value_is_skipped() -> DsigResult<()> {
        let document = build::parse_str(ENVELOPING).expect("well-formed test input");
        let signature = Signature::find(&document)?;
        // The <SignatureValue> above is indented across three lines.
        assert_eq!(signature.signature_value().len(), 32);
        Ok(())
    }

    #[test]
    fn a_missing_signed_info_is_a_structure_error() {
        let document = build::parse_str(
            r##"<Signature xmlns="http://www.w3.org/2000/09/xmldsig#"><SignatureValue/></Signature>"##,
        )
        .expect("well-formed test input");
        assert!(matches!(
            Signature::find(&document),
            Err(DsigError::Structure(_))
        ));
    }

    #[test]
    fn a_signature_in_the_wrong_namespace_is_not_a_signature() {
        let document =
            build::parse_str(r##"<Signature xmlns="urn:not-dsig"/>"##).expect("well-formed input");
        let root = document.document_element().expect("a document element");
        assert!(matches!(
            Signature::parse(&document, root),
            Err(DsigError::Structure(_))
        ));
    }

    #[test]
    fn transform_parameters_are_parsed_per_algorithm() -> DsigResult<()> {
        let document = build::parse_str(
            r##"<Reference xmlns="http://www.w3.org/2000/09/xmldsig#" URI="">
                 <Transforms>
                   <Transform Algorithm="http://www.w3.org/2000/09/xmldsig#enveloped-signature"/>
                   <Transform Algorithm="http://www.w3.org/TR/1999/REC-xpath-19991116">
                     <XPath>not(ancestor-or-self::ds:Signature)</XPath>
                   </Transform>
                   <Transform Algorithm="http://www.w3.org/2002/06/xmldsig-filter2">
                     <XPath xmlns="http://www.w3.org/2002/06/xmldsig-filter2"
                            Filter="subtract">here()/ancestor::Signature[1]</XPath>
                   </Transform>
                   <Transform Algorithm="http://www.w3.org/2001/10/xml-exc-c14n#">
                     <InclusiveNamespaces xmlns="http://www.w3.org/2001/10/xml-exc-c14n#"
                                          PrefixList="a b"/>
                   </Transform>
                 </Transforms>
                 <DigestMethod Algorithm="http://www.w3.org/2001/04/xmlenc#sha256"/>
                 <DigestValue>47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU=</DigestValue>
               </Reference>"##,
        )
        .expect("well-formed test input");
        let root = document.document_element().expect("a document element");
        let reference = Reference::parse(&document, root)?;
        let transforms: Vec<_> = reference.transforms().collect();
        assert_eq!(transforms.len(), 4);
        assert_eq!(
            transforms.first().map(|t| t.parameters()),
            Some(&TransformParameters::None)
        );
        assert!(matches!(
            transforms.get(1).map(|t| t.parameters()),
            Some(TransformParameters::XPath(_))
        ));
        match transforms.get(2).map(|t| t.parameters()) {
            Some(TransformParameters::Filter2(steps)) => {
                assert_eq!(steps.len(), 1);
                assert_eq!(
                    steps.first().map(|step| step.filter()),
                    Some(FilterOperation::Subtract)
                );
            }
            other => panic!("expected Filter2 parameters, got {other:?}"),
        }
        match transforms.get(3).map(|t| t.parameters()) {
            Some(TransformParameters::InclusiveNamespaces(prefixes)) => {
                assert!(prefixes.contains("a") && prefixes.contains("b"));
            }
            other => panic!("expected InclusiveNamespaces, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn key_info_children_are_all_recognised() -> DsigResult<()> {
        let document = build::parse_str(
            r##"<KeyInfo xmlns="http://www.w3.org/2000/09/xmldsig#"
                        xmlns:d11="http://www.w3.org/2009/xmldsig11#" Id="ki">
                 <KeyName>alice</KeyName>
                 <KeyValue><RSAKeyValue><Modulus>AQAB</Modulus><Exponent>AQAB</Exponent></RSAKeyValue></KeyValue>
                 <d11:DEREncodedKeyValue>AQAB</d11:DEREncodedKeyValue>
                 <d11:KeyInfoReference URI="#other"/>
                 <X509Data>
                   <X509SubjectName>CN=Alice</X509SubjectName>
                   <X509IssuerSerial><X509IssuerName>CN=CA</X509IssuerName><X509SerialNumber>7</X509SerialNumber></X509IssuerSerial>
                   <X509SKI>AQAB</X509SKI>
                   <d11:X509Digest Algorithm="http://www.w3.org/2001/04/xmlenc#sha256">AQAB</d11:X509Digest>
                 </X509Data>
                 <PGPData/>
                 <urn:Custom xmlns:urn="urn:example"/>
               </KeyInfo>"##,
        )
        .expect("well-formed test input");
        let root = document.document_element().expect("a document element");
        let key_info = parse_key_info(&document, root)?;
        assert_eq!(key_info.id(), Some("ki"));
        assert_eq!(key_info.items().count(), 7);
        assert_eq!(key_info.key_names().collect::<Vec<_>>(), ["alice"]);
        assert_eq!(key_info.key_info_reference(), Some("#other"));
        let x509 = key_info
            .items()
            .find_map(|item| match item {
                KeyInfoItem::X509Data(data) => Some(data),
                _ => None,
            })
            .expect("an X509Data");
        assert_eq!(x509.subject_names().collect::<Vec<_>>(), ["CN=Alice"]);
        assert_eq!(x509.issuer_serials().count(), 1);
        assert_eq!(x509.subject_key_identifiers().count(), 1);
        assert_eq!(x509.digests().count(), 1);
        Ok(())
    }
}

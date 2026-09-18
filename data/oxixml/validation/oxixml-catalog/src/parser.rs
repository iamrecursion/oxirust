//! Parses an OASIS XML Catalog document into a flattened [`CatalogFile`].

use oxixml_iri::{Iri, IriRef};
use oxixml_xml::XmlVersion;
use oxixml_xml::errors::Error as XmlError;
use oxixml_xml::events::attributes::AttrError;
use oxixml_xml::events::{BytesStart, Event};
use oxixml_xml::name::{Namespace, QName, ResolveResult};
use oxixml_xml::reader::NsReader;

use crate::error::CatalogError;
use crate::model::{CatalogFile, Entry, Prefer};
use crate::normalize::{normalize_public_id, normalize_uri_ref};

/// The catalog namespace name, `urn:oasis:names:tc:entity:xmlns:xml:catalog`
/// (OASIS XML Catalogs 1.1 §3.1).
const CATALOG_NS: Namespace<'static> = Namespace(b"urn:oasis:names:tc:entity:xmlns:xml:catalog");

/// The greatest depth of nested `group` elements this parser will follow before reporting
/// [`CatalogError::GroupNestingTooDeep`] instead of exhausting the native call stack.
///
/// The internal `parse_children`/`dispatch` functions are mutually recursive, calling one
/// another once per `group` nesting level, and each level costs several stack frames of its
/// own — an [`NsReader`] borrow, an absolutized [`Iri`], a resolved `prefer` scope. The reader
/// underlying every catalog document already bounds *raw* element nesting on its own, but at a
/// considerably larger depth chosen for XML documents generally: a `group` chain nested deep
/// enough to still pass the reader's own cap has been confirmed (in a separate process; an
/// in-process `catch_unwind` cannot observe stack exhaustion) to still exhaust the stack here.
/// This mirrors the smaller ceiling this workspace uses for its other recursive-descent parsers
/// (see, for instance, `oxixml_relaxng::compact::parse`'s equivalent guard), rather than the
/// larger ceiling used for plain data-structure nesting, for the same reason: this recursion
/// costs measurably more native stack per level than a plain tree walk does.
pub const MAX_GROUP_DEPTH: usize = 64;

/// Parses `xml` as a catalog document, resolving all relative references
/// against `base`, and flattening every `group`-scoped `xml:base`/`prefer`
/// context into each contained entry.
///
/// `default_prefer` is the `prefer` strategy assumed when neither the root
/// `catalog` element nor any enclosing `group` declares a `prefer` attribute.
pub(crate) fn parse_document(
    xml: &str,
    base: Iri<String>,
    default_prefer: Prefer,
) -> Result<CatalogFile, CatalogError> {
    let origin_uri = base.as_str().to_owned();
    let mut reader = NsReader::from_str(xml);
    reader.config_mut().trim_text(true);

    let (root, root_base, root_prefer, has_children) = loop {
        match reader.read_resolved_event().map_err(CatalogError::Xml)? {
            (ResolveResult::Bound(ns), Event::Start(start)) if ns == CATALOG_NS => {
                let root_base = compute_base(&base, &start)?;
                let root_prefer = compute_prefer(default_prefer, &start)?;
                break (start, root_base, root_prefer, true);
            }
            (ResolveResult::Bound(ns), Event::Empty(start)) if ns == CATALOG_NS => {
                let root_base = compute_base(&base, &start)?;
                let root_prefer = compute_prefer(default_prefer, &start)?;
                break (start, root_base, root_prefer, false);
            }
            (_, Event::Eof) => return Err(CatalogError::MissingRootElement),
            (_, Event::Start(_)) | (_, Event::Empty(_)) => {
                return Err(CatalogError::MissingRootElement);
            }
            _ => {}
        }
    };
    if root.local_name().as_ref() != b"catalog" {
        return Err(CatalogError::MissingRootElement);
    }

    let mut entries = Vec::new();
    if has_children {
        parse_children(
            &mut reader,
            &root_base,
            root_prefer,
            &mut entries,
            "catalog",
            0,
        )?;
    }

    Ok(CatalogFile {
        uri: origin_uri,
        entries,
    })
}

/// Reads events until the `End` matching the element that opened this scope
/// (a `catalog` or `group` start tag), dispatching every child element found
/// along the way.
///
/// `depth` is how many `group` scopes this call is nested inside — 0 for the root `catalog`
/// scope itself — checked against [`MAX_GROUP_DEPTH`] before recursing any further; see its
/// documentation for why.
fn parse_children(
    reader: &mut NsReader<&[u8]>,
    base: &Iri<String>,
    prefer: Prefer,
    out: &mut Vec<Entry>,
    owner: &'static str,
    depth: usize,
) -> Result<(), CatalogError> {
    loop {
        match reader.read_resolved_event().map_err(CatalogError::Xml)? {
            (_, Event::End(_)) => return Ok(()),
            (_, Event::Eof) => {
                return Err(CatalogError::UnexpectedEof {
                    element: owner.to_owned(),
                });
            }
            (ns, Event::Start(start)) => {
                let in_catalog_ns = matches!(ns, ResolveResult::Bound(n) if n == CATALOG_NS);
                dispatch(
                    reader,
                    in_catalog_ns,
                    &start,
                    base,
                    prefer,
                    out,
                    true,
                    depth,
                )?;
            }
            (ns, Event::Empty(start)) => {
                let in_catalog_ns = matches!(ns, ResolveResult::Bound(n) if n == CATALOG_NS);
                dispatch(
                    reader,
                    in_catalog_ns,
                    &start,
                    base,
                    prefer,
                    out,
                    false,
                    depth,
                )?;
            }
            _ => {}
        }
    }
}

/// Handles one child element of a `catalog` or `group` scope: either a nested
/// `group` (recursed into), a recognized entry element (parsed into an
/// [`Entry`]), or anything else (ignored, its subtree skipped).
///
/// `depth` is `parse_children`'s own, passed straight through.
#[allow(clippy::too_many_arguments)]
fn dispatch(
    reader: &mut NsReader<&[u8]>,
    in_catalog_ns: bool,
    start: &BytesStart<'_>,
    base: &Iri<String>,
    prefer: Prefer,
    out: &mut Vec<Entry>,
    has_children: bool,
    depth: usize,
) -> Result<(), CatalogError> {
    if !in_catalog_ns {
        if has_children {
            skip_subtree(reader, start)?;
        }
        return Ok(());
    }

    match start.local_name().as_ref() {
        b"group" => {
            let new_depth = depth + 1;
            if new_depth > MAX_GROUP_DEPTH {
                return Err(CatalogError::GroupNestingTooDeep {
                    limit: MAX_GROUP_DEPTH,
                });
            }
            let new_base = compute_base(base, start)?;
            let new_prefer = compute_prefer(prefer, start)?;
            if has_children {
                parse_children(reader, &new_base, new_prefer, out, "group", new_depth)?;
            }
            return Ok(());
        }
        b"public" => {
            let public_id = req_attr(start, "public", "publicId")?;
            let uri_raw = req_attr(start, "public", "uri")?;
            let own_base = compute_base(base, start)?;
            out.push(Entry::Public {
                public_id: normalize_public_id(&public_id),
                uri: absolutize(&own_base, &uri_raw)?,
                prefer,
            });
        }
        b"system" => {
            let system_id = req_attr(start, "system", "systemId")?;
            let uri_raw = req_attr(start, "system", "uri")?;
            let own_base = compute_base(base, start)?;
            out.push(Entry::System {
                system_id: normalize_uri_ref(&system_id),
                uri: absolutize(&own_base, &uri_raw)?,
            });
        }
        b"uri" => {
            let name_raw = req_attr(start, "uri", "name")?;
            let uri_raw = req_attr(start, "uri", "uri")?;
            let own_base = compute_base(base, start)?;
            out.push(Entry::Uri {
                name: normalize_uri_ref(&name_raw),
                uri: absolutize(&own_base, &uri_raw)?,
            });
        }
        b"rewriteSystem" => {
            let start_str = req_attr(start, "rewriteSystem", "systemIdStartString")?;
            let prefix_raw = req_attr(start, "rewriteSystem", "rewritePrefix")?;
            let own_base = compute_base(base, start)?;
            out.push(Entry::RewriteSystem {
                start: normalize_uri_ref(&start_str),
                prefix: absolutize(&own_base, &prefix_raw)?,
            });
        }
        b"rewriteURI" => {
            let start_str = req_attr(start, "rewriteURI", "uriStartString")?;
            let prefix_raw = req_attr(start, "rewriteURI", "rewritePrefix")?;
            let own_base = compute_base(base, start)?;
            out.push(Entry::RewriteUri {
                start: normalize_uri_ref(&start_str),
                prefix: absolutize(&own_base, &prefix_raw)?,
            });
        }
        b"systemSuffix" => {
            let suffix = req_attr(start, "systemSuffix", "systemIdSuffix")?;
            let uri_raw = req_attr(start, "systemSuffix", "uri")?;
            let own_base = compute_base(base, start)?;
            out.push(Entry::SystemSuffix {
                suffix: normalize_uri_ref(&suffix),
                uri: absolutize(&own_base, &uri_raw)?,
            });
        }
        b"uriSuffix" => {
            let suffix = req_attr(start, "uriSuffix", "uriSuffix")?;
            let uri_raw = req_attr(start, "uriSuffix", "uri")?;
            let own_base = compute_base(base, start)?;
            out.push(Entry::UriSuffix {
                suffix: normalize_uri_ref(&suffix),
                uri: absolutize(&own_base, &uri_raw)?,
            });
        }
        b"delegatePublic" => {
            let start_str = req_attr(start, "delegatePublic", "publicIdStartString")?;
            let catalog_raw = req_attr(start, "delegatePublic", "catalog")?;
            let own_base = compute_base(base, start)?;
            out.push(Entry::DelegatePublic {
                start: normalize_public_id(&start_str),
                catalog: absolutize(&own_base, &catalog_raw)?,
                prefer,
            });
        }
        b"delegateSystem" => {
            let start_str = req_attr(start, "delegateSystem", "systemIdStartString")?;
            let catalog_raw = req_attr(start, "delegateSystem", "catalog")?;
            let own_base = compute_base(base, start)?;
            out.push(Entry::DelegateSystem {
                start: normalize_uri_ref(&start_str),
                catalog: absolutize(&own_base, &catalog_raw)?,
            });
        }
        b"delegateURI" => {
            let start_str = req_attr(start, "delegateURI", "uriStartString")?;
            let catalog_raw = req_attr(start, "delegateURI", "catalog")?;
            let own_base = compute_base(base, start)?;
            out.push(Entry::DelegateUri {
                start: normalize_uri_ref(&start_str),
                catalog: absolutize(&own_base, &catalog_raw)?,
            });
        }
        b"nextCatalog" => {
            let catalog_raw = req_attr(start, "nextCatalog", "catalog")?;
            let own_base = compute_base(base, start)?;
            out.push(Entry::NextCatalog {
                catalog: absolutize(&own_base, &catalog_raw)?,
            });
        }
        _ => {
            if has_children {
                skip_subtree(reader, start)?;
            }
            return Ok(());
        }
    }

    if has_children {
        skip_subtree(reader, start)?;
    }
    Ok(())
}

/// Consumes and discards every event up to and including the `End` event
/// matching `start`'s (qualified) name — used both for foreign-namespace
/// extension elements and for entry elements that were unexpectedly given
/// child content.
fn skip_subtree(reader: &mut NsReader<&[u8]>, start: &BytesStart<'_>) -> Result<(), CatalogError> {
    reader
        .read_to_end(start.name())
        .map_err(CatalogError::Xml)?;
    Ok(())
}

/// Resolves the `xml:base` attribute of `start`, if present, against `base`;
/// otherwise returns `base` unchanged.
fn compute_base(base: &Iri<String>, start: &BytesStart<'_>) -> Result<Iri<String>, CatalogError> {
    match find_attr(start, b"xml:base")? {
        Some(value) => Ok(absolutize_iri(base, &value)?),
        None => Ok(base.clone()),
    }
}

/// Resolves the `prefer` attribute of `start`, if present, against the
/// currently-inherited `prefer` scope.
fn compute_prefer(current: Prefer, start: &BytesStart<'_>) -> Result<Prefer, CatalogError> {
    match find_attr(start, b"prefer")? {
        Some(value) if value == "system" => Ok(Prefer::System),
        Some(value) if value == "public" => Ok(Prefer::Public),
        Some(value) => Err(CatalogError::InvalidPreferValue { value }),
        None => Ok(current),
    }
}

/// Returns the decoded, normalized value of a required attribute, or a
/// [`CatalogError::MissingAttribute`] if it is absent.
fn req_attr(
    start: &BytesStart<'_>,
    element: &str,
    attribute: &str,
) -> Result<String, CatalogError> {
    find_attr(start, attribute.as_bytes())?.ok_or_else(|| CatalogError::MissingAttribute {
        element: element.to_owned(),
        attribute: attribute.to_owned(),
    })
}

/// Returns the decoded, normalized value of `key`, if `start` carries such an
/// attribute.
fn find_attr(start: &BytesStart<'_>, key: &[u8]) -> Result<Option<String>, CatalogError> {
    for attr in start.attributes() {
        let attr = attr.map_err(|e: AttrError| CatalogError::Xml(XmlError::from(e)))?;
        if QName(key) == attr.key {
            let value = attr
                .normalized_value(XmlVersion::Implicit1_0)
                .map_err(CatalogError::Xml)?;
            return Ok(Some(value.into_owned()));
        }
    }
    Ok(None)
}

/// Parses `raw` as an IRI reference and resolves it against `base`, returning
/// the absolutized [`Iri`].
fn absolutize_iri(base: &Iri<String>, raw: &str) -> Result<Iri<String>, CatalogError> {
    let iri_ref = IriRef::parse(raw).map_err(|source| CatalogError::InvalidIri {
        value: raw.to_owned(),
        source,
    })?;
    base.resolve(&iri_ref)
        .map_err(|source| CatalogError::InvalidIri {
            value: raw.to_owned(),
            source,
        })
}

/// Parses `raw` as an IRI reference, resolves it against `base`, and returns
/// the absolutized URI as an owned `String`.
fn absolutize(base: &Iri<String>, raw: &str) -> Result<String, CatalogError> {
    Ok(absolutize_iri(base, raw)?.into_inner())
}

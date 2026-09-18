//! The processing algorithm: walking the tree, resolving `xi:include`, and grafting content.

use oxixml_dom::{Document, NodeId, NodeKind, build};
use oxixml_encoding::{Encoding, detect_bom};
use oxixml_iri::{Iri, IriRef};

use crate::error::XIncludeError;
use crate::fixup;
use crate::loader::ResourceLoader;
use crate::names::{FALLBACK_LOCAL_NAME, INCLUDE_LOCAL_NAME, xinclude_local_name};
use crate::options::XIncludeOptions;
use crate::xpointer;

/// The greatest depth of plain (non-`xi:include`) element nesting [`process_children`] will
/// descend through before reporting [`XIncludeError::ElementNestingTooDeep`] instead of
/// exhausting the native call stack.
///
/// `process_children` recurses once per level of *ordinary* element nesting on its way to
/// finding an `xi:include`, entirely separate from [`XIncludeOptions::max_depth`], which bounds
/// nested resource *acquisitions* rather than plain element nesting. Every document this crate
/// actually reads from text is already bounded well below this by the XML reader's own element
/// nesting cap, so this only guards a [`Document`](oxixml_dom::Document) assembled directly
/// through its public tree-mutation API (`create_element`/`append_child`), which carries no
/// depth limit of its own. Matches the workspace-wide ceiling for structural nesting of
/// DOM-like trees.
const MAX_ELEMENT_DEPTH: usize = 256;

/// The identity of a currently-open inclusion, used for loop detection: the *include location*
/// (the resolved, absolute IRI the resource was fetched from — or, for a self-reference, the
/// including document's own base URI) together with the raw `xpointer` attribute value, exactly
/// as [XInclude 1.0 §4.2.7](https://www.w3.org/TR/xinclude/#loops) defines a loop.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LoopKey {
    location: String,
    xpointer: Option<String>,
}

/// Everything about the *document currently being processed* that stays constant across one
/// call to [`process_document_tree`]: its own pristine copy (for `href=""` self-references), its
/// intrinsic base URI, and the loader/options threaded down from the top-level call.
struct DocCtx<'a> {
    self_snapshot: &'a Document,
    doc_base: &'a Iri<String>,
    loader: &'a dyn ResourceLoader,
    options: &'a XIncludeOptions,
}

/// Runs XInclude processing over `document` in place, using the default [`XIncludeOptions`].
///
/// `base_iri` is the document's own base URI — normally the IRI it was retrieved from — used to
/// resolve every relative `href` and as the target of `href=""` self-references.
///
/// # Errors
///
/// Returns [`XIncludeError`] on the first fatal error encountered (see the type's documentation
/// for what counts as fatal); recoverable resource errors are handled internally via
/// `xi:fallback` and never surface here.
pub fn process_document(
    document: &mut Document,
    base_iri: &str,
    loader: &dyn ResourceLoader,
) -> Result<(), XIncludeError> {
    process_document_with(document, base_iri, loader, &XIncludeOptions::default())
}

/// Like [`process_document`], with explicit [`XIncludeOptions`].
///
/// # Errors
///
/// See [`process_document`].
pub fn process_document_with(
    document: &mut Document,
    base_iri: &str,
    loader: &dyn ResourceLoader,
    options: &XIncludeOptions,
) -> Result<(), XIncludeError> {
    let base = Iri::parse(base_iri.to_owned()).map_err(|source| XIncludeError::InvalidBaseIri {
        value: base_iri.to_owned(),
        source,
    })?;
    let mut loop_stack = Vec::new();
    process_document_tree(document, &base, &mut loop_stack, 0, loader, options)
}

/// Processes every `xi:include` in `doc`, treating `doc` as a freshly acquired document in its
/// own right: `doc_base` is its intrinsic base URI, and a fresh snapshot of `doc` is taken
/// up-front to serve any `href=""` self-reference found while walking it.
fn process_document_tree(
    doc: &mut Document,
    doc_base: &Iri<String>,
    loop_stack: &mut Vec<LoopKey>,
    depth: usize,
    loader: &dyn ResourceLoader,
    options: &XIncludeOptions,
) -> Result<(), XIncludeError> {
    if depth > options.max_depth() {
        return Err(XIncludeError::MaxDepthExceeded);
    }
    let snapshot = doc.clone();
    let ctx = DocCtx {
        self_snapshot: &snapshot,
        doc_base,
        loader,
        options,
    };
    // `element_depth` starts fresh at 0 for every freshly acquired document: it bounds how
    // deep *this* call walks into *this* tree, entirely separate from `depth` above (which
    // bounds the resource-acquisition chain across however many nested documents this ends up
    // visiting).
    process_children(doc, doc.root(), &ctx, loop_stack, depth, 0)
}

/// Walks the children of `parent` in document order, descending into ordinary elements and
/// dispatching `xi:include`/`xi:fallback` elements found in the [`XINCLUDE_NAMESPACE`].
///
/// `element_depth` is how many plain elements this call is nested inside of within the current
/// document tree, checked against [`MAX_ELEMENT_DEPTH`] before descending any further — separate
/// from `depth`, which bounds the resource-acquisition chain (see
/// [`XIncludeOptions::max_depth`]) rather than plain element nesting.
///
/// [`XINCLUDE_NAMESPACE`]: crate::XINCLUDE_NAMESPACE
fn process_children(
    doc: &mut Document,
    parent: NodeId,
    ctx: &DocCtx<'_>,
    loop_stack: &mut Vec<LoopKey>,
    depth: usize,
    element_depth: usize,
) -> Result<(), XIncludeError> {
    if element_depth > MAX_ELEMENT_DEPTH {
        return Err(XIncludeError::ElementNestingTooDeep);
    }
    let children: Vec<NodeId> = doc.children(parent).collect();
    for child in children {
        // A sibling's `xi:include`/`xi:fallback` handling may have already moved or removed
        // this node (for example, when it was inside an `xi:fallback` that got spliced away by
        // an *earlier* iteration is impossible since siblings are independent — but a node can
        // legitimately have been detached if it *was itself* an `xi:include`/`xi:fallback`
        // processed by an earlier step in this very loop). Skip anything no longer in place.
        if doc.parent(child) != Some(parent) {
            continue;
        }
        let Some(local) = xinclude_local_name(doc, child) else {
            if doc.kind(child) == Some(NodeKind::Element) {
                process_children(doc, child, ctx, loop_stack, depth, element_depth + 1)?;
            }
            continue;
        };
        match local {
            INCLUDE_LOCAL_NAME => {
                handle_include(doc, parent, child, ctx, loop_stack, depth, element_depth)?
            }
            FALLBACK_LOCAL_NAME => {
                return Err(XIncludeError::FallbackOutsideInclude { at: child });
            }
            other => {
                return Err(XIncludeError::UnsupportedXIncludeElement {
                    local: other.to_owned(),
                    at: child,
                });
            }
        }
    }
    Ok(())
}

/// Reads and validates one `xi:include` element's attributes, resolves its `href`, and dispatches
/// to text or XML inclusion.
///
/// `element_depth` is [`process_children`]'s own, carried through unchanged: resolving this one
/// `xi:include` costs no *further* plain-element nesting of the surrounding tree, but a failed
/// resource may fall back to `xi:fallback` content still nested exactly this deep.
#[allow(clippy::too_many_arguments)]
fn handle_include(
    doc: &mut Document,
    parent: NodeId,
    include_id: NodeId,
    ctx: &DocCtx<'_>,
    loop_stack: &mut Vec<LoopKey>,
    depth: usize,
    element_depth: usize,
) -> Result<(), XIncludeError> {
    let href = doc
        .get_attribute(include_id, "href")
        .unwrap_or("")
        .to_owned();
    let parse_type = doc
        .get_attribute(include_id, "parse")
        .unwrap_or("xml")
        .to_owned();
    let xpointer_attr = doc.get_attribute(include_id, "xpointer").map(str::to_owned);
    let encoding_attr = doc.get_attribute(include_id, "encoding").map(str::to_owned);
    let accept = doc.get_attribute(include_id, "accept").map(str::to_owned);
    let accept_language = doc
        .get_attribute(include_id, "accept-language")
        .map(str::to_owned);

    if parse_type != "xml" && parse_type != "text" {
        return Err(XIncludeError::UnsupportedParseType {
            value: parse_type,
            at: include_id,
        });
    }
    for (name, value) in [
        ("accept", accept.as_deref()),
        ("accept-language", accept_language.as_deref()),
    ] {
        if let Some(v) = value {
            if !v.bytes().all(|b| (0x20..=0x7E).contains(&b)) {
                return Err(XIncludeError::InvalidHeaderValue {
                    attribute: name,
                    value: v.to_owned(),
                    at: include_id,
                });
            }
        }
    }
    if parse_type == "text" {
        if xpointer_attr.is_some() {
            return Err(XIncludeError::XpointerNotAllowedForText { at: include_id });
        }
        if href.is_empty() {
            return Err(XIncludeError::MissingHref { at: include_id });
        }
    } else if href.is_empty() && xpointer_attr.is_none() {
        return Err(XIncludeError::MissingHref { at: include_id });
    }

    // The base URI in effect at the `xi:include` element itself (self-inclusive: an `xml:base`
    // on the element itself governs how its own `href` is resolved).
    let href_base = fixup::effective_base(doc, include_id, ctx.doc_base)?;

    // XInclude 1.0 §3.1: the content model of `xi:include` is checked whether or not the
    // resource turns out to be reachable — more than one `xi:fallback`, or any other element
    // from the XInclude namespace, is a fatal error in its own right.
    check_include_content(doc, include_id)?;

    let self_reference = href.is_empty();
    let location: Iri<String> = if self_reference {
        ctx.doc_base.clone()
    } else {
        // §4.1.1: `href` is an *IRI reference written in an attribute value*, which must be
        // escaped before it is interpreted as one.
        let escaped = escape_href(&href);
        let href_ref =
            IriRef::<String>::parse(escaped).map_err(|source| XIncludeError::InvalidHref {
                value: href.clone(),
                source,
                at: include_id,
            })?;
        if href_ref.fragment().is_some() {
            return Err(XIncludeError::FragmentInHref {
                value: href,
                at: include_id,
            });
        }
        href_base
            .resolve(&href_ref)
            .map_err(|source| XIncludeError::InvalidHref {
                value: href,
                source,
                at: include_id,
            })?
    };

    if parse_type == "text" {
        return handle_text_include(
            doc,
            parent,
            include_id,
            &location,
            encoding_attr.as_deref(),
            accept.as_deref(),
            accept_language.as_deref(),
            ctx,
            loop_stack,
            depth,
            element_depth,
        );
    }

    // `parse="xml"`: loop detection covers the whole current inclusion chain, since a resource
    // acquired this way is itself recursively processed for further `xi:include` elements.
    let key = LoopKey {
        location: location.as_str().to_owned(),
        xpointer: xpointer_attr.clone(),
    };
    if loop_stack.contains(&key) {
        return Err(XIncludeError::LoopDetected {
            location: key.location,
            xpointer: key.xpointer,
            at: include_id,
        });
    }
    loop_stack.push(key);
    let result = handle_xml_include(
        doc,
        parent,
        include_id,
        &location,
        xpointer_attr.as_deref(),
        accept.as_deref(),
        accept_language.as_deref(),
        self_reference,
        ctx,
        loop_stack,
        depth,
        element_depth,
    );
    loop_stack.pop();
    result
}

/// XInclude 1.0 §4.1.1: turns the `href` attribute's *value* into a URI reference.
///
/// "Each character disallowed by [RFC 3986] — the space, the control characters, and
/// `<`, `>`, `"`, `{`, `}`, `|`, `\`, `^` and `` ` `` — and each character outside US-ASCII
/// must be escaped" as the `%HH` encoding of its UTF-8 bytes. `%` and `#` are deliberately left
/// alone, so an `href` that is already escaped is not escaped twice and a fragment identifier
/// still parses as one (and is then rejected by the caller, per §4.1.1).
fn escape_href(href: &str) -> String {
    let mut out = String::with_capacity(href.len());
    for ch in href.chars() {
        if needs_escaping(ch) {
            let mut buf = [0u8; 4];
            for byte in ch.encode_utf8(&mut buf).as_bytes() {
                out.push('%');
                out.push(hex_digit(byte >> 4));
                out.push(hex_digit(byte & 0x0f));
            }
        } else {
            out.push(ch);
        }
    }
    out
}

/// Whether [`escape_href`] must percent-encode `ch`.
const fn needs_escaping(ch: char) -> bool {
    if !ch.is_ascii() {
        return true;
    }
    matches!(
        ch,
        ' ' | '<' | '>' | '"' | '{' | '}' | '|' | '\\' | '^' | '`'
    ) || (ch as u32) < 0x20
        || ch as u32 == 0x7f
}

/// The upper-case hexadecimal digit of `nibble` (which is always `0..=15`).
const fn hex_digit(nibble: u8) -> char {
    match nibble {
        0..=9 => (b'0' + nibble) as char,
        _ => (b'A' + (nibble - 10)) as char,
    }
}

/// XInclude 1.0 §3.1: checks the content model of an `xi:include` element.
///
/// At most one `xi:fallback` child is allowed, and no other element from the XInclude namespace
/// may appear among its children. Both are fatal errors, independent of whether the referenced
/// resource can be retrieved — which is why this runs before the resource is even requested.
fn check_include_content(doc: &Document, include_id: NodeId) -> Result<(), XIncludeError> {
    let mut fallbacks = 0usize;
    for child in doc.children(include_id) {
        match xinclude_local_name(doc, child) {
            Some(FALLBACK_LOCAL_NAME) => fallbacks += 1,
            Some(other) => {
                return Err(XIncludeError::UnsupportedXIncludeElement {
                    local: other.to_owned(),
                    at: child,
                });
            }
            None => {}
        }
    }
    if fallbacks > 1 {
        return Err(XIncludeError::FallbackCount {
            found: fallbacks,
            at: include_id,
        });
    }
    Ok(())
}

/// Fetches and decodes a `parse="text"` resource, replacing the `xi:include` element with a
/// single text node (or, on a resource error, running `xi:fallback`).
#[allow(clippy::too_many_arguments)]
fn handle_text_include(
    doc: &mut Document,
    parent: NodeId,
    include_id: NodeId,
    location: &Iri<String>,
    encoding_attr: Option<&str>,
    accept: Option<&str>,
    accept_language: Option<&str>,
    ctx: &DocCtx<'_>,
    loop_stack: &mut Vec<LoopKey>,
    depth: usize,
    element_depth: usize,
) -> Result<(), XIncludeError> {
    let bytes = match ctx.loader.load(location.as_str(), accept, accept_language) {
        Ok(bytes) => bytes,
        Err(_) => {
            return use_fallback(
                doc,
                parent,
                include_id,
                ctx,
                loop_stack,
                depth,
                element_depth,
            );
        }
    };

    // XInclude 1.0 §4.3: a byte-order mark, when present, identifies the encoding outright;
    // otherwise fall back to the `encoding` attribute, and finally to UTF-8.
    let (encoding, skip) = match detect_bom(&bytes) {
        Some((encoding, skip)) => (encoding, skip),
        None => {
            let encoding = match encoding_attr {
                Some(label) => match Encoding::for_label(label) {
                    Some(encoding) => encoding,
                    None => {
                        return use_fallback(
                            doc,
                            parent,
                            include_id,
                            ctx,
                            loop_stack,
                            depth,
                            element_depth,
                        );
                    }
                },
                None => Encoding::Utf8,
            };
            (encoding, 0)
        }
    };

    let mut decoder = encoding.new_decoder();
    let mut text = String::new();
    if decoder
        .decode_to_string(&bytes[skip..], &mut text, true)
        .is_err()
    {
        return use_fallback(
            doc,
            parent,
            include_id,
            ctx,
            loop_stack,
            depth,
            element_depth,
        );
    }
    if !text.chars().all(oxixml_dom::is_xml_char) {
        return use_fallback(
            doc,
            parent,
            include_id,
            ctx,
            loop_stack,
            depth,
            element_depth,
        );
    }

    let text_node = doc.create_text_node(text)?;
    doc.replace_child(parent, text_node, include_id)?;
    Ok(())
}

/// Fetches (or, for a self-reference, extracts), recursively XInclude-processes, selects a
/// target via `xpointer`, and grafts a `parse="xml"` resource, applying base/language fixup to
/// the grafted root — or, on a resource error, runs `xi:fallback`.
///
/// For a self-reference (`href=""`), the target is first located *within the pristine
/// snapshot* and copied out into a small standalone document of its own before recursive
/// processing begins. This deliberately avoids re-walking unrelated parts of the including
/// document — in particular the very same `xi:include` element the self-reference started
/// from, which sits elsewhere in the snapshot and would otherwise immediately collide with
/// itself in the loop-detection stack even though the *selected* content has nothing to do with
/// it. A consequence is that a *nested* self-reference found inside a self-referenced fragment
/// resolves against that (smaller) fragment rather than the original document as a whole; this
/// is an intentionally narrow, rarely-exercised edge case of an already-rare feature.
#[allow(clippy::too_many_arguments)]
fn handle_xml_include(
    doc: &mut Document,
    parent: NodeId,
    include_id: NodeId,
    location: &Iri<String>,
    xpointer_attr: Option<&str>,
    accept: Option<&str>,
    accept_language: Option<&str>,
    self_reference: bool,
    ctx: &DocCtx<'_>,
    loop_stack: &mut Vec<LoopKey>,
    depth: usize,
    element_depth: usize,
) -> Result<(), XIncludeError> {
    if depth + 1 > ctx.options.max_depth() {
        return Err(XIncludeError::MaxDepthExceeded);
    }

    let (acquired, target) = if self_reference {
        let selected = match xpointer_attr {
            Some(pointer) => xpointer::evaluate(ctx.self_snapshot, pointer),
            None => ctx.self_snapshot.document_element(),
        };
        let is_element = |&n: &NodeId| ctx.self_snapshot.kind(n) == Some(NodeKind::Element);
        let Some(selected) = selected.filter(is_element) else {
            return use_fallback(
                doc,
                parent,
                include_id,
                ctx,
                loop_stack,
                depth,
                element_depth,
            );
        };
        let mut extracted = Document::new();
        let imported = extracted.import_node(ctx.self_snapshot, selected, true)?;
        extracted.append_child(extracted.root(), imported)?;
        process_document_tree(
            &mut extracted,
            location,
            loop_stack,
            depth + 1,
            ctx.loader,
            ctx.options,
        )?;
        let target = extracted.document_element();
        (extracted, target)
    } else {
        let mut parsed = match ctx.loader.load(location.as_str(), accept, accept_language) {
            Ok(bytes) => match build::parse_reader(bytes.as_slice()) {
                Ok(document) => document,
                Err(source) => {
                    return Err(XIncludeError::MalformedXml {
                        location: location.as_str().to_owned(),
                        source,
                        at: include_id,
                    });
                }
            },
            Err(_) => {
                return use_fallback(
                    doc,
                    parent,
                    include_id,
                    ctx,
                    loop_stack,
                    depth,
                    element_depth,
                );
            }
        };
        process_document_tree(
            &mut parsed,
            location,
            loop_stack,
            depth + 1,
            ctx.loader,
            ctx.options,
        )?;
        let target = match xpointer_attr {
            Some(pointer) => xpointer::evaluate(&parsed, pointer),
            None => parsed.document_element(),
        };
        (parsed, target)
    };

    let Some(target) = target else {
        return use_fallback(
            doc,
            parent,
            include_id,
            ctx,
            loop_stack,
            depth,
            element_depth,
        );
    };
    if acquired.kind(target) != Some(NodeKind::Element) {
        return use_fallback(
            doc,
            parent,
            include_id,
            ctx,
            loop_stack,
            depth,
            element_depth,
        );
    }

    // Fixup compares the item's own (acquired) effective base/language to the ones in effect at
    // its *include parent* — the actual DOM parent of the `xi:include` element, not the
    // `xi:include` element itself (which may carry its own, unrelated `xml:base`).
    let acquired_base = fixup::effective_base(&acquired, target, location)?;
    let parent_base = fixup::effective_base(doc, parent, ctx.doc_base)?;
    let acquired_lang = fixup::effective_lang(&acquired, target);
    let parent_lang = fixup::effective_lang(doc, parent);

    let grafted = doc.import_node(&acquired, target, true)?;

    if acquired_base.as_str() != parent_base.as_str() {
        fixup::apply_base_fixup(doc, grafted, &acquired_base)?;
    }
    if fixup::lang_differs(acquired_lang.as_deref(), parent_lang.as_deref()) {
        fixup::apply_lang_fixup(doc, grafted, acquired_lang.as_deref())?;
    }

    doc.replace_child(parent, grafted, include_id)?;
    Ok(())
}

/// Runs `xi:fallback` processing (XInclude 1.0 §4.4) for an `xi:include` that hit a resource
/// error: requires exactly one `xi:fallback` child, recursively processes its content in place,
/// then splices that content in where the `xi:include` element was.
fn use_fallback(
    doc: &mut Document,
    parent: NodeId,
    include_id: NodeId,
    ctx: &DocCtx<'_>,
    loop_stack: &mut Vec<LoopKey>,
    depth: usize,
    element_depth: usize,
) -> Result<(), XIncludeError> {
    let fallback_children: Vec<NodeId> = doc
        .children(include_id)
        .filter(|&child| xinclude_local_name(doc, child) == Some(FALLBACK_LOCAL_NAME))
        .collect();
    if fallback_children.len() != 1 {
        return Err(XIncludeError::FallbackCount {
            found: fallback_children.len(),
            at: include_id,
        });
    }
    let fallback = fallback_children[0];

    // `xi:fallback` content sits exactly as deep in `doc` as the `xi:include` it replaces, so
    // this continues from the caller's own `element_depth` rather than resetting to 0 — a chain
    // of `xi:include`/`xi:fallback` nested inside one another must still be bounded overall.
    process_children(doc, fallback, ctx, loop_stack, depth, element_depth)?;

    let moved: Vec<NodeId> = doc.children(fallback).collect();
    for child in moved {
        doc.detach(child)?;
        doc.insert_before(parent, child, Some(include_id))?;
    }
    doc.remove_child(parent, include_id)?;
    Ok(())
}

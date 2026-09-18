//! The OASIS XML Catalogs 1.1 §7 resolution algorithm.
//!
//! A single catalog document never has to reach outside itself: matching
//! `public`, `system`, `uri`, `rewriteSystem`, `rewriteURI`, `systemSuffix`
//! and `uriSuffix` entries are all resolved by scanning
//! [`crate::model::CatalogFile::entries`] in document order. Only
//! `nextCatalog` and `delegate*` entries need another document's text, which
//! is why they go through the caller-supplied [`CatalogLoader`] instead of
//! touching the filesystem or network directly.

use std::collections::{HashSet, VecDeque};

use oxixml_iri::Iri;

use crate::model::{CatalogFile, Entry, Prefer};
use crate::normalize::{normalize_public_id, normalize_uri_ref, unwrap_publicid};
use crate::parser::parse_document;

/// The greatest number of `delegatePublic`/`delegateSystem`/`delegateURI` restarts catalog
/// resolution will follow before giving up and reporting no match.
///
/// The `visited` loop-detector stops a delegate chain from re-entering a catalog URI it has
/// already seen, but it does nothing to bound the depth of a chain that only ever delegates to
/// *new* URIs — for instance a misbehaving [`CatalogLoader`] that answers every request with a
/// catalog delegating to yet another, never-before-seen URI. Without this cap that shape of
/// input recurses the native call stack without bound; this is the same "legal, non-repeating
/// chain of distinct resources" hazard that `oxixml-xinclude`'s `XIncludeOptions::max_depth`
/// guards against, for the same reason (their concrete limits differ, chosen independently for
/// each format).
pub const MAX_DELEGATE_DEPTH: usize = 256;

/// Fetches the text of a catalog document referenced by a `nextCatalog` or
/// `delegate*` entry's (already-absolutized) `catalog`/`uriStartString`
/// target.
///
/// [`Catalog`](crate::Catalog) never reads the filesystem or the network on
/// its own — every [`resolve_system_with`](crate::Catalog::resolve_system_with)-family
/// call takes an `Option<&dyn CatalogLoader>`, so the caller decides exactly
/// which catalog URIs may be dereferenced (or passes `None` to resolve using
/// only the entries of the catalog(s) already in hand).
pub trait CatalogLoader {
    /// Returns the textual content of the catalog document identified by the
    /// absolute URI `uri`, or `None` if it is unavailable (not found, denied
    /// by policy, I/O error, and so on).
    ///
    /// A `None` return is not an error: per OASIS XML Catalogs 1.1 §9
    /// ("Errors"), an unreachable catalog simply contributes no matches, and
    /// resolution continues with whatever catalog entries remain.
    fn load(&self, uri: &str) -> Option<String>;
}

/// The result of scanning a single [`CatalogFile`] for a match: either a
/// resolved replacement URI, or a delegation to a fresh, priority-ordered list
/// of other catalogs.
enum Outcome {
    /// A `public`, `system`, `uri`, `rewriteSystem`, `rewriteURI`,
    /// `systemSuffix` or `uriSuffix` entry matched.
    Found(String),
    /// One or more `delegatePublic`, `delegateSystem` or `delegateURI` entries
    /// matched; resolution must continue *only* within these catalogs
    /// (longest-matching-prefix first), abandoning the rest of the original
    /// catalog list.
    Delegate(Vec<String>),
}

/// A catalog file pending examination: either the caller's own document
/// (borrowed, never cloned) or a URI to fetch — lazily, only once it is
/// actually about to be scanned, so that a match in an earlier-queued
/// catalog never causes a later one to be needlessly requested through the
/// [`CatalogLoader`].
enum Pending<'a> {
    Root(&'a CatalogFile),
    Unloaded(String),
}

/// Loads and parses the catalog document at `uri`, or returns `None` if it is
/// unavailable or malformed — either of which simply means "no matches from
/// this catalog" (OASIS XML Catalogs 1.1 §9).
fn load_catalog(
    uri: &str,
    loader: Option<&dyn CatalogLoader>,
    default_prefer: Prefer,
) -> Option<CatalogFile> {
    let text = loader?.load(uri)?;
    let base = Iri::parse(uri.to_owned()).ok()?;
    parse_document(&text, base, default_prefer).ok()
}

/// Returns the (already-absolutized) `catalog` targets of every `nextCatalog`
/// entry in `file`, in document order.
fn next_catalogs(file: &CatalogFile) -> impl Iterator<Item = &str> {
    file.entries.iter().filter_map(|e| match e {
        Entry::NextCatalog { catalog } => Some(catalog.as_str()),
        _ => None,
    })
}

/// Returns the replacement produced by rewriting `subject` with whichever
/// matching `(start, prefix)` pair (extracted from `file`'s entries by
/// `extract`) has the longest `start`, per the "longest match wins" rule
/// shared by `rewriteSystem` and `rewriteURI` (OASIS XML Catalogs 1.1 §7.1.1,
/// §7.2.1). Ties keep the first (document order) candidate.
fn longest_rewrite<'e>(
    file: &'e CatalogFile,
    subject: &str,
    extract: impl Fn(&'e Entry) -> Option<(&'e str, &'e str)>,
) -> Option<String> {
    let mut best: Option<(&str, &str)> = None;
    for entry in &file.entries {
        if let Some((start, prefix)) = extract(entry) {
            if subject.starts_with(start) && best.is_none_or(|(b, _)| start.len() > b.len()) {
                best = Some((start, prefix));
            }
        }
    }
    best.map(|(start, prefix)| format!("{prefix}{}", &subject[start.len()..]))
}

/// Returns the replacement URI of whichever matching `(suffix, uri)` pair
/// (extracted from `file`'s entries by `extract`) has the longest `suffix`,
/// per the `systemSuffix`/`uriSuffix` "longest match wins" rule (OASIS XML
/// Catalogs 1.1 §7.1.3, §7.2.3). Unlike [`longest_rewrite`], the matched entry's
/// `uri` is a direct replacement, not a prefix to splice onto a suffix.
fn longest_suffix<'e>(
    file: &'e CatalogFile,
    subject: &str,
    extract: impl Fn(&'e Entry) -> Option<(&'e str, &'e str)>,
) -> Option<String> {
    let mut best: Option<(&str, &str)> = None;
    for entry in &file.entries {
        if let Some((suffix, uri)) = extract(entry) {
            if subject.ends_with(suffix) && best.is_none_or(|(b, _)| suffix.len() > b.len()) {
                best = Some((suffix, uri));
            }
        }
    }
    best.map(|(_, uri)| uri.to_owned())
}

/// Collects every matching `(start, catalog)` pair (extracted from `file`'s
/// entries by `extract`) whose `start` is a prefix of `subject`, sorted with
/// the longest `start` first (ties keep document order) — the delegate-list
/// construction rule shared by `delegatePublic`, `delegateSystem` and
/// `delegateURI` (OASIS XML Catalogs 1.1 §7.1.2/§7.1.4/§7.2.2).
fn prefix_delegates<'e>(
    file: &'e CatalogFile,
    subject: &str,
    extract: impl Fn(&'e Entry) -> Option<(&'e str, &'e str)>,
) -> Vec<String> {
    let mut matches: Vec<(&str, &str)> = file
        .entries
        .iter()
        .filter_map(extract)
        .filter(|(start, _)| subject.starts_with(start))
        .collect();
    matches.sort_by_key(|(start, _)| std::cmp::Reverse(start.len()));
    matches
        .into_iter()
        .map(|(_, catalog)| catalog.to_owned())
        .collect()
}

/// Scans `file` for a `uri` resolution outcome (OASIS XML Catalogs 1.1 §7.2):
/// direct `uri` match, then longest-matching `rewriteURI`, then
/// longest-matching `uriSuffix`, then `delegateURI` delegation.
fn scan_uri(file: &CatalogFile, target: &str) -> Option<Outcome> {
    for entry in &file.entries {
        if let Entry::Uri { name, uri } = entry {
            if name == target {
                return Some(Outcome::Found(uri.clone()));
            }
        }
    }
    if let Some(uri) = longest_rewrite(file, target, |e| match e {
        Entry::RewriteUri { start, prefix } => Some((start.as_str(), prefix.as_str())),
        _ => None,
    }) {
        return Some(Outcome::Found(uri));
    }
    if let Some(uri) = longest_suffix(file, target, |e| match e {
        Entry::UriSuffix { suffix, uri } => Some((suffix.as_str(), uri.as_str())),
        _ => None,
    }) {
        return Some(Outcome::Found(uri));
    }
    let delegates = prefix_delegates(file, target, |e| match e {
        Entry::DelegateUri { start, catalog } => Some((start.as_str(), catalog.as_str())),
        _ => None,
    });
    if delegates.is_empty() {
        None
    } else {
        Some(Outcome::Delegate(delegates))
    }
}

/// Scans `file` for an external-identifier resolution outcome (OASIS XML
/// Catalogs 1.1 §7.1): when `system_id` is present, direct `system` match,
/// then longest-matching `rewriteSystem`, then longest-matching
/// `systemSuffix`, then `delegateSystem` delegation — all of which take
/// precedence over the `public`/`delegatePublic` path below. The
/// `public`/`delegatePublic` path is attempted only if the entry's `prefer`
/// scope is `Public`, or no `system_id` was supplied at all.
fn scan_external(
    file: &CatalogFile,
    system_id: Option<&str>,
    public_id: Option<&str>,
) -> Option<Outcome> {
    if let Some(sid) = system_id {
        for entry in &file.entries {
            if let Entry::System { system_id: id, uri } = entry {
                if id == sid {
                    return Some(Outcome::Found(uri.clone()));
                }
            }
        }
        if let Some(uri) = longest_rewrite(file, sid, |e| match e {
            Entry::RewriteSystem { start, prefix } => Some((start.as_str(), prefix.as_str())),
            _ => None,
        }) {
            return Some(Outcome::Found(uri));
        }
        if let Some(uri) = longest_suffix(file, sid, |e| match e {
            Entry::SystemSuffix { suffix, uri } => Some((suffix.as_str(), uri.as_str())),
            _ => None,
        }) {
            return Some(Outcome::Found(uri));
        }
        let delegates = prefix_delegates(file, sid, |e| match e {
            Entry::DelegateSystem { start, catalog } => Some((start.as_str(), catalog.as_str())),
            _ => None,
        });
        if !delegates.is_empty() {
            return Some(Outcome::Delegate(delegates));
        }
    }

    let pid = public_id?;
    let system_present = system_id.is_some();
    for entry in &file.entries {
        if let Entry::Public {
            public_id: id,
            uri,
            prefer,
        } = entry
        {
            if (*prefer == Prefer::Public || !system_present) && id == pid {
                return Some(Outcome::Found(uri.clone()));
            }
        }
    }
    let delegates = prefix_delegates(file, pid, |e| match e {
        Entry::DelegatePublic {
            start,
            catalog,
            prefer,
        } if *prefer == Prefer::Public || !system_present => {
            Some((start.as_str(), catalog.as_str()))
        }
        _ => None,
    });
    if delegates.is_empty() {
        None
    } else {
        Some(Outcome::Delegate(delegates))
    }
}

/// Drives the catalog-list traversal of OASIS XML Catalogs 1.1 §7: examines
/// each pending file in turn (a match in an earlier file always wins over a
/// later one); on no match, `nextCatalog` entries extend the list; on a
/// `Delegate` outcome, resolution restarts from scratch with *only* the
/// delegate list. `visited` provides loop detection across the whole call —
/// shared by both the top-level list and every delegate sub-list, so a
/// catalog already examined (by URI) is never re-entered. `visited` alone
/// cannot bound the depth of a chain that only ever delegates to *new* URIs,
/// which is what `depth` and [`MAX_DELEGATE_DEPTH`] are for: once it is
/// reached, the chain is abandoned exactly as OASIS XML Catalogs 1.1 §9
/// already treats an unreachable or malformed catalog — contributing no
/// match, not an error — since delegation this deep is indistinguishable in
/// practice from a misbehaving [`CatalogLoader`] that never terminates.
fn drive<'a>(
    mut queue: VecDeque<Pending<'a>>,
    scan: impl Fn(&CatalogFile) -> Option<Outcome> + Copy,
    loader: Option<&dyn CatalogLoader>,
    default_prefer: Prefer,
    visited: &mut HashSet<String>,
    depth: usize,
) -> Option<String> {
    while let Some(pending) = queue.pop_front() {
        let loaded_owner;
        let file: &CatalogFile = match &pending {
            Pending::Root(f) => f,
            Pending::Unloaded(uri) => match load_catalog(uri, loader, default_prefer) {
                Some(f) => {
                    loaded_owner = f;
                    &loaded_owner
                }
                // Unreachable or malformed: contributes no match; move on to
                // whatever else remains in the queue.
                None => continue,
            },
        };
        match scan(file) {
            Some(Outcome::Found(uri)) => return Some(uri),
            Some(Outcome::Delegate(catalog_uris)) => {
                if depth >= MAX_DELEGATE_DEPTH {
                    return None;
                }
                let mut sub_queue = VecDeque::new();
                for uri in catalog_uris {
                    if visited.insert(uri.clone()) {
                        sub_queue.push_back(Pending::Unloaded(uri));
                    }
                }
                return drive(sub_queue, scan, loader, default_prefer, visited, depth + 1);
            }
            None => {
                for uri in next_catalogs(file) {
                    if visited.insert(uri.to_owned()) {
                        queue.push_back(Pending::Unloaded(uri.to_owned()));
                    }
                }
            }
        }
    }
    None
}

/// Resolves an external identifier (`system_id` and/or `public_id`) against
/// `root`, following `nextCatalog`/`delegate*` entries through `loader`.
pub(crate) fn resolve_external(
    root: &CatalogFile,
    system_id: Option<&str>,
    public_id: Option<&str>,
    loader: Option<&dyn CatalogLoader>,
    default_prefer: Prefer,
) -> Option<String> {
    // §6.4 "URN 'unwrapping'": a system identifier written as an RFC 3151
    // `urn:publicid:` URN *is* a public identifier, so it is unwrapped and
    // resolved through the public path (§7.1.1 step 1). The URN itself is then
    // no longer a system identifier, which also lifts the `prefer="system"`
    // suppression of `public` entries.
    let unwrapped = system_id.and_then(unwrap_publicid);
    let (system_id, public_id) = match &unwrapped {
        Some(from_urn) => (None, public_id.or(Some(from_urn.as_str()))),
        None => (system_id, public_id),
    };
    let system_norm = system_id.map(normalize_uri_ref);
    let public_norm = public_id.map(normalize_public_id);
    let mut visited = HashSet::new();
    visited.insert(root.uri.clone());
    let mut queue = VecDeque::new();
    queue.push_back(Pending::Root(root));
    drive(
        queue,
        |file| scan_external(file, system_norm.as_deref(), public_norm.as_deref()),
        loader,
        default_prefer,
        &mut visited,
        0,
    )
}

/// Resolves a URI reference against `root`, following `nextCatalog`/`delegateURI`
/// entries through `loader`.
pub(crate) fn resolve_uri(
    root: &CatalogFile,
    uri: &str,
    loader: Option<&dyn CatalogLoader>,
    default_prefer: Prefer,
) -> Option<String> {
    // §6.4/§7.2.1 step 1: a URI reference written as an RFC 3151
    // `urn:publicid:` URN is unwrapped into a public identifier and resolved
    // through the `public`/`delegatePublic` path instead.
    if let Some(public_id) = unwrap_publicid(uri) {
        let public_norm = normalize_public_id(&public_id);
        let mut visited = HashSet::new();
        visited.insert(root.uri.clone());
        let mut queue = VecDeque::new();
        queue.push_back(Pending::Root(root));
        return drive(
            queue,
            |file| scan_external(file, None, Some(public_norm.as_str())),
            loader,
            default_prefer,
            &mut visited,
            0,
        );
    }
    let target = normalize_uri_ref(uri);
    let mut visited = HashSet::new();
    visited.insert(root.uri.clone());
    let mut queue = VecDeque::new();
    queue.push_back(Pending::Root(root));
    drive(
        queue,
        |file| scan_uri(file, &target),
        loader,
        default_prefer,
        &mut visited,
        0,
    )
}

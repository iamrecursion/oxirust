//! The focus a template's `xsl:context-item` declaration asks for.
//!
//! [XSLT 3.0 §10.1.1](https://www.w3.org/TR/xslt-30/#context-item) lets a template state whether
//! it needs a context item, whether it wants none at all, and what type it must be. The rules
//! are settled *before* the template's parameters are bound, because `use="absent"` evaluates
//! "the contained sequence constructor, **and any `xsl:param` elements**, with an absent focus".

use super::Context;
use crate::compile::{ContextItemDecl, ContextItemUse};
use crate::error::{XsltError, XsltResult};
use oxixml_infoset::{Item, Sequence};
use oxixml_xpath::XpathValue;

/// Applies the stylesheet's `xsl:global-context-item` declaration to the item the invocation
/// supplied, returning the *global context item* — the focus every global variable and
/// parameter is evaluated with.
///
/// [XSLT 3.0 §3.13](https://www.w3.org/TR/xslt-30/#global-context-item) reads the same way as
/// the per-template declaration of §10.1.1, with its own codes: `use="required"` and nothing
/// supplied is `XTDE3086`, an item outside `as` is `XTTE0590`, and `use="absent"` drops the
/// item, so that a global variable reading `.` reports an absent focus rather than quietly
/// seeing the source document.
///
/// # Errors
///
/// `XTDE3086` for a missing required global context item, `XTTE0590` for one of the wrong type.
pub(crate) fn apply_global_context_item(
    declaration: Option<&ContextItemDecl>,
    supplied: Option<Item>,
) -> XsltResult<Option<Item>> {
    let Some(declaration) = declaration else {
        return Ok(supplied);
    };
    if declaration.usage == ContextItemUse::Absent {
        return Ok(None);
    }
    let Some(item) = supplied else {
        if declaration.usage == ContextItemUse::Required {
            return Err(XsltError::dynamic(
                "XTDE3086",
                "the stylesheet requires a global context item and the invocation supplied none",
            ));
        }
        return Ok(None);
    };
    if let Some(required) = &declaration.as_type {
        let value = XpathValue::from(Sequence::from(item.clone()));
        if !required.matches(&value) {
            return Err(XsltError::dynamic(
                "XTTE0590",
                "the global context item does not match the type xsl:global-context-item requires",
            ));
        }
    }
    Ok(Some(item))
}

/// Applies a template's `xsl:context-item` declaration to the context it is about to run in.
///
/// [XSLT 3.0 §10.1.1](https://www.w3.org/TR/xslt-30/#context-item):
///
/// * `use="required"` — "there must be a context item"; when none was supplied, `XTTE3090`.
/// * `use="absent"` — "the contained sequence constructor, and any `xsl:param` elements, are
///   evaluated with an absent focus", and §6.7 adds that the current template rule is cleared
///   for an `xsl:template` "if and only if the called template specifies
///   `<xsl:context-item use="absent"/>`". It is explicitly *not* an error to call such a
///   template with a focus; the focus is simply dropped.
/// * `as` — the supplied context item must match, with "no attempt … to convert the context
///   item to the required type"; a mismatch carries the same code as `xsl:param`, `XTTE0590`.
///
/// # Errors
///
/// `XTTE3090` for a missing required context item, `XTTE0590` for one of the wrong type.
pub(crate) fn apply_context_item(
    declaration: Option<&ContextItemDecl>,
    context: &mut Context,
) -> XsltResult<()> {
    let Some(declaration) = declaration else {
        return Ok(());
    };
    if declaration.usage == ContextItemUse::Absent {
        context.item = None;
        context.position = 0;
        context.size = 0;
        context.rule = None;
        return Ok(());
    }
    let Some(item) = &context.item else {
        if declaration.usage == ContextItemUse::Required {
            return Err(XsltError::dynamic(
                "XTTE3090",
                "the template requires a context item and none was supplied by the caller",
            ));
        }
        return Ok(());
    };
    if let Some(required) = &declaration.as_type {
        let value = XpathValue::from(Sequence::from(item.clone()));
        if !required.matches(&value) {
            return Err(XsltError::dynamic(
                "XTTE0590",
                "the context item does not match the type xsl:context-item requires",
            ));
        }
    }
    Ok(())
}

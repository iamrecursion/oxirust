//! `xsl:map` and `xsl:map-entry`.
//!
//! [XSLT 3.0 §11.4](https://www.w3.org/TR/xslt-30/#element-map) builds a single map by
//! evaluating `xsl:map`'s content to a sequence of maps — typically nested `xsl:map-entry`, but
//! any map-valued expression counts — and combining them, rejecting a duplicate key; each
//! `xsl:map-entry` in turn builds a single-entry map from its `key` and a value constructed
//! exactly as a variable's is (`Transformer::bound_value`).

use super::{Context, Transformer};
use crate::compile::Instruction;
use crate::error::{XsltError, XsltResult};
use oxixml_infoset::{Item, NodeKind, XdmNode};

/// A short, human-readable name for an item's kind, for a type-error message.
fn item_kind(item: &Item) -> &'static str {
    match item {
        Item::Node(node) => match node.node_kind() {
            NodeKind::Document => "a document node",
            NodeKind::Element => "an element node",
            NodeKind::Attribute => "an attribute node",
            NodeKind::Namespace => "a namespace node",
            NodeKind::Text => "a text node",
            NodeKind::Comment => "a comment node",
            NodeKind::ProcessingInstruction => "a processing-instruction node",
        },
        Item::Atomic(_) => "an atomic value",
        Item::Map(_) => "a map",
        Item::Array(_) => "an array",
        Item::Function(_) => "a function item",
    }
}

impl Transformer {
    /// `xsl:map`: evaluates the content to a sequence of maps and combines them into one.
    ///
    /// Every item the content produces (typically from nested `xsl:map-entry`, but any
    /// map-valued expression counts, such as an `xsl:sequence` selecting a map another
    /// `xsl:map` already built) must itself be a map; the maps are then combined exactly as
    /// `map:merge` with `duplicates="reject"` combines them, so that the same key reached from
    /// two of them is a dynamic error rather than the second one silently winning.
    ///
    /// # Errors
    ///
    /// `XTTE3375` when the content produces an item that is not a map; `XTDE3365` when two of
    /// the maps it does produce share a key; any dynamic error the content itself raises.
    pub(crate) fn evaluate_map(
        &mut self,
        content: &[Instruction],
        context: &Context,
    ) -> XsltResult<oxixml_infoset::XdmMap> {
        let (_, sequence) = self.run_to_sequence(content, None, context)?;
        let mut maps = Vec::with_capacity(sequence.len());
        for item in sequence.iter() {
            match item {
                Item::Map(map) => maps.push(map.clone()),
                other => {
                    return Err(XsltError::dynamic(
                        "XTTE3375",
                        format!(
                            "the content of xsl:map must be a sequence of maps, not {}",
                            item_kind(other)
                        ),
                    ));
                }
            }
        }
        oxixml_infoset::XdmMap::merge(maps, oxixml_infoset::DuplicateKeys::Reject).map_err(|_| {
            XsltError::dynamic(
                "XTDE3365",
                "two of the maps xsl:map's content produced share a key",
            )
        })
    }

    /// `xsl:map-entry`: a single-entry map from `key` and a value built exactly as
    /// [`Transformer::bound_value`] builds a variable's.
    ///
    /// # Errors
    ///
    /// `XPTY0004` when `key` does not atomize to exactly one atomic value; any dynamic error
    /// `key`, `select` or the content itself raises.
    pub(crate) fn evaluate_map_entry(
        &mut self,
        key: &oxixml_xpath::CompiledXPath,
        select: Option<&oxixml_xpath::CompiledXPath>,
        content: &[Instruction],
        context: &Context,
    ) -> XsltResult<oxixml_infoset::XdmMap> {
        let atomized = self.evaluate(key, context)?.into_sequence()?.atomize()?;
        // XSLT 3.0 §11.4: `key` is required and "must evaluate to a single atomic value".
        let key = match atomized.as_slice() {
            [Item::Atomic(value)] => value.clone(),
            [] => {
                return Err(XsltError::dynamic(
                    "XPTY0004",
                    "the key of xsl:map-entry must be a single atomic value, found none",
                ));
            }
            _ => {
                return Err(XsltError::dynamic(
                    "XPTY0004",
                    "the key of xsl:map-entry must be a single atomic value, found more than one",
                ));
            }
        };
        // XSLT 3.0 §11.10: "The associated value is the value obtained by evaluating the
        // expression in the select attribute, or the contained sequence constructor, **with no
        // conversion**." A sequence constructor here is therefore not building a tree the way an
        // `xsl:variable` without `as` does: the items keep their types, so
        // `<xsl:map-entry key="'index'"><xsl:sequence select="."/></xsl:map-entry>` files the
        // integer and not a text node reading it (`result-document-1401`, whose expected JSON
        // has `"index":3` and not `"index":"3"`).
        let value = self.bound_value(select, content, true, None, context)?;
        Ok(oxixml_infoset::XdmMap::new().put(key, value.into_sequence()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::Stylesheet;
    use oxixml_infoset::Tree;

    fn transform(xsl_body: &str, source: &str) -> XsltResult<String> {
        let xsl = format!(
            r#"<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
                 <xsl:output omit-xml-declaration="yes"/>
                 {xsl_body}
               </xsl:stylesheet>"#
        );
        let compiled = Stylesheet::compile_str(&xsl)?;
        let tree = Tree::parse_str(source)
            .map_err(|error| XsltError::dynamic("XTDE0000", error.message().to_owned()))?;
        compiled.transform_to_string(&tree.root())
    }

    #[test]
    fn a_map_entry_builds_a_single_entry_map() -> XsltResult<()> {
        let out = transform(
            r#"<xsl:template match="/">
                 <xsl:variable name="m" as="map(*)">
                   <xsl:map><xsl:map-entry key="'k'" select="42"/></xsl:map>
                 </xsl:variable>
                 <out><xsl:value-of select="map:get($m, 'k')"
                   xmlns:map="http://www.w3.org/2005/xpath-functions/map"/></out>
               </xsl:template>"#,
            "<in/>",
        )?;
        assert_eq!(out, "<out>42</out>");
        Ok(())
    }

    #[test]
    fn a_non_map_item_is_xtte3375() {
        let error = transform(
            r#"<xsl:template match="/">
                 <xsl:variable name="m" as="map(*)"><xsl:map><xsl:sequence select="1"/></xsl:map></xsl:variable>
                 <out><xsl:value-of select="map:size($m)"
                   xmlns:map="http://www.w3.org/2005/xpath-functions/map"/></out>
               </xsl:template>"#,
            "<in/>",
        )
        .expect_err("a bare integer is not a map");
        assert_eq!(error.code(), Some("XTTE3375"));
    }

    #[test]
    fn a_duplicate_key_is_xtde3365() {
        let error = transform(
            r#"<xsl:template match="/">
                 <xsl:variable name="m" as="map(*)">
                   <xsl:map>
                     <xsl:map-entry key="'k'" select="1"/>
                     <xsl:map-entry key="'k'" select="2"/>
                   </xsl:map>
                 </xsl:variable>
                 <out><xsl:value-of select="map:size($m)"
                   xmlns:map="http://www.w3.org/2005/xpath-functions/map"/></out>
               </xsl:template>"#,
            "<in/>",
        )
        .expect_err("the same key twice");
        assert_eq!(error.code(), Some("XTDE3365"));
    }
}

//! Foreign content: MathML and SVG inside HTML
//! ([HTML § 13.2.6.5](https://html.spec.whatwg.org/multipage/parsing.html#parsing-main-inforeign)).
//!
//! HTML tokenizes case-insensitively and namespace-blind, but MathML and SVG
//! are XML vocabularies with camel-case names and real namespaces. The three
//! adjustment tables below undo the damage, and the "in foreign content" rules
//! decide when a tag breaks out of the foreign subtree back into HTML.

use super::insert::AdjustedAttribute;
use super::{Action, TreeBuilder};
use crate::error::codes;
use crate::ns::{Namespace, XLINK_NAMESPACE, XML_NAMESPACE, XMLNS_NAMESPACE};
use crate::token::{Attribute, Tag, Token};

/// The SVG element names whose spelling HTML's lower-casing destroys.
const SVG_TAG_NAMES: &[(&str, &str)] = &[
    ("altglyph", "altGlyph"),
    ("altglyphdef", "altGlyphDef"),
    ("altglyphitem", "altGlyphItem"),
    ("animatecolor", "animateColor"),
    ("animatemotion", "animateMotion"),
    ("animatetransform", "animateTransform"),
    ("clippath", "clipPath"),
    ("feblend", "feBlend"),
    ("fecolormatrix", "feColorMatrix"),
    ("fecomponenttransfer", "feComponentTransfer"),
    ("fecomposite", "feComposite"),
    ("feconvolvematrix", "feConvolveMatrix"),
    ("fediffuselighting", "feDiffuseLighting"),
    ("fedisplacementmap", "feDisplacementMap"),
    ("fedistantlight", "feDistantLight"),
    ("fedropshadow", "feDropShadow"),
    ("feflood", "feFlood"),
    ("fefunca", "feFuncA"),
    ("fefuncb", "feFuncB"),
    ("fefuncg", "feFuncG"),
    ("fefuncr", "feFuncR"),
    ("fegaussianblur", "feGaussianBlur"),
    ("feimage", "feImage"),
    ("femerge", "feMerge"),
    ("femergenode", "feMergeNode"),
    ("femorphology", "feMorphology"),
    ("feoffset", "feOffset"),
    ("fepointlight", "fePointLight"),
    ("fespecularlighting", "feSpecularLighting"),
    ("fespotlight", "feSpotLight"),
    ("fetile", "feTile"),
    ("feturbulence", "feTurbulence"),
    ("foreignobject", "foreignObject"),
    ("glyphref", "glyphRef"),
    ("lineargradient", "linearGradient"),
    ("radialgradient", "radialGradient"),
    ("textpath", "textPath"),
];

/// The SVG attribute names whose spelling HTML's lower-casing destroys.
const SVG_ATTRIBUTE_NAMES: &[(&str, &str)] = &[
    ("attributename", "attributeName"),
    ("attributetype", "attributeType"),
    ("basefrequency", "baseFrequency"),
    ("baseprofile", "baseProfile"),
    ("calcmode", "calcMode"),
    ("clippathunits", "clipPathUnits"),
    ("diffuseconstant", "diffuseConstant"),
    ("edgemode", "edgeMode"),
    ("filterunits", "filterUnits"),
    ("glyphref", "glyphRef"),
    ("gradienttransform", "gradientTransform"),
    ("gradientunits", "gradientUnits"),
    ("kernelmatrix", "kernelMatrix"),
    ("kernelunitlength", "kernelUnitLength"),
    ("keypoints", "keyPoints"),
    ("keysplines", "keySplines"),
    ("keytimes", "keyTimes"),
    ("lengthadjust", "lengthAdjust"),
    ("limitingconeangle", "limitingConeAngle"),
    ("markerheight", "markerHeight"),
    ("markerunits", "markerUnits"),
    ("markerwidth", "markerWidth"),
    ("maskcontentunits", "maskContentUnits"),
    ("maskunits", "maskUnits"),
    ("numoctaves", "numOctaves"),
    ("pathlength", "pathLength"),
    ("patterncontentunits", "patternContentUnits"),
    ("patterntransform", "patternTransform"),
    ("patternunits", "patternUnits"),
    ("pointsatx", "pointsAtX"),
    ("pointsaty", "pointsAtY"),
    ("pointsatz", "pointsAtZ"),
    ("preservealpha", "preserveAlpha"),
    ("preserveaspectratio", "preserveAspectRatio"),
    ("primitiveunits", "primitiveUnits"),
    ("refx", "refX"),
    ("refy", "refY"),
    ("repeatcount", "repeatCount"),
    ("repeatdur", "repeatDur"),
    ("requiredextensions", "requiredExtensions"),
    ("requiredfeatures", "requiredFeatures"),
    ("specularconstant", "specularConstant"),
    ("specularexponent", "specularExponent"),
    ("spreadmethod", "spreadMethod"),
    ("startoffset", "startOffset"),
    ("stddeviation", "stdDeviation"),
    ("stitchtiles", "stitchTiles"),
    ("surfacescale", "surfaceScale"),
    ("systemlanguage", "systemLanguage"),
    ("tablevalues", "tableValues"),
    ("targetx", "targetX"),
    ("targety", "targetY"),
    ("textlength", "textLength"),
    ("viewbox", "viewBox"),
    ("viewtarget", "viewTarget"),
    ("xchannelselector", "xChannelSelector"),
    ("ychannelselector", "yChannelSelector"),
    ("zoomandpan", "zoomAndPan"),
];

/// The attributes that "adjust foreign attributes" moves into a namespace.
const FOREIGN_ATTRIBUTES: &[(&str, &str, &str)] = &[
    ("xlink:actuate", "xlink:actuate", XLINK_NAMESPACE),
    ("xlink:arcrole", "xlink:arcrole", XLINK_NAMESPACE),
    ("xlink:href", "xlink:href", XLINK_NAMESPACE),
    ("xlink:role", "xlink:role", XLINK_NAMESPACE),
    ("xlink:show", "xlink:show", XLINK_NAMESPACE),
    ("xlink:title", "xlink:title", XLINK_NAMESPACE),
    ("xlink:type", "xlink:type", XLINK_NAMESPACE),
    ("xml:lang", "xml:lang", XML_NAMESPACE),
    ("xml:space", "xml:space", XML_NAMESPACE),
    ("xmlns", "xmlns", XMLNS_NAMESPACE),
    ("xmlns:xlink", "xmlns:xlink", XMLNS_NAMESPACE),
];

/// The HTML start tags that break out of foreign content.
const BREAKOUT_TAGS: &[&str] = &[
    "b",
    "big",
    "blockquote",
    "body",
    "br",
    "center",
    "code",
    "dd",
    "div",
    "dl",
    "dt",
    "em",
    "embed",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "head",
    "hr",
    "i",
    "img",
    "li",
    "listing",
    "menu",
    "meta",
    "nobr",
    "ol",
    "p",
    "pre",
    "ruby",
    "s",
    "small",
    "span",
    "strong",
    "strike",
    "sub",
    "sup",
    "table",
    "tt",
    "u",
    "ul",
    "var",
];

/// "Adjust SVG tag names": restores the camel case of an SVG element name.
#[must_use]
pub(crate) fn adjust_svg_tag_name(name: &str) -> &str {
    SVG_TAG_NAMES
        .iter()
        .find(|(lower, _)| *lower == name)
        .map_or(name, |(_, proper)| *proper)
}

/// Applies the MathML, SVG and foreign attribute adjustments to a token's
/// attributes.
pub(crate) fn adjust_attributes(
    attributes: &[Attribute],
    namespace: Namespace,
) -> Vec<AdjustedAttribute> {
    attributes
        .iter()
        .map(|attribute| {
            let name = match namespace {
                Namespace::MathMl if attribute.name == "definitionurl" => "definitionURL",
                Namespace::Svg => SVG_ATTRIBUTE_NAMES
                    .iter()
                    .find(|(lower, _)| *lower == attribute.name)
                    .map_or(attribute.name.as_str(), |(_, proper)| *proper),
                _ => attribute.name.as_str(),
            };
            match FOREIGN_ATTRIBUTES
                .iter()
                .find(|(source, _, _)| *source == name)
            {
                Some((_, qualified, namespace)) => AdjustedAttribute {
                    qualified_name: (*qualified).to_owned(),
                    namespace: Some(namespace),
                    value: attribute.value.clone(),
                },
                None => AdjustedAttribute {
                    qualified_name: name.to_owned(),
                    namespace: None,
                    value: attribute.value.clone(),
                },
            }
        })
        .collect()
}

impl TreeBuilder<'_> {
    /// The "in foreign content" rules.
    pub(crate) fn foreign_content(&mut self, token: Token) -> Action {
        match token {
            Token::Characters(text) => {
                for ch in text.chars() {
                    if ch == '\0' {
                        self.error(codes::UNEXPECTED_NULL_CHARACTER);
                        self.insert_characters("\u{fffd}");
                    } else {
                        if !matches!(ch, '\t' | '\n' | '\u{c}' | '\r' | ' ') {
                            self.frameset_ok = false;
                        }
                        let mut buffer = [0_u8; 4];
                        let text = ch.encode_utf8(&mut buffer).to_owned();
                        self.insert_characters(&text);
                    }
                }
                Action::Done
            }
            Token::Comment(_) | Token::ProcessingInstruction(_) => {
                self.insert_markup(&token, None);
                Action::Done
            }
            Token::Doctype(_) => {
                self.error(codes::UNEXPECTED_DOCTYPE);
                Action::Done
            }
            Token::StartTag(tag) => self.foreign_start_tag(tag),
            Token::EndTag(tag) => self.foreign_end_tag(tag),
            Token::Eof => Action::Done,
        }
    }

    fn foreign_start_tag(&mut self, tag: Tag) -> Action {
        let breaks_out = BREAKOUT_TAGS.contains(&tag.name.as_str())
            || (tag.name == "font"
                && (tag.has_attribute("color")
                    || tag.has_attribute("face")
                    || tag.has_attribute("size")));
        if breaks_out {
            self.error(codes::UNEXPECTED_START_TAG);
            self.pop_back_to_html_content();
            return Action::ReprocessInHtml(Token::StartTag(tag));
        }

        let namespace = self
            .adjusted_current_node()
            .map_or(Namespace::Html, |node| node.namespace);
        let mut tag = tag;
        if namespace == Namespace::Svg {
            tag.name = adjust_svg_tag_name(&tag.name).to_owned();
        }
        let attributes = adjust_attributes(&tag.attributes, namespace);
        self.insert_element(&tag, namespace, attributes);
        if tag.self_closing {
            self.open.pop();
        }
        Action::Done
    }

    /// "While the current node is not a MathML text integration point, an HTML
    /// integration point, or an element in the HTML namespace, pop elements
    /// from the stack of open elements."
    fn pop_back_to_html_content(&mut self) {
        loop {
            let Some(node) = self.current_node().cloned() else {
                return;
            };
            if node.namespace == Namespace::Html
                || (node.namespace == Namespace::MathMl
                    && crate::ns::is_mathml_text_integration_point(&node.name))
                || self.is_html_integration_point(&node)
            {
                return;
            }
            self.open.pop();
        }
    }

    fn foreign_end_tag(&mut self, tag: Tag) -> Action {
        if tag.name == "br" || tag.name == "p" {
            self.error(codes::UNEXPECTED_END_TAG);
            self.pop_back_to_html_content();
            return Action::ReprocessInHtml(Token::EndTag(tag));
        }
        if tag.name == "script"
            && self
                .current_node()
                .is_some_and(|node| node.namespace == Namespace::Svg && node.name == "script")
        {
            self.open.pop();
            return Action::Done;
        }

        let Some(mut index) = self.open.len().checked_sub(1) else {
            return Action::Done;
        };
        if !self.open[index].name.eq_ignore_ascii_case(&tag.name) {
            self.error(codes::UNEXPECTED_END_TAG);
        }
        loop {
            if index == 0 {
                return Action::Done;
            }
            if self.open[index].name.eq_ignore_ascii_case(&tag.name) {
                let id = self.open[index].id;
                self.pop_until_id(id);
                return Action::Done;
            }
            index -= 1;
            if self.open[index].namespace == Namespace::Html {
                return self.dispatch(self.mode, Token::EndTag(tag));
            }
        }
    }
}

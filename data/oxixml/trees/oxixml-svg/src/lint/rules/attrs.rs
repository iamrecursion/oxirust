//! Per-attribute checks: name resolution (mapping an authored, possibly
//! namespaced attribute to an [`AttrId`]), applicability, deprecation, the
//! `style` attribute, and the value-grammar dispatcher
//! [`validate_attr_value`] shared by plain attributes and `style`
//! declarations alike.

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use oxixml_dom::{Document, NodeId};

use crate::error::ValueError;
use crate::lint::diagnostic::{Diagnostic, LintOptions, Severity, codes};
use crate::names::attrs::AttrId;
use crate::names::content::attribute_applies;
use crate::names::elements::ElementId;
use crate::ns::{XLINK_NAMESPACE, XML_NAMESPACE, XMLNS_NAMESPACE};
use crate::style::attr::parse_style_attribute_strict;
use crate::style::property::PropertyId;
use crate::values::aspect::{PreserveAspectRatio, ViewBox};
use crate::values::clock::{ClockDuration, RepeatCount, parse_repeat_dur, parse_timing_list};
use crate::values::color::Color;
use crate::values::iri::{FuncIri, local_reference};
use crate::values::length::{Length, parse_dash_array, parse_number_optional_number};
use crate::values::lex::parse_number;
use crate::values::paint::{Paint, parse_opacity};
use crate::values::path::PathData;
use crate::values::points::parse_points_strict;
use crate::values::transform::TransformList;

/// One `url(#x)`/`href="#x"`-shaped reference found while walking the
/// document, recorded for the document-level dangling-reference pass in
/// `super::refs`.
pub(super) struct RefSite {
    pub(super) node: NodeId,
    pub(super) reference: Box<str>,
}

/// How an authored attribute name maps onto this crate's [`AttrId`]
/// catalog.
enum AttrResolution {
    /// An `xmlns`/`xmlns:*` declaration, a `data-*`/`aria-*` custom
    /// attribute, or an attribute in a namespace this crate does not
    /// model at all: never reported, by design (see the
    /// [`crate::lint::rules`] module documentation).
    Skip,
    /// No [`AttrId`] recognizes this name.
    Unknown,
    /// A recognized attribute.
    Known(AttrId),
}

/// Resolves an authored attribute's namespace and local name to an
/// [`AttrResolution`].
///
/// `xlink:*` and `xml:*` names are matched against the small fixed sets
/// [`AttrId::from_name`] itself expects spelled with their literal prefix
/// (`"xlink:href"`, `"xml:lang"`); this function bridges a
/// namespace-resolved `(namespace, local)` pair (what
/// [`oxixml_dom::node::AttrData`] exposes) to that prefixed spelling
/// without allocating.
///
/// `xml:id` is deliberately **not** specially handled: [`AttrId`] does not
/// model it (only `xml:lang`/`xml:space`/`xml:base` are catalog/mission
/// entries), so it falls through to [`AttrResolution::Unknown`] like any
/// other name this crate's catalog does not cover — an honest, if mild,
/// finding rather than a silent skip, since unlike `data-*`/`aria-*` it is
/// not an open-ended, intentionally-uncataloged prefix.
fn resolve_attr_id(namespace: Option<&str>, local: &str) -> AttrResolution {
    if namespace == Some(XMLNS_NAMESPACE) {
        return AttrResolution::Skip;
    }
    if namespace == Some(XLINK_NAMESPACE) {
        return match local {
            "href" => AttrResolution::Known(AttrId::XlinkHref),
            "show" => AttrResolution::Known(AttrId::XlinkShow),
            "actuate" => AttrResolution::Known(AttrId::XlinkActuate),
            "type" => AttrResolution::Known(AttrId::XlinkType),
            "title" => AttrResolution::Known(AttrId::XlinkTitle),
            "arcrole" => AttrResolution::Known(AttrId::XlinkArcrole),
            "role" => AttrResolution::Known(AttrId::XlinkRole),
            _ => AttrResolution::Unknown,
        };
    }
    if namespace == Some(XML_NAMESPACE) {
        return match local {
            "lang" => AttrResolution::Known(AttrId::XmlLang),
            "space" => AttrResolution::Known(AttrId::XmlSpace),
            "base" => AttrResolution::Known(AttrId::XmlBase),
            _ => AttrResolution::Unknown,
        };
    }
    if namespace.is_some() {
        // A foreign namespace this crate does not model at all (neither
        // XLink nor XML): out of scope, silently skipped.
        return AttrResolution::Skip;
    }
    if local.starts_with("data-") || local.starts_with("aria-") {
        return AttrResolution::Skip;
    }
    match AttrId::from_name(local) {
        Some(id) => AttrResolution::Known(id),
        None => AttrResolution::Unknown,
    }
}

/// Runs every attribute-level check for one authored attribute node:
/// unknown-name, applicability, deprecation, value grammar (including the
/// `style` attribute's own declarations), and reference-site collection.
///
/// Also updates the shared `id_table` (for [`AttrId::Id`]) and
/// `seen_href`/`seen_xlink_href` (for the caller's
/// [`codes::HREF_DUPLICATED`] check, which needs to see every attribute on
/// the element before it can fire).
#[allow(clippy::too_many_arguments)]
pub(super) fn process_attribute(
    doc: &Document,
    element: NodeId,
    kind: ElementId,
    attr_node: NodeId,
    options: &LintOptions,
    diagnostics: &mut Vec<Diagnostic>,
    id_table: &mut HashMap<Box<str>, NodeId>,
    ref_sites: &mut Vec<RefSite>,
    seen_href: &mut bool,
    seen_xlink_href: &mut bool,
) {
    let Some(data) = doc.attr(attr_node) else {
        return;
    };
    let namespace = data.namespace_uri();
    let local = data.local_name();
    let value = data.value();

    match resolve_attr_id(namespace, local) {
        AttrResolution::Skip => {}
        AttrResolution::Unknown => {
            if options.checks_unknown() {
                diagnostics.push(Diagnostic::warning(
                    codes::UNKNOWN_ATTRIBUTE,
                    format!("unknown attribute `{}`", data.qualified_name()),
                    element,
                ));
            }
        }
        AttrResolution::Known(attr) => {
            if attr == AttrId::Id {
                note_id(value, element, id_table, diagnostics);
            }
            if attr == AttrId::Href {
                *seen_href = true;
            }
            if attr == AttrId::XlinkHref {
                *seen_xlink_href = true;
            }

            if options.checks_deprecated() && attr.deprecated_in_svg2() {
                diagnostics.push(Diagnostic::info(
                    codes::DEPRECATED_ATTRIBUTE,
                    format!("`{}` is deprecated in SVG 2", data.qualified_name()),
                    element,
                ));
            }

            if options.checks_unknown() && !attribute_applies(kind, attr) {
                diagnostics.push(Diagnostic::warning(
                    codes::ATTRIBUTE_NOT_APPLICABLE,
                    format!(
                        "`{}` does not apply to `<{}>`",
                        data.qualified_name(),
                        kind.as_str()
                    ),
                    element,
                ));
            }

            if options.checks_values() {
                report_value_check(validate_attr_value(attr, kind, value), diagnostics, element);
            }
            // The `style` attribute's own declarations feed both the value-grammar
            // pass (each declaration's value) and the reference-collection pass
            // (each declaration's `url(#x)`), so it runs whenever either is enabled.
            if attr == AttrId::Style && (options.checks_values() || options.checks_references()) {
                check_style_attribute(value, kind, options, diagnostics, element, ref_sites);
            }

            if options.checks_references() {
                collect_reference_sites(attr, kind, value, element, ref_sites);
            }
        }
    }
}

/// Records `value` as `node`'s `id` in `id_table`, reporting
/// [`codes::DUPLICATE_ID`] when an earlier node already claimed the same
/// value. An empty `id=""` is not recorded (it cannot be a valid XML
/// `Name`, and is not a meaningful reference target either way).
///
/// Shared between [`process_attribute`] (the common case) and
/// `super::Walker::note_id_if_present` (used for elements whose deeper
/// attribute processing is otherwise skipped because their name is
/// unknown).
pub(super) fn note_id(
    value: &str,
    node: NodeId,
    id_table: &mut HashMap<Box<str>, NodeId>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    if value.is_empty() {
        return;
    }
    match id_table.entry(Box::from(value)) {
        Entry::Vacant(entry) => {
            entry.insert(node);
        }
        Entry::Occupied(_) => {
            diagnostics.push(Diagnostic::error(
                codes::DUPLICATE_ID,
                format!("duplicate id `{value}`"),
                node,
            ));
        }
    }
}

fn report_value_check(check: ValueCheck, diagnostics: &mut Vec<Diagnostic>, node: NodeId) {
    if let ValueCheck::Invalid { code, message } = check {
        let severity = if code == codes::NEGATIVE_DIMENSION {
            Severity::Error
        } else {
            Severity::Warning
        };
        diagnostics.push(Diagnostic::new(code, severity, message, node));
    }
}

/// The outcome of validating one attribute or style-declaration value
/// against its grammar.
#[derive(Debug, PartialEq, Eq)]
enum ValueCheck {
    /// No grammar is implemented for this name (see the
    /// [`crate::lint::rules`] module documentation's scope-cut list), or a
    /// keyword specific to this attribute/host short-circuited the check
    /// (a CSS-wide keyword, `none`, `freeze`/`remove`, ...). Nothing to
    /// report.
    Skipped,
    /// The value parsed successfully.
    Valid,
    /// The value is malformed; report it under `code`.
    Invalid { code: &'static str, message: String },
}

/// Validates one attribute value against the grammar this crate
/// implements for it, if any.
///
/// `host` disambiguates the small number of attributes whose meaning (and
/// therefore grammar) depends on which element carries them: `fill`
/// (paint on shapes/text vs. freeze/remove on animation elements, the
/// catalog's own headline example of the "one name, two attributes"
/// pattern — see [`crate::names::attrs`]'s module documentation), `offset`
/// (a gradient `stop`'s position vs. a transfer function's additive
/// constant), and the enumerated-keyword `operator` (`feComposite` vs.
/// `feMorphology`), `type` (`animateTransform` vs. `feColorMatrix` vs. the
/// `feFunc*` transfer functions vs. `feTurbulence`, each with a different
/// keyword set) and `mode` (`feBlend` only).
///
/// A CSS-wide keyword (`inherit`/`initial`/`unset`) is always accepted for
/// any attribute [`AttrId::is_presentation`] reports `true` for, checked
/// before grammar dispatch, per the specification's cascade rules — this
/// intentionally does not further condition on `host`, so (for instance)
/// `fill="inherit"` on an `<animate>` element is accepted too, rather than
/// being run through the freeze/remove check; this is a narrow, harmless
/// simplification (SMIL timing attributes do not really participate in
/// the CSS cascade) rather than a considered design choice.
///
/// See the [`crate::lint::rules`] module documentation for the (short) list
/// of attributes with **no** implemented grammar, which fall through to
/// [`ValueCheck::Skipped`] via the final wildcard arm below.
fn validate_attr_value(attr: AttrId, host: ElementId, value: &str) -> ValueCheck {
    let trimmed = value.trim();
    if attr.is_presentation() && is_css_wide_keyword(trimmed) {
        return ValueCheck::Valid;
    }

    match attr {
        AttrId::D => from_value_error(PathData::parse(value).map(|_| ()), "d"),
        AttrId::Points => from_value_error(parse_points_strict(value).map(|_| ()), "points"),
        AttrId::Transform | AttrId::GradientTransform | AttrId::PatternTransform => {
            from_value_error(TransformList::parse(value).map(|_| ()), attr.as_str())
        }
        AttrId::ViewBox => match ViewBox::parse(value) {
            Ok(_) => ValueCheck::Valid,
            Err(error @ ValueError::OutOfRange { .. }) => ValueCheck::Invalid {
                code: codes::INVALID_VIEW_BOX,
                message: format!("invalid `viewBox`: {error}"),
            },
            Err(error) => ValueCheck::Invalid {
                code: codes::INVALID_ATTRIBUTE_VALUE,
                message: format!("invalid `viewBox`: {error}"),
            },
        },
        AttrId::PreserveAspectRatio => from_value_error(
            PreserveAspectRatio::parse(value).map(|_| ()),
            "preserveAspectRatio",
        ),
        AttrId::X
        | AttrId::Y
        | AttrId::Cx
        | AttrId::Cy
        | AttrId::R
        | AttrId::Rx
        | AttrId::Ry
        | AttrId::Width
        | AttrId::Height
        | AttrId::X1
        | AttrId::Y1
        | AttrId::X2
        | AttrId::Y2
        | AttrId::Dx
        | AttrId::Dy
        | AttrId::Fx
        | AttrId::Fy
        | AttrId::Fr
        | AttrId::TextLength
        | AttrId::FontSize
        | AttrId::StrokeWidth
        | AttrId::StrokeDashoffset
        | AttrId::MarkerWidth
        | AttrId::MarkerHeight
        | AttrId::RefX
        | AttrId::RefY => validate_length(attr, value),
        AttrId::StrokeDasharray => {
            if trimmed == "none" {
                ValueCheck::Valid
            } else {
                from_value_error(parse_dash_array(value).map(|_| ()), "stroke-dasharray")
            }
        }
        AttrId::Fill => {
            if host.is_animation() {
                validate_freeze_remove(trimmed)
            } else {
                from_value_error(Paint::parse(value).map(|_| ()), "fill")
            }
        }
        AttrId::Stroke => {
            if host.is_animation() {
                // `stroke` has no timing-attribute meaning; on an animation
                // element it is already reported as ATTRIBUTE_NOT_APPLICABLE,
                // so it is not additionally paint-parsed here.
                ValueCheck::Skipped
            } else {
                from_value_error(Paint::parse(value).map(|_| ()), "stroke")
            }
        }
        AttrId::Color | AttrId::StopColor | AttrId::FloodColor | AttrId::LightingColor => {
            if trimmed.eq_ignore_ascii_case("currentcolor") {
                ValueCheck::Valid
            } else {
                from_value_error(Color::parse(value).map(|_| ()), attr.as_str())
            }
        }
        AttrId::Opacity
        | AttrId::FillOpacity
        | AttrId::StrokeOpacity
        | AttrId::StopOpacity
        | AttrId::FloodOpacity => from_value_error(parse_opacity(value).map(|_| ()), attr.as_str()),
        AttrId::Dur => from_value_error(ClockDuration::parse(value).map(|_| ()), "dur"),
        AttrId::RepeatDur => from_value_error(parse_repeat_dur(value).map(|_| ()), "repeatDur"),
        AttrId::Begin | AttrId::End => {
            from_value_error(parse_timing_list(value).map(|_| ()), attr.as_str())
        }
        AttrId::RepeatCount => {
            from_value_error(RepeatCount::parse(value).map(|_| ()), "repeatCount")
        }
        AttrId::Href | AttrId::XlinkHref => ValueCheck::Skipped,
        AttrId::StdDeviation
        | AttrId::BaseFrequency
        | AttrId::Order
        | AttrId::Radius
        | AttrId::KernelUnitLength => from_value_error(
            parse_number_optional_number(value).map(|_| ()),
            attr.as_str(),
        ),
        AttrId::Offset => match host {
            ElementId::Stop => from_value_error(parse_opacity(value).map(|_| ()), "offset"),
            ElementId::FeFuncR | ElementId::FeFuncG | ElementId::FeFuncB | ElementId::FeFuncA => {
                from_value_error(parse_number(value).map(|_| ()), "offset")
            }
            _ => ValueCheck::Skipped,
        },
        AttrId::StrokeMiterlimit => match parse_number(value) {
            Ok(n) if n >= 1.0 => ValueCheck::Valid,
            Ok(_) => ValueCheck::Invalid {
                code: codes::INVALID_ATTRIBUTE_VALUE,
                message: format!("`stroke-miterlimit` must be >= 1, found {trimmed:?}"),
            },
            Err(error) => ValueCheck::Invalid {
                code: codes::INVALID_ATTRIBUTE_VALUE,
                message: format!("invalid `stroke-miterlimit` value: {error}"),
            },
        },

        // The purely-enumerated-keyword attributes: a small, fixed,
        // specification-defined vocabulary with no further internal grammar.
        // Grouped by shared vocabulary; every one of these attribute names
        // has exactly one host-independent set of keywords except
        // `operator`, `type` and `mode`, which are overloaded per host
        // element (dispatched the same way `AttrId::Offset` is above).
        AttrId::GradientUnits
        | AttrId::PatternUnits
        | AttrId::PatternContentUnits
        | AttrId::MaskUnits
        | AttrId::MaskContentUnits
        | AttrId::ClipPathUnits
        | AttrId::FilterUnits
        | AttrId::PrimitiveUnits => {
            validate_enum(attr.as_str(), OBJECT_BOUNDING_BOX_UNITS, trimmed)
        }
        AttrId::MarkerUnits => validate_enum(attr.as_str(), MARKER_UNITS, trimmed),
        AttrId::SpreadMethod => validate_enum(attr.as_str(), SPREAD_METHOD, trimmed),
        AttrId::CalcMode => validate_enum(attr.as_str(), CALC_MODE, trimmed),
        AttrId::Restart => validate_enum(attr.as_str(), RESTART, trimmed),
        AttrId::Additive => validate_enum(attr.as_str(), ADDITIVE, trimmed),
        AttrId::Accumulate => validate_enum(attr.as_str(), ACCUMULATE, trimmed),
        AttrId::EdgeMode => validate_enum(attr.as_str(), EDGE_MODE, trimmed),
        AttrId::AttributeType => validate_enum(attr.as_str(), ATTRIBUTE_TYPE, trimmed),
        AttrId::ZoomAndPan => validate_enum(attr.as_str(), ZOOM_AND_PAN, trimmed),
        AttrId::LengthAdjust => validate_enum(attr.as_str(), LENGTH_ADJUST, trimmed),
        AttrId::Method => validate_enum(attr.as_str(), TEXT_PATH_METHOD, trimmed),
        AttrId::Spacing => validate_enum(attr.as_str(), TEXT_PATH_SPACING, trimmed),
        AttrId::Side => validate_enum(attr.as_str(), TEXT_PATH_SIDE, trimmed),
        AttrId::StitchTiles => validate_enum(attr.as_str(), STITCH_TILES, trimmed),
        AttrId::XChannelSelector => validate_enum(attr.as_str(), CHANNEL_SELECTOR, trimmed),
        AttrId::YChannelSelector => validate_enum(attr.as_str(), CHANNEL_SELECTOR, trimmed),
        AttrId::Operator => match host {
            ElementId::FeComposite => validate_enum(attr.as_str(), FE_COMPOSITE_OPERATOR, trimmed),
            ElementId::FeMorphology => {
                validate_enum(attr.as_str(), FE_MORPHOLOGY_OPERATOR, trimmed)
            }
            _ => ValueCheck::Skipped,
        },
        AttrId::Type => match host {
            ElementId::AnimateTransform => {
                validate_enum(attr.as_str(), ANIMATE_TRANSFORM_TYPE, trimmed)
            }
            ElementId::FeColorMatrix => validate_enum(attr.as_str(), FE_COLOR_MATRIX_TYPE, trimmed),
            ElementId::FeFuncR | ElementId::FeFuncG | ElementId::FeFuncB | ElementId::FeFuncA => {
                validate_enum(attr.as_str(), FE_TRANSFER_FUNCTION_TYPE, trimmed)
            }
            ElementId::FeTurbulence => validate_enum(attr.as_str(), FE_TURBULENCE_TYPE, trimmed),
            // `script`'s and `style`'s `type` is a MIME type string, and `a`'s
            // is a MIME type hint for the linked resource — neither is a
            // small enumerated vocabulary, so both stay unchecked.
            _ => ValueCheck::Skipped,
        },
        AttrId::Mode => match host {
            ElementId::FeBlend => validate_enum(attr.as_str(), FE_BLEND_MODE, trimmed),
            _ => ValueCheck::Skipped,
        },

        _ => ValueCheck::Skipped,
    }
}

/// Validates that `trimmed` is exactly one of `allowed`, case-sensitively —
/// the shape shared by every purely-enumerated-keyword SVG attribute.
fn validate_enum(name: &str, allowed: &[&str], trimmed: &str) -> ValueCheck {
    if allowed.contains(&trimmed) {
        ValueCheck::Valid
    } else {
        ValueCheck::Invalid {
            code: codes::INVALID_ATTRIBUTE_VALUE,
            message: format!(
                "`{name}` must be one of {}, found {trimmed:?}",
                allowed.join("/")
            ),
        }
    }
}

/// `gradientUnits`, `patternUnits`, `patternContentUnits`, `maskUnits`,
/// `maskContentUnits`, `clipPathUnits`, `filterUnits` and `primitiveUnits`
/// all share this exact two-keyword vocabulary (SVG 1.1 §7.11 and its clones
/// on `pattern`, `mask`, `clipPath` and `filter`) — see also
/// [`crate::writer::paint_defs::CoordinateUnits`], the same vocabulary
/// modeled as a builder-side enum.
const OBJECT_BOUNDING_BOX_UNITS: &[&str] = &["userSpaceOnUse", "objectBoundingBox"];
/// `marker`'s `markerUnits`.
const MARKER_UNITS: &[&str] = &["strokeWidth", "userSpaceOnUse"];
/// A gradient's `spreadMethod`.
const SPREAD_METHOD: &[&str] = &["pad", "reflect", "repeat"];
/// An animation's `calcMode`.
const CALC_MODE: &[&str] = &["discrete", "linear", "paced", "spline"];
/// An animation's `restart`.
const RESTART: &[&str] = &["always", "whenNotActive", "never"];
/// An animation's `additive`.
const ADDITIVE: &[&str] = &["replace", "sum"];
/// An animation's `accumulate`.
const ACCUMULATE: &[&str] = &["none", "sum"];
/// `feConvolveMatrix`'s and `feGaussianBlur`'s `edgeMode`.
const EDGE_MODE: &[&str] = &["duplicate", "wrap", "none"];
/// An animation's `attributeType`. Case-sensitive exactly as the grammar
/// spells it — `"css"`/`"xml"` are not equivalent to `"CSS"`/`"XML"`.
const ATTRIBUTE_TYPE: &[&str] = &["CSS", "XML", "auto"];
/// `svg`'s and `view`'s `zoomAndPan`.
const ZOOM_AND_PAN: &[&str] = &["disable", "magnify"];
/// `text`/`tspan`/`textPath`/`altGlyph`'s `lengthAdjust`.
const LENGTH_ADJUST: &[&str] = &["spacing", "spacingAndGlyphs"];
/// `textPath`'s `method`.
const TEXT_PATH_METHOD: &[&str] = &["align", "stretch"];
/// `textPath`'s `spacing`.
const TEXT_PATH_SPACING: &[&str] = &["auto", "exact"];
/// `textPath`'s `side` (SVG 2).
const TEXT_PATH_SIDE: &[&str] = &["left", "right"];
/// `feTurbulence`'s `stitchTiles`.
const STITCH_TILES: &[&str] = &["stitch", "noStitch"];
/// `feDisplacementMap`'s `xChannelSelector`/`yChannelSelector`.
const CHANNEL_SELECTOR: &[&str] = &["R", "G", "B", "A"];
/// `feComposite`'s `operator`.
const FE_COMPOSITE_OPERATOR: &[&str] =
    &["over", "in", "out", "atop", "xor", "lighter", "arithmetic"];
/// `feMorphology`'s `operator`.
const FE_MORPHOLOGY_OPERATOR: &[&str] = &["erode", "dilate"];
/// `animateTransform`'s `type`.
const ANIMATE_TRANSFORM_TYPE: &[&str] = &["translate", "scale", "rotate", "skewX", "skewY"];
/// `feColorMatrix`'s `type`.
const FE_COLOR_MATRIX_TYPE: &[&str] = &["matrix", "saturate", "hueRotate", "luminanceToAlpha"];
/// `feFuncR`/`feFuncG`/`feFuncB`/`feFuncA`'s `type`.
const FE_TRANSFER_FUNCTION_TYPE: &[&str] = &["identity", "table", "discrete", "linear", "gamma"];
/// `feTurbulence`'s `type`.
const FE_TURBULENCE_TYPE: &[&str] = &["fractalNoise", "turbulence"];
/// `feBlend`'s `mode`: the full CSS Compositing and Blending Level 1
/// `<blend-mode>` production (SVG 2), a superset of SVG 1.1's original
/// 5-keyword `feBlend` vocabulary (`normal`/`multiply`/`screen`/`darken`/
/// `lighten`), so accepting it covers both.
const FE_BLEND_MODE: &[&str] = &[
    "normal",
    "multiply",
    "screen",
    "overlay",
    "darken",
    "lighten",
    "color-dodge",
    "color-burn",
    "hard-light",
    "soft-light",
    "difference",
    "exclusion",
    "hue",
    "saturation",
    "color",
    "luminosity",
];

fn is_css_wide_keyword(trimmed: &str) -> bool {
    trimmed.eq_ignore_ascii_case("inherit")
        || trimmed.eq_ignore_ascii_case("initial")
        || trimmed.eq_ignore_ascii_case("unset")
}

fn validate_freeze_remove(trimmed: &str) -> ValueCheck {
    if trimmed == "freeze" || trimmed == "remove" {
        ValueCheck::Valid
    } else {
        ValueCheck::Invalid {
            code: codes::INVALID_ATTRIBUTE_VALUE,
            message: format!(
                "`fill` on an animation element must be `freeze` or `remove`, found {trimmed:?}"
            ),
        }
    }
}

/// Turns a `Result<(), ValueError>` from a frozen parser into a
/// [`ValueCheck`], reporting any error as [`codes::INVALID_ATTRIBUTE_VALUE`]
/// with a message that includes the [`ValueError`]'s [`std::fmt::Display`]
/// text (which carries a byte offset into the value).
fn from_value_error(result: Result<(), ValueError>, name: &str) -> ValueCheck {
    match result {
        Ok(()) => ValueCheck::Valid,
        Err(error) => ValueCheck::Invalid {
            code: codes::INVALID_ATTRIBUTE_VALUE,
            message: format!("invalid `{name}` value: {error}"),
        },
    }
}

/// The attributes the specification defines as accepting no negative
/// magnitude: `width`, `height`, `r`, `rx`, `ry`. Checked only after
/// [`Length::parse`] already succeeded.
const NON_NEGATIVE_LENGTH_ATTRS: &[AttrId] = &[
    AttrId::Width,
    AttrId::Height,
    AttrId::R,
    AttrId::Rx,
    AttrId::Ry,
];

/// The attributes SVG 2 additionally allows the bare keyword `auto` on
/// (meaning "derive from context"), checked before [`Length::parse`] is
/// attempted at all.
const AUTO_KEYWORD_LENGTH_ATTRS: &[AttrId] =
    &[AttrId::Width, AttrId::Height, AttrId::Rx, AttrId::Ry];

/// Validates a `<length>`-grammar attribute: SVG 2's `auto` keyword for
/// the attributes that accept it, then [`Length::parse`], then a
/// non-negative check for the attributes that require it.
fn validate_length(attr: AttrId, value: &str) -> ValueCheck {
    if AUTO_KEYWORD_LENGTH_ATTRS.contains(&attr) && value.trim().eq_ignore_ascii_case("auto") {
        return ValueCheck::Valid;
    }
    match Length::parse(value) {
        Ok(length) if NON_NEGATIVE_LENGTH_ATTRS.contains(&attr) && length.value < 0.0 => {
            ValueCheck::Invalid {
                code: codes::NEGATIVE_DIMENSION,
                message: format!("`{}` must not be negative, found {value:?}", attr.as_str()),
            }
        }
        Ok(_) => ValueCheck::Valid,
        Err(error) => ValueCheck::Invalid {
            code: codes::INVALID_ATTRIBUTE_VALUE,
            message: format!("invalid `{}` value: {error}", attr.as_str()),
        },
    }
}

/// Checks a `style` attribute's value: [`codes::INVALID_STYLE_ATTRIBUTE`]
/// if it is not a well-formed declaration list, otherwise each
/// declaration's value (reusing [`validate_attr_value`] via
/// [`validate_declaration_value`]) and each declaration's `url(#x)`
/// references.
fn check_style_attribute(
    value: &str,
    host: ElementId,
    options: &LintOptions,
    diagnostics: &mut Vec<Diagnostic>,
    node: NodeId,
    ref_sites: &mut Vec<RefSite>,
) {
    match parse_style_attribute_strict(value) {
        Err(_) => {
            if options.checks_values() {
                diagnostics.push(Diagnostic::warning(
                    codes::INVALID_STYLE_ATTRIBUTE,
                    "the `style` attribute is not a well-formed declaration list",
                    node,
                ));
            }
        }
        Ok(declarations) => {
            for declaration in &declarations {
                if options.checks_values() {
                    report_value_check(
                        validate_declaration_value(&declaration.name, &declaration.value, host),
                        diagnostics,
                        node,
                    );
                }
                if options.checks_references() {
                    if let Some(attr) = property_name_to_attr(&declaration.name) {
                        collect_reference_sites(attr, host, &declaration.value, node, ref_sites);
                    }
                }
            }
        }
    }
}

/// Validates one `style` declaration's value, by mapping its CSS property
/// name back onto the equivalent [`AttrId`] and reusing
/// [`validate_attr_value`] — the "shared dispatcher" both plain
/// presentation attributes and `style` declarations validate through.
///
/// A declaration whose name is not a [`PropertyId`] this crate models
/// (including every geometry property such as `width`/`cx`/`d`, which SVG 2
/// also promotes to a CSS property but [`PropertyId`] does not enumerate —
/// see that type's module documentation) is [`ValueCheck::Skipped`], not
/// reported as an unknown property: this module has no "unknown property"
/// code, only "unknown attribute".
fn validate_declaration_value(name: &str, value: &str, host: ElementId) -> ValueCheck {
    match property_name_to_attr(name) {
        Some(attr) => validate_attr_value(attr, host, value),
        None => ValueCheck::Skipped,
    }
}

/// Maps a CSS property name (as authored in a `style` declaration) to the
/// [`AttrId`] with the identical spelling, via [`PropertyId::from_name`]
/// as a "is this a real, modeled CSS property" gate. Every [`PropertyId`]
/// this crate defines has an [`AttrId`] counterpart with the same
/// [`PropertyId::as_str`]/[`AttrId::as_str`] spelling (both are drawn from
/// the same presentation-attribute vocabulary), so the second lookup is
/// expected to always succeed when the first does.
fn property_name_to_attr(name: &str) -> Option<AttrId> {
    PropertyId::from_name(name).and_then(|property| AttrId::from_name(property.as_str()))
}

/// Collects the `url(#x)`/`href="#x"`-shaped local reference(s) an
/// attribute or style-declaration value contains, if the attribute is one
/// of the families that can hold one.
fn collect_reference_sites(
    attr: AttrId,
    host: ElementId,
    value: &str,
    node: NodeId,
    ref_sites: &mut Vec<RefSite>,
) {
    match attr {
        AttrId::Href | AttrId::XlinkHref => push_local_reference(value, node, ref_sites),
        AttrId::Fill | AttrId::Stroke => {
            if host.is_animation() {
                return;
            }
            if let Ok(Paint::Iri { href, .. }) = Paint::parse(value) {
                push_local_reference(&href, node, ref_sites);
            }
        }
        AttrId::ClipPath
        | AttrId::Mask
        | AttrId::Filter
        | AttrId::Marker
        | AttrId::MarkerStart
        | AttrId::MarkerMid
        | AttrId::MarkerEnd => {
            if value.trim().eq_ignore_ascii_case("none") {
                return;
            }
            if let Ok(iri) = FuncIri::parse(value) {
                push_local_reference(iri.reference(), node, ref_sites);
            }
        }
        _ => {}
    }
}

/// Pushes `href`'s fragment identifier (the part after `#`) as a
/// [`RefSite`], if `href` is a fragment-only reference at all (an
/// absolute or relative external reference is silently not checked, per
/// [`crate::values::iri::local_reference`]'s own documented scope) and the
/// fragment is non-empty (an empty fragment is not a meaningful reference
/// target; `<use>`'s specific empty-reference case has its own
/// [`codes::EMPTY_USE_REFERENCE`] check elsewhere).
fn push_local_reference(href: &str, node: NodeId, ref_sites: &mut Vec<RefSite>) {
    if let Some(id) = local_reference(href) {
        if !id.is_empty() {
            ref_sites.push(RefSite {
                node,
                reference: id.into(),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- resolve_attr_id ----

    #[test]
    fn plain_attribute_resolves_by_name() {
        assert!(matches!(
            resolve_attr_id(None, "fill"),
            AttrResolution::Known(AttrId::Fill)
        ));
    }

    #[test]
    fn xlink_prefixed_attribute_resolves_via_the_xlink_namespace() {
        assert!(matches!(
            resolve_attr_id(Some(XLINK_NAMESPACE), "href"),
            AttrResolution::Known(AttrId::XlinkHref)
        ));
    }

    #[test]
    fn xml_prefixed_attribute_resolves_via_the_xml_namespace() {
        assert!(matches!(
            resolve_attr_id(Some(XML_NAMESPACE), "lang"),
            AttrResolution::Known(AttrId::XmlLang)
        ));
    }

    #[test]
    fn xml_id_is_unknown_not_skipped() {
        // `xml:id` is a real, standardized attribute, but this crate's
        // `AttrId` catalog does not model it (see the module docs) — it
        // should read as "unknown", not be silently dropped like
        // `data-*`/`aria-*`.
        assert!(matches!(
            resolve_attr_id(Some(XML_NAMESPACE), "id"),
            AttrResolution::Unknown
        ));
    }

    #[test]
    fn xmlns_declarations_are_skipped() {
        assert!(matches!(
            resolve_attr_id(Some(XMLNS_NAMESPACE), "xlink"),
            AttrResolution::Skip
        ));
    }

    #[test]
    fn data_and_aria_attributes_are_skipped() {
        assert!(matches!(
            resolve_attr_id(None, "data-foo"),
            AttrResolution::Skip
        ));
        assert!(matches!(
            resolve_attr_id(None, "aria-hidden"),
            AttrResolution::Skip
        ));
    }

    #[test]
    fn a_foreign_namespace_attribute_is_skipped() {
        assert!(matches!(
            resolve_attr_id(Some("urn:example"), "whatever"),
            AttrResolution::Skip
        ));
    }

    #[test]
    fn an_unrecognized_plain_name_is_unknown() {
        assert!(matches!(
            resolve_attr_id(None, "frobnicate"),
            AttrResolution::Unknown
        ));
    }

    // ---- note_id ----

    #[test]
    fn note_id_records_the_first_occurrence_without_a_diagnostic() {
        let mut table: HashMap<Box<str>, NodeId> = HashMap::new();
        let mut diagnostics = Vec::new();
        let node = NodeId::from_index(1);
        note_id("a", node, &mut table, &mut diagnostics);
        assert!(diagnostics.is_empty());
        assert_eq!(table.get("a"), Some(&node));
    }

    #[test]
    fn note_id_reports_duplicate_id_for_a_later_occurrence_and_keeps_the_first() {
        let mut table: HashMap<Box<str>, NodeId> = HashMap::new();
        let mut diagnostics = Vec::new();
        let first = NodeId::from_index(1);
        let second = NodeId::from_index(2);
        note_id("a", first, &mut table, &mut diagnostics);
        note_id("a", second, &mut table, &mut diagnostics);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].code(), codes::DUPLICATE_ID);
        assert_eq!(diagnostics[0].node(), second);
        assert_eq!(table.get("a"), Some(&first), "XML first-wins");
    }

    #[test]
    fn note_id_ignores_an_empty_id() {
        let mut table: HashMap<Box<str>, NodeId> = HashMap::new();
        let mut diagnostics = Vec::new();
        note_id("", NodeId::from_index(1), &mut table, &mut diagnostics);
        assert!(table.is_empty());
        assert!(diagnostics.is_empty());
    }

    // ---- validate_attr_value: a representative sample per family ----

    #[test]
    fn d_dispatches_to_path_data() {
        assert_eq!(
            validate_attr_value(AttrId::D, ElementId::Path, "M0 0 L1 1"),
            ValueCheck::Valid
        );
        assert!(matches!(
            validate_attr_value(AttrId::D, ElementId::Path, "M0 0 X"),
            ValueCheck::Invalid {
                code: codes::INVALID_ATTRIBUTE_VALUE,
                ..
            }
        ));
    }

    #[test]
    fn view_box_negative_width_is_the_specific_code() {
        assert!(matches!(
            validate_attr_value(AttrId::ViewBox, ElementId::Svg, "0 0 -1 1"),
            ValueCheck::Invalid {
                code: codes::INVALID_VIEW_BOX,
                ..
            }
        ));
    }

    #[test]
    fn view_box_garbage_is_the_generic_code() {
        assert!(matches!(
            validate_attr_value(AttrId::ViewBox, ElementId::Svg, "not a viewbox"),
            ValueCheck::Invalid {
                code: codes::INVALID_ATTRIBUTE_VALUE,
                ..
            }
        ));
    }

    #[test]
    fn stroke_dasharray_accepts_the_none_keyword_first() {
        assert_eq!(
            validate_attr_value(AttrId::StrokeDasharray, ElementId::Rect, "none"),
            ValueCheck::Valid
        );
        assert_eq!(
            validate_attr_value(AttrId::StrokeDasharray, ElementId::Rect, "1,2,3"),
            ValueCheck::Valid
        );
        assert!(matches!(
            validate_attr_value(AttrId::StrokeDasharray, ElementId::Rect, "1,-2"),
            ValueCheck::Invalid { .. }
        ));
    }

    #[test]
    fn offset_means_stop_position_on_a_stop_and_a_plain_number_on_a_transfer_function() {
        assert_eq!(
            validate_attr_value(AttrId::Offset, ElementId::Stop, "50%"),
            ValueCheck::Valid
        );
        assert_eq!(
            validate_attr_value(AttrId::Offset, ElementId::FeFuncR, "0.5"),
            ValueCheck::Valid
        );
        assert_eq!(
            validate_attr_value(AttrId::Offset, ElementId::Rect, "0.5"),
            ValueCheck::Skipped
        );
    }

    #[test]
    fn currentcolor_and_inherit_are_accepted_before_color_grammar() {
        assert_eq!(
            validate_attr_value(AttrId::StopColor, ElementId::Stop, "currentColor"),
            ValueCheck::Valid
        );
        assert_eq!(
            validate_attr_value(AttrId::StopColor, ElementId::Stop, "inherit"),
            ValueCheck::Valid
        );
        assert_eq!(
            validate_attr_value(AttrId::StopColor, ElementId::Stop, "#ff0000"),
            ValueCheck::Valid
        );
        assert!(matches!(
            validate_attr_value(AttrId::StopColor, ElementId::Stop, "not-a-color"),
            ValueCheck::Invalid { .. }
        ));
    }

    #[test]
    fn href_and_xlink_href_have_no_grammar() {
        assert_eq!(
            validate_attr_value(AttrId::Href, ElementId::Use, "anything at all"),
            ValueCheck::Skipped
        );
    }

    #[test]
    fn an_unimplemented_attribute_family_is_skipped() {
        // keyTimes/keySplines/values are a documented, still-open scope cut
        // (see the module documentation): their validity is a cross-attribute
        // cardinality check, not a single value's grammar.
        assert_eq!(
            validate_attr_value(AttrId::KeyTimes, ElementId::Animate, "garbage"),
            ValueCheck::Skipped
        );
    }

    // ---- validate_attr_value: the enumerated-keyword attributes ----

    #[test]
    fn the_object_bounding_box_units_family_shares_one_vocabulary() {
        for attr in [
            AttrId::GradientUnits,
            AttrId::PatternUnits,
            AttrId::PatternContentUnits,
            AttrId::MaskUnits,
            AttrId::MaskContentUnits,
            AttrId::ClipPathUnits,
            AttrId::FilterUnits,
            AttrId::PrimitiveUnits,
        ] {
            assert_eq!(
                validate_attr_value(attr, ElementId::LinearGradient, "userSpaceOnUse"),
                ValueCheck::Valid,
                "{attr:?}"
            );
            assert_eq!(
                validate_attr_value(attr, ElementId::LinearGradient, "objectBoundingBox"),
                ValueCheck::Valid,
                "{attr:?}"
            );
            assert!(
                matches!(
                    validate_attr_value(attr, ElementId::LinearGradient, "bogus"),
                    ValueCheck::Invalid { .. }
                ),
                "{attr:?}"
            );
        }
    }

    #[test]
    fn marker_units_accepts_its_own_two_keywords() {
        assert_eq!(
            validate_attr_value(AttrId::MarkerUnits, ElementId::Marker, "strokeWidth"),
            ValueCheck::Valid
        );
        assert_eq!(
            validate_attr_value(AttrId::MarkerUnits, ElementId::Marker, "userSpaceOnUse"),
            ValueCheck::Valid
        );
        assert!(matches!(
            validate_attr_value(AttrId::MarkerUnits, ElementId::Marker, "objectBoundingBox"),
            ValueCheck::Invalid { .. }
        ));
    }

    #[test]
    fn spread_method_accepts_pad_reflect_repeat_only() {
        for value in ["pad", "reflect", "repeat"] {
            assert_eq!(
                validate_attr_value(AttrId::SpreadMethod, ElementId::RadialGradient, value),
                ValueCheck::Valid,
                "{value}"
            );
        }
        assert!(matches!(
            validate_attr_value(AttrId::SpreadMethod, ElementId::RadialGradient, "whenever"),
            ValueCheck::Invalid { .. }
        ));
    }

    #[test]
    fn calc_mode_rejects_a_value_outside_its_four_keywords() {
        for value in ["discrete", "linear", "paced", "spline"] {
            assert_eq!(
                validate_attr_value(AttrId::CalcMode, ElementId::Animate, value),
                ValueCheck::Valid,
                "{value}"
            );
        }
        assert!(matches!(
            validate_attr_value(AttrId::CalcMode, ElementId::Animate, "whenever"),
            ValueCheck::Invalid { .. }
        ));
    }

    #[test]
    fn restart_additive_and_accumulate_are_checked() {
        assert_eq!(
            validate_attr_value(AttrId::Restart, ElementId::Animate, "whenNotActive"),
            ValueCheck::Valid
        );
        assert!(matches!(
            validate_attr_value(AttrId::Restart, ElementId::Animate, "bogus"),
            ValueCheck::Invalid { .. }
        ));
        assert_eq!(
            validate_attr_value(AttrId::Additive, ElementId::Animate, "sum"),
            ValueCheck::Valid
        );
        assert!(matches!(
            validate_attr_value(AttrId::Additive, ElementId::Animate, "bogus"),
            ValueCheck::Invalid { .. }
        ));
        assert_eq!(
            validate_attr_value(AttrId::Accumulate, ElementId::Animate, "sum"),
            ValueCheck::Valid
        );
        assert!(matches!(
            validate_attr_value(AttrId::Accumulate, ElementId::Animate, "bogus"),
            ValueCheck::Invalid { .. }
        ));
    }

    #[test]
    fn edge_mode_is_checked_on_both_of_its_hosts() {
        for value in ["duplicate", "wrap", "none"] {
            assert_eq!(
                validate_attr_value(AttrId::EdgeMode, ElementId::FeGaussianBlur, value),
                ValueCheck::Valid,
                "{value}"
            );
        }
        assert!(matches!(
            validate_attr_value(AttrId::EdgeMode, ElementId::FeConvolveMatrix, "bogus"),
            ValueCheck::Invalid { .. }
        ));
    }

    #[test]
    fn attribute_type_is_checked_case_sensitively() {
        for value in ["CSS", "XML", "auto"] {
            assert_eq!(
                validate_attr_value(AttrId::AttributeType, ElementId::Animate, value),
                ValueCheck::Valid,
                "{value}"
            );
        }
        assert!(
            matches!(
                validate_attr_value(AttrId::AttributeType, ElementId::Animate, "css"),
                ValueCheck::Invalid { .. }
            ),
            "lowercase css is not the same keyword as CSS"
        );
    }

    #[test]
    fn zoom_and_pan_length_adjust_and_stitch_tiles_are_checked() {
        assert_eq!(
            validate_attr_value(AttrId::ZoomAndPan, ElementId::Svg, "magnify"),
            ValueCheck::Valid
        );
        assert!(matches!(
            validate_attr_value(AttrId::ZoomAndPan, ElementId::Svg, "bogus"),
            ValueCheck::Invalid { .. }
        ));
        assert_eq!(
            validate_attr_value(AttrId::LengthAdjust, ElementId::Text, "spacingAndGlyphs"),
            ValueCheck::Valid
        );
        assert!(matches!(
            validate_attr_value(AttrId::LengthAdjust, ElementId::Text, "bogus"),
            ValueCheck::Invalid { .. }
        ));
        assert_eq!(
            validate_attr_value(AttrId::StitchTiles, ElementId::FeTurbulence, "noStitch"),
            ValueCheck::Valid
        );
        assert!(matches!(
            validate_attr_value(AttrId::StitchTiles, ElementId::FeTurbulence, "bogus"),
            ValueCheck::Invalid { .. }
        ));
    }

    #[test]
    fn text_path_method_spacing_and_side_are_checked() {
        assert_eq!(
            validate_attr_value(AttrId::Method, ElementId::TextPath, "stretch"),
            ValueCheck::Valid
        );
        assert!(matches!(
            validate_attr_value(AttrId::Method, ElementId::TextPath, "bogus"),
            ValueCheck::Invalid { .. }
        ));
        assert_eq!(
            validate_attr_value(AttrId::Spacing, ElementId::TextPath, "exact"),
            ValueCheck::Valid
        );
        assert!(matches!(
            validate_attr_value(AttrId::Spacing, ElementId::TextPath, "bogus"),
            ValueCheck::Invalid { .. }
        ));
        assert_eq!(
            validate_attr_value(AttrId::Side, ElementId::TextPath, "right"),
            ValueCheck::Valid
        );
        assert!(matches!(
            validate_attr_value(AttrId::Side, ElementId::TextPath, "bogus"),
            ValueCheck::Invalid { .. }
        ));
    }

    #[test]
    fn channel_selectors_accept_only_the_four_color_channel_letters() {
        for attr in [AttrId::XChannelSelector, AttrId::YChannelSelector] {
            for value in ["R", "G", "B", "A"] {
                assert_eq!(
                    validate_attr_value(attr, ElementId::FeDisplacementMap, value),
                    ValueCheck::Valid,
                    "{attr:?} {value}"
                );
            }
            assert!(
                matches!(
                    validate_attr_value(attr, ElementId::FeDisplacementMap, "r"),
                    ValueCheck::Invalid { .. }
                ),
                "{attr:?}: lowercase r is not one of the four channel letters"
            );
        }
    }

    #[test]
    fn operator_is_checked_per_host_with_a_different_vocabulary_each() {
        for value in ["over", "in", "out", "atop", "xor", "lighter", "arithmetic"] {
            assert_eq!(
                validate_attr_value(AttrId::Operator, ElementId::FeComposite, value),
                ValueCheck::Valid,
                "{value}"
            );
        }
        assert!(matches!(
            validate_attr_value(AttrId::Operator, ElementId::FeComposite, "erode"),
            ValueCheck::Invalid { .. }
        ));
        for value in ["erode", "dilate"] {
            assert_eq!(
                validate_attr_value(AttrId::Operator, ElementId::FeMorphology, value),
                ValueCheck::Valid,
                "{value}"
            );
        }
        assert!(matches!(
            validate_attr_value(AttrId::Operator, ElementId::FeMorphology, "over"),
            ValueCheck::Invalid { .. }
        ));
    }

    #[test]
    fn type_is_checked_per_host_with_a_different_vocabulary_each() {
        assert_eq!(
            validate_attr_value(AttrId::Type, ElementId::AnimateTransform, "rotate"),
            ValueCheck::Valid
        );
        assert!(matches!(
            validate_attr_value(AttrId::Type, ElementId::AnimateTransform, "matrix"),
            ValueCheck::Invalid { .. }
        ));
        assert_eq!(
            validate_attr_value(AttrId::Type, ElementId::FeColorMatrix, "saturate"),
            ValueCheck::Valid
        );
        assert!(matches!(
            validate_attr_value(AttrId::Type, ElementId::FeColorMatrix, "rotate"),
            ValueCheck::Invalid { .. }
        ));
        for host in [
            ElementId::FeFuncR,
            ElementId::FeFuncG,
            ElementId::FeFuncB,
            ElementId::FeFuncA,
        ] {
            assert_eq!(
                validate_attr_value(AttrId::Type, host, "gamma"),
                ValueCheck::Valid,
                "{host:?}"
            );
        }
        assert_eq!(
            validate_attr_value(AttrId::Type, ElementId::FeTurbulence, "fractalNoise"),
            ValueCheck::Valid
        );
        assert!(matches!(
            validate_attr_value(AttrId::Type, ElementId::FeTurbulence, "gamma"),
            ValueCheck::Invalid { .. }
        ));
    }

    #[test]
    fn type_on_a_free_form_host_stays_unchecked() {
        // `script`/`style`'s `type` is a MIME type string and `a`'s is a MIME
        // type hint, neither a small enumerated vocabulary.
        assert_eq!(
            validate_attr_value(AttrId::Type, ElementId::Script, "module"),
            ValueCheck::Skipped
        );
        assert_eq!(
            validate_attr_value(AttrId::Type, ElementId::Style, "text/css"),
            ValueCheck::Skipped
        );
        assert_eq!(
            validate_attr_value(AttrId::Type, ElementId::A, "text/html"),
            ValueCheck::Skipped
        );
    }

    #[test]
    fn mode_is_checked_only_on_fe_blend() {
        for value in ["normal", "multiply", "color-dodge", "luminosity"] {
            assert_eq!(
                validate_attr_value(AttrId::Mode, ElementId::FeBlend, value),
                ValueCheck::Valid,
                "{value}"
            );
        }
        assert!(matches!(
            validate_attr_value(AttrId::Mode, ElementId::FeBlend, "bogus"),
            ValueCheck::Invalid { .. }
        ));
    }

    // ---- validate_length ----

    #[test]
    fn negative_width_is_negative_dimension() {
        assert!(matches!(
            validate_length(AttrId::Width, "-1"),
            ValueCheck::Invalid {
                code: codes::NEGATIVE_DIMENSION,
                ..
            }
        ));
    }

    #[test]
    fn negative_x_is_not_checked_by_validate_length_callers() {
        // `validate_length` itself does not special-case which attributes
        // reach it; the non-negative rule is scoped by
        // `NON_NEGATIVE_LENGTH_ATTRS`, so calling it directly for `x` (not
        // a member) must not apply the rule even for a negative value.
        assert_eq!(validate_length(AttrId::X, "-1"), ValueCheck::Valid);
    }

    #[test]
    fn auto_is_accepted_only_for_the_documented_attributes() {
        assert_eq!(validate_length(AttrId::Width, "auto"), ValueCheck::Valid);
        assert_eq!(validate_length(AttrId::Rx, "auto"), ValueCheck::Valid);
        assert!(matches!(
            validate_length(AttrId::X, "auto"),
            ValueCheck::Invalid { .. }
        ));
    }

    // ---- collect_reference_sites ----

    #[test]
    fn fill_url_is_collected_as_a_reference_site() {
        let mut sites = Vec::new();
        collect_reference_sites(
            AttrId::Fill,
            ElementId::Rect,
            "url(#g)",
            NodeId::from_index(2),
            &mut sites,
        );
        assert_eq!(sites.len(), 1);
        assert_eq!(&*sites[0].reference, "g");
    }

    #[test]
    fn fill_on_an_animation_element_is_not_a_reference_site() {
        let mut sites = Vec::new();
        collect_reference_sites(
            AttrId::Fill,
            ElementId::Animate,
            "freeze",
            NodeId::from_index(2),
            &mut sites,
        );
        assert!(sites.is_empty());
    }

    #[test]
    fn clip_path_none_is_not_a_reference_site() {
        let mut sites = Vec::new();
        collect_reference_sites(
            AttrId::ClipPath,
            ElementId::G,
            "none",
            NodeId::from_index(2),
            &mut sites,
        );
        assert!(sites.is_empty());
    }

    #[test]
    fn clip_path_url_is_collected() {
        let mut sites = Vec::new();
        collect_reference_sites(
            AttrId::ClipPath,
            ElementId::G,
            "url(#clip)",
            NodeId::from_index(2),
            &mut sites,
        );
        assert_eq!(sites.len(), 1);
        assert_eq!(&*sites[0].reference, "clip");
    }

    // ---- property_name_to_attr ----

    #[test]
    fn property_name_to_attr_round_trips_every_property() {
        for &property in PropertyId::ALL {
            let name = property.as_str();
            assert_eq!(
                property_name_to_attr(name).map(AttrId::as_str),
                Some(name),
                "PropertyId {name:?} has no matching AttrId"
            );
        }
    }

    #[test]
    fn property_name_to_attr_is_none_for_an_unmodeled_geometry_property() {
        // `width` is a CSS property in SVG 2 but this crate's `PropertyId`
        // does not enumerate it (see that type's module documentation).
        assert_eq!(property_name_to_attr("width"), None);
    }
}

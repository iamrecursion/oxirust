//! Shorthand expansion, [XSL 1.1 §5.2] and [§7.31].
//!
//! A shorthand never takes part in inheritance itself; it is rewritten into the
//! individual properties it stands for, which then go through ordinary
//! refinement. XSL fixes a *precedence order* for the interrelated shorthands
//! because XML attributes are unordered:
//!
//! 1. the least precise shorthands (`border`, `margin`, `font`, …);
//! 2. `border-style`, `border-color`, `border-width` (and `border-spacing`,
//!    `background-position`);
//! 3. `border-top`, `border-bottom`, `border-left`, `border-right`;
//! 4. the individual properties, which always win.
//!
//! [XSL 1.1 §5.2]: https://www.w3.org/TR/xsl11/#d0e4778
//! [§7.31]: https://www.w3.org/TR/xsl11/#section-N18327-Shorthand-Properties

use crate::error::{FoError, Result};
use crate::property::datatype::Color;
use crate::property::id::PropertyId;
use crate::property::key::PropertyKey;
use crate::property::parse::{border_width_keyword, parse_uri_specification, split_top_level};

/// The precedence class of a shorthand: lower expands first, so later, more
/// precise specifications overwrite earlier ones.
///
/// Individual (non-shorthand) properties are class `3` and are applied last.
#[must_use]
pub(crate) fn precedence(id: PropertyId) -> u8 {
    match id {
        PropertyId::BorderStyle
        | PropertyId::BorderColor
        | PropertyId::BorderWidth
        | PropertyId::BorderSpacing
        | PropertyId::BackgroundPosition => 1,
        PropertyId::BorderTop
        | PropertyId::BorderBottom
        | PropertyId::BorderLeft
        | PropertyId::BorderRight => 2,
        other if other.is_shorthand() => 0,
        _ => 3,
    }
}

/// Every property a shorthand can set, used for `shorthand="inherit"` and for
/// deciding whether a shorthand applies to a formatting object.
#[must_use]
pub fn expansion_targets(id: PropertyId) -> &'static [PropertyId] {
    use PropertyId as P;
    match id {
        P::Background => &[
            P::BackgroundColor,
            P::BackgroundImage,
            P::BackgroundRepeat,
            P::BackgroundAttachment,
            P::BackgroundPositionHorizontal,
            P::BackgroundPositionVertical,
        ],
        P::BackgroundPosition => &[
            P::BackgroundPositionHorizontal,
            P::BackgroundPositionVertical,
        ],
        P::Border => &[
            P::BorderTopWidth,
            P::BorderTopStyle,
            P::BorderTopColor,
            P::BorderBottomWidth,
            P::BorderBottomStyle,
            P::BorderBottomColor,
            P::BorderLeftWidth,
            P::BorderLeftStyle,
            P::BorderLeftColor,
            P::BorderRightWidth,
            P::BorderRightStyle,
            P::BorderRightColor,
        ],
        P::BorderTop => &[P::BorderTopWidth, P::BorderTopStyle, P::BorderTopColor],
        P::BorderBottom => &[
            P::BorderBottomWidth,
            P::BorderBottomStyle,
            P::BorderBottomColor,
        ],
        P::BorderLeft => &[P::BorderLeftWidth, P::BorderLeftStyle, P::BorderLeftColor],
        P::BorderRight => &[
            P::BorderRightWidth,
            P::BorderRightStyle,
            P::BorderRightColor,
        ],
        P::BorderColor => &[
            P::BorderTopColor,
            P::BorderRightColor,
            P::BorderBottomColor,
            P::BorderLeftColor,
        ],
        P::BorderStyle => &[
            P::BorderTopStyle,
            P::BorderRightStyle,
            P::BorderBottomStyle,
            P::BorderLeftStyle,
        ],
        P::BorderWidth => &[
            P::BorderTopWidth,
            P::BorderRightWidth,
            P::BorderBottomWidth,
            P::BorderLeftWidth,
        ],
        P::BorderSpacing => &[P::BorderSeparation],
        P::Cue => &[P::CueBefore, P::CueAfter],
        P::Font => &[
            P::FontStyle,
            P::FontVariant,
            P::FontWeight,
            P::FontSize,
            P::LineHeight,
            P::FontFamily,
            P::FontStretch,
            P::FontSizeAdjust,
        ],
        P::Margin => &[P::MarginTop, P::MarginRight, P::MarginBottom, P::MarginLeft],
        P::Padding => &[
            P::PaddingTop,
            P::PaddingRight,
            P::PaddingBottom,
            P::PaddingLeft,
        ],
        P::PageBreakAfter => &[P::BreakAfter, P::KeepWithNext],
        P::PageBreakBefore => &[P::BreakBefore, P::KeepWithPrevious],
        P::PageBreakInside => &[P::KeepTogether],
        P::Pause => &[P::PauseBefore, P::PauseAfter],
        P::Position => &[P::RelativePosition, P::AbsolutePosition],
        P::Size => &[P::PageHeight, P::PageWidth],
        P::VerticalAlign => &[
            P::AlignmentBaseline,
            P::AlignmentAdjust,
            P::BaselineShift,
            P::DominantBaseline,
        ],
        P::WhiteSpace => &[
            P::LinefeedTreatment,
            P::WhiteSpaceCollapse,
            P::WhiteSpaceTreatment,
            P::WrapOption,
        ],
        P::XmlLang => &[P::Country, P::Language],
        _ => &[],
    }
}

fn assign(target: PropertyId, value: &str) -> (PropertyKey, String) {
    (PropertyKey::whole(target), value.to_owned())
}

fn component(target: PropertyId, name: &str, value: &str) -> Result<(PropertyKey, String)> {
    let key = PropertyKey::component(target, name).ok_or_else(|| FoError::UnknownComponent {
        property: target.name().to_owned(),
        component: name.to_owned(),
    })?;
    Ok((key, value.to_owned()))
}

fn is_border_style(token: &str) -> bool {
    matches!(
        token,
        "none"
            | "hidden"
            | "dotted"
            | "dashed"
            | "solid"
            | "double"
            | "groove"
            | "ridge"
            | "inset"
            | "outset"
    )
}

fn is_length_like(token: &str) -> bool {
    let candidate = token.trim();
    if candidate.is_empty() {
        return false;
    }
    if !candidate.starts_with(|c: char| c.is_ascii_digit() || c == '.' || c == '-' || c == '+') {
        return false;
    }
    candidate
        .trim_start_matches(['+', '-'])
        .starts_with(|c: char| c.is_ascii_digit() || c == '.')
}

fn is_border_width(token: &str) -> bool {
    border_width_keyword(token).is_some() || is_length_like(token)
}

fn is_color(token: &str) -> bool {
    token == "transparent"
        || token.starts_with('#')
        || token.starts_with("rgb(")
        || token.starts_with("rgb-icc(")
        || token.starts_with("system-color(")
        || Color::from_keyword(token).is_some()
}

/// Distributes one to four values over the top, right, bottom and left edges,
/// as CSS2 (and hence XSL 1.1 §7.31.11) specifies.
fn box_sides(values: &[String]) -> Result<[&str; 4]> {
    Ok(match values.len() {
        1 => [&values[0], &values[0], &values[0], &values[0]],
        2 => [&values[0], &values[1], &values[0], &values[1]],
        3 => [&values[0], &values[1], &values[2], &values[1]],
        4 => [&values[0], &values[1], &values[2], &values[3]],
        _ => {
            return Err(FoError::invalid_value(
                "shorthand",
                values.join(" "),
                "expected one to four values",
            ));
        }
    })
}

fn border_triplet(
    id: PropertyId,
    raw: &str,
    targets: &[PropertyId; 3],
) -> Result<Vec<(PropertyKey, String)>> {
    let mut width = None;
    let mut style = None;
    let mut color = None;
    for token in split_top_level(raw, false) {
        if style.is_none() && is_border_style(&token) {
            style = Some(token);
        } else if width.is_none() && is_border_width(&token) {
            width = Some(token);
        } else if color.is_none() && is_color(&token) {
            color = Some(token);
        } else {
            return Err(FoError::invalid_value(
                id.name(),
                raw,
                format!("`{token}` is not a border width, style or colour"),
            ));
        }
    }
    let mut out = Vec::with_capacity(3);
    out.push(assign(targets[0], width.as_deref().unwrap_or("medium")));
    out.push(assign(targets[1], style.as_deref().unwrap_or("none")));
    if let Some(color) = color {
        out.push(assign(targets[2], &color));
    }
    Ok(out)
}

/// The `background-position` keyword table of XSL 1.1 §7.31.2.
fn background_position(raw: &str) -> Result<(String, String)> {
    let values = split_top_level(raw, false);
    let keyword_position = |token: &str| -> Option<(Option<&'static str>, Option<&'static str>)> {
        Some(match token {
            "left" => (Some("0%"), None),
            "right" => (Some("100%"), None),
            "top" => (None, Some("0%")),
            "bottom" => (None, Some("100%")),
            "center" => (None, None),
            _ => return None,
        })
    };
    match values.len() {
        1 => {
            let token = &values[0];
            if let Some((horizontal, vertical)) = keyword_position(token) {
                Ok((
                    horizontal.unwrap_or("50%").to_owned(),
                    vertical.unwrap_or("50%").to_owned(),
                ))
            } else {
                Ok((token.clone(), "50%".to_owned()))
            }
        }
        2 => {
            let (first, second) = (&values[0], &values[1]);
            match (keyword_position(first), keyword_position(second)) {
                (Some(a), Some(b)) => {
                    let horizontal = a.0.or(b.0).unwrap_or("50%");
                    let vertical = a.1.or(b.1).unwrap_or("50%");
                    Ok((horizontal.to_owned(), vertical.to_owned()))
                }
                _ => Ok((first.clone(), second.clone())),
            }
        }
        _ => Err(FoError::invalid_value(
            "background-position",
            raw,
            "expected one or two values",
        )),
    }
}

fn expand_font(raw: &str) -> Result<Vec<(PropertyKey, String)>> {
    use PropertyId as P;
    let trimmed = raw.trim();
    if matches!(
        trimmed,
        "caption" | "icon" | "menu" | "message-box" | "small-caption" | "status-bar"
    ) {
        // XSL 1.1 §7.31.13: system fonts are reached through `system-font()`.
        return Ok(expansion_targets(P::Font)
            .iter()
            .map(|target| {
                assign(
                    *target,
                    &format!("system-font({trimmed}, {})", target.name()),
                )
            })
            .collect());
    }

    let words = split_top_level(trimmed, false);
    let mut out = vec![
        assign(P::FontStyle, "normal"),
        assign(P::FontVariant, "normal"),
        assign(P::FontWeight, "normal"),
        assign(P::FontStretch, "normal"),
        assign(P::FontSizeAdjust, "none"),
        assign(P::LineHeight, "normal"),
    ];
    let mut index = 0;
    while index < words.len() {
        let word = words[index].as_str();
        match word {
            "italic" | "oblique" | "backslant" => out.push(assign(P::FontStyle, word)),
            "small-caps" => out.push(assign(P::FontVariant, word)),
            "bold" | "bolder" | "lighter" | "100" | "200" | "300" | "400" | "500" | "600"
            | "700" | "800" | "900" => out.push(assign(P::FontWeight, word)),
            "normal" => {}
            _ => break,
        }
        index += 1;
    }
    let size = words.get(index).ok_or_else(|| {
        FoError::invalid_value("font", raw, "the `font` shorthand requires a font size")
    })?;
    match size.split_once('/') {
        Some((size, line_height)) => {
            out.push(assign(P::FontSize, size));
            out.push(assign(P::LineHeight, line_height));
        }
        None => out.push(assign(P::FontSize, size)),
    }
    index += 1;
    let family = words[index..].join(" ");
    if family.trim().is_empty() {
        return Err(FoError::invalid_value(
            "font",
            raw,
            "the `font` shorthand requires a font family",
        ));
    }
    out.push(assign(P::FontFamily, family.trim()));
    Ok(out)
}

fn expand_background(raw: &str) -> Result<Vec<(PropertyKey, String)>> {
    use PropertyId as P;
    let mut out = vec![
        assign(P::BackgroundColor, "transparent"),
        assign(P::BackgroundImage, "none"),
        assign(P::BackgroundRepeat, "repeat"),
        assign(P::BackgroundAttachment, "scroll"),
        assign(P::BackgroundPositionHorizontal, "0%"),
        assign(P::BackgroundPositionVertical, "0%"),
    ];
    let mut position = Vec::new();
    for token in split_top_level(raw, false) {
        if token == "none" || parse_uri_specification(&token).is_some() {
            out.push(assign(P::BackgroundImage, &token));
        } else if matches!(
            token.as_str(),
            "repeat" | "repeat-x" | "repeat-y" | "no-repeat"
        ) {
            out.push(assign(P::BackgroundRepeat, &token));
        } else if matches!(token.as_str(), "scroll" | "fixed") {
            out.push(assign(P::BackgroundAttachment, &token));
        } else if is_color(&token) {
            out.push(assign(P::BackgroundColor, &token));
        } else {
            position.push(token);
        }
    }
    if !position.is_empty() {
        let (horizontal, vertical) = background_position(&position.join(" "))?;
        out.push(assign(P::BackgroundPositionHorizontal, &horizontal));
        out.push(assign(P::BackgroundPositionVertical, &vertical));
    }
    Ok(out)
}

/// Expands one shorthand specification into the individual properties it sets.
///
/// The returned pairs are in application order: later pairs overwrite earlier
/// ones.
pub(crate) fn expand(id: PropertyId, raw: &str) -> Result<Vec<(PropertyKey, String)>> {
    use PropertyId as P;
    let trimmed = raw.trim();
    if trimmed == "inherit" {
        return Ok(expansion_targets(id)
            .iter()
            .map(|target| assign(*target, "inherit"))
            .collect());
    }

    match id {
        P::Background => expand_background(trimmed),
        P::BackgroundPosition => {
            let (horizontal, vertical) = background_position(trimmed)?;
            Ok(vec![
                assign(P::BackgroundPositionHorizontal, &horizontal),
                assign(P::BackgroundPositionVertical, &vertical),
            ])
        }
        P::Border => {
            let mut out = Vec::new();
            for targets in [
                [P::BorderTopWidth, P::BorderTopStyle, P::BorderTopColor],
                [
                    P::BorderBottomWidth,
                    P::BorderBottomStyle,
                    P::BorderBottomColor,
                ],
                [P::BorderLeftWidth, P::BorderLeftStyle, P::BorderLeftColor],
                [
                    P::BorderRightWidth,
                    P::BorderRightStyle,
                    P::BorderRightColor,
                ],
            ] {
                out.extend(border_triplet(id, trimmed, &targets)?);
            }
            Ok(out)
        }
        P::BorderTop => border_triplet(
            id,
            trimmed,
            &[P::BorderTopWidth, P::BorderTopStyle, P::BorderTopColor],
        ),
        P::BorderBottom => border_triplet(
            id,
            trimmed,
            &[
                P::BorderBottomWidth,
                P::BorderBottomStyle,
                P::BorderBottomColor,
            ],
        ),
        P::BorderLeft => border_triplet(
            id,
            trimmed,
            &[P::BorderLeftWidth, P::BorderLeftStyle, P::BorderLeftColor],
        ),
        P::BorderRight => border_triplet(
            id,
            trimmed,
            &[
                P::BorderRightWidth,
                P::BorderRightStyle,
                P::BorderRightColor,
            ],
        ),
        P::BorderColor | P::BorderStyle | P::BorderWidth => {
            let values = split_top_level(trimmed, false);
            let sides = box_sides(&values)?;
            let targets = expansion_targets(id);
            Ok(targets
                .iter()
                .zip(sides)
                .map(|(target, value)| assign(*target, value))
                .collect())
        }
        P::BorderSpacing => {
            let values = split_top_level(trimmed, false);
            let (inline, block) = match values.len() {
                1 => (values[0].clone(), values[0].clone()),
                2 => (values[0].clone(), values[1].clone()),
                _ => {
                    return Err(FoError::invalid_value(
                        id.name(),
                        trimmed,
                        "expected one or two lengths",
                    ));
                }
            };
            Ok(vec![
                component(P::BorderSeparation, "inline-progression-direction", &inline)?,
                component(P::BorderSeparation, "block-progression-direction", &block)?,
            ])
        }
        P::Margin | P::Padding => {
            let values = split_top_level(trimmed, false);
            let sides = box_sides(&values)?;
            Ok(expansion_targets(id)
                .iter()
                .zip(sides)
                .map(|(target, value)| assign(*target, value))
                .collect())
        }
        P::Cue | P::Pause => {
            let values = split_top_level(trimmed, false);
            let targets = expansion_targets(id);
            let (before, after) = match values.len() {
                1 => (values[0].clone(), values[0].clone()),
                2 => (values[0].clone(), values[1].clone()),
                _ => {
                    return Err(FoError::invalid_value(
                        id.name(),
                        trimmed,
                        "expected one or two values",
                    ));
                }
            };
            Ok(vec![
                assign(targets[0], &before),
                assign(targets[1], &after),
            ])
        }
        P::Font => expand_font(trimmed),
        P::PageBreakAfter | P::PageBreakBefore => {
            let after = id == P::PageBreakAfter;
            let (break_value, keep) = match trimmed {
                "auto" => ("auto", "auto"),
                "always" => ("page", "auto"),
                "avoid" => ("auto", "always"),
                "left" => ("even-page", "auto"),
                "right" => ("odd-page", "auto"),
                other => {
                    return Err(FoError::invalid_value(
                        id.name(),
                        other,
                        "expected auto, always, avoid, left or right",
                    ));
                }
            };
            Ok(if after {
                vec![
                    assign(P::BreakAfter, break_value),
                    assign(P::KeepWithNext, keep),
                ]
            } else {
                vec![
                    assign(P::BreakBefore, break_value),
                    assign(P::KeepWithPrevious, keep),
                ]
            })
        }
        P::PageBreakInside => {
            let keep = match trimmed {
                "auto" => "auto",
                "avoid" => "always",
                other => {
                    return Err(FoError::invalid_value(
                        id.name(),
                        other,
                        "expected auto or avoid",
                    ));
                }
            };
            Ok(vec![assign(P::KeepTogether, keep)])
        }
        P::Position => {
            let (relative, absolute) = match trimmed {
                "static" => ("static", "auto"),
                "relative" => ("relative", "auto"),
                "absolute" => ("static", "absolute"),
                "fixed" => ("static", "fixed"),
                other => {
                    return Err(FoError::invalid_value(
                        id.name(),
                        other,
                        "expected static, relative, absolute or fixed",
                    ));
                }
            };
            Ok(vec![
                assign(P::RelativePosition, relative),
                assign(P::AbsolutePosition, absolute),
            ])
        }
        P::Size => {
            let values = split_top_level(trimmed, false);
            if values.len() == 1 && matches!(values[0].as_str(), "auto" | "landscape" | "portrait")
            {
                return Ok(vec![
                    assign(P::PageHeight, "auto"),
                    assign(P::PageWidth, "auto"),
                ]);
            }
            let (width, height) = match values.len() {
                1 => (values[0].clone(), values[0].clone()),
                2 => (values[0].clone(), values[1].clone()),
                _ => {
                    return Err(FoError::invalid_value(
                        id.name(),
                        trimmed,
                        "expected one or two lengths",
                    ));
                }
            };
            Ok(vec![
                assign(P::PageWidth, &width),
                assign(P::PageHeight, &height),
            ])
        }
        P::VerticalAlign => {
            let (baseline, adjust, shift) = match trimmed {
                "baseline" => ("baseline", "auto", "baseline"),
                "top" => ("before-edge", "auto", "baseline"),
                "text-top" => ("text-before-edge", "auto", "baseline"),
                "middle" => ("middle", "auto", "baseline"),
                "bottom" => ("after-edge", "auto", "baseline"),
                "text-bottom" => ("text-after-edge", "auto", "baseline"),
                "sub" => ("baseline", "auto", "sub"),
                "super" => ("baseline", "auto", "super"),
                other => ("baseline", other, "baseline"),
            };
            Ok(vec![
                assign(P::AlignmentBaseline, baseline),
                assign(P::AlignmentAdjust, adjust),
                assign(P::BaselineShift, shift),
                assign(P::DominantBaseline, "auto"),
            ])
        }
        P::WhiteSpace => {
            let (linefeed, collapse, treatment, wrap) = match trimmed {
                "normal" => (
                    "treat-as-space",
                    "true",
                    "ignore-if-surrounding-linefeed",
                    "wrap",
                ),
                "pre" => ("preserve", "false", "preserve", "no-wrap"),
                "nowrap" => (
                    "treat-as-space",
                    "true",
                    "ignore-if-surrounding-linefeed",
                    "no-wrap",
                ),
                other => {
                    return Err(FoError::invalid_value(
                        id.name(),
                        other,
                        "expected normal, pre or nowrap",
                    ));
                }
            };
            Ok(vec![
                assign(P::LinefeedTreatment, linefeed),
                assign(P::WhiteSpaceCollapse, collapse),
                assign(P::WhiteSpaceTreatment, treatment),
                assign(P::WrapOption, wrap),
            ])
        }
        P::XmlLang => {
            let mut parts = trimmed.splitn(2, '-');
            let language = parts.next().unwrap_or(trimmed);
            let mut out = vec![assign(P::Language, language)];
            if let Some(country) = parts.next()
                && !country.is_empty()
            {
                out.push(assign(P::Country, country));
            }
            Ok(out)
        }
        other => Err(FoError::invalid_value(
            other.name(),
            trimmed,
            "not a shorthand property",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expanded(name: &str, value: &str) -> Vec<(String, String)> {
        let id = PropertyId::parse(name).expect("a property");
        expand(id, value)
            .expect("expands")
            .into_iter()
            .map(|(key, value)| (key.to_string(), value))
            .collect()
    }

    fn lookup(pairs: &[(String, String)], name: &str) -> Option<String> {
        pairs
            .iter()
            .rev()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.clone())
    }

    #[test]
    fn margin_distributes_over_four_sides() {
        let pairs = expanded("margin", "1pt 2pt 3pt 4pt");
        assert_eq!(lookup(&pairs, "margin-top").as_deref(), Some("1pt"));
        assert_eq!(lookup(&pairs, "margin-right").as_deref(), Some("2pt"));
        assert_eq!(lookup(&pairs, "margin-bottom").as_deref(), Some("3pt"));
        assert_eq!(lookup(&pairs, "margin-left").as_deref(), Some("4pt"));

        let pairs = expanded("padding", "1pt 2pt");
        assert_eq!(lookup(&pairs, "padding-top").as_deref(), Some("1pt"));
        assert_eq!(lookup(&pairs, "padding-right").as_deref(), Some("2pt"));
        assert_eq!(lookup(&pairs, "padding-bottom").as_deref(), Some("1pt"));
        assert_eq!(lookup(&pairs, "padding-left").as_deref(), Some("2pt"));
    }

    #[test]
    fn border_sets_all_four_edges() {
        let pairs = expanded("border", "1pt solid red");
        for edge in ["top", "bottom", "left", "right"] {
            assert_eq!(
                lookup(&pairs, &format!("border-{edge}-width")).as_deref(),
                Some("1pt")
            );
            assert_eq!(
                lookup(&pairs, &format!("border-{edge}-style")).as_deref(),
                Some("solid")
            );
            assert_eq!(
                lookup(&pairs, &format!("border-{edge}-color")).as_deref(),
                Some("red")
            );
        }
    }

    #[test]
    fn font_shorthand_follows_the_css_grammar() {
        let pairs = expanded("font", "italic bold 12pt/14pt Times New Roman, serif");
        assert_eq!(lookup(&pairs, "font-style").as_deref(), Some("italic"));
        assert_eq!(lookup(&pairs, "font-weight").as_deref(), Some("bold"));
        assert_eq!(lookup(&pairs, "font-size").as_deref(), Some("12pt"));
        assert_eq!(lookup(&pairs, "line-height").as_deref(), Some("14pt"));
        assert_eq!(
            lookup(&pairs, "font-family").as_deref(),
            Some("Times New Roman, serif")
        );
        // Properties the shorthand always resets.
        assert_eq!(lookup(&pairs, "font-stretch").as_deref(), Some("normal"));
        assert_eq!(lookup(&pairs, "font-size-adjust").as_deref(), Some("none"));
    }

    #[test]
    fn css_compatibility_shorthands_map_to_xsl_properties() {
        let pairs = expanded("page-break-before", "always");
        assert_eq!(lookup(&pairs, "break-before").as_deref(), Some("page"));
        assert_eq!(
            lookup(&pairs, "keep-with-previous").as_deref(),
            Some("auto")
        );

        let pairs = expanded("page-break-inside", "avoid");
        assert_eq!(lookup(&pairs, "keep-together").as_deref(), Some("always"));

        let pairs = expanded("position", "absolute");
        assert_eq!(
            lookup(&pairs, "relative-position").as_deref(),
            Some("static")
        );
        assert_eq!(
            lookup(&pairs, "absolute-position").as_deref(),
            Some("absolute")
        );

        let pairs = expanded("white-space", "pre");
        assert_eq!(
            lookup(&pairs, "linefeed-treatment").as_deref(),
            Some("preserve")
        );
        assert_eq!(
            lookup(&pairs, "white-space-collapse").as_deref(),
            Some("false")
        );
        assert_eq!(lookup(&pairs, "wrap-option").as_deref(), Some("no-wrap"));

        let pairs = expanded("xml:lang", "en-GB");
        assert_eq!(lookup(&pairs, "language").as_deref(), Some("en"));
        assert_eq!(lookup(&pairs, "country").as_deref(), Some("GB"));
    }

    #[test]
    fn background_position_keyword_table() {
        assert_eq!(
            background_position("top").expect("maps"),
            ("50%".to_owned(), "0%".to_owned())
        );
        assert_eq!(
            background_position("bottom right").expect("maps"),
            ("100%".to_owned(), "100%".to_owned())
        );
        assert_eq!(
            background_position("2cm").expect("maps"),
            ("2cm".to_owned(), "50%".to_owned())
        );
    }

    #[test]
    fn border_spacing_maps_to_border_separation_components() {
        let pairs = expanded("border-spacing", "2pt 4pt");
        assert_eq!(
            lookup(&pairs, "border-separation.inline-progression-direction").as_deref(),
            Some("2pt")
        );
        assert_eq!(
            lookup(&pairs, "border-separation.block-progression-direction").as_deref(),
            Some("4pt")
        );
    }

    #[test]
    fn inherit_propagates_to_every_expansion_target() {
        let pairs = expanded("border", "inherit");
        assert_eq!(pairs.len(), expansion_targets(PropertyId::Border).len());
        assert!(pairs.iter().all(|(_, value)| value == "inherit"));
    }

    #[test]
    fn precedence_orders_the_interrelated_shorthands() {
        assert!(precedence(PropertyId::Border) < precedence(PropertyId::BorderWidth));
        assert!(precedence(PropertyId::BorderWidth) < precedence(PropertyId::BorderTop));
        assert!(precedence(PropertyId::BorderTop) < precedence(PropertyId::BorderTopWidth));
    }
}

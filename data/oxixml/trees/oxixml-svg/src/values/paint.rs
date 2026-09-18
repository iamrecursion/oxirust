//! The SVG `<paint>` microsyntax (the value grammar of `fill`/`stroke`) and
//! the `fill-opacity`/`stroke-opacity`/`opacity` number-or-percentage
//! microsyntax.
//!
//! # Grammar
//! ```text
//! paint ::= "none"
//!         | "currentColor"
//!         | <color> <icccolor>?
//!         | <funciri> [ "none" | "currentColor" | <color> <icccolor>? ]?
//!         | "context-fill"
//!         | "context-stroke"
//!         | "inherit"
//! ```
//! [`Paint`] models every production above **except** `inherit` and the
//! trailing `<icccolor>` annotation — see "Scope notes" below, which is a
//! **blocking note for callers implementing the CSS cascade**, not an
//! incidental one.
//!
//! `context-fill`/`context-stroke` are only meaningful inside `marker`
//! content referenced from a shape, or inside a `use`-instantiated shadow
//! tree; this module only recognizes the keyword, resolving what it means in
//! context is the typed document layer's job.
//!
//! # Scope notes
//! - **`inherit`**: this is a CSS-wide keyword valid on *any* property, not
//!   specific to `<paint>`'s own grammar, so it has no [`Paint`] variant.
//!   [`Paint::parse("inherit")`](Paint::parse) returns
//!   [`ValueError::UnknownKeyword`]. **Callers implementing the style
//!   cascade must special-case the literal attribute/property value
//!   `"inherit"` before calling [`Paint::parse`]**; this module will not do
//!   it for them.
//! - **`<icccolor>`**: not accepted, matching [`crate::values::color`]'s own
//!   documented cut (a `<paint>` value's `<color>` component is parsed by
//!   that module, so the cut applies transitively here too). A value like
//!   `"blue icc-color(...)"` parses the `blue` and then reports the
//!   remainder as [`ValueError::TrailingData`].
//! - **`hwb()`/`lab()`/`lch()`/`oklab()`/`oklch()`/`color()`**: not accepted
//!   inside the `<color>` component, for the same reason documented in
//!   [`crate::values::color`].
//! - **Numeric leniency**: [`parse_opacity`]'s number (and every number
//!   inside a `<color>` component) is scanned with the shared SVG `<number>`
//!   grammar (see [`crate::values::color`]'s module doc), a lenient superset
//!   of CSS's own `<number>` token — e.g. `parse_opacity("5.")` is accepted
//!   here, where strict CSS would reject the bare trailing point.

use std::fmt;

use crate::error::ValueError;
use crate::values::color::{self, Color};
use crate::values::iri;
use crate::values::lex::Cursor;

/// A resolved `<paint>` value: what `fill`/`stroke` (and the `paint-order`
/// family of properties) hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Paint {
    /// The `none` keyword: paint nothing.
    None,
    /// The `currentColor` keyword: use the computed value of the `color`
    /// property.
    CurrentColor,
    /// A resolved color, from a named keyword, hex notation, or a
    /// `rgb()`/`hsl()` function.
    Color(Color),
    /// The SVG2 `context-fill` keyword.
    ContextFill,
    /// The SVG2 `context-stroke` keyword.
    ContextStroke,
    /// A `url(...)` reference to a paint server element (a gradient or
    /// pattern), with an optional fallback to use if the reference fails to
    /// resolve.
    Iri {
        /// The raw reference text captured from inside `url(...)`, verbatim
        /// (e.g. `"#gradient1"`).
        href: Box<str>,
        /// The fallback value following `url(...)`, if one was given. This
        /// is [`None`] when nothing at all followed the `url(...)` (which is
        /// different from an explicit `none` fallback,
        /// [`Some`]`(`[`PaintFallback::None`]`)`  — the two must round-trip
        /// distinctly).
        fallback: Option<PaintFallback>,
    },
}

/// The fallback value that may follow a `url(...)` reference in a
/// [`Paint::Iri`], used if the reference does not resolve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaintFallback {
    /// The `none` keyword.
    None,
    /// The `currentColor` keyword.
    CurrentColor,
    /// A resolved color.
    Color(Color),
}

impl Paint {
    /// Parses a complete `<paint>` value.
    ///
    /// See the [module documentation](self) for the exact grammar accepted,
    /// and in particular the notes on `inherit` and `<icccolor>`, which this
    /// function does not handle.
    ///
    /// # Errors
    /// Returns [`ValueError`] with a byte offset into `input` if the value
    /// is empty or violates the grammar.
    pub fn parse(input: &str) -> Result<Self, ValueError> {
        let mut cursor = Cursor::new(input);
        cursor.skip_wsp();
        if cursor.is_at_end() {
            return Err(ValueError::Empty);
        }
        let paint = parse_from_cursor(&mut cursor)?;
        cursor.done()?;
        Ok(paint)
    }
}

fn parse_from_cursor(cursor: &mut Cursor<'_>) -> Result<Paint, ValueError> {
    if iri::looks_like_url(cursor) {
        let func = iri::parse_from_cursor(cursor)?;
        cursor.skip_wsp();
        let fallback = if cursor.is_at_end() {
            None
        } else {
            Some(parse_fallback(cursor)?)
        };
        return Ok(Paint::Iri {
            href: func.into_reference(),
            fallback,
        });
    }
    // Peek at a bareword via a cloned cursor: commit it only if it is one of
    // the paint-specific keywords, otherwise leave the real cursor untouched
    // so `color::parse_from_cursor` can do its own (hex/named/function)
    // dispatch from the same starting position. `lex::Cursor` intentionally
    // exposes no seek/rewind operation, so cloning-and-conditionally-
    // committing is how this crate composes cursor-driven sub-parsers.
    let mut probe = cursor.clone();
    let word = probe.keyword();
    if word.eq_ignore_ascii_case("none") {
        *cursor = probe;
        return Ok(Paint::None);
    }
    if word.eq_ignore_ascii_case("currentcolor") {
        *cursor = probe;
        return Ok(Paint::CurrentColor);
    }
    if word.eq_ignore_ascii_case("context-fill") {
        *cursor = probe;
        return Ok(Paint::ContextFill);
    }
    if word.eq_ignore_ascii_case("context-stroke") {
        *cursor = probe;
        return Ok(Paint::ContextStroke);
    }
    let color = color::parse_from_cursor(cursor)?;
    Ok(Paint::Color(color))
}

fn parse_fallback(cursor: &mut Cursor<'_>) -> Result<PaintFallback, ValueError> {
    let mut probe = cursor.clone();
    let word = probe.keyword();
    if word.eq_ignore_ascii_case("none") {
        *cursor = probe;
        return Ok(PaintFallback::None);
    }
    if word.eq_ignore_ascii_case("currentcolor") {
        *cursor = probe;
        return Ok(PaintFallback::CurrentColor);
    }
    let color = color::parse_from_cursor(cursor)?;
    Ok(PaintFallback::Color(color))
}

impl fmt::Display for Paint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => f.write_str("none"),
            Self::CurrentColor => f.write_str("currentColor"),
            Self::Color(color) => write!(f, "{color}"),
            Self::ContextFill => f.write_str("context-fill"),
            Self::ContextStroke => f.write_str("context-stroke"),
            Self::Iri { href, fallback } => {
                iri::write_url(f, href)?;
                if let Some(fallback) = fallback {
                    write!(f, " {fallback}")?;
                }
                Ok(())
            }
        }
    }
}

impl fmt::Display for PaintFallback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => f.write_str("none"),
            Self::CurrentColor => f.write_str("currentColor"),
            Self::Color(color) => write!(f, "{color}"),
        }
    }
}

/// Parses an opacity value (`fill-opacity`, `stroke-opacity`, `opacity`,
/// `stop-opacity`, ...): a `<number>` or `<percentage>`, clamped to
/// `0.0..=1.0` per CSS. Out-of-range magnitudes clamp silently; only grammar
/// violations are errors.
///
/// # Errors
/// Returns [`ValueError`] with a byte offset into `input` if the value is
/// empty or is not a number or percentage.
pub fn parse_opacity(input: &str) -> Result<f64, ValueError> {
    let mut cursor = Cursor::new(input);
    cursor.skip_wsp();
    if cursor.is_at_end() {
        return Err(ValueError::Empty);
    }
    let (value, is_percent) = color::number_or_percent(&mut cursor)?;
    cursor.done()?;
    let normalized = if is_percent { value / 100.0 } else { value };
    Ok(normalized.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn red() -> Color {
        Color::rgb(0xFF, 0x00, 0x00)
    }

    // ---- keyword forms ----

    #[test]
    fn none_keyword() {
        assert_eq!(Paint::parse("none"), Ok(Paint::None));
        assert_eq!(Paint::parse("None"), Ok(Paint::None));
        assert_eq!(Paint::parse("NONE"), Ok(Paint::None));
    }

    #[test]
    fn current_color_keyword() {
        assert_eq!(Paint::parse("currentColor"), Ok(Paint::CurrentColor));
        assert_eq!(Paint::parse("CURRENTCOLOR"), Ok(Paint::CurrentColor));
    }

    #[test]
    fn context_fill_and_stroke_keywords() {
        assert_eq!(Paint::parse("context-fill"), Ok(Paint::ContextFill));
        assert_eq!(Paint::parse("context-stroke"), Ok(Paint::ContextStroke));
    }

    #[test]
    fn inherit_is_not_handled_here_by_design() {
        // See the module doc: the cascade layer must intercept "inherit"
        // itself, before calling into this module at all.
        assert!(matches!(
            Paint::parse("inherit"),
            Err(ValueError::UnknownKeyword { .. })
        ));
    }

    // ---- plain <color> ----

    #[test]
    fn named_color_value() {
        assert_eq!(Paint::parse("red"), Ok(Paint::Color(red())));
    }

    #[test]
    fn hex_color_value() {
        assert_eq!(Paint::parse("#ff0000"), Ok(Paint::Color(red())));
    }

    #[test]
    fn functional_color_value() {
        assert_eq!(Paint::parse("rgb(255,0,0)"), Ok(Paint::Color(red())));
    }

    #[test]
    fn icc_color_annotation_is_trailing_data() {
        let result = Paint::parse("blue icc-color(srgb 0 0 1)");
        assert!(
            matches!(result, Err(ValueError::TrailingData { .. })),
            "got {result:?}"
        );
    }

    // ---- funciri forms, per the spec catalog's §9.3 #16 edge case ----

    #[test]
    fn bare_url_reference_has_no_fallback() {
        assert_eq!(
            Paint::parse("url(#g)"),
            Ok(Paint::Iri {
                href: "#g".into(),
                fallback: None,
            })
        );
    }

    #[test]
    fn url_with_color_fallback() {
        assert_eq!(
            Paint::parse("url(#g) red"),
            Ok(Paint::Iri {
                href: "#g".into(),
                fallback: Some(PaintFallback::Color(red())),
            })
        );
    }

    #[test]
    fn url_with_none_fallback_differs_from_no_fallback() {
        assert_eq!(
            Paint::parse("url(#g) none"),
            Ok(Paint::Iri {
                href: "#g".into(),
                fallback: Some(PaintFallback::None),
            })
        );
        assert_ne!(
            Paint::parse("url(#g) none").expect("explicit none fallback parses"),
            Paint::parse("url(#g)").expect("bare url() with no fallback parses")
        );
    }

    #[test]
    fn url_with_current_color_fallback() {
        assert_eq!(
            Paint::parse("url(#g) currentColor"),
            Ok(Paint::Iri {
                href: "#g".into(),
                fallback: Some(PaintFallback::CurrentColor),
            })
        );
    }

    #[test]
    fn url_quoting_styles_all_resolve_to_the_same_reference() {
        for input in [
            "url(#g)",
            "url( #g )",
            r##"url("#g")"##,
            "url('#g')",
            r##"url( "#g" )"##,
        ] {
            let parsed = Paint::parse(input).unwrap_or_else(|e| panic!("{input:?}: {e}"));
            assert_eq!(
                parsed,
                Paint::Iri {
                    href: "#g".into(),
                    fallback: None,
                },
                "input {input:?}"
            );
        }
    }

    #[test]
    fn context_fill_and_stroke_are_not_valid_fallbacks() {
        // The <paint> grammar's fallback alternatives are only
        // none/currentColor/<color>, not context-fill/context-stroke.
        assert!(Paint::parse("url(#g) context-fill").is_err());
    }

    // ---- error handling ----

    #[test]
    fn empty_value_is_empty_error() {
        assert_eq!(Paint::parse(""), Err(ValueError::Empty));
        assert_eq!(Paint::parse("   "), Err(ValueError::Empty));
    }

    #[test]
    fn garbage_is_rejected() {
        assert!(Paint::parse("not-a-paint-value").is_err());
        assert!(Paint::parse("url(#g) not-a-fallback(").is_err());
    }

    // ---- Display / round-trip (value-level, not textual) ----

    #[test]
    fn round_trip_through_display_and_parse_preserves_the_value() {
        let cases = [
            Paint::None,
            Paint::CurrentColor,
            Paint::ContextFill,
            Paint::ContextStroke,
            Paint::Color(red()),
            Paint::Color(Color::rgba(1, 2, 3, 4)),
            Paint::Iri {
                href: "#g".into(),
                fallback: None,
            },
            Paint::Iri {
                href: "#g".into(),
                fallback: Some(PaintFallback::None),
            },
            Paint::Iri {
                href: "#g".into(),
                fallback: Some(PaintFallback::CurrentColor),
            },
            Paint::Iri {
                href: "#g".into(),
                fallback: Some(PaintFallback::Color(red())),
            },
            Paint::Iri {
                href: "has space".into(),
                fallback: None,
            },
        ];
        for paint in cases {
            let text = paint.to_string();
            assert_eq!(Paint::parse(&text), Ok(paint), "round-trip of {text:?}");
        }
    }

    #[test]
    fn display_uses_spec_keyword_spellings() {
        assert_eq!(Paint::None.to_string(), "none");
        assert_eq!(Paint::CurrentColor.to_string(), "currentColor");
        assert_eq!(Paint::ContextFill.to_string(), "context-fill");
        assert_eq!(Paint::ContextStroke.to_string(), "context-stroke");
        assert_eq!(
            Paint::Iri {
                href: "#g".into(),
                fallback: None
            }
            .to_string(),
            "url(#g)"
        );
    }

    // ---- parse_opacity ----

    #[test]
    fn opacity_accepts_number_or_percentage() {
        assert_eq!(parse_opacity("0.5"), Ok(0.5));
        assert_eq!(parse_opacity("50%"), Ok(0.5));
        assert_eq!(parse_opacity("1"), Ok(1.0));
        assert_eq!(parse_opacity("0"), Ok(0.0));
    }

    #[test]
    fn opacity_clamps_out_of_range_magnitudes() {
        assert_eq!(parse_opacity("1.5"), Ok(1.0));
        assert_eq!(parse_opacity("150%"), Ok(1.0));
        assert_eq!(parse_opacity("-0.5"), Ok(0.0));
        assert_eq!(parse_opacity("-10%"), Ok(0.0));
    }

    #[test]
    fn opacity_rejects_garbage() {
        assert_eq!(parse_opacity(""), Err(ValueError::Empty));
        assert!(parse_opacity("abc").is_err());
        assert!(parse_opacity("50% extra").is_err());
    }

    #[test]
    fn opacity_tolerates_surrounding_whitespace() {
        assert_eq!(parse_opacity("  0.5  "), Ok(0.5));
    }
}

//! The `transform` attribute microsyntax: `<transform-list>`.
//!
//! This is the SVG presentation-attribute grammar (`matrix|translate|scale|
//! rotate|skewX|skewY`, comma-wsp separated), **not** the CSS `transform`
//! *property* grammar SVG 2 additionally allows through `style="transform:
//! ..."` or a stylesheet rule (which adds `translateX/Y/Z`, `scale3d`,
//! `rotate3d`, `matrix3d`, length/percentage units, and `deg/grad/rad/turn`
//! angle units). The two are intentionally separate entry points sharing
//! only the [`Matrix`] output type; this module implements the
//! attribute-grammar entry point only.
//!
//! # Round-tripping short forms
//!
//! [`Transform`] keeps the arguments of each function as the author wrote
//! them: a one-argument `scale(2)` is `Scale { sx: 2.0, sy: None }`, not
//! eagerly expanded to `Scale { sx: 2.0, sy: Some(2.0) }`. [`Display`](fmt::Display)
//! reads that back out, so a value parsed from a short form serializes back
//! to the same short form rather than always writing the fully expanded
//! form. [`Transform::to_matrix`] resolves the omitted argument (`ty`
//! defaults to `0`, `sy` defaults to `sx`, an omitted rotation center means
//! rotation about the origin) when it needs an actual [`Matrix`].
//!
//! # A note on error-variant choice and separator strictness
//!
//! Two different separator rules are in play, matching the two places a
//! separator can go missing:
//!
//! - *Between the arguments of one function* (e.g. the six numbers of
//!   `matrix(...)`), this parser follows the same lenient, sign-as-delimiter
//!   convention [`crate::values::lex::parse_number_list`] and
//!   [`crate::values::aspect`] use elsewhere in the crate, even though the
//!   grammar transcribes a mandatory (non-`?`) `comma-wsp` there: a missing
//!   separator is only accepted when the next character (a sign, digit, or
//!   `.`) could only start a new number, exactly as within path data.
//! - *Between two functions in the list* has no such escape hatch: a
//!   function name always starts with a letter, never a sign or digit, so
//!   `matrix(...)rotate(...)` with zero separator characters is rejected
//!   unconditionally, per the grammar note that comma-wsp is required
//!   there.
//!
//! Both cases report [`ValueError::UnexpectedChar`]/[`ValueError::UnexpectedEnd`]
//! when the value is still incomplete (more arguments or functions are
//! still expected), and [`ValueError::TrailingData`] only once a
//! syntactically complete transform list has already been recognized and
//! something further follows it.

use std::fmt;
use std::str::FromStr;

use crate::error::ValueError;
use crate::geometry::matrix::Matrix;
use crate::values::lex::Cursor;

/// One transform function from a `transform` attribute value.
///
/// See the [module documentation](self) for how the optional arguments
/// round-trip through [`Display`](fmt::Display) and resolve through
/// [`Transform::to_matrix`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Transform {
    /// `matrix(a, b, c, d, e, f)` — a raw affine matrix.
    Matrix(Matrix),
    /// `translate(tx)` or `translate(tx, ty)`.
    Translate {
        /// The x translation.
        tx: f64,
        /// The y translation, when the author wrote a second argument.
        /// `None` means it was omitted, which [`Transform::to_matrix`]
        /// resolves to `0`, not to `tx`.
        ty: Option<f64>,
    },
    /// `scale(s)` or `scale(sx, sy)`.
    Scale {
        /// The x scale factor.
        sx: f64,
        /// The y scale factor, when the author wrote a second argument.
        /// `None` means it was omitted, which [`Transform::to_matrix`]
        /// resolves to `sx` (uniform scaling), not to `1`.
        sy: Option<f64>,
    },
    /// `rotate(angle)` or `rotate(angle, cx, cy)`.
    Rotate {
        /// The rotation angle in degrees.
        angle: f64,
        /// The rotation center, when the author wrote it. `None` means
        /// rotation about the origin.
        center: Option<(f64, f64)>,
    },
    /// `skewX(angle)`, in degrees.
    SkewX(f64),
    /// `skewY(angle)`, in degrees.
    SkewY(f64),
}

impl Transform {
    /// The affine matrix this single transform function represents.
    ///
    /// Omitted arguments resolve per the SVG specification: an omitted
    /// `translate` `ty` is `0` (not `tx`); an omitted `scale` `sy` is `sx`
    /// (uniform scaling, not `1`); an omitted `rotate` center means
    /// rotation about the origin.
    #[must_use]
    pub fn to_matrix(&self) -> Matrix {
        match self {
            Self::Matrix(m) => *m,
            Self::Translate { tx, ty } => Matrix::translate(*tx, ty.unwrap_or(0.0)),
            Self::Scale { sx, sy } => Matrix::scale(*sx, sy.unwrap_or(*sx)),
            Self::Rotate {
                angle,
                center: None,
            } => Matrix::rotate(*angle),
            Self::Rotate {
                angle,
                center: Some((cx, cy)),
            } => Matrix::rotate_about(*angle, *cx, *cy),
            Self::SkewX(angle) => Matrix::skew_x(*angle),
            Self::SkewY(angle) => Matrix::skew_y(*angle),
        }
    }

    /// True when every numeric argument of this transform function is
    /// finite.
    ///
    /// An unusually long digit run in any argument parses to an infinity
    /// rather than an error (`str::parse::<f64>` saturates on overflow
    /// instead of erroring — see [`crate::values::lex::Cursor::number`]), and
    /// [`Transform`]'s own [`Display`](fmt::Display) has no other bound on
    /// its arguments' magnitude. A non-finite `Transform` still formats as a
    /// well-formed function call in every other respect, but the
    /// `inf`/`-inf`/`NaN` argument itself is not valid `<number>` syntax, so
    /// callers that need round-tripping should check this first, the same
    /// way [`crate::values::path::PathData::is_finite`] lets a path-data
    /// caller do.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        match self {
            Self::Matrix(m) => {
                m.a.is_finite()
                    && m.b.is_finite()
                    && m.c.is_finite()
                    && m.d.is_finite()
                    && m.e.is_finite()
                    && m.f.is_finite()
            }
            Self::Translate { tx, ty } => tx.is_finite() && ty.is_none_or(f64::is_finite),
            Self::Scale { sx, sy } => sx.is_finite() && sy.is_none_or(f64::is_finite),
            Self::Rotate { angle, center } => {
                angle.is_finite() && center.is_none_or(|(cx, cy)| cx.is_finite() && cy.is_finite())
            }
            Self::SkewX(angle) | Self::SkewY(angle) => angle.is_finite(),
        }
    }
}

impl fmt::Display for Transform {
    /// Formats this transform function in the same short or long form its
    /// arguments carry, e.g. `translate(1)` when `ty` is `None`,
    /// `translate(1, 2)` when it is `Some(2.0)`.
    ///
    /// Round-trips when [`Transform::is_finite`] is true; see that method.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Matrix(m) => write!(f, "{m}"),
            Self::Translate { tx, ty: None } => write!(f, "translate({tx})"),
            Self::Translate { tx, ty: Some(ty) } => write!(f, "translate({tx}, {ty})"),
            Self::Scale { sx, sy: None } => write!(f, "scale({sx})"),
            Self::Scale { sx, sy: Some(sy) } => write!(f, "scale({sx}, {sy})"),
            Self::Rotate {
                angle,
                center: None,
            } => write!(f, "rotate({angle})"),
            Self::Rotate {
                angle,
                center: Some((cx, cy)),
            } => write!(f, "rotate({angle}, {cx}, {cy})"),
            Self::SkewX(angle) => write!(f, "skewX({angle})"),
            Self::SkewY(angle) => write!(f, "skewY({angle})"),
        }
    }
}

/// A parsed `transform` attribute value: an ordered list of transform
/// functions, applied left-to-right in the order SVG composes them.
///
/// The inner `Vec` is public so callers can build, inspect, and mutate a
/// list directly (`list.0.push(...)`); [`TransformList::new`] is offered
/// as an alternative discoverable through the type's own rustdoc.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TransformList(pub Vec<Transform>);

impl TransformList {
    /// Wraps an already-built sequence of transform operations.
    #[must_use]
    pub const fn new(ops: Vec<Transform>) -> Self {
        Self(ops)
    }

    /// True when the list has no operations. An empty (or white-space-only)
    /// `transform` attribute value parses to this: the grammar's
    /// `transforms?` is optional, unlike [`crate::values::aspect::ViewBox`]
    /// or [`crate::values::aspect::PreserveAspectRatio`], whose values must
    /// not be empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Parses a complete `transform` attribute value.
    ///
    /// ```
    /// use oxixml_svg::values::transform::TransformList;
    ///
    /// let list = TransformList::parse("translate(10,0) scale(2)").expect("valid value");
    /// let (x, y) = list.to_matrix().apply(1.0, 1.0);
    /// assert_eq!((x, y), (12.0, 2.0));
    /// ```
    pub fn parse(input: &str) -> Result<Self, ValueError> {
        let mut cursor = Cursor::new(input);
        let mut ops = Vec::new();

        cursor.skip_wsp();
        if cursor.is_at_end() {
            return Ok(Self(ops));
        }

        loop {
            ops.push(parse_transform(&mut cursor)?);
            let separated = cursor.skip_comma_wsp();
            if cursor.is_at_end() {
                break;
            }
            if !separated {
                // A function name always starts with a letter, so unlike a
                // number there is never a character that can serve as an
                // implicit delimiter here: a missing separator between two
                // functions is always an error. The first function already
                // parsed successfully, so this is content trailing a
                // complete value, not an incomplete one.
                return Err(ValueError::TrailingData {
                    offset: cursor.pos(),
                });
            }
        }

        Ok(Self(ops))
    }

    /// Composes every operation into a single affine matrix.
    ///
    /// SVG writes a transform list left-to-right but *applies* it to a
    /// point right-to-left: the last-written function acts on the point
    /// first, and the first-written function acts last, on the
    /// already-transformed point. Concretely, for `transform="A B C"` the
    /// resulting matrix is `M(A).multiply(&M(B)).multiply(&M(C))` — which
    /// this method builds by folding [`Matrix::multiply`] over the list in
    /// authored order — and satisfies
    /// `to_matrix().apply(p) == A.apply(B.apply(C.apply(p)))`.
    ///
    /// An empty list composes to [`Matrix::identity`].
    #[must_use]
    pub fn to_matrix(&self) -> Matrix {
        self.0
            .iter()
            .fold(Matrix::identity(), |acc, op| acc.multiply(&op.to_matrix()))
    }

    /// True when every operation's [`Transform::is_finite`] is true. An
    /// empty list is finite (vacuously).
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.0.iter().all(Transform::is_finite)
    }
}

impl fmt::Display for TransformList {
    /// Formats each operation with [`Transform`]'s own `Display`, separated
    /// by a single space.
    ///
    /// Round-trips when [`TransformList::is_finite`] is true; see that
    /// method.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for op in &self.0 {
            if !first {
                f.write_str(" ")?;
            }
            write!(f, "{op}")?;
            first = false;
        }
        Ok(())
    }
}

impl FromStr for TransformList {
    type Err = ValueError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

/// The six transform-function names the grammar recognizes, as a fieldless
/// tag distinct from `Transform` itself so [`parse_transform`]'s dispatch
/// on the function name is checked exhaustively by the compiler with no
/// need for a catch-all arm once the name has already been validated.
enum Kind {
    Matrix,
    Translate,
    Scale,
    Rotate,
    SkewX,
    SkewY,
}

/// Maps a scanned keyword to the [`Kind`] it names, or `None` if it is not
/// one of the six recognized transform-function names.
fn keyword_to_kind(name: &str) -> Option<Kind> {
    Some(match name {
        "matrix" => Kind::Matrix,
        "translate" => Kind::Translate,
        "scale" => Kind::Scale,
        "rotate" => Kind::Rotate,
        "skewX" => Kind::SkewX,
        "skewY" => Kind::SkewY,
        _ => return None,
    })
}

/// Parses one `name(args)` transform function starting at the cursor's
/// current position.
fn parse_transform(cursor: &mut Cursor<'_>) -> Result<Transform, ValueError> {
    let name_offset = cursor.pos();
    let name = cursor.keyword();
    if name.is_empty() {
        return Err(match cursor.peek() {
            Some(found) => ValueError::UnexpectedChar {
                offset: cursor.pos(),
                found,
            },
            None => ValueError::UnexpectedEnd {
                offset: cursor.pos(),
            },
        });
    }
    let kind = keyword_to_kind(name).ok_or_else(|| ValueError::UnknownKeyword {
        offset: name_offset,
        keyword: name.into(),
    })?;

    cursor.skip_wsp();
    cursor.expect_char('(')?;
    cursor.skip_wsp();

    let result = match kind {
        Kind::Matrix => {
            let mut components = [0.0_f64; 6];
            for (i, slot) in components.iter_mut().enumerate() {
                if i > 0 {
                    require_comma_wsp(cursor)?;
                }
                *slot = cursor.number()?;
            }
            let [a, b, c, d, e, f] = components;
            Transform::Matrix(Matrix::new(a, b, c, d, e, f))
        }
        Kind::Translate => {
            let tx = cursor.number()?;
            let ty = parse_optional_arg(cursor)?;
            Transform::Translate { tx, ty }
        }
        Kind::Scale => {
            let sx = cursor.number()?;
            let sy = parse_optional_arg(cursor)?;
            Transform::Scale { sx, sy }
        }
        Kind::Rotate => {
            let angle = cursor.number()?;
            cursor.skip_wsp();
            let center = if cursor.peek() == Some(')') {
                None
            } else {
                require_comma_wsp(cursor)?;
                let cx = cursor.number()?;
                require_comma_wsp(cursor)?;
                let cy = cursor.number()?;
                Some((cx, cy))
            };
            Transform::Rotate { angle, center }
        }
        Kind::SkewX => Transform::SkewX(cursor.number()?),
        Kind::SkewY => Transform::SkewY(cursor.number()?),
    };

    cursor.skip_wsp();
    cursor.expect_char(')')?;
    Ok(result)
}

/// Parses the optional second argument of `translate`/`scale`: either
/// nothing (the closing `)` follows, modulo white space) or one more
/// `comma-wsp`-separated number.
fn parse_optional_arg(cursor: &mut Cursor<'_>) -> Result<Option<f64>, ValueError> {
    cursor.skip_wsp();
    if cursor.peek() == Some(')') {
        return Ok(None);
    }
    require_comma_wsp(cursor)?;
    Ok(Some(cursor.number()?))
}

/// Requires a separator between two arguments of the *same* function, with
/// the lenient sign-as-delimiter tolerance documented in the
/// [module documentation](self).
fn require_comma_wsp(cursor: &mut Cursor<'_>) -> Result<(), ValueError> {
    if cursor.skip_comma_wsp() {
        return Ok(());
    }
    match cursor.peek() {
        Some('+' | '-' | '.' | '0'..='9') => Ok(()),
        Some(found) => Err(ValueError::UnexpectedChar {
            offset: cursor.pos(),
            found,
        }),
        None => Err(ValueError::UnexpectedEnd {
            offset: cursor.pos(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Transform::to_matrix ---------------------------------------------

    #[test]
    fn translate_omitted_ty_resolves_to_zero_not_tx() {
        let t = Transform::Translate { tx: 5.0, ty: None };
        assert_eq!(t.to_matrix().apply(0.0, 0.0), (5.0, 0.0));
    }

    #[test]
    fn scale_omitted_sy_resolves_to_sx_not_one() {
        let t = Transform::Scale { sx: 3.0, sy: None };
        assert_eq!(t.to_matrix().apply(1.0, 1.0), (3.0, 3.0));
    }

    #[test]
    fn rotate_omitted_center_is_about_the_origin() {
        let t = Transform::Rotate {
            angle: 90.0,
            center: None,
        };
        let (x, y) = t.to_matrix().apply(1.0, 0.0);
        assert!((x - 0.0).abs() < 1e-9 && (y - 1.0).abs() < 1e-9);
    }

    // -- TransformList::parse: basic shapes -------------------------------

    #[test]
    fn parses_an_empty_or_whitespace_only_value_as_an_empty_list() {
        assert_eq!(
            TransformList::parse("").expect("empty is valid"),
            TransformList(vec![])
        );
        assert_eq!(
            TransformList::parse("   ").expect("whitespace-only is valid"),
            TransformList(vec![])
        );
    }

    #[test]
    fn parses_matrix_with_only_commas_as_separators() {
        let list = TransformList::parse("matrix(1,0,0,1,0,0)").expect("valid matrix");
        assert_eq!(list.0, vec![Transform::Matrix(Matrix::identity())]);
    }

    #[test]
    fn parses_rotate_with_mixed_separators() {
        // Mission edge test: `'rotate(45, 5 5)'` mixes a comma+space
        // separator with a bare-space separator.
        let list = TransformList::parse("rotate(45, 5 5)").expect("valid rotate");
        assert_eq!(
            list.0,
            vec![Transform::Rotate {
                angle: 45.0,
                center: Some((5.0, 5.0)),
            }]
        );
    }

    #[test]
    fn parses_scientific_notation_arguments() {
        let list = TransformList::parse("translate(1e2, 2E-1)").expect("valid translate");
        assert_eq!(
            list.0,
            vec![Transform::Translate {
                tx: 100.0,
                ty: Some(0.2),
            }]
        );
    }

    #[test]
    fn parses_a_comma_between_two_of_three_functions() {
        // Mission edge test: `'translate(1) rotate(2),scale(3)'` — a plain
        // space separates the first pair of functions, a comma the second.
        let list =
            TransformList::parse("translate(1) rotate(2),scale(3)").expect("valid transform list");
        assert_eq!(
            list.0,
            vec![
                Transform::Translate { tx: 1.0, ty: None },
                Transform::Rotate {
                    angle: 2.0,
                    center: None,
                },
                Transform::Scale { sx: 3.0, sy: None },
            ]
        );
    }

    #[test]
    fn a_sign_may_stand_in_for_a_missing_separator_between_arguments() {
        // Deliberate deviation from the grammar's mandatory `comma-wsp`
        // between one function's own arguments; see the module
        // documentation. Distinct from between-functions separators, which
        // remain mandatory (tested below).
        let list = TransformList::parse("translate(1-2)").expect("sign disambiguates the numbers");
        assert_eq!(
            list.0,
            vec![Transform::Translate {
                tx: 1.0,
                ty: Some(-2.0),
            }]
        );
    }

    // -- TransformList::parse: rejections with offsets --------------------

    #[test]
    fn rejects_zero_separator_between_two_functions() {
        let err = TransformList::parse("translate(1)scale(2)")
            .expect_err("no separator between functions is invalid");
        assert_eq!(err, ValueError::TrailingData { offset: 12 });
    }

    #[test]
    fn rejects_an_unknown_function_name() {
        assert_eq!(
            TransformList::parse("foo(1)"),
            Err(ValueError::UnknownKeyword {
                offset: 0,
                keyword: "foo".into(),
            })
        );
    }

    #[test]
    fn rejects_matrix_with_too_few_arguments() {
        let err = TransformList::parse("matrix(1,2,3)").expect_err("matrix needs exactly 6 args");
        assert_eq!(
            err,
            ValueError::UnexpectedChar {
                offset: 12,
                found: ')',
            }
        );
    }

    #[test]
    fn rejects_matrix_with_too_many_arguments() {
        let err =
            TransformList::parse("matrix(1,2,3,4,5,6,7)").expect_err("matrix needs exactly 6 args");
        assert_eq!(
            err,
            ValueError::UnexpectedChar {
                offset: 18,
                found: ',',
            }
        );
    }

    #[test]
    fn rejects_rotate_with_two_arguments() {
        let err = TransformList::parse("rotate(1,2)").expect_err("rotate needs 1 or 3 args");
        assert_eq!(
            err,
            ValueError::UnexpectedChar {
                offset: 10,
                found: ')',
            }
        );
    }

    #[test]
    fn rejects_translate_with_three_arguments() {
        let err =
            TransformList::parse("translate(1,2,3)").expect_err("translate needs 1 or 2 args");
        assert_eq!(
            err,
            ValueError::UnexpectedChar {
                offset: 13,
                found: ',',
            }
        );
    }

    #[test]
    fn rejects_scale_with_three_arguments() {
        let err = TransformList::parse("scale(1,2,3)").expect_err("scale needs 1 or 2 args");
        assert_eq!(
            err,
            ValueError::UnexpectedChar {
                offset: 9,
                found: ',',
            }
        );
    }

    #[test]
    fn rejects_skew_x_with_two_arguments() {
        let err = TransformList::parse("skewX(1,2)").expect_err("skewX needs exactly 1 arg");
        assert_eq!(
            err,
            ValueError::UnexpectedChar {
                offset: 7,
                found: ',',
            }
        );
    }

    #[test]
    fn rejects_skew_y_with_two_arguments() {
        let err = TransformList::parse("skewY(1,2)").expect_err("skewY needs exactly 1 arg");
        assert_eq!(
            err,
            ValueError::UnexpectedChar {
                offset: 7,
                found: ',',
            }
        );
    }

    // -- to_matrix composition order --------------------------------------

    /// The mission's own numeric lock: `translate(10,0)` then `scale(2)`
    /// must map `(1,1)` to `(12,2)` — the leftmost function is applied
    /// last.
    #[test]
    fn to_matrix_composes_left_to_right_authored_order() {
        let list = TransformList::parse("translate(10,0) scale(2)").expect("valid list");
        assert_eq!(list.to_matrix().apply(1.0, 1.0), (12.0, 2.0));
    }

    #[test]
    fn to_matrix_of_an_empty_list_is_the_identity() {
        assert_eq!(
            TransformList::parse("")
                .expect("empty is valid")
                .to_matrix(),
            Matrix::identity()
        );
    }

    // -- Display / round-trip ----------------------------------------------

    /// The load-bearing round-trip test: parses a value containing every
    /// operation kind in both its short and long argument forms, and
    /// checks *value equality* after formatting and re-parsing — not just
    /// that re-parsing succeeds. A `Display` that silently expanded
    /// `scale(2)` into `scale(2, 2)` would still re-parse successfully but
    /// would fail this equality check, since the parsed `Scale { sy: None
    /// }` would no longer match `Scale { sy: Some(2.0) }`.
    #[test]
    fn display_round_trips_every_operation_in_both_short_and_long_form() {
        let source = "matrix(1, 0, 0, 1, 0, 0) translate(1) translate(1, 2) scale(2) \
                       scale(2, 3) rotate(45) rotate(45, 5, 5) skewX(10) skewY(10)";
        let list = TransformList::parse(source).expect("valid transform list");
        assert_eq!(list.0.len(), 9);

        let displayed = list.to_string();
        let round_tripped = TransformList::parse(&displayed).expect("Display output re-parses");
        assert_eq!(round_tripped, list);
    }

    /// The same round-trip property as above, but for numbers that do not
    /// survive `f64`'s `Display` trivially: a very small magnitude, a very
    /// large one, and a long-mantissa value (`0.1 + 0.2`, which is not
    /// exactly `0.3` in binary floating point). Rust's `f64` `Display`
    /// never switches to scientific notation (unlike its `Debug`), instead
    /// writing the full decimal expansion even for extreme magnitudes —
    /// this remains a grammar-valid `<number>` (an arbitrarily long digit
    /// sequence), so it still round-trips; this test pins that it actually
    /// does, rather than assuming it.
    #[test]
    fn display_round_trips_extreme_and_long_mantissa_numbers() {
        let list = TransformList(vec![
            Transform::Translate {
                tx: 1e-7,
                ty: Some(1e20),
            },
            Transform::Scale {
                sx: 0.1 + 0.2,
                sy: None,
            },
            Transform::Rotate {
                angle: -1e20,
                center: Some((1e-7, 0.1 + 0.2)),
            },
        ]);

        let displayed = list.to_string();
        let round_tripped =
            TransformList::parse(&displayed).expect("Display output re-parses even at extremes");
        assert_eq!(round_tripped, list);
    }

    #[test]
    fn is_finite_detects_an_overlong_argument_that_saturated_to_infinity() {
        // Unlike the extreme-but-finite magnitudes above, a long enough
        // digit run saturates `f64::parse` to infinity outright (it never
        // errors on overflow); `is_finite()` is what lets a caller detect
        // that before relying on a round trip through Display, which would
        // otherwise render an unparseable "inf" argument.
        let overlong = "9".repeat(400);
        let list = TransformList::parse(&format!("translate({overlong})"))
            .expect("grammar-valid, just an enormous number");
        assert!(!list.is_finite());
        assert!(!list.0[0].is_finite());

        let finite = TransformList::parse("translate(1) scale(2, 3)").expect("valid");
        assert!(finite.is_finite());
    }

    #[test]
    fn display_matches_the_expected_short_and_long_forms() {
        assert_eq!(
            Transform::Translate { tx: 1.0, ty: None }.to_string(),
            "translate(1)"
        );
        assert_eq!(
            Transform::Translate {
                tx: 1.0,
                ty: Some(2.0)
            }
            .to_string(),
            "translate(1, 2)"
        );
        assert_eq!(
            Transform::Scale { sx: 2.0, sy: None }.to_string(),
            "scale(2)"
        );
        assert_eq!(
            Transform::Scale {
                sx: 2.0,
                sy: Some(3.0)
            }
            .to_string(),
            "scale(2, 3)"
        );
        assert_eq!(
            Transform::Rotate {
                angle: 45.0,
                center: None
            }
            .to_string(),
            "rotate(45)"
        );
        assert_eq!(
            Transform::Rotate {
                angle: 45.0,
                center: Some((5.0, 5.0))
            }
            .to_string(),
            "rotate(45, 5, 5)"
        );
        assert_eq!(Transform::SkewX(10.0).to_string(), "skewX(10)");
        assert_eq!(Transform::SkewY(10.0).to_string(), "skewY(10)");
        assert_eq!(
            Transform::Matrix(Matrix::identity()).to_string(),
            "matrix(1, 0, 0, 1, 0, 0)"
        );
    }

    #[test]
    fn transform_list_display_joins_operations_with_a_single_space() {
        let list = TransformList(vec![
            Transform::Translate { tx: 1.0, ty: None },
            Transform::Scale { sx: 2.0, sy: None },
        ]);
        assert_eq!(list.to_string(), "translate(1) scale(2)");
    }

    #[test]
    fn from_str_delegates_to_parse() {
        let list: TransformList = "translate(1,2)".parse().expect("valid via FromStr");
        assert_eq!(
            list.0,
            vec![Transform::Translate {
                tx: 1.0,
                ty: Some(2.0)
            }]
        );
    }

    #[test]
    fn is_empty_reflects_the_inner_vec() {
        assert!(TransformList::new(vec![]).is_empty());
        assert!(!TransformList(vec![Transform::SkewX(1.0)]).is_empty());
    }
}

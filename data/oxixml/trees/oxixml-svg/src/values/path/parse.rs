//! The `svg-path` grammar parser.
//!
//! The parser drives a [`Cursor`] over the attribute value and appends one
//! [`PathSegment`] per *complete* argument group. Because an incomplete group
//! is never appended, the segment list that has accumulated when an error is
//! raised is exactly the prefix the SVG error-handling rule says to render:
//! everything up to but not including the command that contains the error,
//! plus the correctly formed segments of that command.

use crate::error::ValueError;
use crate::values::lex::Cursor;

use super::segment::PathSegment;

/// Parses `input` as far as the grammar allows.
///
/// Returns the complete segments that were read and, when the input did not
/// parse to its end, the error that stopped it.
pub(super) fn run(input: &str) -> (Vec<PathSegment>, Option<ValueError>) {
    let mut segments = Vec::new();
    let mut cursor = Cursor::new(input);
    cursor.skip_wsp();
    // `svg-path ::= wsp* moveto-drawto-command-groups? wsp*` — an empty or
    // white-space-only value is a valid, empty path.
    if cursor.is_at_end() {
        return (segments, None);
    }
    // Every command group starts with a moveto, so a path whose first token is
    // anything else is entirely in error: nothing at all is rendered.
    match cursor.peek() {
        Some('M' | 'm') => {}
        Some(found) => {
            return (
                Vec::new(),
                Some(ValueError::UnexpectedChar {
                    offset: cursor.pos(),
                    found,
                }),
            );
        }
        None => return (segments, None),
    }
    loop {
        let Some(letter) = cursor.bump() else {
            return (segments, None);
        };
        if let Err(error) = command(&mut cursor, letter, &mut segments) {
            return (segments, Some(error));
        }
        // Only white space may separate one command from the next: the
        // grammar's `comma-wsp` appears inside argument sequences, never
        // between commands.
        cursor.skip_wsp();
        match cursor.peek() {
            None => return (segments, None),
            Some(found) if is_command(found) => {}
            Some(found) => {
                return (
                    segments,
                    Some(ValueError::UnexpectedChar {
                        offset: cursor.pos(),
                        found,
                    }),
                );
            }
        }
    }
}

/// True for the twenty path command letters.
const fn is_command(c: char) -> bool {
    matches!(
        c,
        'M' | 'm'
            | 'L'
            | 'l'
            | 'H'
            | 'h'
            | 'V'
            | 'v'
            | 'C'
            | 'c'
            | 'S'
            | 's'
            | 'Q'
            | 'q'
            | 'T'
            | 't'
            | 'A'
            | 'a'
            | 'Z'
            | 'z'
    )
}

/// Reads the whole argument sequence of the command `letter`, whose letter has
/// already been consumed, appending one segment per argument group.
fn command(
    cursor: &mut Cursor<'_>,
    letter: char,
    out: &mut Vec<PathSegment>,
) -> Result<(), ValueError> {
    let abs = letter.is_ascii_uppercase();
    match letter {
        'Z' | 'z' => {
            out.push(PathSegment::Close);
            Ok(())
        }
        'M' | 'm' => {
            // `moveto-argument-sequence ::= coordinate-pair
            //     | coordinate-pair comma-wsp? lineto-argument-sequence`
            // — the pairs after the first are implicit linetos of the same
            // absoluteness as the moveto's own letter.
            cursor.skip_wsp();
            let (x, y) = coordinate_pair(cursor)?;
            out.push(PathSegment::MoveTo { abs, x, y });
            while more_arguments(cursor) {
                let (x, y) = coordinate_pair(cursor)?;
                out.push(PathSegment::LineTo { abs, x, y });
            }
            Ok(())
        }
        'L' | 'l' => sequence(cursor, out, |cursor| {
            let (x, y) = coordinate_pair(cursor)?;
            Ok(PathSegment::LineTo { abs, x, y })
        }),
        'H' | 'h' => sequence(cursor, out, |cursor| {
            Ok(PathSegment::HorizontalTo {
                abs,
                x: cursor.number()?,
            })
        }),
        'V' | 'v' => sequence(cursor, out, |cursor| {
            Ok(PathSegment::VerticalTo {
                abs,
                y: cursor.number()?,
            })
        }),
        'C' | 'c' => sequence(cursor, out, |cursor| {
            let (x1, y1) = coordinate_pair(cursor)?;
            cursor.skip_comma_wsp();
            let (x2, y2) = coordinate_pair(cursor)?;
            cursor.skip_comma_wsp();
            let (x, y) = coordinate_pair(cursor)?;
            Ok(PathSegment::CurveTo {
                abs,
                x1,
                y1,
                x2,
                y2,
                x,
                y,
            })
        }),
        'S' | 's' => sequence(cursor, out, |cursor| {
            let (x2, y2) = coordinate_pair(cursor)?;
            cursor.skip_comma_wsp();
            let (x, y) = coordinate_pair(cursor)?;
            Ok(PathSegment::SmoothCurveTo { abs, x2, y2, x, y })
        }),
        'Q' | 'q' => sequence(cursor, out, |cursor| {
            let (x1, y1) = coordinate_pair(cursor)?;
            cursor.skip_comma_wsp();
            let (x, y) = coordinate_pair(cursor)?;
            Ok(PathSegment::QuadTo { abs, x1, y1, x, y })
        }),
        'T' | 't' => sequence(cursor, out, |cursor| {
            let (x, y) = coordinate_pair(cursor)?;
            Ok(PathSegment::SmoothQuadTo { abs, x, y })
        }),
        'A' | 'a' => sequence(cursor, out, |cursor| arc(cursor, abs)),
        // `run` only dispatches on letters `is_command` accepted.
        other => Err(ValueError::UnexpectedChar {
            offset: cursor.pos(),
            found: other,
        }),
    }
}

/// Reads one argument sequence: at least one group, then as many further
/// groups as the input continues with.
fn sequence<F>(
    cursor: &mut Cursor<'_>,
    out: &mut Vec<PathSegment>,
    mut group: F,
) -> Result<(), ValueError>
where
    F: FnMut(&mut Cursor<'_>) -> Result<PathSegment, ValueError>,
{
    // `("L"|"l") wsp* lineto-argument-sequence` — only white space may follow
    // the command letter, not a comma.
    cursor.skip_wsp();
    loop {
        let segment = group(cursor)?;
        out.push(segment);
        if !more_arguments(cursor) {
            return Ok(());
        }
    }
}

/// Looks past an optional `comma-wsp` for another argument group.
///
/// The separator is consumed only when a group really follows, so that a
/// separator sitting between the end of an argument sequence and the next
/// command letter — which the grammar does not allow — is still reported at
/// its own offset.
fn more_arguments(cursor: &mut Cursor<'_>) -> bool {
    let saved = cursor.clone();
    cursor.skip_comma_wsp();
    if matches!(cursor.peek(), Some('+' | '-' | '.' | '0'..='9')) {
        true
    } else {
        *cursor = saved;
        false
    }
}

/// Reads `coordinate comma-wsp? coordinate`.
fn coordinate_pair(cursor: &mut Cursor<'_>) -> Result<(f64, f64), ValueError> {
    let x = cursor.number()?;
    cursor.skip_comma_wsp();
    let y = cursor.number()?;
    Ok((x, y))
}

/// Reads one `elliptical-arc-argument`.
///
/// The radii are read with the shared `<number>` scanner rather than the
/// grammar's unsigned `nonnegative-number`: the specification's own
/// out-of-range handling (SVG 1.1 Appendix F.6.2, SVG 2 §9.5.1) says a negative
/// radius has its sign dropped, which presupposes that it parses. The authored
/// sign is preserved here and normalized by the geometry layer.
fn arc(cursor: &mut Cursor<'_>, abs: bool) -> Result<PathSegment, ValueError> {
    let rx = cursor.number()?;
    cursor.skip_comma_wsp();
    let ry = cursor.number()?;
    cursor.skip_comma_wsp();
    let x_axis_rotation = cursor.number()?;
    cursor.skip_comma_wsp();
    let large_arc = cursor.flag()?;
    cursor.skip_comma_wsp();
    let sweep = cursor.flag()?;
    cursor.skip_comma_wsp();
    let (x, y) = coordinate_pair(cursor)?;
    Ok(PathSegment::ArcTo {
        abs,
        rx,
        ry,
        x_axis_rotation,
        large_arc,
        sweep,
        x,
        y,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(input: &str) -> Vec<PathSegment> {
        let (segments, error) = run(input);
        assert_eq!(error, None, "parsing {input:?}");
        segments
    }

    fn error(input: &str) -> ValueError {
        let (_, error) = run(input);
        error.unwrap_or_else(|| panic!("{input:?} was expected to fail"))
    }

    fn move_to(x: f64, y: f64) -> PathSegment {
        PathSegment::MoveTo { abs: true, x, y }
    }

    fn line_to(x: f64, y: f64) -> PathSegment {
        PathSegment::LineTo { abs: true, x, y }
    }

    #[test]
    fn an_empty_value_is_a_valid_empty_path() {
        assert_eq!(parse(""), Vec::new());
        assert_eq!(parse("   "), Vec::new());
        assert_eq!(parse(" \t\r\n "), Vec::new());
    }

    #[test]
    fn leading_and_trailing_white_space_is_allowed() {
        assert_eq!(parse("  M 0 0  "), vec![move_to(0.0, 0.0)]);
    }

    #[test]
    fn a_path_must_start_with_a_moveto() {
        assert_eq!(
            error("L 10 10"),
            ValueError::UnexpectedChar {
                offset: 0,
                found: 'L'
            }
        );
        assert_eq!(
            error("z"),
            ValueError::UnexpectedChar {
                offset: 0,
                found: 'z'
            }
        );
        assert_eq!(
            error("0 0"),
            ValueError::UnexpectedChar {
                offset: 0,
                found: '0'
            }
        );
        assert_eq!(
            error("\t\nZ"),
            ValueError::UnexpectedChar {
                offset: 2,
                found: 'Z'
            }
        );
        // Nothing at all is rendered for such a path.
        let (segments, _) = run("L 10 10 L 20 20");
        assert_eq!(segments, Vec::new());
    }

    #[test]
    fn separators_are_optional_where_the_grammar_is_unambiguous() {
        assert_eq!(
            parse("M10-20L.5.5"),
            vec![
                PathSegment::MoveTo {
                    abs: true,
                    x: 10.0,
                    y: -20.0
                },
                PathSegment::LineTo {
                    abs: true,
                    x: 0.5,
                    y: 0.5
                },
            ]
        );
        assert_eq!(parse("M 10,20"), parse("M10 20"));
        assert_eq!(parse("M 10 , 20"), parse("M10 20"));
    }

    #[test]
    fn exponents_and_bare_points_tokenize_per_the_number_grammar() {
        // `1e-2.5.5` is three numbers: 1e-2, .5 and .5. The third one is a
        // half-read implicit lineto, so only the moveto survives.
        let (segments, error) = run("M 1e-2.5.5");
        assert_eq!(
            segments,
            vec![PathSegment::MoveTo {
                abs: true,
                x: 0.01,
                y: 0.5
            }]
        );
        assert_eq!(error, Some(ValueError::UnexpectedEnd { offset: 10 }));
        // One more number completes the implicit lineto.
        assert_eq!(
            parse("M 1e-2.5.5.5"),
            vec![
                PathSegment::MoveTo {
                    abs: true,
                    x: 0.01,
                    y: 0.5
                },
                PathSegment::LineTo {
                    abs: true,
                    x: 0.5,
                    y: 0.5
                },
            ]
        );
        assert_eq!(
            parse("M1.5E+2 5."),
            vec![PathSegment::MoveTo {
                abs: true,
                x: 150.0,
                y: 5.0
            }]
        );
    }

    #[test]
    fn huge_exponents_saturate_rather_than_failing() {
        let segments = parse("M1e38 1e-38");
        assert_eq!(segments.len(), 1);
        let segments = parse("M1e999 0");
        match segments.as_slice() {
            [PathSegment::MoveTo { x, y, .. }] => {
                assert!(x.is_infinite() && x.is_sign_positive());
                assert_eq!(*y, 0.0);
            }
            other => panic!("unexpected segments {other:?}"),
        }
        assert!(!segments[0].is_finite());
    }

    #[test]
    fn extra_pairs_after_a_moveto_are_implicit_linetos() {
        assert_eq!(
            parse("M0 0 10 10 20 20"),
            vec![move_to(0.0, 0.0), line_to(10.0, 10.0), line_to(20.0, 20.0)]
        );
        assert_eq!(
            parse("m0 0 10 10"),
            vec![
                PathSegment::MoveTo {
                    abs: false,
                    x: 0.0,
                    y: 0.0
                },
                PathSegment::LineTo {
                    abs: false,
                    x: 10.0,
                    y: 10.0
                },
            ]
        );
    }

    #[test]
    fn drawto_commands_repeat_implicitly() {
        assert_eq!(
            parse("M0 0L1 1 2 2 3 3"),
            vec![
                move_to(0.0, 0.0),
                line_to(1.0, 1.0),
                line_to(2.0, 2.0),
                line_to(3.0, 3.0),
            ]
        );
        assert_eq!(
            parse("M0 0H1 2 3"),
            vec![
                move_to(0.0, 0.0),
                PathSegment::HorizontalTo { abs: true, x: 1.0 },
                PathSegment::HorizontalTo { abs: true, x: 2.0 },
                PathSegment::HorizontalTo { abs: true, x: 3.0 },
            ]
        );
        assert_eq!(parse("M240 296q25-100 47 0t47 0t47 0z").len(), 5);
    }

    #[test]
    fn every_command_letter_is_recognized_in_both_cases() {
        let path = "M0 0 L1 1 H2 V3 C4 4 5 5 6 6 S7 7 8 8 Q9 9 10 10 T11 11 \
                    A1 2 3 0 1 12 12 Z \
                    m0 0 l1 1 h2 v3 c4 4 5 5 6 6 s7 7 8 8 q9 9 10 10 t11 11 \
                    a1 2 3 0 1 12 12 z";
        let segments = parse(path);
        let letters: String = segments.iter().map(PathSegment::command).collect();
        assert_eq!(letters, "MLHVCSQTAZmlhvcsqtaZ");
    }

    #[test]
    fn closepath_takes_no_arguments_and_may_repeat() {
        assert_eq!(
            parse("M 0 0 Z Z"),
            vec![move_to(0.0, 0.0), PathSegment::Close, PathSegment::Close]
        );
        assert_eq!(parse("M0,0z"), vec![move_to(0.0, 0.0), PathSegment::Close]);
        assert_eq!(
            parse("M10,10Z L20,20"),
            vec![move_to(10.0, 10.0), PathSegment::Close, line_to(20.0, 20.0)]
        );
    }

    #[test]
    fn arc_flags_lex_as_single_digits() {
        let packed = parse("M0 0a1 1 0 011 0");
        let spaced = parse("M0 0a1 1 0 0,1 1 0");
        let expected = PathSegment::ArcTo {
            abs: false,
            rx: 1.0,
            ry: 1.0,
            x_axis_rotation: 0.0,
            large_arc: false,
            sweep: true,
            x: 1.0,
            y: 0.0,
        };
        assert_eq!(packed, vec![move_to(0.0, 0.0), expected]);
        assert_eq!(spaced, vec![move_to(0.0, 0.0), expected]);
        // The forms seen in the W3C SVG 1.1 test suite.
        assert_eq!(parse("M120,120 h25 a25,25 0 10 -25,25 z").len(), 4);
        assert_eq!(parse("M200,120 h-25 a25,25 0 1125,25 z").len(), 4);
        assert_eq!(parse("M120,200 h25 a25,25 0 1 1-25,-25 z").len(), 4);
    }

    #[test]
    fn a_full_circle_arc_pair_parses() {
        assert_eq!(
            parse("M0 0A1 1 0 1 1 2 0"),
            vec![
                move_to(0.0, 0.0),
                PathSegment::ArcTo {
                    abs: true,
                    rx: 1.0,
                    ry: 1.0,
                    x_axis_rotation: 0.0,
                    large_arc: true,
                    sweep: true,
                    x: 2.0,
                    y: 0.0
                }
            ]
        );
    }

    #[test]
    fn arc_radii_are_stored_as_authored() {
        // The grammar's `nonnegative-number` forbids the sign, but the spec's
        // own out-of-range handling drops it instead, so it must parse. No
        // normalization happens here.
        assert_eq!(
            parse("M0 0A-1 -2 400 0 1 2 0"),
            vec![
                move_to(0.0, 0.0),
                PathSegment::ArcTo {
                    abs: true,
                    rx: -1.0,
                    ry: -2.0,
                    x_axis_rotation: 400.0,
                    large_arc: false,
                    sweep: true,
                    x: 2.0,
                    y: 0.0
                }
            ]
        );
    }

    #[test]
    fn an_out_of_range_flag_is_rejected_at_its_own_offset() {
        assert_eq!(
            error("M0 0A1 1 0 2 1 2 0"),
            ValueError::UnexpectedChar {
                offset: 11,
                found: '2'
            }
        );
        assert_eq!(
            error("M0 0A1 1 0 -1 0 2 0"),
            ValueError::UnexpectedChar {
                offset: 11,
                found: '-'
            }
        );
        // Packing the flags against the rotation makes the greedy number
        // scanner swallow them (`0011` reads as the rotation 11), so the
        // grammar's mandatory separator there needs no separate enforcement.
        assert_eq!(
            error("M0 0a1 1 0011 0"),
            ValueError::UnexpectedEnd { offset: 15 }
        );
    }

    #[test]
    fn an_unknown_command_letter_ends_the_path_at_its_offset() {
        let (segments, error) = run("M10 10 L20 20 X30 30 L40 40");
        assert_eq!(segments, vec![move_to(10.0, 10.0), line_to(20.0, 20.0)]);
        assert_eq!(
            error,
            Some(ValueError::UnexpectedChar {
                offset: 14,
                found: 'X'
            })
        );
    }

    #[test]
    fn an_incomplete_argument_group_is_dropped() {
        let (segments, error) = run("M10,10 L 20,20 30");
        assert_eq!(segments, vec![move_to(10.0, 10.0), line_to(20.0, 20.0)]);
        assert_eq!(error, Some(ValueError::UnexpectedEnd { offset: 17 }));
        let (segments, error) = run("M 10 10 L");
        assert_eq!(segments, vec![move_to(10.0, 10.0)]);
        assert_eq!(error, Some(ValueError::UnexpectedEnd { offset: 9 }));
        let (segments, error) = run("M0 0C1 1 2 2");
        assert_eq!(segments, vec![move_to(0.0, 0.0)]);
        assert_eq!(error, Some(ValueError::UnexpectedEnd { offset: 12 }));
        let (segments, error) = run("M");
        assert_eq!(segments, Vec::new());
        assert_eq!(error, Some(ValueError::UnexpectedEnd { offset: 1 }));
    }

    #[test]
    fn a_separator_may_not_stand_between_two_commands() {
        assert_eq!(
            error("M0 0L1 1,L2 2"),
            ValueError::UnexpectedChar {
                offset: 8,
                found: ','
            }
        );
        assert_eq!(
            error("M0 0L1 1,"),
            ValueError::UnexpectedChar {
                offset: 8,
                found: ','
            }
        );
        // White space between commands is fine.
        assert_eq!(parse("M0 0 L1 1  \n L2 2").len(), 3);
    }

    #[test]
    fn a_separator_may_not_follow_a_command_letter() {
        assert_eq!(error("M,10 10"), ValueError::InvalidNumber { offset: 1 });
        assert_eq!(error("M0 0L,1 1"), ValueError::InvalidNumber { offset: 5 });
    }

    #[test]
    fn a_malformed_number_is_reported_where_it_starts() {
        assert_eq!(error("M0 0L1 x"), ValueError::InvalidNumber { offset: 7 });
        assert_eq!(error("M0 0H+"), ValueError::InvalidNumber { offset: 5 });
    }

    #[test]
    fn the_error_cases_of_the_w3c_test_suite_recover_their_valid_prefix() {
        // `paths-data-18-f.svg`: a stray character inside an argument list.
        let (segments, error) = run("M 20 100 H 40#90");
        assert_eq!(segments.len(), 2);
        assert_eq!(
            error,
            Some(ValueError::UnexpectedChar {
                offset: 13,
                found: '#'
            })
        );
        // `paths-data-20-f.svg`: five ways of writing an unusable arc flag.
        for (input, offset, found) in [
            ("M280,120 h25 a25,25 0 6 0 -25,25 z", 22, '6'),
            ("M360,120 h-25 a25,25 0 1 -1 25,25 z", 25, '-'),
            ("M200,200 h-25 a25,2501 025,-25 z", 27, '-'),
            ("M280,200 h25 a25 25 0 1 7 -25 -25 z", 24, '7'),
            ("M360,200 h-25 a25,25 0 -1 0 25,-25 z", 23, '-'),
        ] {
            let (segments, error) = run(input);
            assert_eq!(segments.len(), 2, "prefix of {input:?}");
            assert_eq!(
                error,
                Some(ValueError::UnexpectedChar { offset, found }),
                "error of {input:?}"
            );
        }
    }

    #[test]
    fn white_space_may_separate_a_command_letter_from_its_arguments() {
        assert_eq!(
            parse("   M   60 100      Q  -40 150      60 200       z  ").len(),
            3
        );
    }
}

//! Hershey-style stroke-based vector font renderer.
//!
//! Provides `put_text_hershey` which draws text using simplified stroke outlines
//! derived from the public-domain Hershey font data.  All printable ASCII
//! characters (0x20–0x7E) are supported.
//!
//! The coordinate system places the baseline at `y = 12` within a 12×16 glyph
//! box.  The `org` argument to `put_text_hershey` is the *bottom-left* origin
//! of the first character, matching the OpenCV `putText` convention.
//!
//! ## Font variant support
//! Font faces 0–4 (`FONT_HERSHEY_SIMPLEX`, `FONT_HERSHEY_PLAIN`,
//! `FONT_HERSHEY_DUPLEX`, `FONT_HERSHEY_COMPLEX`,
//! `FONT_HERSHEY_SCRIPT_SIMPLEX`) are all accepted and route through the same
//! simplex glyph table (variants are a simplified implementation — differences
//! in thickness and scale are left to the caller).  Unknown font face values
//! return [`Cv2Error::UnsupportedFlag`].

use crate::{
    drawing::line,
    error::{Cv2Error, Cv2Result},
    mat::{Mat, Point, Scalar},
};

// ── Glyph data ────────────────────────────────────────────────────────────────

/// A single stroke-font glyph: character advance width and a slice of strokes.
///
/// Each stroke is a sequence of `(x, y)` pairs to be connected with line
/// segments.  The glyph coordinate space has x in `0..=12` and y in `0..=12`,
/// with `y = 12` at the baseline.
struct HersheyGlyph {
    advance: i8,
    strokes: &'static [&'static [(i8, i8)]],
}

// ── Glyph stroke data ─────────────────────────────────────────────────────────

static GLYPH_0: &[&[(i8, i8)]] = &[&[
    (5, 0),
    (3, 0),
    (1, 2),
    (0, 5),
    (0, 7),
    (1, 10),
    (3, 12),
    (5, 12),
    (7, 12),
    (9, 10),
    (10, 7),
    (10, 5),
    (9, 2),
    (7, 0),
    (5, 0),
]];

static GLYPH_1: &[&[(i8, i8)]] = &[&[(2, 3), (4, 1), (4, 12)], &[(1, 12), (7, 12)]];

static GLYPH_2: &[&[(i8, i8)]] = &[&[
    (1, 3),
    (2, 1),
    (4, 0),
    (6, 0),
    (8, 1),
    (9, 3),
    (9, 4),
    (8, 6),
    (1, 11),
    (1, 12),
    (9, 12),
]];

static GLYPH_3: &[&[(i8, i8)]] = &[&[
    (1, 0),
    (9, 0),
    (5, 5),
    (7, 5),
    (9, 7),
    (9, 10),
    (8, 12),
    (5, 12),
    (3, 12),
    (1, 11),
]];

static GLYPH_4: &[&[(i8, i8)]] = &[&[(7, 0), (0, 8), (9, 8)], &[(7, 0), (7, 12)]];

static GLYPH_5: &[&[(i8, i8)]] = &[&[
    (8, 0),
    (1, 0),
    (1, 5),
    (5, 5),
    (8, 6),
    (9, 8),
    (9, 10),
    (7, 12),
    (4, 12),
    (2, 11),
    (1, 9),
]];

static GLYPH_6: &[&[(i8, i8)]] = &[&[
    (8, 1),
    (6, 0),
    (3, 0),
    (1, 3),
    (0, 6),
    (0, 9),
    (1, 11),
    (3, 12),
    (6, 12),
    (8, 11),
    (9, 9),
    (9, 8),
    (8, 6),
    (6, 5),
    (3, 5),
    (1, 6),
    (0, 8),
]];

static GLYPH_7: &[&[(i8, i8)]] = &[&[(0, 0), (9, 0), (3, 12)]];

static GLYPH_8: &[&[(i8, i8)]] = &[&[
    (4, 0),
    (2, 1),
    (1, 3),
    (1, 5),
    (2, 6),
    (5, 7),
    (7, 8),
    (8, 10),
    (8, 11),
    (6, 12),
    (3, 12),
    (1, 11),
    (1, 10),
    (2, 8),
    (5, 7),
    (7, 6),
    (8, 4),
    (8, 2),
    (7, 1),
    (5, 0),
    (4, 0),
]];

static GLYPH_9: &[&[(i8, i8)]] = &[&[
    (1, 11),
    (3, 12),
    (6, 12),
    (8, 11),
    (9, 9),
    (9, 6),
    (8, 4),
    (6, 3),
    (3, 3),
    (1, 4),
    (0, 6),
    (0, 7),
    (1, 9),
    (3, 10),
    (6, 10),
    (8, 9),
    (9, 7),
]];

static GLYPH_A: &[&[(i8, i8)]] = &[&[(0, 12), (4, 0), (8, 12)], &[(1, 8), (7, 8)]];

static GLYPH_B: &[&[(i8, i8)]] = &[
    &[(1, 0), (1, 12)],
    &[(1, 0), (6, 0), (8, 1), (8, 3), (7, 5), (5, 6), (1, 6)],
    &[(1, 6), (6, 6), (8, 7), (8, 10), (7, 11), (5, 12), (1, 12)],
];

static GLYPH_C: &[&[(i8, i8)]] = &[&[
    (8, 2),
    (6, 0),
    (4, 0),
    (2, 1),
    (0, 4),
    (0, 8),
    (2, 11),
    (4, 12),
    (6, 12),
    (8, 10),
]];

static GLYPH_D: &[&[(i8, i8)]] = &[
    &[(1, 0), (1, 12)],
    &[
        (1, 0),
        (5, 0),
        (7, 1),
        (9, 4),
        (9, 8),
        (7, 11),
        (5, 12),
        (1, 12),
    ],
];

static GLYPH_E: &[&[(i8, i8)]] = &[
    &[(1, 0), (1, 12)],
    &[(1, 0), (8, 0)],
    &[(1, 6), (6, 6)],
    &[(1, 12), (8, 12)],
];

static GLYPH_F: &[&[(i8, i8)]] = &[&[(1, 0), (1, 12)], &[(1, 0), (8, 0)], &[(1, 6), (6, 6)]];

static GLYPH_G: &[&[(i8, i8)]] = &[&[
    (8, 2),
    (6, 0),
    (4, 0),
    (2, 1),
    (0, 4),
    (0, 8),
    (2, 11),
    (4, 12),
    (6, 12),
    (8, 10),
    (8, 7),
    (5, 7),
]];

static GLYPH_H: &[&[(i8, i8)]] = &[&[(1, 0), (1, 12)], &[(8, 0), (8, 12)], &[(1, 6), (8, 6)]];

static GLYPH_I: &[&[(i8, i8)]] = &[&[(2, 0), (6, 0)], &[(4, 0), (4, 12)], &[(2, 12), (6, 12)]];

static GLYPH_J: &[&[(i8, i8)]] = &[
    &[(2, 0), (7, 0)],
    &[(5, 0), (5, 10), (4, 12), (2, 12), (1, 10)],
];

static GLYPH_K: &[&[(i8, i8)]] = &[&[(1, 0), (1, 12)], &[(8, 0), (1, 6)], &[(1, 6), (8, 12)]];

static GLYPH_L: &[&[(i8, i8)]] = &[&[(1, 0), (1, 12), (8, 12)]];

static GLYPH_M: &[&[(i8, i8)]] = &[&[(0, 12), (0, 0), (5, 8), (9, 0), (9, 12)]];

static GLYPH_N: &[&[(i8, i8)]] = &[&[(0, 12), (0, 0), (8, 12), (8, 0)]];

// ── Additional uppercase O–Z ──────────────────────────────────────────────────

static GLYPH_O: &[&[(i8, i8)]] = &[&[
    (5, 0),
    (3, 0),
    (1, 2),
    (0, 5),
    (0, 7),
    (1, 10),
    (3, 12),
    (5, 12),
    (7, 12),
    (9, 10),
    (10, 7),
    (10, 5),
    (9, 2),
    (7, 0),
    (5, 0),
]];

static GLYPH_P: &[&[(i8, i8)]] = &[
    &[(1, 0), (1, 12)],
    &[
        (1, 0),
        (6, 0),
        (8, 1),
        (9, 3),
        (9, 5),
        (8, 7),
        (6, 8),
        (1, 8),
    ],
];

static GLYPH_Q: &[&[(i8, i8)]] = &[
    &[
        (5, 0),
        (3, 0),
        (1, 2),
        (0, 5),
        (0, 7),
        (1, 10),
        (3, 12),
        (5, 12),
        (7, 12),
        (9, 10),
        (10, 7),
        (10, 5),
        (9, 2),
        (7, 0),
        (5, 0),
    ],
    &[(6, 9), (10, 13)],
];

static GLYPH_R: &[&[(i8, i8)]] = &[
    &[(1, 0), (1, 12)],
    &[
        (1, 0),
        (6, 0),
        (8, 1),
        (9, 3),
        (9, 5),
        (8, 7),
        (6, 8),
        (1, 8),
    ],
    &[(5, 8), (9, 12)],
];

static GLYPH_S: &[&[(i8, i8)]] = &[&[
    (9, 2),
    (8, 1),
    (6, 0),
    (4, 0),
    (2, 1),
    (1, 3),
    (1, 5),
    (2, 6),
    (4, 7),
    (6, 7),
    (8, 8),
    (9, 10),
    (9, 11),
    (8, 12),
    (6, 12),
    (4, 12),
    (2, 11),
    (1, 10),
]];

static GLYPH_T: &[&[(i8, i8)]] = &[&[(0, 0), (9, 0)], &[(4, 0), (4, 12)]];

static GLYPH_U: &[&[(i8, i8)]] = &[&[
    (1, 0),
    (1, 9),
    (2, 11),
    (4, 12),
    (5, 12),
    (7, 11),
    (8, 9),
    (8, 0),
]];

static GLYPH_V: &[&[(i8, i8)]] = &[&[(0, 0), (4, 12), (8, 0)]];

static GLYPH_W: &[&[(i8, i8)]] = &[&[(0, 0), (2, 12), (5, 5), (7, 12), (9, 0)]];

static GLYPH_X: &[&[(i8, i8)]] = &[&[(0, 0), (8, 12)], &[(8, 0), (0, 12)]];

static GLYPH_Y: &[&[(i8, i8)]] = &[&[(0, 0), (4, 7), (4, 12)], &[(8, 0), (4, 7)]];

static GLYPH_Z: &[&[(i8, i8)]] = &[&[(0, 0), (8, 0), (0, 12), (8, 12)]];

// ── Punctuation and special characters ───────────────────────────────────────

// Space has no strokes (advance only).
static GLYPH_SPACE: &[&[(i8, i8)]] = &[];

// '!'
static GLYPH_EXCLAIM: &[&[(i8, i8)]] = &[&[(4, 0), (4, 8)], &[(4, 10), (4, 12)]];

// '"'
static GLYPH_DQUOTE: &[&[(i8, i8)]] = &[&[(2, 0), (2, 4)], &[(6, 0), (6, 4)]];

// '#'
static GLYPH_HASH: &[&[(i8, i8)]] = &[
    &[(3, 0), (1, 12)],
    &[(7, 0), (5, 12)],
    &[(0, 4), (9, 4)],
    &[(0, 8), (9, 8)],
];

// '$'
static GLYPH_DOLLAR: &[&[(i8, i8)]] = &[
    &[(4, 0), (4, 12)],
    &[
        (8, 2),
        (7, 1),
        (5, 0),
        (3, 0),
        (1, 2),
        (1, 4),
        (3, 5),
        (6, 6),
        (8, 7),
        (8, 9),
        (7, 11),
        (5, 12),
        (3, 12),
        (1, 10),
    ],
];

// '%'
static GLYPH_PERCENT: &[&[(i8, i8)]] = &[
    &[(0, 0), (9, 12)],
    &[
        (2, 0),
        (1, 1),
        (1, 3),
        (2, 4),
        (4, 4),
        (5, 3),
        (5, 1),
        (4, 0),
        (2, 0),
    ],
    &[
        (5, 8),
        (4, 9),
        (4, 11),
        (5, 12),
        (7, 12),
        (8, 11),
        (8, 9),
        (7, 8),
        (5, 8),
    ],
];

// '&'
static GLYPH_AMPERSAND: &[&[(i8, i8)]] = &[&[
    (9, 12),
    (4, 0),
    (2, 0),
    (1, 2),
    (1, 4),
    (3, 6),
    (5, 8),
    (5, 10),
    (4, 12),
    (2, 12),
    (1, 10),
    (1, 9),
    (5, 5),
    (8, 12),
]];

// '\''
static GLYPH_SQUOTE: &[&[(i8, i8)]] = &[&[(3, 0), (4, 0), (4, 4)]];

// '('
static GLYPH_LPAREN: &[&[(i8, i8)]] = &[&[(5, 0), (3, 2), (2, 5), (2, 7), (3, 10), (5, 12)]];

// ')'
static GLYPH_RPAREN: &[&[(i8, i8)]] = &[&[(3, 0), (5, 2), (6, 5), (6, 7), (5, 10), (3, 12)]];

// '*'
static GLYPH_STAR: &[&[(i8, i8)]] = &[&[(4, 1), (4, 7)], &[(1, 3), (7, 5)], &[(7, 3), (1, 5)]];

// '+'
static GLYPH_PLUS: &[&[(i8, i8)]] = &[&[(4, 2), (4, 10)], &[(1, 6), (7, 6)]];

// ','
static GLYPH_COMMA: &[&[(i8, i8)]] = &[&[(3, 10), (3, 12), (2, 13)]];

// '-'
static GLYPH_MINUS: &[&[(i8, i8)]] = &[&[(1, 6), (7, 6)]];

// '.'
static GLYPH_PERIOD: &[&[(i8, i8)]] = &[&[(3, 11), (4, 11), (4, 12), (3, 12), (3, 11)]];

// '/'
static GLYPH_SLASH: &[&[(i8, i8)]] = &[&[(0, 12), (8, 0)]];

// ':'
static GLYPH_COLON: &[&[(i8, i8)]] = &[
    &[(3, 3), (4, 3), (4, 4), (3, 4), (3, 3)],
    &[(3, 10), (4, 10), (4, 11), (3, 11), (3, 10)],
];

// ';'
static GLYPH_SEMICOLON: &[&[(i8, i8)]] = &[
    &[(3, 3), (4, 3), (4, 4), (3, 4), (3, 3)],
    &[(3, 10), (3, 12), (2, 13)],
];

// '<'
static GLYPH_LESS: &[&[(i8, i8)]] = &[&[(8, 2), (1, 6), (8, 10)]];

// '='
static GLYPH_EQ: &[&[(i8, i8)]] = &[&[(1, 4), (8, 4)], &[(1, 8), (8, 8)]];

// '>'
static GLYPH_GT: &[&[(i8, i8)]] = &[&[(1, 2), (8, 6), (1, 10)]];

// '?'
static GLYPH_QUESTION: &[&[(i8, i8)]] = &[
    &[
        (1, 2),
        (2, 1),
        (3, 0),
        (5, 0),
        (7, 1),
        (8, 3),
        (8, 5),
        (6, 7),
        (4, 8),
        (4, 10),
    ],
    &[(4, 11), (4, 12)],
];

// '@'
static GLYPH_AT: &[&[(i8, i8)]] = &[&[
    (6, 5),
    (5, 4),
    (3, 4),
    (2, 5),
    (2, 7),
    (3, 8),
    (5, 8),
    (6, 7),
    (6, 4),
    (7, 4),
    (8, 5),
    (8, 8),
    (7, 10),
    (5, 11),
    (3, 11),
    (1, 10),
    (0, 8),
    (0, 5),
    (1, 3),
    (3, 1),
    (5, 0),
    (7, 0),
    (9, 1),
    (10, 3),
]];

// '['
static GLYPH_LBRACKET: &[&[(i8, i8)]] = &[&[(4, 0), (2, 0), (2, 12), (4, 12)]];

// '\'
static GLYPH_BACKSLASH: &[&[(i8, i8)]] = &[&[(0, 0), (8, 12)]];

// ']'
static GLYPH_RBRACKET: &[&[(i8, i8)]] = &[&[(2, 0), (4, 0), (4, 12), (2, 12)]];

// '^'
static GLYPH_CARET: &[&[(i8, i8)]] = &[&[(2, 5), (4, 0), (6, 5)]];

// '_'
static GLYPH_UNDERSCORE: &[&[(i8, i8)]] = &[&[(0, 12), (9, 12)]];

// '`'
static GLYPH_BACKTICK: &[&[(i8, i8)]] = &[&[(3, 0), (4, 1), (4, 3)]];

// '{' '|' '}'
static GLYPH_LBRACE: &[&[(i8, i8)]] =
    &[&[(5, 0), (4, 1), (4, 5), (3, 6), (4, 7), (4, 11), (5, 12)]];
static GLYPH_PIPE: &[&[(i8, i8)]] = &[&[(4, 0), (4, 12)]];
static GLYPH_RBRACE: &[&[(i8, i8)]] =
    &[&[(3, 0), (4, 1), (4, 5), (5, 6), (4, 7), (4, 11), (3, 12)]];

// '~'
static GLYPH_TILDE: &[&[(i8, i8)]] = &[&[
    (1, 6),
    (1, 4),
    (2, 3),
    (4, 3),
    (6, 5),
    (7, 5),
    (8, 4),
    (8, 6),
]];

// ── Lowercase letters a–z ─────────────────────────────────────────────────────

static GLYPH_LA: &[&[(i8, i8)]] = &[
    &[(8, 4), (8, 12)],
    &[
        (8, 6),
        (7, 5),
        (5, 4),
        (3, 4),
        (1, 6),
        (1, 9),
        (3, 12),
        (5, 12),
        (7, 11),
        (8, 9),
    ],
];

static GLYPH_LB: &[&[(i8, i8)]] = &[
    &[(1, 0), (1, 12)],
    &[
        (1, 4),
        (2, 5),
        (4, 5),
        (6, 6),
        (7, 8),
        (7, 10),
        (6, 12),
        (4, 12),
        (2, 11),
        (1, 9),
    ],
];

static GLYPH_LC: &[&[(i8, i8)]] = &[&[
    (8, 5),
    (7, 4),
    (5, 4),
    (3, 5),
    (1, 7),
    (1, 9),
    (3, 12),
    (5, 12),
    (7, 11),
    (8, 10),
]];

static GLYPH_LD: &[&[(i8, i8)]] = &[
    &[(8, 0), (8, 12)],
    &[
        (8, 6),
        (7, 5),
        (5, 4),
        (3, 4),
        (1, 6),
        (1, 9),
        (3, 12),
        (5, 12),
        (7, 11),
        (8, 9),
    ],
];

static GLYPH_LE: &[&[(i8, i8)]] = &[&[
    (1, 8),
    (8, 8),
    (8, 6),
    (7, 5),
    (5, 4),
    (3, 4),
    (1, 6),
    (1, 9),
    (3, 12),
    (5, 12),
    (7, 11),
    (8, 10),
]];

static GLYPH_LF: &[&[(i8, i8)]] = &[&[(5, 0), (4, 0), (3, 1), (3, 12)], &[(1, 5), (7, 5)]];

static GLYPH_LG: &[&[(i8, i8)]] = &[
    &[(8, 4), (8, 15), (7, 16), (5, 16), (3, 15)],
    &[
        (8, 6),
        (7, 5),
        (5, 4),
        (3, 4),
        (1, 6),
        (1, 9),
        (3, 12),
        (5, 12),
        (7, 11),
        (8, 9),
    ],
];

static GLYPH_LH: &[&[(i8, i8)]] = &[
    &[(1, 0), (1, 12)],
    &[(1, 5), (3, 4), (5, 4), (7, 5), (8, 7), (8, 12)],
];

static GLYPH_LI: &[&[(i8, i8)]] = &[&[(3, 4), (3, 12)], &[(3, 1), (3, 2)]];

static GLYPH_LJ: &[&[(i8, i8)]] = &[
    &[(5, 4), (5, 14), (4, 16), (2, 16), (1, 14)],
    &[(5, 1), (5, 2)],
];

static GLYPH_LK: &[&[(i8, i8)]] = &[&[(1, 0), (1, 12)], &[(7, 4), (1, 8)], &[(3, 7), (7, 12)]];

static GLYPH_LL: &[&[(i8, i8)]] = &[&[(3, 0), (3, 12)]];

static GLYPH_LM: &[&[(i8, i8)]] = &[&[
    (1, 12),
    (1, 4),
    (4, 4),
    (5, 5),
    (5, 12),
    (5, 5),
    (8, 4),
    (9, 5),
    (9, 12),
]];

static GLYPH_LN: &[&[(i8, i8)]] = &[&[(1, 12), (1, 4), (7, 11), (7, 4)]];

static GLYPH_LO: &[&[(i8, i8)]] = &[&[
    (5, 4),
    (3, 4),
    (1, 6),
    (1, 9),
    (3, 12),
    (5, 12),
    (7, 10),
    (7, 7),
    (5, 5),
    (3, 5),
]];

static GLYPH_LP: &[&[(i8, i8)]] = &[
    &[(1, 4), (1, 16)],
    &[
        (1, 4),
        (3, 4),
        (5, 5),
        (7, 7),
        (7, 9),
        (5, 12),
        (3, 12),
        (1, 10),
    ],
];

static GLYPH_LQ: &[&[(i8, i8)]] = &[
    &[(8, 4), (8, 16)],
    &[
        (8, 6),
        (7, 5),
        (5, 4),
        (3, 4),
        (1, 6),
        (1, 9),
        (3, 12),
        (5, 12),
        (7, 11),
        (8, 9),
    ],
];

static GLYPH_LR: &[&[(i8, i8)]] = &[
    &[(1, 4), (1, 12)],
    &[(1, 6), (2, 5), (4, 4), (6, 4), (8, 5)],
];

static GLYPH_LS: &[&[(i8, i8)]] = &[&[
    (8, 5),
    (7, 4),
    (5, 4),
    (3, 5),
    (3, 7),
    (5, 8),
    (7, 9),
    (8, 10),
    (8, 11),
    (7, 12),
    (5, 12),
    (3, 11),
]];

static GLYPH_LT: &[&[(i8, i8)]] = &[&[(3, 0), (3, 10), (4, 12), (6, 12)], &[(1, 5), (6, 5)]];

static GLYPH_LU: &[&[(i8, i8)]] = &[&[(1, 4), (1, 10), (2, 11), (4, 12), (6, 12), (8, 11), (8, 4)]];

static GLYPH_LV: &[&[(i8, i8)]] = &[&[(1, 4), (4, 12), (7, 4)]];

static GLYPH_LW: &[&[(i8, i8)]] = &[&[(1, 4), (2, 12), (4, 8), (6, 12), (7, 4)]];

static GLYPH_LX: &[&[(i8, i8)]] = &[&[(1, 4), (7, 12)], &[(7, 4), (1, 12)]];

static GLYPH_LY: &[&[(i8, i8)]] = &[
    &[(1, 4), (4, 10)],
    &[(7, 4), (4, 10), (4, 14), (3, 16), (1, 16)],
];

static GLYPH_LZ: &[&[(i8, i8)]] = &[&[(1, 4), (7, 4), (1, 12), (7, 12)]];

// ── Glyph lookup ──────────────────────────────────────────────────────────────

fn glyph_for_char(c: char) -> Option<HersheyGlyph> {
    let (advance, strokes): (i8, &'static [&'static [(i8, i8)]]) = match c {
        // ── Punctuation / special ────────────────────────────────────────────
        ' ' => (6, GLYPH_SPACE),
        '!' => (5, GLYPH_EXCLAIM),
        '"' => (8, GLYPH_DQUOTE),
        '#' => (11, GLYPH_HASH),
        '$' => (11, GLYPH_DOLLAR),
        '%' => (11, GLYPH_PERCENT),
        '&' => (12, GLYPH_AMPERSAND),
        '\'' => (5, GLYPH_SQUOTE),
        '(' => (7, GLYPH_LPAREN),
        ')' => (7, GLYPH_RPAREN),
        '*' => (9, GLYPH_STAR),
        '+' => (9, GLYPH_PLUS),
        ',' => (5, GLYPH_COMMA),
        '-' => (9, GLYPH_MINUS),
        '.' => (5, GLYPH_PERIOD),
        '/' => (9, GLYPH_SLASH),
        // ── Digits ───────────────────────────────────────────────────────────
        '0' => (11, GLYPH_0),
        '1' => (8, GLYPH_1),
        '2' => (11, GLYPH_2),
        '3' => (11, GLYPH_3),
        '4' => (11, GLYPH_4),
        '5' => (11, GLYPH_5),
        '6' => (11, GLYPH_6),
        '7' => (11, GLYPH_7),
        '8' => (11, GLYPH_8),
        '9' => (11, GLYPH_9),
        // ── More punctuation ─────────────────────────────────────────────────
        ':' => (5, GLYPH_COLON),
        ';' => (5, GLYPH_SEMICOLON),
        '<' => (9, GLYPH_LESS),
        '=' => (9, GLYPH_EQ),
        '>' => (9, GLYPH_GT),
        '?' => (9, GLYPH_QUESTION),
        '@' => (13, GLYPH_AT),
        // ── Uppercase A–Z ────────────────────────────────────────────────────
        'A' => (11, GLYPH_A),
        'B' => (11, GLYPH_B),
        'C' => (11, GLYPH_C),
        'D' => (11, GLYPH_D),
        'E' => (11, GLYPH_E),
        'F' => (11, GLYPH_F),
        'G' => (11, GLYPH_G),
        'H' => (11, GLYPH_H),
        'I' => (9, GLYPH_I),
        'J' => (11, GLYPH_J),
        'K' => (11, GLYPH_K),
        'L' => (11, GLYPH_L),
        'M' => (11, GLYPH_M),
        'N' => (11, GLYPH_N),
        'O' => (11, GLYPH_O),
        'P' => (11, GLYPH_P),
        'Q' => (11, GLYPH_Q),
        'R' => (11, GLYPH_R),
        'S' => (11, GLYPH_S),
        'T' => (11, GLYPH_T),
        'U' => (11, GLYPH_U),
        'V' => (11, GLYPH_V),
        'W' => (11, GLYPH_W),
        'X' => (11, GLYPH_X),
        'Y' => (11, GLYPH_Y),
        'Z' => (11, GLYPH_Z),
        // ── More punctuation ─────────────────────────────────────────────────
        '[' => (7, GLYPH_LBRACKET),
        '\\' => (9, GLYPH_BACKSLASH),
        ']' => (7, GLYPH_RBRACKET),
        '^' => (9, GLYPH_CARET),
        '_' => (11, GLYPH_UNDERSCORE),
        '`' => (6, GLYPH_BACKTICK),
        // ── Lowercase a–z ────────────────────────────────────────────────────
        'a' => (11, GLYPH_LA),
        'b' => (11, GLYPH_LB),
        'c' => (11, GLYPH_LC),
        'd' => (11, GLYPH_LD),
        'e' => (11, GLYPH_LE),
        'f' => (9, GLYPH_LF),
        'g' => (11, GLYPH_LG),
        'h' => (11, GLYPH_LH),
        'i' => (5, GLYPH_LI),
        'j' => (7, GLYPH_LJ),
        'k' => (11, GLYPH_LK),
        'l' => (5, GLYPH_LL),
        'm' => (14, GLYPH_LM),
        'n' => (11, GLYPH_LN),
        'o' => (11, GLYPH_LO),
        'p' => (11, GLYPH_LP),
        'q' => (11, GLYPH_LQ),
        'r' => (9, GLYPH_LR),
        's' => (11, GLYPH_LS),
        't' => (9, GLYPH_LT),
        'u' => (11, GLYPH_LU),
        'v' => (11, GLYPH_LV),
        'w' => (11, GLYPH_LW),
        'x' => (11, GLYPH_LX),
        'y' => (11, GLYPH_LY),
        'z' => (11, GLYPH_LZ),
        // ── Final punctuation ─────────────────────────────────────────────────
        '{' => (7, GLYPH_LBRACE),
        '|' => (5, GLYPH_PIPE),
        '}' => (7, GLYPH_RBRACE),
        '~' => (11, GLYPH_TILDE),
        _ => return None,
    };
    Some(HersheyGlyph { advance, strokes })
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Render text using Hershey stroke vector outlines.
///
/// All printable ASCII characters (space through `~`) are supported.
///
/// Accepted `font_face` values (matching `crate::constants::font`):
/// - `0` (`FONT_HERSHEY_SIMPLEX`) — single-stroke simplex (full implementation)
/// - `1` (`FONT_HERSHEY_PLAIN`) — same glyph table, smaller scale intended
/// - `2` (`FONT_HERSHEY_DUPLEX`) — same glyph table, double-stroke by caller
/// - `3` (`FONT_HERSHEY_COMPLEX`) — same glyph table, complex variant simplified
/// - `4` (`FONT_HERSHEY_TRIPLEX`) — same glyph table, triplex variant simplified
///
/// All accepted values route through the same simplex glyph data; differences
/// in visual style (weight, extra strokes) are the caller's responsibility via
/// `thickness`.  Font face values outside 0–4 return
/// [`Cv2Error::UnsupportedFlag`].
///
/// `org` is the bottom-left origin of the text baseline (matching the OpenCV
/// `putText` convention).  Glyphs are scaled by `font_scale`; `thickness`
/// controls the line width of each stroke.
///
/// # Errors
/// Returns [`Cv2Error::UnsupportedFlag`] when `font_face` is outside the
/// accepted range 0–4.
pub fn put_text_hershey(
    img: &mut Mat,
    text: &str,
    org: Point,
    font_face: i32,
    font_scale: f64,
    color: Scalar,
    thickness: i32,
) -> Cv2Result<()> {
    // Accept font faces 0–4; all route through the simplex glyph table.
    if !(0..=4).contains(&font_face) {
        return Err(Cv2Error::UnsupportedFlag {
            name: "put_text_hershey: unsupported font_face",
            value: font_face,
        });
    }

    let scale = font_scale;
    let mut x_offset = 0.0f64;

    for c in text.chars() {
        let Some(glyph) = glyph_for_char(c) else {
            // Unknown char: advance by a default width without drawing.
            x_offset += 11.0 * scale;
            continue;
        };

        for stroke in glyph.strokes {
            let n = stroke.len();
            if n < 2 {
                continue;
            }
            for i in 0..n - 1 {
                let (sx1, sy1) = stroke[i];
                let (sx2, sy2) = stroke[i + 1];
                // Transform glyph coordinates to image coordinates.
                // y=12 is the baseline in glyph space; org.y is the baseline
                // in image space, so glyph rows above the baseline map to
                // smaller image y values.
                let x1 = (org.x as f64 + x_offset + sx1 as f64 * scale) as i32;
                let y1 = (org.y as f64 - (12.0 - sy1 as f64) * scale) as i32;
                let x2 = (org.x as f64 + x_offset + sx2 as f64 * scale) as i32;
                let y2 = (org.y as f64 - (12.0 - sy2 as f64) * scale) as i32;
                line(
                    img,
                    Point { x: x1, y: y1 },
                    Point { x: x2, y: y2 },
                    color,
                    thickness,
                )?;
            }
        }

        x_offset += glyph.advance as f64 * scale;
    }

    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        constants::{
            FONT_HERSHEY_COMPLEX, FONT_HERSHEY_DUPLEX, FONT_HERSHEY_PLAIN, FONT_HERSHEY_SIMPLEX,
            FONT_HERSHEY_TRIPLEX,
        },
        mat::{Mat, MatType, Point, Scalar},
    };

    fn blank(w: usize, h: usize) -> Mat {
        Mat::new(h, w, MatType::CV_8UC3)
    }

    fn nonzero(m: &Mat) -> usize {
        m.data.iter().filter(|&&v| v > 0).count()
    }

    #[test]
    fn test_digit_produces_pixels() {
        let mut img = blank(200, 60);
        put_text_hershey(
            &mut img,
            "0",
            Point { x: 10, y: 50 },
            FONT_HERSHEY_SIMPLEX,
            2.0,
            Scalar(255.0, 255.0, 255.0, 255.0),
            1,
        )
        .unwrap();
        assert!(
            nonzero(&img) > 0,
            "digit '0' should produce non-zero pixels"
        );
    }

    #[test]
    fn test_letter_produces_pixels() {
        let mut img = blank(200, 60);
        put_text_hershey(
            &mut img,
            "A",
            Point { x: 10, y: 50 },
            FONT_HERSHEY_SIMPLEX,
            2.0,
            Scalar(255.0, 255.0, 255.0, 255.0),
            1,
        )
        .unwrap();
        assert!(
            nonzero(&img) > 0,
            "letter 'A' should produce non-zero pixels"
        );
    }

    #[test]
    fn test_two_chars_more_pixels_than_one() {
        let mut img1 = blank(300, 60);
        let mut img2 = blank(300, 60);
        put_text_hershey(
            &mut img1,
            "A",
            Point { x: 0, y: 50 },
            FONT_HERSHEY_SIMPLEX,
            2.0,
            Scalar(255.0, 255.0, 255.0, 255.0),
            1,
        )
        .unwrap();
        put_text_hershey(
            &mut img2,
            "AN",
            Point { x: 0, y: 50 },
            FONT_HERSHEY_SIMPLEX,
            2.0,
            Scalar(255.0, 255.0, 255.0, 255.0),
            1,
        )
        .unwrap();
        assert!(
            nonzero(&img2) > nonzero(&img1),
            "two chars should produce more pixels than one"
        );
    }

    #[test]
    fn test_unsupported_font_errors() {
        let mut img = blank(100, 50);
        // Font face 99 is outside 0–4 range.
        let res = put_text_hershey(
            &mut img,
            "A",
            Point { x: 0, y: 20 },
            99,
            1.0,
            Scalar(255.0, 255.0, 255.0, 255.0),
            1,
        );
        assert!(res.is_err());
        // Font face 5 is also outside the accepted range.
        let res2 = put_text_hershey(
            &mut img,
            "A",
            Point { x: 0, y: 20 },
            5,
            1.0,
            Scalar(255.0, 255.0, 255.0, 255.0),
            1,
        );
        assert!(res2.is_err());
    }

    // ── New tests for Slice 6 ────────────────────────────────────────────────

    /// All printable ASCII chars (0x20–0x7E) must return Some from glyph_for_char.
    #[test]
    fn test_all_printable_ascii_glyphs() {
        for code in 0x20u8..=0x7Eu8 {
            let c = code as char;
            assert!(
                glyph_for_char(c).is_some(),
                "glyph_for_char({:?}) returned None",
                c
            );
        }
    }

    /// Non-space visible chars must have at least one stroke.
    #[test]
    fn test_glyph_strokes_non_empty_for_visible() {
        for code in 0x21u8..=0x7Eu8 {
            let c = code as char;
            let glyph = glyph_for_char(c).unwrap_or_else(|| panic!("missing glyph for {:?}", c));
            assert!(!glyph.strokes.is_empty(), "char {:?} has zero strokes", c);
        }
    }

    /// Font faces 0–4 must all succeed (they all use the same glyph table).
    #[test]
    fn test_put_text_all_font_faces() {
        let faces = [
            FONT_HERSHEY_SIMPLEX,
            FONT_HERSHEY_PLAIN,
            FONT_HERSHEY_DUPLEX,
            FONT_HERSHEY_COMPLEX,
            FONT_HERSHEY_TRIPLEX,
        ];
        for &face in &faces {
            let mut img = blank(300, 80);
            let res = put_text_hershey(
                &mut img,
                "Hello",
                Point { x: 5, y: 60 },
                face,
                1.5,
                Scalar(200.0, 200.0, 200.0, 255.0),
                1,
            );
            assert!(res.is_ok(), "font_face {} returned error: {:?}", face, res);
        }
    }
}

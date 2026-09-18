//! Expression serialization for TLExpr and EinsumGraph.
//!
//! Provides custom S-expression text format and compact binary format for
//! serializing/deserializing logical expressions and computation graphs.
//! This enables saving/loading compiled expressions for caching, transfer, and debugging.

mod binary;
mod sexpr;

use crate::TLExpr;

pub use binary::{from_binary, graph_from_binary, graph_to_binary, to_binary};
pub use sexpr::{from_sexpr, to_sexpr};

// ============================================================================
// Error type
// ============================================================================

/// Error type for serialization operations.
#[derive(Debug, Clone)]
pub enum ExprSerializeError {
    /// I/O related error
    IoError(String),
    /// Format/parsing error
    FormatError(String),
    /// Unknown variant tag encountered
    UnknownVariant(String),
    /// Binary format version mismatch
    VersionMismatch { expected: u32, got: u32 },
    /// Invalid magic bytes in binary header
    InvalidMagic,
    /// Input was truncated (unexpected end)
    TruncatedInput,
    /// UTF-8 decoding error
    Utf8Error(String),
}

impl std::fmt::Display for ExprSerializeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(msg) => write!(f, "IO error: {msg}"),
            Self::FormatError(msg) => write!(f, "Format error: {msg}"),
            Self::UnknownVariant(v) => write!(f, "Unknown variant: {v}"),
            Self::VersionMismatch { expected, got } => {
                write!(f, "Version mismatch: expected {expected}, got {got}")
            }
            Self::InvalidMagic => write!(f, "Invalid magic bytes"),
            Self::TruncatedInput => write!(f, "Truncated input"),
            Self::Utf8Error(msg) => write!(f, "UTF-8 error: {msg}"),
        }
    }
}

impl std::error::Error for ExprSerializeError {}

// ============================================================================
// ExprFormat
// ============================================================================

/// Serialization format selector.
#[derive(Debug, Clone, PartialEq)]
pub enum ExprFormat {
    /// S-expression text format: `(And (Not (Var "x")) (Pred "p" (Var "y")))`
    SExpr,
    /// Compact binary format with magic + version header
    Binary,
}

// ============================================================================
// Constants (shared between binary and sexpr modules)
// ============================================================================

/// Binary format magic bytes for single expressions: "TLEX"
pub(crate) const TLEX_MAGIC: [u8; 4] = [0x54, 0x4C, 0x45, 0x58];
/// Binary format magic bytes for batch expressions: "TLBX"
pub(crate) const TLBX_MAGIC: [u8; 4] = [0x54, 0x4C, 0x42, 0x58];
/// Binary format magic bytes for graphs: "TLGR"
pub(crate) const TLGR_MAGIC: [u8; 4] = [0x54, 0x4C, 0x47, 0x52];
/// Current binary format version
pub(crate) const FORMAT_VER: u32 = 1;

// ============================================================================
// Tag assignments for TLExpr variants (u8)
// ============================================================================

pub(crate) const TAG_PRED: u8 = 0;
pub(crate) const TAG_AND: u8 = 1;
pub(crate) const TAG_OR: u8 = 2;
pub(crate) const TAG_NOT: u8 = 3;
pub(crate) const TAG_EXISTS: u8 = 4;
pub(crate) const TAG_FORALL: u8 = 5;
pub(crate) const TAG_IMPLY: u8 = 6;
pub(crate) const TAG_SCORE: u8 = 7;
pub(crate) const TAG_ADD: u8 = 8;
pub(crate) const TAG_SUB: u8 = 9;
pub(crate) const TAG_MUL: u8 = 10;
pub(crate) const TAG_DIV: u8 = 11;
pub(crate) const TAG_POW: u8 = 12;
pub(crate) const TAG_MOD: u8 = 13;
pub(crate) const TAG_MIN: u8 = 14;
pub(crate) const TAG_MAX: u8 = 15;
pub(crate) const TAG_ABS: u8 = 16;
pub(crate) const TAG_FLOOR: u8 = 17;
pub(crate) const TAG_CEIL: u8 = 18;
pub(crate) const TAG_ROUND: u8 = 19;
pub(crate) const TAG_SQRT: u8 = 20;
pub(crate) const TAG_EXP: u8 = 21;
pub(crate) const TAG_LOG: u8 = 22;
pub(crate) const TAG_SIN: u8 = 23;
pub(crate) const TAG_COS: u8 = 24;
pub(crate) const TAG_TAN: u8 = 25;
pub(crate) const TAG_EQ: u8 = 26;
pub(crate) const TAG_LT: u8 = 27;
pub(crate) const TAG_GT: u8 = 28;
pub(crate) const TAG_LTE: u8 = 29;
pub(crate) const TAG_GTE: u8 = 30;
pub(crate) const TAG_IF_THEN_ELSE: u8 = 31;
pub(crate) const TAG_CONSTANT: u8 = 32;
pub(crate) const TAG_AGGREGATE: u8 = 33;
pub(crate) const TAG_LET: u8 = 34;
pub(crate) const TAG_BOX: u8 = 35;
pub(crate) const TAG_DIAMOND: u8 = 36;
pub(crate) const TAG_NEXT: u8 = 37;
pub(crate) const TAG_EVENTUALLY: u8 = 38;
pub(crate) const TAG_ALWAYS: u8 = 39;
pub(crate) const TAG_UNTIL: u8 = 40;
pub(crate) const TAG_TNORM: u8 = 41;
pub(crate) const TAG_TCONORM: u8 = 42;
pub(crate) const TAG_FUZZY_NOT: u8 = 43;
pub(crate) const TAG_FUZZY_IMPLICATION: u8 = 44;
pub(crate) const TAG_SOFT_EXISTS: u8 = 45;
pub(crate) const TAG_SOFT_FORALL: u8 = 46;
pub(crate) const TAG_WEIGHTED_RULE: u8 = 47;
pub(crate) const TAG_PROBABILISTIC_CHOICE: u8 = 48;
pub(crate) const TAG_RELEASE: u8 = 49;
pub(crate) const TAG_WEAK_UNTIL: u8 = 50;
pub(crate) const TAG_STRONG_RELEASE: u8 = 51;
pub(crate) const TAG_LAMBDA: u8 = 52;
pub(crate) const TAG_APPLY: u8 = 53;
pub(crate) const TAG_SET_MEMBERSHIP: u8 = 54;
pub(crate) const TAG_SET_UNION: u8 = 55;
pub(crate) const TAG_SET_INTERSECTION: u8 = 56;
pub(crate) const TAG_SET_DIFFERENCE: u8 = 57;
pub(crate) const TAG_SET_CARDINALITY: u8 = 58;
pub(crate) const TAG_EMPTY_SET: u8 = 59;
pub(crate) const TAG_SET_COMPREHENSION: u8 = 60;
pub(crate) const TAG_COUNTING_EXISTS: u8 = 61;
pub(crate) const TAG_COUNTING_FORALL: u8 = 62;
pub(crate) const TAG_EXACT_COUNT: u8 = 63;
pub(crate) const TAG_MAJORITY: u8 = 64;
pub(crate) const TAG_LEAST_FIXPOINT: u8 = 65;
pub(crate) const TAG_GREATEST_FIXPOINT: u8 = 66;
pub(crate) const TAG_NOMINAL: u8 = 67;
pub(crate) const TAG_AT: u8 = 68;
pub(crate) const TAG_SOMEWHERE: u8 = 69;
pub(crate) const TAG_EVERYWHERE: u8 = 70;
pub(crate) const TAG_ALL_DIFFERENT: u8 = 71;
pub(crate) const TAG_GLOBAL_CARDINALITY: u8 = 72;
pub(crate) const TAG_ABDUCIBLE: u8 = 73;
pub(crate) const TAG_EXPLAIN: u8 = 74;
pub(crate) const TAG_SYMBOL_LITERAL: u8 = 75;
pub(crate) const TAG_MATCH: u8 = 76;
pub(crate) const TAG_PATTERN_CONST_SYMBOL: u8 = 0;
pub(crate) const TAG_PATTERN_CONST_NUMBER: u8 = 1;
pub(crate) const TAG_PATTERN_WILDCARD: u8 = 2;

// Term tags
pub(crate) const TERM_TAG_VAR: u8 = 0;
pub(crate) const TERM_TAG_CONST: u8 = 1;
pub(crate) const TERM_TAG_TYPED: u8 = 2;

// AggregateOp tags
pub(crate) const AGG_COUNT: u8 = 0;
pub(crate) const AGG_SUM: u8 = 1;
pub(crate) const AGG_AVERAGE: u8 = 2;
pub(crate) const AGG_MAX: u8 = 3;
pub(crate) const AGG_MIN: u8 = 4;
pub(crate) const AGG_PRODUCT: u8 = 5;
pub(crate) const AGG_ANY: u8 = 6;
pub(crate) const AGG_ALL: u8 = 7;

// TNormKind tags
pub(crate) const TNORM_MINIMUM: u8 = 0;
pub(crate) const TNORM_PRODUCT: u8 = 1;
pub(crate) const TNORM_LUKASIEWICZ: u8 = 2;
pub(crate) const TNORM_DRASTIC: u8 = 3;
pub(crate) const TNORM_NILPOTENT_MINIMUM: u8 = 4;
pub(crate) const TNORM_HAMACHER: u8 = 5;

// TCoNormKind tags
pub(crate) const TCONORM_MAXIMUM: u8 = 0;
pub(crate) const TCONORM_PROBABILISTIC_SUM: u8 = 1;
pub(crate) const TCONORM_BOUNDED_SUM: u8 = 2;
pub(crate) const TCONORM_DRASTIC: u8 = 3;
pub(crate) const TCONORM_NILPOTENT_MAXIMUM: u8 = 4;
pub(crate) const TCONORM_HAMACHER: u8 = 5;

// FuzzyNegationKind tags
pub(crate) const FNEG_STANDARD: u8 = 0;
pub(crate) const FNEG_SUGENO: u8 = 1;
pub(crate) const FNEG_YAGER: u8 = 2;

// FuzzyImplicationKind tags
pub(crate) const FIMP_GODEL: u8 = 0;
pub(crate) const FIMP_LUKASIEWICZ: u8 = 1;
pub(crate) const FIMP_REICHENBACH: u8 = 2;
pub(crate) const FIMP_KLEENE_DIENES: u8 = 3;
pub(crate) const FIMP_RESCHER: u8 = 4;
pub(crate) const FIMP_GOGUEN: u8 = 5;

// OpType tags for graph serialization
pub(crate) const OP_EINSUM: u8 = 0;
pub(crate) const OP_ELEM_UNARY: u8 = 1;
pub(crate) const OP_ELEM_BINARY: u8 = 2;
pub(crate) const OP_REDUCE: u8 = 3;

// ============================================================================
// Fingerprint (FNV-1a)
// ============================================================================

/// Compute a 64-bit FNV-1a hash/fingerprint of a `TLExpr` for caching.
pub fn expr_fingerprint(expr: &TLExpr) -> u64 {
    let bin = to_binary(expr);
    fnv1a_hash(&bin[8..]) // skip magic + version, hash only payload
}

/// FNV-1a 64-bit hash implementation.
fn fnv1a_hash(data: &[u8]) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x00000100000001B3;
    let mut hash = FNV_OFFSET_BASIS;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Compare two serialized binary forms for equality without deserializing.
pub fn binary_equal(a: &[u8], b: &[u8]) -> bool {
    a == b
}

// ============================================================================
// SerializationStats
// ============================================================================

/// Statistics about the serialized form of a `TLExpr`.
#[derive(Debug, Clone)]
pub struct SerializationStats {
    /// Size of the S-expression representation in bytes
    pub sexpr_bytes: usize,
    /// Size of the binary representation in bytes
    pub binary_bytes: usize,
    /// Compression ratio: sexpr_bytes / binary_bytes
    pub compression_ratio: f64,
    /// Number of AST nodes in the expression
    pub node_count: usize,
    /// Maximum nesting depth
    pub max_depth: usize,
}

impl SerializationStats {
    /// Compute serialization statistics for a `TLExpr`.
    pub fn compute(expr: &TLExpr) -> Self {
        let sexpr_str = to_sexpr(expr);
        let bin = to_binary(expr);
        let sexpr_bytes = sexpr_str.len();
        let binary_bytes = bin.len();
        let compression_ratio = if binary_bytes > 0 {
            sexpr_bytes as f64 / binary_bytes as f64
        } else {
            0.0
        };
        let node_count = count_nodes(expr);
        let max_depth = compute_depth(expr);
        Self {
            sexpr_bytes,
            binary_bytes,
            compression_ratio,
            node_count,
            max_depth,
        }
    }

    /// Return a human-readable summary string.
    pub fn summary(&self) -> String {
        format!(
            "sexpr={} bytes, binary={} bytes, ratio={:.2}, nodes={}, depth={}",
            self.sexpr_bytes,
            self.binary_bytes,
            self.compression_ratio,
            self.node_count,
            self.max_depth
        )
    }
}

fn count_nodes(expr: &TLExpr) -> usize {
    let mut count = 1usize;
    visit_children(expr, &mut |child| count += count_nodes(child));
    count
}

fn compute_depth(expr: &TLExpr) -> usize {
    let mut max_child_depth = 0usize;
    visit_children(expr, &mut |child| {
        let d = compute_depth(child);
        if d > max_child_depth {
            max_child_depth = d;
        }
    });
    1 + max_child_depth
}

/// Visit immediate children of a `TLExpr`.
fn visit_children(expr: &TLExpr, f: &mut impl FnMut(&TLExpr)) {
    match expr {
        TLExpr::Pred { .. }
        | TLExpr::Constant(_)
        | TLExpr::EmptySet
        | TLExpr::Nominal { .. }
        | TLExpr::AllDifferent { .. }
        | TLExpr::Abducible { .. } => {}

        TLExpr::Not(e)
        | TLExpr::Score(e)
        | TLExpr::Abs(e)
        | TLExpr::Floor(e)
        | TLExpr::Ceil(e)
        | TLExpr::Round(e)
        | TLExpr::Sqrt(e)
        | TLExpr::Exp(e)
        | TLExpr::Log(e)
        | TLExpr::Sin(e)
        | TLExpr::Cos(e)
        | TLExpr::Tan(e)
        | TLExpr::Box(e)
        | TLExpr::Diamond(e)
        | TLExpr::Next(e)
        | TLExpr::Eventually(e)
        | TLExpr::Always(e) => f(e),

        TLExpr::And(a, b)
        | TLExpr::Or(a, b)
        | TLExpr::Imply(a, b)
        | TLExpr::Add(a, b)
        | TLExpr::Sub(a, b)
        | TLExpr::Mul(a, b)
        | TLExpr::Div(a, b)
        | TLExpr::Pow(a, b)
        | TLExpr::Mod(a, b)
        | TLExpr::Min(a, b)
        | TLExpr::Max(a, b)
        | TLExpr::Eq(a, b)
        | TLExpr::Lt(a, b)
        | TLExpr::Gt(a, b)
        | TLExpr::Lte(a, b)
        | TLExpr::Gte(a, b) => {
            f(a);
            f(b);
        }

        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => {
            f(condition);
            f(then_branch);
            f(else_branch);
        }

        TLExpr::Exists { body, .. }
        | TLExpr::ForAll { body, .. }
        | TLExpr::Majority { body, .. }
        | TLExpr::SetComprehension {
            condition: body, ..
        } => f(body),

        TLExpr::Aggregate { body, .. } => f(body),

        TLExpr::Let { value, body, .. } => {
            f(value);
            f(body);
        }

        TLExpr::Until { before, after } | TLExpr::WeakUntil { before, after } => {
            f(before);
            f(after);
        }

        TLExpr::Release { released, releaser } | TLExpr::StrongRelease { released, releaser } => {
            f(released);
            f(releaser);
        }

        TLExpr::TNorm { left, right, .. } | TLExpr::TCoNorm { left, right, .. } => {
            f(left);
            f(right);
        }

        TLExpr::FuzzyNot { expr: e, .. } => f(e),
        TLExpr::FuzzyImplication {
            premise,
            conclusion,
            ..
        } => {
            f(premise);
            f(conclusion);
        }

        TLExpr::SoftExists { body, .. } | TLExpr::SoftForAll { body, .. } => f(body),
        TLExpr::WeightedRule { rule, .. } => f(rule),

        TLExpr::ProbabilisticChoice { alternatives } => {
            for (_, alt_expr) in alternatives {
                f(alt_expr);
            }
        }

        TLExpr::Lambda { body, .. }
        | TLExpr::LeastFixpoint { body, .. }
        | TLExpr::GreatestFixpoint { body, .. } => f(body),

        TLExpr::Apply { function, argument } => {
            f(function);
            f(argument);
        }

        TLExpr::SetMembership { element, set } => {
            f(element);
            f(set);
        }
        TLExpr::SetUnion { left, right }
        | TLExpr::SetIntersection { left, right }
        | TLExpr::SetDifference { left, right } => {
            f(left);
            f(right);
        }

        TLExpr::SetCardinality { set } => f(set),

        TLExpr::CountingExists { body, .. }
        | TLExpr::CountingForAll { body, .. }
        | TLExpr::ExactCount { body, .. } => f(body),

        TLExpr::At { formula, .. }
        | TLExpr::Somewhere { formula }
        | TLExpr::Everywhere { formula }
        | TLExpr::Explain { formula } => f(formula),

        TLExpr::GlobalCardinality { values, .. } => {
            for v in values {
                f(v);
            }
        }

        TLExpr::SymbolLiteral(_) => {}

        TLExpr::Match { scrutinee, arms } => {
            f(scrutinee);
            for (_, body) in arms {
                f(body);
            }
        }
    }
}

// ============================================================================
// Batch serialization
// ============================================================================

/// Serialize multiple expressions efficiently into a single binary blob.
pub fn batch_to_binary(exprs: &[TLExpr]) -> Vec<u8> {
    let mut w = binary::BinWriter::new();
    w.write_magic(&TLBX_MAGIC);
    w.write_u32(FORMAT_VER);
    w.write_u32(exprs.len() as u32);
    for expr in exprs {
        binary::write_expr_bin(expr, &mut w);
    }
    w.into_bytes()
}

/// Deserialize multiple expressions from a batch binary blob.
pub fn batch_from_binary(bytes: &[u8]) -> Result<Vec<TLExpr>, ExprSerializeError> {
    let mut r = binary::BinReader::new(bytes);
    let magic = r.read_magic()?;
    if magic != TLBX_MAGIC {
        return Err(ExprSerializeError::InvalidMagic);
    }
    let version = r.read_u32()?;
    if version != FORMAT_VER {
        return Err(ExprSerializeError::VersionMismatch {
            expected: FORMAT_VER,
            got: version,
        });
    }
    let count = r.read_u32()? as usize;
    let mut exprs = Vec::with_capacity(count);
    for _ in 0..count {
        exprs.push(binary::read_expr_bin(&mut r)?);
    }
    Ok(exprs)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EinsumGraph, EinsumNode, TLExpr, TNormKind, Term};

    fn simple_pred(name: &str, arg: &str) -> TLExpr {
        TLExpr::Pred {
            name: name.to_string(),
            args: vec![Term::Var(arg.to_string())],
        }
    }

    #[test]
    fn test_sexpr_variable() {
        let expr = simple_pred("P", "x");
        let s = to_sexpr(&expr);
        assert!(s.contains("x"));
        assert!(s.contains("Pred"));
    }

    #[test]
    fn test_sexpr_constant() {
        let expr = TLExpr::Constant(3.15);
        let s = to_sexpr(&expr);
        assert!(s.contains("3.15"));
        assert!(s.contains("Constant"));
    }

    #[test]
    fn test_sexpr_not() {
        let expr = TLExpr::Not(Box::new(simple_pred("P", "x")));
        let s = to_sexpr(&expr);
        assert!(s.contains("Not"));
        assert!(s.contains("Pred"));
    }

    #[test]
    fn test_sexpr_and() {
        let a = simple_pred("P", "x");
        let b = simple_pred("Q", "y");
        let expr = TLExpr::And(Box::new(a), Box::new(b));
        let s = to_sexpr(&expr);
        assert!(s.contains("And"));
        assert!(s.contains("\"P\""));
        assert!(s.contains("\"Q\""));
    }

    #[test]
    fn test_sexpr_roundtrip_simple() {
        let expr = TLExpr::Constant(42.0);
        let s = to_sexpr(&expr);
        let parsed = from_sexpr(&s).expect("parse failed");
        assert_eq!(parsed, expr);
    }

    #[test]
    fn test_sexpr_roundtrip_nested() {
        let inner = TLExpr::And(
            Box::new(simple_pred("P", "x")),
            Box::new(TLExpr::Not(Box::new(simple_pred("Q", "y")))),
        );
        let expr = TLExpr::ForAll {
            var: "x".to_string(),
            domain: "Entity".to_string(),
            body: Box::new(inner),
        };
        let s = to_sexpr(&expr);
        let parsed = from_sexpr(&s).expect("parse failed");
        assert_eq!(parsed, expr);
    }

    #[test]
    fn test_sexpr_parse_error() {
        let result = from_sexpr("not valid sexpr )))");
        assert!(result.is_err());
    }

    #[test]
    fn test_binary_roundtrip_variable() {
        let expr = simple_pred("P", "x");
        let bin = to_binary(&expr);
        let parsed = from_binary(&bin).expect("binary parse failed");
        assert_eq!(parsed, expr);
    }

    #[test]
    fn test_binary_roundtrip_constant() {
        let expr = TLExpr::Constant(2.719);
        let bin = to_binary(&expr);
        let parsed = from_binary(&bin).expect("binary parse failed");
        assert_eq!(parsed, expr);
    }

    #[test]
    fn test_binary_roundtrip_not() {
        let expr = TLExpr::Not(Box::new(TLExpr::Constant(1.0)));
        let bin = to_binary(&expr);
        let parsed = from_binary(&bin).expect("binary parse failed");
        assert_eq!(parsed, expr);
    }

    #[test]
    fn test_binary_roundtrip_and() {
        let expr = TLExpr::And(
            Box::new(TLExpr::Constant(1.0)),
            Box::new(TLExpr::Constant(2.0)),
        );
        let bin = to_binary(&expr);
        let parsed = from_binary(&bin).expect("binary parse failed");
        assert_eq!(parsed, expr);
    }

    #[test]
    fn test_binary_roundtrip_nested() {
        let leaf = simple_pred("leaf", "z");
        let nested = TLExpr::Imply(
            Box::new(TLExpr::And(
                Box::new(TLExpr::Exists {
                    var: "x".to_string(),
                    domain: "D".to_string(),
                    body: Box::new(leaf.clone()),
                }),
                Box::new(TLExpr::Not(Box::new(leaf))),
            )),
            Box::new(TLExpr::Constant(99.9)),
        );
        let bin = to_binary(&nested);
        let parsed = from_binary(&bin).expect("binary parse failed");
        assert_eq!(parsed, nested);
    }

    #[test]
    fn test_binary_magic_check() {
        let expr = TLExpr::Constant(1.0);
        let bin = to_binary(&expr);
        assert_eq!(&bin[..4], &TLEX_MAGIC);
    }

    #[test]
    fn test_binary_invalid_magic() {
        let data = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x00, 0x00, 0x00];
        let result = from_binary(&data);
        assert!(matches!(result, Err(ExprSerializeError::InvalidMagic)));
    }

    #[test]
    fn test_binary_truncated() {
        let data = vec![0x54, 0x4C, 0x45]; // incomplete magic
        let result = from_binary(&data);
        assert!(matches!(result, Err(ExprSerializeError::TruncatedInput)));
    }

    #[test]
    fn test_expr_fingerprint_same() {
        let a = TLExpr::And(
            Box::new(TLExpr::Constant(1.0)),
            Box::new(TLExpr::Constant(2.0)),
        );
        let b = TLExpr::And(
            Box::new(TLExpr::Constant(1.0)),
            Box::new(TLExpr::Constant(2.0)),
        );
        assert_eq!(expr_fingerprint(&a), expr_fingerprint(&b));
    }

    #[test]
    fn test_expr_fingerprint_different() {
        let a = TLExpr::Constant(1.0);
        let b = TLExpr::Constant(2.0);
        assert_ne!(expr_fingerprint(&a), expr_fingerprint(&b));
    }

    #[test]
    fn test_binary_equal_true() {
        let expr = TLExpr::Or(
            Box::new(TLExpr::Constant(1.0)),
            Box::new(TLExpr::Constant(2.0)),
        );
        let a = to_binary(&expr);
        let b = to_binary(&expr);
        assert!(binary_equal(&a, &b));
    }

    #[test]
    fn test_serialization_stats() {
        let expr = TLExpr::And(
            Box::new(simple_pred("P", "x")),
            Box::new(TLExpr::Not(Box::new(simple_pred("Q", "y")))),
        );
        let stats = SerializationStats::compute(&expr);
        assert!(stats.sexpr_bytes > 0);
        assert!(stats.binary_bytes > 0);
        assert!(stats.node_count > 0);
        assert!(stats.max_depth > 0);
        let summary = stats.summary();
        assert!(summary.contains("bytes"));
    }

    #[test]
    fn test_batch_roundtrip() {
        let exprs = vec![
            TLExpr::Constant(1.0),
            TLExpr::Not(Box::new(TLExpr::Constant(2.0))),
            TLExpr::And(
                Box::new(simple_pred("P", "x")),
                Box::new(simple_pred("Q", "y")),
            ),
        ];
        let bin = batch_to_binary(&exprs);
        let parsed = batch_from_binary(&bin).expect("batch parse failed");
        assert_eq!(parsed.len(), exprs.len());
        for (a, b) in exprs.iter().zip(parsed.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_graph_binary_roundtrip() {
        let mut graph = EinsumGraph::new();
        let _a = graph.add_tensor("A");
        let _b = graph.add_tensor("B");
        let _c = graph.add_tensor("C");
        graph
            .add_node(EinsumNode::einsum("ik,kj->ij", vec![0, 1], vec![2]))
            .expect("add node failed");
        graph.add_output(2).expect("add output failed");

        let bin = graph_to_binary(&graph);
        let parsed = graph_from_binary(&bin).expect("graph parse failed");
        assert_eq!(parsed.tensors, graph.tensors);
        assert_eq!(parsed.inputs, graph.inputs);
        assert_eq!(parsed.outputs, graph.outputs);
        assert_eq!(parsed.nodes.len(), graph.nodes.len());
    }

    #[test]
    fn test_sexpr_roundtrip_empty_set() {
        let expr = TLExpr::EmptySet;
        let s = to_sexpr(&expr);
        let parsed = from_sexpr(&s).expect("parse failed");
        assert_eq!(parsed, expr);
    }

    #[test]
    fn test_binary_roundtrip_lambda() {
        let expr = TLExpr::Lambda {
            var: "x".to_string(),
            var_type: Some("Int".to_string()),
            body: Box::new(TLExpr::Constant(42.0)),
        };
        let bin = to_binary(&expr);
        let parsed = from_binary(&bin).expect("binary parse failed");
        assert_eq!(parsed, expr);
    }

    #[test]
    fn test_binary_roundtrip_fuzzy() {
        let expr = TLExpr::TNorm {
            kind: TNormKind::Lukasiewicz,
            left: Box::new(TLExpr::Constant(0.5)),
            right: Box::new(TLExpr::Constant(0.7)),
        };
        let bin = to_binary(&expr);
        let parsed = from_binary(&bin).expect("binary parse failed");
        assert_eq!(parsed, expr);
    }
}

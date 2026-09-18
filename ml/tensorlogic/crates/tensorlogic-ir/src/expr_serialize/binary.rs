//! Binary serialization and deserialization for TLExpr and EinsumGraph.

use std::collections::HashMap;

use crate::{
    AggregateOp, EinsumGraph, EinsumNode, FuzzyImplicationKind, FuzzyNegationKind, Metadata,
    OpType, TCoNormKind, TLExpr, TNormKind, Term, TypeAnnotation,
};

use super::ExprSerializeError;
use super::{
    AGG_ALL, AGG_ANY, AGG_AVERAGE, AGG_COUNT, AGG_MAX, AGG_MIN, AGG_PRODUCT, AGG_SUM, FIMP_GODEL,
    FIMP_GOGUEN, FIMP_KLEENE_DIENES, FIMP_LUKASIEWICZ, FIMP_REICHENBACH, FIMP_RESCHER,
    FNEG_STANDARD, FNEG_SUGENO, FNEG_YAGER, FORMAT_VER, OP_EINSUM, OP_ELEM_BINARY, OP_ELEM_UNARY,
    OP_REDUCE, TAG_ABDUCIBLE, TAG_ABS, TAG_ADD, TAG_AGGREGATE, TAG_ALL_DIFFERENT, TAG_ALWAYS,
    TAG_AND, TAG_APPLY, TAG_AT, TAG_BOX, TAG_CEIL, TAG_CONSTANT, TAG_COS, TAG_COUNTING_EXISTS,
    TAG_COUNTING_FORALL, TAG_DIAMOND, TAG_DIV, TAG_EMPTY_SET, TAG_EQ, TAG_EVENTUALLY,
    TAG_EVERYWHERE, TAG_EXACT_COUNT, TAG_EXISTS, TAG_EXP, TAG_EXPLAIN, TAG_FLOOR, TAG_FORALL,
    TAG_FUZZY_IMPLICATION, TAG_FUZZY_NOT, TAG_GLOBAL_CARDINALITY, TAG_GREATEST_FIXPOINT, TAG_GT,
    TAG_GTE, TAG_IF_THEN_ELSE, TAG_IMPLY, TAG_LAMBDA, TAG_LEAST_FIXPOINT, TAG_LET, TAG_LOG, TAG_LT,
    TAG_LTE, TAG_MAJORITY, TAG_MATCH, TAG_MAX, TAG_MIN, TAG_MOD, TAG_MUL, TAG_NEXT, TAG_NOMINAL,
    TAG_NOT, TAG_OR, TAG_PATTERN_CONST_NUMBER, TAG_PATTERN_CONST_SYMBOL, TAG_PATTERN_WILDCARD,
    TAG_POW, TAG_PRED, TAG_PROBABILISTIC_CHOICE, TAG_RELEASE, TAG_ROUND, TAG_SCORE,
    TAG_SET_CARDINALITY, TAG_SET_COMPREHENSION, TAG_SET_DIFFERENCE, TAG_SET_INTERSECTION,
    TAG_SET_MEMBERSHIP, TAG_SET_UNION, TAG_SIN, TAG_SOFT_EXISTS, TAG_SOFT_FORALL, TAG_SOMEWHERE,
    TAG_SQRT, TAG_STRONG_RELEASE, TAG_SUB, TAG_SYMBOL_LITERAL, TAG_TAN, TAG_TCONORM, TAG_TNORM,
    TAG_UNTIL, TAG_WEAK_UNTIL, TAG_WEIGHTED_RULE, TCONORM_BOUNDED_SUM, TCONORM_DRASTIC,
    TCONORM_HAMACHER, TCONORM_MAXIMUM, TCONORM_NILPOTENT_MAXIMUM, TCONORM_PROBABILISTIC_SUM,
    TERM_TAG_CONST, TERM_TAG_TYPED, TERM_TAG_VAR, TLEX_MAGIC, TLGR_MAGIC, TNORM_DRASTIC,
    TNORM_HAMACHER, TNORM_LUKASIEWICZ, TNORM_MINIMUM, TNORM_NILPOTENT_MINIMUM, TNORM_PRODUCT,
};

/// A writer that accumulates bytes.
pub(super) struct BinWriter {
    buf: Vec<u8>,
}

impl BinWriter {
    pub(super) fn new() -> Self {
        Self { buf: Vec::new() }
    }

    pub(super) fn write_u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    pub(super) fn write_u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_i32(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_f64(&mut self, v: f64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_string(&mut self, s: &str) {
        self.write_u32(s.len() as u32);
        self.buf.extend_from_slice(s.as_bytes());
    }

    pub(super) fn write_magic(&mut self, magic: &[u8; 4]) {
        self.buf.extend_from_slice(magic);
    }

    pub(super) fn into_bytes(self) -> Vec<u8> {
        self.buf
    }
}

/// A reader that consumes bytes from a slice.
pub(super) struct BinReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BinReader<'a> {
    pub(super) fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    pub(super) fn read_u8(&mut self) -> Result<u8, ExprSerializeError> {
        if self.remaining() < 1 {
            return Err(ExprSerializeError::TruncatedInput);
        }
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    pub(super) fn read_u32(&mut self) -> Result<u32, ExprSerializeError> {
        if self.remaining() < 4 {
            return Err(ExprSerializeError::TruncatedInput);
        }
        let bytes: [u8; 4] = self.data[self.pos..self.pos + 4]
            .try_into()
            .map_err(|_| ExprSerializeError::TruncatedInput)?;
        self.pos += 4;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_u64(&mut self) -> Result<u64, ExprSerializeError> {
        if self.remaining() < 8 {
            return Err(ExprSerializeError::TruncatedInput);
        }
        let bytes: [u8; 8] = self.data[self.pos..self.pos + 8]
            .try_into()
            .map_err(|_| ExprSerializeError::TruncatedInput)?;
        self.pos += 8;
        Ok(u64::from_le_bytes(bytes))
    }

    fn read_i32(&mut self) -> Result<i32, ExprSerializeError> {
        if self.remaining() < 4 {
            return Err(ExprSerializeError::TruncatedInput);
        }
        let bytes: [u8; 4] = self.data[self.pos..self.pos + 4]
            .try_into()
            .map_err(|_| ExprSerializeError::TruncatedInput)?;
        self.pos += 4;
        Ok(i32::from_le_bytes(bytes))
    }

    fn read_f64(&mut self) -> Result<f64, ExprSerializeError> {
        if self.remaining() < 8 {
            return Err(ExprSerializeError::TruncatedInput);
        }
        let bytes: [u8; 8] = self.data[self.pos..self.pos + 8]
            .try_into()
            .map_err(|_| ExprSerializeError::TruncatedInput)?;
        self.pos += 8;
        Ok(f64::from_le_bytes(bytes))
    }

    fn read_str(&mut self) -> Result<String, ExprSerializeError> {
        let len = self.read_u32()? as usize;
        if self.remaining() < len {
            return Err(ExprSerializeError::TruncatedInput);
        }
        let s = std::str::from_utf8(&self.data[self.pos..self.pos + len])
            .map_err(|e| ExprSerializeError::Utf8Error(e.to_string()))?
            .to_string();
        self.pos += len;
        Ok(s)
    }

    pub(super) fn read_magic(&mut self) -> Result<[u8; 4], ExprSerializeError> {
        if self.remaining() < 4 {
            return Err(ExprSerializeError::TruncatedInput);
        }
        let magic: [u8; 4] = self.data[self.pos..self.pos + 4]
            .try_into()
            .map_err(|_| ExprSerializeError::TruncatedInput)?;
        self.pos += 4;
        Ok(magic)
    }
}

/// Serialize a `TLExpr` to compact binary bytes.
pub fn to_binary(expr: &TLExpr) -> Vec<u8> {
    let mut w = BinWriter::new();
    w.write_magic(&TLEX_MAGIC);
    w.write_u32(FORMAT_VER);
    write_expr_bin(expr, &mut w);
    w.into_bytes()
}

/// Deserialize a `TLExpr` from binary bytes.
pub fn from_binary(bytes: &[u8]) -> Result<TLExpr, ExprSerializeError> {
    let mut r = BinReader::new(bytes);
    let magic = r.read_magic()?;
    if magic != TLEX_MAGIC {
        return Err(ExprSerializeError::InvalidMagic);
    }
    let version = r.read_u32()?;
    if version != FORMAT_VER {
        return Err(ExprSerializeError::VersionMismatch {
            expected: FORMAT_VER,
            got: version,
        });
    }
    read_expr_bin(&mut r)
}

fn write_term_bin(term: &Term, w: &mut BinWriter) {
    match term {
        Term::Var(name) => {
            w.write_u8(TERM_TAG_VAR);
            w.write_string(name);
        }
        Term::Const(name) => {
            w.write_u8(TERM_TAG_CONST);
            w.write_string(name);
        }
        Term::Typed {
            value,
            type_annotation,
        } => {
            w.write_u8(TERM_TAG_TYPED);
            write_term_bin(value, w);
            w.write_string(&type_annotation.type_name);
        }
    }
}

fn read_term_bin(r: &mut BinReader<'_>) -> Result<Term, ExprSerializeError> {
    let tag = r.read_u8()?;
    match tag {
        TERM_TAG_VAR => Ok(Term::Var(r.read_str()?)),
        TERM_TAG_CONST => Ok(Term::Const(r.read_str()?)),
        TERM_TAG_TYPED => {
            let value = read_term_bin(r)?;
            let type_name = r.read_str()?;
            Ok(Term::Typed {
                value: Box::new(value),
                type_annotation: TypeAnnotation::new(type_name),
            })
        }
        _ => Err(ExprSerializeError::UnknownVariant(format!(
            "Term tag {tag}"
        ))),
    }
}

fn write_optional_string(s: &Option<String>, w: &mut BinWriter) {
    match s {
        Some(val) => {
            w.write_u8(1);
            w.write_string(val);
        }
        None => w.write_u8(0),
    }
}

fn read_optional_string(r: &mut BinReader<'_>) -> Result<Option<String>, ExprSerializeError> {
    let has = r.read_u8()?;
    if has == 0 {
        Ok(None)
    } else {
        Ok(Some(r.read_str()?))
    }
}

fn write_string_vec(v: &[String], w: &mut BinWriter) {
    w.write_u32(v.len() as u32);
    for s in v {
        w.write_string(s);
    }
}

fn read_string_vec(r: &mut BinReader<'_>) -> Result<Vec<String>, ExprSerializeError> {
    let count = r.read_u32()? as usize;
    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        result.push(r.read_str()?);
    }
    Ok(result)
}

fn write_usize_vec(v: &[usize], w: &mut BinWriter) {
    w.write_u32(v.len() as u32);
    for &val in v {
        w.write_u64(val as u64);
    }
}

fn read_usize_vec(r: &mut BinReader<'_>) -> Result<Vec<usize>, ExprSerializeError> {
    let count = r.read_u32()? as usize;
    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        result.push(r.read_u64()? as usize);
    }
    Ok(result)
}

pub(super) fn write_expr_bin(expr: &TLExpr, w: &mut BinWriter) {
    match expr {
        TLExpr::Pred { name, args } => {
            w.write_u8(TAG_PRED);
            w.write_string(name);
            w.write_u32(args.len() as u32);
            for arg in args {
                write_term_bin(arg, w);
            }
        }
        TLExpr::And(a, b) => {
            w.write_u8(TAG_AND);
            write_expr_bin(a, w);
            write_expr_bin(b, w);
        }
        TLExpr::Or(a, b) => {
            w.write_u8(TAG_OR);
            write_expr_bin(a, w);
            write_expr_bin(b, w);
        }
        TLExpr::Not(e) => {
            w.write_u8(TAG_NOT);
            write_expr_bin(e, w);
        }
        TLExpr::Exists { var, domain, body } => {
            w.write_u8(TAG_EXISTS);
            w.write_string(var);
            w.write_string(domain);
            write_expr_bin(body, w);
        }
        TLExpr::ForAll { var, domain, body } => {
            w.write_u8(TAG_FORALL);
            w.write_string(var);
            w.write_string(domain);
            write_expr_bin(body, w);
        }
        TLExpr::Imply(a, b) => {
            w.write_u8(TAG_IMPLY);
            write_expr_bin(a, w);
            write_expr_bin(b, w);
        }
        TLExpr::Score(e) => {
            w.write_u8(TAG_SCORE);
            write_expr_bin(e, w);
        }
        TLExpr::Add(a, b) => {
            w.write_u8(TAG_ADD);
            write_expr_bin(a, w);
            write_expr_bin(b, w);
        }
        TLExpr::Sub(a, b) => {
            w.write_u8(TAG_SUB);
            write_expr_bin(a, w);
            write_expr_bin(b, w);
        }
        TLExpr::Mul(a, b) => {
            w.write_u8(TAG_MUL);
            write_expr_bin(a, w);
            write_expr_bin(b, w);
        }
        TLExpr::Div(a, b) => {
            w.write_u8(TAG_DIV);
            write_expr_bin(a, w);
            write_expr_bin(b, w);
        }
        TLExpr::Pow(a, b) => {
            w.write_u8(TAG_POW);
            write_expr_bin(a, w);
            write_expr_bin(b, w);
        }
        TLExpr::Mod(a, b) => {
            w.write_u8(TAG_MOD);
            write_expr_bin(a, w);
            write_expr_bin(b, w);
        }
        TLExpr::Min(a, b) => {
            w.write_u8(TAG_MIN);
            write_expr_bin(a, w);
            write_expr_bin(b, w);
        }
        TLExpr::Max(a, b) => {
            w.write_u8(TAG_MAX);
            write_expr_bin(a, w);
            write_expr_bin(b, w);
        }
        TLExpr::Abs(e) => {
            w.write_u8(TAG_ABS);
            write_expr_bin(e, w);
        }
        TLExpr::Floor(e) => {
            w.write_u8(TAG_FLOOR);
            write_expr_bin(e, w);
        }
        TLExpr::Ceil(e) => {
            w.write_u8(TAG_CEIL);
            write_expr_bin(e, w);
        }
        TLExpr::Round(e) => {
            w.write_u8(TAG_ROUND);
            write_expr_bin(e, w);
        }
        TLExpr::Sqrt(e) => {
            w.write_u8(TAG_SQRT);
            write_expr_bin(e, w);
        }
        TLExpr::Exp(e) => {
            w.write_u8(TAG_EXP);
            write_expr_bin(e, w);
        }
        TLExpr::Log(e) => {
            w.write_u8(TAG_LOG);
            write_expr_bin(e, w);
        }
        TLExpr::Sin(e) => {
            w.write_u8(TAG_SIN);
            write_expr_bin(e, w);
        }
        TLExpr::Cos(e) => {
            w.write_u8(TAG_COS);
            write_expr_bin(e, w);
        }
        TLExpr::Tan(e) => {
            w.write_u8(TAG_TAN);
            write_expr_bin(e, w);
        }
        TLExpr::Eq(a, b) => {
            w.write_u8(TAG_EQ);
            write_expr_bin(a, w);
            write_expr_bin(b, w);
        }
        TLExpr::Lt(a, b) => {
            w.write_u8(TAG_LT);
            write_expr_bin(a, w);
            write_expr_bin(b, w);
        }
        TLExpr::Gt(a, b) => {
            w.write_u8(TAG_GT);
            write_expr_bin(a, w);
            write_expr_bin(b, w);
        }
        TLExpr::Lte(a, b) => {
            w.write_u8(TAG_LTE);
            write_expr_bin(a, w);
            write_expr_bin(b, w);
        }
        TLExpr::Gte(a, b) => {
            w.write_u8(TAG_GTE);
            write_expr_bin(a, w);
            write_expr_bin(b, w);
        }
        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => {
            w.write_u8(TAG_IF_THEN_ELSE);
            write_expr_bin(condition, w);
            write_expr_bin(then_branch, w);
            write_expr_bin(else_branch, w);
        }
        TLExpr::Constant(v) => {
            w.write_u8(TAG_CONSTANT);
            w.write_f64(*v);
        }
        TLExpr::Aggregate {
            op,
            var,
            domain,
            body,
            group_by,
        } => {
            w.write_u8(TAG_AGGREGATE);
            w.write_u8(aggregate_op_tag(op));
            w.write_string(var);
            w.write_string(domain);
            write_expr_bin(body, w);
            match group_by {
                Some(gb) => {
                    w.write_u8(1);
                    write_string_vec(gb, w);
                }
                None => w.write_u8(0),
            }
        }
        TLExpr::Let { var, value, body } => {
            w.write_u8(TAG_LET);
            w.write_string(var);
            write_expr_bin(value, w);
            write_expr_bin(body, w);
        }
        TLExpr::Box(e) => {
            w.write_u8(TAG_BOX);
            write_expr_bin(e, w);
        }
        TLExpr::Diamond(e) => {
            w.write_u8(TAG_DIAMOND);
            write_expr_bin(e, w);
        }
        TLExpr::Next(e) => {
            w.write_u8(TAG_NEXT);
            write_expr_bin(e, w);
        }
        TLExpr::Eventually(e) => {
            w.write_u8(TAG_EVENTUALLY);
            write_expr_bin(e, w);
        }
        TLExpr::Always(e) => {
            w.write_u8(TAG_ALWAYS);
            write_expr_bin(e, w);
        }
        TLExpr::Until { before, after } => {
            w.write_u8(TAG_UNTIL);
            write_expr_bin(before, w);
            write_expr_bin(after, w);
        }
        TLExpr::TNorm { kind, left, right } => {
            w.write_u8(TAG_TNORM);
            w.write_u8(tnorm_kind_tag(kind));
            write_expr_bin(left, w);
            write_expr_bin(right, w);
        }
        TLExpr::TCoNorm { kind, left, right } => {
            w.write_u8(TAG_TCONORM);
            w.write_u8(tconorm_kind_tag(kind));
            write_expr_bin(left, w);
            write_expr_bin(right, w);
        }
        TLExpr::FuzzyNot { kind, expr: e } => {
            w.write_u8(TAG_FUZZY_NOT);
            write_fuzzy_neg_kind_bin(kind, w);
            write_expr_bin(e, w);
        }
        TLExpr::FuzzyImplication {
            kind,
            premise,
            conclusion,
        } => {
            w.write_u8(TAG_FUZZY_IMPLICATION);
            w.write_u8(fuzzy_imp_kind_tag(kind));
            write_expr_bin(premise, w);
            write_expr_bin(conclusion, w);
        }
        TLExpr::SoftExists {
            var,
            domain,
            body,
            temperature,
        } => {
            w.write_u8(TAG_SOFT_EXISTS);
            w.write_string(var);
            w.write_string(domain);
            w.write_f64(*temperature);
            write_expr_bin(body, w);
        }
        TLExpr::SoftForAll {
            var,
            domain,
            body,
            temperature,
        } => {
            w.write_u8(TAG_SOFT_FORALL);
            w.write_string(var);
            w.write_string(domain);
            w.write_f64(*temperature);
            write_expr_bin(body, w);
        }
        TLExpr::WeightedRule { weight, rule } => {
            w.write_u8(TAG_WEIGHTED_RULE);
            w.write_f64(*weight);
            write_expr_bin(rule, w);
        }
        TLExpr::ProbabilisticChoice { alternatives } => {
            w.write_u8(TAG_PROBABILISTIC_CHOICE);
            w.write_u32(alternatives.len() as u32);
            for (prob, alt_expr) in alternatives {
                w.write_f64(*prob);
                write_expr_bin(alt_expr, w);
            }
        }
        TLExpr::Release { released, releaser } => {
            w.write_u8(TAG_RELEASE);
            write_expr_bin(released, w);
            write_expr_bin(releaser, w);
        }
        TLExpr::WeakUntil { before, after } => {
            w.write_u8(TAG_WEAK_UNTIL);
            write_expr_bin(before, w);
            write_expr_bin(after, w);
        }
        TLExpr::StrongRelease { released, releaser } => {
            w.write_u8(TAG_STRONG_RELEASE);
            write_expr_bin(released, w);
            write_expr_bin(releaser, w);
        }
        TLExpr::Lambda {
            var,
            var_type,
            body,
        } => {
            w.write_u8(TAG_LAMBDA);
            w.write_string(var);
            write_optional_string(var_type, w);
            write_expr_bin(body, w);
        }
        TLExpr::Apply { function, argument } => {
            w.write_u8(TAG_APPLY);
            write_expr_bin(function, w);
            write_expr_bin(argument, w);
        }
        TLExpr::SetMembership { element, set } => {
            w.write_u8(TAG_SET_MEMBERSHIP);
            write_expr_bin(element, w);
            write_expr_bin(set, w);
        }
        TLExpr::SetUnion { left, right } => {
            w.write_u8(TAG_SET_UNION);
            write_expr_bin(left, w);
            write_expr_bin(right, w);
        }
        TLExpr::SetIntersection { left, right } => {
            w.write_u8(TAG_SET_INTERSECTION);
            write_expr_bin(left, w);
            write_expr_bin(right, w);
        }
        TLExpr::SetDifference { left, right } => {
            w.write_u8(TAG_SET_DIFFERENCE);
            write_expr_bin(left, w);
            write_expr_bin(right, w);
        }
        TLExpr::SetCardinality { set } => {
            w.write_u8(TAG_SET_CARDINALITY);
            write_expr_bin(set, w);
        }
        TLExpr::EmptySet => {
            w.write_u8(TAG_EMPTY_SET);
        }
        TLExpr::SetComprehension {
            var,
            domain,
            condition,
        } => {
            w.write_u8(TAG_SET_COMPREHENSION);
            w.write_string(var);
            w.write_string(domain);
            write_expr_bin(condition, w);
        }
        TLExpr::CountingExists {
            var,
            domain,
            body,
            min_count,
        } => {
            w.write_u8(TAG_COUNTING_EXISTS);
            w.write_string(var);
            w.write_string(domain);
            w.write_u64(*min_count as u64);
            write_expr_bin(body, w);
        }
        TLExpr::CountingForAll {
            var,
            domain,
            body,
            min_count,
        } => {
            w.write_u8(TAG_COUNTING_FORALL);
            w.write_string(var);
            w.write_string(domain);
            w.write_u64(*min_count as u64);
            write_expr_bin(body, w);
        }
        TLExpr::ExactCount {
            var,
            domain,
            body,
            count,
        } => {
            w.write_u8(TAG_EXACT_COUNT);
            w.write_string(var);
            w.write_string(domain);
            w.write_u64(*count as u64);
            write_expr_bin(body, w);
        }
        TLExpr::Majority { var, domain, body } => {
            w.write_u8(TAG_MAJORITY);
            w.write_string(var);
            w.write_string(domain);
            write_expr_bin(body, w);
        }
        TLExpr::LeastFixpoint { var, body } => {
            w.write_u8(TAG_LEAST_FIXPOINT);
            w.write_string(var);
            write_expr_bin(body, w);
        }
        TLExpr::GreatestFixpoint { var, body } => {
            w.write_u8(TAG_GREATEST_FIXPOINT);
            w.write_string(var);
            write_expr_bin(body, w);
        }
        TLExpr::Nominal { name } => {
            w.write_u8(TAG_NOMINAL);
            w.write_string(name);
        }
        TLExpr::At { nominal, formula } => {
            w.write_u8(TAG_AT);
            w.write_string(nominal);
            write_expr_bin(formula, w);
        }
        TLExpr::Somewhere { formula } => {
            w.write_u8(TAG_SOMEWHERE);
            write_expr_bin(formula, w);
        }
        TLExpr::Everywhere { formula } => {
            w.write_u8(TAG_EVERYWHERE);
            write_expr_bin(formula, w);
        }
        TLExpr::AllDifferent { variables } => {
            w.write_u8(TAG_ALL_DIFFERENT);
            write_string_vec(variables, w);
        }
        TLExpr::GlobalCardinality {
            variables,
            values,
            min_occurrences,
            max_occurrences,
        } => {
            w.write_u8(TAG_GLOBAL_CARDINALITY);
            write_string_vec(variables, w);
            w.write_u32(values.len() as u32);
            for val in values {
                write_expr_bin(val, w);
            }
            write_usize_vec(min_occurrences, w);
            write_usize_vec(max_occurrences, w);
        }
        TLExpr::Abducible { name, cost } => {
            w.write_u8(TAG_ABDUCIBLE);
            w.write_string(name);
            w.write_f64(*cost);
        }
        TLExpr::Explain { formula } => {
            w.write_u8(TAG_EXPLAIN);
            write_expr_bin(formula, w);
        }
        TLExpr::SymbolLiteral(s) => {
            w.write_u8(TAG_SYMBOL_LITERAL);
            w.write_string(s);
        }
        TLExpr::Match { scrutinee, arms } => {
            w.write_u8(TAG_MATCH);
            write_expr_bin(scrutinee, w);
            w.write_u32(arms.len() as u32);
            for (pat, body) in arms {
                match pat {
                    crate::pattern::MatchPattern::ConstSymbol(s) => {
                        w.write_u8(TAG_PATTERN_CONST_SYMBOL);
                        w.write_string(s);
                    }
                    crate::pattern::MatchPattern::ConstNumber(n) => {
                        w.write_u8(TAG_PATTERN_CONST_NUMBER);
                        w.write_f64(*n);
                    }
                    crate::pattern::MatchPattern::Wildcard => {
                        w.write_u8(TAG_PATTERN_WILDCARD);
                    }
                }
                write_expr_bin(body, w);
            }
        }
    }
}

pub(super) fn read_expr_bin(r: &mut BinReader<'_>) -> Result<TLExpr, ExprSerializeError> {
    let tag = r.read_u8()?;
    match tag {
        TAG_PRED => {
            let name = r.read_str()?;
            let count = r.read_u32()? as usize;
            let mut args = Vec::with_capacity(count);
            for _ in 0..count {
                args.push(read_term_bin(r)?);
            }
            Ok(TLExpr::Pred { name, args })
        }
        TAG_AND => {
            let a = read_expr_bin(r)?;
            let b = read_expr_bin(r)?;
            Ok(TLExpr::And(Box::new(a), Box::new(b)))
        }
        TAG_OR => {
            let a = read_expr_bin(r)?;
            let b = read_expr_bin(r)?;
            Ok(TLExpr::Or(Box::new(a), Box::new(b)))
        }
        TAG_NOT => {
            let e = read_expr_bin(r)?;
            Ok(TLExpr::Not(Box::new(e)))
        }
        TAG_EXISTS => {
            let var = r.read_str()?;
            let domain = r.read_str()?;
            let body = read_expr_bin(r)?;
            Ok(TLExpr::Exists {
                var,
                domain,
                body: Box::new(body),
            })
        }
        TAG_FORALL => {
            let var = r.read_str()?;
            let domain = r.read_str()?;
            let body = read_expr_bin(r)?;
            Ok(TLExpr::ForAll {
                var,
                domain,
                body: Box::new(body),
            })
        }
        TAG_IMPLY => {
            let a = read_expr_bin(r)?;
            let b = read_expr_bin(r)?;
            Ok(TLExpr::Imply(Box::new(a), Box::new(b)))
        }
        TAG_SCORE => {
            let e = read_expr_bin(r)?;
            Ok(TLExpr::Score(Box::new(e)))
        }
        TAG_ADD => read_binary_expr(r, TLExpr::Add),
        TAG_SUB => read_binary_expr(r, TLExpr::Sub),
        TAG_MUL => read_binary_expr(r, TLExpr::Mul),
        TAG_DIV => read_binary_expr(r, TLExpr::Div),
        TAG_POW => read_binary_expr(r, TLExpr::Pow),
        TAG_MOD => read_binary_expr(r, TLExpr::Mod),
        TAG_MIN => read_binary_expr(r, TLExpr::Min),
        TAG_MAX => read_binary_expr(r, TLExpr::Max),
        TAG_ABS => read_unary_expr(r, TLExpr::Abs),
        TAG_FLOOR => read_unary_expr(r, TLExpr::Floor),
        TAG_CEIL => read_unary_expr(r, TLExpr::Ceil),
        TAG_ROUND => read_unary_expr(r, TLExpr::Round),
        TAG_SQRT => read_unary_expr(r, TLExpr::Sqrt),
        TAG_EXP => read_unary_expr(r, TLExpr::Exp),
        TAG_LOG => read_unary_expr(r, TLExpr::Log),
        TAG_SIN => read_unary_expr(r, TLExpr::Sin),
        TAG_COS => read_unary_expr(r, TLExpr::Cos),
        TAG_TAN => read_unary_expr(r, TLExpr::Tan),
        TAG_EQ => read_binary_expr(r, TLExpr::Eq),
        TAG_LT => read_binary_expr(r, TLExpr::Lt),
        TAG_GT => read_binary_expr(r, TLExpr::Gt),
        TAG_LTE => read_binary_expr(r, TLExpr::Lte),
        TAG_GTE => read_binary_expr(r, TLExpr::Gte),
        TAG_IF_THEN_ELSE => {
            let cond = read_expr_bin(r)?;
            let then_b = read_expr_bin(r)?;
            let else_b = read_expr_bin(r)?;
            Ok(TLExpr::IfThenElse {
                condition: Box::new(cond),
                then_branch: Box::new(then_b),
                else_branch: Box::new(else_b),
            })
        }
        TAG_CONSTANT => {
            let v = r.read_f64()?;
            Ok(TLExpr::Constant(v))
        }
        TAG_AGGREGATE => {
            let op_tag = r.read_u8()?;
            let op = read_aggregate_op_tag(op_tag)?;
            let var = r.read_str()?;
            let domain = r.read_str()?;
            let body = read_expr_bin(r)?;
            let has_gb = r.read_u8()?;
            let group_by = if has_gb == 0 {
                None
            } else {
                Some(read_string_vec(r)?)
            };
            Ok(TLExpr::Aggregate {
                op,
                var,
                domain,
                body: Box::new(body),
                group_by,
            })
        }
        TAG_LET => {
            let var = r.read_str()?;
            let value = read_expr_bin(r)?;
            let body = read_expr_bin(r)?;
            Ok(TLExpr::Let {
                var,
                value: Box::new(value),
                body: Box::new(body),
            })
        }
        TAG_BOX => read_unary_expr(r, TLExpr::Box),
        TAG_DIAMOND => read_unary_expr(r, TLExpr::Diamond),
        TAG_NEXT => read_unary_expr(r, TLExpr::Next),
        TAG_EVENTUALLY => read_unary_expr(r, TLExpr::Eventually),
        TAG_ALWAYS => read_unary_expr(r, TLExpr::Always),
        TAG_UNTIL => {
            let before = read_expr_bin(r)?;
            let after = read_expr_bin(r)?;
            Ok(TLExpr::Until {
                before: Box::new(before),
                after: Box::new(after),
            })
        }
        TAG_TNORM => {
            let kind_tag = r.read_u8()?;
            let kind = read_tnorm_kind_tag(kind_tag)?;
            let left = read_expr_bin(r)?;
            let right = read_expr_bin(r)?;
            Ok(TLExpr::TNorm {
                kind,
                left: Box::new(left),
                right: Box::new(right),
            })
        }
        TAG_TCONORM => {
            let kind_tag = r.read_u8()?;
            let kind = read_tconorm_kind_tag(kind_tag)?;
            let left = read_expr_bin(r)?;
            let right = read_expr_bin(r)?;
            Ok(TLExpr::TCoNorm {
                kind,
                left: Box::new(left),
                right: Box::new(right),
            })
        }
        TAG_FUZZY_NOT => {
            let kind = read_fuzzy_neg_kind_bin(r)?;
            let e = read_expr_bin(r)?;
            Ok(TLExpr::FuzzyNot {
                kind,
                expr: Box::new(e),
            })
        }
        TAG_FUZZY_IMPLICATION => {
            let kind_tag = r.read_u8()?;
            let kind = read_fuzzy_imp_kind_tag(kind_tag)?;
            let premise = read_expr_bin(r)?;
            let conclusion = read_expr_bin(r)?;
            Ok(TLExpr::FuzzyImplication {
                kind,
                premise: Box::new(premise),
                conclusion: Box::new(conclusion),
            })
        }
        TAG_SOFT_EXISTS => {
            let var = r.read_str()?;
            let domain = r.read_str()?;
            let temperature = r.read_f64()?;
            let body = read_expr_bin(r)?;
            Ok(TLExpr::SoftExists {
                var,
                domain,
                body: Box::new(body),
                temperature,
            })
        }
        TAG_SOFT_FORALL => {
            let var = r.read_str()?;
            let domain = r.read_str()?;
            let temperature = r.read_f64()?;
            let body = read_expr_bin(r)?;
            Ok(TLExpr::SoftForAll {
                var,
                domain,
                body: Box::new(body),
                temperature,
            })
        }
        TAG_WEIGHTED_RULE => {
            let weight = r.read_f64()?;
            let rule = read_expr_bin(r)?;
            Ok(TLExpr::WeightedRule {
                weight,
                rule: Box::new(rule),
            })
        }
        TAG_PROBABILISTIC_CHOICE => {
            let count = r.read_u32()? as usize;
            let mut alternatives = Vec::with_capacity(count);
            for _ in 0..count {
                let prob = r.read_f64()?;
                let alt_expr = read_expr_bin(r)?;
                alternatives.push((prob, alt_expr));
            }
            Ok(TLExpr::ProbabilisticChoice { alternatives })
        }
        TAG_RELEASE => {
            let released = read_expr_bin(r)?;
            let releaser = read_expr_bin(r)?;
            Ok(TLExpr::Release {
                released: Box::new(released),
                releaser: Box::new(releaser),
            })
        }
        TAG_WEAK_UNTIL => {
            let before = read_expr_bin(r)?;
            let after = read_expr_bin(r)?;
            Ok(TLExpr::WeakUntil {
                before: Box::new(before),
                after: Box::new(after),
            })
        }
        TAG_STRONG_RELEASE => {
            let released = read_expr_bin(r)?;
            let releaser = read_expr_bin(r)?;
            Ok(TLExpr::StrongRelease {
                released: Box::new(released),
                releaser: Box::new(releaser),
            })
        }
        TAG_LAMBDA => {
            let var = r.read_str()?;
            let var_type = read_optional_string(r)?;
            let body = read_expr_bin(r)?;
            Ok(TLExpr::Lambda {
                var,
                var_type,
                body: Box::new(body),
            })
        }
        TAG_APPLY => {
            let function = read_expr_bin(r)?;
            let argument = read_expr_bin(r)?;
            Ok(TLExpr::Apply {
                function: Box::new(function),
                argument: Box::new(argument),
            })
        }
        TAG_SET_MEMBERSHIP => {
            let element = read_expr_bin(r)?;
            let set = read_expr_bin(r)?;
            Ok(TLExpr::SetMembership {
                element: Box::new(element),
                set: Box::new(set),
            })
        }
        TAG_SET_UNION => read_binary_expr(r, |a, b| TLExpr::SetUnion { left: a, right: b }),
        TAG_SET_INTERSECTION => {
            read_binary_expr(r, |a, b| TLExpr::SetIntersection { left: a, right: b })
        }
        TAG_SET_DIFFERENCE => {
            read_binary_expr(r, |a, b| TLExpr::SetDifference { left: a, right: b })
        }
        TAG_SET_CARDINALITY => read_unary_expr(r, |e| TLExpr::SetCardinality { set: e }),
        TAG_EMPTY_SET => Ok(TLExpr::EmptySet),
        TAG_SET_COMPREHENSION => {
            let var = r.read_str()?;
            let domain = r.read_str()?;
            let condition = read_expr_bin(r)?;
            Ok(TLExpr::SetComprehension {
                var,
                domain,
                condition: Box::new(condition),
            })
        }
        TAG_COUNTING_EXISTS => {
            let var = r.read_str()?;
            let domain = r.read_str()?;
            let min_count = r.read_u64()? as usize;
            let body = read_expr_bin(r)?;
            Ok(TLExpr::CountingExists {
                var,
                domain,
                body: Box::new(body),
                min_count,
            })
        }
        TAG_COUNTING_FORALL => {
            let var = r.read_str()?;
            let domain = r.read_str()?;
            let min_count = r.read_u64()? as usize;
            let body = read_expr_bin(r)?;
            Ok(TLExpr::CountingForAll {
                var,
                domain,
                body: Box::new(body),
                min_count,
            })
        }
        TAG_EXACT_COUNT => {
            let var = r.read_str()?;
            let domain = r.read_str()?;
            let count = r.read_u64()? as usize;
            let body = read_expr_bin(r)?;
            Ok(TLExpr::ExactCount {
                var,
                domain,
                body: Box::new(body),
                count,
            })
        }
        TAG_MAJORITY => {
            let var = r.read_str()?;
            let domain = r.read_str()?;
            let body = read_expr_bin(r)?;
            Ok(TLExpr::Majority {
                var,
                domain,
                body: Box::new(body),
            })
        }
        TAG_LEAST_FIXPOINT => {
            let var = r.read_str()?;
            let body = read_expr_bin(r)?;
            Ok(TLExpr::LeastFixpoint {
                var,
                body: Box::new(body),
            })
        }
        TAG_GREATEST_FIXPOINT => {
            let var = r.read_str()?;
            let body = read_expr_bin(r)?;
            Ok(TLExpr::GreatestFixpoint {
                var,
                body: Box::new(body),
            })
        }
        TAG_NOMINAL => {
            let name = r.read_str()?;
            Ok(TLExpr::Nominal { name })
        }
        TAG_AT => {
            let nominal = r.read_str()?;
            let formula = read_expr_bin(r)?;
            Ok(TLExpr::At {
                nominal,
                formula: Box::new(formula),
            })
        }
        TAG_SOMEWHERE => read_unary_expr(r, |e| TLExpr::Somewhere { formula: e }),
        TAG_EVERYWHERE => read_unary_expr(r, |e| TLExpr::Everywhere { formula: e }),
        TAG_ALL_DIFFERENT => {
            let variables = read_string_vec(r)?;
            Ok(TLExpr::AllDifferent { variables })
        }
        TAG_GLOBAL_CARDINALITY => {
            let variables = read_string_vec(r)?;
            let val_count = r.read_u32()? as usize;
            let mut values = Vec::with_capacity(val_count);
            for _ in 0..val_count {
                values.push(read_expr_bin(r)?);
            }
            let min_occurrences = read_usize_vec(r)?;
            let max_occurrences = read_usize_vec(r)?;
            Ok(TLExpr::GlobalCardinality {
                variables,
                values,
                min_occurrences,
                max_occurrences,
            })
        }
        TAG_ABDUCIBLE => {
            let name = r.read_str()?;
            let cost = r.read_f64()?;
            Ok(TLExpr::Abducible { name, cost })
        }
        TAG_EXPLAIN => read_unary_expr(r, |e| TLExpr::Explain { formula: e }),
        TAG_SYMBOL_LITERAL => {
            let s = r.read_str()?;
            Ok(TLExpr::SymbolLiteral(s))
        }
        TAG_MATCH => {
            let scrutinee = read_expr_bin(r)?;
            let arm_count = r.read_u32()? as usize;
            let mut arms = Vec::with_capacity(arm_count);
            for _ in 0..arm_count {
                let pat_tag = r.read_u8()?;
                let pat = match pat_tag {
                    TAG_PATTERN_CONST_SYMBOL => {
                        let s = r.read_str()?;
                        crate::pattern::MatchPattern::ConstSymbol(s)
                    }
                    TAG_PATTERN_CONST_NUMBER => {
                        let n = r.read_f64()?;
                        crate::pattern::MatchPattern::ConstNumber(n)
                    }
                    TAG_PATTERN_WILDCARD => crate::pattern::MatchPattern::Wildcard,
                    other => {
                        return Err(ExprSerializeError::UnknownVariant(format!(
                            "pattern tag {other}"
                        )));
                    }
                };
                let body = read_expr_bin(r)?;
                arms.push((pat, Box::new(body)));
            }
            Ok(TLExpr::Match {
                scrutinee: Box::new(scrutinee),
                arms,
            })
        }
        _ => Err(ExprSerializeError::UnknownVariant(format!(
            "binary tag {tag}"
        ))),
    }
}

fn read_unary_expr(
    r: &mut BinReader<'_>,
    ctor: impl FnOnce(Box<TLExpr>) -> TLExpr,
) -> Result<TLExpr, ExprSerializeError> {
    let e = read_expr_bin(r)?;
    Ok(ctor(Box::new(e)))
}

fn read_binary_expr(
    r: &mut BinReader<'_>,
    ctor: impl FnOnce(Box<TLExpr>, Box<TLExpr>) -> TLExpr,
) -> Result<TLExpr, ExprSerializeError> {
    let a = read_expr_bin(r)?;
    let b = read_expr_bin(r)?;
    Ok(ctor(Box::new(a), Box::new(b)))
}

fn aggregate_op_tag(op: &AggregateOp) -> u8 {
    match op {
        AggregateOp::Count => AGG_COUNT,
        AggregateOp::Sum => AGG_SUM,
        AggregateOp::Average => AGG_AVERAGE,
        AggregateOp::Max => AGG_MAX,
        AggregateOp::Min => AGG_MIN,
        AggregateOp::Product => AGG_PRODUCT,
        AggregateOp::Any => AGG_ANY,
        AggregateOp::All => AGG_ALL,
    }
}

fn read_aggregate_op_tag(tag: u8) -> Result<AggregateOp, ExprSerializeError> {
    match tag {
        AGG_COUNT => Ok(AggregateOp::Count),
        AGG_SUM => Ok(AggregateOp::Sum),
        AGG_AVERAGE => Ok(AggregateOp::Average),
        AGG_MAX => Ok(AggregateOp::Max),
        AGG_MIN => Ok(AggregateOp::Min),
        AGG_PRODUCT => Ok(AggregateOp::Product),
        AGG_ANY => Ok(AggregateOp::Any),
        AGG_ALL => Ok(AggregateOp::All),
        _ => Err(ExprSerializeError::UnknownVariant(format!(
            "AggregateOp tag {tag}"
        ))),
    }
}

fn tnorm_kind_tag(kind: &TNormKind) -> u8 {
    match kind {
        TNormKind::Minimum => TNORM_MINIMUM,
        TNormKind::Product => TNORM_PRODUCT,
        TNormKind::Lukasiewicz => TNORM_LUKASIEWICZ,
        TNormKind::Drastic => TNORM_DRASTIC,
        TNormKind::NilpotentMinimum => TNORM_NILPOTENT_MINIMUM,
        TNormKind::Hamacher => TNORM_HAMACHER,
    }
}

fn read_tnorm_kind_tag(tag: u8) -> Result<TNormKind, ExprSerializeError> {
    match tag {
        TNORM_MINIMUM => Ok(TNormKind::Minimum),
        TNORM_PRODUCT => Ok(TNormKind::Product),
        TNORM_LUKASIEWICZ => Ok(TNormKind::Lukasiewicz),
        TNORM_DRASTIC => Ok(TNormKind::Drastic),
        TNORM_NILPOTENT_MINIMUM => Ok(TNormKind::NilpotentMinimum),
        TNORM_HAMACHER => Ok(TNormKind::Hamacher),
        _ => Err(ExprSerializeError::UnknownVariant(format!(
            "TNormKind tag {tag}"
        ))),
    }
}

fn tconorm_kind_tag(kind: &TCoNormKind) -> u8 {
    match kind {
        TCoNormKind::Maximum => TCONORM_MAXIMUM,
        TCoNormKind::ProbabilisticSum => TCONORM_PROBABILISTIC_SUM,
        TCoNormKind::BoundedSum => TCONORM_BOUNDED_SUM,
        TCoNormKind::Drastic => TCONORM_DRASTIC,
        TCoNormKind::NilpotentMaximum => TCONORM_NILPOTENT_MAXIMUM,
        TCoNormKind::Hamacher => TCONORM_HAMACHER,
    }
}

fn read_tconorm_kind_tag(tag: u8) -> Result<TCoNormKind, ExprSerializeError> {
    match tag {
        TCONORM_MAXIMUM => Ok(TCoNormKind::Maximum),
        TCONORM_PROBABILISTIC_SUM => Ok(TCoNormKind::ProbabilisticSum),
        TCONORM_BOUNDED_SUM => Ok(TCoNormKind::BoundedSum),
        TCONORM_DRASTIC => Ok(TCoNormKind::Drastic),
        TCONORM_NILPOTENT_MAXIMUM => Ok(TCoNormKind::NilpotentMaximum),
        TCONORM_HAMACHER => Ok(TCoNormKind::Hamacher),
        _ => Err(ExprSerializeError::UnknownVariant(format!(
            "TCoNormKind tag {tag}"
        ))),
    }
}

fn write_fuzzy_neg_kind_bin(kind: &FuzzyNegationKind, w: &mut BinWriter) {
    match kind {
        FuzzyNegationKind::Standard => w.write_u8(FNEG_STANDARD),
        FuzzyNegationKind::Sugeno { lambda } => {
            w.write_u8(FNEG_SUGENO);
            w.write_i32(*lambda);
        }
        FuzzyNegationKind::Yager { w: wval } => {
            w.write_u8(FNEG_YAGER);
            w.write_u32(*wval);
        }
    }
}

fn read_fuzzy_neg_kind_bin(r: &mut BinReader<'_>) -> Result<FuzzyNegationKind, ExprSerializeError> {
    let tag = r.read_u8()?;
    match tag {
        FNEG_STANDARD => Ok(FuzzyNegationKind::Standard),
        FNEG_SUGENO => {
            let lambda = r.read_i32()?;
            Ok(FuzzyNegationKind::Sugeno { lambda })
        }
        FNEG_YAGER => {
            let w = r.read_u32()?;
            Ok(FuzzyNegationKind::Yager { w })
        }
        _ => Err(ExprSerializeError::UnknownVariant(format!(
            "FuzzyNegationKind tag {tag}"
        ))),
    }
}

fn fuzzy_imp_kind_tag(kind: &FuzzyImplicationKind) -> u8 {
    match kind {
        FuzzyImplicationKind::Godel => FIMP_GODEL,
        FuzzyImplicationKind::Lukasiewicz => FIMP_LUKASIEWICZ,
        FuzzyImplicationKind::Reichenbach => FIMP_REICHENBACH,
        FuzzyImplicationKind::KleeneDienes => FIMP_KLEENE_DIENES,
        FuzzyImplicationKind::Rescher => FIMP_RESCHER,
        FuzzyImplicationKind::Goguen => FIMP_GOGUEN,
    }
}

fn read_fuzzy_imp_kind_tag(tag: u8) -> Result<FuzzyImplicationKind, ExprSerializeError> {
    match tag {
        FIMP_GODEL => Ok(FuzzyImplicationKind::Godel),
        FIMP_LUKASIEWICZ => Ok(FuzzyImplicationKind::Lukasiewicz),
        FIMP_REICHENBACH => Ok(FuzzyImplicationKind::Reichenbach),
        FIMP_KLEENE_DIENES => Ok(FuzzyImplicationKind::KleeneDienes),
        FIMP_RESCHER => Ok(FuzzyImplicationKind::Rescher),
        FIMP_GOGUEN => Ok(FuzzyImplicationKind::Goguen),
        _ => Err(ExprSerializeError::UnknownVariant(format!(
            "FuzzyImplicationKind tag {tag}"
        ))),
    }
}

// ============================================================================
// Graph serialization
// ============================================================================

/// Serialize an `EinsumGraph` to binary bytes.
pub fn graph_to_binary(graph: &EinsumGraph) -> Vec<u8> {
    let mut w = BinWriter::new();
    w.write_magic(&TLGR_MAGIC);
    w.write_u32(FORMAT_VER);

    // tensors
    write_string_vec(&graph.tensors, &mut w);

    // nodes
    w.write_u32(graph.nodes.len() as u32);
    for node in &graph.nodes {
        write_optype_bin(&node.op, &mut w);
        write_usize_vec(&node.inputs, &mut w);
        write_usize_vec(&node.outputs, &mut w);
        // metadata: write presence flag + name if present
        match &node.metadata {
            Some(meta) => {
                w.write_u8(1);
                write_optional_string(&meta.name, &mut w);
            }
            None => w.write_u8(0),
        }
    }

    // inputs
    write_usize_vec(&graph.inputs, &mut w);
    // outputs
    write_usize_vec(&graph.outputs, &mut w);

    // tensor_metadata count
    w.write_u32(graph.tensor_metadata.len() as u32);
    // Sort keys for deterministic output
    let mut keys: Vec<&usize> = graph.tensor_metadata.keys().collect();
    keys.sort();
    for &key in &keys {
        if let Some(meta) = graph.tensor_metadata.get(key) {
            w.write_u64(*key as u64);
            write_optional_string(&meta.name, &mut w);
        }
    }

    w.into_bytes()
}

/// Deserialize an `EinsumGraph` from binary bytes.
pub fn graph_from_binary(bytes: &[u8]) -> Result<EinsumGraph, ExprSerializeError> {
    let mut r = BinReader::new(bytes);
    let magic = r.read_magic()?;
    if magic != TLGR_MAGIC {
        return Err(ExprSerializeError::InvalidMagic);
    }
    let version = r.read_u32()?;
    if version != FORMAT_VER {
        return Err(ExprSerializeError::VersionMismatch {
            expected: FORMAT_VER,
            got: version,
        });
    }

    let tensors = read_string_vec(&mut r)?;
    let node_count = r.read_u32()? as usize;
    let mut nodes = Vec::with_capacity(node_count);
    for _ in 0..node_count {
        let op = read_optype_bin(&mut r)?;
        let inputs = read_usize_vec(&mut r)?;
        let outputs = read_usize_vec(&mut r)?;
        let has_meta = r.read_u8()?;
        let metadata = if has_meta != 0 {
            let name = read_optional_string(&mut r)?;
            let mut meta = Metadata::new();
            if let Some(n) = name {
                meta = meta.with_name(n);
            }
            Some(meta)
        } else {
            None
        };
        nodes.push(EinsumNode {
            op,
            inputs,
            outputs,
            metadata,
        });
    }

    let inputs = read_usize_vec(&mut r)?;
    let outputs = read_usize_vec(&mut r)?;

    let meta_count = r.read_u32()? as usize;
    let mut tensor_metadata = HashMap::new();
    for _ in 0..meta_count {
        let key = r.read_u64()? as usize;
        let name_opt = read_optional_string(&mut r)?;
        let mut meta = Metadata::new();
        if let Some(n) = name_opt {
            meta = meta.with_name(n);
        }
        tensor_metadata.insert(key, meta);
    }

    Ok(EinsumGraph {
        tensors,
        nodes,
        inputs,
        outputs,
        tensor_metadata,
    })
}

fn write_optype_bin(op: &OpType, w: &mut BinWriter) {
    match op {
        OpType::Einsum { spec } => {
            w.write_u8(OP_EINSUM);
            w.write_string(spec);
        }
        OpType::ElemUnary { op: op_name } => {
            w.write_u8(OP_ELEM_UNARY);
            w.write_string(op_name);
        }
        OpType::ElemBinary { op: op_name } => {
            w.write_u8(OP_ELEM_BINARY);
            w.write_string(op_name);
        }
        OpType::Reduce { op: op_name, axes } => {
            w.write_u8(OP_REDUCE);
            w.write_string(op_name);
            write_usize_vec(axes, w);
        }
    }
}

fn read_optype_bin(r: &mut BinReader<'_>) -> Result<OpType, ExprSerializeError> {
    let tag = r.read_u8()?;
    match tag {
        OP_EINSUM => {
            let spec = r.read_str()?;
            Ok(OpType::Einsum { spec })
        }
        OP_ELEM_UNARY => {
            let op_name = r.read_str()?;
            Ok(OpType::ElemUnary { op: op_name })
        }
        OP_ELEM_BINARY => {
            let op_name = r.read_str()?;
            Ok(OpType::ElemBinary { op: op_name })
        }
        OP_REDUCE => {
            let op_name = r.read_str()?;
            let axes = read_usize_vec(r)?;
            Ok(OpType::Reduce { op: op_name, axes })
        }
        _ => Err(ExprSerializeError::UnknownVariant(format!(
            "OpType tag {tag}"
        ))),
    }
}

//! S-expression serialization and parsing for TLExpr.

use crate::{
    AggregateOp, FuzzyImplicationKind, FuzzyNegationKind, TCoNormKind, TLExpr, TNormKind, Term,
    TypeAnnotation,
};

use super::ExprSerializeError;

/// Serialize a `TLExpr` to an S-expression string.
pub fn to_sexpr(expr: &TLExpr) -> String {
    let mut buf = String::new();
    write_sexpr(expr, &mut buf);
    buf
}

fn write_sexpr(expr: &TLExpr, buf: &mut String) {
    match expr {
        TLExpr::Pred { name, args } => {
            buf.push_str("(Pred ");
            write_quoted(name, buf);
            for arg in args {
                buf.push(' ');
                write_term_sexpr(arg, buf);
            }
            buf.push(')');
        }
        TLExpr::And(a, b) => write_binary_sexpr("And", a, b, buf),
        TLExpr::Or(a, b) => write_binary_sexpr("Or", a, b, buf),
        TLExpr::Not(e) => write_unary_sexpr("Not", e, buf),
        TLExpr::Exists { var, domain, body } => {
            buf.push_str("(Exists ");
            write_quoted(var, buf);
            buf.push(' ');
            write_quoted(domain, buf);
            buf.push(' ');
            write_sexpr(body, buf);
            buf.push(')');
        }
        TLExpr::ForAll { var, domain, body } => {
            buf.push_str("(ForAll ");
            write_quoted(var, buf);
            buf.push(' ');
            write_quoted(domain, buf);
            buf.push(' ');
            write_sexpr(body, buf);
            buf.push(')');
        }
        TLExpr::Imply(a, b) => write_binary_sexpr("Imply", a, b, buf),
        TLExpr::Score(e) => write_unary_sexpr("Score", e, buf),
        TLExpr::Add(a, b) => write_binary_sexpr("Add", a, b, buf),
        TLExpr::Sub(a, b) => write_binary_sexpr("Sub", a, b, buf),
        TLExpr::Mul(a, b) => write_binary_sexpr("Mul", a, b, buf),
        TLExpr::Div(a, b) => write_binary_sexpr("Div", a, b, buf),
        TLExpr::Pow(a, b) => write_binary_sexpr("Pow", a, b, buf),
        TLExpr::Mod(a, b) => write_binary_sexpr("Mod", a, b, buf),
        TLExpr::Min(a, b) => write_binary_sexpr("Min", a, b, buf),
        TLExpr::Max(a, b) => write_binary_sexpr("Max", a, b, buf),
        TLExpr::Abs(e) => write_unary_sexpr("Abs", e, buf),
        TLExpr::Floor(e) => write_unary_sexpr("Floor", e, buf),
        TLExpr::Ceil(e) => write_unary_sexpr("Ceil", e, buf),
        TLExpr::Round(e) => write_unary_sexpr("Round", e, buf),
        TLExpr::Sqrt(e) => write_unary_sexpr("Sqrt", e, buf),
        TLExpr::Exp(e) => write_unary_sexpr("Exp", e, buf),
        TLExpr::Log(e) => write_unary_sexpr("Log", e, buf),
        TLExpr::Sin(e) => write_unary_sexpr("Sin", e, buf),
        TLExpr::Cos(e) => write_unary_sexpr("Cos", e, buf),
        TLExpr::Tan(e) => write_unary_sexpr("Tan", e, buf),
        TLExpr::Eq(a, b) => write_binary_sexpr("Eq", a, b, buf),
        TLExpr::Lt(a, b) => write_binary_sexpr("Lt", a, b, buf),
        TLExpr::Gt(a, b) => write_binary_sexpr("Gt", a, b, buf),
        TLExpr::Lte(a, b) => write_binary_sexpr("Lte", a, b, buf),
        TLExpr::Gte(a, b) => write_binary_sexpr("Gte", a, b, buf),
        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => {
            buf.push_str("(IfThenElse ");
            write_sexpr(condition, buf);
            buf.push(' ');
            write_sexpr(then_branch, buf);
            buf.push(' ');
            write_sexpr(else_branch, buf);
            buf.push(')');
        }
        TLExpr::Constant(v) => {
            buf.push_str("(Constant ");
            buf.push_str(&format!("{v}"));
            buf.push(')');
        }
        TLExpr::Aggregate {
            op,
            var,
            domain,
            body,
            group_by,
        } => {
            buf.push_str("(Aggregate ");
            buf.push_str(aggregate_op_name(op));
            buf.push(' ');
            write_quoted(var, buf);
            buf.push(' ');
            write_quoted(domain, buf);
            buf.push(' ');
            write_sexpr(body, buf);
            if let Some(gb) = group_by {
                buf.push_str(" (GroupBy");
                for g in gb {
                    buf.push(' ');
                    write_quoted(g, buf);
                }
                buf.push(')');
            }
            buf.push(')');
        }
        TLExpr::Let { var, value, body } => {
            buf.push_str("(Let ");
            write_quoted(var, buf);
            buf.push(' ');
            write_sexpr(value, buf);
            buf.push(' ');
            write_sexpr(body, buf);
            buf.push(')');
        }
        TLExpr::Box(e) => write_unary_sexpr("Box", e, buf),
        TLExpr::Diamond(e) => write_unary_sexpr("Diamond", e, buf),
        TLExpr::Next(e) => write_unary_sexpr("Next", e, buf),
        TLExpr::Eventually(e) => write_unary_sexpr("Eventually", e, buf),
        TLExpr::Always(e) => write_unary_sexpr("Always", e, buf),
        TLExpr::Until { before, after } => write_binary_sexpr("Until", before, after, buf),
        TLExpr::TNorm { kind, left, right } => {
            buf.push_str("(TNorm ");
            buf.push_str(tnorm_kind_name(kind));
            buf.push(' ');
            write_sexpr(left, buf);
            buf.push(' ');
            write_sexpr(right, buf);
            buf.push(')');
        }
        TLExpr::TCoNorm { kind, left, right } => {
            buf.push_str("(TCoNorm ");
            buf.push_str(tconorm_kind_name(kind));
            buf.push(' ');
            write_sexpr(left, buf);
            buf.push(' ');
            write_sexpr(right, buf);
            buf.push(')');
        }
        TLExpr::FuzzyNot { kind, expr: e } => {
            buf.push_str("(FuzzyNot ");
            write_fuzzy_neg_kind(kind, buf);
            buf.push(' ');
            write_sexpr(e, buf);
            buf.push(')');
        }
        TLExpr::FuzzyImplication {
            kind,
            premise,
            conclusion,
        } => {
            buf.push_str("(FuzzyImplication ");
            buf.push_str(fuzzy_imp_kind_name(kind));
            buf.push(' ');
            write_sexpr(premise, buf);
            buf.push(' ');
            write_sexpr(conclusion, buf);
            buf.push(')');
        }
        TLExpr::SoftExists {
            var,
            domain,
            body,
            temperature,
        } => {
            buf.push_str("(SoftExists ");
            write_quoted(var, buf);
            buf.push(' ');
            write_quoted(domain, buf);
            buf.push(' ');
            buf.push_str(&format!("{temperature}"));
            buf.push(' ');
            write_sexpr(body, buf);
            buf.push(')');
        }
        TLExpr::SoftForAll {
            var,
            domain,
            body,
            temperature,
        } => {
            buf.push_str("(SoftForAll ");
            write_quoted(var, buf);
            buf.push(' ');
            write_quoted(domain, buf);
            buf.push(' ');
            buf.push_str(&format!("{temperature}"));
            buf.push(' ');
            write_sexpr(body, buf);
            buf.push(')');
        }
        TLExpr::WeightedRule { weight, rule } => {
            buf.push_str("(WeightedRule ");
            buf.push_str(&format!("{weight}"));
            buf.push(' ');
            write_sexpr(rule, buf);
            buf.push(')');
        }
        TLExpr::ProbabilisticChoice { alternatives } => {
            buf.push_str("(ProbabilisticChoice");
            for (prob, alt_expr) in alternatives {
                buf.push_str(" (");
                buf.push_str(&format!("{prob}"));
                buf.push(' ');
                write_sexpr(alt_expr, buf);
                buf.push(')');
            }
            buf.push(')');
        }
        TLExpr::Release { released, releaser } => {
            write_binary_sexpr("Release", released, releaser, buf);
        }
        TLExpr::WeakUntil { before, after } => {
            write_binary_sexpr("WeakUntil", before, after, buf);
        }
        TLExpr::StrongRelease { released, releaser } => {
            write_binary_sexpr("StrongRelease", released, releaser, buf);
        }
        TLExpr::Lambda {
            var,
            var_type,
            body,
        } => {
            buf.push_str("(Lambda ");
            write_quoted(var, buf);
            buf.push(' ');
            match var_type {
                Some(t) => write_quoted(t, buf),
                None => buf.push_str("None"),
            }
            buf.push(' ');
            write_sexpr(body, buf);
            buf.push(')');
        }
        TLExpr::Apply { function, argument } => {
            write_binary_sexpr("Apply", function, argument, buf);
        }
        TLExpr::SetMembership { element, set } => {
            write_binary_sexpr("SetMembership", element, set, buf);
        }
        TLExpr::SetUnion { left, right } => write_binary_sexpr("SetUnion", left, right, buf),
        TLExpr::SetIntersection { left, right } => {
            write_binary_sexpr("SetIntersection", left, right, buf);
        }
        TLExpr::SetDifference { left, right } => {
            write_binary_sexpr("SetDifference", left, right, buf);
        }
        TLExpr::SetCardinality { set } => write_unary_sexpr("SetCardinality", set, buf),
        TLExpr::EmptySet => buf.push_str("(EmptySet)"),
        TLExpr::SetComprehension {
            var,
            domain,
            condition,
        } => {
            buf.push_str("(SetComprehension ");
            write_quoted(var, buf);
            buf.push(' ');
            write_quoted(domain, buf);
            buf.push(' ');
            write_sexpr(condition, buf);
            buf.push(')');
        }
        TLExpr::CountingExists {
            var,
            domain,
            body,
            min_count,
        } => {
            buf.push_str("(CountingExists ");
            write_quoted(var, buf);
            buf.push(' ');
            write_quoted(domain, buf);
            buf.push(' ');
            buf.push_str(&format!("{min_count}"));
            buf.push(' ');
            write_sexpr(body, buf);
            buf.push(')');
        }
        TLExpr::CountingForAll {
            var,
            domain,
            body,
            min_count,
        } => {
            buf.push_str("(CountingForAll ");
            write_quoted(var, buf);
            buf.push(' ');
            write_quoted(domain, buf);
            buf.push(' ');
            buf.push_str(&format!("{min_count}"));
            buf.push(' ');
            write_sexpr(body, buf);
            buf.push(')');
        }
        TLExpr::ExactCount {
            var,
            domain,
            body,
            count,
        } => {
            buf.push_str("(ExactCount ");
            write_quoted(var, buf);
            buf.push(' ');
            write_quoted(domain, buf);
            buf.push(' ');
            buf.push_str(&format!("{count}"));
            buf.push(' ');
            write_sexpr(body, buf);
            buf.push(')');
        }
        TLExpr::Majority { var, domain, body } => {
            buf.push_str("(Majority ");
            write_quoted(var, buf);
            buf.push(' ');
            write_quoted(domain, buf);
            buf.push(' ');
            write_sexpr(body, buf);
            buf.push(')');
        }
        TLExpr::LeastFixpoint { var, body } => {
            buf.push_str("(LeastFixpoint ");
            write_quoted(var, buf);
            buf.push(' ');
            write_sexpr(body, buf);
            buf.push(')');
        }
        TLExpr::GreatestFixpoint { var, body } => {
            buf.push_str("(GreatestFixpoint ");
            write_quoted(var, buf);
            buf.push(' ');
            write_sexpr(body, buf);
            buf.push(')');
        }
        TLExpr::Nominal { name } => {
            buf.push_str("(Nominal ");
            write_quoted(name, buf);
            buf.push(')');
        }
        TLExpr::At { nominal, formula } => {
            buf.push_str("(At ");
            write_quoted(nominal, buf);
            buf.push(' ');
            write_sexpr(formula, buf);
            buf.push(')');
        }
        TLExpr::Somewhere { formula } => write_unary_sexpr("Somewhere", formula, buf),
        TLExpr::Everywhere { formula } => write_unary_sexpr("Everywhere", formula, buf),
        TLExpr::AllDifferent { variables } => {
            buf.push_str("(AllDifferent");
            for v in variables {
                buf.push(' ');
                write_quoted(v, buf);
            }
            buf.push(')');
        }
        TLExpr::GlobalCardinality {
            variables,
            values,
            min_occurrences,
            max_occurrences,
        } => {
            buf.push_str("(GlobalCardinality (Vars");
            for v in variables {
                buf.push(' ');
                write_quoted(v, buf);
            }
            buf.push_str(") (Values");
            for val in values {
                buf.push(' ');
                write_sexpr(val, buf);
            }
            buf.push_str(") (MinOcc");
            for m in min_occurrences {
                buf.push(' ');
                buf.push_str(&format!("{m}"));
            }
            buf.push_str(") (MaxOcc");
            for m in max_occurrences {
                buf.push(' ');
                buf.push_str(&format!("{m}"));
            }
            buf.push_str("))");
        }
        TLExpr::Abducible { name, cost } => {
            buf.push_str("(Abducible ");
            write_quoted(name, buf);
            buf.push(' ');
            buf.push_str(&format!("{cost}"));
            buf.push(')');
        }
        TLExpr::Explain { formula } => write_unary_sexpr("Explain", formula, buf),
        TLExpr::SymbolLiteral(s) => {
            buf.push_str("(SymbolLiteral ");
            write_quoted(s, buf);
            buf.push(')');
        }
        TLExpr::Match { scrutinee, arms } => {
            buf.push_str("(Match ");
            write_sexpr(scrutinee, buf);
            for (pat, body) in arms {
                buf.push_str(" (");
                match pat {
                    crate::pattern::MatchPattern::ConstSymbol(s) => {
                        buf.push_str("Symbol ");
                        write_quoted(s, buf);
                    }
                    crate::pattern::MatchPattern::ConstNumber(n) => {
                        buf.push_str("Num ");
                        buf.push_str(&format!("{n}"));
                    }
                    crate::pattern::MatchPattern::Wildcard => {
                        buf.push('_');
                    }
                }
                buf.push(' ');
                write_sexpr(body, buf);
                buf.push(')');
            }
            buf.push(')');
        }
    }
}

fn write_unary_sexpr(tag: &str, child: &TLExpr, buf: &mut String) {
    buf.push('(');
    buf.push_str(tag);
    buf.push(' ');
    write_sexpr(child, buf);
    buf.push(')');
}

fn write_binary_sexpr(tag: &str, a: &TLExpr, b: &TLExpr, buf: &mut String) {
    buf.push('(');
    buf.push_str(tag);
    buf.push(' ');
    write_sexpr(a, buf);
    buf.push(' ');
    write_sexpr(b, buf);
    buf.push(')');
}

fn write_quoted(s: &str, buf: &mut String) {
    buf.push('"');
    for ch in s.chars() {
        match ch {
            '"' => buf.push_str("\\\""),
            '\\' => buf.push_str("\\\\"),
            _ => buf.push(ch),
        }
    }
    buf.push('"');
}

fn write_term_sexpr(term: &Term, buf: &mut String) {
    match term {
        Term::Var(name) => {
            buf.push_str("(Var ");
            write_quoted(name, buf);
            buf.push(')');
        }
        Term::Const(name) => {
            buf.push_str("(Const ");
            write_quoted(name, buf);
            buf.push(')');
        }
        Term::Typed {
            value,
            type_annotation,
        } => {
            buf.push_str("(Typed ");
            write_term_sexpr(value, buf);
            buf.push(' ');
            write_quoted(&type_annotation.type_name, buf);
            buf.push(')');
        }
    }
}

fn aggregate_op_name(op: &AggregateOp) -> &'static str {
    match op {
        AggregateOp::Count => "Count",
        AggregateOp::Sum => "Sum",
        AggregateOp::Average => "Average",
        AggregateOp::Max => "Max",
        AggregateOp::Min => "Min",
        AggregateOp::Product => "Product",
        AggregateOp::Any => "Any",
        AggregateOp::All => "All",
    }
}

fn tnorm_kind_name(kind: &TNormKind) -> &'static str {
    match kind {
        TNormKind::Minimum => "Minimum",
        TNormKind::Product => "Product",
        TNormKind::Lukasiewicz => "Lukasiewicz",
        TNormKind::Drastic => "Drastic",
        TNormKind::NilpotentMinimum => "NilpotentMinimum",
        TNormKind::Hamacher => "Hamacher",
    }
}

fn tconorm_kind_name(kind: &TCoNormKind) -> &'static str {
    match kind {
        TCoNormKind::Maximum => "Maximum",
        TCoNormKind::ProbabilisticSum => "ProbabilisticSum",
        TCoNormKind::BoundedSum => "BoundedSum",
        TCoNormKind::Drastic => "Drastic",
        TCoNormKind::NilpotentMaximum => "NilpotentMaximum",
        TCoNormKind::Hamacher => "Hamacher",
    }
}

fn write_fuzzy_neg_kind(kind: &FuzzyNegationKind, buf: &mut String) {
    match kind {
        FuzzyNegationKind::Standard => buf.push_str("Standard"),
        FuzzyNegationKind::Sugeno { lambda } => {
            buf.push_str(&format!("(Sugeno {lambda})"));
        }
        FuzzyNegationKind::Yager { w } => {
            buf.push_str(&format!("(Yager {w})"));
        }
    }
}

fn fuzzy_imp_kind_name(kind: &FuzzyImplicationKind) -> &'static str {
    match kind {
        FuzzyImplicationKind::Godel => "Godel",
        FuzzyImplicationKind::Lukasiewicz => "Lukasiewicz",
        FuzzyImplicationKind::Reichenbach => "Reichenbach",
        FuzzyImplicationKind::KleeneDienes => "KleeneDienes",
        FuzzyImplicationKind::Rescher => "Rescher",
        FuzzyImplicationKind::Goguen => "Goguen",
    }
}

// ============================================================================
// S-expression parser (recursive descent)
// ============================================================================

/// Parse a `TLExpr` from an S-expression string.
pub fn from_sexpr(input: &str) -> Result<TLExpr, ExprSerializeError> {
    let tokens = tokenize_sexpr(input)?;
    let mut pos = 0;
    let result = parse_expr(&tokens, &mut pos)?;
    if pos < tokens.len() {
        return Err(ExprSerializeError::FormatError(
            "Unexpected trailing tokens".to_string(),
        ));
    }
    Ok(result)
}

#[derive(Debug, Clone)]
enum SToken {
    LParen,
    RParen,
    Str(String),
    Ident(String),
    Num(f64),
}

fn tokenize_sexpr(input: &str) -> Result<Vec<SToken>, ExprSerializeError> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            ' ' | '\t' | '\n' | '\r' => i += 1,
            '(' => {
                tokens.push(SToken::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(SToken::RParen);
                i += 1;
            }
            '"' => {
                i += 1;
                let mut s = String::new();
                while i < chars.len() && chars[i] != '"' {
                    if chars[i] == '\\' && i + 1 < chars.len() {
                        i += 1;
                        s.push(chars[i]);
                    } else {
                        s.push(chars[i]);
                    }
                    i += 1;
                }
                if i >= chars.len() {
                    return Err(ExprSerializeError::FormatError(
                        "Unterminated string".to_string(),
                    ));
                }
                i += 1; // skip closing quote
                tokens.push(SToken::Str(s));
            }
            c if c == '-' || c == '+' || c.is_ascii_digit() => {
                let start = i;
                if c == '-' || c == '+' {
                    i += 1;
                }
                // Check if it's actually a number
                if i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    while i < chars.len()
                        && (chars[i].is_ascii_digit()
                            || chars[i] == '.'
                            || chars[i] == 'e'
                            || chars[i] == 'E')
                    {
                        i += 1;
                    }
                    let num_str: String = chars[start..i].iter().collect();
                    let val: f64 = num_str.parse().map_err(|e: std::num::ParseFloatError| {
                        ExprSerializeError::FormatError(format!("Invalid number '{num_str}': {e}"))
                    })?;
                    tokens.push(SToken::Num(val));
                } else {
                    // It's an identifier starting with - or +
                    while i < chars.len()
                        && !matches!(chars[i], ' ' | '\t' | '\n' | '\r' | '(' | ')' | '"')
                    {
                        i += 1;
                    }
                    let ident: String = chars[start..i].iter().collect();
                    tokens.push(SToken::Ident(ident));
                }
            }
            _ => {
                let start = i;
                while i < chars.len()
                    && !matches!(chars[i], ' ' | '\t' | '\n' | '\r' | '(' | ')' | '"')
                {
                    i += 1;
                }
                let ident: String = chars[start..i].iter().collect();
                tokens.push(SToken::Ident(ident));
            }
        }
    }
    Ok(tokens)
}

fn expect_lparen(tokens: &[SToken], pos: &mut usize) -> Result<(), ExprSerializeError> {
    if *pos >= tokens.len() {
        return Err(ExprSerializeError::TruncatedInput);
    }
    if matches!(tokens[*pos], SToken::LParen) {
        *pos += 1;
        Ok(())
    } else {
        Err(ExprSerializeError::FormatError("Expected '('".to_string()))
    }
}

fn expect_rparen(tokens: &[SToken], pos: &mut usize) -> Result<(), ExprSerializeError> {
    if *pos >= tokens.len() {
        return Err(ExprSerializeError::TruncatedInput);
    }
    if matches!(tokens[*pos], SToken::RParen) {
        *pos += 1;
        Ok(())
    } else {
        Err(ExprSerializeError::FormatError("Expected ')'".to_string()))
    }
}

fn read_ident(tokens: &[SToken], pos: &mut usize) -> Result<String, ExprSerializeError> {
    if *pos >= tokens.len() {
        return Err(ExprSerializeError::TruncatedInput);
    }
    if let SToken::Ident(s) = &tokens[*pos] {
        let result = s.clone();
        *pos += 1;
        Ok(result)
    } else {
        Err(ExprSerializeError::FormatError(format!(
            "Expected identifier, got {:?}",
            tokens[*pos]
        )))
    }
}

fn read_string(tokens: &[SToken], pos: &mut usize) -> Result<String, ExprSerializeError> {
    if *pos >= tokens.len() {
        return Err(ExprSerializeError::TruncatedInput);
    }
    if let SToken::Str(s) = &tokens[*pos] {
        let result = s.clone();
        *pos += 1;
        Ok(result)
    } else {
        Err(ExprSerializeError::FormatError(format!(
            "Expected string, got {:?}",
            tokens[*pos]
        )))
    }
}

fn read_num(tokens: &[SToken], pos: &mut usize) -> Result<f64, ExprSerializeError> {
    if *pos >= tokens.len() {
        return Err(ExprSerializeError::TruncatedInput);
    }
    match &tokens[*pos] {
        SToken::Num(v) => {
            let result = *v;
            *pos += 1;
            Ok(result)
        }
        SToken::Ident(s) => {
            // Try to parse as number (handles edge cases)
            let result: f64 = s.parse().map_err(|_| {
                ExprSerializeError::FormatError(format!("Expected number, got '{s}'"))
            })?;
            *pos += 1;
            Ok(result)
        }
        other => Err(ExprSerializeError::FormatError(format!(
            "Expected number, got {other:?}"
        ))),
    }
}

fn read_usize(tokens: &[SToken], pos: &mut usize) -> Result<usize, ExprSerializeError> {
    let v = read_num(tokens, pos)?;
    Ok(v as usize)
}

fn at_rparen(tokens: &[SToken], pos: usize) -> bool {
    pos < tokens.len() && matches!(tokens[pos], SToken::RParen)
}

fn parse_expr(tokens: &[SToken], pos: &mut usize) -> Result<TLExpr, ExprSerializeError> {
    expect_lparen(tokens, pos)?;
    let tag = read_ident(tokens, pos)?;
    let result = match tag.as_str() {
        "Pred" => {
            let name = read_string(tokens, pos)?;
            let mut args = Vec::new();
            while !at_rparen(tokens, *pos) {
                args.push(parse_term(tokens, pos)?);
            }
            TLExpr::Pred { name, args }
        }
        "And" => {
            let a = parse_expr(tokens, pos)?;
            let b = parse_expr(tokens, pos)?;
            TLExpr::And(Box::new(a), Box::new(b))
        }
        "Or" => {
            let a = parse_expr(tokens, pos)?;
            let b = parse_expr(tokens, pos)?;
            TLExpr::Or(Box::new(a), Box::new(b))
        }
        "Not" => {
            let e = parse_expr(tokens, pos)?;
            TLExpr::Not(Box::new(e))
        }
        "Exists" => {
            let var = read_string(tokens, pos)?;
            let domain = read_string(tokens, pos)?;
            let body = parse_expr(tokens, pos)?;
            TLExpr::Exists {
                var,
                domain,
                body: Box::new(body),
            }
        }
        "ForAll" => {
            let var = read_string(tokens, pos)?;
            let domain = read_string(tokens, pos)?;
            let body = parse_expr(tokens, pos)?;
            TLExpr::ForAll {
                var,
                domain,
                body: Box::new(body),
            }
        }
        "Imply" => {
            let a = parse_expr(tokens, pos)?;
            let b = parse_expr(tokens, pos)?;
            TLExpr::Imply(Box::new(a), Box::new(b))
        }
        "Score" => {
            let e = parse_expr(tokens, pos)?;
            TLExpr::Score(Box::new(e))
        }
        "Add" => parse_binary_op(tokens, pos, TLExpr::Add)?,
        "Sub" => parse_binary_op(tokens, pos, TLExpr::Sub)?,
        "Mul" => parse_binary_op(tokens, pos, TLExpr::Mul)?,
        "Div" => parse_binary_op(tokens, pos, TLExpr::Div)?,
        "Pow" => parse_binary_op(tokens, pos, TLExpr::Pow)?,
        "Mod" => parse_binary_op(tokens, pos, TLExpr::Mod)?,
        "Min" => parse_binary_op(tokens, pos, TLExpr::Min)?,
        "Max" => parse_binary_op(tokens, pos, TLExpr::Max)?,
        "Abs" => parse_unary_op(tokens, pos, TLExpr::Abs)?,
        "Floor" => parse_unary_op(tokens, pos, TLExpr::Floor)?,
        "Ceil" => parse_unary_op(tokens, pos, TLExpr::Ceil)?,
        "Round" => parse_unary_op(tokens, pos, TLExpr::Round)?,
        "Sqrt" => parse_unary_op(tokens, pos, TLExpr::Sqrt)?,
        "Exp" => parse_unary_op(tokens, pos, TLExpr::Exp)?,
        "Log" => parse_unary_op(tokens, pos, TLExpr::Log)?,
        "Sin" => parse_unary_op(tokens, pos, TLExpr::Sin)?,
        "Cos" => parse_unary_op(tokens, pos, TLExpr::Cos)?,
        "Tan" => parse_unary_op(tokens, pos, TLExpr::Tan)?,
        "Eq" => parse_binary_op(tokens, pos, TLExpr::Eq)?,
        "Lt" => parse_binary_op(tokens, pos, TLExpr::Lt)?,
        "Gt" => parse_binary_op(tokens, pos, TLExpr::Gt)?,
        "Lte" => parse_binary_op(tokens, pos, TLExpr::Lte)?,
        "Gte" => parse_binary_op(tokens, pos, TLExpr::Gte)?,
        "IfThenElse" => {
            let cond = parse_expr(tokens, pos)?;
            let then_b = parse_expr(tokens, pos)?;
            let else_b = parse_expr(tokens, pos)?;
            TLExpr::IfThenElse {
                condition: Box::new(cond),
                then_branch: Box::new(then_b),
                else_branch: Box::new(else_b),
            }
        }
        "Constant" => {
            let v = read_num(tokens, pos)?;
            TLExpr::Constant(v)
        }
        "Aggregate" => {
            let op_name = read_ident(tokens, pos)?;
            let op = parse_aggregate_op(&op_name)?;
            let var = read_string(tokens, pos)?;
            let domain = read_string(tokens, pos)?;
            let body = parse_expr(tokens, pos)?;
            let group_by = if !at_rparen(tokens, *pos) {
                // parse (GroupBy ...)
                expect_lparen(tokens, pos)?;
                let gb_tag = read_ident(tokens, pos)?;
                if gb_tag != "GroupBy" {
                    return Err(ExprSerializeError::FormatError(format!(
                        "Expected GroupBy, got {gb_tag}"
                    )));
                }
                let mut gb = Vec::new();
                while !at_rparen(tokens, *pos) {
                    gb.push(read_string(tokens, pos)?);
                }
                expect_rparen(tokens, pos)?;
                Some(gb)
            } else {
                None
            };
            TLExpr::Aggregate {
                op,
                var,
                domain,
                body: Box::new(body),
                group_by,
            }
        }
        "Let" => {
            let var = read_string(tokens, pos)?;
            let value = parse_expr(tokens, pos)?;
            let body = parse_expr(tokens, pos)?;
            TLExpr::Let {
                var,
                value: Box::new(value),
                body: Box::new(body),
            }
        }
        "Box" => parse_unary_op(tokens, pos, TLExpr::Box)?,
        "Diamond" => parse_unary_op(tokens, pos, TLExpr::Diamond)?,
        "Next" => parse_unary_op(tokens, pos, TLExpr::Next)?,
        "Eventually" => parse_unary_op(tokens, pos, TLExpr::Eventually)?,
        "Always" => parse_unary_op(tokens, pos, TLExpr::Always)?,
        "Until" => {
            let a = parse_expr(tokens, pos)?;
            let b = parse_expr(tokens, pos)?;
            TLExpr::Until {
                before: Box::new(a),
                after: Box::new(b),
            }
        }
        "TNorm" => {
            let kind_name = read_ident(tokens, pos)?;
            let kind = parse_tnorm_kind(&kind_name)?;
            let left = parse_expr(tokens, pos)?;
            let right = parse_expr(tokens, pos)?;
            TLExpr::TNorm {
                kind,
                left: Box::new(left),
                right: Box::new(right),
            }
        }
        "TCoNorm" => {
            let kind_name = read_ident(tokens, pos)?;
            let kind = parse_tconorm_kind(&kind_name)?;
            let left = parse_expr(tokens, pos)?;
            let right = parse_expr(tokens, pos)?;
            TLExpr::TCoNorm {
                kind,
                left: Box::new(left),
                right: Box::new(right),
            }
        }
        "FuzzyNot" => {
            let kind = parse_fuzzy_neg_kind(tokens, pos)?;
            let e = parse_expr(tokens, pos)?;
            TLExpr::FuzzyNot {
                kind,
                expr: Box::new(e),
            }
        }
        "FuzzyImplication" => {
            let kind_name = read_ident(tokens, pos)?;
            let kind = parse_fuzzy_imp_kind(&kind_name)?;
            let premise = parse_expr(tokens, pos)?;
            let conclusion = parse_expr(tokens, pos)?;
            TLExpr::FuzzyImplication {
                kind,
                premise: Box::new(premise),
                conclusion: Box::new(conclusion),
            }
        }
        "SoftExists" => {
            let var = read_string(tokens, pos)?;
            let domain = read_string(tokens, pos)?;
            let temperature = read_num(tokens, pos)?;
            let body = parse_expr(tokens, pos)?;
            TLExpr::SoftExists {
                var,
                domain,
                body: Box::new(body),
                temperature,
            }
        }
        "SoftForAll" => {
            let var = read_string(tokens, pos)?;
            let domain = read_string(tokens, pos)?;
            let temperature = read_num(tokens, pos)?;
            let body = parse_expr(tokens, pos)?;
            TLExpr::SoftForAll {
                var,
                domain,
                body: Box::new(body),
                temperature,
            }
        }
        "WeightedRule" => {
            let weight = read_num(tokens, pos)?;
            let rule = parse_expr(tokens, pos)?;
            TLExpr::WeightedRule {
                weight,
                rule: Box::new(rule),
            }
        }
        "ProbabilisticChoice" => {
            let mut alts = Vec::new();
            while !at_rparen(tokens, *pos) {
                expect_lparen(tokens, pos)?;
                let prob = read_num(tokens, pos)?;
                let alt_expr = parse_expr(tokens, pos)?;
                expect_rparen(tokens, pos)?;
                alts.push((prob, alt_expr));
            }
            TLExpr::ProbabilisticChoice { alternatives: alts }
        }
        "Release" => {
            let a = parse_expr(tokens, pos)?;
            let b = parse_expr(tokens, pos)?;
            TLExpr::Release {
                released: Box::new(a),
                releaser: Box::new(b),
            }
        }
        "WeakUntil" => {
            let a = parse_expr(tokens, pos)?;
            let b = parse_expr(tokens, pos)?;
            TLExpr::WeakUntil {
                before: Box::new(a),
                after: Box::new(b),
            }
        }
        "StrongRelease" => {
            let a = parse_expr(tokens, pos)?;
            let b = parse_expr(tokens, pos)?;
            TLExpr::StrongRelease {
                released: Box::new(a),
                releaser: Box::new(b),
            }
        }
        "Lambda" => {
            let var = read_string(tokens, pos)?;
            let var_type = if *pos < tokens.len() {
                match &tokens[*pos] {
                    SToken::Ident(s) if s == "None" => {
                        *pos += 1;
                        None
                    }
                    SToken::Str(s) => {
                        let result = s.clone();
                        *pos += 1;
                        Some(result)
                    }
                    _ => None,
                }
            } else {
                None
            };
            let body = parse_expr(tokens, pos)?;
            TLExpr::Lambda {
                var,
                var_type,
                body: Box::new(body),
            }
        }
        "Apply" => parse_binary_op(tokens, pos, |a, b| TLExpr::Apply {
            function: a,
            argument: b,
        })?,
        "SetMembership" => parse_binary_op(tokens, pos, |a, b| TLExpr::SetMembership {
            element: a,
            set: b,
        })?,
        "SetUnion" => parse_binary_op(tokens, pos, |a, b| TLExpr::SetUnion { left: a, right: b })?,
        "SetIntersection" => parse_binary_op(tokens, pos, |a, b| TLExpr::SetIntersection {
            left: a,
            right: b,
        })?,
        "SetDifference" => parse_binary_op(tokens, pos, |a, b| TLExpr::SetDifference {
            left: a,
            right: b,
        })?,
        "SetCardinality" => parse_unary_op(tokens, pos, |e| TLExpr::SetCardinality { set: e })?,
        "EmptySet" => TLExpr::EmptySet,
        "SetComprehension" => {
            let var = read_string(tokens, pos)?;
            let domain = read_string(tokens, pos)?;
            let condition = parse_expr(tokens, pos)?;
            TLExpr::SetComprehension {
                var,
                domain,
                condition: Box::new(condition),
            }
        }
        "CountingExists" => {
            let var = read_string(tokens, pos)?;
            let domain = read_string(tokens, pos)?;
            let min_count = read_usize(tokens, pos)?;
            let body = parse_expr(tokens, pos)?;
            TLExpr::CountingExists {
                var,
                domain,
                body: Box::new(body),
                min_count,
            }
        }
        "CountingForAll" => {
            let var = read_string(tokens, pos)?;
            let domain = read_string(tokens, pos)?;
            let min_count = read_usize(tokens, pos)?;
            let body = parse_expr(tokens, pos)?;
            TLExpr::CountingForAll {
                var,
                domain,
                body: Box::new(body),
                min_count,
            }
        }
        "ExactCount" => {
            let var = read_string(tokens, pos)?;
            let domain = read_string(tokens, pos)?;
            let count = read_usize(tokens, pos)?;
            let body = parse_expr(tokens, pos)?;
            TLExpr::ExactCount {
                var,
                domain,
                body: Box::new(body),
                count,
            }
        }
        "Majority" => {
            let var = read_string(tokens, pos)?;
            let domain = read_string(tokens, pos)?;
            let body = parse_expr(tokens, pos)?;
            TLExpr::Majority {
                var,
                domain,
                body: Box::new(body),
            }
        }
        "LeastFixpoint" => {
            let var = read_string(tokens, pos)?;
            let body = parse_expr(tokens, pos)?;
            TLExpr::LeastFixpoint {
                var,
                body: Box::new(body),
            }
        }
        "GreatestFixpoint" => {
            let var = read_string(tokens, pos)?;
            let body = parse_expr(tokens, pos)?;
            TLExpr::GreatestFixpoint {
                var,
                body: Box::new(body),
            }
        }
        "Nominal" => {
            let name = read_string(tokens, pos)?;
            TLExpr::Nominal { name }
        }
        "At" => {
            let nominal = read_string(tokens, pos)?;
            let formula = parse_expr(tokens, pos)?;
            TLExpr::At {
                nominal,
                formula: Box::new(formula),
            }
        }
        "Somewhere" => parse_unary_op(tokens, pos, |e| TLExpr::Somewhere { formula: e })?,
        "Everywhere" => parse_unary_op(tokens, pos, |e| TLExpr::Everywhere { formula: e })?,
        "AllDifferent" => {
            let mut variables = Vec::new();
            while !at_rparen(tokens, *pos) {
                variables.push(read_string(tokens, pos)?);
            }
            TLExpr::AllDifferent { variables }
        }
        "GlobalCardinality" => {
            // (Vars ...)
            expect_lparen(tokens, pos)?;
            let _ = read_ident(tokens, pos)?; // "Vars"
            let mut variables = Vec::new();
            while !at_rparen(tokens, *pos) {
                variables.push(read_string(tokens, pos)?);
            }
            expect_rparen(tokens, pos)?;
            // (Values ...)
            expect_lparen(tokens, pos)?;
            let _ = read_ident(tokens, pos)?;
            let mut values = Vec::new();
            while !at_rparen(tokens, *pos) {
                values.push(parse_expr(tokens, pos)?);
            }
            expect_rparen(tokens, pos)?;
            // (MinOcc ...)
            expect_lparen(tokens, pos)?;
            let _ = read_ident(tokens, pos)?;
            let mut min_occurrences = Vec::new();
            while !at_rparen(tokens, *pos) {
                min_occurrences.push(read_usize(tokens, pos)?);
            }
            expect_rparen(tokens, pos)?;
            // (MaxOcc ...)
            expect_lparen(tokens, pos)?;
            let _ = read_ident(tokens, pos)?;
            let mut max_occurrences = Vec::new();
            while !at_rparen(tokens, *pos) {
                max_occurrences.push(read_usize(tokens, pos)?);
            }
            expect_rparen(tokens, pos)?;
            TLExpr::GlobalCardinality {
                variables,
                values,
                min_occurrences,
                max_occurrences,
            }
        }
        "Abducible" => {
            let name = read_string(tokens, pos)?;
            let cost = read_num(tokens, pos)?;
            TLExpr::Abducible { name, cost }
        }
        "Explain" => parse_unary_op(tokens, pos, |e| TLExpr::Explain { formula: e })?,
        "SymbolLiteral" => {
            let s = read_string(tokens, pos)?;
            TLExpr::SymbolLiteral(s)
        }
        "Match" => {
            let scrutinee = parse_expr(tokens, pos)?;
            let mut arms = Vec::new();
            while !at_rparen(tokens, *pos) {
                expect_lparen(tokens, pos)?;
                // The arm pattern tag is an Ident token
                if *pos >= tokens.len() {
                    return Err(ExprSerializeError::TruncatedInput);
                }
                let pat = match &tokens[*pos].clone() {
                    SToken::Ident(s) if s == "_" => {
                        *pos += 1;
                        crate::pattern::MatchPattern::Wildcard
                    }
                    SToken::Ident(s) if s == "Symbol" => {
                        *pos += 1;
                        let sym = read_string(tokens, pos)?;
                        crate::pattern::MatchPattern::ConstSymbol(sym)
                    }
                    SToken::Ident(s) if s == "Num" => {
                        *pos += 1;
                        let n = read_num(tokens, pos)?;
                        crate::pattern::MatchPattern::ConstNumber(n)
                    }
                    other => {
                        return Err(ExprSerializeError::FormatError(format!(
                            "Unknown pattern tag: {other:?}"
                        )));
                    }
                };
                let body = parse_expr(tokens, pos)?;
                expect_rparen(tokens, pos)?;
                arms.push((pat, Box::new(body)));
            }
            TLExpr::Match {
                scrutinee: Box::new(scrutinee),
                arms,
            }
        }
        other => return Err(ExprSerializeError::UnknownVariant(other.to_string())),
    };
    expect_rparen(tokens, pos)?;
    Ok(result)
}

fn parse_unary_op(
    tokens: &[SToken],
    pos: &mut usize,
    ctor: impl FnOnce(Box<TLExpr>) -> TLExpr,
) -> Result<TLExpr, ExprSerializeError> {
    let e = parse_expr(tokens, pos)?;
    Ok(ctor(Box::new(e)))
}

fn parse_binary_op(
    tokens: &[SToken],
    pos: &mut usize,
    ctor: impl FnOnce(Box<TLExpr>, Box<TLExpr>) -> TLExpr,
) -> Result<TLExpr, ExprSerializeError> {
    let a = parse_expr(tokens, pos)?;
    let b = parse_expr(tokens, pos)?;
    Ok(ctor(Box::new(a), Box::new(b)))
}

fn parse_term(tokens: &[SToken], pos: &mut usize) -> Result<Term, ExprSerializeError> {
    expect_lparen(tokens, pos)?;
    let tag = read_ident(tokens, pos)?;
    let result = match tag.as_str() {
        "Var" => {
            let name = read_string(tokens, pos)?;
            Term::Var(name)
        }
        "Const" => {
            let name = read_string(tokens, pos)?;
            Term::Const(name)
        }
        "Typed" => {
            let value = parse_term(tokens, pos)?;
            let type_name = read_string(tokens, pos)?;
            Term::Typed {
                value: Box::new(value),
                type_annotation: TypeAnnotation::new(type_name),
            }
        }
        other => return Err(ExprSerializeError::UnknownVariant(format!("Term::{other}"))),
    };
    expect_rparen(tokens, pos)?;
    Ok(result)
}

fn parse_aggregate_op(name: &str) -> Result<AggregateOp, ExprSerializeError> {
    match name {
        "Count" => Ok(AggregateOp::Count),
        "Sum" => Ok(AggregateOp::Sum),
        "Average" => Ok(AggregateOp::Average),
        "Max" => Ok(AggregateOp::Max),
        "Min" => Ok(AggregateOp::Min),
        "Product" => Ok(AggregateOp::Product),
        "Any" => Ok(AggregateOp::Any),
        "All" => Ok(AggregateOp::All),
        other => Err(ExprSerializeError::UnknownVariant(format!(
            "AggregateOp::{other}"
        ))),
    }
}

fn parse_tnorm_kind(name: &str) -> Result<TNormKind, ExprSerializeError> {
    match name {
        "Minimum" => Ok(TNormKind::Minimum),
        "Product" => Ok(TNormKind::Product),
        "Lukasiewicz" => Ok(TNormKind::Lukasiewicz),
        "Drastic" => Ok(TNormKind::Drastic),
        "NilpotentMinimum" => Ok(TNormKind::NilpotentMinimum),
        "Hamacher" => Ok(TNormKind::Hamacher),
        other => Err(ExprSerializeError::UnknownVariant(format!(
            "TNormKind::{other}"
        ))),
    }
}

fn parse_tconorm_kind(name: &str) -> Result<TCoNormKind, ExprSerializeError> {
    match name {
        "Maximum" => Ok(TCoNormKind::Maximum),
        "ProbabilisticSum" => Ok(TCoNormKind::ProbabilisticSum),
        "BoundedSum" => Ok(TCoNormKind::BoundedSum),
        "Drastic" => Ok(TCoNormKind::Drastic),
        "NilpotentMaximum" => Ok(TCoNormKind::NilpotentMaximum),
        "Hamacher" => Ok(TCoNormKind::Hamacher),
        other => Err(ExprSerializeError::UnknownVariant(format!(
            "TCoNormKind::{other}"
        ))),
    }
}

fn parse_fuzzy_neg_kind(
    tokens: &[SToken],
    pos: &mut usize,
) -> Result<FuzzyNegationKind, ExprSerializeError> {
    if *pos >= tokens.len() {
        return Err(ExprSerializeError::TruncatedInput);
    }
    match &tokens[*pos] {
        SToken::Ident(s) if s == "Standard" => {
            *pos += 1;
            Ok(FuzzyNegationKind::Standard)
        }
        SToken::LParen => {
            *pos += 1;
            let kind_name = read_ident(tokens, pos)?;
            match kind_name.as_str() {
                "Sugeno" => {
                    let lambda = read_num(tokens, pos)? as i32;
                    expect_rparen(tokens, pos)?;
                    Ok(FuzzyNegationKind::Sugeno { lambda })
                }
                "Yager" => {
                    let w = read_num(tokens, pos)? as u32;
                    expect_rparen(tokens, pos)?;
                    Ok(FuzzyNegationKind::Yager { w })
                }
                other => Err(ExprSerializeError::UnknownVariant(format!(
                    "FuzzyNegationKind::{other}"
                ))),
            }
        }
        other => Err(ExprSerializeError::FormatError(format!(
            "Expected fuzzy negation kind, got {other:?}"
        ))),
    }
}

fn parse_fuzzy_imp_kind(name: &str) -> Result<FuzzyImplicationKind, ExprSerializeError> {
    match name {
        "Godel" => Ok(FuzzyImplicationKind::Godel),
        "Lukasiewicz" => Ok(FuzzyImplicationKind::Lukasiewicz),
        "Reichenbach" => Ok(FuzzyImplicationKind::Reichenbach),
        "KleeneDienes" => Ok(FuzzyImplicationKind::KleeneDienes),
        "Rescher" => Ok(FuzzyImplicationKind::Rescher),
        "Goguen" => Ok(FuzzyImplicationKind::Goguen),
        other => Err(ExprSerializeError::UnknownVariant(format!(
            "FuzzyImplicationKind::{other}"
        ))),
    }
}

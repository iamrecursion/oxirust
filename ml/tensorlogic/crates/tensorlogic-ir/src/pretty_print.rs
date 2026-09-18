//! Human-readable pretty printing for TLExpr.
//!
//! Renders logical expressions using standard mathematical notation:
//! `forall x in Person. exists y in Person. knows(x, y) -> friends(x, y)`
//!
//! Supports both Unicode and ASCII symbol sets, configurable indentation,
//! maximum line width, and optional type annotations.

use std::fmt::Write;

use crate::{TLExpr, Term};

/// Symbol set for rendering logical connectives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymbolSet {
    /// Unicode mathematical symbols: `\u{2200}` `\u{2203}` `\u{2227}` `\u{2228}` `\u{00ac}` `\u{2192}` `\u{2208}`
    Unicode,
    /// ASCII-only symbols: `forall` `exists` `&` `|` `!` `->` `in`
    Ascii,
}

/// Configuration for pretty printing.
#[derive(Debug, Clone)]
pub struct PrettyConfig {
    /// Symbol set to use for rendering
    pub symbols: SymbolSet,
    /// Number of spaces per indentation level
    pub indent_width: usize,
    /// Maximum line width before wrapping (advisory)
    pub max_width: usize,
    /// Whether to show type annotations on terms
    pub show_types: bool,
}

impl Default for PrettyConfig {
    fn default() -> Self {
        Self {
            symbols: SymbolSet::Unicode,
            indent_width: 2,
            max_width: 80,
            show_types: false,
        }
    }
}

impl PrettyConfig {
    /// Create a Unicode-symbol configuration.
    pub fn unicode() -> Self {
        Self::default()
    }

    /// Create an ASCII-only configuration.
    pub fn ascii() -> Self {
        Self {
            symbols: SymbolSet::Ascii,
            ..Self::default()
        }
    }

    /// Set the indentation width.
    pub fn with_indent(mut self, w: usize) -> Self {
        self.indent_width = w;
        self
    }

    /// Set the maximum line width.
    pub fn with_max_width(mut self, w: usize) -> Self {
        self.max_width = w;
        self
    }

    /// Enable or disable type annotation display.
    pub fn with_types(mut self, v: bool) -> Self {
        self.show_types = v;
        self
    }

    // ---- symbol helpers ----

    fn sym_and(&self) -> &str {
        match self.symbols {
            SymbolSet::Unicode => "\u{2227}",
            SymbolSet::Ascii => "&",
        }
    }

    fn sym_or(&self) -> &str {
        match self.symbols {
            SymbolSet::Unicode => "\u{2228}",
            SymbolSet::Ascii => "|",
        }
    }

    fn sym_not(&self) -> &str {
        match self.symbols {
            SymbolSet::Unicode => "\u{00ac}",
            SymbolSet::Ascii => "!",
        }
    }

    fn sym_implies(&self) -> &str {
        match self.symbols {
            SymbolSet::Unicode => "\u{2192}",
            SymbolSet::Ascii => "->",
        }
    }

    fn sym_forall(&self) -> &str {
        match self.symbols {
            SymbolSet::Unicode => "\u{2200}",
            SymbolSet::Ascii => "forall",
        }
    }

    fn sym_exists(&self) -> &str {
        match self.symbols {
            SymbolSet::Unicode => "\u{2203}",
            SymbolSet::Ascii => "exists",
        }
    }

    fn sym_in(&self) -> &str {
        match self.symbols {
            SymbolSet::Unicode => "\u{2208}",
            SymbolSet::Ascii => "in",
        }
    }

    fn sym_box(&self) -> &str {
        match self.symbols {
            SymbolSet::Unicode => "\u{25a1}",
            SymbolSet::Ascii => "[]",
        }
    }

    fn sym_diamond(&self) -> &str {
        match self.symbols {
            SymbolSet::Unicode => "\u{25c7}",
            SymbolSet::Ascii => "<>",
        }
    }

    fn sym_lambda(&self) -> &str {
        match self.symbols {
            SymbolSet::Unicode => "\u{03bb}",
            SymbolSet::Ascii => "\\",
        }
    }

    fn sym_mu(&self) -> &str {
        match self.symbols {
            SymbolSet::Unicode => "\u{03bc}",
            SymbolSet::Ascii => "mu",
        }
    }

    fn sym_nu(&self) -> &str {
        match self.symbols {
            SymbolSet::Unicode => "\u{03bd}",
            SymbolSet::Ascii => "nu",
        }
    }

    fn sym_union(&self) -> &str {
        match self.symbols {
            SymbolSet::Unicode => "\u{222a}",
            SymbolSet::Ascii => "U",
        }
    }

    fn sym_intersection(&self) -> &str {
        match self.symbols {
            SymbolSet::Unicode => "\u{2229}",
            SymbolSet::Ascii => "^",
        }
    }

    fn sym_setminus(&self) -> &str {
        match self.symbols {
            SymbolSet::Unicode => "\\",
            SymbolSet::Ascii => "\\",
        }
    }

    fn sym_emptyset(&self) -> &str {
        match self.symbols {
            SymbolSet::Unicode => "\u{2205}",
            SymbolSet::Ascii => "{}",
        }
    }

    fn sym_element_of(&self) -> &str {
        // re-use ∈ for set membership
        self.sym_in()
    }

    fn sym_at(&self) -> &str {
        "@"
    }

    fn sym_sigma(&self) -> &str {
        match self.symbols {
            SymbolSet::Unicode => "\u{03c3}",
            SymbolSet::Ascii => "sigma",
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Pretty-print a [`TLExpr`] to a string with the given configuration.
pub fn pretty_print(expr: &TLExpr, config: &PrettyConfig) -> String {
    let mut buf = String::with_capacity(128);
    // write_expr never actually fails on a String sink
    let _ = write_expr(expr, config, &mut buf);
    buf
}

/// Pretty-print a [`TLExpr`] with the default Unicode configuration.
pub fn pretty(expr: &TLExpr) -> String {
    pretty_print(expr, &PrettyConfig::default())
}

/// Pretty-print a [`Term`] to a string (uses Unicode config).
pub fn pretty_term(term: &Term) -> String {
    pretty_term_with(term, &PrettyConfig::default())
}

/// Pretty-print a [`Term`] with the given configuration.
pub fn pretty_term_with(term: &Term, config: &PrettyConfig) -> String {
    let mut buf = String::with_capacity(32);
    let _ = write_term(term, config, &mut buf);
    buf
}

// ---------------------------------------------------------------------------
// Term rendering
// ---------------------------------------------------------------------------

fn write_term(term: &Term, config: &PrettyConfig, buf: &mut String) -> std::fmt::Result {
    match term {
        Term::Var(name) => write!(buf, "{name}"),
        Term::Const(name) => write!(buf, "{name}"),
        Term::Typed {
            value,
            type_annotation,
        } => {
            write_term(value, config, buf)?;
            if config.show_types {
                write!(buf, ":{}", type_annotation.type_name)?;
            }
            Ok(())
        }
    }
}

fn write_term_list(args: &[Term], config: &PrettyConfig, buf: &mut String) -> std::fmt::Result {
    for (i, arg) in args.iter().enumerate() {
        if i > 0 {
            write!(buf, ", ")?;
        }
        write_term(arg, config, buf)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Expression rendering
// ---------------------------------------------------------------------------

fn write_expr(expr: &TLExpr, cfg: &PrettyConfig, buf: &mut String) -> std::fmt::Result {
    match expr {
        // ---- predicates ----
        TLExpr::Pred { name, args } => {
            write!(buf, "{name}(")?;
            write_term_list(args, cfg, buf)?;
            write!(buf, ")")
        }

        // ---- logical connectives ----
        TLExpr::And(l, r) => write_binop(l, r, cfg, buf, cfg.sym_and()),
        TLExpr::Or(l, r) => write_binop(l, r, cfg, buf, cfg.sym_or()),
        TLExpr::Imply(l, r) => write_binop(l, r, cfg, buf, cfg.sym_implies()),
        TLExpr::Not(e) => {
            write!(buf, "{}", cfg.sym_not())?;
            write_subexpr(e, cfg, buf)
        }

        // ---- quantifiers ----
        TLExpr::Exists { var, domain, body } => {
            write!(
                buf,
                "{}{} {} {}. ",
                cfg.sym_exists(),
                var,
                cfg.sym_in(),
                domain
            )?;
            write_expr(body, cfg, buf)
        }
        TLExpr::ForAll { var, domain, body } => {
            write!(
                buf,
                "{}{} {} {}. ",
                cfg.sym_forall(),
                var,
                cfg.sym_in(),
                domain
            )?;
            write_expr(body, cfg, buf)
        }

        // ---- arithmetic ----
        TLExpr::Add(l, r) => write_binop(l, r, cfg, buf, "+"),
        TLExpr::Sub(l, r) => write_binop(l, r, cfg, buf, "-"),
        TLExpr::Mul(l, r) => write_binop(l, r, cfg, buf, "*"),
        TLExpr::Div(l, r) => write_binop(l, r, cfg, buf, "/"),
        TLExpr::Pow(l, r) => write_binop(l, r, cfg, buf, "^"),
        TLExpr::Mod(l, r) => write_binop(l, r, cfg, buf, "%"),
        TLExpr::Min(l, r) => write_func2("min", l, r, cfg, buf),
        TLExpr::Max(l, r) => write_func2("max", l, r, cfg, buf),

        // ---- unary math ----
        TLExpr::Abs(e) => write_func1("abs", e, cfg, buf),
        TLExpr::Floor(e) => write_func1("floor", e, cfg, buf),
        TLExpr::Ceil(e) => write_func1("ceil", e, cfg, buf),
        TLExpr::Round(e) => write_func1("round", e, cfg, buf),
        TLExpr::Sqrt(e) => write_func1("sqrt", e, cfg, buf),
        TLExpr::Exp(e) => write_func1("exp", e, cfg, buf),
        TLExpr::Log(e) => write_func1("log", e, cfg, buf),
        TLExpr::Sin(e) => write_func1("sin", e, cfg, buf),
        TLExpr::Cos(e) => write_func1("cos", e, cfg, buf),
        TLExpr::Tan(e) => write_func1("tan", e, cfg, buf),

        // ---- comparisons ----
        TLExpr::Eq(l, r) => write_binop(l, r, cfg, buf, "="),
        TLExpr::Lt(l, r) => write_binop(l, r, cfg, buf, "<"),
        TLExpr::Gt(l, r) => write_binop(l, r, cfg, buf, ">"),
        TLExpr::Lte(l, r) => {
            let sym = match cfg.symbols {
                SymbolSet::Unicode => "\u{2264}",
                SymbolSet::Ascii => "<=",
            };
            write_binop(l, r, cfg, buf, sym)
        }
        TLExpr::Gte(l, r) => {
            let sym = match cfg.symbols {
                SymbolSet::Unicode => "\u{2265}",
                SymbolSet::Ascii => ">=",
            };
            write_binop(l, r, cfg, buf, sym)
        }

        // ---- conditional ----
        TLExpr::IfThenElse {
            condition,
            then_branch,
            else_branch,
        } => {
            write!(buf, "if ")?;
            write_expr(condition, cfg, buf)?;
            write!(buf, " then ")?;
            write_expr(then_branch, cfg, buf)?;
            write!(buf, " else ")?;
            write_expr(else_branch, cfg, buf)
        }

        // ---- constants ----
        TLExpr::Constant(v) => {
            // Format without trailing zeros where possible
            if v.fract() == 0.0 && v.abs() < 1e15 {
                write!(buf, "{}", *v as i64)
            } else {
                write!(buf, "{v}")
            }
        }

        // ---- score ----
        TLExpr::Score(e) => {
            write!(buf, "{}(", cfg.sym_sigma())?;
            write_expr(e, cfg, buf)?;
            write!(buf, ")")
        }

        // ---- aggregation ----
        TLExpr::Aggregate {
            op,
            var,
            domain,
            body,
            group_by,
        } => {
            write!(buf, "{op:?}({var} {} {domain}", cfg.sym_in())?;
            if let Some(groups) = group_by {
                write!(buf, " group_by [{}]", groups.join(", "))?;
            }
            write!(buf, ". ")?;
            write_expr(body, cfg, buf)?;
            write!(buf, ")")
        }

        // ---- let binding ----
        TLExpr::Let { var, value, body } => {
            write!(buf, "let {var} = ")?;
            write_expr(value, cfg, buf)?;
            write!(buf, " in ")?;
            write_expr(body, cfg, buf)
        }

        // ---- modal logic ----
        TLExpr::Box(e) => {
            write!(buf, "{}", cfg.sym_box())?;
            write_subexpr(e, cfg, buf)
        }
        TLExpr::Diamond(e) => {
            write!(buf, "{}", cfg.sym_diamond())?;
            write_subexpr(e, cfg, buf)
        }

        // ---- temporal logic ----
        TLExpr::Next(e) => {
            write!(buf, "X ")?;
            write_subexpr(e, cfg, buf)
        }
        TLExpr::Eventually(e) => {
            write!(buf, "F ")?;
            write_subexpr(e, cfg, buf)
        }
        TLExpr::Always(e) => {
            write!(buf, "G ")?;
            write_subexpr(e, cfg, buf)
        }
        TLExpr::Until { before, after } => write_binop(before, after, cfg, buf, "U"),
        TLExpr::Release { released, releaser } => write_binop(released, releaser, cfg, buf, "R"),
        TLExpr::WeakUntil { before, after } => write_binop(before, after, cfg, buf, "W"),
        TLExpr::StrongRelease { released, releaser } => {
            write_binop(released, releaser, cfg, buf, "M")
        }

        // ---- fuzzy logic ----
        TLExpr::TNorm { kind, left, right } => {
            write!(buf, "T_{kind:?}(")?;
            write_expr(left, cfg, buf)?;
            write!(buf, ", ")?;
            write_expr(right, cfg, buf)?;
            write!(buf, ")")
        }
        TLExpr::TCoNorm { kind, left, right } => {
            write!(buf, "S_{kind:?}(")?;
            write_expr(left, cfg, buf)?;
            write!(buf, ", ")?;
            write_expr(right, cfg, buf)?;
            write!(buf, ")")
        }
        TLExpr::FuzzyNot { kind, expr } => {
            write!(buf, "FN_{kind:?}(")?;
            write_expr(expr, cfg, buf)?;
            write!(buf, ")")
        }
        TLExpr::FuzzyImplication {
            kind,
            premise,
            conclusion,
        } => {
            write!(buf, "FI_{kind:?}(")?;
            write_expr(premise, cfg, buf)?;
            write!(buf, ", ")?;
            write_expr(conclusion, cfg, buf)?;
            write!(buf, ")")
        }

        // ---- probabilistic ----
        TLExpr::SoftExists {
            var,
            domain,
            body,
            temperature,
        } => {
            write!(
                buf,
                "soft_{}{} {} {} [t={temperature}]. ",
                cfg.sym_exists(),
                var,
                cfg.sym_in(),
                domain
            )?;
            write_expr(body, cfg, buf)
        }
        TLExpr::SoftForAll {
            var,
            domain,
            body,
            temperature,
        } => {
            write!(
                buf,
                "soft_{}{} {} {} [t={temperature}]. ",
                cfg.sym_forall(),
                var,
                cfg.sym_in(),
                domain
            )?;
            write_expr(body, cfg, buf)
        }
        TLExpr::WeightedRule { weight, rule } => {
            write!(buf, "{weight}:: ")?;
            write_expr(rule, cfg, buf)
        }
        TLExpr::ProbabilisticChoice { alternatives } => {
            write!(buf, "choice(")?;
            for (i, (prob, expr)) in alternatives.iter().enumerate() {
                if i > 0 {
                    write!(buf, "; ")?;
                }
                write!(buf, "{prob}: ")?;
                write_expr(expr, cfg, buf)?;
            }
            write!(buf, ")")
        }

        // ---- higher-order ----
        TLExpr::Lambda {
            var,
            var_type,
            body,
        } => {
            write!(buf, "{}{var}", cfg.sym_lambda())?;
            if let Some(ty) = var_type {
                write!(buf, ":{ty}")?;
            }
            write!(buf, ". ")?;
            write_expr(body, cfg, buf)
        }
        TLExpr::Apply { function, argument } => {
            write_subexpr(function, cfg, buf)?;
            write!(buf, "(")?;
            write_expr(argument, cfg, buf)?;
            write!(buf, ")")
        }

        // ---- set operations ----
        TLExpr::SetMembership { element, set } => {
            write_subexpr(element, cfg, buf)?;
            write!(buf, " {} ", cfg.sym_element_of())?;
            write_subexpr(set, cfg, buf)
        }
        TLExpr::SetUnion { left, right } => write_binop(left, right, cfg, buf, cfg.sym_union()),
        TLExpr::SetIntersection { left, right } => {
            write_binop(left, right, cfg, buf, cfg.sym_intersection())
        }
        TLExpr::SetDifference { left, right } => {
            write_binop(left, right, cfg, buf, cfg.sym_setminus())
        }
        TLExpr::SetCardinality { set } => {
            write!(buf, "|")?;
            write_expr(set, cfg, buf)?;
            write!(buf, "|")
        }
        TLExpr::EmptySet => write!(buf, "{}", cfg.sym_emptyset()),
        TLExpr::SetComprehension {
            var,
            domain,
            condition,
        } => {
            write!(buf, "{{ {var} : {domain} | ")?;
            write_expr(condition, cfg, buf)?;
            write!(buf, " }}")
        }

        // ---- counting quantifiers ----
        TLExpr::CountingExists {
            var,
            domain,
            body,
            min_count,
        } => {
            let sym_e = cfg.sym_exists();
            let sym_in = cfg.sym_in();
            write!(buf, "{sym_e}>={min_count} {var} {sym_in} {domain}. ")?;
            write_expr(body, cfg, buf)
        }
        TLExpr::CountingForAll {
            var,
            domain,
            body,
            min_count,
        } => {
            let sym_a = cfg.sym_forall();
            let sym_in = cfg.sym_in();
            write!(buf, "{sym_a}>={min_count} {var} {sym_in} {domain}. ")?;
            write_expr(body, cfg, buf)
        }
        TLExpr::ExactCount {
            var,
            domain,
            body,
            count,
        } => {
            let sym_e = cfg.sym_exists();
            let sym_in = cfg.sym_in();
            write!(buf, "{sym_e}={count} {var} {sym_in} {domain}. ")?;
            write_expr(body, cfg, buf)
        }
        TLExpr::Majority { var, domain, body } => {
            let sym_in = cfg.sym_in();
            write!(buf, "majority {var} {sym_in} {domain}. ")?;
            write_expr(body, cfg, buf)
        }

        // ---- fixed points ----
        TLExpr::LeastFixpoint { var, body } => {
            write!(buf, "{}{var}. ", cfg.sym_mu())?;
            write_expr(body, cfg, buf)
        }
        TLExpr::GreatestFixpoint { var, body } => {
            write!(buf, "{}{var}. ", cfg.sym_nu())?;
            write_expr(body, cfg, buf)
        }

        // ---- hybrid logic ----
        TLExpr::Nominal { name } => write!(buf, "{}{name}", cfg.sym_at()),
        TLExpr::At { nominal, formula } => {
            write!(buf, "{}{nominal} ", cfg.sym_at())?;
            write_subexpr(formula, cfg, buf)
        }
        TLExpr::Somewhere { formula } => {
            write!(buf, "E ")?;
            write_subexpr(formula, cfg, buf)
        }
        TLExpr::Everywhere { formula } => {
            write!(buf, "A ")?;
            write_subexpr(formula, cfg, buf)
        }

        // ---- constraint programming ----
        TLExpr::AllDifferent { variables } => {
            write!(buf, "all_different({})", variables.join(", "))
        }
        TLExpr::GlobalCardinality {
            variables,
            values,
            min_occurrences,
            max_occurrences,
        } => {
            write!(buf, "gcc([{}], [", variables.join(", "))?;
            for (i, val) in values.iter().enumerate() {
                if i > 0 {
                    write!(buf, ", ")?;
                }
                write_expr(val, cfg, buf)?;
            }
            write!(buf, "], min={min_occurrences:?}, max={max_occurrences:?})")
        }

        // ---- abductive reasoning ----
        TLExpr::Abducible { name, cost } => write!(buf, "abd({name}, {cost})"),
        TLExpr::Explain { formula } => {
            write!(buf, "explain(")?;
            write_expr(formula, cfg, buf)?;
            write!(buf, ")")
        }

        // ---- symbol literal ----
        TLExpr::SymbolLiteral(s) => write!(buf, ":{s}"),

        // ---- pattern matching ----
        TLExpr::Match { scrutinee, arms } => {
            write!(buf, "match ")?;
            write_subexpr(scrutinee, cfg, buf)?;
            write!(buf, " {{")?;
            for (pat, body) in arms {
                writeln!(buf)?;
                for _ in 0..cfg.indent_width {
                    write!(buf, " ")?;
                }
                write!(buf, "{pat} => ")?;
                write_expr(body, cfg, buf)?;
                write!(buf, ",")?;
            }
            write!(buf, "\n}}")
        }
    }
}

/// Write a sub-expression, wrapping in parentheses if it is compound.
fn write_subexpr(expr: &TLExpr, cfg: &PrettyConfig, buf: &mut String) -> std::fmt::Result {
    if needs_parens(expr) {
        write!(buf, "(")?;
        write_expr(expr, cfg, buf)?;
        write!(buf, ")")
    } else {
        write_expr(expr, cfg, buf)
    }
}

/// Binary operator helper: `(left op right)`.
fn write_binop(
    left: &TLExpr,
    right: &TLExpr,
    cfg: &PrettyConfig,
    buf: &mut String,
    op: &str,
) -> std::fmt::Result {
    write!(buf, "(")?;
    write_expr(left, cfg, buf)?;
    write!(buf, " {op} ")?;
    write_expr(right, cfg, buf)?;
    write!(buf, ")")
}

/// Unary function helper: `name(inner)`.
fn write_func1(
    name: &str,
    inner: &TLExpr,
    cfg: &PrettyConfig,
    buf: &mut String,
) -> std::fmt::Result {
    write!(buf, "{name}(")?;
    write_expr(inner, cfg, buf)?;
    write!(buf, ")")
}

/// Binary function helper: `name(a, b)`.
fn write_func2(
    name: &str,
    a: &TLExpr,
    b: &TLExpr,
    cfg: &PrettyConfig,
    buf: &mut String,
) -> std::fmt::Result {
    write!(buf, "{name}(")?;
    write_expr(a, cfg, buf)?;
    write!(buf, ", ")?;
    write_expr(b, cfg, buf)?;
    write!(buf, ")")
}

/// Determine whether a sub-expression needs parenthesisation for clarity.
fn needs_parens(expr: &TLExpr) -> bool {
    matches!(
        expr,
        TLExpr::And(..)
            | TLExpr::Or(..)
            | TLExpr::Imply(..)
            | TLExpr::Add(..)
            | TLExpr::Sub(..)
            | TLExpr::Mul(..)
            | TLExpr::Div(..)
            | TLExpr::Pow(..)
            | TLExpr::Mod(..)
            | TLExpr::Eq(..)
            | TLExpr::Lt(..)
            | TLExpr::Gt(..)
            | TLExpr::Lte(..)
            | TLExpr::Gte(..)
            | TLExpr::Until { .. }
            | TLExpr::Release { .. }
            | TLExpr::WeakUntil { .. }
            | TLExpr::StrongRelease { .. }
            | TLExpr::SetUnion { .. }
            | TLExpr::SetIntersection { .. }
            | TLExpr::SetDifference { .. }
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TLExpr, Term};

    #[test]
    fn test_pretty_pred() {
        let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
        assert_eq!(pretty(&expr), "knows(x, y)");
    }

    #[test]
    fn test_pretty_and_unicode() {
        let a = TLExpr::pred("P", vec![Term::var("x")]);
        let b = TLExpr::pred("Q", vec![Term::var("y")]);
        let expr = TLExpr::and(a, b);
        let out = pretty(&expr);
        assert!(out.contains('\u{2227}'), "expected unicode AND: {out}");
    }

    #[test]
    fn test_pretty_and_ascii() {
        let a = TLExpr::pred("P", vec![Term::var("x")]);
        let b = TLExpr::pred("Q", vec![Term::var("y")]);
        let expr = TLExpr::and(a, b);
        let out = pretty_print(&expr, &PrettyConfig::ascii());
        assert!(out.contains('&'), "expected & in ascii mode: {out}");
    }

    #[test]
    fn test_pretty_or() {
        let a = TLExpr::pred("P", vec![]);
        let b = TLExpr::pred("Q", vec![]);
        let expr = TLExpr::or(a, b);
        let out = pretty(&expr);
        assert!(out.contains('\u{2228}'), "expected unicode OR: {out}");
    }

    #[test]
    fn test_pretty_not() {
        let a = TLExpr::pred("P", vec![Term::var("x")]);
        let expr = TLExpr::negate(a);
        let out = pretty(&expr);
        assert!(out.contains('\u{00ac}'), "expected unicode NOT: {out}");
    }

    #[test]
    fn test_pretty_exists() {
        let body = TLExpr::pred("P", vec![Term::var("x")]);
        let expr = TLExpr::exists("x", "Person", body);
        let out = pretty(&expr);
        assert!(out.contains("\u{2203}x"), "expected exists with var: {out}");
        assert!(out.contains("Person"), "expected domain: {out}");
    }

    #[test]
    fn test_pretty_forall() {
        let body = TLExpr::pred("P", vec![Term::var("x")]);
        let expr = TLExpr::forall("x", "D", body);
        let out = pretty(&expr);
        assert!(out.contains("\u{2200}x"), "expected forall with var: {out}");
    }

    #[test]
    fn test_pretty_implication() {
        let a = TLExpr::pred("P", vec![]);
        let b = TLExpr::pred("Q", vec![]);
        let expr = TLExpr::imply(a, b);
        let out = pretty(&expr);
        assert!(out.contains('\u{2192}'), "expected unicode arrow: {out}");
    }

    #[test]
    fn test_pretty_constant() {
        let expr = TLExpr::Constant(0.5);
        assert_eq!(pretty(&expr), "0.5");
    }

    #[test]
    fn test_pretty_constant_integer() {
        let expr = TLExpr::Constant(3.0);
        assert_eq!(pretty(&expr), "3");
    }

    #[test]
    fn test_pretty_add() {
        let a = TLExpr::Constant(1.0);
        let b = TLExpr::Constant(2.0);
        let expr = TLExpr::add(a, b);
        let out = pretty(&expr);
        assert!(out.contains('+'), "expected +: {out}");
    }

    #[test]
    fn test_pretty_nested() {
        let p = TLExpr::pred("P", vec![]);
        let q = TLExpr::pred("Q", vec![]);
        let r = TLExpr::pred("R", vec![]);
        let and_expr = TLExpr::and(p, q);
        let or_expr = TLExpr::or(and_expr, r);
        let out = pretty(&or_expr);
        // The inner AND should be parenthesised inside the OR
        assert!(out.contains('('), "expected parens in nested: {out}");
        assert!(out.contains('\u{2227}'), "expected AND symbol: {out}");
        assert!(out.contains('\u{2228}'), "expected OR symbol: {out}");
    }

    #[test]
    fn test_pretty_term_var() {
        let t = Term::var("hello");
        assert_eq!(pretty_term(&t), "hello");
    }

    #[test]
    fn test_pretty_term_const() {
        let t = Term::constant("42");
        assert_eq!(pretty_term(&t), "42");
    }

    #[test]
    fn test_config_default_is_unicode() {
        let cfg = PrettyConfig::default();
        assert_eq!(cfg.symbols, SymbolSet::Unicode);
    }

    #[test]
    fn test_config_ascii() {
        let cfg = PrettyConfig::ascii();
        assert_eq!(cfg.symbols, SymbolSet::Ascii);
    }

    #[test]
    fn test_config_indent() {
        let cfg = PrettyConfig::default().with_indent(4);
        assert_eq!(cfg.indent_width, 4);
    }

    #[test]
    fn test_pretty_deterministic() {
        let expr = TLExpr::and(
            TLExpr::pred("P", vec![Term::var("x")]),
            TLExpr::or(TLExpr::pred("Q", vec![]), TLExpr::Constant(1.0)),
        );
        let a = pretty(&expr);
        let b = pretty(&expr);
        assert_eq!(a, b, "pretty printing must be deterministic");
    }

    #[test]
    fn test_pretty_convenience() {
        let expr = TLExpr::pred("P", vec![Term::var("x")]);
        // `pretty` is just a shorthand for `pretty_print` with default config
        let a = pretty(&expr);
        let b = pretty_print(&expr, &PrettyConfig::default());
        assert_eq!(a, b);
    }
}

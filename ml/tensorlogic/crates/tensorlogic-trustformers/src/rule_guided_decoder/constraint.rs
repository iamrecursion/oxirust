//! Logical-constraint compilation for rule-guided decoding.
//!
//! This module wraps a [`tensorlogic_ir::TLExpr`] constraint and compiles it
//! into a lightweight, per-token lookup representation that the decoder can
//! consult at every sampling step.
//!
//! ## Scope
//!
//! The following subset of `TLExpr` is supported:
//!
//! * [`TLExpr::Pred`] — treated as an allow-list of symbol names that the
//!   candidate token must match.
//! * [`TLExpr::And`] — intersection of its operands' constraints.
//! * [`TLExpr::Or`] — union of its operands' constraints.
//! * [`TLExpr::Not`] — inverts the classification emitted by the inner
//!   constraint (allow ↔ forbid for known symbols; unknown tokens remain a
//!   soft penalty regardless of negation).
//! * [`TLExpr::Imply`] — compiled as `Not(premise) Or conclusion`.
//!
//! Any other variant (e.g. `Exists`, `ForAll`) collapses to
//! [`ConstraintVerdict::SoftPenalty`]`(0.0)` (no-op).
//! See [`extend_tlexpr_support`] for the next extension point.
//!
//! ## Token-to-symbol mapping
//!
//! The compiled constraint needs to know which *symbol name* each token
//! corresponds to — vocabulary encodings are deeply application-specific.
//! Callers supply a mapper `Fn(TokenId) -> Option<SymbolName>`; an empty option
//! means "this token is unknown / has no symbolic identity".

use std::collections::HashSet;

use tensorlogic_ir::TLExpr;

#[cfg(test)]
use crate::rule_guided_decoder::error::RuleGuidedError;
use crate::rule_guided_decoder::error::RuleGuidedResult;

/// Logical token identifier used by the beam-search backend (`usize`).
pub type TokenId = usize;

/// Verdict produced by a compiled constraint about a candidate token.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConstraintVerdict {
    /// Token is explicitly allowed — no logit adjustment needed.
    Allowed,
    /// Token is explicitly forbidden — hard masking should set the logit
    /// to `-inf`; soft masking treats it as the maximum-penalty case.
    Forbidden,
    /// Soft penalty expressed as a non-negative "violation magnitude" that
    /// the soft-penalty mask multiplies by `lambda`.
    SoftPenalty(f64),
}

/// Token → symbol-name mapper. `None` means the token has no symbolic identity
/// (constraint evaluation is conservative — see [`RuleConstraint::evaluate`]).
pub type TokenSymbolMapper = dyn Fn(TokenId) -> Option<String> + Send + Sync;

/// Compiled representation of a single `TLExpr` constraint.
///
/// Three forms coexist:
///
/// 1. **Eager allow-table** — a finite set of symbol names that must match
///    (`complement = false`).  Used when the constraint resolves to an
///    allow-list (e.g. `Pred`, `Or`, `And`).
/// 2. **Eager deny-table** — a finite set of symbol names that must *not*
///    match (`complement = true`).  Used when a `Not` wrapper is compiled
///    around an inner allow-list.
/// 3. **Fallback pass-through** — used when the expression hit an unsupported
///    variant.  In that case [`RuleConstraint::evaluate`] returns
///    `ConstraintVerdict::SoftPenalty(0.0)` and the decoder behaves as if no
///    constraint was present.
pub struct RuleConstraint {
    /// Original TLExpr (kept for diagnostics and lazy re-compilation).
    source: TLExpr,
    /// Symbol names for membership testing, if the expression is enumerable.
    ///
    /// `None` means "constraint is non-enumerable" (unsupported variant).
    /// Combined with `complement`, the semantics are:
    ///   - `complement = false`: token must map to a name *in* this set.
    ///   - `complement = true`: token must map to a name *outside* this set.
    allow_set: Option<HashSet<String>>,
    /// When `true`, the allow-set acts as a *deny*-set: symbols inside it
    /// are `Forbidden`, symbols outside it are `Allowed`.  In both cases
    /// tokens with no symbolic identity yield `SoftPenalty(1.0)`.
    complement: bool,
    /// Mapper from token ids to symbol names.  Stored so `evaluate` can be
    /// called many times without re-compiling.
    mapper: Box<TokenSymbolMapper>,
    /// Set to true when compilation succeeded against a recognised subset of
    /// `TLExpr`.  When false, the constraint silently no-ops.
    supported: bool,
}

impl std::fmt::Debug for RuleConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuleConstraint")
            .field("source", &self.source)
            .field("allow_set", &self.allow_set)
            .field("complement", &self.complement)
            .field("supported", &self.supported)
            .finish_non_exhaustive()
    }
}

impl RuleConstraint {
    /// Compile a constraint from a `TLExpr` and a token → symbol-name mapper.
    ///
    /// * If the expression only uses supported variants, the returned
    ///   constraint eagerly enumerates allowed symbol names into a `HashSet`.
    /// * Otherwise, the constraint is still constructed but evaluates to a
    ///   no-op (soft-penalty of zero).  This makes the decoder forward-
    ///   compatible: new `TLExpr` variants don't break existing call sites.
    pub fn compile<M>(expr: TLExpr, mapper: M) -> RuleGuidedResult<Self>
    where
        M: Fn(TokenId) -> Option<String> + Send + Sync + 'static,
    {
        let mut builder = AllowSetBuilder::default();
        let supported = builder.visit(&expr)?;
        let (allow_set, complement) = if supported {
            let (set, comp) = builder.finalize();
            (Some(set), comp)
        } else {
            (None, false)
        };
        Ok(Self {
            source: expr,
            allow_set,
            complement,
            mapper: Box::new(mapper),
            supported,
        })
    }

    /// Evaluate the constraint against `(prefix, candidate)`.
    ///
    /// `prefix` is the token sequence already committed to the beam; it is
    /// not used by the current allow-list compiler but is part of the contract
    /// so stateful constraints (e.g. "no token X after token Y") remain
    /// implementable without an API break — see [`extend_tlexpr_support`].
    pub fn evaluate(&self, prefix: &[TokenId], candidate: TokenId) -> ConstraintVerdict {
        let _ = prefix; // Reserved for future stateful predicates.
        if !self.supported {
            return ConstraintVerdict::SoftPenalty(0.0);
        }
        let allow_set = match &self.allow_set {
            Some(set) => set,
            None => return ConstraintVerdict::SoftPenalty(0.0),
        };

        let symbol = (self.mapper)(candidate);
        match symbol {
            Some(name) => {
                let in_set = allow_set.contains(&name);
                // complement=false: in_set → Allowed, !in_set → Forbidden
                // complement=true:  in_set → Forbidden, !in_set → Allowed
                if in_set ^ self.complement {
                    ConstraintVerdict::Allowed
                } else {
                    ConstraintVerdict::Forbidden
                }
            }
            None => {
                // Unknown tokens (e.g. punctuation with no symbol) are treated
                // conservatively as a soft violation so the decoder slightly
                // prefers fully-symbolic completions without banning them.
                // This behaviour is the same regardless of the complement flag:
                // we cannot classify an unknown token as definitively Allowed
                // even when the constraint is negated.
                ConstraintVerdict::SoftPenalty(1.0)
            }
        }
    }

    /// Read-only access to the compiled allow/deny-list, if any.
    pub fn allow_set(&self) -> Option<&HashSet<String>> {
        self.allow_set.as_ref()
    }

    /// `true` when the set is acting as a *deny*-list (negated constraint).
    pub fn is_complement(&self) -> bool {
        self.complement
    }

    /// `true` when the constraint was compiled against a supported subset of
    /// `TLExpr`.  `false` means the constraint is a no-op.
    pub fn is_supported(&self) -> bool {
        self.supported
    }

    /// Original expression.
    pub fn source(&self) -> &TLExpr {
        &self.source
    }
}

// ---------------------------------------------------------------------------
// Allow-set compiler
// ---------------------------------------------------------------------------

/// Internal result type returned by `AllowSetBuilder::classify`.
///
/// The `bool` is the complement flag: `false` means the set is an allow-list,
/// `true` means it is a deny-list (the `Not` wrapper flips it).
type ClassifyResult = Option<(HashSet<String>, bool)>;

#[derive(Default)]
struct AllowSetBuilder {
    /// Accumulated (set, complement) pair, set by `visit`.
    current: Option<(HashSet<String>, bool)>,
}

impl AllowSetBuilder {
    /// Recursively visit `expr`, updating `self.current`.
    ///
    /// Returns `true` when every sub-expression fell into the supported
    /// subset; `false` signals the caller to drop the compiled table and
    /// fall back to the no-op path.
    fn visit(&mut self, expr: &TLExpr) -> RuleGuidedResult<bool> {
        let pair = match self.classify(expr)? {
            Some(p) => p,
            None => return Ok(false),
        };
        self.current = Some(pair);
        Ok(true)
    }

    /// Returns `(set, complement)` after visiting.
    fn finalize(self) -> (HashSet<String>, bool) {
        self.current.unwrap_or_default()
    }

    /// Attempt to fold `expr` into `(allow_set, complement)`.
    ///
    /// Returns `Ok(None)` when the expression uses an unsupported variant.
    fn classify(&self, expr: &TLExpr) -> RuleGuidedResult<ClassifyResult> {
        match expr {
            TLExpr::Pred { name, args } => {
                // Treat the predicate's atoms as allowed symbol names.
                // The predicate name itself is allowed as a symbol too —
                // this matches the usual convention where tokenizers emit a
                // "type" token (e.g., `entity(Alice)`).
                let mut set = HashSet::with_capacity(1 + args.len());
                set.insert(name.clone());
                for arg in args {
                    match arg {
                        tensorlogic_ir::Term::Const(s) => {
                            set.insert(s.clone());
                        }
                        tensorlogic_ir::Term::Var(_) => {
                            // Variables are unbound: the predicate doesn't
                            // restrict the vocabulary symbolically.
                        }
                        tensorlogic_ir::Term::Typed { value, .. } => {
                            if let tensorlogic_ir::Term::Const(s) = value.as_ref() {
                                set.insert(s.clone());
                            }
                        }
                    }
                }
                Ok(Some((set, false)))
            }
            TLExpr::And(lhs, rhs) => {
                let (l, lc) = match self.classify(lhs)? {
                    Some(p) => p,
                    None => return Ok(None),
                };
                let (r, rc) = match self.classify(rhs)? {
                    Some(p) => p,
                    None => return Ok(None),
                };
                // AND of two normal allow-lists → intersection, still an allow-list.
                // We only handle the case where both sides have the same complement
                // flag (mixed complement semantics require a universe, which we
                // don't have, so fall back to unsupported).
                if lc == rc {
                    let combined: HashSet<String> = if lc {
                        // Both are deny-lists: A_deny AND B_deny → union of denied
                        // symbols (token must avoid both sets to pass either deny-check).
                        l.union(&r).cloned().collect()
                    } else {
                        l.intersection(&r).cloned().collect()
                    };
                    Ok(Some((combined, lc)))
                } else {
                    Ok(None)
                }
            }
            TLExpr::Or(lhs, rhs) => {
                let (l, lc) = match self.classify(lhs)? {
                    Some(p) => p,
                    None => return Ok(None),
                };
                let (r, rc) = match self.classify(rhs)? {
                    Some(p) => p,
                    None => return Ok(None),
                };
                if lc == rc {
                    let combined: HashSet<String> = if lc {
                        // Both deny-lists: OR → intersection (token passes if it
                        // avoids at least one deny-set, i.e. is outside both).
                        l.intersection(&r).cloned().collect()
                    } else {
                        l.union(&r).cloned().collect()
                    };
                    Ok(Some((combined, lc)))
                } else {
                    Ok(None)
                }
            }
            TLExpr::Not(inner) => {
                // Compile the inner expression, then flip its complement flag.
                // This gives correct double-negation elimination for free:
                // Not(Not(x)) → inner compiles with complement=false, flip →
                // complement=true, outer flip → complement=false again.
                match self.classify(inner)? {
                    Some((set, comp)) => Ok(Some((set, !comp))),
                    None => Ok(None),
                }
            }
            TLExpr::Imply(premise, conclusion) => {
                // p → q  ≡  ¬p ∨ q
                // Rewrite at compile-time and recurse.
                let not_p = TLExpr::Not(premise.clone());
                let rewritten = TLExpr::Or(Box::new(not_p), conclusion.clone());
                self.classify(&rewritten)
            }
            // Exists/ForAll/stateful connectives require enumerating a domain
            // or inspecting the prefix — not available at compile time.
            _ => Ok(None),
        }
    }
}

/// Documentation marker: extension point for additional `TLExpr` variants.
///
/// Today we handle `Pred`, `And`, `Or`, `Not`, and `Imply`.
/// To add support for `Exists`/`ForAll`, a domain-enumeration mechanism
/// is needed (the variable domain is not available at compile time).
/// Stateful connectives (those whose truth depends on the generated prefix)
/// should introduce a new arm in [`RuleConstraint::evaluate`] that inspects
/// `prefix` rather than the static allow-set.
pub const fn extend_tlexpr_support() {}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::Term;

    fn mk_pred(name: &str, consts: &[&str]) -> TLExpr {
        TLExpr::Pred {
            name: name.into(),
            args: consts.iter().map(|c| Term::Const((*c).into())).collect(),
        }
    }

    fn demo_mapper() -> impl Fn(TokenId) -> Option<String> + Send + Sync + 'static {
        |tid: TokenId| match tid {
            1 => Some("Alice".into()),
            2 => Some("Bob".into()),
            3 => Some("entity".into()),
            _ => None,
        }
    }

    #[test]
    fn predicate_allow_list_accepts_named_consts() {
        let expr = mk_pred("entity", &["Alice"]);
        let rc = RuleConstraint::compile(expr, demo_mapper()).expect("compile");
        assert!(rc.is_supported());
        assert_eq!(rc.evaluate(&[], 1), ConstraintVerdict::Allowed);
        assert_eq!(rc.evaluate(&[], 2), ConstraintVerdict::Forbidden);
        assert_eq!(rc.evaluate(&[], 3), ConstraintVerdict::Allowed);
    }

    #[test]
    fn conjunction_intersects_allow_sets() {
        // entity(Alice) AND entity(Bob) — only the shared "entity" symbol
        // remains, so Alice/Bob tokens become Forbidden.
        let a = mk_pred("entity", &["Alice"]);
        let b = mk_pred("entity", &["Bob"]);
        let expr = TLExpr::And(Box::new(a), Box::new(b));
        let rc = RuleConstraint::compile(expr, demo_mapper()).expect("compile");
        assert!(rc.is_supported());
        assert_eq!(rc.evaluate(&[], 1), ConstraintVerdict::Forbidden);
        assert_eq!(rc.evaluate(&[], 2), ConstraintVerdict::Forbidden);
        assert_eq!(rc.evaluate(&[], 3), ConstraintVerdict::Allowed);
    }

    #[test]
    fn disjunction_unions_allow_sets() {
        let a = mk_pred("entity", &["Alice"]);
        let b = mk_pred("entity", &["Bob"]);
        let expr = TLExpr::Or(Box::new(a), Box::new(b));
        let rc = RuleConstraint::compile(expr, demo_mapper()).expect("compile");
        assert!(rc.is_supported());
        assert_eq!(rc.evaluate(&[], 1), ConstraintVerdict::Allowed);
        assert_eq!(rc.evaluate(&[], 2), ConstraintVerdict::Allowed);
        assert_eq!(rc.evaluate(&[], 3), ConstraintVerdict::Allowed);
    }

    #[test]
    fn unsupported_variant_returns_soft_noop() {
        // Exists requires domain enumeration — not available at compile time,
        // so it must fall back to the unsupported/no-op path.
        let body = mk_pred("entity", &["Alice"]);
        let expr = TLExpr::Exists {
            var: "x".to_string(),
            domain: "Person".to_string(),
            body: Box::new(body),
        };
        let rc = RuleConstraint::compile(expr, demo_mapper()).expect("compile");
        assert!(!rc.is_supported());
        assert_eq!(rc.evaluate(&[], 1), ConstraintVerdict::SoftPenalty(0.0));
    }

    #[test]
    fn unknown_token_yields_soft_penalty() {
        let expr = mk_pred("entity", &["Alice"]);
        let rc = RuleConstraint::compile(expr, demo_mapper()).expect("compile");
        // Token id 99 has no mapping -> conservative SoftPenalty(1.0).
        assert_eq!(rc.evaluate(&[], 99), ConstraintVerdict::SoftPenalty(1.0));
    }

    #[test]
    fn empty_intersection_forbids_all_known_tokens() {
        // entity(Alice) AND user(Charlie) — disjoint allow sets except that
        // "entity" / "user" are the predicate names.  Tokens for Alice/Bob
        // all become Forbidden.
        let a = mk_pred("entity", &["Alice"]);
        let b = mk_pred("user", &["Charlie"]);
        let expr = TLExpr::And(Box::new(a), Box::new(b));
        let rc = RuleConstraint::compile(expr, demo_mapper()).expect("compile");
        assert!(rc.is_supported());
        // "entity" is in allow_set(a) but not allow_set(b); intersection is
        // empty, so every known symbol is forbidden.
        assert_eq!(rc.evaluate(&[], 1), ConstraintVerdict::Forbidden);
        assert_eq!(rc.evaluate(&[], 3), ConstraintVerdict::Forbidden);
    }

    #[test]
    fn error_type_has_useful_display() {
        // Sanity check that RuleGuidedError links correctly.
        let err: RuleGuidedError =
            RuleGuidedError::CompilationError("synthetic failure".to_string());
        assert!(err.to_string().contains("synthetic"));
    }

    #[test]
    fn not_pred_forbids_inner_allows_rest() {
        // Not(entity(Alice)) — Alice/entity are forbidden; Bob is allowed.
        let inner = mk_pred("entity", &["Alice"]);
        let expr = TLExpr::Not(Box::new(inner));
        let rc = RuleConstraint::compile(expr, demo_mapper()).expect("compile");
        assert!(rc.is_supported());
        assert!(rc.is_complement());
        // Token 1 = "Alice" → in deny-set → Forbidden
        assert_eq!(rc.evaluate(&[], 1), ConstraintVerdict::Forbidden);
        // Token 3 = "entity" → in deny-set → Forbidden
        assert_eq!(rc.evaluate(&[], 3), ConstraintVerdict::Forbidden);
        // Token 2 = "Bob" → not in deny-set → Allowed
        assert_eq!(rc.evaluate(&[], 2), ConstraintVerdict::Allowed);
        // Token 99 = unknown → SoftPenalty(1.0) regardless of complement
        assert_eq!(rc.evaluate(&[], 99), ConstraintVerdict::SoftPenalty(1.0));
    }

    #[test]
    fn double_negation_is_identity() {
        // Not(Not(entity(Alice))) should behave like entity(Alice).
        let inner = mk_pred("entity", &["Alice"]);
        let single = TLExpr::Not(Box::new(inner.clone()));
        let double = TLExpr::Not(Box::new(single));
        let rc = RuleConstraint::compile(double, demo_mapper()).expect("compile");
        assert!(rc.is_supported());
        // Should NOT be complement — double negation cancels out.
        assert!(!rc.is_complement());
        assert_eq!(rc.evaluate(&[], 1), ConstraintVerdict::Allowed); // Alice
        assert_eq!(rc.evaluate(&[], 2), ConstraintVerdict::Forbidden); // Bob
        assert_eq!(rc.evaluate(&[], 3), ConstraintVerdict::Allowed); // entity
    }

    #[test]
    fn imply_p_q_is_not_p_or_q() {
        // entity(Alice) → entity(Bob)  ≡  ¬entity(Alice) ∨ entity(Bob)
        // Allow-set of ¬entity(Alice) = {Alice, entity} deny, i.e. anything NOT {Alice, entity}
        // Allow-set of entity(Bob) = {Bob, entity}
        // The OR of a deny-list and an allow-list: mixed complement → unsupported (None).
        // So this particular implication falls back gracefully to no-op.
        let p = mk_pred("entity", &["Alice"]);
        let q = mk_pred("entity", &["Bob"]);
        let expr = TLExpr::Imply(Box::new(p), Box::new(q));
        let rc = RuleConstraint::compile(expr, demo_mapper()).expect("compile");
        // Mixed complement sides → unsupported → soft no-op
        assert!(!rc.is_supported());
        assert_eq!(rc.evaluate(&[], 1), ConstraintVerdict::SoftPenalty(0.0));
    }

    #[test]
    fn imply_p_q_same_complement_succeeds() {
        // Not(entity(Alice)) → Not(entity(Bob))
        // ≡ ¬(¬entity(Alice)) ∨ ¬entity(Bob)
        // ≡ entity(Alice) ∨ ¬entity(Bob)
        // Both sides after rewrite: entity(Alice) is a normal allow-list (comp=false),
        // Not(entity(Bob)) is a deny-list (comp=true) — still mixed, still None.
        // But Not(Not(A)) → Not(Not(entity(Bob))) → same issue. Let's test
        // the canonical case where both sides of Imply are plain Preds that
        // produce deny-lists after the Not wrapping at the top level.
        // Imply(Not(A), Not(B)) → Or(Not(Not(A)), Not(B)) → Or(A, Not(B)) → mixed → None.
        // For a pure deny-list implication:
        // Imply(Not(pred_a), Not(pred_b)) is OR(A_allow, Not(B_allow)):
        // left=allow(false), right=deny(true) → mixed → None.
        // The simplest all-same-complement case: wrap the whole Imply in Not.
        // Not(Imply(A,B)) ≡ Not(Or(Not(A), B)) ≡ Not(Or(deny(A_set), allow(B_set))):
        // again mixed. This shows the mixed-complement limitation is expected.
        // What does succeed: Imply where both compile to allow-lists after rewriting.
        // That only happens when the premise is already a Not(...), making
        // Not(Not(premise)) = allow-list, so both sides are allow-lists.
        let a = mk_pred("entity", &["Alice"]);
        let b = mk_pred("entity", &["Bob"]);
        // Imply(Not(entity(Alice)), entity(Bob))
        // = Or(Not(Not(entity(Alice))), entity(Bob))
        // = Or(entity(Alice), entity(Bob))
        // Both compile to allow-lists with comp=false → union = {Alice, entity, Bob}
        let not_a = TLExpr::Not(Box::new(a));
        let expr = TLExpr::Imply(Box::new(not_a), Box::new(b));
        let rc = RuleConstraint::compile(expr, demo_mapper()).expect("compile");
        assert!(rc.is_supported());
        assert!(!rc.is_complement());
        // All three symbols are allowed
        assert_eq!(rc.evaluate(&[], 1), ConstraintVerdict::Allowed); // Alice
        assert_eq!(rc.evaluate(&[], 2), ConstraintVerdict::Allowed); // Bob
        assert_eq!(rc.evaluate(&[], 3), ConstraintVerdict::Allowed); // entity
    }
}

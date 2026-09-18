//! Structural diff between two TLExpr trees.
//!
//! Identifies additions, removals, and modifications at each
//! node in the expression tree. Useful for debugging incremental
//! compilation and tracking rule evolution.

use serde::{Deserialize, Serialize};
use tensorlogic_ir::TLExpr;

/// Kind of change between two expression nodes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DiffKind {
    /// No change
    Unchanged,
    /// Node was added (present in new, absent in old)
    Added,
    /// Node was removed (present in old, absent in new)
    Removed,
    /// Node type changed (e.g., And -> Or)
    TypeChanged { old_type: String, new_type: String },
    /// Node parameters changed (e.g., different predicate name)
    ParameterChanged {
        old_value: String,
        new_value: String,
    },
    /// Children changed (recurse into sub-diffs)
    ChildrenChanged,
}

impl DiffKind {
    /// Returns `true` if this represents an actual change (not `Unchanged`).
    pub fn is_change(&self) -> bool {
        !matches!(self, DiffKind::Unchanged)
    }
}

/// A single diff entry at a specific path in the expression tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffEntry {
    /// Path from root (e.g., `["left", "body", "arg0"]`)
    pub path: Vec<String>,
    /// Kind of change
    pub kind: DiffKind,
    /// Human-readable description
    pub description: String,
}

/// Complete diff result between two expressions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExprDiff {
    /// All diff entries found during comparison.
    pub entries: Vec<DiffEntry>,
}

impl ExprDiff {
    /// Create an empty diff.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` if the two expressions are structurally identical.
    pub fn is_identical(&self) -> bool {
        self.entries.is_empty() || self.entries.iter().all(|e| !e.kind.is_change())
    }

    /// Count the number of actual changes (excluding `Unchanged` entries).
    pub fn change_count(&self) -> usize {
        self.entries.iter().filter(|e| e.kind.is_change()).count()
    }

    /// Filter to only addition entries.
    pub fn additions(&self) -> Vec<&DiffEntry> {
        self.entries
            .iter()
            .filter(|e| matches!(e.kind, DiffKind::Added))
            .collect()
    }

    /// Filter to only removal entries.
    pub fn removals(&self) -> Vec<&DiffEntry> {
        self.entries
            .iter()
            .filter(|e| matches!(e.kind, DiffKind::Removed))
            .collect()
    }

    /// Filter to only modification entries (TypeChanged or ParameterChanged).
    pub fn modifications(&self) -> Vec<&DiffEntry> {
        self.entries
            .iter()
            .filter(|e| {
                matches!(
                    e.kind,
                    DiffKind::TypeChanged { .. } | DiffKind::ParameterChanged { .. }
                )
            })
            .collect()
    }

    /// Human-readable summary of the diff.
    pub fn summary(&self) -> String {
        format!(
            "{} changes ({} added, {} removed, {} modified)",
            self.change_count(),
            self.additions().len(),
            self.removals().len(),
            self.modifications().len()
        )
    }
}

/// Compute the structural diff between two expressions.
pub fn expr_diff(old: &TLExpr, new: &TLExpr) -> ExprDiff {
    let mut diff = ExprDiff::new();
    compare_recursive(old, new, &[], &mut diff);
    diff
}

/// Get a short type tag for an expression variant.
pub fn expr_type_tag(expr: &TLExpr) -> String {
    match expr {
        TLExpr::Pred { .. } => "Pred".to_string(),
        TLExpr::And(..) => "And".to_string(),
        TLExpr::Or(..) => "Or".to_string(),
        TLExpr::Not(..) => "Not".to_string(),
        TLExpr::Exists { .. } => "Exists".to_string(),
        TLExpr::ForAll { .. } => "ForAll".to_string(),
        TLExpr::Imply(..) => "Imply".to_string(),
        TLExpr::Score(..) => "Score".to_string(),
        TLExpr::Add(..) => "Add".to_string(),
        TLExpr::Sub(..) => "Sub".to_string(),
        TLExpr::Mul(..) => "Mul".to_string(),
        TLExpr::Div(..) => "Div".to_string(),
        TLExpr::Pow(..) => "Pow".to_string(),
        TLExpr::Mod(..) => "Mod".to_string(),
        TLExpr::Min(..) => "Min".to_string(),
        TLExpr::Max(..) => "Max".to_string(),
        TLExpr::Abs(..) => "Abs".to_string(),
        TLExpr::Floor(..) => "Floor".to_string(),
        TLExpr::Ceil(..) => "Ceil".to_string(),
        TLExpr::Round(..) => "Round".to_string(),
        TLExpr::Sqrt(..) => "Sqrt".to_string(),
        TLExpr::Exp(..) => "Exp".to_string(),
        TLExpr::Log(..) => "Log".to_string(),
        TLExpr::Sin(..) => "Sin".to_string(),
        TLExpr::Cos(..) => "Cos".to_string(),
        TLExpr::Tan(..) => "Tan".to_string(),
        TLExpr::Eq(..) => "Eq".to_string(),
        TLExpr::Lt(..) => "Lt".to_string(),
        TLExpr::Gt(..) => "Gt".to_string(),
        TLExpr::Lte(..) => "Lte".to_string(),
        TLExpr::Gte(..) => "Gte".to_string(),
        TLExpr::IfThenElse { .. } => "IfThenElse".to_string(),
        TLExpr::Constant(..) => "Constant".to_string(),
        TLExpr::Aggregate { .. } => "Aggregate".to_string(),
        TLExpr::Let { .. } => "Let".to_string(),
        TLExpr::Box(..) => "Box".to_string(),
        TLExpr::Diamond(..) => "Diamond".to_string(),
        TLExpr::Next(..) => "Next".to_string(),
        TLExpr::Eventually(..) => "Eventually".to_string(),
        TLExpr::Always(..) => "Always".to_string(),
        TLExpr::Until { .. } => "Until".to_string(),
        TLExpr::TNorm { .. } => "TNorm".to_string(),
        TLExpr::TCoNorm { .. } => "TCoNorm".to_string(),
        TLExpr::FuzzyNot { .. } => "FuzzyNot".to_string(),
        TLExpr::FuzzyImplication { .. } => "FuzzyImplication".to_string(),
        TLExpr::SoftExists { .. } => "SoftExists".to_string(),
        TLExpr::SoftForAll { .. } => "SoftForAll".to_string(),
        TLExpr::WeightedRule { .. } => "WeightedRule".to_string(),
        TLExpr::ProbabilisticChoice { .. } => "ProbabilisticChoice".to_string(),
        TLExpr::Release { .. } => "Release".to_string(),
        TLExpr::WeakUntil { .. } => "WeakUntil".to_string(),
        TLExpr::StrongRelease { .. } => "StrongRelease".to_string(),
        TLExpr::Lambda { .. } => "Lambda".to_string(),
        TLExpr::Apply { .. } => "Apply".to_string(),
        TLExpr::SetMembership { .. } => "SetMembership".to_string(),
        TLExpr::SetUnion { .. } => "SetUnion".to_string(),
        TLExpr::SetIntersection { .. } => "SetIntersection".to_string(),
        TLExpr::SetDifference { .. } => "SetDifference".to_string(),
        TLExpr::SetCardinality { .. } => "SetCardinality".to_string(),
        TLExpr::EmptySet => "EmptySet".to_string(),
        TLExpr::SetComprehension { .. } => "SetComprehension".to_string(),
        TLExpr::CountingExists { .. } => "CountingExists".to_string(),
        TLExpr::CountingForAll { .. } => "CountingForAll".to_string(),
        TLExpr::ExactCount { .. } => "ExactCount".to_string(),
        TLExpr::Majority { .. } => "Majority".to_string(),
        TLExpr::LeastFixpoint { .. } => "LeastFixpoint".to_string(),
        TLExpr::GreatestFixpoint { .. } => "GreatestFixpoint".to_string(),
        TLExpr::Nominal { .. } => "Nominal".to_string(),
        TLExpr::At { .. } => "At".to_string(),
        TLExpr::Somewhere { .. } => "Somewhere".to_string(),
        TLExpr::Everywhere { .. } => "Everywhere".to_string(),
        TLExpr::AllDifferent { .. } => "AllDifferent".to_string(),
        TLExpr::GlobalCardinality { .. } => "GlobalCardinality".to_string(),
        TLExpr::Abducible { .. } => "Abducible".to_string(),
        TLExpr::Explain { .. } => "Explain".to_string(),
        TLExpr::SymbolLiteral(_) => "SymbolLiteral".to_string(),
        TLExpr::Match { .. } => "Match".to_string(),
    }
}

/// Compare two children at a named path position.
fn compare_child(
    old: &TLExpr,
    new: &TLExpr,
    parent_path: &[String],
    child_name: &str,
    diff: &mut ExprDiff,
) {
    let mut path = parent_path.to_vec();
    path.push(child_name.to_string());
    compare_recursive(old, new, &path, diff);
}

/// Record an addition entry at the given path.
fn record_added(path: &[String], child_name: &str, desc: &str, diff: &mut ExprDiff) {
    let mut p = path.to_vec();
    p.push(child_name.to_string());
    diff.entries.push(DiffEntry {
        path: p,
        kind: DiffKind::Added,
        description: desc.to_string(),
    });
}

/// Record a removal entry at the given path.
fn record_removed(path: &[String], child_name: &str, desc: &str, diff: &mut ExprDiff) {
    let mut p = path.to_vec();
    p.push(child_name.to_string());
    diff.entries.push(DiffEntry {
        path: p,
        kind: DiffKind::Removed,
        description: desc.to_string(),
    });
}

/// Compare arguments lists, reporting per-element changes.
fn compare_args(
    old_args: &[tensorlogic_ir::Term],
    new_args: &[tensorlogic_ir::Term],
    path: &[String],
    diff: &mut ExprDiff,
) {
    let common_len = old_args.len().min(new_args.len());
    for i in 0..common_len {
        if old_args[i] != new_args[i] {
            let mut p = path.to_vec();
            p.push(format!("arg{}", i));
            diff.entries.push(DiffEntry {
                path: p,
                kind: DiffKind::ParameterChanged {
                    old_value: format!("{:?}", old_args[i]),
                    new_value: format!("{:?}", new_args[i]),
                },
                description: format!("Arg {} changed", i),
            });
        }
    }
    for i in common_len..new_args.len() {
        record_added(
            path,
            &format!("arg{}", i),
            &format!("Arg {} added", i),
            diff,
        );
    }
    for i in common_len..old_args.len() {
        record_removed(
            path,
            &format!("arg{}", i),
            &format!("Arg {} removed", i),
            diff,
        );
    }
}

/// Compare string parameters at a given field name.
fn compare_string_param(
    old_val: &str,
    new_val: &str,
    path: &[String],
    field: &str,
    label: &str,
    diff: &mut ExprDiff,
) {
    if old_val != new_val {
        let mut p = path.to_vec();
        p.push(field.to_string());
        diff.entries.push(DiffEntry {
            path: p,
            kind: DiffKind::ParameterChanged {
                old_value: old_val.to_string(),
                new_value: new_val.to_string(),
            },
            description: format!("{}: {} -> {}", label, old_val, new_val),
        });
    }
}

/// Compare f64 parameters at a given field name.
fn compare_f64_param(
    old_val: f64,
    new_val: f64,
    path: &[String],
    field: &str,
    label: &str,
    diff: &mut ExprDiff,
) {
    if (old_val - new_val).abs() > f64::EPSILON {
        let mut p = path.to_vec();
        p.push(field.to_string());
        diff.entries.push(DiffEntry {
            path: p,
            kind: DiffKind::ParameterChanged {
                old_value: format!("{}", old_val),
                new_value: format!("{}", new_val),
            },
            description: format!("{}: {} -> {}", label, old_val, new_val),
        });
    }
}

/// Compare usize parameters at a given field name.
fn compare_usize_param(
    old_val: usize,
    new_val: usize,
    path: &[String],
    field: &str,
    label: &str,
    diff: &mut ExprDiff,
) {
    if old_val != new_val {
        let mut p = path.to_vec();
        p.push(field.to_string());
        diff.entries.push(DiffEntry {
            path: p,
            kind: DiffKind::ParameterChanged {
                old_value: format!("{}", old_val),
                new_value: format!("{}", new_val),
            },
            description: format!("{}: {} -> {}", label, old_val, new_val),
        });
    }
}

/// Recursively compare two expression trees.
fn compare_recursive(old: &TLExpr, new: &TLExpr, path: &[String], diff: &mut ExprDiff) {
    let old_tag = expr_type_tag(old);
    let new_tag = expr_type_tag(new);

    if old_tag != new_tag {
        diff.entries.push(DiffEntry {
            path: path.to_vec(),
            kind: DiffKind::TypeChanged {
                old_type: old_tag.clone(),
                new_type: new_tag.clone(),
            },
            description: format!("Changed from {} to {}", old_tag, new_tag),
        });
        return;
    }

    match (old, new) {
        // Pred: compare name and args
        (TLExpr::Pred { name: n1, args: a1 }, TLExpr::Pred { name: n2, args: a2 }) => {
            compare_string_param(n1, n2, path, "name", "Predicate name", diff);
            compare_args(a1, a2, path, diff);
        }

        // Binary logical/arithmetic ops
        (TLExpr::And(l1, r1), TLExpr::And(l2, r2))
        | (TLExpr::Or(l1, r1), TLExpr::Or(l2, r2))
        | (TLExpr::Imply(l1, r1), TLExpr::Imply(l2, r2))
        | (TLExpr::Add(l1, r1), TLExpr::Add(l2, r2))
        | (TLExpr::Sub(l1, r1), TLExpr::Sub(l2, r2))
        | (TLExpr::Mul(l1, r1), TLExpr::Mul(l2, r2))
        | (TLExpr::Div(l1, r1), TLExpr::Div(l2, r2))
        | (TLExpr::Pow(l1, r1), TLExpr::Pow(l2, r2))
        | (TLExpr::Mod(l1, r1), TLExpr::Mod(l2, r2))
        | (TLExpr::Min(l1, r1), TLExpr::Min(l2, r2))
        | (TLExpr::Max(l1, r1), TLExpr::Max(l2, r2))
        | (TLExpr::Eq(l1, r1), TLExpr::Eq(l2, r2))
        | (TLExpr::Lt(l1, r1), TLExpr::Lt(l2, r2))
        | (TLExpr::Gt(l1, r1), TLExpr::Gt(l2, r2))
        | (TLExpr::Lte(l1, r1), TLExpr::Lte(l2, r2))
        | (TLExpr::Gte(l1, r1), TLExpr::Gte(l2, r2)) => {
            compare_child(l1, l2, path, "left", diff);
            compare_child(r1, r2, path, "right", diff);
        }

        // Unary ops
        (TLExpr::Not(c1), TLExpr::Not(c2))
        | (TLExpr::Score(c1), TLExpr::Score(c2))
        | (TLExpr::Abs(c1), TLExpr::Abs(c2))
        | (TLExpr::Floor(c1), TLExpr::Floor(c2))
        | (TLExpr::Ceil(c1), TLExpr::Ceil(c2))
        | (TLExpr::Round(c1), TLExpr::Round(c2))
        | (TLExpr::Sqrt(c1), TLExpr::Sqrt(c2))
        | (TLExpr::Exp(c1), TLExpr::Exp(c2))
        | (TLExpr::Log(c1), TLExpr::Log(c2))
        | (TLExpr::Sin(c1), TLExpr::Sin(c2))
        | (TLExpr::Cos(c1), TLExpr::Cos(c2))
        | (TLExpr::Tan(c1), TLExpr::Tan(c2))
        | (TLExpr::Box(c1), TLExpr::Box(c2))
        | (TLExpr::Diamond(c1), TLExpr::Diamond(c2))
        | (TLExpr::Next(c1), TLExpr::Next(c2))
        | (TLExpr::Eventually(c1), TLExpr::Eventually(c2))
        | (TLExpr::Always(c1), TLExpr::Always(c2)) => {
            compare_child(c1, c2, path, "child", diff);
        }

        // Quantifiers: Exists / ForAll
        (
            TLExpr::Exists {
                var: v1,
                domain: d1,
                body: b1,
            },
            TLExpr::Exists {
                var: v2,
                domain: d2,
                body: b2,
            },
        )
        | (
            TLExpr::ForAll {
                var: v1,
                domain: d1,
                body: b1,
            },
            TLExpr::ForAll {
                var: v2,
                domain: d2,
                body: b2,
            },
        ) => {
            compare_string_param(v1, v2, path, "var", "Variable", diff);
            compare_string_param(d1, d2, path, "domain", "Domain", diff);
            compare_child(b1, b2, path, "body", diff);
        }

        // Constant
        (TLExpr::Constant(v1), TLExpr::Constant(v2)) => {
            compare_f64_param(*v1, *v2, path, "value", "Constant", diff);
        }

        // IfThenElse
        (
            TLExpr::IfThenElse {
                condition: c1,
                then_branch: t1,
                else_branch: e1,
            },
            TLExpr::IfThenElse {
                condition: c2,
                then_branch: t2,
                else_branch: e2,
            },
        ) => {
            compare_child(c1, c2, path, "condition", diff);
            compare_child(t1, t2, path, "then_branch", diff);
            compare_child(e1, e2, path, "else_branch", diff);
        }

        // Aggregate
        (
            TLExpr::Aggregate {
                op: op1,
                var: v1,
                domain: d1,
                body: b1,
                group_by: g1,
            },
            TLExpr::Aggregate {
                op: op2,
                var: v2,
                domain: d2,
                body: b2,
                group_by: g2,
            },
        ) => {
            if op1 != op2 {
                let mut p = path.to_vec();
                p.push("op".to_string());
                diff.entries.push(DiffEntry {
                    path: p,
                    kind: DiffKind::ParameterChanged {
                        old_value: format!("{:?}", op1),
                        new_value: format!("{:?}", op2),
                    },
                    description: format!("Aggregate op: {:?} -> {:?}", op1, op2),
                });
            }
            compare_string_param(v1, v2, path, "var", "Variable", diff);
            compare_string_param(d1, d2, path, "domain", "Domain", diff);
            compare_child(b1, b2, path, "body", diff);
            if g1 != g2 {
                let mut p = path.to_vec();
                p.push("group_by".to_string());
                diff.entries.push(DiffEntry {
                    path: p,
                    kind: DiffKind::ParameterChanged {
                        old_value: format!("{:?}", g1),
                        new_value: format!("{:?}", g2),
                    },
                    description: "Group-by changed".to_string(),
                });
            }
        }

        // Let binding
        (
            TLExpr::Let {
                var: v1,
                value: val1,
                body: b1,
            },
            TLExpr::Let {
                var: v2,
                value: val2,
                body: b2,
            },
        ) => {
            compare_string_param(v1, v2, path, "var", "Variable", diff);
            compare_child(val1, val2, path, "value", diff);
            compare_child(b1, b2, path, "body", diff);
        }

        // Until / WeakUntil
        (
            TLExpr::Until {
                before: b1,
                after: a1,
            },
            TLExpr::Until {
                before: b2,
                after: a2,
            },
        )
        | (
            TLExpr::WeakUntil {
                before: b1,
                after: a1,
            },
            TLExpr::WeakUntil {
                before: b2,
                after: a2,
            },
        ) => {
            compare_child(b1, b2, path, "before", diff);
            compare_child(a1, a2, path, "after", diff);
        }

        // Release / StrongRelease
        (
            TLExpr::Release {
                released: r1,
                releaser: l1,
            },
            TLExpr::Release {
                released: r2,
                releaser: l2,
            },
        )
        | (
            TLExpr::StrongRelease {
                released: r1,
                releaser: l1,
            },
            TLExpr::StrongRelease {
                released: r2,
                releaser: l2,
            },
        ) => {
            compare_child(r1, r2, path, "released", diff);
            compare_child(l1, l2, path, "releaser", diff);
        }

        // TNorm
        (
            TLExpr::TNorm {
                kind: k1,
                left: l1,
                right: r1,
            },
            TLExpr::TNorm {
                kind: k2,
                left: l2,
                right: r2,
            },
        ) => {
            if k1 != k2 {
                let mut p = path.to_vec();
                p.push("kind".to_string());
                diff.entries.push(DiffEntry {
                    path: p,
                    kind: DiffKind::ParameterChanged {
                        old_value: format!("{:?}", k1),
                        new_value: format!("{:?}", k2),
                    },
                    description: format!("TNorm kind: {:?} -> {:?}", k1, k2),
                });
            }
            compare_child(l1, l2, path, "left", diff);
            compare_child(r1, r2, path, "right", diff);
        }

        // TCoNorm
        (
            TLExpr::TCoNorm {
                kind: k1,
                left: l1,
                right: r1,
            },
            TLExpr::TCoNorm {
                kind: k2,
                left: l2,
                right: r2,
            },
        ) => {
            if k1 != k2 {
                let mut p = path.to_vec();
                p.push("kind".to_string());
                diff.entries.push(DiffEntry {
                    path: p,
                    kind: DiffKind::ParameterChanged {
                        old_value: format!("{:?}", k1),
                        new_value: format!("{:?}", k2),
                    },
                    description: format!("TCoNorm kind: {:?} -> {:?}", k1, k2),
                });
            }
            compare_child(l1, l2, path, "left", diff);
            compare_child(r1, r2, path, "right", diff);
        }

        // FuzzyNot
        (TLExpr::FuzzyNot { kind: k1, expr: e1 }, TLExpr::FuzzyNot { kind: k2, expr: e2 }) => {
            if k1 != k2 {
                let mut p = path.to_vec();
                p.push("kind".to_string());
                diff.entries.push(DiffEntry {
                    path: p,
                    kind: DiffKind::ParameterChanged {
                        old_value: format!("{:?}", k1),
                        new_value: format!("{:?}", k2),
                    },
                    description: format!("FuzzyNot kind: {:?} -> {:?}", k1, k2),
                });
            }
            compare_child(e1, e2, path, "expr", diff);
        }

        // FuzzyImplication
        (
            TLExpr::FuzzyImplication {
                kind: k1,
                premise: p1,
                conclusion: c1,
            },
            TLExpr::FuzzyImplication {
                kind: k2,
                premise: p2,
                conclusion: c2,
            },
        ) => {
            if k1 != k2 {
                let mut p = path.to_vec();
                p.push("kind".to_string());
                diff.entries.push(DiffEntry {
                    path: p,
                    kind: DiffKind::ParameterChanged {
                        old_value: format!("{:?}", k1),
                        new_value: format!("{:?}", k2),
                    },
                    description: format!("FuzzyImplication kind: {:?} -> {:?}", k1, k2),
                });
            }
            compare_child(p1, p2, path, "premise", diff);
            compare_child(c1, c2, path, "conclusion", diff);
        }

        // SoftExists
        (
            TLExpr::SoftExists {
                var: v1,
                domain: d1,
                body: b1,
                temperature: t1,
            },
            TLExpr::SoftExists {
                var: v2,
                domain: d2,
                body: b2,
                temperature: t2,
            },
        ) => {
            compare_string_param(v1, v2, path, "var", "Variable", diff);
            compare_string_param(d1, d2, path, "domain", "Domain", diff);
            compare_child(b1, b2, path, "body", diff);
            compare_f64_param(*t1, *t2, path, "temperature", "Temperature", diff);
        }

        // SoftForAll
        (
            TLExpr::SoftForAll {
                var: v1,
                domain: d1,
                body: b1,
                temperature: t1,
            },
            TLExpr::SoftForAll {
                var: v2,
                domain: d2,
                body: b2,
                temperature: t2,
            },
        ) => {
            compare_string_param(v1, v2, path, "var", "Variable", diff);
            compare_string_param(d1, d2, path, "domain", "Domain", diff);
            compare_child(b1, b2, path, "body", diff);
            compare_f64_param(*t1, *t2, path, "temperature", "Temperature", diff);
        }

        // WeightedRule
        (
            TLExpr::WeightedRule {
                weight: w1,
                rule: r1,
            },
            TLExpr::WeightedRule {
                weight: w2,
                rule: r2,
            },
        ) => {
            compare_f64_param(*w1, *w2, path, "weight", "Weight", diff);
            compare_child(r1, r2, path, "rule", diff);
        }

        // ProbabilisticChoice
        (
            TLExpr::ProbabilisticChoice { alternatives: a1 },
            TLExpr::ProbabilisticChoice { alternatives: a2 },
        ) => {
            let common_len = a1.len().min(a2.len());
            for i in 0..common_len {
                compare_f64_param(
                    a1[i].0,
                    a2[i].0,
                    path,
                    &format!("alt{}_prob", i),
                    &format!("Alternative {} probability", i),
                    diff,
                );
                compare_child(&a1[i].1, &a2[i].1, path, &format!("alt{}_expr", i), diff);
            }
            for i in common_len..a2.len() {
                record_added(
                    path,
                    &format!("alt{}", i),
                    &format!("Alternative {} added", i),
                    diff,
                );
            }
            for i in common_len..a1.len() {
                record_removed(
                    path,
                    &format!("alt{}", i),
                    &format!("Alternative {} removed", i),
                    diff,
                );
            }
        }

        // Lambda
        (
            TLExpr::Lambda {
                var: v1,
                var_type: t1,
                body: b1,
            },
            TLExpr::Lambda {
                var: v2,
                var_type: t2,
                body: b2,
            },
        ) => {
            compare_string_param(v1, v2, path, "var", "Variable", diff);
            if t1 != t2 {
                let mut p = path.to_vec();
                p.push("var_type".to_string());
                diff.entries.push(DiffEntry {
                    path: p,
                    kind: DiffKind::ParameterChanged {
                        old_value: format!("{:?}", t1),
                        new_value: format!("{:?}", t2),
                    },
                    description: format!("Type annotation: {:?} -> {:?}", t1, t2),
                });
            }
            compare_child(b1, b2, path, "body", diff);
        }

        // Apply
        (
            TLExpr::Apply {
                function: f1,
                argument: a1,
            },
            TLExpr::Apply {
                function: f2,
                argument: a2,
            },
        ) => {
            compare_child(f1, f2, path, "function", diff);
            compare_child(a1, a2, path, "argument", diff);
        }

        // Set ops with left/right
        (
            TLExpr::SetMembership {
                element: e1,
                set: s1,
            },
            TLExpr::SetMembership {
                element: e2,
                set: s2,
            },
        ) => {
            compare_child(e1, e2, path, "element", diff);
            compare_child(s1, s2, path, "set", diff);
        }

        (
            TLExpr::SetUnion {
                left: l1,
                right: r1,
            },
            TLExpr::SetUnion {
                left: l2,
                right: r2,
            },
        )
        | (
            TLExpr::SetIntersection {
                left: l1,
                right: r1,
            },
            TLExpr::SetIntersection {
                left: l2,
                right: r2,
            },
        )
        | (
            TLExpr::SetDifference {
                left: l1,
                right: r1,
            },
            TLExpr::SetDifference {
                left: l2,
                right: r2,
            },
        ) => {
            compare_child(l1, l2, path, "left", diff);
            compare_child(r1, r2, path, "right", diff);
        }

        // SetCardinality
        (TLExpr::SetCardinality { set: s1 }, TLExpr::SetCardinality { set: s2 }) => {
            compare_child(s1, s2, path, "set", diff);
        }

        // EmptySet
        (TLExpr::EmptySet, TLExpr::EmptySet) => {
            // identical
        }

        // SetComprehension
        (
            TLExpr::SetComprehension {
                var: v1,
                domain: d1,
                condition: c1,
            },
            TLExpr::SetComprehension {
                var: v2,
                domain: d2,
                condition: c2,
            },
        ) => {
            compare_string_param(v1, v2, path, "var", "Variable", diff);
            compare_string_param(d1, d2, path, "domain", "Domain", diff);
            compare_child(c1, c2, path, "condition", diff);
        }

        // Counting quantifiers
        (
            TLExpr::CountingExists {
                var: v1,
                domain: d1,
                body: b1,
                min_count: mc1,
            },
            TLExpr::CountingExists {
                var: v2,
                domain: d2,
                body: b2,
                min_count: mc2,
            },
        )
        | (
            TLExpr::CountingForAll {
                var: v1,
                domain: d1,
                body: b1,
                min_count: mc1,
            },
            TLExpr::CountingForAll {
                var: v2,
                domain: d2,
                body: b2,
                min_count: mc2,
            },
        ) => {
            compare_string_param(v1, v2, path, "var", "Variable", diff);
            compare_string_param(d1, d2, path, "domain", "Domain", diff);
            compare_child(b1, b2, path, "body", diff);
            compare_usize_param(*mc1, *mc2, path, "min_count", "Min count", diff);
        }

        // ExactCount
        (
            TLExpr::ExactCount {
                var: v1,
                domain: d1,
                body: b1,
                count: c1,
            },
            TLExpr::ExactCount {
                var: v2,
                domain: d2,
                body: b2,
                count: c2,
            },
        ) => {
            compare_string_param(v1, v2, path, "var", "Variable", diff);
            compare_string_param(d1, d2, path, "domain", "Domain", diff);
            compare_child(b1, b2, path, "body", diff);
            compare_usize_param(*c1, *c2, path, "count", "Count", diff);
        }

        // Majority
        (
            TLExpr::Majority {
                var: v1,
                domain: d1,
                body: b1,
            },
            TLExpr::Majority {
                var: v2,
                domain: d2,
                body: b2,
            },
        ) => {
            compare_string_param(v1, v2, path, "var", "Variable", diff);
            compare_string_param(d1, d2, path, "domain", "Domain", diff);
            compare_child(b1, b2, path, "body", diff);
        }

        // Fixed-point operators
        (
            TLExpr::LeastFixpoint { var: v1, body: b1 },
            TLExpr::LeastFixpoint { var: v2, body: b2 },
        )
        | (
            TLExpr::GreatestFixpoint { var: v1, body: b1 },
            TLExpr::GreatestFixpoint { var: v2, body: b2 },
        ) => {
            compare_string_param(v1, v2, path, "var", "Variable", diff);
            compare_child(b1, b2, path, "body", diff);
        }

        // Nominal
        (TLExpr::Nominal { name: n1 }, TLExpr::Nominal { name: n2 }) => {
            compare_string_param(n1, n2, path, "name", "Nominal name", diff);
        }

        // At
        (
            TLExpr::At {
                nominal: n1,
                formula: f1,
            },
            TLExpr::At {
                nominal: n2,
                formula: f2,
            },
        ) => {
            compare_string_param(n1, n2, path, "nominal", "Nominal", diff);
            compare_child(f1, f2, path, "formula", diff);
        }

        // Somewhere / Everywhere
        (TLExpr::Somewhere { formula: f1 }, TLExpr::Somewhere { formula: f2 })
        | (TLExpr::Everywhere { formula: f1 }, TLExpr::Everywhere { formula: f2 }) => {
            compare_child(f1, f2, path, "formula", diff);
        }

        // Explain
        (TLExpr::Explain { formula: f1 }, TLExpr::Explain { formula: f2 }) => {
            compare_child(f1, f2, path, "formula", diff);
        }

        // AllDifferent
        (TLExpr::AllDifferent { variables: vars1 }, TLExpr::AllDifferent { variables: vars2 }) => {
            if vars1 != vars2 {
                let mut p = path.to_vec();
                p.push("variables".to_string());
                diff.entries.push(DiffEntry {
                    path: p,
                    kind: DiffKind::ParameterChanged {
                        old_value: format!("{:?}", vars1),
                        new_value: format!("{:?}", vars2),
                    },
                    description: "Variables list changed".to_string(),
                });
            }
        }

        // GlobalCardinality
        (
            TLExpr::GlobalCardinality {
                variables: vars1,
                values: vals1,
                min_occurrences: min1,
                max_occurrences: max1,
            },
            TLExpr::GlobalCardinality {
                variables: vars2,
                values: vals2,
                min_occurrences: min2,
                max_occurrences: max2,
            },
        ) => {
            if vars1 != vars2 {
                let mut p = path.to_vec();
                p.push("variables".to_string());
                diff.entries.push(DiffEntry {
                    path: p,
                    kind: DiffKind::ParameterChanged {
                        old_value: format!("{:?}", vars1),
                        new_value: format!("{:?}", vars2),
                    },
                    description: "Variables list changed".to_string(),
                });
            }
            if vals1 != vals2 {
                let mut p = path.to_vec();
                p.push("values".to_string());
                diff.entries.push(DiffEntry {
                    path: p,
                    kind: DiffKind::ParameterChanged {
                        old_value: format!("{:?}", vals1),
                        new_value: format!("{:?}", vals2),
                    },
                    description: "Values list changed".to_string(),
                });
            }
            if min1 != min2 {
                let mut p = path.to_vec();
                p.push("min_occurrences".to_string());
                diff.entries.push(DiffEntry {
                    path: p,
                    kind: DiffKind::ParameterChanged {
                        old_value: format!("{:?}", min1),
                        new_value: format!("{:?}", min2),
                    },
                    description: "Min occurrences changed".to_string(),
                });
            }
            if max1 != max2 {
                let mut p = path.to_vec();
                p.push("max_occurrences".to_string());
                diff.entries.push(DiffEntry {
                    path: p,
                    kind: DiffKind::ParameterChanged {
                        old_value: format!("{:?}", max1),
                        new_value: format!("{:?}", max2),
                    },
                    description: "Max occurrences changed".to_string(),
                });
            }
        }

        // Abducible
        (TLExpr::Abducible { name: n1, cost: c1 }, TLExpr::Abducible { name: n2, cost: c2 }) => {
            compare_string_param(n1, n2, path, "name", "Abducible name", diff);
            compare_f64_param(*c1, *c2, path, "cost", "Cost", diff);
        }

        // SymbolLiteral
        (TLExpr::SymbolLiteral(s1), TLExpr::SymbolLiteral(s2)) => {
            compare_string_param(s1, s2, path, "symbol", "Symbol", diff);
        }

        // Match
        (
            TLExpr::Match {
                scrutinee: sc1,
                arms: a1,
            },
            TLExpr::Match {
                scrutinee: sc2,
                arms: a2,
            },
        ) => {
            compare_child(sc1, sc2, path, "scrutinee", diff);
            if a1.len() != a2.len() {
                diff.entries.push(DiffEntry {
                    path: path.to_vec(),
                    kind: DiffKind::ParameterChanged {
                        old_value: format!("{} arms", a1.len()),
                        new_value: format!("{} arms", a2.len()),
                    },
                    description: "Match arm count changed".to_string(),
                });
            } else {
                for (i, ((p1, b1), (p2, b2))) in a1.iter().zip(a2.iter()).enumerate() {
                    if p1 != p2 {
                        diff.entries.push(DiffEntry {
                            path: path.to_vec(),
                            kind: DiffKind::ParameterChanged {
                                old_value: format!("{p1}"),
                                new_value: format!("{p2}"),
                            },
                            description: format!("arm[{i}] pattern changed"),
                        });
                    }
                    compare_child(b1, b2, path, &format!("arm[{i}]"), diff);
                }
            }
        }

        // Catch-all: same type tag but not matched above (should not happen
        // if all variants are covered, but kept for safety)
        _ => {
            let old_dbg = format!("{:?}", old);
            let new_dbg = format!("{:?}", new);
            if old_dbg != new_dbg {
                diff.entries.push(DiffEntry {
                    path: path.to_vec(),
                    kind: DiffKind::ParameterChanged {
                        old_value: old_dbg,
                        new_value: new_dbg,
                    },
                    description: "Expression content changed".to_string(),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorlogic_ir::Term;

    fn pred_a() -> TLExpr {
        TLExpr::pred("a", vec![Term::var("x")])
    }

    fn pred_b() -> TLExpr {
        TLExpr::pred("b", vec![Term::var("x")])
    }

    fn pred_c() -> TLExpr {
        TLExpr::pred("c", vec![Term::var("y")])
    }

    #[test]
    fn test_diff_identical() {
        let e = pred_a();
        let diff = expr_diff(&e, &e);
        assert!(diff.is_identical());
        assert_eq!(diff.change_count(), 0);
    }

    #[test]
    fn test_diff_different_type() {
        let old = TLExpr::and(pred_a(), pred_b());
        let new = TLExpr::or(pred_a(), pred_b());
        let diff = expr_diff(&old, &new);
        assert!(!diff.is_identical());
        assert!(diff.entries.iter().any(
            |e| matches!(&e.kind, DiffKind::TypeChanged { old_type, new_type }
                if old_type == "And" && new_type == "Or")
        ));
    }

    #[test]
    fn test_diff_pred_name_change() {
        let old = TLExpr::pred("a", vec![Term::var("x")]);
        let new = TLExpr::pred("b", vec![Term::var("x")]);
        let diff = expr_diff(&old, &new);
        assert!(!diff.is_identical());
        assert!(diff.entries.iter().any(
            |e| matches!(&e.kind, DiffKind::ParameterChanged { old_value, new_value }
                if old_value == "a" && new_value == "b")
        ));
    }

    #[test]
    fn test_diff_pred_arg_change() {
        let old = TLExpr::pred("p", vec![Term::var("x")]);
        let new = TLExpr::pred("p", vec![Term::var("y")]);
        let diff = expr_diff(&old, &new);
        assert!(!diff.is_identical());
        assert_eq!(diff.change_count(), 1);
        let entry = &diff.entries[0];
        assert_eq!(entry.path, vec!["arg0".to_string()]);
        assert!(matches!(&entry.kind, DiffKind::ParameterChanged { .. }));
    }

    #[test]
    fn test_diff_constant_change() {
        let old = TLExpr::Constant(1.0);
        let new = TLExpr::Constant(2.0);
        let diff = expr_diff(&old, &new);
        assert!(!diff.is_identical());
        assert_eq!(diff.change_count(), 1);
        assert!(diff
            .entries
            .iter()
            .any(|e| matches!(&e.kind, DiffKind::ParameterChanged { .. })));
    }

    #[test]
    fn test_diff_added_not() {
        let old = pred_a();
        let new = TLExpr::negate(pred_a());
        let diff = expr_diff(&old, &new);
        assert!(!diff.is_identical());
        assert!(diff.entries.iter().any(
            |e| matches!(&e.kind, DiffKind::TypeChanged { old_type, new_type }
                if old_type == "Pred" && new_type == "Not")
        ));
    }

    #[test]
    fn test_diff_change_count() {
        let old = TLExpr::and(pred_a(), pred_b());
        let new = TLExpr::and(pred_b(), pred_c());
        let diff = expr_diff(&old, &new);
        // left child: name a->b; right child: name b->c, arg0 x->y
        assert!(diff.change_count() >= 2);
    }

    #[test]
    fn test_diff_summary() {
        let old = TLExpr::Constant(1.0);
        let new = TLExpr::Constant(2.0);
        let diff = expr_diff(&old, &new);
        let s = diff.summary();
        assert!(s.contains("changes"));
        assert!(s.contains("modified"));
    }

    #[test]
    fn test_diff_additions() {
        let mut diff = ExprDiff::new();
        diff.entries.push(DiffEntry {
            path: vec!["a".to_string()],
            kind: DiffKind::Added,
            description: "added".to_string(),
        });
        diff.entries.push(DiffEntry {
            path: vec!["b".to_string()],
            kind: DiffKind::Removed,
            description: "removed".to_string(),
        });
        assert_eq!(diff.additions().len(), 1);
        assert_eq!(diff.additions()[0].path, vec!["a".to_string()]);
    }

    #[test]
    fn test_diff_removals() {
        let mut diff = ExprDiff::new();
        diff.entries.push(DiffEntry {
            path: vec!["a".to_string()],
            kind: DiffKind::Added,
            description: "added".to_string(),
        });
        diff.entries.push(DiffEntry {
            path: vec!["b".to_string()],
            kind: DiffKind::Removed,
            description: "removed".to_string(),
        });
        assert_eq!(diff.removals().len(), 1);
        assert_eq!(diff.removals()[0].path, vec!["b".to_string()]);
    }

    #[test]
    fn test_diff_modifications() {
        let mut diff = ExprDiff::new();
        diff.entries.push(DiffEntry {
            path: vec![],
            kind: DiffKind::TypeChanged {
                old_type: "And".to_string(),
                new_type: "Or".to_string(),
            },
            description: "type".to_string(),
        });
        diff.entries.push(DiffEntry {
            path: vec![],
            kind: DiffKind::ParameterChanged {
                old_value: "a".to_string(),
                new_value: "b".to_string(),
            },
            description: "param".to_string(),
        });
        diff.entries.push(DiffEntry {
            path: vec![],
            kind: DiffKind::Added,
            description: "added".to_string(),
        });
        assert_eq!(diff.modifications().len(), 2);
    }

    #[test]
    fn test_diff_kind_is_change() {
        assert!(!DiffKind::Unchanged.is_change());
        assert!(DiffKind::Added.is_change());
        assert!(DiffKind::Removed.is_change());
        assert!(DiffKind::ChildrenChanged.is_change());
        assert!((DiffKind::TypeChanged {
            old_type: "A".to_string(),
            new_type: "B".to_string(),
        })
        .is_change());
        assert!((DiffKind::ParameterChanged {
            old_value: "a".to_string(),
            new_value: "b".to_string(),
        })
        .is_change());
    }

    #[test]
    fn test_diff_nested_change() {
        let old = TLExpr::and(pred_a(), pred_b());
        let new = TLExpr::and(pred_a(), pred_c());
        let diff = expr_diff(&old, &new);
        assert!(!diff.is_identical());
        // Right child changed: name b->c, arg y instead of x
        assert!(diff.change_count() >= 1);
        // Check that at least one path starts with "right"
        assert!(diff
            .entries
            .iter()
            .any(|e| e.path.first().is_some_and(|p| p == "right")));
    }

    #[test]
    fn test_diff_quantifier_change() {
        let body = pred_a();
        let old = TLExpr::exists("x", "D", body.clone());
        let new = TLExpr::forall("x", "D", body);
        let diff = expr_diff(&old, &new);
        assert!(!diff.is_identical());
        assert!(diff.entries.iter().any(
            |e| matches!(&e.kind, DiffKind::TypeChanged { old_type, new_type }
                if old_type == "Exists" && new_type == "ForAll")
        ));
    }

    #[test]
    fn test_diff_entry_path() {
        let old = TLExpr::and(TLExpr::or(pred_a(), pred_b()), TLExpr::Constant(1.0));
        let new = TLExpr::and(TLExpr::or(pred_a(), pred_c()), TLExpr::Constant(1.0));
        let diff = expr_diff(&old, &new);
        assert!(!diff.is_identical());
        // The change is at left -> right -> (name or arg0)
        assert!(diff
            .entries
            .iter()
            .any(|e| e.path.len() >= 2 && e.path[0] == "left" && e.path[1] == "right"));
    }

    #[test]
    fn test_expr_type_tag_pred() {
        let e = pred_a();
        assert_eq!(expr_type_tag(&e), "Pred");
    }

    #[test]
    fn test_expr_type_tag_and() {
        let e = TLExpr::and(pred_a(), pred_b());
        assert_eq!(expr_type_tag(&e), "And");
    }

    #[test]
    fn test_diff_default_empty() {
        let diff = ExprDiff::new();
        assert!(diff.entries.is_empty());
        assert!(diff.is_identical());
        assert_eq!(diff.change_count(), 0);
    }
}

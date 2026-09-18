//! Join ordering optimization for logic queries.
//!
//! Given a set of predicates/atoms to be joined, finds the cheapest join order
//! based on selectivity estimates. This is a classic Datalog/DB query optimization
//! problem relevant to TensorLogic's inference engine.
//!
//! ## Strategies
//!
//! - **Dynamic Programming (System R style)**: For small numbers of relations
//!   (≤ `max_relations`), enumerates all subset partitions to find the optimal plan.
//! - **Greedy**: For larger queries, repeatedly picks the cheapest next join. O(n²).

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// Relation
// ---------------------------------------------------------------------------

/// A relation (predicate) with arity and estimated cardinality.
#[derive(Debug, Clone)]
pub struct Relation {
    /// Name of the relation / predicate.
    pub name: String,
    /// Number of columns (arity).
    pub arity: usize,
    /// Estimated number of rows (tuples).
    pub estimated_rows: u64,
    /// Which columns have known bindings (constants / previously bound variables).
    pub bound_columns: BTreeSet<usize>,
}

impl Relation {
    /// Create a new relation with no bound columns.
    pub fn new(name: impl Into<String>, arity: usize, estimated_rows: u64) -> Self {
        Self {
            name: name.into(),
            arity,
            estimated_rows,
            bound_columns: BTreeSet::new(),
        }
    }

    /// Mark a column as bound and return self (builder pattern).
    pub fn with_binding(mut self, col: usize) -> Self {
        self.bound_columns.insert(col);
        self
    }

    /// Fraction of columns that are bound. Returns 0.0 when arity is 0.
    pub fn selectivity(&self) -> f64 {
        if self.arity == 0 {
            return 0.0;
        }
        self.bound_columns.len() as f64 / self.arity as f64
    }
}

// ---------------------------------------------------------------------------
// JoinCondition
// ---------------------------------------------------------------------------

/// A join condition between two relations (equi-join on one column each).
#[derive(Debug, Clone)]
pub struct JoinCondition {
    pub left_relation: String,
    pub left_column: usize,
    pub right_relation: String,
    pub right_column: usize,
}

// ---------------------------------------------------------------------------
// JoinPlanNode
// ---------------------------------------------------------------------------

/// A node in a join plan tree.
#[derive(Debug, Clone)]
pub enum JoinPlanNode {
    /// Leaf: scan a single relation.
    Scan {
        relation: String,
        estimated_cost: u64,
    },
    /// Hash join (good for larger inner relations).
    HashJoin {
        left: Box<JoinPlanNode>,
        right: Box<JoinPlanNode>,
        conditions: Vec<JoinCondition>,
        estimated_cost: u64,
        estimated_rows: u64,
    },
    /// Nested-loop join (fallback / small inner).
    NestedLoopJoin {
        left: Box<JoinPlanNode>,
        right: Box<JoinPlanNode>,
        conditions: Vec<JoinCondition>,
        estimated_cost: u64,
        estimated_rows: u64,
    },
}

impl JoinPlanNode {
    /// Total estimated cost of this sub-plan.
    pub fn cost(&self) -> u64 {
        match self {
            Self::Scan { estimated_cost, .. } => *estimated_cost,
            Self::HashJoin { estimated_cost, .. } => *estimated_cost,
            Self::NestedLoopJoin { estimated_cost, .. } => *estimated_cost,
        }
    }

    /// Estimated number of output rows.
    pub fn estimated_output_rows(&self) -> u64 {
        match self {
            Self::Scan { estimated_cost, .. } => *estimated_cost, // rows == cost for scans
            Self::HashJoin { estimated_rows, .. } => *estimated_rows,
            Self::NestedLoopJoin { estimated_rows, .. } => *estimated_rows,
        }
    }

    /// Depth of the plan tree (leaf = 1).
    pub fn depth(&self) -> usize {
        match self {
            Self::Scan { .. } => 1,
            Self::HashJoin { left, right, .. } | Self::NestedLoopJoin { left, right, .. } => {
                1 + left.depth().max(right.depth())
            }
        }
    }

    /// Collect all relation names involved in this sub-plan.
    pub fn relations_involved(&self) -> Vec<String> {
        let mut out = Vec::new();
        self.collect_relations(&mut out);
        out
    }

    fn collect_relations(&self, out: &mut Vec<String>) {
        match self {
            Self::Scan { relation, .. } => out.push(relation.clone()),
            Self::HashJoin { left, right, .. } | Self::NestedLoopJoin { left, right, .. } => {
                left.collect_relations(out);
                right.collect_relations(out);
            }
        }
    }

    /// Recursive helper for `format_tree`.
    fn format_tree_inner(&self, indent: usize, buf: &mut String) {
        let pad = " ".repeat(indent);
        match self {
            Self::Scan {
                relation,
                estimated_cost,
            } => {
                buf.push_str(&format!("{pad}Scan({relation}, cost={estimated_cost})\n"));
            }
            Self::HashJoin {
                left,
                right,
                estimated_cost,
                estimated_rows,
                ..
            } => {
                buf.push_str(&format!(
                    "{pad}HashJoin(cost={estimated_cost}, rows={estimated_rows})\n"
                ));
                left.format_tree_inner(indent + 2, buf);
                right.format_tree_inner(indent + 2, buf);
            }
            Self::NestedLoopJoin {
                left,
                right,
                estimated_cost,
                estimated_rows,
                ..
            } => {
                buf.push_str(&format!(
                    "{pad}NestedLoopJoin(cost={estimated_cost}, rows={estimated_rows})\n"
                ));
                left.format_tree_inner(indent + 2, buf);
                right.format_tree_inner(indent + 2, buf);
            }
        }
    }

    /// Recursive DOT helper. Returns the node id assigned.
    fn format_dot_inner(&self, counter: &mut usize, buf: &mut String) -> usize {
        let id = *counter;
        *counter += 1;
        match self {
            Self::Scan {
                relation,
                estimated_cost,
            } => {
                buf.push_str(&format!(
                    "  n{id} [label=\"Scan({relation})\\ncost={estimated_cost}\"];\n"
                ));
            }
            Self::HashJoin {
                left,
                right,
                estimated_cost,
                estimated_rows,
                ..
            } => {
                buf.push_str(&format!(
                    "  n{id} [label=\"HashJoin\\ncost={estimated_cost} rows={estimated_rows}\"];\n"
                ));
                let lid = left.format_dot_inner(counter, buf);
                let rid = right.format_dot_inner(counter, buf);
                buf.push_str(&format!("  n{id} -> n{lid};\n"));
                buf.push_str(&format!("  n{id} -> n{rid};\n"));
            }
            Self::NestedLoopJoin {
                left,
                right,
                estimated_cost,
                estimated_rows,
                ..
            } => {
                buf.push_str(&format!(
                    "  n{id} [label=\"NLJoin\\ncost={estimated_cost} rows={estimated_rows}\"];\n"
                ));
                let lid = left.format_dot_inner(counter, buf);
                let rid = right.format_dot_inner(counter, buf);
                buf.push_str(&format!("  n{id} -> n{lid};\n"));
                buf.push_str(&format!("  n{id} -> n{rid};\n"));
            }
        }
        id
    }
}

// ---------------------------------------------------------------------------
// JoinStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for a join plan.
#[derive(Debug, Clone)]
pub struct JoinStats {
    pub relations_scanned: usize,
    pub joins_performed: usize,
    pub total_estimated_cost: u64,
    pub total_estimated_rows: u64,
    pub plan_depth: usize,
}

// ---------------------------------------------------------------------------
// JoinPlan
// ---------------------------------------------------------------------------

/// A complete join plan with root node and statistics.
#[derive(Debug, Clone)]
pub struct JoinPlan {
    pub root: JoinPlanNode,
    pub stats: JoinStats,
}

impl JoinPlan {
    /// Indented tree representation.
    pub fn format_tree(&self) -> String {
        let mut buf = String::new();
        self.root.format_tree_inner(0, &mut buf);
        buf
    }

    /// DOT graph representation.
    pub fn format_dot(&self) -> String {
        let mut buf = String::from("digraph JoinPlan {\n");
        let mut counter = 0usize;
        self.root.format_dot_inner(&mut counter, &mut buf);
        buf.push_str("}\n");
        buf
    }

    /// Total estimated cost of the plan.
    pub fn total_cost(&self) -> u64 {
        self.root.cost()
    }
}

// ---------------------------------------------------------------------------
// JoinOptimizerConfig
// ---------------------------------------------------------------------------

/// Configuration for the join order optimizer.
#[derive(Debug, Clone)]
pub struct JoinOptimizerConfig {
    /// Above this count the optimizer falls back to greedy.
    pub max_relations: usize,
    /// Use hash join when the inner relation exceeds this many rows.
    pub hash_join_threshold: u64,
    /// Assumed selectivity when unknown.
    pub default_selectivity: f64,
    /// When true, place the smaller relation on the left (build side) in hash joins.
    pub prefer_small_left: bool,
}

impl Default for JoinOptimizerConfig {
    fn default() -> Self {
        Self {
            max_relations: 10,
            hash_join_threshold: 100,
            default_selectivity: 0.1,
            prefer_small_left: true,
        }
    }
}

// ---------------------------------------------------------------------------
// JoinOrderError
// ---------------------------------------------------------------------------

/// Errors from the join optimizer.
#[derive(Debug, Clone)]
pub enum JoinOrderError {
    /// No relations were provided.
    NoRelations,
    /// The join graph is disconnected.
    DisconnectedGraph(String),
    /// Too many relations for the requested strategy.
    TooManyRelations { count: usize, max: usize },
    /// A join condition references a relation not in the input set.
    InvalidCondition(String),
}

impl fmt::Display for JoinOrderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoRelations => write!(f, "no relations provided for join ordering"),
            Self::DisconnectedGraph(msg) => write!(f, "disconnected join graph: {msg}"),
            Self::TooManyRelations { count, max } => {
                write!(
                    f,
                    "too many relations ({count}) for exhaustive search (max {max})"
                )
            }
            Self::InvalidCondition(msg) => write!(f, "invalid join condition: {msg}"),
        }
    }
}

impl std::error::Error for JoinOrderError {}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Enumerate all subsets of size `k` from `{0..n-1}`.
fn subsets_of_size(n: usize, k: usize) -> Vec<BTreeSet<usize>> {
    let mut result = Vec::new();
    if k > n {
        return result;
    }
    let mut indices: Vec<usize> = (0..k).collect();
    loop {
        result.push(indices.iter().copied().collect());
        // Advance to next combination
        let mut i = k;
        loop {
            if i == 0 {
                return result;
            }
            i -= 1;
            if indices[i] != i + n - k {
                break;
            }
            if i == 0 {
                return result;
            }
        }
        indices[i] += 1;
        for j in (i + 1)..k {
            indices[j] = indices[j - 1] + 1;
        }
    }
}

/// Selectivity estimation helper.
///
/// Returns a value in `(0, 1]` estimating the fraction of the cross-product
/// that survives the join.
pub fn estimate_selectivity(
    left_rows: u64,
    right_rows: u64,
    num_conditions: usize,
    default_selectivity: f64,
) -> f64 {
    if num_conditions == 0 {
        return 1.0; // cross product
    }
    let max_side = left_rows.max(right_rows).max(1) as f64;
    // Each condition filters by roughly 1/max(|L|,|R|), capped by default_selectivity.
    let per_cond = (1.0 / max_side).max(default_selectivity);
    let sel = per_cond.powi(num_conditions as i32);
    sel.clamp(f64::MIN_POSITIVE, 1.0)
}

// ---------------------------------------------------------------------------
// JoinOrderOptimizer
// ---------------------------------------------------------------------------

/// The join order optimizer.
pub struct JoinOrderOptimizer {
    config: JoinOptimizerConfig,
}

impl JoinOrderOptimizer {
    /// Create with explicit configuration.
    pub fn new(config: JoinOptimizerConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    pub fn with_default() -> Self {
        Self::new(JoinOptimizerConfig::default())
    }

    /// Find optimal join order for a set of relations with join conditions.
    ///
    /// For `<= max_relations`: dynamic programming (System R style).
    /// For `> max_relations`: greedy (smallest estimated output first).
    pub fn optimize(
        &self,
        relations: &[Relation],
        conditions: &[JoinCondition],
    ) -> Result<JoinPlan, JoinOrderError> {
        if relations.is_empty() {
            return Err(JoinOrderError::NoRelations);
        }

        // Validate conditions reference known relations
        let known: HashSet<&str> = relations.iter().map(|r| r.name.as_str()).collect();
        for c in conditions {
            if !known.contains(c.left_relation.as_str()) {
                return Err(JoinOrderError::InvalidCondition(format!(
                    "unknown relation '{}'",
                    c.left_relation
                )));
            }
            if !known.contains(c.right_relation.as_str()) {
                return Err(JoinOrderError::InvalidCondition(format!(
                    "unknown relation '{}'",
                    c.right_relation
                )));
            }
        }

        let root = if relations.len() > self.config.max_relations {
            self.greedy_order(relations, conditions)?
        } else {
            self.dp_order(relations, conditions)?
        };

        let rels = root.relations_involved();
        let joins = if rels.len() > 1 { rels.len() - 1 } else { 0 };
        let stats = JoinStats {
            relations_scanned: rels.len(),
            joins_performed: joins,
            total_estimated_cost: root.cost(),
            total_estimated_rows: root.estimated_output_rows(),
            plan_depth: root.depth(),
        };

        Ok(JoinPlan { root, stats })
    }

    /// Greedy join ordering: always pick the cheapest next join.
    fn greedy_order(
        &self,
        relations: &[Relation],
        conditions: &[JoinCondition],
    ) -> Result<JoinPlanNode, JoinOrderError> {
        if relations.len() == 1 {
            let r = &relations[0];
            return Ok(JoinPlanNode::Scan {
                relation: r.name.clone(),
                estimated_cost: r.estimated_rows,
            });
        }

        // Build initial nodes sorted by estimated_rows ascending.
        let mut nodes: Vec<JoinPlanNode> = {
            let mut v: Vec<_> = relations.iter().collect();
            v.sort_by_key(|r| r.estimated_rows);
            v.into_iter()
                .map(|r| JoinPlanNode::Scan {
                    relation: r.name.clone(),
                    estimated_cost: r.estimated_rows,
                })
                .collect()
        };

        while nodes.len() > 1 {
            let mut best_i = 0;
            let mut best_j = 1;
            let mut best_cost = u64::MAX;
            let mut best_rows = u64::MAX;

            for i in 0..nodes.len() {
                for j in (i + 1)..nodes.len() {
                    let left_rels: HashSet<String> =
                        nodes[i].relations_involved().into_iter().collect();
                    let right_rels: HashSet<String> =
                        nodes[j].relations_involved().into_iter().collect();
                    let conds = Self::find_conditions(&left_rels, &right_rels, conditions);
                    let (cost, rows) = self.estimate_join_cost(&nodes[i], &nodes[j], &conds);
                    if cost < best_cost || (cost == best_cost && rows < best_rows) {
                        best_cost = cost;
                        best_rows = rows;
                        best_i = i;
                        best_j = j;
                    }
                }
            }

            // Remove j first (larger index) then i.
            let right_node = nodes.remove(best_j);
            let left_node = nodes.remove(best_i);

            let left_rels: HashSet<String> = left_node.relations_involved().into_iter().collect();
            let right_rels: HashSet<String> = right_node.relations_involved().into_iter().collect();
            let conds = Self::find_conditions(&left_rels, &right_rels, conditions);
            let (cost, rows) = self.estimate_join_cost(&left_node, &right_node, &conds);

            let joined = self.make_join_node(left_node, right_node, conds, cost, rows);
            nodes.push(joined);
        }

        // Safety: we checked len >= 1 above and the loop leaves exactly 1 element.
        Ok(nodes
            .into_iter()
            .next()
            .unwrap_or_else(|| JoinPlanNode::Scan {
                relation: String::new(),
                estimated_cost: 0,
            }))
    }

    /// Dynamic programming (System R style) for small number of relations.
    ///
    /// `dp[S]` = best plan for subset S of relations.
    fn dp_order(
        &self,
        relations: &[Relation],
        conditions: &[JoinCondition],
    ) -> Result<JoinPlanNode, JoinOrderError> {
        let n = relations.len();
        if n == 1 {
            let r = &relations[0];
            return Ok(JoinPlanNode::Scan {
                relation: r.name.clone(),
                estimated_cost: r.estimated_rows,
            });
        }

        // Map index → relation name.
        let idx_to_name: Vec<&str> = relations.iter().map(|r| r.name.as_str()).collect();

        // dp table: BTreeSet<usize> → (best plan, cost)
        let mut dp: HashMap<BTreeSet<usize>, (JoinPlanNode, u64)> = HashMap::new();

        // Base case: single relations.
        for (i, r) in relations.iter().enumerate() {
            let mut set = BTreeSet::new();
            set.insert(i);
            let node = JoinPlanNode::Scan {
                relation: r.name.clone(),
                estimated_cost: r.estimated_rows,
            };
            dp.insert(set, (node, r.estimated_rows));
        }

        // Enumerate subset sizes 2..=n
        for size in 2..=n {
            let subsets = subsets_of_size(n, size);
            for subset in &subsets {
                let mut best: Option<(JoinPlanNode, u64)> = None;

                // Try all non-empty proper subsets s1 of subset.
                // We enumerate s1 as subsets of `subset` with size 1..size-1.
                let elems: Vec<usize> = subset.iter().copied().collect();
                let m = elems.len();

                for s1_size in 1..m {
                    let s1_subsets = subsets_of_size(m, s1_size);
                    for s1_indices in &s1_subsets {
                        let s1: BTreeSet<usize> =
                            s1_indices.iter().map(|&idx| elems[idx]).collect();
                        let s2: BTreeSet<usize> = subset.difference(&s1).copied().collect();

                        if s2.is_empty() {
                            continue;
                        }

                        let (left_plan, _left_cost) = match dp.get(&s1) {
                            Some(v) => v,
                            None => continue,
                        };
                        let (right_plan, _right_cost) = match dp.get(&s2) {
                            Some(v) => v,
                            None => continue,
                        };

                        // Find join conditions between s1 and s2
                        let left_names: HashSet<String> =
                            s1.iter().map(|&i| idx_to_name[i].to_string()).collect();
                        let right_names: HashSet<String> =
                            s2.iter().map(|&i| idx_to_name[i].to_string()).collect();
                        let conds = Self::find_conditions(&left_names, &right_names, conditions);

                        let (cost, rows) = self.estimate_join_cost(left_plan, right_plan, &conds);

                        if best.as_ref().is_none_or(|(_, bc)| cost < *bc) {
                            let node = self.make_join_node(
                                left_plan.clone(),
                                right_plan.clone(),
                                conds,
                                cost,
                                rows,
                            );
                            best = Some((node, cost));
                        }
                    }
                }

                if let Some(entry) = best {
                    dp.insert(subset.clone(), entry);
                }
            }
        }

        // Retrieve full set.
        let full: BTreeSet<usize> = (0..n).collect();
        dp.remove(&full).map(|(node, _)| node).ok_or_else(|| {
            JoinOrderError::DisconnectedGraph(
                "could not find a plan covering all relations".to_string(),
            )
        })
    }

    /// Estimate the cost of joining two sub-plans.
    fn estimate_join_cost(
        &self,
        left: &JoinPlanNode,
        right: &JoinPlanNode,
        conditions: &[JoinCondition],
    ) -> (u64, u64) {
        let left_rows = left.estimated_output_rows().max(1);
        let right_rows = right.estimated_output_rows().max(1);

        let selectivity = estimate_selectivity(
            left_rows,
            right_rows,
            conditions.len(),
            self.config.default_selectivity,
        );

        let output_rows =
            ((left_rows as f64 * right_rows as f64 * selectivity).ceil() as u64).max(1);

        let use_hash = right_rows > self.config.hash_join_threshold;
        let join_cost = if use_hash {
            // hash join: build + probe ≈ left + right + output
            left_rows + right_rows + output_rows
        } else {
            // nested loop: left * right (but at least left + right)
            (left_rows.saturating_mul(right_rows)).max(left_rows + right_rows)
        };

        let total_cost = left
            .cost()
            .saturating_add(right.cost())
            .saturating_add(join_cost);
        (total_cost, output_rows)
    }

    /// Find applicable join conditions between two sets of relations.
    fn find_conditions(
        left_rels: &HashSet<String>,
        right_rels: &HashSet<String>,
        all_conditions: &[JoinCondition],
    ) -> Vec<JoinCondition> {
        all_conditions
            .iter()
            .filter(|c| {
                (left_rels.contains(&c.left_relation) && right_rels.contains(&c.right_relation))
                    || (left_rels.contains(&c.right_relation)
                        && right_rels.contains(&c.left_relation))
            })
            .cloned()
            .collect()
    }

    /// Construct the appropriate join node based on config.
    fn make_join_node(
        &self,
        left: JoinPlanNode,
        right: JoinPlanNode,
        conditions: Vec<JoinCondition>,
        estimated_cost: u64,
        estimated_rows: u64,
    ) -> JoinPlanNode {
        let right_rows = right.estimated_output_rows();
        let use_hash = right_rows > self.config.hash_join_threshold;

        let (left, right) = if self.config.prefer_small_left && use_hash {
            if left.estimated_output_rows() > right.estimated_output_rows() {
                (right, left)
            } else {
                (left, right)
            }
        } else {
            (left, right)
        };

        if use_hash {
            JoinPlanNode::HashJoin {
                left: Box::new(left),
                right: Box::new(right),
                conditions,
                estimated_cost,
                estimated_rows,
            }
        } else {
            JoinPlanNode::NestedLoopJoin {
                left: Box::new(left),
                right: Box::new(right),
                conditions,
                estimated_cost,
                estimated_rows,
            }
        }
    }
}

impl Default for JoinOrderOptimizer {
    fn default() -> Self {
        Self::with_default()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relation_new() {
        let r = Relation::new("users", 3, 1000);
        assert_eq!(r.name, "users");
        assert_eq!(r.arity, 3);
        assert_eq!(r.estimated_rows, 1000);
        assert!(r.bound_columns.is_empty());
    }

    #[test]
    fn test_relation_with_binding() {
        let r = Relation::new("users", 3, 1000)
            .with_binding(0)
            .with_binding(2);
        assert!(r.bound_columns.contains(&0));
        assert!(r.bound_columns.contains(&2));
        assert!(!r.bound_columns.contains(&1));
        assert_eq!(r.bound_columns.len(), 2);
    }

    #[test]
    fn test_relation_selectivity() {
        let r = Relation::new("users", 4, 1000)
            .with_binding(0)
            .with_binding(1);
        let sel = r.selectivity();
        assert!((sel - 0.5).abs() < 1e-10);

        let r_zero = Relation::new("empty", 0, 0);
        assert!((r_zero.selectivity() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_join_config_default() {
        let cfg = JoinOptimizerConfig::default();
        assert_eq!(cfg.max_relations, 10);
        assert_eq!(cfg.hash_join_threshold, 100);
        assert!((cfg.default_selectivity - 0.1).abs() < 1e-10);
        assert!(cfg.prefer_small_left);
    }

    #[test]
    fn test_greedy_single_relation() {
        let opt = JoinOrderOptimizer::with_default();
        let rels = vec![Relation::new("users", 3, 100)];
        let plan = opt.optimize(&rels, &[]).expect("should succeed");
        assert!(matches!(plan.root, JoinPlanNode::Scan { .. }));
        assert_eq!(plan.stats.relations_scanned, 1);
        assert_eq!(plan.stats.joins_performed, 0);
    }

    #[test]
    fn test_greedy_two_relations() {
        let opt = JoinOrderOptimizer::with_default();
        let rels = vec![
            Relation::new("users", 2, 500),
            Relation::new("orders", 3, 2000),
        ];
        let conds = vec![JoinCondition {
            left_relation: "users".to_string(),
            left_column: 0,
            right_relation: "orders".to_string(),
            right_column: 1,
        }];
        let plan = opt.optimize(&rels, &conds).expect("should succeed");
        assert_eq!(plan.stats.relations_scanned, 2);
        assert_eq!(plan.stats.joins_performed, 1);
        assert!(plan.root.cost() > 0);
    }

    #[test]
    fn test_greedy_three_relations() {
        let opt = JoinOrderOptimizer::with_default();
        let rels = vec![
            Relation::new("a", 2, 100),
            Relation::new("b", 2, 200),
            Relation::new("c", 2, 300),
        ];
        let conds = vec![
            JoinCondition {
                left_relation: "a".to_string(),
                left_column: 0,
                right_relation: "b".to_string(),
                right_column: 0,
            },
            JoinCondition {
                left_relation: "b".to_string(),
                left_column: 1,
                right_relation: "c".to_string(),
                right_column: 0,
            },
        ];
        let plan = opt.optimize(&rels, &conds).expect("should succeed");
        assert_eq!(plan.stats.relations_scanned, 3);
        assert_eq!(plan.stats.joins_performed, 2);
        assert!(plan.root.depth() >= 2);
    }

    #[test]
    fn test_dp_two_relations() {
        let opt = JoinOrderOptimizer::with_default();
        let rels = vec![Relation::new("x", 2, 50), Relation::new("y", 2, 80)];
        let conds = vec![JoinCondition {
            left_relation: "x".to_string(),
            left_column: 0,
            right_relation: "y".to_string(),
            right_column: 0,
        }];
        let plan = opt.optimize(&rels, &conds).expect("should succeed");
        assert_eq!(plan.stats.relations_scanned, 2);
        assert_eq!(plan.stats.joins_performed, 1);
    }

    #[test]
    fn test_dp_three_relations() {
        let opt = JoinOrderOptimizer::with_default();
        let rels = vec![
            Relation::new("r1", 2, 10),
            Relation::new("r2", 2, 20),
            Relation::new("r3", 2, 30),
        ];
        let conds = vec![
            JoinCondition {
                left_relation: "r1".to_string(),
                left_column: 0,
                right_relation: "r2".to_string(),
                right_column: 0,
            },
            JoinCondition {
                left_relation: "r2".to_string(),
                left_column: 1,
                right_relation: "r3".to_string(),
                right_column: 0,
            },
        ];
        let plan = opt.optimize(&rels, &conds).expect("should succeed");
        assert_eq!(plan.stats.relations_scanned, 3);
        assert_eq!(plan.stats.joins_performed, 2);
        assert!(plan.root.depth() >= 2);
    }

    #[test]
    fn test_optimize_uses_greedy_when_too_many() {
        let cfg = JoinOptimizerConfig {
            max_relations: 2,
            ..Default::default()
        };
        let opt = JoinOrderOptimizer::new(cfg);
        let rels = vec![
            Relation::new("a", 2, 10),
            Relation::new("b", 2, 20),
            Relation::new("c", 2, 30),
        ];
        let conds = vec![
            JoinCondition {
                left_relation: "a".to_string(),
                left_column: 0,
                right_relation: "b".to_string(),
                right_column: 0,
            },
            JoinCondition {
                left_relation: "b".to_string(),
                left_column: 1,
                right_relation: "c".to_string(),
                right_column: 0,
            },
        ];
        // Should succeed using greedy fallback (3 > max_relations=2)
        let plan = opt.optimize(&rels, &conds).expect("greedy fallback");
        assert_eq!(plan.stats.relations_scanned, 3);
    }

    #[test]
    fn test_optimize_no_relations_error() {
        let opt = JoinOrderOptimizer::with_default();
        let result = opt.optimize(&[], &[]);
        assert!(result.is_err());
        assert!(matches!(result, Err(JoinOrderError::NoRelations)));
    }

    #[test]
    fn test_join_plan_node_cost() {
        let node = JoinPlanNode::Scan {
            relation: "t".to_string(),
            estimated_cost: 42,
        };
        assert_eq!(node.cost(), 42);
        assert!(node.cost() > 0);
    }

    #[test]
    fn test_join_plan_node_depth() {
        let leaf = JoinPlanNode::Scan {
            relation: "t".to_string(),
            estimated_cost: 10,
        };
        assert_eq!(leaf.depth(), 1);

        let join = JoinPlanNode::HashJoin {
            left: Box::new(JoinPlanNode::Scan {
                relation: "a".to_string(),
                estimated_cost: 5,
            }),
            right: Box::new(JoinPlanNode::Scan {
                relation: "b".to_string(),
                estimated_cost: 10,
            }),
            conditions: vec![],
            estimated_cost: 20,
            estimated_rows: 8,
        };
        assert_eq!(join.depth(), 2);
    }

    #[test]
    fn test_join_plan_node_relations() {
        let join = JoinPlanNode::HashJoin {
            left: Box::new(JoinPlanNode::Scan {
                relation: "a".to_string(),
                estimated_cost: 5,
            }),
            right: Box::new(JoinPlanNode::Scan {
                relation: "b".to_string(),
                estimated_cost: 10,
            }),
            conditions: vec![],
            estimated_cost: 20,
            estimated_rows: 8,
        };
        let rels = join.relations_involved();
        assert!(rels.contains(&"a".to_string()));
        assert!(rels.contains(&"b".to_string()));
        assert_eq!(rels.len(), 2);
    }

    #[test]
    fn test_join_plan_format_tree() {
        let opt = JoinOrderOptimizer::with_default();
        let rels = vec![Relation::new("a", 2, 100), Relation::new("b", 2, 200)];
        let conds = vec![JoinCondition {
            left_relation: "a".to_string(),
            left_column: 0,
            right_relation: "b".to_string(),
            right_column: 0,
        }];
        let plan = opt.optimize(&rels, &conds).expect("ok");
        let tree = plan.format_tree();
        assert!(!tree.is_empty());
    }

    #[test]
    fn test_join_plan_format_dot() {
        let opt = JoinOrderOptimizer::with_default();
        let rels = vec![Relation::new("a", 2, 100), Relation::new("b", 2, 200)];
        let conds = vec![JoinCondition {
            left_relation: "a".to_string(),
            left_column: 0,
            right_relation: "b".to_string(),
            right_column: 0,
        }];
        let plan = opt.optimize(&rels, &conds).expect("ok");
        let dot = plan.format_dot();
        assert!(dot.contains("digraph"));
    }

    #[test]
    fn test_estimate_selectivity() {
        let sel = estimate_selectivity(1000, 2000, 1, 0.1);
        assert!(sel > 0.0);
        assert!(sel <= 1.0);

        // No conditions → cross product selectivity = 1.0
        let sel_cross = estimate_selectivity(100, 100, 0, 0.1);
        assert!((sel_cross - 1.0).abs() < 1e-10);

        // Multiple conditions → smaller selectivity
        let sel_one = estimate_selectivity(100, 200, 1, 0.1);
        let sel_two = estimate_selectivity(100, 200, 2, 0.1);
        assert!(sel_two < sel_one);
    }

    #[test]
    fn test_find_conditions() {
        let conds = vec![
            JoinCondition {
                left_relation: "a".to_string(),
                left_column: 0,
                right_relation: "b".to_string(),
                right_column: 0,
            },
            JoinCondition {
                left_relation: "b".to_string(),
                left_column: 1,
                right_relation: "c".to_string(),
                right_column: 0,
            },
        ];

        let left: HashSet<String> = ["a".to_string()].into_iter().collect();
        let right: HashSet<String> = ["b".to_string()].into_iter().collect();
        let found = JoinOrderOptimizer::find_conditions(&left, &right, &conds);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].left_relation, "a");

        let left2: HashSet<String> = ["a".to_string()].into_iter().collect();
        let right2: HashSet<String> = ["c".to_string()].into_iter().collect();
        let found2 = JoinOrderOptimizer::find_conditions(&left2, &right2, &conds);
        assert_eq!(found2.len(), 0);
    }

    #[test]
    fn test_join_stats() {
        let opt = JoinOrderOptimizer::with_default();
        let rels = vec![
            Relation::new("a", 2, 100),
            Relation::new("b", 2, 200),
            Relation::new("c", 2, 300),
        ];
        let conds = vec![
            JoinCondition {
                left_relation: "a".to_string(),
                left_column: 0,
                right_relation: "b".to_string(),
                right_column: 0,
            },
            JoinCondition {
                left_relation: "b".to_string(),
                left_column: 1,
                right_relation: "c".to_string(),
                right_column: 0,
            },
        ];
        let plan = opt.optimize(&rels, &conds).expect("ok");
        assert_eq!(plan.stats.relations_scanned, 3);
        assert_eq!(plan.stats.joins_performed, 2);
        assert!(plan.stats.total_estimated_cost > 0);
        assert!(plan.stats.total_estimated_rows > 0);
        assert!(plan.stats.plan_depth >= 2);
    }

    #[test]
    fn test_join_order_error_display() {
        let e1 = JoinOrderError::NoRelations;
        assert!(!e1.to_string().is_empty());

        let e2 = JoinOrderError::DisconnectedGraph("parts missing".to_string());
        assert!(e2.to_string().contains("disconnected"));

        let e3 = JoinOrderError::TooManyRelations { count: 20, max: 10 };
        assert!(e3.to_string().contains("20"));

        let e4 = JoinOrderError::InvalidCondition("bad ref".to_string());
        assert!(e4.to_string().contains("invalid"));
    }
}

/// Represents a SPARQL triple pattern
#[derive(Debug, Clone, PartialEq)]
pub struct TriplePattern {
    pub subject: PatternElement,
    pub predicate: PatternElement,
    pub object: PatternElement,
}

/// Element in a triple pattern (variable or constant)
#[derive(Debug, Clone, PartialEq)]
pub enum PatternElement {
    Variable(String),
    Constant(String),
}

/// Filter condition in SPARQL
#[derive(Debug, Clone, PartialEq)]
pub enum FilterCondition {
    Equals(String, String),
    NotEquals(String, String),
    GreaterThan(String, String),
    LessThan(String, String),
    GreaterOrEqual(String, String),
    LessOrEqual(String, String),
    Regex(String, String),
    Bound(String),
    IsIri(String),
    IsLiteral(String),
}

/// Aggregate function in SPARQL
#[derive(Debug, Clone, PartialEq)]
pub enum AggregateFunction {
    /// COUNT aggregate - counts solutions
    Count {
        variable: Option<String>,
        distinct: bool,
    },
    /// SUM aggregate - sums numeric values
    Sum { variable: String, distinct: bool },
    /// AVG aggregate - computes average
    Avg { variable: String, distinct: bool },
    /// MIN aggregate - finds minimum value
    Min { variable: String },
    /// MAX aggregate - finds maximum value
    Max { variable: String },
    /// GROUP_CONCAT aggregate - concatenates strings
    GroupConcat {
        variable: String,
        separator: Option<String>,
        distinct: bool,
    },
    /// SAMPLE aggregate - returns arbitrary value
    Sample { variable: String },
}

/// A projection element that can be a variable or an aggregate expression
#[derive(Debug, Clone, PartialEq)]
pub enum SelectElement {
    /// Simple variable projection
    Variable(String),
    /// Aggregate expression with optional alias
    Aggregate {
        function: AggregateFunction,
        alias: Option<String>,
    },
}

/// Right-hand side of a BIND clause.
///
/// Intentionally narrow: only Term-level expressions supported this release.
/// Arithmetic and function-call expressions require per-row executor context
/// (the same limitation present on `FilterCondition` execution) and are
/// filed as a follow-up.
#[derive(Debug, Clone, PartialEq)]
pub enum BindExpr {
    /// A single `PatternElement` — either a constant/IRI or a variable reference.
    Term(PatternElement),
}

/// Graph pattern in SPARQL (supports complex patterns)
#[derive(Debug, Clone, PartialEq)]
pub enum GraphPattern {
    /// Basic triple pattern
    Triple(TriplePattern),
    /// Conjunction of patterns (implicit AND)
    Group(Vec<GraphPattern>),
    /// OPTIONAL pattern (left-outer join)
    Optional(Box<GraphPattern>),
    /// UNION pattern (disjunction)
    Union(Box<GraphPattern>, Box<GraphPattern>),
    /// FILTER constraint
    Filter(FilterCondition),
    /// BIND (expr AS ?var) — bind expression result to a new variable.
    Bind(BindExpr, String),
    /// VALUES (?v1 ?v2 …) { (t1 t2) … } — inline value table.
    Values(Vec<String>, Vec<Vec<PatternElement>>),
}

/// Type of SPARQL query
#[derive(Debug, Clone, PartialEq)]
pub enum QueryType {
    /// SELECT query - returns variable bindings
    Select {
        /// Projection elements (variables and aggregates)
        projections: Vec<SelectElement>,
        /// Legacy field for simple variable names (for backward compatibility)
        select_vars: Vec<String>,
        distinct: bool,
    },
    /// ASK query - returns boolean (existence check)
    Ask,
    /// DESCRIBE query - returns RDF description of resources
    Describe { resources: Vec<String> },
    /// CONSTRUCT query - constructs new RDF triples
    Construct { template: Vec<TriplePattern> },
}

/// Compiled SPARQL query
#[derive(Debug, Clone)]
pub struct SparqlQuery {
    /// Type of query (SELECT, ASK, DESCRIBE, CONSTRUCT)
    pub query_type: QueryType,
    /// WHERE clause graph patterns
    pub where_pattern: GraphPattern,
    /// GROUP BY variables
    pub group_by: Vec<String>,
    /// HAVING conditions (applied after grouping)
    pub having: Vec<FilterCondition>,
    /// Solution modifiers
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub order_by: Vec<String>,
}

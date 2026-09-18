//! Compile-execute-validate pipeline tying together TensorLogic rules and RDF.
//!
//! [`ValidationExecutor`] compiles a [`TLExpr`] to an [`tensorlogic_ir::EinsumGraph`] using
//! [`tensorlogic_compiler`], executes it with [`Scirs2Exec`] via the
//! [`TlAutodiff`] trait, generates a SHACL-style [`ValidationReport`], and can
//! export the execution result as RDF Turtle.
//!
//! # Pipeline overview
//!
//! ```text
//! TLExpr
//!   │  compile_to_einsum()
//!   ▼
//! EinsumGraph
//!   │  Scirs2Exec::forward()
//!   ▼
//! Scirs2Tensor  ──►  ExecutionResult
//!                        │  generate_validation_report()
//!                        ▼
//!                   ValidationReport
//!                        │  export_as_rdf()
//!                        ▼
//!                   Turtle string
//! ```

use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::time::Instant;

use tensorlogic_compiler::compile_to_einsum;
use tensorlogic_infer::{ExecutorError, TlAutodiff};
use tensorlogic_ir::TLExpr;
use tensorlogic_scirs_backend::Scirs2Exec;

use crate::shacl::validation::{ValidationReport, ValidationResult, ValidationSeverity};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`ValidationExecutor`].
///
/// All fields have sensible defaults available via `Default::default()` /
/// `ValidationExecutorConfig::default()`.
#[derive(Debug, Clone)]
pub struct ValidationExecutorConfig {
    /// Maximum number of elements allowed per output tensor.
    ///
    /// If the forward pass produces a tensor with more elements than this limit,
    /// [`ValidationExecutorError::TensorTooLarge`] is returned.  Default: 65536.
    pub max_tensor_size: usize,

    /// Decimal places used when formatting float values in the RDF Turtle export.
    /// Default: 6.
    pub float_precision: usize,

    /// Base IRI for generated RDF triples.  Default: `"https://tensorlogic.local/"`.
    pub base_iri: String,
}

impl Default for ValidationExecutorConfig {
    fn default() -> Self {
        Self {
            max_tensor_size: 65536,
            float_precision: 6,
            base_iri: "https://tensorlogic.local/".into(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Timing / statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Timing and size statistics captured during [`ValidationExecutor::execute_rule`].
#[derive(Debug, Clone)]
pub struct ExecutionStats {
    /// Wall-clock microseconds spent in the compilation phase.
    pub compile_time_us: u64,
    /// Wall-clock microseconds spent in the executor forward pass.
    pub execute_time_us: u64,
    /// Number of operation nodes in the compiled [`tensorlogic_ir::EinsumGraph`].
    pub graph_node_count: usize,
    /// Number of output tensors collected from the forward pass.
    pub output_tensor_count: usize,
    /// Total number of scalar elements across all output tensors.
    pub total_elements: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Output tensor wrapper
// ─────────────────────────────────────────────────────────────────────────────

/// A named, flattened snapshot of one output tensor.
#[derive(Debug, Clone)]
pub struct ExecutionTensor {
    /// Logical name for this tensor (e.g. `"output"`).
    pub name: String,
    /// Shape of the tensor (product equals `values.len()`).
    pub shape: Vec<usize>,
    /// Flattened element values in row-major order.
    pub values: Vec<f64>,
}

impl ExecutionTensor {
    /// Returns `true` if any element is `NaN`.
    pub fn has_nan(&self) -> bool {
        self.values.iter().any(|v| v.is_nan())
    }

    /// Returns `true` if any element is positive or negative infinity.
    pub fn has_inf(&self) -> bool {
        self.values.iter().any(|v| v.is_infinite())
    }

    /// Returns `true` when every element is a finite number (not NaN, not Inf).
    pub fn all_finite(&self) -> bool {
        self.values.iter().all(|v| v.is_finite())
    }

    /// Minimum element value, or `None` if the tensor is empty.
    pub fn min_value(&self) -> Option<f64> {
        self.values.iter().copied().reduce(f64::min)
    }

    /// Maximum element value, or `None` if the tensor is empty.
    pub fn max_value(&self) -> Option<f64> {
        self.values.iter().copied().reduce(f64::max)
    }

    /// Count of elements that are not finite (NaN or Inf).
    pub fn non_finite_count(&self) -> usize {
        self.values.iter().filter(|v| !v.is_finite()).count()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Execution result
// ─────────────────────────────────────────────────────────────────────────────

/// The result of a successful [`ValidationExecutor::execute_rule`] call.
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// `Debug` representation of the input [`TLExpr`].
    pub expression_repr: String,
    /// Number of operation nodes in the compiled graph.
    pub graph_node_count: usize,
    /// Output tensors produced by the forward pass.
    pub output_tensors: Vec<ExecutionTensor>,
    /// Timing and size statistics for this execution.
    pub stats: ExecutionStats,
}

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during the compile-execute-validate pipeline.
#[derive(Debug)]
pub enum ValidationExecutorError {
    /// The [`tensorlogic_compiler`] rejected the expression.
    ///
    /// The inner [`anyhow::Error`] carries the full diagnostic chain produced
    /// by the compiler (type errors, unsupported constructs, etc.).
    Compile(anyhow::Error),
    /// The [`Scirs2Exec`] forward pass failed.
    Execute(ExecutorError),
    /// An output tensor exceeded the configured `max_tensor_size` limit.
    TensorTooLarge {
        /// Name of the offending tensor.
        name: String,
        /// Actual number of elements.
        size: usize,
        /// Configured maximum.
        max: usize,
    },
    /// The compiled [`tensorlogic_ir::EinsumGraph`] contains no tensors or nodes.
    EmptyGraph,
}

impl fmt::Display for ValidationExecutorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compile(e) => write!(f, "compilation error: {e}"),
            Self::Execute(e) => write!(f, "executor error: {e}"),
            Self::TensorTooLarge { name, size, max } => write!(
                f,
                "output tensor '{name}' has {size} elements which exceeds the limit of {max}"
            ),
            Self::EmptyGraph => write!(f, "compiled graph is empty (no tensors or nodes)"),
        }
    }
}

impl std::error::Error for ValidationExecutorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            // anyhow::Error does not implement std::error::Error directly
            // so we cannot return it as a source; the Display chain already
            // includes the full context.
            Self::Compile(_) => None,
            Self::Execute(e) => Some(e),
            Self::TensorTooLarge { .. } | Self::EmptyGraph => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Executor
// ─────────────────────────────────────────────────────────────────────────────

/// Compile-execute-validate pipeline for TensorLogic expressions.
///
/// # Example
///
/// ```rust
/// use tensorlogic_ir::{TLExpr, Term};
/// use tensorlogic_oxirs_bridge::{ValidationExecutor, ValidationExecutorConfig};
///
/// let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
/// let executor = ValidationExecutor::new(ValidationExecutorConfig::default());
/// let result = executor.execute_rule(&expr).unwrap();
/// let report = executor.generate_validation_report(&result);
/// assert!(report.conforms);
/// let rdf = executor.export_as_rdf(&result);
/// assert!(rdf.contains("@prefix tl:"));
/// ```
pub struct ValidationExecutor {
    config: ValidationExecutorConfig,
}

impl ValidationExecutor {
    /// Create a new executor with the given configuration.
    pub fn new(config: ValidationExecutorConfig) -> Self {
        Self { config }
    }

    /// Return a reference to the active configuration.
    pub fn config(&self) -> &ValidationExecutorConfig {
        &self.config
    }

    // ── Core pipeline ──────────────────────────────────────────────────────

    /// Compile `expr` to an [`tensorlogic_ir::EinsumGraph`], execute it with placeholder
    /// input tensors via [`Scirs2Exec`], and return an [`ExecutionResult`].
    ///
    /// # Errors
    ///
    /// - [`ValidationExecutorError::Compile`] — compiler rejected the expression.
    /// - [`ValidationExecutorError::EmptyGraph`] — compiled graph is trivially empty.
    /// - [`ValidationExecutorError::Execute`] — forward pass failed.
    /// - [`ValidationExecutorError::TensorTooLarge`] — output exceeds configured limit.
    pub fn execute_rule(&self, expr: &TLExpr) -> Result<ExecutionResult, ValidationExecutorError> {
        // ── Phase 1: compile ──────────────────────────────────────────────
        let t_compile_start = Instant::now();
        let graph = compile_to_einsum(expr).map_err(ValidationExecutorError::Compile)?;
        let compile_time_us = t_compile_start.elapsed().as_micros() as u64;

        if graph.is_empty() {
            return Err(ValidationExecutorError::EmptyGraph);
        }

        // ── Phase 2: pre-populate placeholder tensors ─────────────────────
        //
        // The executor looks up named tensors in `self.tensors`.  Tensors
        // whose names start with `const_` are handled automatically by the
        // forward pass (it parses the numeric suffix).  All others get a
        // deterministic scalar placeholder so that the graph can execute
        // without real data.
        let t_exec_start = Instant::now();
        let mut exec = Scirs2Exec::new();

        for (i, tensor_name) in graph.tensors.iter().enumerate() {
            // Strip any axis-annotation suffix (e.g. "age[a]" → "age")
            let base_name = tensor_name
                .split('[')
                .next()
                .unwrap_or(tensor_name.as_str());

            if base_name.starts_with("const_") || tensor_name.starts_with("const_") {
                // Auto-handled by the forward pass — nothing to pre-load.
                continue;
            }

            // Deterministic non-zero placeholder: 0.1 + 0.1*(i % 9).
            // Use a 1-element 1-D tensor (shape [1]) rather than a 0-D scalar
            // so that einsum specs with at least one index can address axis 0
            // without a shape-mismatch error.
            let val = 0.1 + 0.1 * (i % 9) as f64;
            let placeholder = scirs2_core::ndarray::Array1::from_vec(vec![val]).into_dyn();
            exec.add_tensor(tensor_name.clone(), placeholder);
        }

        // ── Phase 3: forward pass ─────────────────────────────────────────
        let result_tensor = exec
            .forward(&graph)
            .map_err(ValidationExecutorError::Execute)?;
        let execute_time_us = t_exec_start.elapsed().as_micros() as u64;

        // ── Phase 4: collect output ───────────────────────────────────────
        let shape: Vec<usize> = result_tensor.shape().to_vec();
        let values: Vec<f64> = result_tensor.iter().copied().collect();
        let total_elements = values.len();

        if total_elements > self.config.max_tensor_size {
            return Err(ValidationExecutorError::TensorTooLarge {
                name: "output".to_string(),
                size: total_elements,
                max: self.config.max_tensor_size,
            });
        }

        let output_tensor = ExecutionTensor {
            name: "output".to_string(),
            shape,
            values,
        };

        let graph_node_count = graph.nodes.len();

        Ok(ExecutionResult {
            expression_repr: format!("{expr:?}"),
            graph_node_count,
            output_tensors: vec![output_tensor],
            stats: ExecutionStats {
                compile_time_us,
                execute_time_us,
                graph_node_count,
                output_tensor_count: 1,
                total_elements,
            },
        })
    }

    // ── Validation report ──────────────────────────────────────────────────

    /// Generate a SHACL-style [`ValidationReport`] from an [`ExecutionResult`].
    ///
    /// The report conforms (`report.conforms == true`) when every output tensor
    /// contains only finite values.  A `sh:Violation`-severity
    /// [`ValidationResult`] is added for each tensor that contains NaN or Inf
    /// elements, carrying the tensor name and non-finite count in the message.
    pub fn generate_validation_report(&self, result: &ExecutionResult) -> ValidationReport {
        let mut report = ValidationReport::new();

        for tensor in &result.output_tensors {
            if !tensor.all_finite() {
                let non_finite = tensor.non_finite_count();
                let has_nan = tensor.has_nan();
                let has_inf = tensor.has_inf();

                let kind_desc = match (has_nan, has_inf) {
                    (true, true) => "NaN and Inf values",
                    (true, false) => "NaN values",
                    (false, true) => "Inf values",
                    (false, false) => "non-finite values",
                };

                let message = format!(
                    "Output tensor '{}' contains {} {} (shape: {:?})",
                    tensor.name, non_finite, kind_desc, tensor.shape,
                );

                let focus_node = format!("{}tensor/{}", self.config.base_iri, tensor.name);
                let source_shape = format!("{}shape/FiniteValueConstraint", self.config.base_iri);
                let constraint_component = format!(
                    "{}constraint/FiniteValueConstraintComponent",
                    self.config.base_iri
                );

                let vr =
                    ValidationResult::new(focus_node, source_shape, constraint_component, message)
                        .with_severity(ValidationSeverity::Violation)
                        .with_value(format!("{non_finite} non-finite elements"));

                report.add_result(vr);
            }
        }

        report
    }

    // ── RDF Turtle export ──────────────────────────────────────────────────

    /// Serialise an [`ExecutionResult`] as an RDF Turtle string.
    ///
    /// The generated document uses a `tl:` prefix for the configured
    /// `base_iri` and declares standard XSD types.  Each execution is
    /// identified by a stable IRI derived from the `expression_repr` hash so
    /// that repeated calls for the same expression produce consistent IRIs.
    ///
    /// Each output tensor is appended as a blank-node `tl:OutputTensor` with
    /// shape, element count, allFinite, min, and max.
    pub fn export_as_rdf(&self, result: &ExecutionResult) -> String {
        let base_iri = &self.config.base_iri;
        let prec = self.config.float_precision;

        // Stable hash of the expression representation.
        let exec_hash = {
            let mut h = DefaultHasher::new();
            result.expression_repr.hash(&mut h);
            h.finish()
        };

        // Escape the expression repr for use in a Turtle string literal.
        let escaped_repr = escape_turtle_literal(&result.expression_repr);

        // Whether all outputs are fully finite.
        let all_conforms = result.output_tensors.iter().all(|t| t.all_finite());

        let mut out = String::with_capacity(512);

        // Prefix declarations
        out.push_str(&format!("@prefix tl: <{base_iri}> .\n"));
        out.push_str("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n");
        out.push('\n');

        // Main execution result node
        out.push_str(&format!(
            "tl:exec_{exec_hash:016x} a tl:ExecutionResult ;\n"
        ));
        out.push_str(&format!("    tl:expressionRepr \"{escaped_repr}\" ;\n"));
        out.push_str(&format!(
            "    tl:graphNodeCount {graph_node_count}^^xsd:integer ;\n",
            graph_node_count = result.graph_node_count
        ));
        out.push_str(&format!(
            "    tl:compileTimeUs {compile_us}^^xsd:integer ;\n",
            compile_us = result.stats.compile_time_us
        ));
        out.push_str(&format!(
            "    tl:executeTimeUs {execute_us}^^xsd:integer ;\n",
            execute_us = result.stats.execute_time_us
        ));
        out.push_str(&format!(
            "    tl:totalElements {total}^^xsd:integer ;\n",
            total = result.stats.total_elements
        ));
        out.push_str(&format!(
            "    tl:conforms {conforms}^^xsd:boolean",
            conforms = all_conforms
        ));

        if result.output_tensors.is_empty() {
            out.push_str(" .\n");
        } else {
            // Link to blank-node tensor descriptions
            out.push_str(" ;\n");
            let tensor_count = result.output_tensors.len();
            for (idx, tensor) in result.output_tensors.iter().enumerate() {
                let is_last = idx == tensor_count - 1;
                let tensor_node = format_tensor_blank_node(tensor, prec);
                if is_last {
                    out.push_str(&format!("    tl:outputTensor {tensor_node} .\n"));
                } else {
                    out.push_str(&format!("    tl:outputTensor {tensor_node} ;\n"));
                }
            }
        }

        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Escape a string for safe embedding in a Turtle string literal.
fn escape_turtle_literal(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out
}

/// Format an [`ExecutionTensor`] as an inline Turtle blank node.
fn format_tensor_blank_node(tensor: &ExecutionTensor, prec: usize) -> String {
    let shape_str: Vec<String> = tensor.shape.iter().map(|d| d.to_string()).collect();
    let shape_literal = shape_str.join(",");
    let all_finite = tensor.all_finite();

    let min_str = tensor
        .min_value()
        .map(|v| format!("{v:.prec$}"))
        .unwrap_or_else(|| "null".to_string());
    let max_str = tensor
        .max_value()
        .map(|v| format!("{v:.prec$}"))
        .unwrap_or_else(|| "null".to_string());

    let mut node = String::new();
    node.push_str("[\n");
    node.push_str("        a tl:OutputTensor ;\n");
    node.push_str(&format!(
        "        tl:tensorName \"{name}\" ;\n",
        name = escape_turtle_literal(&tensor.name)
    ));
    node.push_str(&format!("        tl:shape \"{shape_literal}\" ;\n",));
    node.push_str(&format!(
        "        tl:elementCount {count}^^xsd:integer ;\n",
        count = tensor.values.len()
    ));
    node.push_str(&format!(
        "        tl:allFinite {all_finite}^^xsd:boolean ;\n",
    ));
    node.push_str(&format!(
        "        tl:minValue \"{min_str}\"^^xsd:decimal ;\n",
    ));
    node.push_str(&format!("        tl:maxValue \"{max_str}\"^^xsd:decimal\n",));
    node.push_str("    ]");
    node
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use tensorlogic_ir::{TLExpr, Term};

    use super::*;

    fn default_executor() -> ValidationExecutor {
        ValidationExecutor::new(ValidationExecutorConfig::default())
    }

    #[test]
    fn test_simple_predicate_compiles_and_runs() {
        let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
        let executor = default_executor();
        let result = executor
            .execute_rule(&expr)
            .expect("execute simple predicate");
        // A simple predicate may compile to a graph with zero computational nodes
        // (pass-through with no connectives).  We only verify it ran successfully.
        assert!(
            !result.output_tensors.is_empty(),
            "expected at least one output tensor"
        );
    }

    #[test]
    fn test_finite_output_conforms() {
        let expr = TLExpr::pred("p", vec![Term::var("x")]);
        let executor = default_executor();
        let result = executor.execute_rule(&expr).expect("execute predicate");
        let report = executor.generate_validation_report(&result);
        // Placeholder inputs are finite, so the result should conform.
        assert!(report.conforms, "finite outputs should conform");
    }

    #[test]
    fn test_export_rdf_contains_required_prefixes() {
        let expr = TLExpr::pred("q", vec![Term::var("a")]);
        let executor = default_executor();
        let result = executor.execute_rule(&expr).expect("execute");
        let rdf = executor.export_as_rdf(&result);
        assert!(rdf.contains("@prefix tl:"), "missing tl: prefix");
        assert!(rdf.contains("@prefix xsd:"), "missing xsd: prefix");
        assert!(
            rdf.contains("tl:ExecutionResult"),
            "missing ExecutionResult type"
        );
    }

    #[test]
    fn test_export_rdf_conforms_field() {
        let expr = TLExpr::pred("r", vec![Term::var("x")]);
        let executor = default_executor();
        let result = executor.execute_rule(&expr).expect("execute");
        let rdf = executor.export_as_rdf(&result);
        assert!(
            rdf.contains("tl:conforms true") || rdf.contains("tl:conforms false"),
            "missing conforms field in RDF: {rdf}"
        );
    }

    #[test]
    fn test_execution_stats_recorded() {
        // Use a conjunctive expression to guarantee at least one computational node.
        let p = TLExpr::pred("s", vec![Term::var("x")]);
        let q = TLExpr::pred("t", vec![Term::var("x")]);
        let expr = TLExpr::and(p, q);
        let executor = default_executor();
        let result = executor
            .execute_rule(&expr)
            .expect("execute AND expression");
        // An AND expression always compiles to at least one node.
        assert!(
            result.stats.graph_node_count > 0,
            "expected at least one graph node"
        );
    }

    #[test]
    fn test_max_tensor_size_zero_returns_error_or_empty() {
        let config = ValidationExecutorConfig {
            max_tensor_size: 0,
            ..Default::default()
        };
        let expr = TLExpr::pred("t", vec![Term::var("x")]);
        let executor = ValidationExecutor::new(config);
        // Either succeeds with an empty tensor or fails with TensorTooLarge.
        // Neither path should panic.
        let _ = executor.execute_rule(&expr);
    }

    #[test]
    fn test_execution_tensor_helpers_all_finite() {
        let t = ExecutionTensor {
            name: "test".to_string(),
            shape: vec![3],
            values: vec![1.0, 2.0, 3.0],
        };
        assert!(t.all_finite());
        assert!(!t.has_nan());
        assert!(!t.has_inf());
        assert_eq!(t.min_value(), Some(1.0));
        assert_eq!(t.max_value(), Some(3.0));
        assert_eq!(t.non_finite_count(), 0);
    }

    #[test]
    fn test_execution_tensor_helpers_with_nan() {
        let t = ExecutionTensor {
            name: "bad".to_string(),
            shape: vec![2],
            values: vec![f64::NAN, 1.0],
        };
        assert!(!t.all_finite());
        assert!(t.has_nan());
        assert_eq!(t.non_finite_count(), 1);
    }

    #[test]
    fn test_error_display_empty_graph() {
        let e = ValidationExecutorError::EmptyGraph;
        assert!(e.to_string().contains("empty"), "unexpected: {e}");
    }

    #[test]
    fn test_error_display_tensor_too_large() {
        let e = ValidationExecutorError::TensorTooLarge {
            name: "out".into(),
            size: 100,
            max: 50,
        };
        let s = e.to_string();
        assert!(s.contains("out"), "unexpected: {s}");
        assert!(s.contains("100"), "unexpected: {s}");
        assert!(s.contains("50"), "unexpected: {s}");
    }

    #[test]
    fn test_escape_turtle_literal_special_chars() {
        let raw = "Hello\nworld\\foo\"bar";
        let escaped = escape_turtle_literal(raw);
        assert!(escaped.contains("\\n"), "newline not escaped");
        assert!(escaped.contains("\\\\"), "backslash not escaped");
        assert!(escaped.contains("\\\""), "quote not escaped");
    }

    #[test]
    fn test_validation_report_for_infinite_tensor() {
        let executor = default_executor();
        let result = ExecutionResult {
            expression_repr: "test".to_string(),
            graph_node_count: 1,
            output_tensors: vec![ExecutionTensor {
                name: "output".to_string(),
                shape: vec![1],
                values: vec![f64::INFINITY],
            }],
            stats: ExecutionStats {
                compile_time_us: 0,
                execute_time_us: 0,
                graph_node_count: 1,
                output_tensor_count: 1,
                total_elements: 1,
            },
        };
        let report = executor.generate_validation_report(&result);
        assert!(!report.conforms, "Inf tensor should not conform");
        assert!(
            !report.results.is_empty(),
            "expected at least one violation"
        );
    }
}

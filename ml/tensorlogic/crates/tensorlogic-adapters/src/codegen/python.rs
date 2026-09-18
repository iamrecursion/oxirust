use std::fmt::Write as FmtWrite;

use crate::{DomainInfo, PredicateInfo, SymbolTable};

use super::rust::RustCodegen;

/// Code generator for Python type stubs and PyO3 bindings.
///
/// This generator creates Python type stubs (.pyi) and optionally PyO3
/// binding code from TensorLogic schemas.
pub struct PythonCodegen {
    /// Module name
    module_name: String,
    /// Whether to generate PyO3 bindings (vs. just stubs)
    generate_pyo3: bool,
    /// Whether to include docstrings
    include_docs: bool,
    /// Whether to generate dataclass decorators
    use_dataclasses: bool,
}

impl PythonCodegen {
    /// Create a new Python code generator.
    pub fn new(module_name: impl Into<String>) -> Self {
        Self {
            module_name: module_name.into(),
            generate_pyo3: false,
            include_docs: true,
            use_dataclasses: true,
        }
    }

    /// Set whether to generate PyO3 bindings.
    pub fn with_pyo3(mut self, enable: bool) -> Self {
        self.generate_pyo3 = enable;
        self
    }

    /// Set whether to include docstrings.
    pub fn with_docs(mut self, enable: bool) -> Self {
        self.include_docs = enable;
        self
    }

    /// Set whether to use dataclasses.
    pub fn with_dataclasses(mut self, enable: bool) -> Self {
        self.use_dataclasses = enable;
        self
    }

    /// Generate complete Python module from a symbol table.
    pub fn generate(&self, table: &SymbolTable) -> String {
        if self.generate_pyo3 {
            self.generate_pyo3_bindings(table)
        } else {
            self.generate_type_stubs(table)
        }
    }

    /// Generate Python type stubs (.pyi file).
    fn generate_type_stubs(&self, table: &SymbolTable) -> String {
        let mut code = String::new();

        // Module header
        writeln!(code, "\"\"\"").expect("writing to String is infallible");
        writeln!(code, "Generated from TensorLogic schema")
            .expect("writing to String is infallible");
        writeln!(code, "Module: {}", self.module_name).expect("writing to String is infallible");
        writeln!(code).expect("writing to String is infallible");
        writeln!(code, "This code was automatically generated.")
            .expect("writing to String is infallible");
        writeln!(code, "DO NOT EDIT MANUALLY.").expect("writing to String is infallible");
        writeln!(code, "\"\"\"").expect("writing to String is infallible");
        writeln!(code).expect("writing to String is infallible");

        // Imports
        writeln!(code, "from typing import NewType, Final")
            .expect("writing to String is infallible");
        if self.use_dataclasses {
            writeln!(code, "from dataclasses import dataclass")
                .expect("writing to String is infallible");
        }
        writeln!(code).expect("writing to String is infallible");

        // Generate domain types
        writeln!(code, "# ==========================================")
            .expect("writing to String is infallible");
        writeln!(code, "# Domain Types").expect("writing to String is infallible");
        writeln!(code, "# ==========================================")
            .expect("writing to String is infallible");
        writeln!(code).expect("writing to String is infallible");

        for domain in table.domains.values() {
            self.generate_domain_stub(&mut code, domain);
            writeln!(code).expect("writing to String is infallible");
        }

        // Generate predicate types
        writeln!(code, "# ==========================================")
            .expect("writing to String is infallible");
        writeln!(code, "# Predicate Types").expect("writing to String is infallible");
        writeln!(code, "# ==========================================")
            .expect("writing to String is infallible");
        writeln!(code).expect("writing to String is infallible");

        for predicate in table.predicates.values() {
            self.generate_predicate_stub(&mut code, predicate, table);
            writeln!(code).expect("writing to String is infallible");
        }

        // Generate schema metadata
        writeln!(code, "# ==========================================")
            .expect("writing to String is infallible");
        writeln!(code, "# Schema Metadata").expect("writing to String is infallible");
        writeln!(code, "# ==========================================")
            .expect("writing to String is infallible");
        writeln!(code).expect("writing to String is infallible");
        self.generate_schema_metadata_stub(&mut code, table);

        code
    }

    /// Generate PyO3 Rust bindings.
    fn generate_pyo3_bindings(&self, table: &SymbolTable) -> String {
        let mut code = String::new();

        // Module header
        writeln!(code, "//! PyO3 bindings for TensorLogic schema")
            .expect("writing to String is infallible");
        writeln!(code, "//! Module: {}", self.module_name)
            .expect("writing to String is infallible");
        writeln!(code, "//!").expect("writing to String is infallible");
        writeln!(code, "//! This code was automatically generated.")
            .expect("writing to String is infallible");
        writeln!(code, "//! DO NOT EDIT MANUALLY.").expect("writing to String is infallible");
        writeln!(code).expect("writing to String is infallible");

        writeln!(code, "use pyo3::prelude::*;").expect("writing to String is infallible");
        writeln!(code).expect("writing to String is infallible");

        // Generate domain classes
        writeln!(code, "// ==========================================")
            .expect("writing to String is infallible");
        writeln!(code, "// Domain Types").expect("writing to String is infallible");
        writeln!(code, "// ==========================================")
            .expect("writing to String is infallible");
        writeln!(code).expect("writing to String is infallible");

        for domain in table.domains.values() {
            self.generate_domain_pyo3(&mut code, domain);
            writeln!(code).expect("writing to String is infallible");
        }

        // Generate predicate classes
        writeln!(code, "// ==========================================")
            .expect("writing to String is infallible");
        writeln!(code, "// Predicate Types").expect("writing to String is infallible");
        writeln!(code, "// ==========================================")
            .expect("writing to String is infallible");
        writeln!(code).expect("writing to String is infallible");

        for predicate in table.predicates.values() {
            self.generate_predicate_pyo3(&mut code, predicate);
            writeln!(code).expect("writing to String is infallible");
        }

        // Generate module registration
        writeln!(code, "// ==========================================")
            .expect("writing to String is infallible");
        writeln!(code, "// Module Registration").expect("writing to String is infallible");
        writeln!(code, "// ==========================================")
            .expect("writing to String is infallible");
        writeln!(code).expect("writing to String is infallible");
        self.generate_module_registration(&mut code, table);

        code
    }

    /// Generate Python type stub for a domain.
    fn generate_domain_stub(&self, code: &mut String, domain: &DomainInfo) {
        let type_name = Self::to_python_class_name(&domain.name);

        // NewType for branded ID
        writeln!(code, "{} = NewType('{}', int)", type_name, type_name)
            .expect("writing to String is infallible");
        writeln!(code).expect("writing to String is infallible");

        // Cardinality constant
        writeln!(
            code,
            "{}_CARDINALITY: Final[int] = {}",
            domain.name.to_uppercase(),
            domain.cardinality
        )
        .expect("writing to String is infallible");
        writeln!(code).expect("writing to String is infallible");

        // Validator function
        writeln!(code, "def is_valid_{}(id: int) -> bool:", domain.name)
            .expect("writing to String is infallible");
        if self.include_docs {
            writeln!(code, "    \"\"\"").expect("writing to String is infallible");
            if let Some(ref desc) = domain.description {
                writeln!(code, "    {}", desc).expect("writing to String is infallible");
                writeln!(code).expect("writing to String is infallible");
            }
            writeln!(code, "    Validate {} ID.", type_name)
                .expect("writing to String is infallible");
            writeln!(code).expect("writing to String is infallible");
            writeln!(code, "    Args:").expect("writing to String is infallible");
            writeln!(code, "        id: The ID to validate")
                .expect("writing to String is infallible");
            writeln!(code).expect("writing to String is infallible");
            writeln!(code, "    Returns:").expect("writing to String is infallible");
            writeln!(
                code,
                "        True if id is in range [0, {}), False otherwise",
                domain.cardinality
            )
            .expect("writing to String is infallible");
            writeln!(code, "    \"\"\"").expect("writing to String is infallible");
        }
        writeln!(code, "    ...").expect("writing to String is infallible");
    }

    /// Generate Python type stub for a predicate.
    fn generate_predicate_stub(
        &self,
        code: &mut String,
        predicate: &PredicateInfo,
        _table: &SymbolTable,
    ) {
        let class_name = Self::to_python_class_name(&predicate.name);

        if self.include_docs {
            writeln!(code, "\"\"\"").expect("writing to String is infallible");
            if let Some(ref desc) = predicate.description {
                writeln!(code, "{}", desc).expect("writing to String is infallible");
            } else {
                writeln!(code, "Predicate: {}", predicate.name)
                    .expect("writing to String is infallible");
            }
            writeln!(code).expect("writing to String is infallible");
            writeln!(code, "Arity: {}", predicate.arg_domains.len())
                .expect("writing to String is infallible");
            writeln!(code, "\"\"\"").expect("writing to String is infallible");
        }

        if self.use_dataclasses {
            writeln!(code, "@dataclass(frozen=True)").expect("writing to String is infallible");
        }

        writeln!(code, "class {}:", class_name).expect("writing to String is infallible");

        if self.include_docs && predicate.description.is_none() {
            writeln!(code, "    \"\"\"{}\"\"\"", predicate.name)
                .expect("writing to String is infallible");
        }

        // Add fields
        for (i, domain_name) in predicate.arg_domains.iter().enumerate() {
            let field_name = format!("arg{}", i);
            let field_type = Self::to_python_class_name(domain_name);
            writeln!(code, "    {}: {}", field_name, field_type)
                .expect("writing to String is infallible");
        }

        if predicate.arg_domains.is_empty() {
            writeln!(code, "    pass").expect("writing to String is infallible");
        }
    }

    /// Generate PyO3 class for a domain.
    fn generate_domain_pyo3(&self, code: &mut String, domain: &DomainInfo) {
        let type_name = Self::to_python_class_name(&domain.name);

        writeln!(code, "#[pyclass]").expect("writing to String is infallible");
        writeln!(code, "#[derive(Clone, Copy, Debug)]").expect("writing to String is infallible");
        writeln!(code, "pub struct {} {{", type_name).expect("writing to String is infallible");
        writeln!(code, "    #[pyo3(get)]").expect("writing to String is infallible");
        writeln!(code, "    pub id: usize,").expect("writing to String is infallible");
        writeln!(code, "}}").expect("writing to String is infallible");
        writeln!(code).expect("writing to String is infallible");

        writeln!(code, "#[pymethods]").expect("writing to String is infallible");
        writeln!(code, "impl {} {{", type_name).expect("writing to String is infallible");

        // Constructor
        writeln!(code, "    #[new]").expect("writing to String is infallible");
        writeln!(code, "    pub fn new(id: usize) -> PyResult<Self> {{")
            .expect("writing to String is infallible");
        writeln!(code, "        if id >= {} {{", domain.cardinality)
            .expect("writing to String is infallible");
        writeln!(
            code,
            "            return Err(pyo3::exceptions::PyValueError::new_err("
        )
        .expect("writing to String is infallible");
        writeln!(
            code,
            "                format!(\"ID {{}} exceeds cardinality {}\", id)",
            domain.cardinality
        )
        .expect("writing to String is infallible");
        writeln!(code, "            ));").expect("writing to String is infallible");
        writeln!(code, "        }}").expect("writing to String is infallible");
        writeln!(code, "        Ok(Self {{ id }})").expect("writing to String is infallible");
        writeln!(code, "    }}").expect("writing to String is infallible");
        writeln!(code).expect("writing to String is infallible");

        // String representation
        writeln!(code, "    fn __repr__(&self) -> String {{")
            .expect("writing to String is infallible");
        writeln!(code, "        format!(\"{}({{}})\", self.id)", type_name)
            .expect("writing to String is infallible");
        writeln!(code, "    }}").expect("writing to String is infallible");

        writeln!(code, "}}").expect("writing to String is infallible");
    }

    /// Generate PyO3 class for a predicate.
    fn generate_predicate_pyo3(&self, code: &mut String, predicate: &PredicateInfo) {
        let type_name = Self::to_python_class_name(&predicate.name);

        writeln!(code, "#[pyclass]").expect("writing to String is infallible");
        writeln!(code, "#[derive(Clone, Debug)]").expect("writing to String is infallible");
        writeln!(code, "pub struct {} {{", type_name).expect("writing to String is infallible");

        for (i, domain_name) in predicate.arg_domains.iter().enumerate() {
            let field_type = Self::to_python_class_name(domain_name);
            writeln!(code, "    #[pyo3(get)]").expect("writing to String is infallible");
            writeln!(code, "    pub arg{}: {},", i, field_type)
                .expect("writing to String is infallible");
        }

        writeln!(code, "}}").expect("writing to String is infallible");
        writeln!(code).expect("writing to String is infallible");

        writeln!(code, "#[pymethods]").expect("writing to String is infallible");
        writeln!(code, "impl {} {{", type_name).expect("writing to String is infallible");

        // Constructor
        writeln!(code, "    #[new]").expect("writing to String is infallible");
        write!(code, "    pub fn new(").expect("writing to String is infallible");
        for (i, domain_name) in predicate.arg_domains.iter().enumerate() {
            if i > 0 {
                write!(code, ", ").expect("writing to String is infallible");
            }
            write!(
                code,
                "arg{}: {}",
                i,
                Self::to_python_class_name(domain_name)
            )
            .expect("writing to String is infallible");
        }
        writeln!(code, ") -> Self {{").expect("writing to String is infallible");

        if predicate.arg_domains.is_empty() {
            writeln!(code, "        Self {{}}").expect("writing to String is infallible");
        } else {
            write!(code, "        Self {{ ").expect("writing to String is infallible");
            for i in 0..predicate.arg_domains.len() {
                if i > 0 {
                    write!(code, ", ").expect("writing to String is infallible");
                }
                write!(code, "arg{}", i).expect("writing to String is infallible");
            }
            writeln!(code, " }}").expect("writing to String is infallible");
        }
        writeln!(code, "    }}").expect("writing to String is infallible");

        writeln!(code, "}}").expect("writing to String is infallible");
    }

    /// Generate module registration for PyO3.
    fn generate_module_registration(&self, code: &mut String, table: &SymbolTable) {
        writeln!(code, "#[pymodule]").expect("writing to String is infallible");
        writeln!(
            code,
            "fn {}(_py: Python, m: &PyModule) -> PyResult<()> {{",
            self.module_name.replace('-', "_")
        )
        .expect("writing to String is infallible");

        // Register domain classes
        for domain in table.domains.values() {
            let type_name = Self::to_python_class_name(&domain.name);
            writeln!(code, "    m.add_class::<{}>()?;", type_name)
                .expect("writing to String is infallible");
        }

        // Register predicate classes
        for predicate in table.predicates.values() {
            let type_name = Self::to_python_class_name(&predicate.name);
            writeln!(code, "    m.add_class::<{}>()?;", type_name)
                .expect("writing to String is infallible");
        }

        writeln!(code, "    Ok(())").expect("writing to String is infallible");
        writeln!(code, "}}").expect("writing to String is infallible");
    }

    /// Generate schema metadata stub.
    fn generate_schema_metadata_stub(&self, code: &mut String, table: &SymbolTable) {
        writeln!(code, "class SchemaMetadata:").expect("writing to String is infallible");
        if self.include_docs {
            writeln!(code, "    \"\"\"Schema metadata and statistics\"\"\"")
                .expect("writing to String is infallible");
        }
        writeln!(
            code,
            "    DOMAIN_COUNT: Final[int] = {}",
            table.domains.len()
        )
        .expect("writing to String is infallible");
        writeln!(
            code,
            "    PREDICATE_COUNT: Final[int] = {}",
            table.predicates.len()
        )
        .expect("writing to String is infallible");

        let total_card: usize = table.domains.values().map(|d| d.cardinality).sum();
        writeln!(code, "    TOTAL_CARDINALITY: Final[int] = {}", total_card)
            .expect("writing to String is infallible");
    }

    /// Convert a name to Python class name (PascalCase).
    fn to_python_class_name(name: &str) -> String {
        RustCodegen::to_type_name(name) // Reuse Rust converter
    }
}

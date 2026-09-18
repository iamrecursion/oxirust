use std::fmt::Write as FmtWrite;

use crate::{DomainInfo, PredicateInfo, SymbolTable};

/// Code generator for Rust types from schemas.
pub struct RustCodegen {
    /// Module name for generated code
    module_name: String,
    /// Whether to derive common traits
    derive_common: bool,
    /// Whether to include documentation comments
    include_docs: bool,
}

impl RustCodegen {
    /// Create a new Rust code generator.
    pub fn new(module_name: impl Into<String>) -> Self {
        Self {
            module_name: module_name.into(),
            derive_common: true,
            include_docs: true,
        }
    }

    /// Set whether to derive common traits (Clone, Debug, etc.).
    pub fn with_common_derives(mut self, enable: bool) -> Self {
        self.derive_common = enable;
        self
    }

    /// Set whether to include documentation comments.
    pub fn with_docs(mut self, enable: bool) -> Self {
        self.include_docs = enable;
        self
    }

    /// Generate complete Rust module from a symbol table.
    pub fn generate(&self, table: &SymbolTable) -> String {
        let mut code = String::new();

        // Module header
        writeln!(code, "//! Generated from TensorLogic schema.")
            .expect("writing to String is infallible");
        writeln!(code, "//! Module: {}", self.module_name)
            .expect("writing to String is infallible");
        writeln!(code, "//!").expect("writing to String is infallible");
        writeln!(code, "//! This code was automatically generated.")
            .expect("writing to String is infallible");
        writeln!(code, "//! DO NOT EDIT MANUALLY.").expect("writing to String is infallible");
        writeln!(code).expect("writing to String is infallible");

        // Use statements
        writeln!(code, "#![allow(dead_code)]").expect("writing to String is infallible");
        writeln!(code).expect("writing to String is infallible");

        // Generate domain types
        writeln!(code, "// ============================================")
            .expect("writing to String is infallible");
        writeln!(code, "// Domain Types").expect("writing to String is infallible");
        writeln!(code, "// ============================================")
            .expect("writing to String is infallible");
        writeln!(code).expect("writing to String is infallible");

        for domain in table.domains.values() {
            self.generate_domain(&mut code, domain);
            writeln!(code).expect("writing to String is infallible");
        }

        // Generate predicate types
        writeln!(code, "// ============================================")
            .expect("writing to String is infallible");
        writeln!(code, "// Predicate Types").expect("writing to String is infallible");
        writeln!(code, "// ============================================")
            .expect("writing to String is infallible");
        writeln!(code).expect("writing to String is infallible");

        for predicate in table.predicates.values() {
            self.generate_predicate(&mut code, predicate, table);
            writeln!(code).expect("writing to String is infallible");
        }

        // Generate schema metadata type
        writeln!(code, "// ============================================")
            .expect("writing to String is infallible");
        writeln!(code, "// Schema Metadata").expect("writing to String is infallible");
        writeln!(code, "// ============================================")
            .expect("writing to String is infallible");
        writeln!(code).expect("writing to String is infallible");
        self.generate_schema_metadata(&mut code, table);

        code
    }

    /// Generate domain type.
    fn generate_domain(&self, code: &mut String, domain: &DomainInfo) {
        if self.include_docs {
            if let Some(ref desc) = domain.description {
                writeln!(code, "/// {}", desc).expect("writing to String is infallible");
            } else {
                writeln!(code, "/// Domain: {}", domain.name)
                    .expect("writing to String is infallible");
            }
            writeln!(code, "///").expect("writing to String is infallible");
            writeln!(code, "/// Cardinality: {}", domain.cardinality)
                .expect("writing to String is infallible");
        }

        if self.derive_common {
            writeln!(code, "#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]")
                .expect("writing to String is infallible");
        }

        let type_name = Self::to_type_name(&domain.name);
        writeln!(code, "pub struct {}(pub usize);", type_name)
            .expect("writing to String is infallible");
        writeln!(code).expect("writing to String is infallible");

        // Generate constructor and accessors
        writeln!(code, "impl {} {{", type_name).expect("writing to String is infallible");
        writeln!(
            code,
            "    /// Maximum valid ID for this domain (exclusive)."
        )
        .expect("writing to String is infallible");
        writeln!(
            code,
            "    pub const CARDINALITY: usize = {};",
            domain.cardinality
        )
        .expect("writing to String is infallible");
        writeln!(code).expect("writing to String is infallible");

        writeln!(code, "    /// Create a new {} instance.", type_name)
            .expect("writing to String is infallible");
        writeln!(code, "    ///").expect("writing to String is infallible");
        writeln!(code, "    /// # Panics").expect("writing to String is infallible");
        writeln!(code, "    ///").expect("writing to String is infallible");
        writeln!(code, "    /// Panics if `id >= {}`.", domain.cardinality)
            .expect("writing to String is infallible");
        writeln!(code, "    pub fn new(id: usize) -> Self {{")
            .expect("writing to String is infallible");
        writeln!(code, "        assert!(id < Self::CARDINALITY, \"ID {{}} exceeds cardinality {{}}\", id, Self::CARDINALITY);", ).expect("writing to String is infallible");
        writeln!(code, "        Self(id)").expect("writing to String is infallible");
        writeln!(code, "    }}").expect("writing to String is infallible");
        writeln!(code).expect("writing to String is infallible");

        writeln!(
            code,
            "    /// Create a new {} instance without bounds checking.",
            type_name
        )
        .expect("writing to String is infallible");
        writeln!(code, "    ///").expect("writing to String is infallible");
        writeln!(code, "    /// # Safety").expect("writing to String is infallible");
        writeln!(code, "    ///").expect("writing to String is infallible");
        writeln!(
            code,
            "    /// Caller must ensure `id < {}`.",
            domain.cardinality
        )
        .expect("writing to String is infallible");
        writeln!(
            code,
            "    pub unsafe fn new_unchecked(id: usize) -> Self {{"
        )
        .expect("writing to String is infallible");
        writeln!(code, "        Self(id)").expect("writing to String is infallible");
        writeln!(code, "    }}").expect("writing to String is infallible");
        writeln!(code).expect("writing to String is infallible");

        writeln!(code, "    /// Get the underlying ID.").expect("writing to String is infallible");
        writeln!(code, "    pub fn id(&self) -> usize {{")
            .expect("writing to String is infallible");
        writeln!(code, "        self.0").expect("writing to String is infallible");
        writeln!(code, "    }}").expect("writing to String is infallible");

        writeln!(code, "}}").expect("writing to String is infallible");
    }

    /// Generate predicate type.
    fn generate_predicate(
        &self,
        code: &mut String,
        predicate: &PredicateInfo,
        _table: &SymbolTable,
    ) {
        if self.include_docs {
            if let Some(ref desc) = predicate.description {
                writeln!(code, "/// {}", desc).expect("writing to String is infallible");
            } else {
                writeln!(code, "/// Predicate: {}", predicate.name)
                    .expect("writing to String is infallible");
            }
            writeln!(code, "///").expect("writing to String is infallible");
            writeln!(code, "/// Arity: {}", predicate.arg_domains.len())
                .expect("writing to String is infallible");

            if let Some(ref constraints) = predicate.constraints {
                if !constraints.properties.is_empty() {
                    writeln!(code, "///").expect("writing to String is infallible");
                    writeln!(code, "/// Properties:").expect("writing to String is infallible");
                    for prop in &constraints.properties {
                        writeln!(code, "/// - {:?}", prop)
                            .expect("writing to String is infallible");
                    }
                }
            }
        }

        if self.derive_common {
            writeln!(code, "#[derive(Clone, Debug, PartialEq, Eq, Hash)]")
                .expect("writing to String is infallible");
        }

        let type_name = Self::to_type_name(&predicate.name);

        // Generate struct with typed fields
        if predicate.arg_domains.is_empty() {
            // Nullary predicate
            writeln!(code, "pub struct {};", type_name).expect("writing to String is infallible");
        } else if predicate.arg_domains.len() == 1 {
            // Unary predicate
            let domain_type = Self::to_type_name(&predicate.arg_domains[0]);
            writeln!(code, "pub struct {}(pub {});", type_name, domain_type)
                .expect("writing to String is infallible");
        } else {
            // N-ary predicate - use tuple struct
            write!(code, "pub struct {}(", type_name).expect("writing to String is infallible");
            for (i, domain_name) in predicate.arg_domains.iter().enumerate() {
                if i > 0 {
                    write!(code, ", ").expect("writing to String is infallible");
                }
                write!(code, "pub {}", Self::to_type_name(domain_name))
                    .expect("writing to String is infallible");
            }
            writeln!(code, ");").expect("writing to String is infallible");
        }

        writeln!(code).expect("writing to String is infallible");

        // Generate constructor and accessors
        writeln!(code, "impl {} {{", type_name).expect("writing to String is infallible");

        if !predicate.arg_domains.is_empty() {
            // Constructor
            writeln!(code, "    /// Create a new {} instance.", type_name)
                .expect("writing to String is infallible");
            write!(code, "    pub fn new(").expect("writing to String is infallible");
            for (i, domain_name) in predicate.arg_domains.iter().enumerate() {
                if i > 0 {
                    write!(code, ", ").expect("writing to String is infallible");
                }
                write!(code, "arg{}: {}", i, Self::to_type_name(domain_name))
                    .expect("writing to String is infallible");
            }
            writeln!(code, ") -> Self {{").expect("writing to String is infallible");

            if predicate.arg_domains.len() == 1 {
                writeln!(code, "        Self(arg0)").expect("writing to String is infallible");
            } else {
                write!(code, "        Self(").expect("writing to String is infallible");
                for i in 0..predicate.arg_domains.len() {
                    if i > 0 {
                        write!(code, ", ").expect("writing to String is infallible");
                    }
                    write!(code, "arg{}", i).expect("writing to String is infallible");
                }
                writeln!(code, ")").expect("writing to String is infallible");
            }
            writeln!(code, "    }}").expect("writing to String is infallible");
            writeln!(code).expect("writing to String is infallible");

            // Accessor methods
            for (i, domain_name) in predicate.arg_domains.iter().enumerate() {
                writeln!(code, "    /// Get argument {}.", i)
                    .expect("writing to String is infallible");
                writeln!(
                    code,
                    "    pub fn arg{}(&self) -> {} {{",
                    i,
                    Self::to_type_name(domain_name)
                )
                .expect("writing to String is infallible");
                if predicate.arg_domains.len() == 1 {
                    writeln!(code, "        self.0").expect("writing to String is infallible");
                } else {
                    writeln!(code, "        self.{}", i).expect("writing to String is infallible");
                }
                writeln!(code, "    }}").expect("writing to String is infallible");
                writeln!(code).expect("writing to String is infallible");
            }
        }

        writeln!(code, "}}").expect("writing to String is infallible");
    }

    /// Generate schema metadata type.
    fn generate_schema_metadata(&self, code: &mut String, table: &SymbolTable) {
        writeln!(code, "/// Schema metadata and statistics.")
            .expect("writing to String is infallible");
        writeln!(code, "pub struct SchemaMetadata;").expect("writing to String is infallible");
        writeln!(code).expect("writing to String is infallible");

        writeln!(code, "impl SchemaMetadata {{").expect("writing to String is infallible");
        writeln!(code, "    /// Number of domains in the schema.")
            .expect("writing to String is infallible");
        writeln!(
            code,
            "    pub const DOMAIN_COUNT: usize = {};",
            table.domains.len()
        )
        .expect("writing to String is infallible");
        writeln!(code).expect("writing to String is infallible");

        writeln!(code, "    /// Number of predicates in the schema.")
            .expect("writing to String is infallible");
        writeln!(
            code,
            "    pub const PREDICATE_COUNT: usize = {};",
            table.predicates.len()
        )
        .expect("writing to String is infallible");
        writeln!(code).expect("writing to String is infallible");

        writeln!(code, "    /// Total cardinality across all domains.")
            .expect("writing to String is infallible");
        let total_card: usize = table.domains.values().map(|d| d.cardinality).sum();
        writeln!(
            code,
            "    pub const TOTAL_CARDINALITY: usize = {};",
            total_card
        )
        .expect("writing to String is infallible");

        writeln!(code, "}}").expect("writing to String is infallible");
    }

    /// Convert a domain/predicate name to a Rust type name (PascalCase).
    pub(super) fn to_type_name(name: &str) -> String {
        // Simple conversion: capitalize first letter of each word
        name.split('_')
            .map(|word| {
                let mut chars = word.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => first.to_uppercase().chain(chars).collect(),
                }
            })
            .collect()
    }
}

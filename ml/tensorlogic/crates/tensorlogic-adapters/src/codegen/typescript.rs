use std::fmt::Write as FmtWrite;

use crate::{DomainInfo, PredicateInfo, SymbolTable};

use super::rust::RustCodegen;

/// Code generator for TypeScript definitions from symbol tables.
///
/// This generator creates TypeScript interface and type definitions
/// from TensorLogic schemas, enabling type-safe TypeScript development.
pub struct TypeScriptCodegen {
    /// Module name
    module_name: String,
    /// Whether to export types
    export_types: bool,
    /// Whether to include JSDoc comments
    include_jsdoc: bool,
    /// Whether to generate validation functions
    generate_validators: bool,
}

impl TypeScriptCodegen {
    /// Create a new TypeScript code generator.
    pub fn new(module_name: impl Into<String>) -> Self {
        Self {
            module_name: module_name.into(),
            export_types: true,
            include_jsdoc: true,
            generate_validators: true,
        }
    }

    /// Set whether to export types.
    pub fn with_exports(mut self, enable: bool) -> Self {
        self.export_types = enable;
        self
    }

    /// Set whether to include JSDoc comments.
    pub fn with_jsdoc(mut self, enable: bool) -> Self {
        self.include_jsdoc = enable;
        self
    }

    /// Set whether to generate validator functions.
    pub fn with_validators(mut self, enable: bool) -> Self {
        self.generate_validators = enable;
        self
    }

    /// Generate complete TypeScript module from a symbol table.
    pub fn generate(&self, table: &SymbolTable) -> String {
        let mut code = String::new();

        // Module header
        writeln!(code, "/**").expect("writing to String is infallible");
        writeln!(code, " * Generated from TensorLogic schema")
            .expect("writing to String is infallible");
        writeln!(code, " * Module: {}", self.module_name).expect("writing to String is infallible");
        writeln!(code, " *").expect("writing to String is infallible");
        writeln!(code, " * This code was automatically generated.")
            .expect("writing to String is infallible");
        writeln!(code, " * DO NOT EDIT MANUALLY.").expect("writing to String is infallible");
        writeln!(code, " */").expect("writing to String is infallible");
        writeln!(code).expect("writing to String is infallible");

        // Generate domain types
        writeln!(code, "// ==========================================")
            .expect("writing to String is infallible");
        writeln!(code, "// Domain Types").expect("writing to String is infallible");
        writeln!(code, "// ==========================================")
            .expect("writing to String is infallible");
        writeln!(code).expect("writing to String is infallible");

        for domain in table.domains.values() {
            self.generate_domain_type(&mut code, domain);
            writeln!(code).expect("writing to String is infallible");
        }

        // Generate predicate types
        writeln!(code, "// ==========================================")
            .expect("writing to String is infallible");
        writeln!(code, "// Predicate Types").expect("writing to String is infallible");
        writeln!(code, "// ==========================================")
            .expect("writing to String is infallible");
        writeln!(code).expect("writing to String is infallible");

        for predicate in table.predicates.values() {
            self.generate_predicate_type(&mut code, predicate, table);
            writeln!(code).expect("writing to String is infallible");
        }

        // Generate validator functions if enabled
        if self.generate_validators {
            writeln!(code, "// ==========================================")
                .expect("writing to String is infallible");
            writeln!(code, "// Validator Functions").expect("writing to String is infallible");
            writeln!(code, "// ==========================================")
                .expect("writing to String is infallible");
            writeln!(code).expect("writing to String is infallible");

            for domain in table.domains.values() {
                self.generate_domain_validator(&mut code, domain);
                writeln!(code).expect("writing to String is infallible");
            }
        }

        // Generate schema metadata
        writeln!(code, "// ==========================================")
            .expect("writing to String is infallible");
        writeln!(code, "// Schema Metadata").expect("writing to String is infallible");
        writeln!(code, "// ==========================================")
            .expect("writing to String is infallible");
        writeln!(code).expect("writing to String is infallible");
        self.generate_schema_metadata(&mut code, table);

        code
    }

    /// Generate TypeScript interface for a domain.
    fn generate_domain_type(&self, code: &mut String, domain: &DomainInfo) {
        let type_name = Self::to_typescript_type_name(&domain.name);
        let export = if self.export_types { "export " } else { "" };

        if self.include_jsdoc {
            writeln!(code, "/**").expect("writing to String is infallible");
            if let Some(ref desc) = domain.description {
                writeln!(code, " * {}", desc).expect("writing to String is infallible");
            } else {
                writeln!(code, " * Domain: {}", domain.name)
                    .expect("writing to String is infallible");
            }
            writeln!(code, " *").expect("writing to String is infallible");
            writeln!(code, " * Cardinality: {}", domain.cardinality)
                .expect("writing to String is infallible");
            writeln!(code, " */").expect("writing to String is infallible");
        }

        writeln!(code, "{}interface {} {{", export, type_name)
            .expect("writing to String is infallible");
        writeln!(code, "  readonly id: number;").expect("writing to String is infallible");
        writeln!(code, "}}").expect("writing to String is infallible");
        writeln!(code).expect("writing to String is infallible");

        // Generate branded type for stronger typing
        writeln!(
            code,
            "{}type {}Id = number & {{ readonly __brand: '{}' }};",
            export, type_name, type_name
        )
        .expect("writing to String is infallible");
    }

    /// Generate TypeScript interface for a predicate.
    fn generate_predicate_type(
        &self,
        code: &mut String,
        predicate: &PredicateInfo,
        _table: &SymbolTable,
    ) {
        let type_name = Self::to_typescript_type_name(&predicate.name);
        let export = if self.export_types { "export " } else { "" };

        if self.include_jsdoc {
            writeln!(code, "/**").expect("writing to String is infallible");
            if let Some(ref desc) = predicate.description {
                writeln!(code, " * {}", desc).expect("writing to String is infallible");
            } else {
                writeln!(code, " * Predicate: {}", predicate.name)
                    .expect("writing to String is infallible");
            }
            writeln!(code, " *").expect("writing to String is infallible");
            writeln!(code, " * Arity: {}", predicate.arg_domains.len())
                .expect("writing to String is infallible");

            if let Some(ref constraints) = predicate.constraints {
                if !constraints.properties.is_empty() {
                    writeln!(code, " *").expect("writing to String is infallible");
                    writeln!(code, " * Properties:").expect("writing to String is infallible");
                    for prop in &constraints.properties {
                        writeln!(code, " * - {:?}", prop).expect("writing to String is infallible");
                    }
                }
            }
            writeln!(code, " */").expect("writing to String is infallible");
        }

        writeln!(code, "{}interface {} {{", export, type_name)
            .expect("writing to String is infallible");
        writeln!(code, "  readonly id: string;").expect("writing to String is infallible");

        // Add argument fields
        for (i, domain_name) in predicate.arg_domains.iter().enumerate() {
            let field_name = format!("arg{}", i);
            let field_type = format!("{}Id", Self::to_typescript_type_name(domain_name));
            writeln!(code, "  readonly {}: {};", field_name, field_type)
                .expect("writing to String is infallible");
        }

        writeln!(code, "}}").expect("writing to String is infallible");
    }

    /// Generate validator function for a domain.
    fn generate_domain_validator(&self, code: &mut String, domain: &DomainInfo) {
        let type_name = Self::to_typescript_type_name(&domain.name);
        let export = if self.export_types { "export " } else { "" };

        writeln!(code, "/**").expect("writing to String is infallible");
        writeln!(code, " * Validate {} ID", type_name).expect("writing to String is infallible");
        writeln!(
            code,
            " * @param id - The ID to validate (must be in range [0, {}))",
            domain.cardinality
        )
        .expect("writing to String is infallible");
        writeln!(code, " * @returns true if valid, false otherwise")
            .expect("writing to String is infallible");
        writeln!(code, " */").expect("writing to String is infallible");

        writeln!(
            code,
            "{}function is{}Id(id: number): id is {}Id {{",
            export, type_name, type_name
        )
        .expect("writing to String is infallible");
        writeln!(
            code,
            "  return Number.isInteger(id) && id >= 0 && id < {};",
            domain.cardinality
        )
        .expect("writing to String is infallible");
        writeln!(code, "}}").expect("writing to String is infallible");
    }

    /// Generate schema metadata constant.
    fn generate_schema_metadata(&self, code: &mut String, table: &SymbolTable) {
        let export = if self.export_types { "export " } else { "" };

        writeln!(code, "/**").expect("writing to String is infallible");
        writeln!(code, " * Schema metadata and statistics")
            .expect("writing to String is infallible");
        writeln!(code, " */").expect("writing to String is infallible");

        writeln!(code, "{}const SCHEMA_METADATA = {{", export)
            .expect("writing to String is infallible");
        writeln!(code, "  domainCount: {},", table.domains.len())
            .expect("writing to String is infallible");
        writeln!(code, "  predicateCount: {},", table.predicates.len())
            .expect("writing to String is infallible");

        let total_card: usize = table.domains.values().map(|d| d.cardinality).sum();
        writeln!(code, "  totalCardinality: {},", total_card)
            .expect("writing to String is infallible");

        writeln!(code, "  domains: {{").expect("writing to String is infallible");
        for domain in table.domains.values() {
            writeln!(
                code,
                "    '{}': {{ cardinality: {} }},",
                domain.name, domain.cardinality
            )
            .expect("writing to String is infallible");
        }
        writeln!(code, "  }},").expect("writing to String is infallible");

        writeln!(code, "  predicates: {{").expect("writing to String is infallible");
        for predicate in table.predicates.values() {
            writeln!(
                code,
                "    '{}': {{ arity: {} }},",
                predicate.name,
                predicate.arg_domains.len()
            )
            .expect("writing to String is infallible");
        }
        writeln!(code, "  }},").expect("writing to String is infallible");

        writeln!(code, "}} as const;").expect("writing to String is infallible");
    }

    /// Convert a name to TypeScript type name (PascalCase).
    fn to_typescript_type_name(name: &str) -> String {
        RustCodegen::to_type_name(name) // Reuse Rust converter
    }
}

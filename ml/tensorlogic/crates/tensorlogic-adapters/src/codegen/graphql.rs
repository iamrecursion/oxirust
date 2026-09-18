use std::fmt::Write as FmtWrite;

use crate::{DomainInfo, PredicateInfo, SymbolTable};

use super::rust::RustCodegen;

/// Code generator for GraphQL schemas from symbol tables.
///
/// This generator creates GraphQL type definitions, queries, and mutations
/// from TensorLogic schemas, enabling API development with type-safe schemas.
pub struct GraphQLCodegen {
    /// Schema name
    schema_name: String,
    /// Whether to include descriptions
    include_descriptions: bool,
    /// Whether to generate Query type
    generate_queries: bool,
    /// Whether to generate Mutation type
    generate_mutations: bool,
}

impl GraphQLCodegen {
    /// Create a new GraphQL code generator.
    pub fn new(schema_name: impl Into<String>) -> Self {
        Self {
            schema_name: schema_name.into(),
            include_descriptions: true,
            generate_queries: true,
            generate_mutations: false,
        }
    }

    /// Set whether to include descriptions.
    pub fn with_descriptions(mut self, enable: bool) -> Self {
        self.include_descriptions = enable;
        self
    }

    /// Set whether to generate Query type.
    pub fn with_queries(mut self, enable: bool) -> Self {
        self.generate_queries = enable;
        self
    }

    /// Set whether to generate Mutation type.
    pub fn with_mutations(mut self, enable: bool) -> Self {
        self.generate_mutations = enable;
        self
    }

    /// Generate complete GraphQL schema from a symbol table.
    pub fn generate(&self, table: &SymbolTable) -> String {
        let mut schema = String::new();

        // Schema header
        writeln!(schema, "# Generated GraphQL Schema").expect("writing to String is infallible");
        writeln!(schema, "# Schema: {}", self.schema_name)
            .expect("writing to String is infallible");
        writeln!(schema, "#").expect("writing to String is infallible");
        writeln!(
            schema,
            "# This schema was automatically generated from TensorLogic."
        )
        .expect("writing to String is infallible");
        writeln!(schema, "# DO NOT EDIT MANUALLY.").expect("writing to String is infallible");
        writeln!(schema).expect("writing to String is infallible");

        // Generate domain types
        writeln!(schema, "# ==========================================")
            .expect("writing to String is infallible");
        writeln!(schema, "# Domain Types").expect("writing to String is infallible");
        writeln!(schema, "# ==========================================")
            .expect("writing to String is infallible");
        writeln!(schema).expect("writing to String is infallible");

        for domain in table.domains.values() {
            self.generate_domain_type(&mut schema, domain);
            writeln!(schema).expect("writing to String is infallible");
        }

        // Generate predicate types
        writeln!(schema, "# ==========================================")
            .expect("writing to String is infallible");
        writeln!(schema, "# Predicate Types").expect("writing to String is infallible");
        writeln!(schema, "# ==========================================")
            .expect("writing to String is infallible");
        writeln!(schema).expect("writing to String is infallible");

        for predicate in table.predicates.values() {
            self.generate_predicate_type(&mut schema, predicate, table);
            writeln!(schema).expect("writing to String is infallible");
        }

        // Generate Query type
        if self.generate_queries {
            writeln!(schema, "# ==========================================")
                .expect("writing to String is infallible");
            writeln!(schema, "# Query Operations").expect("writing to String is infallible");
            writeln!(schema, "# ==========================================")
                .expect("writing to String is infallible");
            writeln!(schema).expect("writing to String is infallible");
            self.generate_query_type(&mut schema, table);
            writeln!(schema).expect("writing to String is infallible");
        }

        // Generate Mutation type
        if self.generate_mutations {
            writeln!(schema, "# ==========================================")
                .expect("writing to String is infallible");
            writeln!(schema, "# Mutation Operations").expect("writing to String is infallible");
            writeln!(schema, "# ==========================================")
                .expect("writing to String is infallible");
            writeln!(schema).expect("writing to String is infallible");
            self.generate_mutation_type(&mut schema, table);
            writeln!(schema).expect("writing to String is infallible");
        }

        // Schema definition
        writeln!(schema, "# ==========================================")
            .expect("writing to String is infallible");
        writeln!(schema, "# Schema Definition").expect("writing to String is infallible");
        writeln!(schema, "# ==========================================")
            .expect("writing to String is infallible");
        writeln!(schema).expect("writing to String is infallible");
        writeln!(schema, "schema {{").expect("writing to String is infallible");
        if self.generate_queries {
            writeln!(schema, "  query: Query").expect("writing to String is infallible");
        }
        if self.generate_mutations {
            writeln!(schema, "  mutation: Mutation").expect("writing to String is infallible");
        }
        writeln!(schema, "}}").expect("writing to String is infallible");

        schema
    }

    /// Generate GraphQL type for a domain.
    fn generate_domain_type(&self, schema: &mut String, domain: &DomainInfo) {
        let type_name = Self::to_graphql_type_name(&domain.name);

        if self.include_descriptions {
            if let Some(ref desc) = domain.description {
                writeln!(schema, "\"\"\"\n{}\n\"\"\"", desc)
                    .expect("writing to String is infallible");
            } else {
                writeln!(schema, "\"\"\"\nDomain: {}\n\"\"\"", domain.name)
                    .expect("writing to String is infallible");
            }
        }

        writeln!(schema, "type {} {{", type_name).expect("writing to String is infallible");
        writeln!(schema, "  \"Unique identifier\"").expect("writing to String is infallible");
        writeln!(schema, "  id: ID!").expect("writing to String is infallible");
        writeln!(
            schema,
            "  \"Integer index (0 to {})\"",
            domain.cardinality - 1
        )
        .expect("writing to String is infallible");
        writeln!(schema, "  index: Int!").expect("writing to String is infallible");
        writeln!(schema, "}}").expect("writing to String is infallible");
    }

    /// Generate GraphQL type for a predicate.
    fn generate_predicate_type(
        &self,
        schema: &mut String,
        predicate: &PredicateInfo,
        _table: &SymbolTable,
    ) {
        let type_name = Self::to_graphql_type_name(&predicate.name);

        if self.include_descriptions {
            if let Some(ref desc) = predicate.description {
                writeln!(schema, "\"\"\"\n{}\n\"\"\"", desc)
                    .expect("writing to String is infallible");
            } else {
                writeln!(schema, "\"\"\"\nPredicate: {}\n\"\"\"", predicate.name)
                    .expect("writing to String is infallible");
            }
        }

        writeln!(schema, "type {} {{", type_name).expect("writing to String is infallible");

        // Add ID field
        writeln!(schema, "  \"Unique identifier\"").expect("writing to String is infallible");
        writeln!(schema, "  id: ID!").expect("writing to String is infallible");

        // Add argument fields
        for (i, domain_name) in predicate.arg_domains.iter().enumerate() {
            let field_name = format!("arg{}", i);
            let field_type = Self::to_graphql_type_name(domain_name);
            writeln!(schema, "  \"Argument {} of type {}\"", i, domain_name)
                .expect("writing to String is infallible");
            writeln!(schema, "  {}: {}!", field_name, field_type)
                .expect("writing to String is infallible");
        }

        writeln!(schema, "}}").expect("writing to String is infallible");
    }

    /// Generate Query type.
    fn generate_query_type(&self, schema: &mut String, table: &SymbolTable) {
        writeln!(schema, "\"\"\"").expect("writing to String is infallible");
        writeln!(schema, "Root query type for retrieving data")
            .expect("writing to String is infallible");
        writeln!(schema, "\"\"\"").expect("writing to String is infallible");
        writeln!(schema, "type Query {{").expect("writing to String is infallible");

        // Domain queries
        for domain in table.domains.values() {
            let type_name = Self::to_graphql_type_name(&domain.name);
            let field_name = Self::to_graphql_field_name(&domain.name);

            writeln!(schema, "  \"Get {} by ID\"", domain.name)
                .expect("writing to String is infallible");
            writeln!(schema, "  {}(id: ID!): {}", field_name, type_name)
                .expect("writing to String is infallible");
            writeln!(schema).expect("writing to String is infallible");

            writeln!(schema, "  \"List all {}s\"", domain.name)
                .expect("writing to String is infallible");
            writeln!(schema, "  {}s: [{}!]!", field_name, type_name)
                .expect("writing to String is infallible");
            writeln!(schema).expect("writing to String is infallible");
        }

        // Predicate queries
        for predicate in table.predicates.values() {
            let type_name = Self::to_graphql_type_name(&predicate.name);
            let field_name = Self::to_graphql_field_name(&predicate.name);

            writeln!(schema, "  \"Query {} predicate\"", predicate.name)
                .expect("writing to String is infallible");

            // Build query with argument filters
            write!(schema, "  {}(", field_name).expect("writing to String is infallible");
            for (i, domain_name) in predicate.arg_domains.iter().enumerate() {
                if i > 0 {
                    write!(schema, ", ").expect("writing to String is infallible");
                }
                let arg_type = Self::to_graphql_type_name(domain_name);
                write!(schema, "arg{}: {}", i, arg_type).expect("writing to String is infallible");
            }
            writeln!(schema, "): [{}!]!", type_name).expect("writing to String is infallible");
            writeln!(schema).expect("writing to String is infallible");
        }

        writeln!(schema, "}}").expect("writing to String is infallible");
    }

    /// Generate Mutation type.
    fn generate_mutation_type(&self, schema: &mut String, table: &SymbolTable) {
        writeln!(schema, "\"\"\"").expect("writing to String is infallible");
        writeln!(schema, "Root mutation type for modifying data")
            .expect("writing to String is infallible");
        writeln!(schema, "\"\"\"").expect("writing to String is infallible");
        writeln!(schema, "type Mutation {{").expect("writing to String is infallible");

        // Predicate mutations (add/remove)
        for predicate in table.predicates.values() {
            let type_name = Self::to_graphql_type_name(&predicate.name);

            // Add mutation
            writeln!(schema, "  \"Add {} instance\"", predicate.name)
                .expect("writing to String is infallible");
            write!(schema, "  add{}(", type_name).expect("writing to String is infallible");
            for (i, domain_name) in predicate.arg_domains.iter().enumerate() {
                if i > 0 {
                    write!(schema, ", ").expect("writing to String is infallible");
                }
                let arg_type = Self::to_graphql_type_name(domain_name);
                write!(schema, "arg{}: {}!", i, arg_type).expect("writing to String is infallible");
            }
            writeln!(schema, "): {}!", type_name).expect("writing to String is infallible");
            writeln!(schema).expect("writing to String is infallible");

            // Remove mutation
            writeln!(schema, "  \"Remove {} instance\"", predicate.name)
                .expect("writing to String is infallible");
            writeln!(schema, "  remove{}(id: ID!): Boolean!", type_name)
                .expect("writing to String is infallible");
            writeln!(schema).expect("writing to String is infallible");
        }

        writeln!(schema, "}}").expect("writing to String is infallible");
    }

    /// Convert a name to GraphQL type name (PascalCase).
    fn to_graphql_type_name(name: &str) -> String {
        RustCodegen::to_type_name(name) // Reuse Rust converter
    }

    /// Convert a name to GraphQL field name (camelCase).
    pub(super) fn to_graphql_field_name(name: &str) -> String {
        let parts: Vec<&str> = name.split('_').collect();
        if parts.is_empty() {
            return String::new();
        }

        let mut result = parts[0].to_lowercase();
        for part in &parts[1..] {
            if let Some(first_char) = part.chars().next() {
                result.push_str(&first_char.to_uppercase().to_string());
                result.push_str(&part[first_char.len_utf8()..]);
            }
        }
        result
    }
}

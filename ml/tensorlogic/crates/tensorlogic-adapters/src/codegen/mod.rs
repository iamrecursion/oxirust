//! Code generation from schemas.
//!
//! This module provides utilities for generating code in various target languages
//! from TensorLogic schemas, enabling type-safe programming and API generation.
//!
//! Supported targets:
//! - Rust: Type definitions with bounds checking
//! - GraphQL: Schema definitions for API development
//! - TypeScript: Interface and type definitions
//! - Python: Type stubs and PyO3 bindings

mod graphql;
mod python;
mod rust;
mod typescript;

pub use graphql::GraphQLCodegen;
pub use python::PythonCodegen;
pub use rust::RustCodegen;
pub use typescript::TypeScriptCodegen;

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{DomainInfo, PredicateInfo, SymbolTable};

    #[test]
    fn test_to_type_name() {
        assert_eq!(RustCodegen::to_type_name("person"), "Person");
        assert_eq!(RustCodegen::to_type_name("Person"), "Person");
        assert_eq!(RustCodegen::to_type_name("student_record"), "StudentRecord");
        assert_eq!(RustCodegen::to_type_name("HTTP_Request"), "HTTPRequest");
    }

    #[test]
    fn test_generate_simple_schema() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100).with_description("A person entity"))
            .expect("unwrap");

        let codegen = RustCodegen::new("test_module");
        let code = codegen.generate(&table);

        assert!(code.contains("pub struct Person(pub usize);"));
        assert!(code.contains("CARDINALITY: usize = 100"));
        assert!(code.contains("A person entity"));
    }

    #[test]
    fn test_generate_predicate() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100))
            .expect("unwrap");

        let pred = PredicateInfo::new("knows", vec!["Person".to_string(), "Person".to_string()])
            .with_description("Person knows another person");
        table.add_predicate(pred).expect("unwrap");

        let codegen = RustCodegen::new("test_module");
        let code = codegen.generate(&table);

        assert!(code.contains("pub struct Knows(pub Person, pub Person);"));
        assert!(code.contains("Person knows another person"));
        assert!(code.contains("pub fn new(arg0: Person, arg1: Person)"));
    }

    #[test]
    fn test_generate_without_docs() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100).with_description("A person"))
            .expect("unwrap");

        let codegen = RustCodegen::new("test_module").with_docs(false);
        let code = codegen.generate(&table);

        // Should not contain descriptive doc comments (module header is ok)
        assert!(!code.contains("/// A person"));
        assert!(!code.contains("/// Cardinality:"));
        // Should still contain the struct
        assert!(code.contains("pub struct Person"));
    }

    #[test]
    fn test_generate_without_derives() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100))
            .expect("unwrap");

        let codegen = RustCodegen::new("test_module").with_common_derives(false);
        let code = codegen.generate(&table);

        // Should not contain derive attributes
        assert!(!code.contains("#[derive("));
        // Should still contain the struct
        assert!(code.contains("pub struct Person"));
    }

    #[test]
    fn test_generate_unary_predicate() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100))
            .expect("unwrap");

        let pred = PredicateInfo::new("adult", vec!["Person".to_string()]);
        table.add_predicate(pred).expect("unwrap");

        let codegen = RustCodegen::new("test_module");
        let code = codegen.generate(&table);

        assert!(code.contains("pub struct Adult(pub Person);"));
        assert!(code.contains("pub fn new(arg0: Person)"));
    }

    #[test]
    fn test_generate_metadata() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100))
            .expect("unwrap");
        table
            .add_domain(DomainInfo::new("Course", 50))
            .expect("unwrap");

        let pred = PredicateInfo::new("enrolled", vec!["Person".to_string(), "Course".to_string()]);
        table.add_predicate(pred).expect("unwrap");

        let codegen = RustCodegen::new("test_module");
        let code = codegen.generate(&table);

        assert!(code.contains("DOMAIN_COUNT: usize = 2"));
        assert!(code.contains("PREDICATE_COUNT: usize = 1"));
        assert!(code.contains("TOTAL_CARDINALITY: usize = 150"));
    }

    // GraphQL code generation tests
    #[test]
    fn test_graphql_field_name_conversion() {
        assert_eq!(GraphQLCodegen::to_graphql_field_name("person"), "person");
        assert_eq!(
            GraphQLCodegen::to_graphql_field_name("student_record"),
            "studentRecord"
        );
        assert_eq!(
            GraphQLCodegen::to_graphql_field_name("http_request"),
            "httpRequest"
        );
    }

    #[test]
    fn test_graphql_generate_simple_schema() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100).with_description("A person entity"))
            .expect("unwrap");

        let codegen = GraphQLCodegen::new("TestSchema");
        let schema = codegen.generate(&table);

        assert!(schema.contains("# Generated GraphQL Schema"));
        assert!(schema.contains("type Person {"));
        assert!(schema.contains("id: ID!"));
        assert!(schema.contains("index: Int!"));
        assert!(schema.contains("A person entity"));
    }

    #[test]
    fn test_graphql_generate_with_predicate() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100))
            .expect("unwrap");

        let pred = PredicateInfo::new("knows", vec!["Person".to_string(), "Person".to_string()])
            .with_description("Person knows another person");
        table.add_predicate(pred).expect("unwrap");

        let codegen = GraphQLCodegen::new("TestSchema");
        let schema = codegen.generate(&table);

        assert!(schema.contains("type Knows {"));
        assert!(schema.contains("arg0: Person!"));
        assert!(schema.contains("arg1: Person!"));
        assert!(schema.contains("Person knows another person"));
    }

    #[test]
    fn test_graphql_generate_queries() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100))
            .expect("unwrap");

        let pred = PredicateInfo::new("adult", vec!["Person".to_string()]);
        table.add_predicate(pred).expect("unwrap");

        let codegen = GraphQLCodegen::new("TestSchema").with_queries(true);
        let schema = codegen.generate(&table);

        assert!(schema.contains("type Query {"));
        assert!(schema.contains("person(id: ID!): Person"));
        assert!(schema.contains("persons: [Person!]!"));
        assert!(schema.contains("adult(arg0: Person): [Adult!]!"));
    }

    #[test]
    fn test_graphql_generate_mutations() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100))
            .expect("unwrap");

        let pred = PredicateInfo::new("adult", vec!["Person".to_string()]);
        table.add_predicate(pred).expect("unwrap");

        let codegen = GraphQLCodegen::new("TestSchema")
            .with_queries(false)
            .with_mutations(true);
        let schema = codegen.generate(&table);

        assert!(schema.contains("type Mutation {"));
        assert!(schema.contains("addAdult(arg0: Person!): Adult!"));
        assert!(schema.contains("removeAdult(id: ID!): Boolean!"));
        assert!(!schema.contains("type Query"));
    }

    #[test]
    fn test_graphql_without_descriptions() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100).with_description("A person"))
            .expect("unwrap");

        let codegen = GraphQLCodegen::new("TestSchema").with_descriptions(false);
        let schema = codegen.generate(&table);

        assert!(!schema.contains("A person"));
        assert!(schema.contains("type Person {"));
    }

    #[test]
    fn test_graphql_schema_definition() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100))
            .expect("unwrap");

        let codegen = GraphQLCodegen::new("TestSchema")
            .with_queries(true)
            .with_mutations(true);
        let schema = codegen.generate(&table);

        assert!(schema.contains("schema {"));
        assert!(schema.contains("query: Query"));
        assert!(schema.contains("mutation: Mutation"));
    }

    #[test]
    fn test_graphql_complex_predicate() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Student", 80))
            .expect("unwrap");
        table
            .add_domain(DomainInfo::new("Course", 50))
            .expect("unwrap");
        table
            .add_domain(DomainInfo::new("Grade", 5))
            .expect("unwrap");

        let pred = PredicateInfo::new(
            "grade",
            vec![
                "Student".to_string(),
                "Course".to_string(),
                "Grade".to_string(),
            ],
        );
        table.add_predicate(pred).expect("unwrap");

        let codegen = GraphQLCodegen::new("TestSchema");
        let schema = codegen.generate(&table);

        assert!(schema.contains("type Grade {"));
        assert!(schema.contains("arg0: Student!"));
        assert!(schema.contains("arg1: Course!"));
        assert!(schema.contains("arg2: Grade!"));
    }

    // TypeScript code generation tests
    #[test]
    fn test_typescript_generate_simple_schema() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100).with_description("A person entity"))
            .expect("unwrap");

        let codegen = TypeScriptCodegen::new("test_module");
        let code = codegen.generate(&table);

        assert!(code.contains("export interface Person {"));
        assert!(code.contains("readonly id: number;"));
        assert!(code.contains("export type PersonId = number"));
        assert!(code.contains("A person entity"));
        assert!(code.contains("Cardinality: 100"));
    }

    #[test]
    fn test_typescript_generate_with_predicate() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100))
            .expect("unwrap");

        let pred = PredicateInfo::new("knows", vec!["Person".to_string(), "Person".to_string()])
            .with_description("Person knows another person");
        table.add_predicate(pred).expect("unwrap");

        let codegen = TypeScriptCodegen::new("test_module");
        let code = codegen.generate(&table);

        assert!(code.contains("export interface Knows {"));
        assert!(code.contains("readonly arg0: PersonId;"));
        assert!(code.contains("readonly arg1: PersonId;"));
        assert!(code.contains("Person knows another person"));
    }

    #[test]
    fn test_typescript_generate_validators() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100))
            .expect("unwrap");

        let codegen = TypeScriptCodegen::new("test_module").with_validators(true);
        let code = codegen.generate(&table);

        assert!(code.contains("export function isPersonId(id: number): id is PersonId {"));
        assert!(code.contains("Number.isInteger(id) && id >= 0 && id < 100"));
    }

    #[test]
    fn test_typescript_without_exports() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100))
            .expect("unwrap");

        let codegen = TypeScriptCodegen::new("test_module").with_exports(false);
        let code = codegen.generate(&table);

        // Should not have export keywords
        let export_count = code.matches("export ").count();
        assert_eq!(export_count, 0);
        assert!(code.contains("interface Person {"));
    }

    #[test]
    fn test_typescript_without_jsdoc() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100).with_description("A person"))
            .expect("unwrap");

        let codegen = TypeScriptCodegen::new("test_module").with_jsdoc(false);
        let code = codegen.generate(&table);

        assert!(!code.contains("A person"));
        assert!(code.contains("export interface Person {"));
    }

    #[test]
    fn test_typescript_metadata_generation() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100))
            .expect("unwrap");
        table
            .add_domain(DomainInfo::new("Course", 50))
            .expect("unwrap");

        let pred = PredicateInfo::new("enrolled", vec!["Person".to_string(), "Course".to_string()]);
        table.add_predicate(pred).expect("unwrap");

        let codegen = TypeScriptCodegen::new("test_module");
        let code = codegen.generate(&table);

        assert!(code.contains("export const SCHEMA_METADATA = {"));
        assert!(code.contains("domainCount: 2,"));
        assert!(code.contains("predicateCount: 1,"));
        assert!(code.contains("totalCardinality: 150,"));
    }

    // Python code generation tests
    #[test]
    fn test_python_generate_simple_stubs() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100).with_description("A person entity"))
            .expect("unwrap");

        let codegen = PythonCodegen::new("test_module");
        let code = codegen.generate(&table);

        assert!(code.contains("Person = NewType('Person', int)"));
        assert!(code.contains("PERSON_CARDINALITY: Final[int] = 100"));
        assert!(code.contains("def is_valid_Person(id: int) -> bool:"));
        assert!(code.contains("A person entity"));
    }

    #[test]
    fn test_python_generate_with_predicate() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100))
            .expect("unwrap");

        let pred = PredicateInfo::new("knows", vec!["Person".to_string(), "Person".to_string()])
            .with_description("Person knows another person");
        table.add_predicate(pred).expect("unwrap");

        let codegen = PythonCodegen::new("test_module");
        let code = codegen.generate(&table);

        assert!(code.contains("@dataclass(frozen=True)"));
        assert!(code.contains("class Knows:"));
        assert!(code.contains("Person knows another person"));
        assert!(code.contains("arg0: Person"));
        assert!(code.contains("arg1: Person"));
    }

    #[test]
    fn test_python_generate_pyo3_bindings() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100))
            .expect("unwrap");

        let codegen = PythonCodegen::new("test_module").with_pyo3(true);
        let code = codegen.generate(&table);

        assert!(code.contains("use pyo3::prelude::*;"));
        assert!(code.contains("#[pyclass]"));
        assert!(code.contains("pub struct Person {"));
        assert!(code.contains("#[pyo3(get)]"));
        assert!(code.contains("pub id: usize,"));
        assert!(code.contains("#[pymethods]"));
        assert!(code.contains("#[new]"));
        assert!(code.contains("fn __repr__(&self)"));
    }

    #[test]
    fn test_python_without_docs() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100).with_description("A person"))
            .expect("unwrap");

        let codegen = PythonCodegen::new("test_module").with_docs(false);
        let code = codegen.generate(&table);

        // Should not contain docstrings (except the module header)
        let docstring_count = code.matches("\"\"\"").count();
        assert_eq!(docstring_count, 2); // Only module header
    }

    #[test]
    fn test_python_without_dataclasses() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100))
            .expect("unwrap");

        let pred = PredicateInfo::new("adult", vec!["Person".to_string()]);
        table.add_predicate(pred).expect("unwrap");

        let codegen = PythonCodegen::new("test_module").with_dataclasses(false);
        let code = codegen.generate(&table);

        assert!(!code.contains("@dataclass"));
        assert!(code.contains("class Adult:"));
    }

    #[test]
    fn test_python_pyo3_module_registration() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100))
            .expect("unwrap");
        table
            .add_domain(DomainInfo::new("Course", 50))
            .expect("unwrap");

        let pred = PredicateInfo::new("enrolled", vec!["Person".to_string(), "Course".to_string()]);
        table.add_predicate(pred).expect("unwrap");

        let codegen = PythonCodegen::new("test_module").with_pyo3(true);
        let code = codegen.generate(&table);

        assert!(code.contains("#[pymodule]"));
        assert!(code.contains("fn test_module(_py: Python, m: &PyModule)"));
        assert!(code.contains("m.add_class::<Person>()?;"));
        assert!(code.contains("m.add_class::<Course>()?;"));
        assert!(code.contains("m.add_class::<Enrolled>()?;"));
    }

    #[test]
    fn test_python_metadata_generation() {
        let mut table = SymbolTable::new();
        table
            .add_domain(DomainInfo::new("Person", 100))
            .expect("unwrap");
        table
            .add_domain(DomainInfo::new("Course", 50))
            .expect("unwrap");

        let codegen = PythonCodegen::new("test_module");
        let code = codegen.generate(&table);

        assert!(code.contains("class SchemaMetadata:"));
        assert!(code.contains("DOMAIN_COUNT: Final[int] = 2"));
        assert!(code.contains("TOTAL_CARDINALITY: Final[int] = 150"));
    }
}

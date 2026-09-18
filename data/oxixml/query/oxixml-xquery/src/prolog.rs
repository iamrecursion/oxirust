//! The syntax tree of a module and its prolog.
//!
//! [XQuery 3.1 §4](https://www.w3.org/TR/xquery-31/#id-query-prolog) splits a module into a
//! *prolog*, which sets the static context, and either a *query body* (a main module) or nothing
//! at all (a library module). Every declaration keeps the source text of the types it mentions,
//! because a type is only ever needed as an XPath `instance of` or `treat as` operand.

use crate::ast::Syntax;
use crate::context::{BoundarySpace, ConstructionMode, CopyNamespaces, EmptyOrder, OrderingMode};

/// A parsed main or library module.
#[derive(Debug, Clone)]
pub(crate) struct Module {
    /// `(prefix, target namespace)` for a library module; `None` for a main module.
    pub(crate) module_namespace: Option<(String, String)>,
    pub(crate) prolog: Prolog,
    /// The query body; a library module has none.
    pub(crate) body: Option<Syntax>,
    /// What the parser noticed on the way through: the ranges to blank, every variable
    /// reference and binding, and the annotation names used inside sequence types.
    pub(crate) seen: crate::parser::Observations,
}

/// Everything a prolog can declare.
#[derive(Debug, Clone, Default)]
pub(crate) struct Prolog {
    pub(crate) version: Option<String>,
    pub(crate) encoding: Option<String>,
    pub(crate) namespaces: Vec<(String, String)>,
    pub(crate) default_element_namespace: Option<String>,
    pub(crate) default_function_namespace: Option<String>,
    pub(crate) boundary_space: Option<BoundarySpace>,
    pub(crate) default_collation: Option<String>,
    pub(crate) base_uri: Option<String>,
    pub(crate) construction: Option<ConstructionMode>,
    pub(crate) ordering: Option<OrderingMode>,
    pub(crate) empty_order: Option<EmptyOrder>,
    pub(crate) copy_namespaces: Option<CopyNamespaces>,
    pub(crate) decimal_formats: Vec<DecimalFormatDeclaration>,
    pub(crate) schema_imports: Vec<SchemaImport>,
    pub(crate) module_imports: Vec<ModuleImport>,
    pub(crate) context_item: Option<ContextItemDeclaration>,
    pub(crate) variables: Vec<VariableDeclaration>,
    pub(crate) functions: Vec<FunctionDeclaration>,
    pub(crate) options: Vec<(String, String)>,
}

/// `declare decimal-format` or `declare default decimal-format`.
#[derive(Debug, Clone)]
pub(crate) struct DecimalFormatDeclaration {
    /// `None` for `declare default decimal-format`.
    pub(crate) name: Option<String>,
    pub(crate) properties: Vec<(String, String)>,
}

/// `import schema …`, which this processor can record but not honour.
#[derive(Debug, Clone)]
pub(crate) struct SchemaImport {
    pub(crate) prefix: Option<String>,
    /// `true` for `import schema default element namespace "…"`.
    pub(crate) default_element_namespace: bool,
    pub(crate) target_namespace: String,
}

/// `import module namespace p = "…" at "…"`.
#[derive(Debug, Clone)]
pub(crate) struct ModuleImport {
    pub(crate) prefix: Option<String>,
    pub(crate) target_namespace: String,
    pub(crate) locations: Vec<String>,
}

/// `declare context item as T := E` or `… external`.
#[derive(Debug, Clone)]
pub(crate) struct ContextItemDeclaration {
    pub(crate) declared_type: Option<String>,
    pub(crate) value: Option<Syntax>,
    pub(crate) external: bool,
}

/// `declare variable $x as T := E` or `… external`.
#[derive(Debug, Clone)]
pub(crate) struct VariableDeclaration {
    pub(crate) name: String,
    pub(crate) declared_type: Option<String>,
    pub(crate) value: Option<Syntax>,
    pub(crate) external: bool,
    pub(crate) annotations: Vec<String>,
}

/// `declare function f($x as T) as R { … }` or `… external`.
#[derive(Debug, Clone)]
pub(crate) struct FunctionDeclaration {
    pub(crate) name: String,
    pub(crate) parameters: Vec<(String, Option<String>)>,
    pub(crate) result_type: Option<String>,
    pub(crate) body: Option<Syntax>,
    pub(crate) external: bool,
    pub(crate) annotations: Vec<String>,
}

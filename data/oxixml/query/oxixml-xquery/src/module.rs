//! Module resolution: the callback `import module` consults, and an in-memory implementation.
//!
//! [XQuery 3.1 §4.11](https://www.w3.org/TR/xquery-31/#id-module-import) leaves the mapping from
//! a target namespace and a list of location hints to actual module source entirely to the
//! implementation. A Pure-Rust library has no business reading the filesystem or the network
//! behind its caller's back, so nothing is importable until an application installs a
//! [`ModuleResolver`] with
//! [`StaticQueryContext::with_module_resolver`](crate::StaticQueryContext::with_module_resolver);
//! an `import module` with no resolver raises the static error `XQST0059`.
//!
//! # Examples
//!
//! ```
//! use oxixml_xquery::{InMemoryModules, StaticQueryContext, XQuery};
//! use std::sync::Arc;
//!
//! let library = InMemoryModules::new().with_module(
//!     "urn:example:math",
//!     r#"module namespace m = "urn:example:math";
//!        declare function m:double($n as xs:integer) as xs:integer { 2 * $n };"#,
//! );
//!
//! let compiled = XQuery::compile(
//!     r#"import module namespace m = "urn:example:math"; m:double(21)"#,
//!     StaticQueryContext::new().with_module_resolver(Arc::new(library)),
//! )?;
//! assert_eq!(compiled.execute_to_string(&Default::default())?, "42");
//! # Ok::<(), oxixml_xquery::XQueryError>(())
//! ```

use std::collections::BTreeMap;
use std::fmt;

/// Supplies the source text of the library modules an `import module` names.
pub trait ModuleResolver: fmt::Debug + Send + Sync {
    /// The source of every library module that contributes to `target_namespace`.
    ///
    /// `locations` holds the `at` hints the import gave, which may be empty. Returning `None`
    /// — or an empty vector — makes the import raise `XQST0059`.
    fn resolve(&self, target_namespace: &str, locations: &[String]) -> Option<Vec<String>>;
}

/// A [`ModuleResolver`] backed by a map from target namespace to module source.
///
/// This is what a test, an embedded application, or a caller that has already read its modules
/// from wherever they live wants: no I/O, and complete control over what is importable.
#[derive(Debug, Clone, Default)]
pub struct InMemoryModules {
    modules: BTreeMap<String, Vec<String>>,
}

impl InMemoryModules {
    /// An empty library.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one module's source under `target_namespace`.
    ///
    /// Several modules may contribute to the same namespace; they are returned in insertion
    /// order.
    #[must_use]
    pub fn with_module(
        mut self,
        target_namespace: impl Into<String>,
        source: impl Into<String>,
    ) -> Self {
        self.modules
            .entry(target_namespace.into())
            .or_default()
            .push(source.into());
        self
    }

    /// Adds one module's source in place.
    pub fn insert(&mut self, target_namespace: impl Into<String>, source: impl Into<String>) {
        self.modules
            .entry(target_namespace.into())
            .or_default()
            .push(source.into());
    }

    /// Whether any module is registered under `target_namespace`.
    #[must_use]
    pub fn contains(&self, target_namespace: &str) -> bool {
        self.modules.contains_key(target_namespace)
    }

    /// Every registered target namespace.
    pub fn namespaces(&self) -> impl Iterator<Item = &str> {
        self.modules.keys().map(String::as_str)
    }
}

impl ModuleResolver for InMemoryModules {
    fn resolve(&self, target_namespace: &str, _locations: &[String]) -> Option<Vec<String>> {
        self.modules.get(target_namespace).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_in_memory_resolver_returns_what_was_registered() {
        let library = InMemoryModules::new()
            .with_module("urn:a", "module namespace a = \"urn:a\";")
            .with_module("urn:a", "module namespace a = \"urn:a\";")
            .with_module("urn:b", "module namespace b = \"urn:b\";");
        assert!(library.contains("urn:a"));
        assert_eq!(library.namespaces().count(), 2);
        assert_eq!(library.resolve("urn:a", &[]).map(|s| s.len()), Some(2));
        assert!(library.resolve("urn:missing", &[]).is_none());
    }

    #[test]
    fn insert_matches_the_builder() {
        let mut library = InMemoryModules::new();
        library.insert("urn:a", "module namespace a = \"urn:a\";");
        assert!(library.contains("urn:a"));
    }
}

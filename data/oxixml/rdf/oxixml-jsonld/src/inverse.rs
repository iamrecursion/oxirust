//! The [inverse context](https://www.w3.org/TR/json-ld11-api/#inverse-context-creation) and the
//! [Term Selection](https://www.w3.org/TR/json-ld11-api/#term-selection) algorithm.
//!
//! An inverse context is a reverse lookup table that maps a container mapping plus a type or
//! language mapping back to the shortest term of an active context. Compaction consults it to pick
//! the term that best expresses a given IRI/value pair.

use crate::context::Context;
use std::collections::BTreeMap;

/// The special key standing for "no mapping at all".
pub const NONE: &str = "@none";
/// The key under which an explicitly null language mapping is stored.
pub const NULL: &str = "@null";
/// The key standing for "any mapping is acceptable".
pub const ANY: &str = "@any";

/// The term tables of one container mapping.
#[derive(Debug, Default)]
pub struct TypeLanguageMap {
    /// Terms keyed by language mapping (or `lang_dir`, `@null`, `@none`, `@any`).
    pub language: BTreeMap<String, String>,
    /// Terms keyed by type mapping (or `@reverse`, `@none`, `@any`).
    pub type_map: BTreeMap<String, String>,
    /// Terms keyed by `@none` only — the "any mapping goes" fallback.
    pub any: BTreeMap<String, String>,
}

impl TypeLanguageMap {
    /// The table addressed by a `type/language` selector (`@type`, `@language`, or `@any`).
    #[must_use]
    pub fn table(&self, type_language: &str) -> &BTreeMap<String, String> {
        match type_language {
            "@type" => &self.type_map,
            "@any" => &self.any,
            _ => &self.language,
        }
    }
}

/// An inverse context: IRI → container → [`TypeLanguageMap`].
#[derive(Debug, Default)]
pub struct InverseContext {
    entries: BTreeMap<String, BTreeMap<String, TypeLanguageMap>>,
}

impl InverseContext {
    /// Builds the inverse context of `active_context`
    /// ([Inverse Context Creation](https://www.w3.org/TR/json-ld11-api/#inverse-context-creation)).
    #[must_use]
    pub fn build(active_context: &Context) -> Self {
        // 2)
        let default_language = active_context
            .default_language
            .as_ref()
            .map_or_else(|| NONE.to_owned(), |language| language.to_lowercase());
        let mut entries: BTreeMap<String, BTreeMap<String, TypeLanguageMap>> = BTreeMap::new();
        // 3) shortest term first, ties broken lexicographically
        let mut terms: Vec<(&String, &crate::context::TermDefinition)> =
            active_context.term_definitions.iter().collect();
        terms.sort_by(|a, b| a.0.len().cmp(&b.0.len()).then_with(|| a.0.cmp(b.0)));

        for (term, definition) in terms {
            // 3.1)
            let Some(var) = definition.iri() else {
                continue;
            };
            // 3.2)
            let container = if definition.container_mapping.is_empty() {
                NONE.to_owned()
            } else {
                let mut sorted = definition.container_mapping.clone();
                sorted.sort_unstable();
                sorted.concat()
            };
            // 3.3-3.7)
            let container_map = entries.entry(var.to_owned()).or_default();
            let type_language = container_map.entry(container).or_insert_with(|| {
                let mut fresh = TypeLanguageMap::default();
                fresh.any.insert(NONE.to_owned(), term.clone());
                fresh
            });
            if definition.reverse_property {
                // 3.10)
                type_language
                    .type_map
                    .entry("@reverse".to_owned())
                    .or_insert_with(|| term.clone());
            } else if definition.type_mapping.as_deref() == Some(NONE) {
                // 3.11)
                type_language
                    .language
                    .entry(ANY.to_owned())
                    .or_insert_with(|| term.clone());
                type_language
                    .type_map
                    .entry(ANY.to_owned())
                    .or_insert_with(|| term.clone());
            } else if let Some(type_mapping) = &definition.type_mapping {
                // 3.12)
                type_language
                    .type_map
                    .entry(type_mapping.clone())
                    .or_insert_with(|| term.clone());
            } else if definition.language_mapping.is_some()
                && definition.direction_mapping.is_some()
            {
                // 3.13)
                let language = definition.language_mapping.clone().flatten();
                let direction = definition.direction_mapping.flatten();
                let lang_dir = match (language, direction) {
                    (Some(language), Some(direction)) => {
                        format!("{language}_{}", direction.as_str()).to_lowercase()
                    }
                    (Some(language), None) => language.to_lowercase(),
                    (None, Some(direction)) => format!("_{}", direction.as_str()),
                    (None, None) => NULL.to_owned(),
                };
                type_language
                    .language
                    .entry(lang_dir)
                    .or_insert_with(|| term.clone());
            } else if let Some(language_mapping) = &definition.language_mapping {
                // 3.14)
                let language = language_mapping
                    .as_ref()
                    .map_or_else(|| NULL.to_owned(), |language| language.to_lowercase());
                type_language
                    .language
                    .entry(language)
                    .or_insert_with(|| term.clone());
            } else if let Some(direction_mapping) = &definition.direction_mapping {
                // 3.15)
                let direction = direction_mapping
                    .map_or_else(|| NONE.to_owned(), |dir| format!("_{}", dir.as_str()));
                type_language
                    .language
                    .entry(direction)
                    .or_insert_with(|| term.clone());
            } else if let Some(default_direction) = active_context.default_direction {
                // 3.16)
                let lang_dir =
                    format!("{default_language}_{}", default_direction.as_str()).to_lowercase();
                type_language
                    .language
                    .entry(lang_dir)
                    .or_insert_with(|| term.clone());
                type_language
                    .language
                    .entry(NONE.to_owned())
                    .or_insert_with(|| term.clone());
                type_language
                    .type_map
                    .entry(NONE.to_owned())
                    .or_insert_with(|| term.clone());
            } else {
                // 3.17)
                type_language
                    .language
                    .entry(default_language.clone())
                    .or_insert_with(|| term.clone());
                type_language
                    .language
                    .entry(NONE.to_owned())
                    .or_insert_with(|| term.clone());
                type_language
                    .type_map
                    .entry(NONE.to_owned())
                    .or_insert_with(|| term.clone());
            }
        }
        Self { entries }
    }

    /// Whether the inverse context has any term for `var`.
    #[must_use]
    pub fn contains(&self, var: &str) -> bool {
        self.entries.contains_key(var)
    }

    /// The [Term Selection](https://www.w3.org/TR/json-ld11-api/#term-selection) algorithm.
    #[must_use]
    pub fn select_term(
        &self,
        var: &str,
        containers: &[String],
        type_language: &str,
        preferred_values: &[String],
    ) -> Option<String> {
        let container_map = self.entries.get(var)?;
        for container in containers {
            let Some(type_language_map) = container_map.get(container) else {
                continue;
            };
            let value_map = type_language_map.table(type_language);
            for item in preferred_values {
                if let Some(term) = value_map.get(item) {
                    return Some(term.clone());
                }
            }
        }
        None
    }
}

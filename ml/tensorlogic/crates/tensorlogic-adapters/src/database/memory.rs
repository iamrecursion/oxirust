//! In-memory database implementation for testing and development.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::{SchemaDatabase, SchemaId, SchemaMetadata, SchemaVersion};
use crate::{AdapterError, SymbolTable};

/// In-memory database implementation for testing and development.
///
/// This provides a simple in-memory store that implements the SchemaDatabase trait
/// without requiring external database dependencies. Useful for:
/// - Testing
/// - Development
/// - Small-scale applications
/// - Temporary storage
pub struct MemoryDatabase {
    schemas: HashMap<SchemaId, StoredSchema>,
    next_id: u64,
    name_index: HashMap<String, Vec<SchemaId>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct StoredSchema {
    id: SchemaId,
    name: String,
    version: u32,
    table: SymbolTable,
    created_at: u64,
    updated_at: u64,
    description: Option<String>,
}

impl MemoryDatabase {
    /// Create a new empty memory database.
    pub fn new() -> Self {
        Self {
            schemas: HashMap::new(),
            next_id: 1,
            name_index: HashMap::new(),
        }
    }

    /// Get current timestamp (Unix epoch seconds).
    fn current_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("SystemTime after UNIX_EPOCH")
            .as_secs()
    }

    /// Find latest version for a schema name.
    fn find_latest_version(&self, name: &str) -> Option<SchemaId> {
        self.name_index.get(name).and_then(|ids| {
            ids.iter()
                .filter_map(|id| self.schemas.get(id))
                .max_by_key(|s| s.version)
                .map(|s| s.id)
        })
    }
}

impl Default for MemoryDatabase {
    fn default() -> Self {
        Self::new()
    }
}

impl SchemaDatabase for MemoryDatabase {
    fn store_schema(&mut self, name: &str, table: &SymbolTable) -> Result<SchemaId, AdapterError> {
        let now = Self::current_timestamp();

        // Check if schema with this name exists
        let version = if let Some(existing_id) = self.find_latest_version(name) {
            if let Some(existing) = self.schemas.get(&existing_id) {
                existing.version + 1
            } else {
                1
            }
        } else {
            1
        };

        let id = SchemaId(self.next_id);
        self.next_id += 1;

        let stored = StoredSchema {
            id,
            name: name.to_string(),
            version,
            table: table.clone(),
            created_at: now,
            updated_at: now,
            description: None,
        };

        self.schemas.insert(id, stored);

        // Update name index
        self.name_index
            .entry(name.to_string())
            .or_default()
            .push(id);

        Ok(id)
    }

    fn load_schema(&self, id: SchemaId) -> Result<SymbolTable, AdapterError> {
        self.schemas
            .get(&id)
            .map(|s| s.table.clone())
            .ok_or_else(|| {
                AdapterError::InvalidOperation(format!("Schema with ID {:?} not found", id))
            })
    }

    fn load_schema_by_name(&self, name: &str) -> Result<SymbolTable, AdapterError> {
        let id = self.find_latest_version(name).ok_or_else(|| {
            AdapterError::InvalidOperation(format!("Schema '{}' not found", name))
        })?;

        self.load_schema(id)
    }

    fn list_schemas(&self) -> Result<Vec<SchemaMetadata>, AdapterError> {
        let mut metadata: Vec<SchemaMetadata> = self
            .schemas
            .values()
            .map(|s| SchemaMetadata {
                id: s.id,
                name: s.name.clone(),
                version: s.version,
                created_at: s.created_at,
                updated_at: s.updated_at,
                num_domains: s.table.domains.len(),
                num_predicates: s.table.predicates.len(),
                num_variables: s.table.variables.len(),
                description: s.description.clone(),
            })
            .collect();

        metadata.sort_by_key(|m| m.name.clone());
        Ok(metadata)
    }

    fn delete_schema(&mut self, id: SchemaId) -> Result<(), AdapterError> {
        if let Some(schema) = self.schemas.remove(&id) {
            // Remove from name index
            if let Some(ids) = self.name_index.get_mut(&schema.name) {
                ids.retain(|&i| i != id);
                if ids.is_empty() {
                    self.name_index.remove(&schema.name);
                }
            }
            Ok(())
        } else {
            Err(AdapterError::InvalidOperation(format!(
                "Schema with ID {:?} not found",
                id
            )))
        }
    }

    fn search_schemas(&self, pattern: &str) -> Result<Vec<SchemaMetadata>, AdapterError> {
        let pattern_lower = pattern.to_lowercase();
        let mut results: Vec<SchemaMetadata> = self
            .schemas
            .values()
            .filter(|s| s.name.to_lowercase().contains(&pattern_lower))
            .map(|s| SchemaMetadata {
                id: s.id,
                name: s.name.clone(),
                version: s.version,
                created_at: s.created_at,
                updated_at: s.updated_at,
                num_domains: s.table.domains.len(),
                num_predicates: s.table.predicates.len(),
                num_variables: s.table.variables.len(),
                description: s.description.clone(),
            })
            .collect();

        results.sort_by_key(|m| m.name.clone());
        Ok(results)
    }

    fn get_schema_history(&self, name: &str) -> Result<Vec<SchemaVersion>, AdapterError> {
        let ids = self.name_index.get(name).ok_or_else(|| {
            AdapterError::InvalidOperation(format!("Schema '{}' not found", name))
        })?;

        let mut versions: Vec<SchemaVersion> = ids
            .iter()
            .filter_map(|id| {
                self.schemas.get(id).map(|s| SchemaVersion {
                    version: s.version,
                    timestamp: s.created_at,
                    description: format!("Version {}", s.version),
                    schema_id: s.id,
                })
            })
            .collect();

        versions.sort_by_key(|v| v.version);
        Ok(versions)
    }
}

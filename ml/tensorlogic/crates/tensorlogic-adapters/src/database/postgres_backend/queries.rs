//! PostgreSQL async query helpers for schema storage.

use tokio_postgres::Client;

use super::{PostgreSQLDatabase, SchemaId, SchemaMetadata, SchemaVersion};
use crate::{AdapterError, DomainInfo, PredicateInfo, SymbolTable};

/// Generate CREATE TABLE statements for PostgreSQL.
///
/// PostgreSQL uses SERIAL instead of AUTOINCREMENT.
pub(super) fn create_tables_postgres_sql() -> Vec<String> {
    vec![
        // Schemas table
        r#"
        CREATE TABLE IF NOT EXISTS schemas (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            created_at BIGINT NOT NULL,
            updated_at BIGINT NOT NULL,
            description TEXT,
            UNIQUE(name, version)
        )
        "#
        .to_string(),
        // Domains table
        r#"
        CREATE TABLE IF NOT EXISTS domains (
            id SERIAL PRIMARY KEY,
            schema_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            cardinality BIGINT NOT NULL,
            description TEXT,
            metadata TEXT,
            FOREIGN KEY (schema_id) REFERENCES schemas(id) ON DELETE CASCADE,
            UNIQUE(schema_id, name)
        )
        "#
        .to_string(),
        // Predicates table
        r#"
        CREATE TABLE IF NOT EXISTS predicates (
            id SERIAL PRIMARY KEY,
            schema_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            arity INTEGER NOT NULL,
            description TEXT,
            constraints TEXT,
            metadata TEXT,
            FOREIGN KEY (schema_id) REFERENCES schemas(id) ON DELETE CASCADE,
            UNIQUE(schema_id, name)
        )
        "#
        .to_string(),
        // Predicate arguments table
        r#"
        CREATE TABLE IF NOT EXISTS predicate_arguments (
            id SERIAL PRIMARY KEY,
            predicate_id INTEGER NOT NULL,
            position INTEGER NOT NULL,
            domain_name TEXT NOT NULL,
            FOREIGN KEY (predicate_id) REFERENCES predicates(id) ON DELETE CASCADE,
            UNIQUE(predicate_id, position)
        )
        "#
        .to_string(),
        // Variables table
        r#"
        CREATE TABLE IF NOT EXISTS variables (
            id SERIAL PRIMARY KEY,
            schema_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            domain_name TEXT NOT NULL,
            FOREIGN KEY (schema_id) REFERENCES schemas(id) ON DELETE CASCADE,
            UNIQUE(schema_id, name)
        )
        "#
        .to_string(),
        // Indexes
        "CREATE INDEX IF NOT EXISTS idx_schemas_name ON schemas(name)".to_string(),
        "CREATE INDEX IF NOT EXISTS idx_domains_schema ON domains(schema_id)".to_string(),
        "CREATE INDEX IF NOT EXISTS idx_predicates_schema ON predicates(schema_id)".to_string(),
    ]
}

/// Store a schema asynchronously.
pub(super) async fn store_schema(
    client: &Client,
    name: &str,
    table: &SymbolTable,
) -> Result<SchemaId, AdapterError> {
    let now = PostgreSQLDatabase::current_timestamp();

    // Check if schema exists
    let existing_version: Option<i32> = client
        .query_opt("SELECT MAX(version) FROM schemas WHERE name = $1", &[&name])
        .await
        .ok()
        .flatten()
        .and_then(|row| row.get(0));

    let version = existing_version.map(|v| v + 1).unwrap_or(1);

    // Insert schema
    let row = client
        .query_one(
            "INSERT INTO schemas (name, version, created_at, updated_at) VALUES ($1, $2, $3, $4) RETURNING id",
            &[&name, &version, &now, &now],
        )
        .await
        .map_err(|e| AdapterError::InvalidOperation(format!("Failed to insert schema: {}", e)))?;

    let schema_id: i32 = row.get(0);

    // Store domains
    for (domain_name, domain) in &table.domains {
        let metadata_json = serde_json::to_string(&domain.metadata).ok();
        client
            .execute(
                "INSERT INTO domains (schema_id, name, cardinality, description, metadata) VALUES ($1, $2, $3, $4, $5)",
                &[
                    &schema_id,
                    &domain_name.as_str(),
                    &(domain.cardinality as i64),
                    &domain.description.as_ref(),
                    &metadata_json.as_ref(),
                ],
            )
            .await
            .map_err(|e| {
                AdapterError::InvalidOperation(format!("Failed to insert domain: {}", e))
            })?;
    }

    // Store predicates
    for (predicate_name, predicate) in &table.predicates {
        let constraints_json = serde_json::to_string(&predicate.constraints).ok();
        let metadata_json = serde_json::to_string(&predicate.metadata).ok();

        let pred_row = client
            .query_one(
                "INSERT INTO predicates (schema_id, name, arity, description, constraints, metadata) VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
                &[
                    &schema_id,
                    &predicate_name.as_str(),
                    &(predicate.arg_domains.len() as i32),
                    &predicate.description.as_ref(),
                    &constraints_json.as_ref(),
                    &metadata_json.as_ref(),
                ],
            )
            .await
            .map_err(|e| {
                AdapterError::InvalidOperation(format!("Failed to insert predicate: {}", e))
            })?;

        let predicate_id: i32 = pred_row.get(0);

        // Store argument domains
        for (position, domain_name) in predicate.arg_domains.iter().enumerate() {
            client
                .execute(
                    "INSERT INTO predicate_arguments (predicate_id, position, domain_name) VALUES ($1, $2, $3)",
                    &[&predicate_id, &(position as i32), &domain_name.as_str()],
                )
                .await
                .map_err(|e| {
                    AdapterError::InvalidOperation(format!(
                        "Failed to insert predicate argument: {}",
                        e
                    ))
                })?;
        }
    }

    // Store variables
    for (var_name, domain_name) in &table.variables {
        client
            .execute(
                "INSERT INTO variables (schema_id, name, domain_name) VALUES ($1, $2, $3)",
                &[&schema_id, &var_name.as_str(), &domain_name.as_str()],
            )
            .await
            .map_err(|e| {
                AdapterError::InvalidOperation(format!("Failed to insert variable: {}", e))
            })?;
    }

    Ok(SchemaId(schema_id as u64))
}

/// Load a schema by ID asynchronously.
pub(super) async fn load_schema(
    client: &Client,
    id: SchemaId,
) -> Result<SymbolTable, AdapterError> {
    let schema_id = id.0 as i32;

    // Verify schema exists
    client
        .query_opt("SELECT id FROM schemas WHERE id = $1", &[&schema_id])
        .await
        .map_err(|e| AdapterError::InvalidOperation(format!("Database error: {}", e)))?
        .ok_or_else(|| {
            AdapterError::InvalidOperation(format!("Schema with ID {:?} not found", id))
        })?;

    let mut table = SymbolTable::new();

    // Load domains
    let domain_rows = client
        .query(
            "SELECT name, cardinality, description, metadata FROM domains WHERE schema_id = $1",
            &[&schema_id],
        )
        .await
        .map_err(|e| AdapterError::InvalidOperation(format!("Failed to query domains: {}", e)))?;

    for row in domain_rows {
        let name: String = row.get(0);
        let cardinality: i64 = row.get(1);
        let description: Option<String> = row.get(2);
        let metadata_json: Option<String> = row.get(3);

        let mut domain = DomainInfo::new(&name, cardinality as usize);
        if let Some(desc) = description {
            domain = domain.with_description(desc);
        }
        if let Some(meta_str) = metadata_json {
            if let Ok(metadata) = serde_json::from_str(&meta_str) {
                domain.metadata = metadata;
            }
        }

        table.domains.insert(name, domain);
    }

    // Load predicates
    let predicate_rows = client
        .query(
            "SELECT id, name, arity, description, constraints, metadata FROM predicates WHERE schema_id = $1",
            &[&schema_id],
        )
        .await
        .map_err(|e| {
            AdapterError::InvalidOperation(format!("Failed to query predicates: {}", e))
        })?;

    for pred_row in predicate_rows {
        let predicate_id: i32 = pred_row.get(0);
        let name: String = pred_row.get(1);
        let _arity: i32 = pred_row.get(2);
        let description: Option<String> = pred_row.get(3);
        let constraints_json: Option<String> = pred_row.get(4);
        let metadata_json: Option<String> = pred_row.get(5);

        // Load argument domains
        let arg_rows = client
            .query(
                "SELECT position, domain_name FROM predicate_arguments WHERE predicate_id = $1 ORDER BY position",
                &[&predicate_id],
            )
            .await
            .map_err(|e| {
                AdapterError::InvalidOperation(format!(
                    "Failed to query predicate args: {}",
                    e
                ))
            })?;

        let arg_domains: Vec<String> = arg_rows.iter().map(|row| row.get(1)).collect();

        let mut predicate = PredicateInfo::new(&name, arg_domains);
        if let Some(desc) = description {
            predicate = predicate.with_description(desc);
        }
        if let Some(constraints_str) = constraints_json {
            if let Ok(constraints) = serde_json::from_str(&constraints_str) {
                predicate.constraints = constraints;
            }
        }
        if let Some(meta_str) = metadata_json {
            if let Ok(metadata) = serde_json::from_str(&meta_str) {
                predicate.metadata = metadata;
            }
        }

        table.predicates.insert(name, predicate);
    }

    // Load variables
    let var_rows = client
        .query(
            "SELECT name, domain_name FROM variables WHERE schema_id = $1",
            &[&schema_id],
        )
        .await
        .map_err(|e| AdapterError::InvalidOperation(format!("Failed to query variables: {}", e)))?;

    for row in var_rows {
        let name: String = row.get(0);
        let domain_name: String = row.get(1);
        table.variables.insert(name, domain_name);
    }

    Ok(table)
}

/// Load a schema by name (latest version) asynchronously.
pub(super) async fn load_schema_by_name(
    client: &Client,
    name: &str,
) -> Result<SymbolTable, AdapterError> {
    let row = client
        .query_opt(
            "SELECT id FROM schemas WHERE name = $1 ORDER BY version DESC LIMIT 1",
            &[&name],
        )
        .await
        .map_err(|e| AdapterError::InvalidOperation(format!("Database error: {}", e)))?
        .ok_or_else(|| AdapterError::InvalidOperation(format!("Schema '{}' not found", name)))?;

    let schema_id: i32 = row.get(0);
    load_schema(client, SchemaId(schema_id as u64)).await
}

/// List all schemas asynchronously.
pub(super) async fn list_schemas(client: &Client) -> Result<Vec<SchemaMetadata>, AdapterError> {
    let rows = client
        .query(
            r#"
            SELECT s.id, s.name, s.version, s.created_at, s.updated_at, s.description,
                   (SELECT COUNT(*) FROM domains WHERE schema_id = s.id) as num_domains,
                   (SELECT COUNT(*) FROM predicates WHERE schema_id = s.id) as num_predicates,
                   (SELECT COUNT(*) FROM variables WHERE schema_id = s.id) as num_variables
            FROM schemas s
            ORDER BY s.name, s.version DESC
            "#,
            &[],
        )
        .await
        .map_err(|e| AdapterError::InvalidOperation(format!("Failed to query schemas: {}", e)))?;

    let schemas = rows
        .iter()
        .map(|row| SchemaMetadata {
            id: SchemaId(row.get::<_, i32>(0) as u64),
            name: row.get(1),
            version: row.get::<_, i32>(2) as u32,
            created_at: row.get::<_, i64>(3) as u64,
            updated_at: row.get::<_, i64>(4) as u64,
            num_domains: row.get::<_, i64>(6) as usize,
            num_predicates: row.get::<_, i64>(7) as usize,
            num_variables: row.get::<_, i64>(8) as usize,
            description: row.get(5),
        })
        .collect();

    Ok(schemas)
}

/// Delete a schema by ID asynchronously.
pub(super) async fn delete_schema(client: &Client, id: SchemaId) -> Result<(), AdapterError> {
    let schema_id = id.0 as i32;
    let affected = client
        .execute("DELETE FROM schemas WHERE id = $1", &[&schema_id])
        .await
        .map_err(|e| AdapterError::InvalidOperation(format!("Failed to delete schema: {}", e)))?;

    if affected == 0 {
        return Err(AdapterError::InvalidOperation(format!(
            "Schema with ID {:?} not found",
            id
        )));
    }

    Ok(())
}

/// Search schemas by pattern asynchronously.
pub(super) async fn search_schemas(
    client: &Client,
    pattern: &str,
) -> Result<Vec<SchemaMetadata>, AdapterError> {
    let search_pattern = format!("%{}%", pattern);
    let rows = client
        .query(
            r#"
            SELECT s.id, s.name, s.version, s.created_at, s.updated_at, s.description,
                   (SELECT COUNT(*) FROM domains WHERE schema_id = s.id) as num_domains,
                   (SELECT COUNT(*) FROM predicates WHERE schema_id = s.id) as num_predicates,
                   (SELECT COUNT(*) FROM variables WHERE schema_id = s.id) as num_variables
            FROM schemas s
            WHERE s.name LIKE $1
            ORDER BY s.name, s.version DESC
            "#,
            &[&search_pattern],
        )
        .await
        .map_err(|e| AdapterError::InvalidOperation(format!("Failed to query schemas: {}", e)))?;

    let schemas = rows
        .iter()
        .map(|row| SchemaMetadata {
            id: SchemaId(row.get::<_, i32>(0) as u64),
            name: row.get(1),
            version: row.get::<_, i32>(2) as u32,
            created_at: row.get::<_, i64>(3) as u64,
            updated_at: row.get::<_, i64>(4) as u64,
            num_domains: row.get::<_, i64>(6) as usize,
            num_predicates: row.get::<_, i64>(7) as usize,
            num_variables: row.get::<_, i64>(8) as usize,
            description: row.get(5),
        })
        .collect();

    Ok(schemas)
}

/// Get schema history asynchronously.
pub(super) async fn get_schema_history(
    client: &Client,
    name: &str,
) -> Result<Vec<SchemaVersion>, AdapterError> {
    let rows = client
        .query(
            "SELECT version, created_at, id FROM schemas WHERE name = $1 ORDER BY version ASC",
            &[&name],
        )
        .await
        .map_err(|e| AdapterError::InvalidOperation(format!("Failed to query versions: {}", e)))?;

    if rows.is_empty() {
        return Err(AdapterError::InvalidOperation(format!(
            "Schema '{}' not found",
            name
        )));
    }

    let versions = rows
        .iter()
        .map(|row| {
            let version: i32 = row.get(0);
            let timestamp: i64 = row.get(1);
            let schema_id: i32 = row.get(2);

            SchemaVersion {
                version: version as u32,
                timestamp: timestamp as u64,
                description: format!("Version {}", version),
                schema_id: SchemaId(schema_id as u64),
            }
        })
        .collect();

    Ok(versions)
}

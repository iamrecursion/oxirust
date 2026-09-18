//! PostgreSQL database backend for schema storage.

use tokio_postgres::{Client, NoTls};

use super::{SchemaId, SchemaMetadata, SchemaVersion};
use crate::{AdapterError, SymbolTable};

mod queries;

/// PostgreSQL database backend for schema storage.
///
/// This implementation provides persistent storage using PostgreSQL
/// with async support. The database schema is automatically created on first use.
///
/// # Example
///
/// ```no_run
/// # #[cfg(feature = "postgres")]
/// # {
/// use tensorlogic_adapters::{PostgreSQLDatabase, SymbolTable, DomainInfo};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let mut db = PostgreSQLDatabase::new("host=localhost user=postgres").await?;
/// let mut table = SymbolTable::new();
/// table.add_domain(DomainInfo::new("Person", 100))?;
///
/// let id = db.store_schema_async("test", &table).await?;
/// let loaded = db.load_schema_async(id).await?;
/// # Ok(())
/// # }
/// # }
/// ```
pub struct PostgreSQLDatabase {
    pub(super) client: Client,
}

impl PostgreSQLDatabase {
    /// Create a new PostgreSQL database connection.
    ///
    /// The connection string should be in the format:
    /// `host=localhost user=postgres password=password dbname=tensorlogic`
    pub async fn new(connection_string: &str) -> Result<Self, AdapterError> {
        let (client, connection) = tokio_postgres::connect(connection_string, NoTls)
            .await
            .map_err(|e| {
                AdapterError::InvalidOperation(format!("Failed to connect to PostgreSQL: {}", e))
            })?;

        // Spawn connection in background
        tokio::spawn(async move {
            if let Err(e) = connection.await {
                eprintln!("PostgreSQL connection error: {}", e);
            }
        });

        let mut db = Self { client };
        db.initialize_schema_async().await?;
        Ok(db)
    }

    /// Initialize the database schema (create tables if they don't exist).
    async fn initialize_schema_async(&mut self) -> Result<(), AdapterError> {
        for sql in queries::create_tables_postgres_sql() {
            self.client.execute(&sql, &[]).await.map_err(|e| {
                AdapterError::InvalidOperation(format!("Failed to create tables: {}", e))
            })?;
        }
        Ok(())
    }

    /// Get current timestamp (Unix epoch seconds).
    pub(super) fn current_timestamp() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("SystemTime after UNIX_EPOCH")
            .as_secs() as i64
    }

    /// Store a schema asynchronously.
    pub async fn store_schema_async(
        &mut self,
        name: &str,
        table: &SymbolTable,
    ) -> Result<SchemaId, AdapterError> {
        queries::store_schema(&self.client, name, table).await
    }

    /// Load a schema by ID asynchronously.
    pub async fn load_schema_async(&self, id: SchemaId) -> Result<SymbolTable, AdapterError> {
        queries::load_schema(&self.client, id).await
    }

    /// Load a schema by name (latest version) asynchronously.
    pub async fn load_schema_by_name_async(&self, name: &str) -> Result<SymbolTable, AdapterError> {
        queries::load_schema_by_name(&self.client, name).await
    }

    /// List all schemas asynchronously.
    pub async fn list_schemas_async(&self) -> Result<Vec<SchemaMetadata>, AdapterError> {
        queries::list_schemas(&self.client).await
    }

    /// Delete a schema by ID asynchronously.
    pub async fn delete_schema_async(&mut self, id: SchemaId) -> Result<(), AdapterError> {
        queries::delete_schema(&self.client, id).await
    }

    /// Search schemas by pattern asynchronously.
    pub async fn search_schemas_async(
        &self,
        pattern: &str,
    ) -> Result<Vec<SchemaMetadata>, AdapterError> {
        queries::search_schemas(&self.client, pattern).await
    }

    /// Get schema history asynchronously.
    pub async fn get_schema_history_async(
        &self,
        name: &str,
    ) -> Result<Vec<SchemaVersion>, AdapterError> {
        queries::get_schema_history(&self.client, name).await
    }
}

//! Collection management
//!
//! Provides collection capabilities:
//! - Hierarchical folder organization
//! - Static collections (manual asset management)
//! - Smart collections (dynamic queries)
//! - Collection sharing and permissions
//! - Collection export

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use crate::database::Database;
use crate::search::SearchEngine;
use crate::{MamError, Result};

/// Collection manager
pub struct CollectionManager {
    db: Arc<Database>,
    #[allow(dead_code)]
    search: Arc<SearchEngine>,
}

/// Collection record
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Collection {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub parent_id: Option<Uuid>,
    pub is_smart: bool,
    pub smart_query: Option<serde_json::Value>,
    pub created_by: Option<Uuid>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Collection item (asset in collection)
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct CollectionItem {
    pub id: Uuid,
    pub collection_id: Uuid,
    pub asset_id: Uuid,
    pub position: Option<i32>,
    pub added_by: Option<Uuid>,
    pub added_at: DateTime<Utc>,
}

/// Create collection request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateCollectionRequest {
    pub name: String,
    pub description: Option<String>,
    pub parent_id: Option<Uuid>,
    pub is_smart: bool,
    pub smart_query: Option<SmartQuery>,
    pub created_by: Option<Uuid>,
}

/// Update collection request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateCollectionRequest {
    pub name: Option<String>,
    pub description: Option<String>,
    pub parent_id: Option<Uuid>,
    pub smart_query: Option<SmartQuery>,
}

/// Smart collection query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartQuery {
    pub conditions: Vec<QueryCondition>,
    pub operator: QueryOperator,
    pub sort_by: Option<String>,
    pub sort_order: Option<String>,
    pub limit: Option<i32>,
}

/// Query condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryCondition {
    pub field: String,
    pub operator: ConditionOperator,
    pub value: serde_json::Value,
}

/// Query operator (AND/OR)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum QueryOperator {
    And,
    Or,
}

/// Condition operator
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ConditionOperator {
    Equals,
    NotEquals,
    Contains,
    StartsWith,
    EndsWith,
    GreaterThan,
    LessThan,
    GreaterOrEqual,
    LessOrEqual,
    In,
    NotIn,
}

/// Columns on `assets` that a smart-collection [`QueryCondition::field`] or
/// [`SmartQuery::sort_by`] may reference. Column/table identifiers can never be SQL
/// bind parameters (`$N` placeholders only substitute *values*), so this allowlist —
/// not escaping — is what makes it safe to interpolate a user-supplied field name
/// into the query text in [`CollectionManager::build_condition_sql`] and
/// [`CollectionManager::execute_smart_query`]. Deliberately excludes JSONB/array
/// columns (`custom_metadata`, `keywords`, `categories`, which need different
/// comparison operators than the scalar ones supported here) and identifiers with no
/// legitimate end-user filtering use (`file_path`, `checksum`). Mirrors the `assets`
/// table definition in `migrations/20240101000000_initial_schema.sql`.
const ALLOWED_ASSET_COLUMNS: &[&str] = &[
    "id",
    "filename",
    "file_size",
    "mime_type",
    "duration_ms",
    "width",
    "height",
    "frame_rate",
    "video_codec",
    "audio_codec",
    "bit_rate",
    "title",
    "description",
    "copyright",
    "license",
    "creator",
    "status",
    "created_by",
    "created_at",
    "updated_at",
];

/// Validates `field` against [`ALLOWED_ASSET_COLUMNS`], returning it unchanged (an
/// allowlisted identifier, not attacker-controlled data) on success.
fn validate_asset_column(field: &str) -> Result<&str> {
    ALLOWED_ASSET_COLUMNS
        .iter()
        .find(|&&allowed| allowed == field)
        .copied()
        .ok_or_else(|| {
            MamError::InvalidInput(format!(
                "smart query references an unknown or disallowed column: {field}"
            ))
        })
}

/// A smart-query condition/sort value, ready to be bound as a real SQL parameter via
/// `sqlx::Query::bind` instead of interpolated into the query text.
/// [`QueryCondition::value`] arrives as an untyped `serde_json::Value` (deserialized
/// from user-authored smart-collection JSON — see
/// [`CollectionManager::execute_smart_query`]'s doc comment), so each condition's
/// value is converted to one of these variants up front via [`BindValue::from_json`].
enum BindValue {
    Text(String),
    Int(i64),
    Float(f64),
    Bool(bool),
}

impl BindValue {
    fn from_json(value: &serde_json::Value) -> Result<Self> {
        match value {
            serde_json::Value::String(s) => Ok(Self::Text(s.clone())),
            serde_json::Value::Bool(b) => Ok(Self::Bool(*b)),
            serde_json::Value::Number(n) => n
                .as_i64()
                .map(Self::Int)
                .or_else(|| n.as_f64().map(Self::Float))
                .ok_or_else(|| {
                    MamError::InvalidInput(format!(
                        "smart query condition value is not a representable number: {n}"
                    ))
                }),
            other => Err(MamError::InvalidInput(format!(
                "smart query condition value must be a string, number, or boolean, not: {other}"
            ))),
        }
    }
}

/// Collection with item count
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionWithCount {
    pub collection: Collection,
    pub item_count: i64,
}

/// Collection tree node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionTreeNode {
    pub collection: Collection,
    pub children: Vec<CollectionTreeNode>,
    pub item_count: i64,
}

impl CollectionManager {
    /// Create a new collection manager
    #[must_use]
    pub fn new(db: Arc<Database>, search: Arc<SearchEngine>) -> Self {
        Self { db, search }
    }

    /// Create a new collection
    ///
    /// # Errors
    ///
    /// Returns an error if the insert fails
    pub async fn create_collection(&self, req: CreateCollectionRequest) -> Result<Collection> {
        // Validate parent exists if specified
        if let Some(parent_id) = req.parent_id {
            self.get_collection(parent_id).await?;
        }

        let smart_query_json = req.smart_query.and_then(|q| serde_json::to_value(q).ok());

        let collection = sqlx::query_as::<_, Collection>(
            "INSERT INTO collections (id, name, description, parent_id, is_smart, smart_query, created_by, created_at, updated_at)
             VALUES ($1, $2, $3, $4, $5, $6, $7, NOW(), NOW())
             RETURNING *"
        )
        .bind(Uuid::new_v4())
        .bind(&req.name)
        .bind(req.description)
        .bind(req.parent_id)
        .bind(req.is_smart)
        .bind(smart_query_json)
        .bind(req.created_by)
        .fetch_one(self.db.pool())
        .await?;

        Ok(collection)
    }

    /// Get collection by ID
    ///
    /// # Errors
    ///
    /// Returns an error if the collection is not found
    pub async fn get_collection(&self, collection_id: Uuid) -> Result<Collection> {
        let collection = sqlx::query_as::<_, Collection>("SELECT * FROM collections WHERE id = $1")
            .bind(collection_id)
            .fetch_one(self.db.pool())
            .await
            .map_err(|_| MamError::CollectionNotFound(collection_id))?;

        Ok(collection)
    }

    /// Update collection
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails
    pub async fn update_collection(
        &self,
        collection_id: Uuid,
        req: UpdateCollectionRequest,
    ) -> Result<Collection> {
        // Validate parent exists if specified
        if let Some(parent_id) = req.parent_id {
            if parent_id != collection_id {
                self.get_collection(parent_id).await?;
            } else {
                return Err(MamError::InvalidInput(
                    "Collection cannot be its own parent".to_string(),
                ));
            }
        }

        let smart_query_json = req.smart_query.and_then(|q| serde_json::to_value(q).ok());

        let collection = sqlx::query_as::<_, Collection>(
            "UPDATE collections SET
                name = COALESCE($2, name),
                description = COALESCE($3, description),
                parent_id = COALESCE($4, parent_id),
                smart_query = COALESCE($5, smart_query),
                updated_at = NOW()
             WHERE id = $1
             RETURNING *",
        )
        .bind(collection_id)
        .bind(req.name)
        .bind(req.description)
        .bind(req.parent_id)
        .bind(smart_query_json)
        .fetch_one(self.db.pool())
        .await?;

        Ok(collection)
    }

    /// Delete collection
    ///
    /// # Errors
    ///
    /// Returns an error if the delete fails
    pub async fn delete_collection(&self, collection_id: Uuid) -> Result<()> {
        // This will cascade delete items and child collections
        sqlx::query("DELETE FROM collections WHERE id = $1")
            .bind(collection_id)
            .execute(self.db.pool())
            .await?;

        Ok(())
    }

    /// List collections
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub async fn list_collections(
        &self,
        parent_id: Option<Uuid>,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<CollectionWithCount>> {
        let collections = if let Some(pid) = parent_id {
            sqlx::query_as::<_, Collection>(
                "SELECT * FROM collections WHERE parent_id = $1 ORDER BY name LIMIT $2 OFFSET $3",
            )
            .bind(pid)
            .bind(limit)
            .bind(offset)
            .fetch_all(self.db.pool())
            .await?
        } else {
            sqlx::query_as::<_, Collection>(
                "SELECT * FROM collections WHERE parent_id IS NULL ORDER BY name LIMIT $1 OFFSET $2"
            )
            .bind(limit)
            .bind(offset)
            .fetch_all(self.db.pool())
            .await?
        };

        let mut result = Vec::new();
        for collection in collections {
            let count = self.get_collection_item_count(collection.id).await?;
            result.push(CollectionWithCount {
                collection,
                item_count: count,
            });
        }

        Ok(result)
    }

    /// Get collection tree (hierarchical structure)
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub async fn get_collection_tree(&self) -> Result<Vec<CollectionTreeNode>> {
        // Get root collections
        let roots = sqlx::query_as::<_, Collection>(
            "SELECT * FROM collections WHERE parent_id IS NULL ORDER BY name",
        )
        .fetch_all(self.db.pool())
        .await?;

        let mut tree = Vec::new();
        for root in roots {
            let node = self.build_tree_node(root).await?;
            tree.push(node);
        }

        Ok(tree)
    }

    /// Build collection tree node recursively
    fn build_tree_node<'a>(
        &'a self,
        collection: Collection,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<CollectionTreeNode>> + Send + 'a>>
    {
        Box::pin(async move {
            let children_collections = sqlx::query_as::<_, Collection>(
                "SELECT * FROM collections WHERE parent_id = $1 ORDER BY name",
            )
            .bind(collection.id)
            .fetch_all(self.db.pool())
            .await?;

            let mut children = Vec::new();
            for child in children_collections {
                let child_node = self.build_tree_node(child).await?;
                children.push(child_node);
            }

            let item_count = self.get_collection_item_count(collection.id).await?;

            Ok(CollectionTreeNode {
                collection,
                children,
                item_count,
            })
        })
    }

    /// Add asset to collection
    ///
    /// # Errors
    ///
    /// Returns an error if the insert fails
    pub async fn add_asset(
        &self,
        collection_id: Uuid,
        asset_id: Uuid,
        position: Option<i32>,
        added_by: Option<Uuid>,
    ) -> Result<CollectionItem> {
        // Check collection exists
        let collection = self.get_collection(collection_id).await?;

        if collection.is_smart {
            return Err(MamError::InvalidInput(
                "Cannot manually add assets to smart collection".to_string(),
            ));
        }

        let item = sqlx::query_as::<_, CollectionItem>(
            "INSERT INTO collection_items (id, collection_id, asset_id, position, added_by, added_at)
             VALUES ($1, $2, $3, $4, $5, NOW())
             ON CONFLICT (collection_id, asset_id) DO UPDATE SET position = $4
             RETURNING *"
        )
        .bind(Uuid::new_v4())
        .bind(collection_id)
        .bind(asset_id)
        .bind(position)
        .bind(added_by)
        .fetch_one(self.db.pool())
        .await?;

        Ok(item)
    }

    /// Remove asset from collection
    ///
    /// # Errors
    ///
    /// Returns an error if the delete fails
    pub async fn remove_asset(&self, collection_id: Uuid, asset_id: Uuid) -> Result<()> {
        sqlx::query("DELETE FROM collection_items WHERE collection_id = $1 AND asset_id = $2")
            .bind(collection_id)
            .bind(asset_id)
            .execute(self.db.pool())
            .await?;

        Ok(())
    }

    /// Get assets in collection
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub async fn get_collection_assets(
        &self,
        collection_id: Uuid,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<Uuid>> {
        let collection = self.get_collection(collection_id).await?;

        let asset_ids = if collection.is_smart {
            // Execute smart query
            self.execute_smart_query(collection_id, limit, offset)
                .await?
        } else {
            // Get static collection items
            let items = sqlx::query_as::<_, CollectionItem>(
                "SELECT * FROM collection_items
                 WHERE collection_id = $1
                 ORDER BY COALESCE(position, 999999), added_at
                 LIMIT $2 OFFSET $3",
            )
            .bind(collection_id)
            .bind(limit)
            .bind(offset)
            .fetch_all(self.db.pool())
            .await?;

            items.into_iter().map(|i| i.asset_id).collect()
        };

        Ok(asset_ids)
    }

    /// Get collection item count
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub async fn get_collection_item_count(&self, collection_id: Uuid) -> Result<i64> {
        let collection = self.get_collection(collection_id).await?;

        let count = if collection.is_smart {
            // For smart collections, execute query and count results
            // This is a simplified version - in production you'd optimize this
            let assets = self.execute_smart_query(collection_id, 999999, 0).await?;
            assets.len() as i64
        } else {
            let count: i64 = sqlx::query_scalar(
                "SELECT COUNT(*) FROM collection_items WHERE collection_id = $1",
            )
            .bind(collection_id)
            .fetch_one(self.db.pool())
            .await?;
            count
        };

        Ok(count)
    }

    /// Execute smart collection query
    ///
    /// `query` (deserialized from `collection.smart_query`) is USER-AUTHORED data —
    /// an API caller creates a smart collection by supplying its own filter
    /// conditions, sort column, and sort order, all stored as JSON and replayed here
    /// every time the collection is viewed. Every identifier (`field`, `sort_by`) is
    /// therefore validated against [`ALLOWED_ASSET_COLUMNS`] before being interpolated
    /// (identifiers can never be bind parameters), and every condition *value* is
    /// routed through [`BindValue`]/`.bind()` rather than string interpolation. See
    /// [`Self::build_condition_sql`] for the per-condition detail.
    async fn execute_smart_query(
        &self,
        collection_id: Uuid,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<Uuid>> {
        let collection = self.get_collection(collection_id).await?;

        let smart_query = collection
            .smart_query
            .ok_or_else(|| MamError::Internal("Smart collection has no query".to_string()))?;

        let query: SmartQuery = serde_json::from_value(smart_query)?;

        // Build SQL query from conditions.
        let mut sql = String::from("SELECT id FROM assets WHERE status != 'deleted'");
        let mut binds: Vec<BindValue> = Vec::new();
        let mut param_num: i32 = 1;

        for (i, condition) in query.conditions.iter().enumerate() {
            if i > 0 {
                match query.operator {
                    QueryOperator::And => sql.push_str(" AND "),
                    QueryOperator::Or => sql.push_str(" OR "),
                }
            } else {
                sql.push_str(" AND ");
            }

            let (condition_sql, condition_binds) =
                self.build_condition_sql(condition, &mut param_num)?;
            sql.push_str(&condition_sql);
            binds.extend(condition_binds);
        }

        if let Some(sort_by) = &query.sort_by {
            let column = validate_asset_column(sort_by)?;
            // Only ASC/DESC are meaningful SQL keywords here; anything else falls
            // back to ASC rather than being interpolated verbatim.
            let order = match query.sort_order.as_deref() {
                Some(o) if o.eq_ignore_ascii_case("desc") => "DESC",
                _ => "ASC",
            };
            sql.push_str(&format!(" ORDER BY {column} {order}"));
        }

        sql.push_str(&format!(" LIMIT ${param_num} OFFSET ${}", param_num + 1));
        binds.push(BindValue::Int(limit));
        binds.push(BindValue::Int(offset));

        // Execute query, binding every collected value in the exact order its
        // placeholder was appended above. `sql` now contains only allowlisted
        // column identifiers and `$N` placeholders -- audited safe for sqlx 0.9's
        // `SqlSafeStr` gate (see the doc comment above and on `build_condition_sql`).
        let mut q = sqlx::query_scalar::<_, Uuid>(sqlx::AssertSqlSafe(sql));
        for bind in binds {
            q = match bind {
                BindValue::Text(s) => q.bind(s),
                BindValue::Int(i) => q.bind(i),
                BindValue::Float(f) => q.bind(f),
                BindValue::Bool(b) => q.bind(b),
            };
        }
        let rows = q.fetch_all(self.db.pool()).await?;

        Ok(rows)
    }

    /// Builds one condition's SQL fragment (using `$N` placeholders numbered from
    /// `*param_num` onward) plus the values to bind for it.
    ///
    /// `condition.field`/`condition.value` are user-authored (see
    /// [`Self::execute_smart_query`]'s doc comment) — `field` is validated against
    /// [`ALLOWED_ASSET_COLUMNS`] (the previous implementation interpolated it
    /// completely unchecked, allowing an arbitrary column/expression to be injected),
    /// and `value` is converted to a [`BindValue`] and returned for the caller to
    /// `.bind()` rather than being spliced into the SQL text via `format!` (the
    /// previous implementation interpolated the raw JSON value directly into the
    /// query, including inside a hand-rolled `'...'` string literal for
    /// Equals/NotEquals/Contains/StartsWith/EndsWith with no escaping — a SQL
    /// injection vulnerability: any `value` containing a `'` could break out of the
    /// literal and inject arbitrary SQL).
    fn build_condition_sql(
        &self,
        condition: &QueryCondition,
        param_num: &mut i32,
    ) -> Result<(String, Vec<BindValue>)> {
        let field = validate_asset_column(&condition.field)?;
        let mut next_placeholder = || {
            let p = *param_num;
            *param_num += 1;
            format!("${p}")
        };

        match condition.operator {
            ConditionOperator::Equals => {
                let v = BindValue::from_json(&condition.value)?;
                Ok((format!("{field} = {}", next_placeholder()), vec![v]))
            }
            ConditionOperator::NotEquals => {
                let v = BindValue::from_json(&condition.value)?;
                Ok((format!("{field} != {}", next_placeholder()), vec![v]))
            }
            ConditionOperator::Contains => {
                let text = condition.value.as_str().ok_or_else(|| {
                    MamError::InvalidInput("Contains requires a string value".to_string())
                })?;
                Ok((
                    format!("{field} ILIKE {}", next_placeholder()),
                    vec![BindValue::Text(format!("%{text}%"))],
                ))
            }
            ConditionOperator::StartsWith => {
                let text = condition.value.as_str().ok_or_else(|| {
                    MamError::InvalidInput("StartsWith requires a string value".to_string())
                })?;
                Ok((
                    format!("{field} ILIKE {}", next_placeholder()),
                    vec![BindValue::Text(format!("{text}%"))],
                ))
            }
            ConditionOperator::EndsWith => {
                let text = condition.value.as_str().ok_or_else(|| {
                    MamError::InvalidInput("EndsWith requires a string value".to_string())
                })?;
                Ok((
                    format!("{field} ILIKE {}", next_placeholder()),
                    vec![BindValue::Text(format!("%{text}"))],
                ))
            }
            ConditionOperator::GreaterThan => {
                let v = BindValue::from_json(&condition.value)?;
                Ok((format!("{field} > {}", next_placeholder()), vec![v]))
            }
            ConditionOperator::LessThan => {
                let v = BindValue::from_json(&condition.value)?;
                Ok((format!("{field} < {}", next_placeholder()), vec![v]))
            }
            ConditionOperator::GreaterOrEqual => {
                let v = BindValue::from_json(&condition.value)?;
                Ok((format!("{field} >= {}", next_placeholder()), vec![v]))
            }
            ConditionOperator::LessOrEqual => {
                let v = BindValue::from_json(&condition.value)?;
                Ok((format!("{field} <= {}", next_placeholder()), vec![v]))
            }
            ConditionOperator::In | ConditionOperator::NotIn => {
                let items = condition.value.as_array().ok_or_else(|| {
                    MamError::InvalidInput("In/NotIn requires an array value".to_string())
                })?;
                if items.is_empty() {
                    return Err(MamError::InvalidInput(
                        "In/NotIn requires a non-empty array value".to_string(),
                    ));
                }
                let mut binds = Vec::with_capacity(items.len());
                let mut placeholders = Vec::with_capacity(items.len());
                for item in items {
                    binds.push(BindValue::from_json(item)?);
                    placeholders.push(next_placeholder());
                }
                let keyword = if matches!(condition.operator, ConditionOperator::In) {
                    "IN"
                } else {
                    "NOT IN"
                };
                Ok((
                    format!("{field} {keyword} ({})", placeholders.join(", ")),
                    binds,
                ))
            }
        }
    }

    /// Reorder assets in collection
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails
    pub async fn reorder_assets(
        &self,
        collection_id: Uuid,
        asset_positions: Vec<(Uuid, i32)>,
    ) -> Result<()> {
        let collection = self.get_collection(collection_id).await?;

        if collection.is_smart {
            return Err(MamError::InvalidInput(
                "Cannot reorder assets in smart collection".to_string(),
            ));
        }

        // Update positions in transaction
        for (asset_id, position) in asset_positions {
            sqlx::query(
                "UPDATE collection_items SET position = $3
                 WHERE collection_id = $1 AND asset_id = $2",
            )
            .bind(collection_id)
            .bind(asset_id)
            .bind(position)
            .execute(self.db.pool())
            .await?;
        }

        Ok(())
    }

    /// Check if asset is in collection
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub async fn is_asset_in_collection(
        &self,
        collection_id: Uuid,
        asset_id: Uuid,
    ) -> Result<bool> {
        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM collection_items WHERE collection_id = $1 AND asset_id = $2",
        )
        .bind(collection_id)
        .bind(asset_id)
        .fetch_one(self.db.pool())
        .await?;

        Ok(count > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smart_query_serialization() {
        let query = SmartQuery {
            conditions: vec![QueryCondition {
                field: "mime_type".to_string(),
                operator: ConditionOperator::Equals,
                value: serde_json::json!("video/mp4"),
            }],
            operator: QueryOperator::And,
            sort_by: Some("created_at".to_string()),
            sort_order: Some("DESC".to_string()),
            limit: Some(100),
        };

        let json = serde_json::to_string(&query).expect("should succeed in test");
        let deserialized: SmartQuery = serde_json::from_str(&json).expect("should succeed in test");

        assert_eq!(deserialized.conditions.len(), 1);
        assert_eq!(deserialized.limit, Some(100));
    }

    #[test]
    fn test_query_operator() {
        let op = QueryOperator::And;
        let json = serde_json::to_string(&op).expect("should succeed in test");
        assert!(json.contains("And"));
    }

    #[test]
    fn test_condition_operator() {
        let op = ConditionOperator::Contains;
        let json = serde_json::to_string(&op).expect("should succeed in test");
        assert!(json.contains("Contains"));
    }
}

//! Folder structure and smart collections
//!
//! Provides organization features for:
//! - Virtual folder hierarchy
//! - Smart collections (dynamic queries)
//! - Manual collections
//! - Nested collections
//! - Collection permissions
//! - Folder templates
//! - Automatic organization rules

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use crate::database::Database;
use crate::search::SearchEngine;
use crate::{MamError, Result};

/// Folder manager handles folder and collection operations
pub struct FolderManager {
    db: Arc<Database>,
    #[allow(dead_code)]
    search: Arc<SearchEngine>,
}

/// Folder in virtual hierarchy
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Folder {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub parent_id: Option<Uuid>,
    pub path: String,
    pub icon: Option<String>,
    pub color: Option<String>,
    pub created_by: Option<Uuid>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Collection (manual or smart)
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Collection {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub folder_id: Option<Uuid>,
    pub collection_type: String,
    pub query: Option<serde_json::Value>,
    pub sort_order: Option<String>,
    pub thumbnail_asset_id: Option<Uuid>,
    pub is_public: bool,
    pub created_by: Option<Uuid>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Collection type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CollectionType {
    /// Manual collection (user-defined assets)
    Manual,
    /// Smart collection (dynamic query)
    Smart,
    /// Template collection
    Template,
}

impl CollectionType {
    /// Convert to string
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Manual => "manual",
            Self::Smart => "smart",
            Self::Template => "template",
        }
    }
}

impl std::str::FromStr for CollectionType {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "manual" => Ok(Self::Manual),
            "smart" => Ok(Self::Smart),
            "template" => Ok(Self::Template),
            _ => Err(format!("Invalid collection type: {s}")),
        }
    }
}

/// Collection item (for manual collections)
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct CollectionItem {
    pub collection_id: Uuid,
    pub asset_id: Uuid,
    pub sort_order: i32,
    pub added_by: Option<Uuid>,
    pub created_at: DateTime<Utc>,
}

/// Smart collection query builder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartCollectionQuery {
    pub filters: Vec<QueryFilter>,
    pub sort_by: Option<String>,
    pub sort_direction: Option<String>,
    pub limit: Option<i64>,
}

/// Query filter for smart collections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryFilter {
    pub field: String,
    pub operator: FilterOperator,
    pub value: serde_json::Value,
}

/// Filter operator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterOperator {
    /// Equals
    Equals,
    /// Not equals
    NotEquals,
    /// Greater than
    GreaterThan,
    /// Less than
    LessThan,
    /// Greater than or equal
    GreaterThanOrEqual,
    /// Less than or equal
    LessThanOrEqual,
    /// Contains (for strings/arrays)
    Contains,
    /// Does not contain
    NotContains,
    /// Starts with
    StartsWith,
    /// Ends with
    EndsWith,
    /// In list
    In,
    /// Not in list
    NotIn,
    /// Between two values
    Between,
    /// Is null
    IsNull,
    /// Is not null
    IsNotNull,
}

impl FilterOperator {
    /// Convert to SQL operator
    #[must_use]
    pub const fn to_sql(&self) -> &'static str {
        match self {
            Self::Equals => "=",
            Self::NotEquals => "!=",
            Self::GreaterThan => ">",
            Self::LessThan => "<",
            Self::GreaterThanOrEqual => ">=",
            Self::LessThanOrEqual => "<=",
            Self::Contains => "LIKE",
            Self::NotContains => "NOT LIKE",
            Self::StartsWith => "LIKE",
            Self::EndsWith => "LIKE",
            Self::In => "IN",
            Self::NotIn => "NOT IN",
            Self::Between => "BETWEEN",
            Self::IsNull => "IS NULL",
            Self::IsNotNull => "IS NOT NULL",
        }
    }
}

/// Folder hierarchy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FolderHierarchy {
    pub folder: Folder,
    pub children: Vec<FolderHierarchy>,
    pub collections: Vec<Collection>,
}

/// Organization rule for automatic folder assignment
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct OrganizationRule {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub priority: i32,
    pub is_active: bool,
    pub conditions: serde_json::Value,
    pub target_folder_id: Uuid,
    pub created_by: Option<Uuid>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Folder template
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct FolderTemplate {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub structure: serde_json::Value,
    pub created_by: Option<Uuid>,
    pub created_at: DateTime<Utc>,
}

impl FolderManager {
    /// Create a new folder manager
    #[must_use]
    pub fn new(db: Arc<Database>, search: Arc<SearchEngine>) -> Self {
        Self { db, search }
    }

    /// Create a folder
    ///
    /// # Errors
    ///
    /// Returns an error if creation fails
    pub async fn create_folder(
        &self,
        name: String,
        description: Option<String>,
        parent_id: Option<Uuid>,
        created_by: Option<Uuid>,
    ) -> Result<Folder> {
        // Build path
        let path = if let Some(parent_id) = parent_id {
            let parent = self.get_folder(parent_id).await?;
            format!("{}/{}", parent.path, name)
        } else {
            format!("/{name}")
        };

        let folder = sqlx::query_as::<_, Folder>(
            "INSERT INTO folders
             (id, name, description, parent_id, path, created_by, created_at, updated_at)
             VALUES ($1, $2, $3, $4, $5, $6, NOW(), NOW())
             RETURNING *",
        )
        .bind(Uuid::new_v4())
        .bind(&name)
        .bind(description)
        .bind(parent_id)
        .bind(&path)
        .bind(created_by)
        .fetch_one(self.db.pool())
        .await?;

        Ok(folder)
    }

    /// Get folder by ID
    ///
    /// # Errors
    ///
    /// Returns an error if folder not found
    pub async fn get_folder(&self, folder_id: Uuid) -> Result<Folder> {
        let folder = sqlx::query_as::<_, Folder>("SELECT * FROM folders WHERE id = $1")
            .bind(folder_id)
            .fetch_one(self.db.pool())
            .await?;

        Ok(folder)
    }

    /// Update folder
    ///
    /// # Errors
    ///
    /// Returns an error if update fails
    pub async fn update_folder(
        &self,
        folder_id: Uuid,
        name: Option<String>,
        description: Option<String>,
        parent_id: Option<Option<Uuid>>,
    ) -> Result<Folder> {
        let current = self.get_folder(folder_id).await?;

        let new_name = name.unwrap_or(current.name.clone());
        let new_parent_id = parent_id.unwrap_or(current.parent_id);

        // Rebuild path
        let new_path = if let Some(pid) = new_parent_id {
            let parent = self.get_folder(pid).await?;
            format!("{}/{}", parent.path, new_name)
        } else {
            format!("/{new_name}")
        };

        let folder = sqlx::query_as::<_, Folder>(
            "UPDATE folders SET
                name = $2,
                description = COALESCE($3, description),
                parent_id = $4,
                path = $5,
                updated_at = NOW()
             WHERE id = $1
             RETURNING *",
        )
        .bind(folder_id)
        .bind(&new_name)
        .bind(description)
        .bind(new_parent_id)
        .bind(&new_path)
        .fetch_one(self.db.pool())
        .await?;

        Ok(folder)
    }

    /// Delete folder
    ///
    /// # Errors
    ///
    /// Returns an error if deletion fails
    pub async fn delete_folder(&self, folder_id: Uuid) -> Result<()> {
        // Check for children
        let children: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM folders WHERE parent_id = $1")
            .bind(folder_id)
            .fetch_one(self.db.pool())
            .await?;

        if children > 0 {
            return Err(MamError::InvalidInput(
                "Cannot delete folder with children".to_string(),
            ));
        }

        // Check for collections
        let collections: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM collections WHERE folder_id = $1")
                .bind(folder_id)
                .fetch_one(self.db.pool())
                .await?;

        if collections > 0 {
            return Err(MamError::InvalidInput(
                "Cannot delete folder with collections".to_string(),
            ));
        }

        sqlx::query("DELETE FROM folders WHERE id = $1")
            .bind(folder_id)
            .execute(self.db.pool())
            .await?;

        Ok(())
    }

    /// Get folder hierarchy
    ///
    /// # Errors
    ///
    /// Returns an error if query fails
    pub async fn get_folder_hierarchy(&self) -> Result<Vec<FolderHierarchy>> {
        let folders = sqlx::query_as::<_, Folder>("SELECT * FROM folders ORDER BY path")
            .fetch_all(self.db.pool())
            .await?;

        let collections =
            sqlx::query_as::<_, Collection>("SELECT * FROM collections ORDER BY name")
                .fetch_all(self.db.pool())
                .await?;

        let folder_map: HashMap<Uuid, Folder> = folders.iter().map(|f| (f.id, f.clone())).collect();
        let mut collection_map: HashMap<Option<Uuid>, Vec<Collection>> = HashMap::new();

        for collection in collections {
            collection_map
                .entry(collection.folder_id)
                .or_default()
                .push(collection);
        }

        let root_folders: Vec<Folder> = folders
            .iter()
            .filter(|f| f.parent_id.is_none())
            .cloned()
            .collect();

        let hierarchy: Vec<FolderHierarchy> = root_folders
            .iter()
            .map(|f| Self::build_folder_hierarchy(f.clone(), &folder_map, &collection_map))
            .collect();

        Ok(hierarchy)
    }

    fn build_folder_hierarchy(
        folder: Folder,
        folder_map: &HashMap<Uuid, Folder>,
        collection_map: &HashMap<Option<Uuid>, Vec<Collection>>,
    ) -> FolderHierarchy {
        let children: Vec<FolderHierarchy> = folder_map
            .values()
            .filter(|f| f.parent_id == Some(folder.id))
            .cloned()
            .map(|child| Self::build_folder_hierarchy(child, folder_map, collection_map))
            .collect();

        let collections = collection_map
            .get(&Some(folder.id))
            .cloned()
            .unwrap_or_default();

        FolderHierarchy {
            folder,
            children,
            collections,
        }
    }

    /// Create a collection
    ///
    /// # Errors
    ///
    /// Returns an error if creation fails
    pub async fn create_collection(
        &self,
        name: String,
        description: Option<String>,
        folder_id: Option<Uuid>,
        collection_type: CollectionType,
        query: Option<SmartCollectionQuery>,
        created_by: Option<Uuid>,
    ) -> Result<Collection> {
        let query_json = query.and_then(|q| serde_json::to_value(q).ok());

        let collection = sqlx::query_as::<_, Collection>(
            "INSERT INTO collections
             (id, name, description, folder_id, collection_type, query, is_public, created_by, created_at, updated_at)
             VALUES ($1, $2, $3, $4, $5, $6, false, $7, NOW(), NOW())
             RETURNING *",
        )
        .bind(Uuid::new_v4())
        .bind(&name)
        .bind(description)
        .bind(folder_id)
        .bind(collection_type.as_str())
        .bind(query_json)
        .bind(created_by)
        .fetch_one(self.db.pool())
        .await?;

        Ok(collection)
    }

    /// Get collection by ID
    ///
    /// # Errors
    ///
    /// Returns an error if collection not found
    pub async fn get_collection(&self, collection_id: Uuid) -> Result<Collection> {
        let collection = sqlx::query_as::<_, Collection>("SELECT * FROM collections WHERE id = $1")
            .bind(collection_id)
            .fetch_one(self.db.pool())
            .await?;

        Ok(collection)
    }

    /// Update collection
    ///
    /// # Errors
    ///
    /// Returns an error if update fails
    pub async fn update_collection(
        &self,
        collection_id: Uuid,
        name: Option<String>,
        description: Option<String>,
        query: Option<SmartCollectionQuery>,
    ) -> Result<Collection> {
        let query_json = query.and_then(|q| serde_json::to_value(q).ok());

        let collection = sqlx::query_as::<_, Collection>(
            "UPDATE collections SET
                name = COALESCE($2, name),
                description = COALESCE($3, description),
                query = COALESCE($4, query),
                updated_at = NOW()
             WHERE id = $1
             RETURNING *",
        )
        .bind(collection_id)
        .bind(name)
        .bind(description)
        .bind(query_json)
        .fetch_one(self.db.pool())
        .await?;

        Ok(collection)
    }

    /// Delete collection
    ///
    /// # Errors
    ///
    /// Returns an error if deletion fails
    pub async fn delete_collection(&self, collection_id: Uuid) -> Result<()> {
        sqlx::query("DELETE FROM collections WHERE id = $1")
            .bind(collection_id)
            .execute(self.db.pool())
            .await?;

        Ok(())
    }

    /// Add asset to manual collection
    ///
    /// # Errors
    ///
    /// Returns an error if operation fails
    pub async fn add_to_collection(
        &self,
        collection_id: Uuid,
        asset_id: Uuid,
        added_by: Option<Uuid>,
    ) -> Result<()> {
        // Verify collection is manual
        let collection = self.get_collection(collection_id).await?;
        if collection.collection_type != CollectionType::Manual.as_str() {
            return Err(MamError::InvalidInput(
                "Can only add assets to manual collections".to_string(),
            ));
        }

        // Get next sort order
        let max_order: Option<i32> = sqlx::query_scalar(
            "SELECT MAX(sort_order) FROM collection_items WHERE collection_id = $1",
        )
        .bind(collection_id)
        .fetch_one(self.db.pool())
        .await?;

        let sort_order = max_order.unwrap_or(0) + 1;

        sqlx::query(
            "INSERT INTO collection_items (collection_id, asset_id, sort_order, added_by, created_at)
             VALUES ($1, $2, $3, $4, NOW())
             ON CONFLICT (collection_id, asset_id) DO NOTHING",
        )
        .bind(collection_id)
        .bind(asset_id)
        .bind(sort_order)
        .bind(added_by)
        .execute(self.db.pool())
        .await?;

        Ok(())
    }

    /// Remove asset from collection
    ///
    /// # Errors
    ///
    /// Returns an error if operation fails
    pub async fn remove_from_collection(&self, collection_id: Uuid, asset_id: Uuid) -> Result<()> {
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
    /// Returns an error if query fails
    pub async fn get_collection_assets(&self, collection_id: Uuid) -> Result<Vec<Uuid>> {
        let collection = self.get_collection(collection_id).await?;

        if collection.collection_type == CollectionType::Manual.as_str() {
            // Manual collection - get from collection_items
            let asset_ids = sqlx::query_scalar::<_, Uuid>(
                "SELECT asset_id FROM collection_items
                 WHERE collection_id = $1
                 ORDER BY sort_order",
            )
            .bind(collection_id)
            .fetch_all(self.db.pool())
            .await?;

            Ok(asset_ids)
        } else {
            // Smart collection - execute query
            // Placeholder: In production, would parse and execute the query
            Ok(Vec::new())
        }
    }

    /// Create organization rule
    ///
    /// # Errors
    ///
    /// Returns an error if creation fails
    pub async fn create_organization_rule(
        &self,
        name: String,
        description: Option<String>,
        priority: i32,
        conditions: serde_json::Value,
        target_folder_id: Uuid,
        created_by: Option<Uuid>,
    ) -> Result<OrganizationRule> {
        let rule = sqlx::query_as::<_, OrganizationRule>(
            "INSERT INTO organization_rules
             (id, name, description, priority, is_active, conditions, target_folder_id, created_by, created_at, updated_at)
             VALUES ($1, $2, $3, $4, true, $5, $6, $7, NOW(), NOW())
             RETURNING *",
        )
        .bind(Uuid::new_v4())
        .bind(&name)
        .bind(description)
        .bind(priority)
        .bind(&conditions)
        .bind(target_folder_id)
        .bind(created_by)
        .fetch_one(self.db.pool())
        .await?;

        Ok(rule)
    }

    /// Apply organization rules to asset
    ///
    /// # Errors
    ///
    /// Returns an error if rule application fails
    pub async fn apply_organization_rules(&self, asset_id: Uuid) -> Result<Option<Uuid>> {
        // Get all active rules ordered by priority
        let rules = sqlx::query_as::<_, OrganizationRule>(
            "SELECT * FROM organization_rules WHERE is_active = true ORDER BY priority",
        )
        .fetch_all(self.db.pool())
        .await?;

        // Apply first matching rule
        for rule in rules {
            // Placeholder: In production, would evaluate conditions
            // For now, return first rule's target folder
            if self
                .evaluate_rule_conditions(asset_id, &rule.conditions)
                .await?
            {
                return Ok(Some(rule.target_folder_id));
            }
        }

        Ok(None)
    }

    async fn evaluate_rule_conditions(
        &self,
        _asset_id: Uuid,
        _conditions: &serde_json::Value,
    ) -> Result<bool> {
        // Placeholder: Would parse and evaluate conditions
        Ok(false)
    }

    /// Create folder template
    ///
    /// # Errors
    ///
    /// Returns an error if creation fails
    pub async fn create_folder_template(
        &self,
        name: String,
        description: Option<String>,
        structure: serde_json::Value,
        created_by: Option<Uuid>,
    ) -> Result<FolderTemplate> {
        let template = sqlx::query_as::<_, FolderTemplate>(
            "INSERT INTO folder_templates
             (id, name, description, structure, created_by, created_at)
             VALUES ($1, $2, $3, $4, $5, NOW())
             RETURNING *",
        )
        .bind(Uuid::new_v4())
        .bind(&name)
        .bind(description)
        .bind(&structure)
        .bind(created_by)
        .fetch_one(self.db.pool())
        .await?;

        Ok(template)
    }

    /// Apply folder template
    ///
    /// # Errors
    ///
    /// Returns an error if template application fails
    pub async fn apply_folder_template(
        &self,
        template_id: Uuid,
        parent_folder_id: Option<Uuid>,
        created_by: Option<Uuid>,
    ) -> Result<Vec<Uuid>> {
        let template =
            sqlx::query_as::<_, FolderTemplate>("SELECT * FROM folder_templates WHERE id = $1")
                .bind(template_id)
                .fetch_one(self.db.pool())
                .await?;

        // Placeholder: Would parse structure and create folders
        // For now, create a simple folder
        let folder = self
            .create_folder(
                template.name.clone(),
                template.description.clone(),
                parent_folder_id,
                created_by,
            )
            .await?;

        Ok(vec![folder.id])
    }
}

// ── FolderSync: filesystem watcher → MamStorage upload ──────────────────────

use crate::event_bus::MamEvent;
use crate::storage::MamStorage;
use std::collections::HashSet;
use std::path::PathBuf;
use tokio::sync::broadcast;

/// Errors produced by [`FolderSync`].
#[derive(Debug, thiserror::Error)]
pub enum FolderSyncError {
    /// The notify watcher could not be started.
    #[error("Watch error: {0}")]
    Watch(String),
    /// A storage operation failed.
    #[error("Storage error: {0}")]
    Storage(#[from] crate::storage::MamStorageError),
    /// An I/O error occurred.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<notify::Error> for FolderSyncError {
    fn from(e: notify::Error) -> Self {
        Self::Watch(e.to_string())
    }
}

/// Watches a directory for new/modified files and uploads them to `MamStorage`,
/// emitting [`MamEvent`] notifications via a broadcast channel.
pub struct FolderSync {
    watched_path: PathBuf,
    storage: Arc<MamStorage>,
    event_tx: broadcast::Sender<MamEvent>,
}

impl FolderSync {
    /// Create a new `FolderSync`.
    ///
    /// `event_tx` is used to broadcast [`MamEvent::AssetIngested`] and
    /// [`MamEvent::AssetDeleted`] whenever the watched folder changes.
    #[must_use]
    pub fn new(
        watched_path: PathBuf,
        storage: Arc<MamStorage>,
        event_tx: broadcast::Sender<MamEvent>,
    ) -> Self {
        Self {
            watched_path,
            storage,
            event_tx,
        }
    }

    /// Start watching the directory for filesystem events.
    ///
    /// This method spawns a background task; it returns as soon as the watcher
    /// is registered.  On each `Create` or `Modify` event the file is
    /// fingerprinted, uploaded (de-duplicated by fingerprint), and an
    /// [`MamEvent::AssetIngested`] event is broadcast.
    ///
    /// # Errors
    ///
    /// Returns [`FolderSyncError::Watch`] if the watcher cannot be registered.
    pub async fn start(&self) -> std::result::Result<(), FolderSyncError> {
        use notify::{RecommendedWatcher, RecursiveMode, Watcher};
        use std::sync::mpsc;

        let (tx, rx) = mpsc::channel::<notify::Result<notify::Event>>();
        let mut watcher = RecommendedWatcher::new(tx, notify::Config::default())?;
        watcher.watch(&self.watched_path, RecursiveMode::Recursive)?;

        let storage = Arc::clone(&self.storage);
        let event_tx = self.event_tx.clone();

        tokio::task::spawn_blocking(move || {
            // Keep watcher alive
            let _watcher = watcher;
            let mut seen: HashSet<String> = HashSet::new();

            for res in rx {
                match res {
                    Ok(event) => {
                        use notify::EventKind;
                        match &event.kind {
                            EventKind::Create(_) | EventKind::Modify(_) => {
                                for path in &event.paths {
                                    if path.is_file() {
                                        let path_str = path.to_string_lossy().to_string();
                                        if let Ok(data) = std::fs::read(path) {
                                            let fp = Self::fingerprint_bytes(&data);
                                            if seen.contains(&fp) {
                                                continue;
                                            }
                                            seen.insert(fp.clone());
                                            let key = path_str.clone();
                                            let size = data.len() as u64;

                                            // Upload asynchronously via a nested Tokio handle.
                                            let storage2 = Arc::clone(&storage);
                                            let key2 = key.clone();
                                            tokio::runtime::Handle::try_current()
                                                .map(|handle| {
                                                    handle.block_on(async move {
                                                        if let Err(e) =
                                                            storage2.put(&key2, &data).await
                                                        {
                                                            tracing::warn!(
                                                                "FolderSync upload failed: {e}"
                                                            );
                                                        }
                                                    });
                                                })
                                                .ok();

                                            let _ = event_tx.send(MamEvent::AssetIngested {
                                                asset_id: fp,
                                                path: path_str,
                                                size_bytes: size,
                                            });
                                        }
                                    }
                                }
                            }
                            EventKind::Remove(_) => {
                                for path in &event.paths {
                                    let _ = event_tx.send(MamEvent::AssetDeleted {
                                        asset_id: path.to_string_lossy().to_string(),
                                    });
                                }
                            }
                            _ => {}
                        }
                    }
                    Err(e) => {
                        tracing::warn!("FolderSync watch error: {e}");
                    }
                }
            }
        });

        Ok(())
    }

    /// Walk the watched directory once, uploading any files not yet known.
    ///
    /// Returns the number of newly uploaded files.
    ///
    /// # Errors
    ///
    /// Returns [`FolderSyncError::Io`] if directory traversal fails, or
    /// [`FolderSyncError::Storage`] if any upload fails.
    pub async fn sync_once(&self) -> std::result::Result<usize, FolderSyncError> {
        let mut count = 0usize;
        let mut seen: HashSet<String> = HashSet::new();

        let mut read_dir = tokio::fs::read_dir(&self.watched_path).await?;
        while let Some(entry) = read_dir.next_entry().await? {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let data = tokio::fs::read(&path).await?;
            let fp = Self::fingerprint_bytes(&data);
            if seen.contains(&fp) {
                continue;
            }
            seen.insert(fp.clone());

            let key = path.to_string_lossy().to_string();
            let size = data.len() as u64;

            // Check if already present; upload only if not.
            if !self.storage.exists(&key).await.unwrap_or(false) {
                self.storage.put(&key, &data).await?;
                let _ = self.event_tx.send(MamEvent::AssetIngested {
                    asset_id: fp,
                    path: key,
                    size_bytes: size,
                });
                count += 1;
            }
        }
        Ok(count)
    }

    /// Compute a SHA-256 fingerprint for `data`, returned as a hex string.
    #[must_use]
    fn fingerprint_bytes(data: &[u8]) -> String {
        // Use sha2 (already a MAM dep) for simple content fingerprinting.
        use sha2::{Digest, Sha256};
        let hash = Sha256::digest(data);
        hash.iter().fold(String::with_capacity(64), |mut acc, b| {
            use std::fmt::Write as _;
            let _ = write!(acc, "{b:02x}");
            acc
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collection_type_as_str() {
        assert_eq!(CollectionType::Manual.as_str(), "manual");
        assert_eq!(CollectionType::Smart.as_str(), "smart");
        assert_eq!(CollectionType::Template.as_str(), "template");
    }

    #[test]
    fn test_collection_type_from_str() {
        use std::str::FromStr;
        assert_eq!(
            CollectionType::from_str("manual").ok(),
            Some(CollectionType::Manual)
        );
        assert_eq!(
            CollectionType::from_str("smart").ok(),
            Some(CollectionType::Smart)
        );
        assert!(CollectionType::from_str("invalid").is_err());
    }

    #[test]
    fn test_filter_operator_to_sql() {
        assert_eq!(FilterOperator::Equals.to_sql(), "=");
        assert_eq!(FilterOperator::GreaterThan.to_sql(), ">");
        assert_eq!(FilterOperator::Contains.to_sql(), "LIKE");
        assert_eq!(FilterOperator::In.to_sql(), "IN");
    }

    #[test]
    fn test_smart_collection_query() {
        let query = SmartCollectionQuery {
            filters: vec![QueryFilter {
                field: "mime_type".to_string(),
                operator: FilterOperator::Equals,
                value: serde_json::json!("video/mp4"),
            }],
            sort_by: Some("created_at".to_string()),
            sort_direction: Some("DESC".to_string()),
            limit: Some(100),
        };

        assert_eq!(query.filters.len(), 1);
        assert_eq!(query.sort_by, Some("created_at".to_string()));
    }

    #[test]
    fn test_query_filter() {
        let filter = QueryFilter {
            field: "duration_ms".to_string(),
            operator: FilterOperator::GreaterThan,
            value: serde_json::json!(5000),
        };

        assert_eq!(filter.field, "duration_ms");
        assert_eq!(filter.operator, FilterOperator::GreaterThan);
    }
}

//! Tag management with hierarchical tags
//!
//! Provides comprehensive tag management for:
//! - Hierarchical tag structure
//! - Tag autocomplete
//! - Tag synonyms
//! - Bulk tagging
//! - Tag usage statistics

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::database::Database;
use crate::{MamError, Result};

/// Tag manager handles tag operations
pub struct TagManager {
    db: Arc<Database>,
    /// Cache of tag hierarchy for fast lookup
    tag_cache: Arc<RwLock<HashMap<Uuid, Tag>>>,
    /// Tag name to ID mapping
    name_cache: Arc<RwLock<HashMap<String, Uuid>>>,
}

/// Tag with hierarchical structure
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Tag {
    pub id: Uuid,
    pub name: String,
    pub slug: String,
    pub description: Option<String>,
    pub parent_id: Option<Uuid>,
    pub color: Option<String>,
    pub icon: Option<String>,
    pub usage_count: i32,
    pub created_by: Option<Uuid>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Tag synonym
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct TagSynonym {
    pub id: Uuid,
    pub tag_id: Uuid,
    pub synonym: String,
    pub created_at: DateTime<Utc>,
}

/// Asset tag association
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct AssetTag {
    pub asset_id: Uuid,
    pub tag_id: Uuid,
    pub added_by: Option<Uuid>,
    pub created_at: DateTime<Utc>,
}

/// Tag creation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateTagRequest {
    pub name: String,
    pub description: Option<String>,
    pub parent_id: Option<Uuid>,
    pub color: Option<String>,
    pub icon: Option<String>,
}

/// Tag hierarchy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagHierarchy {
    pub tag: Tag,
    pub children: Vec<TagHierarchy>,
    pub path: Vec<String>,
}

/// Tag statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagStatistics {
    pub tag_id: Uuid,
    pub tag_name: String,
    pub usage_count: i32,
    pub asset_count: i32,
    pub last_used: Option<DateTime<Utc>>,
}

impl TagManager {
    /// Create a new tag manager
    #[must_use]
    pub fn new(db: Arc<Database>) -> Self {
        Self {
            db,
            tag_cache: Arc::new(RwLock::new(HashMap::new())),
            name_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new tag
    ///
    /// # Errors
    ///
    /// Returns an error if creation fails
    pub async fn create_tag(&self, req: CreateTagRequest, created_by: Option<Uuid>) -> Result<Tag> {
        // Generate slug from name
        let slug = Self::slugify(&req.name);

        // Check if slug already exists
        let existing = sqlx::query_as::<_, Tag>("SELECT * FROM tags WHERE slug = $1")
            .bind(&slug)
            .fetch_optional(self.db.pool())
            .await?;

        if existing.is_some() {
            return Err(MamError::InvalidInput(format!(
                "Tag with slug '{}' already exists",
                slug
            )));
        }

        // Validate parent exists if specified
        if let Some(parent_id) = req.parent_id {
            self.get_tag(parent_id).await?;
        }

        let tag = sqlx::query_as::<_, Tag>(
            "INSERT INTO tags
             (id, name, slug, description, parent_id, color, icon, usage_count, created_by, created_at, updated_at)
             VALUES ($1, $2, $3, $4, $5, $6, $7, 0, $8, NOW(), NOW())
             RETURNING *",
        )
        .bind(Uuid::new_v4())
        .bind(&req.name)
        .bind(&slug)
        .bind(&req.description)
        .bind(req.parent_id)
        .bind(&req.color)
        .bind(&req.icon)
        .bind(created_by)
        .fetch_one(self.db.pool())
        .await?;

        // Update cache
        self.tag_cache.write().await.insert(tag.id, tag.clone());
        self.name_cache
            .write()
            .await
            .insert(tag.name.clone(), tag.id);

        Ok(tag)
    }

    /// Get tag by ID
    ///
    /// # Errors
    ///
    /// Returns an error if tag not found
    pub async fn get_tag(&self, tag_id: Uuid) -> Result<Tag> {
        // Check cache first
        {
            let cache = self.tag_cache.read().await;
            if let Some(tag) = cache.get(&tag_id) {
                return Ok(tag.clone());
            }
        }

        // Load from database
        let tag = sqlx::query_as::<_, Tag>("SELECT * FROM tags WHERE id = $1")
            .bind(tag_id)
            .fetch_one(self.db.pool())
            .await?;

        // Update cache
        self.tag_cache.write().await.insert(tag.id, tag.clone());

        Ok(tag)
    }

    /// Get tag by name
    ///
    /// # Errors
    ///
    /// Returns an error if tag not found
    pub async fn get_tag_by_name(&self, name: &str) -> Result<Tag> {
        // Check name cache first
        {
            let cache = self.name_cache.read().await;
            if let Some(tag_id) = cache.get(name) {
                return self.get_tag(*tag_id).await;
            }
        }

        // Load from database
        let tag = sqlx::query_as::<_, Tag>("SELECT * FROM tags WHERE name = $1")
            .bind(name)
            .fetch_one(self.db.pool())
            .await?;

        // Update cache
        self.tag_cache.write().await.insert(tag.id, tag.clone());
        self.name_cache
            .write()
            .await
            .insert(tag.name.clone(), tag.id);

        Ok(tag)
    }

    /// Get tag by slug
    ///
    /// # Errors
    ///
    /// Returns an error if tag not found
    pub async fn get_tag_by_slug(&self, slug: &str) -> Result<Tag> {
        let tag = sqlx::query_as::<_, Tag>("SELECT * FROM tags WHERE slug = $1")
            .bind(slug)
            .fetch_one(self.db.pool())
            .await?;

        // Update cache
        self.tag_cache.write().await.insert(tag.id, tag.clone());
        self.name_cache
            .write()
            .await
            .insert(tag.name.clone(), tag.id);

        Ok(tag)
    }

    /// Update tag
    ///
    /// # Errors
    ///
    /// Returns an error if update fails
    pub async fn update_tag(
        &self,
        tag_id: Uuid,
        name: Option<String>,
        description: Option<String>,
        parent_id: Option<Option<Uuid>>,
        color: Option<String>,
        icon: Option<String>,
    ) -> Result<Tag> {
        let current = self.get_tag(tag_id).await?;

        let new_name = name.unwrap_or(current.name.clone());
        let new_slug = Self::slugify(&new_name);

        let tag = sqlx::query_as::<_, Tag>(
            "UPDATE tags SET
                name = $2,
                slug = $3,
                description = COALESCE($4, description),
                parent_id = COALESCE($5, parent_id),
                color = COALESCE($6, color),
                icon = COALESCE($7, icon),
                updated_at = NOW()
             WHERE id = $1
             RETURNING *",
        )
        .bind(tag_id)
        .bind(&new_name)
        .bind(&new_slug)
        .bind(description)
        .bind(parent_id)
        .bind(color)
        .bind(icon)
        .fetch_one(self.db.pool())
        .await?;

        // Clear cache
        self.clear_cache().await;

        Ok(tag)
    }

    /// Delete tag
    ///
    /// # Errors
    ///
    /// Returns an error if deletion fails
    pub async fn delete_tag(&self, tag_id: Uuid) -> Result<()> {
        // Check if tag has children
        let children: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM tags WHERE parent_id = $1")
            .bind(tag_id)
            .fetch_one(self.db.pool())
            .await?;

        if children > 0 {
            return Err(MamError::InvalidInput(
                "Cannot delete tag with children".to_string(),
            ));
        }

        // Delete tag
        sqlx::query("DELETE FROM tags WHERE id = $1")
            .bind(tag_id)
            .execute(self.db.pool())
            .await?;

        // Clear cache
        self.clear_cache().await;

        Ok(())
    }

    /// List all tags
    ///
    /// # Errors
    ///
    /// Returns an error if query fails
    pub async fn list_tags(&self) -> Result<Vec<Tag>> {
        let tags = sqlx::query_as::<_, Tag>("SELECT * FROM tags ORDER BY name")
            .fetch_all(self.db.pool())
            .await?;

        Ok(tags)
    }

    /// Get tag hierarchy
    ///
    /// # Errors
    ///
    /// Returns an error if query fails
    pub async fn get_tag_hierarchy(&self) -> Result<Vec<TagHierarchy>> {
        let tags = self.list_tags().await?;

        let tag_map: HashMap<Uuid, Tag> = tags.iter().map(|t| (t.id, t.clone())).collect();

        let mut root_tags: Vec<Tag> = tags
            .iter()
            .filter(|t| t.parent_id.is_none())
            .cloned()
            .collect();

        root_tags.sort_by(|a, b| a.name.cmp(&b.name));

        let hierarchy: Vec<TagHierarchy> = root_tags
            .iter()
            .map(|t| Self::build_hierarchy(t.clone(), &tag_map, Vec::new()))
            .collect();

        Ok(hierarchy)
    }

    fn build_hierarchy(tag: Tag, tag_map: &HashMap<Uuid, Tag>, path: Vec<String>) -> TagHierarchy {
        let mut current_path = path.clone();
        current_path.push(tag.name.clone());

        let children: Vec<TagHierarchy> = tag_map
            .values()
            .filter(|t| t.parent_id == Some(tag.id))
            .cloned()
            .map(|child| Self::build_hierarchy(child, tag_map, current_path.clone()))
            .collect();

        TagHierarchy {
            tag,
            children,
            path: current_path,
        }
    }

    /// Add tag to asset
    ///
    /// # Errors
    ///
    /// Returns an error if operation fails
    pub async fn add_tag_to_asset(
        &self,
        asset_id: Uuid,
        tag_id: Uuid,
        added_by: Option<Uuid>,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO asset_tags (asset_id, tag_id, added_by, created_at)
             VALUES ($1, $2, $3, NOW())
             ON CONFLICT (asset_id, tag_id) DO NOTHING",
        )
        .bind(asset_id)
        .bind(tag_id)
        .bind(added_by)
        .execute(self.db.pool())
        .await?;

        // Increment usage count
        sqlx::query(
            "UPDATE tags SET usage_count = usage_count + 1, updated_at = NOW() WHERE id = $1",
        )
        .bind(tag_id)
        .execute(self.db.pool())
        .await?;

        Ok(())
    }

    /// Remove tag from asset
    ///
    /// # Errors
    ///
    /// Returns an error if operation fails
    pub async fn remove_tag_from_asset(&self, asset_id: Uuid, tag_id: Uuid) -> Result<()> {
        let result = sqlx::query("DELETE FROM asset_tags WHERE asset_id = $1 AND tag_id = $2")
            .bind(asset_id)
            .bind(tag_id)
            .execute(self.db.pool())
            .await?;

        if result.rows_affected() > 0 {
            // Decrement usage count
            sqlx::query(
                "UPDATE tags SET usage_count = GREATEST(usage_count - 1, 0), updated_at = NOW() WHERE id = $1",
            )
            .bind(tag_id)
            .execute(self.db.pool())
            .await?;
        }

        Ok(())
    }

    /// Get tags for asset
    ///
    /// # Errors
    ///
    /// Returns an error if query fails
    pub async fn get_asset_tags(&self, asset_id: Uuid) -> Result<Vec<Tag>> {
        let tags = sqlx::query_as::<_, Tag>(
            "SELECT t.* FROM tags t
             INNER JOIN asset_tags at ON t.id = at.tag_id
             WHERE at.asset_id = $1
             ORDER BY t.name",
        )
        .bind(asset_id)
        .fetch_all(self.db.pool())
        .await?;

        Ok(tags)
    }

    /// Bulk tag assets
    ///
    /// # Errors
    ///
    /// Returns an error if operation fails
    pub async fn bulk_tag_assets(
        &self,
        asset_ids: Vec<Uuid>,
        tag_ids: Vec<Uuid>,
        added_by: Option<Uuid>,
    ) -> Result<()> {
        for asset_id in &asset_ids {
            for tag_id in &tag_ids {
                self.add_tag_to_asset(*asset_id, *tag_id, added_by).await?;
            }
        }

        Ok(())
    }

    /// Search tags by name (autocomplete)
    ///
    /// # Errors
    ///
    /// Returns an error if query fails
    pub async fn search_tags(&self, query: &str, limit: i64) -> Result<Vec<Tag>> {
        let pattern = format!("%{query}%");

        let tags = sqlx::query_as::<_, Tag>(
            "SELECT * FROM tags
             WHERE name ILIKE $1
             ORDER BY usage_count DESC, name
             LIMIT $2",
        )
        .bind(&pattern)
        .bind(limit)
        .fetch_all(self.db.pool())
        .await?;

        Ok(tags)
    }

    /// Add tag synonym
    ///
    /// # Errors
    ///
    /// Returns an error if operation fails
    pub async fn add_synonym(&self, tag_id: Uuid, synonym: String) -> Result<TagSynonym> {
        let syn = sqlx::query_as::<_, TagSynonym>(
            "INSERT INTO tag_synonyms (id, tag_id, synonym, created_at)
             VALUES ($1, $2, $3, NOW())
             RETURNING *",
        )
        .bind(Uuid::new_v4())
        .bind(tag_id)
        .bind(&synonym)
        .fetch_one(self.db.pool())
        .await?;

        Ok(syn)
    }

    /// Get tag synonyms
    ///
    /// # Errors
    ///
    /// Returns an error if query fails
    pub async fn get_synonyms(&self, tag_id: Uuid) -> Result<Vec<TagSynonym>> {
        let synonyms = sqlx::query_as::<_, TagSynonym>(
            "SELECT * FROM tag_synonyms WHERE tag_id = $1 ORDER BY synonym",
        )
        .bind(tag_id)
        .fetch_all(self.db.pool())
        .await?;

        Ok(synonyms)
    }

    /// Get tag statistics
    ///
    /// # Errors
    ///
    /// Returns an error if query fails
    pub async fn get_tag_statistics(&self, tag_id: Uuid) -> Result<TagStatistics> {
        let tag = self.get_tag(tag_id).await?;

        let asset_count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM asset_tags WHERE tag_id = $1")
                .bind(tag_id)
                .fetch_one(self.db.pool())
                .await?;

        let last_used: Option<DateTime<Utc>> =
            sqlx::query_scalar("SELECT MAX(created_at) FROM asset_tags WHERE tag_id = $1")
                .bind(tag_id)
                .fetch_one(self.db.pool())
                .await?;

        Ok(TagStatistics {
            tag_id,
            tag_name: tag.name,
            usage_count: tag.usage_count,
            asset_count: asset_count as i32,
            last_used,
        })
    }

    /// Get most used tags
    ///
    /// # Errors
    ///
    /// Returns an error if query fails
    pub async fn get_most_used_tags(&self, limit: i64) -> Result<Vec<TagStatistics>> {
        let tags =
            sqlx::query_as::<_, Tag>("SELECT * FROM tags ORDER BY usage_count DESC LIMIT $1")
                .bind(limit)
                .fetch_all(self.db.pool())
                .await?;

        let mut stats = Vec::new();
        for tag in tags {
            if let Ok(stat) = self.get_tag_statistics(tag.id).await {
                stats.push(stat);
            }
        }

        Ok(stats)
    }

    /// Generate slug from name
    fn slugify(name: &str) -> String {
        name.to_lowercase()
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '-' {
                    c
                } else if c.is_whitespace() {
                    '-'
                } else {
                    '_'
                }
            })
            .collect::<String>()
            .trim_matches('-')
            .to_string()
    }

    /// Clear tag cache
    pub async fn clear_cache(&self) {
        self.tag_cache.write().await.clear();
        self.name_cache.write().await.clear();
    }

    /// Load all tags into cache
    ///
    /// # Errors
    ///
    /// Returns an error if query fails
    pub async fn load_cache(&self) -> Result<()> {
        let tags = self.list_tags().await?;

        let mut tag_cache = self.tag_cache.write().await;
        let mut name_cache = self.name_cache.write().await;

        tag_cache.clear();
        name_cache.clear();

        for tag in tags {
            tag_cache.insert(tag.id, tag.clone());
            name_cache.insert(tag.name.clone(), tag.id);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slugify() {
        assert_eq!(TagManager::slugify("Hello World"), "hello-world");
        assert_eq!(TagManager::slugify("Test Tag 123"), "test-tag-123");
        assert_eq!(TagManager::slugify("C++ Programming"), "c__-programming");
        assert_eq!(TagManager::slugify("  Spaces  "), "spaces");
    }

    #[test]
    fn test_create_tag_request() {
        let req = CreateTagRequest {
            name: "Documentary".to_string(),
            description: Some("Documentary films".to_string()),
            parent_id: None,
            color: Some("#FF0000".to_string()),
            icon: Some("film".to_string()),
        };

        assert_eq!(req.name, "Documentary");
        assert_eq!(req.color, Some("#FF0000".to_string()));
    }

    #[test]
    fn test_tag_statistics() {
        let stats = TagStatistics {
            tag_id: Uuid::new_v4(),
            tag_name: "Test".to_string(),
            usage_count: 100,
            asset_count: 50,
            last_used: Some(Utc::now()),
        };

        assert_eq!(stats.tag_name, "Test");
        assert_eq!(stats.usage_count, 100);
        assert_eq!(stats.asset_count, 50);
    }
}

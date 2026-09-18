//! Collection and playlist models.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Media collection (playlist).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Collection {
    /// Unique collection ID
    pub id: String,
    /// Owner user ID
    pub user_id: String,
    /// Collection name
    pub name: String,
    /// Description
    pub description: Option<String>,
    /// Thumbnail path
    pub thumbnail_path: Option<String>,
    /// Creation timestamp
    pub created_at: i64,
    /// Last update timestamp
    pub updated_at: i64,
}

impl Collection {
    /// Creates a new collection.
    #[must_use]
    pub fn new(user_id: String, name: String, description: Option<String>) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            id: Uuid::new_v4().to_string(),
            user_id,
            name,
            description,
            thumbnail_path: None,
            created_at: now,
            updated_at: now,
        }
    }
}

/// Item in a collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionItem {
    /// Collection ID
    pub collection_id: String,
    /// Media ID
    pub media_id: String,
    /// Position in collection (for ordering)
    pub position: i32,
    /// Timestamp when added
    pub added_at: i64,
}

impl CollectionItem {
    /// Creates a new collection item.
    #[must_use]
    pub fn new(collection_id: String, media_id: String, position: i32) -> Self {
        Self {
            collection_id,
            media_id,
            position,
            added_at: chrono::Utc::now().timestamp(),
        }
    }
}

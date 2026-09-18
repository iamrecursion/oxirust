//! Collections and playlists API endpoints.

use crate::{
    auth::AuthUser,
    error::{ServerError, ServerResult},
    models::collection::{Collection, CollectionItem},
    AppState,
};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Create collection request.
#[derive(Debug, Deserialize)]
pub struct CreateCollectionRequest {
    /// Collection name
    pub name: String,
    /// Description
    pub description: Option<String>,
}

/// Update collection request.
#[derive(Debug, Deserialize)]
pub struct UpdateCollectionRequest {
    /// New name
    pub name: Option<String>,
    /// New description
    pub description: Option<String>,
}

/// Add item request.
#[derive(Debug, Deserialize)]
pub struct AddItemRequest {
    /// Media ID
    pub media_id: String,
    /// Position (optional, defaults to end)
    pub position: Option<i32>,
}

/// Collection with items.
#[derive(Debug, Serialize)]
pub struct CollectionWithItems {
    /// Collection info
    #[serde(flatten)]
    pub collection: Collection,
    /// Items in the collection
    pub items: Vec<CollectionItem>,
}

/// Lists collections for the current user.
pub async fn list_collections(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
) -> ServerResult<impl IntoResponse> {
    let rows = crate::db::query(
        r"
        SELECT * FROM collections
        WHERE user_id = ?
        ORDER BY created_at DESC
        ",
    )
    .bind(&auth_user.user_id)
    .fetch_all(state.db.pool())
    .await?;

    let collections: Vec<Collection> = rows
        .iter()
        .map(|row| Collection {
            id: row.get("id"),
            user_id: row.get("user_id"),
            name: row.get("name"),
            description: row.get("description"),
            thumbnail_path: row.get("thumbnail_path"),
            created_at: row.get("created_at"),
            updated_at: row.get("updated_at"),
        })
        .collect();

    Ok(Json(collections))
}

/// Creates a new collection.
pub async fn create_collection(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
    Json(req): Json<CreateCollectionRequest>,
) -> ServerResult<impl IntoResponse> {
    if req.name.is_empty() {
        return Err(ServerError::BadRequest(
            "Collection name is required".to_string(),
        ));
    }

    let collection = Collection::new(auth_user.user_id, req.name, req.description);

    crate::db::query(
        r"
        INSERT INTO collections (id, user_id, name, description, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ",
    )
    .bind(&collection.id)
    .bind(&collection.user_id)
    .bind(&collection.name)
    .bind(&collection.description)
    .bind(collection.created_at)
    .bind(collection.updated_at)
    .execute(state.db.pool())
    .await?;

    Ok((StatusCode::CREATED, Json(collection)))
}

/// Gets a collection by ID with its items.
pub async fn get_collection(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
    Path(collection_id): Path<String>,
) -> ServerResult<impl IntoResponse> {
    // Get collection
    let row = crate::db::query(
        r"
        SELECT * FROM collections WHERE id = ?
        ",
    )
    .bind(&collection_id)
    .fetch_one(state.db.pool())
    .await
    .map_err(|_| ServerError::NotFound(format!("Collection not found: {}", collection_id)))?;

    let user_id: String = row.get("user_id");

    // Verify ownership
    if user_id != auth_user.user_id && !auth_user.is_admin() {
        return Err(ServerError::Forbidden(
            "Access denied to this collection".to_string(),
        ));
    }

    let collection = Collection {
        id: row.get("id"),
        user_id: row.get("user_id"),
        name: row.get("name"),
        description: row.get("description"),
        thumbnail_path: row.get("thumbnail_path"),
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
    };

    // Get items
    let item_rows = crate::db::query(
        r"
        SELECT * FROM collection_items
        WHERE collection_id = ?
        ORDER BY position
        ",
    )
    .bind(&collection_id)
    .fetch_all(state.db.pool())
    .await?;

    let items: Vec<CollectionItem> = item_rows
        .iter()
        .map(|row| CollectionItem {
            collection_id: row.get("collection_id"),
            media_id: row.get("media_id"),
            position: row.get("position"),
            added_at: row.get("added_at"),
        })
        .collect();

    Ok(Json(CollectionWithItems { collection, items }))
}

/// Updates a collection.
pub async fn update_collection(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
    Path(collection_id): Path<String>,
    Json(req): Json<UpdateCollectionRequest>,
) -> ServerResult<impl IntoResponse> {
    // Verify ownership
    let user_id: String = crate::db::query_scalar("SELECT user_id FROM collections WHERE id = ?")
        .bind(&collection_id)
        .fetch_one(state.db.pool())
        .await
        .map_err(|_| ServerError::NotFound(format!("Collection not found: {}", collection_id)))?;

    if user_id != auth_user.user_id && !auth_user.is_admin() {
        return Err(ServerError::Forbidden(
            "Access denied to this collection".to_string(),
        ));
    }

    // Update fields
    if let Some(name) = &req.name {
        crate::db::query("UPDATE collections SET name = ?, updated_at = ? WHERE id = ?")
            .bind(name)
            .bind(chrono::Utc::now().timestamp())
            .bind(&collection_id)
            .execute(state.db.pool())
            .await?;
    }

    if let Some(description) = &req.description {
        crate::db::query("UPDATE collections SET description = ?, updated_at = ? WHERE id = ?")
            .bind(description)
            .bind(chrono::Utc::now().timestamp())
            .bind(&collection_id)
            .execute(state.db.pool())
            .await?;
    }

    Ok(StatusCode::NO_CONTENT)
}

/// Deletes a collection.
pub async fn delete_collection(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
    Path(collection_id): Path<String>,
) -> ServerResult<impl IntoResponse> {
    // Verify ownership
    let user_id: String = crate::db::query_scalar("SELECT user_id FROM collections WHERE id = ?")
        .bind(&collection_id)
        .fetch_one(state.db.pool())
        .await
        .map_err(|_| ServerError::NotFound(format!("Collection not found: {}", collection_id)))?;

    if user_id != auth_user.user_id && !auth_user.is_admin() {
        return Err(ServerError::Forbidden(
            "Access denied to this collection".to_string(),
        ));
    }

    // Delete collection (cascades to items)
    crate::db::query("DELETE FROM collections WHERE id = ?")
        .bind(&collection_id)
        .execute(state.db.pool())
        .await?;

    Ok(StatusCode::NO_CONTENT)
}

/// Adds an item to a collection.
pub async fn add_item(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
    Path(collection_id): Path<String>,
    Json(req): Json<AddItemRequest>,
) -> ServerResult<impl IntoResponse> {
    // Verify collection ownership
    let user_id: String = crate::db::query_scalar("SELECT user_id FROM collections WHERE id = ?")
        .bind(&collection_id)
        .fetch_one(state.db.pool())
        .await
        .map_err(|_| ServerError::NotFound(format!("Collection not found: {}", collection_id)))?;

    if user_id != auth_user.user_id && !auth_user.is_admin() {
        return Err(ServerError::Forbidden(
            "Access denied to this collection".to_string(),
        ));
    }

    // Verify media exists
    let _media = state.library.get_media(&req.media_id).await?;

    // Check if item already in collection
    let exists = crate::db::query_scalar::<i64, _>(
        "SELECT COUNT(*) FROM collection_items WHERE collection_id = ? AND media_id = ?",
    )
    .bind(&collection_id)
    .bind(&req.media_id)
    .fetch_one(state.db.pool())
    .await?;

    if exists > 0 {
        return Err(ServerError::Conflict(
            "Item already in collection".to_string(),
        ));
    }

    // Determine position
    let position = if let Some(pos) = req.position {
        pos
    } else {
        // Add to end
        let max_pos: Option<i32> = crate::db::query_scalar(
            "SELECT MAX(position) FROM collection_items WHERE collection_id = ?",
        )
        .bind(&collection_id)
        .fetch_one(state.db.pool())
        .await?;

        max_pos.unwrap_or(-1) + 1
    };

    let item = CollectionItem::new(collection_id, req.media_id, position);

    crate::db::query(
        r"
        INSERT INTO collection_items (collection_id, media_id, position, added_at)
        VALUES (?, ?, ?, ?)
        ",
    )
    .bind(&item.collection_id)
    .bind(&item.media_id)
    .bind(item.position)
    .bind(item.added_at)
    .execute(state.db.pool())
    .await?;

    Ok((StatusCode::CREATED, Json(item)))
}

/// Removes an item from a collection.
pub async fn remove_item(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
    Path((collection_id, media_id)): Path<(String, String)>,
) -> ServerResult<impl IntoResponse> {
    // Verify collection ownership
    let user_id: String = crate::db::query_scalar("SELECT user_id FROM collections WHERE id = ?")
        .bind(&collection_id)
        .fetch_one(state.db.pool())
        .await
        .map_err(|_| ServerError::NotFound(format!("Collection not found: {}", collection_id)))?;

    if user_id != auth_user.user_id && !auth_user.is_admin() {
        return Err(ServerError::Forbidden(
            "Access denied to this collection".to_string(),
        ));
    }

    // Remove item
    crate::db::query("DELETE FROM collection_items WHERE collection_id = ? AND media_id = ?")
        .bind(&collection_id)
        .bind(&media_id)
        .execute(state.db.pool())
        .await?;

    Ok(StatusCode::NO_CONTENT)
}

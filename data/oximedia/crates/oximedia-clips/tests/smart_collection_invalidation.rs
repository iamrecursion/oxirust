//! Integration tests for `SmartCollection` field-dependency auto-invalidation.
//!
//! When a clip's metadata changes via [`ClipManager::update_clip`], only the
//! smart collections whose rules depend on a *changed* field should have their
//! caches invalidated. Collections whose rules depend on unrelated fields keep
//! their cached results.
//!
//! These tests use `:memory:` SQLite (the established single-handle pattern used
//! throughout this crate's unit tests); no external DB server is required.

#![cfg(not(target_arch = "wasm32"))]

use oximedia_clips::{
    Clip, ClipManager, Comparison, MatchMode, Rating, SmartCollection, SmartRule,
};
use std::path::PathBuf;

/// Builds a fresh in-memory manager.
async fn make_manager() -> ClipManager {
    ClipManager::new(":memory:")
        .await
        .expect("ClipManager::new should succeed on :memory:")
}

/// Registers a single-rating-rule smart collection and returns nothing; the
/// caller inspects it via `update_smart_collections` + a fresh read is not
/// available, so tests re-register and re-read through the manager's own API.
fn rating_collection() -> SmartCollection {
    SmartCollection::new(
        "High Rated",
        vec![SmartRule::Rating {
            operator: Comparison::GreaterThanOrEqual,
            value: Rating::FourStars,
        }],
        MatchMode::All,
    )
}

#[tokio::test]
async fn test_update_clip_invalidates_dependent_collection() {
    let manager = make_manager().await;

    // Add a clip (rating-relevant).
    let mut clip = Clip::new(PathBuf::from("/footage/a.mov"));
    clip.set_rating(Rating::FiveStars);
    let clip_id = manager
        .add_clip(clip)
        .await
        .expect("add_clip should succeed");

    // Register a Rating-dependent smart collection.
    manager
        .create_smart_collection(rating_collection())
        .expect("create_smart_collection should succeed");

    // Populate the cache: after this, the collection is valid.
    manager
        .update_smart_collections()
        .await
        .expect("update_smart_collections should succeed");

    // Sanity: cache is valid right after the update.
    manager
        .with_smart_collections(|cols| {
            let col = &cols[0];
            assert!(
                !col.needs_refresh(),
                "collection should be fresh after update_smart_collections"
            );
            assert!(
                col.cached_clip_ids().is_some(),
                "cached ids should be present after update"
            );
        })
        .expect("inspect should succeed");

    // Change the clip's rating → Rating field changed → collection invalidated.
    let mut updated = manager
        .get_clip(&clip_id)
        .await
        .expect("get_clip should succeed");
    updated.set_rating(Rating::OneStar);
    manager
        .update_clip(updated)
        .await
        .expect("update_clip should succeed");

    manager
        .with_smart_collections(|cols| {
            let col = &cols[0];
            assert!(
                col.needs_refresh(),
                "rating-dependent collection must need refresh after rating change"
            );
            assert!(
                col.cached_clip_ids().is_none(),
                "cache must be cleared after invalidation"
            );
        })
        .expect("inspect should succeed");
}

#[tokio::test]
async fn test_unrelated_field_change_keeps_cache_valid() {
    let manager = make_manager().await;

    let mut clip = Clip::new(PathBuf::from("/footage/b.mov"));
    clip.set_rating(Rating::FiveStars);
    let clip_id = manager
        .add_clip(clip)
        .await
        .expect("add_clip should succeed");

    // Rating-dependent collection.
    manager
        .create_smart_collection(rating_collection())
        .expect("create_smart_collection should succeed");
    manager
        .update_smart_collections()
        .await
        .expect("update_smart_collections should succeed");

    // Change ONLY the name (does not affect the Rating rule).
    let mut updated = manager
        .get_clip(&clip_id)
        .await
        .expect("get_clip should succeed");
    updated.set_name("Renamed Clip");
    manager
        .update_clip(updated)
        .await
        .expect("update_clip should succeed");

    manager
        .with_smart_collections(|cols| {
            let col = &cols[0];
            assert!(
                !col.needs_refresh(),
                "name change must NOT invalidate a rating-only collection"
            );
            assert!(
                col.cached_clip_ids().is_some(),
                "cache must remain valid after unrelated field change"
            );
        })
        .expect("inspect should succeed");
}

#[tokio::test]
async fn test_keyword_collection_invalidated_on_add_keyword() {
    let manager = make_manager().await;

    let clip = Clip::new(PathBuf::from("/footage/c.mov"));
    let clip_id = manager
        .add_clip(clip)
        .await
        .expect("add_clip should succeed");

    // Keyword-dependent collection.
    let keyword_col = SmartCollection::new(
        "Interviews",
        vec![SmartRule::Keyword {
            keyword: "interview".to_string(),
        }],
        MatchMode::All,
    );
    manager
        .create_smart_collection(keyword_col)
        .expect("create_smart_collection should succeed");
    manager
        .update_smart_collections()
        .await
        .expect("update_smart_collections should succeed");

    // Confirm fresh before the change.
    manager
        .with_smart_collections(|cols| {
            assert!(!cols[0].needs_refresh());
        })
        .expect("inspect should succeed");

    // Add a keyword → Keywords field changed → collection invalidated.
    let mut updated = manager
        .get_clip(&clip_id)
        .await
        .expect("get_clip should succeed");
    updated.add_keyword("interview");
    manager
        .update_clip(updated)
        .await
        .expect("update_clip should succeed");

    manager
        .with_smart_collections(|cols| {
            let col = &cols[0];
            assert!(
                col.needs_refresh(),
                "keyword-dependent collection must need refresh after add_keyword"
            );
            assert!(col.cached_clip_ids().is_none());
        })
        .expect("inspect should succeed");
}

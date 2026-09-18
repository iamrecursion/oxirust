//! Smoke tests verifying that all 16 orphan modules are reachable and
//! that their core types compile and behave correctly.

// ── asset_versioning ─────────────────────────────────────────────────────────

#[test]
fn test_asset_versioning_semantic_version() {
    use oximedia_mam::asset_versioning::SemanticVersion;
    let v = SemanticVersion::new(1, 2, 3);
    assert_eq!(v.major, 1);
    assert_eq!(v.minor, 2);
    assert_eq!(v.patch, 3);
    let bumped = v.bump_patch();
    assert_eq!(bumped.patch, 4);
}

#[test]
fn test_asset_versioning_tree_empty() {
    use oximedia_mam::asset_versioning::VersionTree;
    let tree = VersionTree::new(uuid::Uuid::new_v4());
    assert_eq!(tree.len(), 0);
}

// ── materialized_path ────────────────────────────────────────────────────────

#[test]
fn test_materialized_path_tree_empty() {
    use oximedia_mam::materialized_path::PathTree;
    let tree = PathTree::new();
    assert_eq!(tree.len(), 0);
    assert!(tree.is_empty());
}

#[test]
fn test_materialized_path_insert_root() {
    use oximedia_mam::materialized_path::PathTree;
    let mut tree = PathTree::new();
    tree.insert_root("root-1", "Root Node 1")
        .expect("insert ok");
    assert_eq!(tree.len(), 1);
    assert_eq!(tree.nodes_at_depth(0).len(), 1);
}

// ── version_compare ──────────────────────────────────────────────────────────

#[test]
fn test_version_compare_field_change_types() {
    use oximedia_mam::version_compare::FieldChange;
    let addition = FieldChange::new("genre", None, Some("drama".to_string()));
    assert!(addition.is_addition());
    assert!(!addition.is_removal());
    let removal = FieldChange::new("rating", Some("PG".to_string()), None);
    assert!(removal.is_removal());
    let modification = FieldChange::new("title", Some("Old".to_string()), Some("New".to_string()));
    assert!(modification.is_modification());
}

// ── relationship ─────────────────────────────────────────────────────────────

#[test]
fn test_relationship_graph_add_and_query() {
    use oximedia_mam::relationship::{AssetRelationship, RelationshipGraph};
    let mut g = RelationshipGraph::new();
    g.add(AssetRelationship::new(1, 2, "derived_from"));
    g.add(AssetRelationship::new(1, 3, "part_of"));
    let rels = g.related_to(1);
    assert_eq!(rels.len(), 2);
    assert_eq!(g.len(), 2);
}

// ── incremental_index ────────────────────────────────────────────────────────

#[test]
fn test_incremental_index_wal_push_and_drain() {
    use oximedia_mam::incremental_index::{IndexDocument, IndexOperation, IndexWal};
    let wal = IndexWal::new();
    let mut doc = IndexDocument::new("asset-1");
    doc.add_field("title", "Test Asset");
    wal.push(IndexOperation::Upsert(doc)).expect("push ok");
    let batch = wal.drain(10).expect("drain ok");
    assert_eq!(batch.len(), 1);
}

// ── asset_cache ──────────────────────────────────────────────────────────────

#[test]
fn test_asset_cache_insert_and_get() {
    use oximedia_mam::asset_cache::{AssetCache, AssetCacheKey, CachePolicy};
    use std::time::Duration;
    let policy = CachePolicy {
        ttl: Duration::from_secs(60),
        max_entries: 10,
        sliding_ttl: false,
    };
    let mut cache = AssetCache::new(policy);
    let key = AssetCacheKey::AssetById(uuid::Uuid::new_v4().to_string());
    cache.insert(key.clone(), serde_json::json!({"name": "test"}));
    assert!(cache.get(&key).is_some());
    assert_eq!(cache.len(), 1);
}

// ── metadata_schema ──────────────────────────────────────────────────────────

#[test]
fn test_metadata_schema_field_data_type_name() {
    use oximedia_mam::metadata_schema::FieldDataType;
    assert_eq!(FieldDataType::String.type_name(), "string");
    assert_eq!(FieldDataType::Integer.type_name(), "integer");
    assert_eq!(FieldDataType::Boolean.type_name(), "boolean");
}

// ── media_rights_clearance ───────────────────────────────────────────────────

#[test]
fn test_media_rights_clearance_territory_display() {
    use oximedia_mam::media_rights_clearance::Territory;
    let t = Territory::Worldwide;
    assert_eq!(t.display(), "Worldwide");
    let us = Territory::Country("US".to_string());
    assert!(us.includes_country("US"));
    assert!(!us.includes_country("GB"));
}

// ── timeline_marker ──────────────────────────────────────────────────────────

#[test]
fn test_timeline_marker_kinds() {
    use oximedia_mam::timeline_marker::MarkerKind;
    assert_eq!(MarkerKind::Chapter.label(), "Chapter");
    assert_eq!(MarkerKind::Cue.label(), "Cue");
    assert_eq!(MarkerKind::AdBreak.label(), "Ad Break");
    let custom = MarkerKind::Custom("special".to_string());
    assert_eq!(custom.label(), "special"); // Custom variant returns the inner string
}

// ── batch_metadata ───────────────────────────────────────────────────────────

#[test]
fn test_batch_metadata_noop_executor() {
    use oximedia_mam::batch_metadata::{BatchBuffer, MetadataCommand, MetadataField, NoopExecutor};
    use uuid::Uuid;
    let executor = NoopExecutor::default();
    let mut buf = BatchBuffer::new(executor, 100);
    buf.push(MetadataCommand::Set {
        asset_id: Uuid::new_v4(),
        field: MetadataField::Title("Test".to_string()),
    });
    let stats = buf.flush().expect("flush ok");
    assert_eq!(stats.commands_executed, 1);
}

// ── asset_qc ─────────────────────────────────────────────────────────────────

#[test]
fn test_asset_qc_engine_broadcast_defaults() {
    use oximedia_mam::asset_qc::QcEngine;
    let engine = QcEngine::broadcast_defaults();
    assert!(engine.checker_count() > 0);
}

// ── pool_tuning ──────────────────────────────────────────────────────────────

#[test]
fn test_pool_tuning_metrics_construction() {
    use oximedia_mam::pool_tuning::PoolMetrics;
    let m = PoolMetrics::new(4, 6, 0, 20, 2);
    assert_eq!(m.active_connections, 4);
    assert_eq!(m.idle_connections, 6);
    assert_eq!(m.total_connections, 10);
}

// ── bulk_update ──────────────────────────────────────────────────────────────

#[test]
fn test_bulk_update_apply_to() {
    use oximedia_mam::bulk_update::{BulkMetadataUpdate, MetadataStore};
    let mut store = MetadataStore::new();
    let mut update = BulkMetadataUpdate::new();
    update.set("genre", "drama");
    update.set("status", "approved");
    let changed = update.apply_to(&[1, 2, 3], &mut store);
    assert_eq!(changed, 3);
    assert_eq!(store.get(1, "genre"), Some("drama"));
    assert_eq!(store.get(2, "status"), Some("approved"));
}

// ── workflow_approval ────────────────────────────────────────────────────────

#[test]
fn test_workflow_approval_quorum_policy() {
    use oximedia_mam::workflow_approval::QuorumPolicy;
    assert!(QuorumPolicy::AnyOne.is_met(5, 1));
    assert!(!QuorumPolicy::All.is_met(3, 2));
    assert!(QuorumPolicy::All.is_met(2, 2));
    assert!(QuorumPolicy::Majority.is_met(3, 2));
    assert!(!QuorumPolicy::Majority.is_met(3, 1));
}

#[test]
fn test_workflow_approval_approver_role() {
    use oximedia_mam::workflow_approval::ApproverRole;
    let role = ApproverRole::new("legal");
    assert_eq!(role.name(), "legal");
}

// ── search_warming ───────────────────────────────────────────────────────────

#[test]
fn test_search_warming_config_default() {
    use oximedia_mam::search_warming::WarmingConfig;
    let cfg = WarmingConfig::default();
    assert!(cfg.top_k_queries > 0);
    assert!(cfg.min_query_frequency > 0);
}

#[test]
fn test_search_warming_frequency_tracker() {
    use oximedia_mam::search_warming::QueryFrequencyTracker;
    let mut tracker = QueryFrequencyTracker::default();
    tracker.record("documentary films");
    tracker.record("documentary films");
    tracker.record("news archive");
    assert_eq!(tracker.count("documentary films"), 2);
    assert_eq!(tracker.count("news archive"), 1);
    assert_eq!(tracker.count("missing query"), 0);
}

// ── rights_link ──────────────────────────────────────────────────────────────

#[test]
fn test_rights_link_perpetual() {
    use oximedia_mam::rights_link::AssetRightsLink;
    let link = AssetRightsLink::new(42, 100);
    assert_eq!(link.asset_id, 42);
    assert_eq!(link.rights_id, 100);
    // Perpetual link is active at any timestamp
    assert!(link.is_active_at(0));
    assert!(link.is_active_at(u64::MAX));
}

#[test]
fn test_rights_link_with_expiry() {
    use oximedia_mam::rights_link::AssetRightsLink;
    let link = AssetRightsLink::new(1, 200).with_expiry(1_000);
    assert!(link.is_active_at(999));
    assert!(!link.is_active_at(1_000)); // at-expiry = expired
}

#[test]
fn test_rights_linker_active_filtering() {
    use oximedia_mam::rights_link::{AssetRightsLink, RightsLinker};
    let mut linker = RightsLinker::new();
    linker.link(AssetRightsLink::new(1, 100).with_expiry(500));
    linker.link(AssetRightsLink::new(2, 200)); // perpetual
    let active = linker.active_rights(1_000);
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].asset_id, 2);
}

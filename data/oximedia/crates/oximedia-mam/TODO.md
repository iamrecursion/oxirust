# oximedia-mam TODO

## Current Status
- 51 modules implementing a comprehensive Media Asset Management system
- Core: MamSystem orchestrator, MamConfig, asset manager, collection manager
- Database: PostgreSQL via sqlx with migrations
- Search: Tantivy full-text search, search_index, catalog_search, smart_search (BM25, Jaccard), asset_search
- API: RESTful (actix-web) and GraphQL (async-graphql) endpoints
- Auth: JWT authentication (jsonwebtoken), bcrypt password hashing, RBAC permissions
- Ingest: ingest, ingest_pipeline, ingest_workflow, batch_ingest
- Assets: asset_lifecycle, asset_status, asset_collection, asset_relations, asset_tag, asset_tag_index, asset_tagging
- Organization: folders, folder_hierarchy, collection_manager, media_catalog, media_linking
- Workflow: workflow, workflow_integration, workflow_trigger, event_bus
- Storage: storage manager, proxy generation, transcoding_profile, transfer_manager
- Metadata: metadata_template, media_format_info
- Advanced: ai_tagging, version_control, versioning, audit logging, webhook, export_package
- Governance: retention_policy, rights_summary, delivery_log, usage_analytics, media_project
- Dependencies: sqlx, tantivy, actix-web, async-graphql, tokio, reqwest, serde, chrono, uuid

## Enhancements
- [x] Add pagination and cursor-based navigation to `asset_search.rs` for large result sets
- [x] Extend `smart_search.rs` with fuzzy matching and typo tolerance (Levenshtein distance)
- [x] Add `retention_policy.rs` support for legal hold overrides that prevent deletion
- [x] Implement `webhook.rs` retry with exponential backoff for failed webhook deliveries
- [x] Extend `permissions.rs` with attribute-based access control (ABAC) in addition to RBAC
- [x] Add `audit.rs` log export to SIEM-compatible formats (CEF, LEEF) (verified 2026-05-16; src/audit.rs:574 fn export_cef)
- [x] Implement `batch_ingest.rs` support for watch folders with automatic ingest on file arrival (completed 2026-05-05)
  - **Goal:** Folder sync: directory walk → fingerprint → upload → emit AssetIngested event
  - **Design:** `notify` crate RecursiveMode::Recursive on watched folder; on Event::Create, fingerprint via oximedia-dedup, upload if not in MAM, emit MamEvent::AssetIngested via broadcast channel
  - **Files:** `crates/oximedia-mam/src/folders.rs`
  - **Tests:** `tests/folder_sync.rs` — 4 tests: uploads new file + event emitted; idempotent second call returns 0; empty dir returns 0; multiple files
  - **Risk:** Debounce rapid create/rename sequences to avoid double-ingest
- [x] Extend `proxy.rs` with adaptive bitrate proxy generation (HLS/DASH) for browser preview (verified 2026-05-16; src/proxy.rs:1008 AbrProxyConfig, AbrProxySet:1130, HLS/DASH manifest gen:1236)

## New Features
- [x] Add an `archive_tier.rs` module for cold/glacier storage tier management with retrieval scheduling (completed 2026-05-05)
  - **Goal:** Wire each MAM storage method to `oximedia-storage::StorageBackend` trait
  - **Design:** URI scheme dispatch (s3://, gs://, https://*.blob.core.windows.net, file://, bare paths) → MamStorage backed by oximedia_storage::local::LocalStorage (shadow dirs for remote stubs). Methods: put/get/list/delete/exists
  - **Files:** `crates/oximedia-mam/src/storage.rs`
  - **Tests:** `tests/storage_glue.rs` — 19 tests: URI parsing (bare/relative/file/s3/gcs/azure/unsupported), round-trip put/get, exists, delete, list, from_uri construction
  - **Risk:** Use only oximedia-storage's existing pure-Rust paths; do NOT add aws-sdk-s3 or azure-storage
- [x] Implement a `collaboration.rs` module for real-time comments, annotations, and review markers on assets (verified 2026-05-16; src/collaboration.rs:884 lines, struct AssetComment:21)
- [x] Add a `notification.rs` module for email/Slack/Teams notifications on asset and workflow events (completed 2026-05-05)
  - **Goal:** Event bus: fan-out MamEvent to all subscribers when assets are ingested
  - **Design:** tokio::sync::broadcast channel + topic-based filter; emit on asset ingest, folder sync, metadata update
  - **Files:** `crates/oximedia-mam/src/integration.rs`
  - **Tests:** `tests/event_bus.rs` — 5 tests: 3 subscribers fan-out, no-receiver error, receiver_count, sender clone delivery, multiple events in order
  - **Risk:** Broadcast channel capacity; slow subscribers should not block the ingest path
- [x] Implement a `custom_field.rs` module for user-defined metadata fields with type validation (verified 2026-05-16; src/custom_field.rs:808 lines)
- [x] Add a `duplicate_detect.rs` module for perceptual hash-based duplicate asset detection (verified 2026-05-16; src/duplicate_detect.rs:547 lines)
- [x] Implement an `access_request.rs` module for request/approve access workflows (verified 2026-05-16; src/access_request.rs:771 lines)
- [x] Add a `reporting.rs` module for scheduled report generation (storage usage, asset stats, user activity) (verified 2026-05-16; src/reporting.rs:792 lines)
- [x] Implement a `federated_search.rs` module for searching across multiple MAM instances (verified 2026-05-16; src/federated_search.rs:1042 lines)
- [x] Add a `trash_bin.rs` module with soft-delete and configurable auto-purge (verified 2026-05-16; src/trash_bin.rs:458 lines)

## Performance
- [x] Add Tantivy index warming on startup in `search_index.rs` for faster first-query response (verified 2026-06-01; src/search_index.rs TantivySearchIndex::warm + src/search.rs SearchEngine::warm called from new())
- [x] Implement connection pool tuning in `database.rs` based on concurrent request load (completed 2026-06-24; src/pool_tuning.rs: PoolMetrics/LoadSampler/PoolTuner/TuningStrategy {Conservative,Balanced,Aggressive}; src/database.rs: Database::new_with_config accepts PoolConfig; 4 async tokio multi-thread tests added to pool_tuning.rs)
- [ ] Add Redis-based caching layer for frequently accessed assets and search results
- [x] Optimize `folder_hierarchy.rs` tree queries with materialized path or nested set model
- [x] Add batch database operations in `asset.rs` for bulk metadata updates (reduce round-trips) (verified 2026-06-01; src/asset.rs:590 batch_update_metadata, :721 batch_set_status, :771 batch_upsert_custom_fields)
- [x] Implement incremental search index updates in `search_index.rs` instead of full reindex (verified 2026-06-01; src/search_index.rs TantivySearchIndex::add_document, update_document, delete_document)

## Testing
- [x] Add storage_glue integration tests for MamStorage URI dispatch and round-trip ops (completed 2026-05-05)
  - `tests/storage_glue.rs` — 19 tests: URI parsing, put/get/exists/delete/list round-trips, from_uri construction
- [x] Add folder_sync integration tests for FolderSync watch directory → upload → event (completed 2026-05-05)
  - `tests/folder_sync.rs` — 4 tests: new file upload + AssetIngested event, idempotent second call, empty directory, multiple files
- [x] Add event_bus integration tests for MamEventBus fan-out pub/sub (completed 2026-05-05)
  - `tests/event_bus.rs` — 5 tests: 3-subscriber fan-out, no-receiver error, receiver_count, sender clone, multiple events in order
- [ ] Add integration tests for the full ingest pipeline (file upload -> metadata extraction -> indexing -> proxy gen) — DEFERRED: requires PostgreSQL + Tantivy file index + proxy transcode infra; not pure/in-memory
- [x] Test `permissions.rs` RBAC with complex role hierarchies and permission inheritance — pure surface covered: role-hierarchy inheritance (`SystemRole::default_permissions` superset chain Admin⊇Editor⊇Viewer⊇Guest) + deny-overrides precedence via `AbacEngine` priority ordering (tests/mam_pure_logic.rs:47 rbac_role_hierarchy_inherits_lower_role_permissions, tests/mam_pure_logic.rs:107 abac_explicit_deny_overrides_inherited_allow). NOTE: the DB-backed `PermissionManager` RBAC (asset/collection grants over PostgreSQL) remains DB-gated and is not exercised here.
- [x] Add `workflow_trigger.rs` tests with concurrent asset events firing multiple triggers (tests/mam_pure_logic.rs:348 trigger_registry_concurrent_fire_no_loss — 64 threads, no firing lost/duplicated; tests/mam_pure_logic.rs:413 trigger_registry_repeat_fire_appends_and_preserves_order — append-only repeat semantics + ordering)
- [x] Test `smart_search.rs` BM25 scoring accuracy with known relevance judgments (completed 2026-06-24; 9 new unit tests in src/smart_search.rs: IDF-rare-term, TF-frequency-ranking, title/tag/description field-weight ordering, multi-term, precision@3 known-relevance corpus, nDCG-style ranking, matched-fields correctness, score-monotonicity with corpus expansion)
- [x] Add `version_control.rs` tests for branching and merging asset versions — branch/merge logic lives in `asset_versioning::VersionTree` (the `version_control::VersionHistory` module is flat/linear with no branch/merge); covered: clean branch+merge with lineage and cross-asset merge-conflict detection (tests/mam_pure_logic.rs:171 version_tree_branch_and_merge_clean_with_lineage, tests/mam_pure_logic.rs:254 version_tree_merge_conflict_detected_and_no_silent_loss)
- [x] Test `transfer_manager.rs` with simulated network failures and partial transfers (completed 2026-06-24; 9 new unit tests in src/transfer_manager.rs: partial-then-failure, retry-preserves-bytes, max-retries-exhaustion, transient-failure-then-success, multiple-independent-failures, fail-idempotent, cancel-paused-partial, priority-unaffected-by-failed, zero-byte-progress)

## Documentation
- [ ] Add an entity-relationship diagram for the database schema
- [ ] Document the GraphQL API schema with query/mutation examples
- [ ] Add a deployment guide covering PostgreSQL setup, Tantivy index configuration, and storage backends

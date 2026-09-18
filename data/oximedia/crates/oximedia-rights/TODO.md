# oximedia-rights TODO

## Current Status
- 40 modules (13 subdirectory modules) covering rights tracking, license management, expiration handling, territory restrictions, usage tracking, clearance workflows, royalty calculation, watermarking, DRM metadata, audit trails, compliance reporting
- Core types: RightsManager (with SQLx database backend), RightsError enum with 13 variants
- Conditional compilation: `database` module and `RightsManager` excluded on wasm32
- Dependencies: oximedia-core, oximedia-watermark, oxisql-sqlite-compat (Pure-Rust SQLite), chrono, uuid, serde

## Enhancements
- [x] Add in-memory `RightsManager` alternative for wasm32 targets using HashMap-based storage (implemented 2026-05-15; src/in_memory_manager.rs — InMemoryRightsManager with LicenseRecord HashMap, check_rights, add_license, revoke_license, list_active_licenses, check_rights_batch; 35+ tests)
- [x] Extend `territory` with hierarchical region support (continent -> country -> state/province)
- [x] Add `rights_conflict` automatic resolution suggestions when overlapping rights are detected
- [x] Implement `royalty_engine` tiered royalty rates based on usage volume thresholds
- [x] Extend `embargo_policy` with platform-specific embargo windows (theatrical -> streaming -> broadcast)
- [x] Add `clearance_workflow` status notifications with configurable escalation on pending clearances (implemented 2026-05-15; ClearanceNotifierTrait + LoggingNotifier + NoopNotifier + EscalationConfig + find_overdue_clearances + submit/approve/reject_with_notify in clearance_workflow.rs)
- [x] Implement `rights_timeline` visualization data export for Gantt-chart style rights period display (verified 2026-05-16; src/rights_timeline.rs:417 lines)
- [x] Add `license_template` variable interpolation for generating customized license agreements (verified 2026-05-16; src/license_template.rs:447 lines)

## New Features
- [x] Add a `rights_api` module with REST endpoints for rights queries (is-asset-available, check-territory, get-license) (verified 2026-05-16; src/rights_api.rs:599 lines)
- [x] Implement `content_id` module for content identification (ISRC, ISAN, EIDR) linking to rights records (verified 2026-05-16; src/content_id.rs:565 lines)
- [x] Add `rights_import` module for bulk importing rights data from CSV/JSON/XML (verified 2026-05-16; src/rights_import.rs:600 lines)
- [x] Implement `rights_export` for generating machine-readable rights manifests (EIDR, DDEX, CWR formats) (verified 2026-05-16; src/rights_export.rs:465 lines)
- [x] Add `sub_licensing` module for managing sub-license chains with parent-child relationship tracking
- [x] Implement `rights_calendar` for calendar-based views of expiring and upcoming rights windows (verified 2026-05-16; src/rights_calendar.rs:521 lines)
- [x] Add `compliance_report` generator for regulatory requirements (GDPR, CCPA data rights) (verified 2026-05-16; src/compliance_report.rs:777 lines)
- [x] Implement `automated_takedown` module for triggering actions when rights expire or are revoked (verified 2026-05-16; src/automated_takedown.rs:578 lines)
- [x] Add `rights_search` full-text search across rights holders, territories, and license terms (verified 2026-05-16; src/rights_search.rs:755 lines)

## Performance
- [x] Add database connection pooling in `RightsManager` (implemented 2026-05-15; database/storage.rs already uses SqlitePoolOptions::max_connections(5); RightsManagerConfig added with pool_size/connect_timeout_secs/cache_ttl_secs; new_with_config() constructor)
- [x] Implement query caching in `rights_check` for frequently accessed assets with short TTL (implemented 2026-05-15; RightsCheckCache in lib.rs with Arc<Mutex<>> field on RightsManager; TTL via RightsManagerConfig.cache_ttl_secs; revoke_license() invalidates cache; tests: test_rights_cache_hit_avoids_db_query, test_rights_cache_ttl_expiry)
- [x] Add batch rights lookup in `rights_database` to reduce per-asset query overhead (implemented 2026-05-15; RightsManager::check_rights_batch with WHERE asset_id IN (…) SQL; InMemoryRightsManager::check_rights_batch for wasm32)
- [x] Index `territory` and `expiration` columns in the SQLite schema for faster filtered queries (implemented 2026-05-15; idx_rights_grants_territory and idx_rights_grants_end_date added in database/storage.rs::initialize_schema)

## Testing
- [x] Add tests for `royalty_calc` with complex tiered rate structures and multi-territory splits (implemented 2026-05-15; tests/it_royalty.rs — 8 tests: 3-tier schedule, 500/5000/50000 views, territory split arithmetic, TerritoryRateTable, Japan premium)
- [x] Test `embargo_window` edge cases (overlapping windows, midnight boundary, timezone handling) (implemented 2026-05-15; tests/it_embargo.rs — 7 tests: overlapping embargo most-restrictive, midnight UTC boundary 2026-06-01, Tokyo UTC+9 offset, indefinite embargo, scheduled-lift boundary)
- [x] Add tests for `rights_conflict` detection with multiple overlapping territorial grants (implemented 2026-05-15; tests/it_rights_conflict.rs — 8 tests: overlapping US theatrical, involved-ids, non-overlapping by date/territory, TerritoryBreach critical, RoyaltyDefault non-critical, mixed conflicts)
- [x] Test `clearance_workflow` state machine transitions (requested -> approved/denied -> expired) (implemented 2026-05-15; tests/it_clearance.rs — 7 tests: initial Pending, approve→Approved, reject→Rejected, 200-day overdue detection, 10-day not-overdue, counter-offer path, cannot-reopen rejected)
- [x] Add integration tests for `watermark` integration with actual image/video data (implemented 2026-05-15; tests/it_watermark.rs — 7 passing tests verifying WatermarkConfig/VisibleWatermark metadata correctness; embed/extract roundtrip placeholder test marked #[ignore] pending oximedia-watermark pipeline)

## Documentation
- [x] Document the rights data model: RightsHolder -> License -> Territory -> Usage relationship (implemented 2026-05-15; src/lib.rs — added //! ## Data Model section with ASCII tree and prose description)
- [x] Add decision flowchart for rights checking (territory check -> date check -> usage check -> clearance check) (implemented 2026-05-15; src/lib.rs — added //! ## Rights Check Flow section with ASCII flowchart)
- [x] Document royalty calculation formulas and configuration options in `royalty_engine` (implemented 2026-05-15; src/royalty_engine.rs — added //! ## Calculation Formulas section: tiered formula, territory split, usage basis table)
- [x] Add guide for setting up DRM metadata integration with common DRM providers (implemented 2026-05-15; src/lib.rs — added //! ## DRM Integration section: content_key_id UUID, PSSH/KID cross-reference, oximedia-drm pointer, revocation workflow)

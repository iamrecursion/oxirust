//! OxiMedia Media Asset Management (MAM) System
//!
//! This crate provides a comprehensive media asset management system with:
//! - PostgreSQL backend for scalable storage
//! - Full-text search using Tantivy
//! - RESTful and GraphQL APIs
//! - Workflow engine with approval/review processes
//! - Collection management with hierarchical organization
//! - Asset ingest with metadata extraction
//! - Advanced search capabilities including faceted and content-based search
//! - Proxy and thumbnail generation (including HLS/DASH ABR)
//! - User management and RBAC
//! - Cloud storage integration (S3, Azure, GCS)
//! - Webhook and event system
//! - Tag management with hierarchical tags
//! - Audit logging with SIEM export (CEF/LEEF)
//! - Folder structure and smart collections
//! - Watch-folder batch ingest
//! - Cold/glacier archive tier management
//! - Real-time collaboration (comments, annotations, review markers)
//! - Email/Slack/Teams notifications
//! - User-defined custom metadata fields
//! - Access request/approve workflows
//! - Scheduled reporting
//! - Federated search across multiple MAM instances
//! - Soft-delete trash bin

/// Request/approve access workflows for media assets.
pub mod access_request;
/// AI-powered metadata enrichment: rule-based and neural tag/caption/category generation.
pub mod ai_enrichment;
pub mod ai_tagging;
pub mod api;
/// Cold/glacier storage tier management with automatic demotion policies.
pub mod archive_tier;
pub mod asset;
/// Asset LRU/LFU cache with TTL eviction for frequently accessed media.
pub mod asset_cache;
pub mod asset_collection;
pub mod asset_lifecycle;
/// Quality-control pass definitions and per-asset QC state tracking.
pub mod asset_qc;
/// Asset relationship graph for tracking connections between media assets.
pub mod asset_relations;
/// Structured asset search engine with field-level filtering.
pub mod asset_search;
pub mod asset_status;
/// Scoped asset tagging with inverted index for fast tag-to-asset lookups.
pub mod asset_tag;
/// Inverted tag index for fast tag-to-asset lookups and co-occurrence analysis.
pub mod asset_tag_index;
/// Automatic and manual asset tagging with taxonomy support.
pub mod asset_tagging;
/// Version history snapshots, diff generation, and restore for media assets.
pub mod asset_versioning;
pub mod audit;
pub mod batch_ingest;
/// Bulk metadata updates across multiple assets in a single transactional pass.
pub mod batch_metadata;
/// Bulk operations on media assets with progress tracking.
pub mod bulk_operation;
/// Atomic bulk field-update engine with undo log support.
pub mod bulk_update;
/// Catalog search filters, entries, searcher, and result sets.
pub mod catalog_search;
/// Real-time comments, annotations, and review markers on assets.
pub mod collaboration;
pub mod collection;
pub mod collection_manager;
/// User-defined metadata fields with type validation.
pub mod custom_field;
pub mod database;
/// Delivery log for recording and querying asset delivery history.
pub mod delivery_log;
/// Perceptual hash-based duplicate asset detection.
pub mod duplicate_detect;
/// In-process publish/subscribe event bus for MAM domain events.
pub mod event_bus;
pub mod export_package;
/// Federated search across multiple MAM instances.
pub mod federated_search;
/// Hierarchical folder tree for organising media assets.
pub mod folder_hierarchy;
pub mod folders;
/// Incremental search-index rebuilds for large asset catalogues.
pub mod incremental_index;
pub mod ingest;
pub mod ingest_pipeline;
pub mod ingest_workflow;
pub mod integration;
/// Materialised-path adjacency table for efficient hierarchical queries.
pub mod materialized_path;
pub mod media_catalog;
/// Media format family classification and format information registry.
pub mod media_format_info;
pub mod media_linking;
/// Media project lifecycle management with status tracking.
pub mod media_project;
/// Broadcast rights clearance workflow with territory and window tracking.
pub mod media_rights_clearance;
/// Metadata schema definitions, field types, and validation rules.
pub mod metadata_schema;
/// Metadata template definitions and template library.
pub mod metadata_template;
pub mod notification;
pub mod permissions;
/// Database connection-pool sizing and tuning heuristics.
pub mod pool_tuning;
pub mod proxy;
/// Email/Slack/Teams notifications for asset and workflow events.
/// Asset connection graph: typed relationships between media assets.
pub mod relationship;
/// Scheduled report generation (storage usage, asset stats, etc.).
pub mod reporting;
/// Data retention policy engine for automated asset lifecycle management.
pub mod retention_policy;
/// Rights relationship links: connect assets to rights records.
pub mod rights_link;
/// Rights coverage summary and builder for asset rights tracking.
pub mod rights_summary;
pub mod search;
pub mod search_index;
/// Search-index pre-warming strategies and scheduled warming jobs.
pub mod search_warming;
/// Enhanced search with BM25-inspired scoring, faceted filtering, and Jaccard similarity.
pub mod smart_search;
pub mod storage;
pub mod tags;
/// SMPTE and custom timeline markers for editorial annotations.
pub mod timeline_marker;
pub mod transcoding_profile;
/// File transfer job manager with status tracking and retry support.
pub mod transfer_manager;
/// Soft-delete trash bin with configurable auto-purge TTL.
pub mod trash_bin;
/// Asset usage analytics and access pattern tracking.
pub mod usage_analytics;
/// Semantic versioning comparisons and range resolution for asset versions.
pub mod version_compare;
pub mod version_control;
pub mod versioning;
pub mod webhook;
pub mod workflow;
/// Workflow approval steps, approver sets, and escalation rules.
pub mod workflow_approval;
pub mod workflow_integration;
/// Event-driven trigger rules that fire workflow actions on asset events.
pub mod workflow_trigger;

use std::sync::Arc;
use thiserror::Error;

/// Main MAM system structure that coordinates all subsystems
pub struct MamSystem {
    /// Database connection pool
    db: Arc<database::Database>,
    /// Search engine instance
    search: Arc<search::SearchEngine>,
    /// Asset manager
    asset_manager: Arc<asset::AssetManager>,
    /// Collection manager
    collection_manager: Arc<collection::CollectionManager>,
    /// Ingest system
    ingest_system: Arc<ingest::IngestSystem>,
    /// Workflow engine
    workflow_engine: Arc<workflow::WorkflowEngine>,
    /// Proxy manager
    proxy_manager: Arc<proxy::ProxyManager>,
    /// Permission manager
    permission_manager: Arc<permissions::PermissionManager>,
    /// Storage manager
    storage_manager: Arc<storage::StorageManager>,
    /// Webhook manager
    webhook_manager: Arc<webhook::WebhookManager>,
    /// Tag manager
    tag_manager: Arc<tags::TagManager>,
    /// Audit logger
    audit_logger: Arc<audit::AuditLogger>,
    /// Folder manager
    folder_manager: Arc<folders::FolderManager>,
    /// Integration manager
    integration_manager: Arc<integration::IntegrationManager>,
}

/// MAM system configuration
#[derive(Debug, Clone)]
pub struct MamConfig {
    /// PostgreSQL database URL
    pub database_url: String,
    /// Tantivy index path
    pub index_path: String,
    /// Storage root path for media files
    pub storage_path: String,
    /// Proxy/thumbnail storage path
    pub proxy_path: String,
    /// Maximum concurrent ingests
    pub max_concurrent_ingests: usize,
    /// Enable email notifications
    pub enable_email: bool,
    /// SMTP server for email notifications
    pub smtp_server: Option<String>,
    /// JWT secret for authentication
    pub jwt_secret: String,
    /// API rate limit (requests per minute)
    pub rate_limit: u32,
}

impl Default for MamConfig {
    fn default() -> Self {
        Self {
            database_url: "postgres://localhost/oximedia_mam".to_string(),
            index_path: "/var/lib/oximedia/search_index".to_string(),
            storage_path: "/var/lib/oximedia/storage".to_string(),
            proxy_path: "/var/lib/oximedia/proxies".to_string(),
            max_concurrent_ingests: 4,
            enable_email: false,
            smtp_server: None,
            jwt_secret: "change-me-in-production".to_string(),
            rate_limit: 100,
        }
    }
}

/// MAM system errors
#[derive(Error, Debug)]
pub enum MamError {
    /// Database operation failed
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    /// Search engine error
    #[error("Search error: {0}")]
    Search(#[from] tantivy::TantivyError),

    /// Asset not found
    #[error("Asset not found: {0}")]
    AssetNotFound(uuid::Uuid),

    /// Collection not found
    #[error("Collection not found: {0}")]
    CollectionNotFound(uuid::Uuid),

    /// Workflow not found
    #[error("Workflow not found: {0}")]
    WorkflowNotFound(uuid::Uuid),

    /// Permission denied
    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    /// Invalid metadata
    #[error("Invalid metadata: {0}")]
    InvalidMetadata(String),

    /// Ingest failed
    #[error("Ingest failed: {0}")]
    IngestFailed(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Authentication error
    #[error("Authentication error: {0}")]
    Authentication(String),

    /// Authorization error
    #[error("Authorization error: {0}")]
    Authorization(String),

    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<tantivy::directory::error::OpenDirectoryError> for MamError {
    fn from(e: tantivy::directory::error::OpenDirectoryError) -> Self {
        Self::Internal(format!("Failed to open search index: {e}"))
    }
}

impl From<tantivy::query::QueryParserError> for MamError {
    fn from(e: tantivy::query::QueryParserError) -> Self {
        Self::InvalidInput(format!("Invalid search query: {e}"))
    }
}

impl From<async_graphql::Error> for MamError {
    fn from(e: async_graphql::Error) -> Self {
        Self::Internal(format!("GraphQL error: {e:?}"))
    }
}

/// Result type for MAM operations
pub type Result<T> = std::result::Result<T, MamError>;

impl MamSystem {
    /// Create a new MAM system instance
    ///
    /// # Arguments
    ///
    /// * `config` - MAM system configuration
    ///
    /// # Errors
    ///
    /// Returns an error if database connection or search index initialization fails
    pub async fn new(config: MamConfig) -> Result<Self> {
        // Initialize database
        let db = Arc::new(database::Database::new(&config.database_url).await?);

        // Run migrations
        db.run_migrations().await?;

        // Initialize search engine
        let search = Arc::new(search::SearchEngine::new(&config.index_path)?);

        // Initialize asset manager
        let asset_manager = Arc::new(asset::AssetManager::new(
            Arc::clone(&db),
            config.storage_path.clone(),
        ));

        // Initialize collection manager
        let collection_manager = Arc::new(collection::CollectionManager::new(
            Arc::clone(&db),
            Arc::clone(&search),
        ));

        // Initialize ingest system
        let ingest_system = Arc::new(ingest::IngestSystem::new(
            Arc::clone(&db),
            Arc::clone(&search),
            Arc::clone(&asset_manager),
            config.clone(),
        ));

        // Initialize workflow engine
        let workflow_engine = Arc::new(workflow::WorkflowEngine::new(
            Arc::clone(&db),
            config.clone(),
        ));

        // Initialize proxy manager
        let proxy_manager = Arc::new(proxy::ProxyManager::new(
            Arc::clone(&db),
            config.proxy_path.clone(),
            format!("{}/thumbnails", config.proxy_path),
            config.max_concurrent_ingests,
        ));

        // Initialize permission manager
        let permission_manager = Arc::new(permissions::PermissionManager::new(Arc::clone(&db)));

        // Initialize storage manager
        let storage_manager = Arc::new(storage::StorageManager::new(
            Arc::clone(&db),
            "local".to_string(),
        ));

        // Initialize webhook manager
        let webhook_manager = Arc::new(webhook::WebhookManager::new(Arc::clone(&db)));

        // Initialize tag manager
        let tag_manager = Arc::new(tags::TagManager::new(Arc::clone(&db)));

        // Initialize audit logger
        let audit_logger = Arc::new(audit::AuditLogger::new(Arc::clone(&db)));

        // Initialize folder manager
        let folder_manager = Arc::new(folders::FolderManager::new(
            Arc::clone(&db),
            Arc::clone(&search),
        ));

        // Initialize integration manager
        let integration_manager = Arc::new(integration::IntegrationManager::new());

        Ok(Self {
            db,
            search,
            asset_manager,
            collection_manager,
            ingest_system,
            workflow_engine,
            proxy_manager,
            permission_manager,
            storage_manager,
            webhook_manager,
            tag_manager,
            audit_logger,
            folder_manager,
            integration_manager,
        })
    }

    /// Get database reference
    #[must_use]
    pub fn database(&self) -> &Arc<database::Database> {
        &self.db
    }

    /// Get search engine reference
    #[must_use]
    pub fn search(&self) -> &Arc<search::SearchEngine> {
        &self.search
    }

    /// Get asset manager reference
    #[must_use]
    pub fn asset_manager(&self) -> &Arc<asset::AssetManager> {
        &self.asset_manager
    }

    /// Get collection manager reference
    #[must_use]
    pub fn collection_manager(&self) -> &Arc<collection::CollectionManager> {
        &self.collection_manager
    }

    /// Get ingest system reference
    #[must_use]
    pub fn ingest_system(&self) -> &Arc<ingest::IngestSystem> {
        &self.ingest_system
    }

    /// Get workflow engine reference
    #[must_use]
    pub fn workflow_engine(&self) -> &Arc<workflow::WorkflowEngine> {
        &self.workflow_engine
    }

    /// Get proxy manager reference
    #[must_use]
    pub fn proxy_manager(&self) -> &Arc<proxy::ProxyManager> {
        &self.proxy_manager
    }

    /// Get permission manager reference
    #[must_use]
    pub fn permission_manager(&self) -> &Arc<permissions::PermissionManager> {
        &self.permission_manager
    }

    /// Get storage manager reference
    #[must_use]
    pub fn storage_manager(&self) -> &Arc<storage::StorageManager> {
        &self.storage_manager
    }

    /// Get webhook manager reference
    #[must_use]
    pub fn webhook_manager(&self) -> &Arc<webhook::WebhookManager> {
        &self.webhook_manager
    }

    /// Get tag manager reference
    #[must_use]
    pub fn tag_manager(&self) -> &Arc<tags::TagManager> {
        &self.tag_manager
    }

    /// Get audit logger reference
    #[must_use]
    pub fn audit_logger(&self) -> &Arc<audit::AuditLogger> {
        &self.audit_logger
    }

    /// Get folder manager reference
    #[must_use]
    pub fn folder_manager(&self) -> &Arc<folders::FolderManager> {
        &self.folder_manager
    }

    /// Get integration manager reference
    #[must_use]
    pub fn integration_manager(&self) -> &Arc<integration::IntegrationManager> {
        &self.integration_manager
    }

    /// Shutdown the MAM system gracefully
    pub async fn shutdown(&self) -> Result<()> {
        tracing::info!("Shutting down MAM system");

        // Stop ingest system
        self.ingest_system.shutdown().await?;

        // Stop workflow engine
        self.workflow_engine.shutdown().await?;

        // Close database connections
        self.db.close().await?;

        tracing::info!("MAM system shutdown complete");
        Ok(())
    }

    /// Get system health status
    pub async fn health(&self) -> Result<HealthStatus> {
        let db_healthy = self.db.check_health().await.is_ok();
        let search_healthy = self.search.check_health().is_ok();

        let asset_count = self.db.get_asset_count().await.unwrap_or(0);
        let collection_count = self.db.get_collection_count().await.unwrap_or(0);

        Ok(HealthStatus {
            database: db_healthy,
            search: search_healthy,
            asset_count,
            collection_count,
            version: env!("CARGO_PKG_VERSION").to_string(),
        })
    }

    /// Get system statistics
    pub async fn statistics(&self) -> Result<SystemStatistics> {
        let stats = self.db.get_statistics().await?;
        Ok(stats)
    }
}

/// System health status
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HealthStatus {
    /// Database connection healthy
    pub database: bool,
    /// Search engine healthy
    pub search: bool,
    /// Total number of assets
    pub asset_count: i64,
    /// Total number of collections
    pub collection_count: i64,
    /// System version
    pub version: String,
}

/// System statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SystemStatistics {
    /// Total assets
    pub total_assets: i64,
    /// Total collections
    pub total_collections: i64,
    /// Total storage used (bytes)
    pub storage_used: i64,
    /// Total users
    pub total_users: i64,
    /// Total workflows
    pub total_workflows: i64,
    /// Active ingests
    pub active_ingests: i64,
    /// Failed ingests (last 24h)
    pub failed_ingests_24h: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = MamConfig::default();
        assert_eq!(config.max_concurrent_ingests, 4);
        assert_eq!(config.rate_limit, 100);
        assert!(!config.enable_email);
    }

    #[test]
    fn test_health_status_serialization() {
        let status = HealthStatus {
            database: true,
            search: true,
            asset_count: 1000,
            collection_count: 50,
            version: "0.1.0".to_string(),
        };

        let json = serde_json::to_string(&status).expect("should succeed in test");
        let deserialized: HealthStatus =
            serde_json::from_str(&json).expect("should succeed in test");

        assert_eq!(deserialized.asset_count, 1000);
        assert_eq!(deserialized.collection_count, 50);
    }
}

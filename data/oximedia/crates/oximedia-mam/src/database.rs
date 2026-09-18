//! Database layer for MAM system
//!
//! Provides PostgreSQL schema design and database operations for:
//! - Asset storage and metadata
//! - Collection hierarchy
//! - Version tracking
//! - User management and permissions
//! - Workflow state

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{postgres::PgPoolOptions, PgPool, Row};
use uuid::Uuid;

use crate::{MamError, Result, SystemStatistics};

/// Database connection and operations
pub struct Database {
    pool: PgPool,
}

/// User account
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct User {
    pub id: Uuid,
    pub username: String,
    pub email: String,
    pub password_hash: String,
    pub full_name: Option<String>,
    pub role: UserRole,
    pub is_active: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// User role for RBAC
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "user_role", rename_all = "lowercase")]
pub enum UserRole {
    Admin,
    Editor,
    Viewer,
    Guest,
}

impl std::fmt::Display for UserRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Admin => write!(f, "admin"),
            Self::Editor => write!(f, "editor"),
            Self::Viewer => write!(f, "viewer"),
            Self::Guest => write!(f, "guest"),
        }
    }
}

/// Permission types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "permission_type", rename_all = "lowercase")]
pub enum PermissionType {
    Read,
    Write,
    Delete,
    Share,
    Admin,
}

/// Asset permission
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct AssetPermission {
    pub id: Uuid,
    pub asset_id: Uuid,
    pub user_id: Option<Uuid>,
    pub role: Option<UserRole>,
    pub permission: PermissionType,
    pub created_at: DateTime<Utc>,
}

/// Collection permission
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct CollectionPermission {
    pub id: Uuid,
    pub collection_id: Uuid,
    pub user_id: Option<Uuid>,
    pub role: Option<UserRole>,
    pub permission: PermissionType,
    pub created_at: DateTime<Utc>,
}

/// Asset version record
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct AssetVersion {
    pub id: Uuid,
    pub asset_id: Uuid,
    pub version_number: i32,
    pub file_path: String,
    pub file_size: i64,
    pub checksum: String,
    pub created_by: Uuid,
    pub created_at: DateTime<Utc>,
    pub comment: Option<String>,
}

/// Audit log entry
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct AuditLog {
    pub id: Uuid,
    pub user_id: Option<Uuid>,
    pub action: String,
    pub resource_type: String,
    pub resource_id: Uuid,
    pub details: Option<serde_json::Value>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub created_at: DateTime<Utc>,
}

impl Database {
    /// Create a new database connection with default pool settings (50 max connections).
    ///
    /// For production deployments, prefer [`Database::new_with_config`] which accepts
    /// pool sizing produced by [`crate::pool_tuning::PoolTuner`].
    ///
    /// # Errors
    ///
    /// Returns an error if the database connection fails
    pub async fn new(database_url: &str) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(50)
            .connect(database_url)
            .await?;

        Ok(Self { pool })
    }

    /// Create a new database connection using a [`crate::pool_tuning::PoolConfig`].
    ///
    /// This integrates the adaptive pool-tuning layer: callers may use a
    /// [`crate::pool_tuning::PoolTuner`] to derive a `PoolConfig` that reflects
    /// observed concurrency, then pass it here when (re-)creating the pool.
    ///
    /// ```text
    /// // Conceptual usage — requires a live PostgreSQL server:
    /// //
    /// //   let sampler = Arc::new(LoadSampler::new(Duration::from_secs(60)));
    /// //   let tuner   = PoolTuner::new(sampler, TuningStrategy::Balanced,
    /// //                                PoolConfig::default(), 2, 200);
    /// //   let metrics = PoolMetrics::new(active, idle, pending, max, min);
    /// //   let rec     = tuner.evaluate(&metrics);
    /// //   tuner.apply(&rec);
    /// //   let db = Database::new_with_config(DATABASE_URL, tuner.current_config()).await?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the database connection fails
    pub async fn new_with_config(
        database_url: &str,
        config: crate::pool_tuning::PoolConfig,
    ) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .min_connections(config.min_connections)
            .max_connections(config.max_connections)
            .acquire_timeout(config.acquire_timeout)
            .idle_timeout(config.idle_timeout)
            .max_lifetime(config.max_lifetime)
            .connect(database_url)
            .await?;

        Ok(Self { pool })
    }

    /// Run database migrations
    ///
    /// # Errors
    ///
    /// Returns an error if migrations fail
    pub async fn run_migrations(&self) -> Result<()> {
        sqlx::migrate!("./migrations")
            .run(&self.pool)
            .await
            .map_err(|e| MamError::Internal(format!("Migration failed: {e}")))?;

        Ok(())
    }

    /// Close database connections
    ///
    /// # Errors
    ///
    /// Returns an error if closing connections fails
    pub async fn close(&self) -> Result<()> {
        self.pool.close().await;
        Ok(())
    }

    /// Check database health
    ///
    /// # Errors
    ///
    /// Returns an error if the health check fails
    pub async fn check_health(&self) -> Result<()> {
        sqlx::query("SELECT 1").execute(&self.pool).await?;
        Ok(())
    }

    /// Get total asset count
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub async fn get_asset_count(&self) -> Result<i64> {
        let row = sqlx::query("SELECT COUNT(*) as count FROM assets")
            .fetch_one(&self.pool)
            .await?;
        Ok(row.get("count"))
    }

    /// Get total collection count
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub async fn get_collection_count(&self) -> Result<i64> {
        let row = sqlx::query("SELECT COUNT(*) as count FROM collections")
            .fetch_one(&self.pool)
            .await?;
        Ok(row.get("count"))
    }

    /// Get system statistics
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub async fn get_statistics(&self) -> Result<SystemStatistics> {
        let assets: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM assets")
            .fetch_one(&self.pool)
            .await?;

        let collections: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM collections")
            .fetch_one(&self.pool)
            .await?;

        let storage: i64 = sqlx::query_scalar(
            "SELECT COALESCE(SUM(file_size), 0) FROM assets WHERE file_size IS NOT NULL",
        )
        .fetch_one(&self.pool)
        .await?;

        let users: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM users")
            .fetch_one(&self.pool)
            .await?;

        let workflows: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM workflows")
            .fetch_one(&self.pool)
            .await?;

        let active_ingests: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM ingest_jobs WHERE status = 'processing'")
                .fetch_one(&self.pool)
                .await?;

        let failed_ingests: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM ingest_jobs
             WHERE status = 'failed' AND created_at > NOW() - INTERVAL '24 hours'",
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(SystemStatistics {
            total_assets: assets,
            total_collections: collections,
            storage_used: storage,
            total_users: users,
            total_workflows: workflows,
            active_ingests,
            failed_ingests_24h: failed_ingests,
        })
    }

    // User management

    /// Create a new user
    ///
    /// # Errors
    ///
    /// Returns an error if the insert fails
    pub async fn create_user(
        &self,
        username: &str,
        email: &str,
        password_hash: &str,
        full_name: Option<&str>,
        role: UserRole,
    ) -> Result<User> {
        let user = sqlx::query_as::<_, User>(
            "INSERT INTO users (id, username, email, password_hash, full_name, role, is_active, created_at, updated_at)
             VALUES ($1, $2, $3, $4, $5, $6, true, NOW(), NOW())
             RETURNING *"
        )
        .bind(Uuid::new_v4())
        .bind(username)
        .bind(email)
        .bind(password_hash)
        .bind(full_name)
        .bind(role)
        .fetch_one(&self.pool)
        .await?;

        Ok(user)
    }

    /// Get user by ID
    ///
    /// # Errors
    ///
    /// Returns an error if the user is not found
    pub async fn get_user(&self, user_id: Uuid) -> Result<User> {
        let user = sqlx::query_as::<_, User>("SELECT * FROM users WHERE id = $1")
            .bind(user_id)
            .fetch_one(&self.pool)
            .await?;

        Ok(user)
    }

    /// Get user by username
    ///
    /// # Errors
    ///
    /// Returns an error if the user is not found
    pub async fn get_user_by_username(&self, username: &str) -> Result<User> {
        let user = sqlx::query_as::<_, User>("SELECT * FROM users WHERE username = $1")
            .bind(username)
            .fetch_one(&self.pool)
            .await?;

        Ok(user)
    }

    /// Update user
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails
    pub async fn update_user(
        &self,
        user_id: Uuid,
        email: Option<&str>,
        full_name: Option<&str>,
        role: Option<UserRole>,
    ) -> Result<User> {
        let user = sqlx::query_as::<_, User>(
            "UPDATE users
             SET email = COALESCE($2, email),
                 full_name = COALESCE($3, full_name),
                 role = COALESCE($4, role),
                 updated_at = NOW()
             WHERE id = $1
             RETURNING *",
        )
        .bind(user_id)
        .bind(email)
        .bind(full_name)
        .bind(role)
        .fetch_one(&self.pool)
        .await?;

        Ok(user)
    }

    /// Delete user (soft delete)
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails
    pub async fn delete_user(&self, user_id: Uuid) -> Result<()> {
        sqlx::query("UPDATE users SET is_active = false WHERE id = $1")
            .bind(user_id)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    /// List all users
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub async fn list_users(&self, limit: i64, offset: i64) -> Result<Vec<User>> {
        let users = sqlx::query_as::<_, User>(
            "SELECT * FROM users WHERE is_active = true ORDER BY created_at DESC LIMIT $1 OFFSET $2"
        )
        .bind(limit)
        .bind(offset)
        .fetch_all(&self.pool)
        .await?;

        Ok(users)
    }

    // Asset permissions

    /// Grant asset permission
    ///
    /// # Errors
    ///
    /// Returns an error if the insert fails
    #[allow(clippy::too_many_arguments)]
    pub async fn grant_asset_permission(
        &self,
        asset_id: Uuid,
        user_id: Option<Uuid>,
        role: Option<UserRole>,
        permission: PermissionType,
    ) -> Result<AssetPermission> {
        let perm = sqlx::query_as::<_, AssetPermission>(
            "INSERT INTO asset_permissions (id, asset_id, user_id, role, permission, created_at)
             VALUES ($1, $2, $3, $4, $5, NOW())
             RETURNING *",
        )
        .bind(Uuid::new_v4())
        .bind(asset_id)
        .bind(user_id)
        .bind(role)
        .bind(permission)
        .fetch_one(&self.pool)
        .await?;

        Ok(perm)
    }

    /// Revoke asset permission
    ///
    /// # Errors
    ///
    /// Returns an error if the delete fails
    pub async fn revoke_asset_permission(&self, permission_id: Uuid) -> Result<()> {
        sqlx::query("DELETE FROM asset_permissions WHERE id = $1")
            .bind(permission_id)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    /// Check if user has asset permission
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub async fn check_asset_permission(
        &self,
        asset_id: Uuid,
        user_id: Uuid,
        permission: PermissionType,
    ) -> Result<bool> {
        // Get user's role
        let user = self.get_user(user_id).await?;

        // Admin has all permissions
        if user.role == UserRole::Admin {
            return Ok(true);
        }

        // Check specific user permission
        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM asset_permissions
             WHERE asset_id = $1
             AND (user_id = $2 OR role = $3)
             AND permission >= $4",
        )
        .bind(asset_id)
        .bind(user_id)
        .bind(user.role)
        .bind(permission)
        .fetch_one(&self.pool)
        .await?;

        Ok(count > 0)
    }

    /// Get asset permissions
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub async fn get_asset_permissions(&self, asset_id: Uuid) -> Result<Vec<AssetPermission>> {
        let perms = sqlx::query_as::<_, AssetPermission>(
            "SELECT * FROM asset_permissions WHERE asset_id = $1 ORDER BY created_at",
        )
        .bind(asset_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(perms)
    }

    // Collection permissions

    /// Grant collection permission
    ///
    /// # Errors
    ///
    /// Returns an error if the insert fails
    #[allow(clippy::too_many_arguments)]
    pub async fn grant_collection_permission(
        &self,
        collection_id: Uuid,
        user_id: Option<Uuid>,
        role: Option<UserRole>,
        permission: PermissionType,
    ) -> Result<CollectionPermission> {
        let perm = sqlx::query_as::<_, CollectionPermission>(
            "INSERT INTO collection_permissions (id, collection_id, user_id, role, permission, created_at)
             VALUES ($1, $2, $3, $4, $5, NOW())
             RETURNING *"
        )
        .bind(Uuid::new_v4())
        .bind(collection_id)
        .bind(user_id)
        .bind(role)
        .bind(permission)
        .fetch_one(&self.pool)
        .await?;

        Ok(perm)
    }

    /// Check if user has collection permission
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub async fn check_collection_permission(
        &self,
        collection_id: Uuid,
        user_id: Uuid,
        permission: PermissionType,
    ) -> Result<bool> {
        let user = self.get_user(user_id).await?;

        if user.role == UserRole::Admin {
            return Ok(true);
        }

        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM collection_permissions
             WHERE collection_id = $1
             AND (user_id = $2 OR role = $3)
             AND permission >= $4",
        )
        .bind(collection_id)
        .bind(user_id)
        .bind(user.role)
        .bind(permission)
        .fetch_one(&self.pool)
        .await?;

        Ok(count > 0)
    }

    // Version tracking

    /// Create asset version
    ///
    /// # Errors
    ///
    /// Returns an error if the insert fails
    #[allow(clippy::too_many_arguments)]
    pub async fn create_asset_version(
        &self,
        asset_id: Uuid,
        version_number: i32,
        file_path: &str,
        file_size: i64,
        checksum: &str,
        created_by: Uuid,
        comment: Option<&str>,
    ) -> Result<AssetVersion> {
        let version = sqlx::query_as::<_, AssetVersion>(
            "INSERT INTO asset_versions
             (id, asset_id, version_number, file_path, file_size, checksum, created_by, created_at, comment)
             VALUES ($1, $2, $3, $4, $5, $6, $7, NOW(), $8)
             RETURNING *"
        )
        .bind(Uuid::new_v4())
        .bind(asset_id)
        .bind(version_number)
        .bind(file_path)
        .bind(file_size)
        .bind(checksum)
        .bind(created_by)
        .bind(comment)
        .fetch_one(&self.pool)
        .await?;

        Ok(version)
    }

    /// Get asset versions
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub async fn get_asset_versions(&self, asset_id: Uuid) -> Result<Vec<AssetVersion>> {
        let versions = sqlx::query_as::<_, AssetVersion>(
            "SELECT * FROM asset_versions WHERE asset_id = $1 ORDER BY version_number DESC",
        )
        .bind(asset_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(versions)
    }

    /// Get latest version number
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub async fn get_latest_version_number(&self, asset_id: Uuid) -> Result<i32> {
        let version: Option<i32> = sqlx::query_scalar(
            "SELECT MAX(version_number) FROM asset_versions WHERE asset_id = $1",
        )
        .bind(asset_id)
        .fetch_one(&self.pool)
        .await?;

        Ok(version.unwrap_or(0))
    }

    // Audit logging

    /// Create audit log entry
    ///
    /// # Errors
    ///
    /// Returns an error if the insert fails
    #[allow(clippy::too_many_arguments)]
    pub async fn create_audit_log(
        &self,
        user_id: Option<Uuid>,
        action: &str,
        resource_type: &str,
        resource_id: Uuid,
        details: Option<serde_json::Value>,
        ip_address: Option<&str>,
        user_agent: Option<&str>,
    ) -> Result<AuditLog> {
        let log = sqlx::query_as::<_, AuditLog>(
            "INSERT INTO audit_logs
             (id, user_id, action, resource_type, resource_id, details, ip_address, user_agent, created_at)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
             RETURNING *"
        )
        .bind(Uuid::new_v4())
        .bind(user_id)
        .bind(action)
        .bind(resource_type)
        .bind(resource_id)
        .bind(details)
        .bind(ip_address)
        .bind(user_agent)
        .fetch_one(&self.pool)
        .await?;

        Ok(log)
    }

    /// Get audit logs for resource
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub async fn get_audit_logs(
        &self,
        resource_type: &str,
        resource_id: Uuid,
        limit: i64,
    ) -> Result<Vec<AuditLog>> {
        let logs = sqlx::query_as::<_, AuditLog>(
            "SELECT * FROM audit_logs
             WHERE resource_type = $1 AND resource_id = $2
             ORDER BY created_at DESC
             LIMIT $3",
        )
        .bind(resource_type)
        .bind(resource_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(logs)
    }

    /// Get pool reference
    #[must_use]
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }
}

/// Database schema SQL
pub const SCHEMA_SQL: &str = r#"
-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    role user_role NOT NULL DEFAULT 'viewer',
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);

-- Assets table
CREATE TABLE IF NOT EXISTS assets (
    id UUID PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(1024) NOT NULL,
    file_size BIGINT,
    mime_type VARCHAR(255),
    checksum VARCHAR(64) NOT NULL,

    -- Technical metadata
    duration_ms BIGINT,
    width INTEGER,
    height INTEGER,
    frame_rate DECIMAL(10, 4),
    video_codec VARCHAR(255),
    audio_codec VARCHAR(255),
    bit_rate BIGINT,

    -- Descriptive metadata
    title VARCHAR(512),
    description TEXT,
    keywords TEXT[],
    categories TEXT[],

    -- Rights metadata
    copyright VARCHAR(512),
    license VARCHAR(255),
    creator VARCHAR(255),

    -- Custom metadata (JSONB for flexibility)
    custom_metadata JSONB,

    -- Status and tracking
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT unique_checksum UNIQUE(checksum)
);

CREATE INDEX idx_assets_checksum ON assets(checksum);
CREATE INDEX idx_assets_filename ON assets(filename);
CREATE INDEX idx_assets_mime_type ON assets(mime_type);
CREATE INDEX idx_assets_created_at ON assets(created_at DESC);
CREATE INDEX idx_assets_status ON assets(status);
CREATE INDEX idx_assets_keywords ON assets USING GIN(keywords);
CREATE INDEX idx_assets_custom_metadata ON assets USING GIN(custom_metadata);

-- Collections table
CREATE TABLE IF NOT EXISTS collections (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    parent_id UUID REFERENCES collections(id) ON DELETE CASCADE,
    is_smart BOOLEAN NOT NULL DEFAULT false,
    smart_query JSONB,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_collections_parent ON collections(parent_id);
CREATE INDEX idx_collections_name ON collections(name);
CREATE INDEX idx_collections_created_at ON collections(created_at DESC);

-- Collection items (many-to-many)
CREATE TABLE IF NOT EXISTS collection_items (
    id UUID PRIMARY KEY,
    collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    position INTEGER,
    added_by UUID REFERENCES users(id),
    added_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT unique_collection_asset UNIQUE(collection_id, asset_id)
);

CREATE INDEX idx_collection_items_collection ON collection_items(collection_id);
CREATE INDEX idx_collection_items_asset ON collection_items(asset_id);

-- Asset versions
CREATE TABLE IF NOT EXISTS asset_versions (
    id UUID PRIMARY KEY,
    asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    file_path VARCHAR(1024) NOT NULL,
    file_size BIGINT NOT NULL,
    checksum VARCHAR(64) NOT NULL,
    created_by UUID NOT NULL REFERENCES users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    comment TEXT,
    CONSTRAINT unique_asset_version UNIQUE(asset_id, version_number)
);

CREATE INDEX idx_asset_versions_asset ON asset_versions(asset_id);

-- Asset permissions
CREATE TABLE IF NOT EXISTS asset_permissions (
    id UUID PRIMARY KEY,
    asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    role user_role,
    permission permission_type NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT check_user_or_role CHECK (user_id IS NOT NULL OR role IS NOT NULL)
);

CREATE INDEX idx_asset_permissions_asset ON asset_permissions(asset_id);
CREATE INDEX idx_asset_permissions_user ON asset_permissions(user_id);

-- Collection permissions
CREATE TABLE IF NOT EXISTS collection_permissions (
    id UUID PRIMARY KEY,
    collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    role user_role,
    permission permission_type NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT check_user_or_role_coll CHECK (user_id IS NOT NULL OR role IS NOT NULL)
);

CREATE INDEX idx_collection_permissions_collection ON collection_permissions(collection_id);
CREATE INDEX idx_collection_permissions_user ON collection_permissions(user_id);

-- Workflows
CREATE TABLE IF NOT EXISTS workflows (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    workflow_type VARCHAR(50) NOT NULL,
    config JSONB NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_workflows_type ON workflows(workflow_type);
CREATE INDEX idx_workflows_active ON workflows(is_active);

-- Workflow instances
CREATE TABLE IF NOT EXISTS workflow_instances (
    id UUID PRIMARY KEY,
    workflow_id UUID NOT NULL REFERENCES workflows(id),
    asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    current_state VARCHAR(255),
    state_data JSONB,
    started_by UUID REFERENCES users(id),
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    error_message TEXT
);

CREATE INDEX idx_workflow_instances_workflow ON workflow_instances(workflow_id);
CREATE INDEX idx_workflow_instances_asset ON workflow_instances(asset_id);
CREATE INDEX idx_workflow_instances_status ON workflow_instances(status);

-- Workflow tasks
CREATE TABLE IF NOT EXISTS workflow_tasks (
    id UUID PRIMARY KEY,
    instance_id UUID NOT NULL REFERENCES workflow_instances(id) ON DELETE CASCADE,
    task_type VARCHAR(255) NOT NULL,
    assigned_to UUID REFERENCES users(id),
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    due_date TIMESTAMPTZ,
    completed_by UUID REFERENCES users(id),
    completed_at TIMESTAMPTZ,
    comment TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_workflow_tasks_instance ON workflow_tasks(instance_id);
CREATE INDEX idx_workflow_tasks_assigned ON workflow_tasks(assigned_to);
CREATE INDEX idx_workflow_tasks_status ON workflow_tasks(status);

-- Comments/annotations
CREATE TABLE IF NOT EXISTS comments (
    id UUID PRIMARY KEY,
    asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id),
    parent_id UUID REFERENCES comments(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    timecode_ms BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_comments_asset ON comments(asset_id);
CREATE INDEX idx_comments_user ON comments(user_id);
CREATE INDEX idx_comments_parent ON comments(parent_id);

-- Ingest jobs
CREATE TABLE IF NOT EXISTS ingest_jobs (
    id UUID PRIMARY KEY,
    source_path VARCHAR(1024) NOT NULL,
    asset_id UUID REFERENCES assets(id),
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    progress INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    metadata JSONB,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

CREATE INDEX idx_ingest_jobs_status ON ingest_jobs(status);
CREATE INDEX idx_ingest_jobs_created_at ON ingest_jobs(created_at DESC);

-- Audit logs
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    action VARCHAR(255) NOT NULL,
    resource_type VARCHAR(255) NOT NULL,
    resource_id UUID NOT NULL,
    details JSONB,
    ip_address VARCHAR(45),
    user_agent TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_audit_logs_resource ON audit_logs(resource_type, resource_id);
CREATE INDEX idx_audit_logs_user ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at DESC);

-- Create enums
CREATE TYPE user_role AS ENUM ('admin', 'editor', 'viewer', 'guest');
CREATE TYPE permission_type AS ENUM ('read', 'write', 'delete', 'share', 'admin');
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_role_serialization() {
        let role = UserRole::Admin;
        let json = serde_json::to_string(&role).expect("should succeed in test");
        assert!(json.contains("Admin"));
    }

    #[test]
    fn test_permission_type_serialization() {
        let perm = PermissionType::Write;
        let json = serde_json::to_string(&perm).expect("should succeed in test");
        assert!(json.contains("Write"));
    }
}

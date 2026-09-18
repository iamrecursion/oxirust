//! Admin API endpoints for server management.
//!
//! Provides privileged routes for user administration, audit log querying,
//! server statistics, config inspection, and database maintenance.
//! All routes require `UserRole::Admin`.

use crate::{
    auth::AuthUser,
    error::{ServerError, ServerResult},
    AppState,
};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// ── Response types ────────────────────────────────────────────────────────────

/// Summary of a user account returned by admin list endpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSummary {
    /// Unique user ID.
    pub id: String,
    /// Username.
    pub username: String,
    /// User role string ("admin" | "user" | "guest").
    pub role: String,
    /// Account creation timestamp (Unix seconds).
    pub created_at: i64,
}

/// Paginated list of users returned by `GET /admin/users`.
#[derive(Debug, Serialize)]
pub struct AdminUserListResponse {
    /// The user summaries for this page.
    pub users: Vec<UserSummary>,
    /// Total number of users matching the filter.
    pub total: i64,
    /// Current page (1-based).
    pub page: i64,
    /// Page size requested.
    pub per_page: i64,
}

/// A single entry from the audit log.
#[derive(Debug, Serialize)]
pub struct AuditEntry {
    /// Row ID in the audit_log table.
    pub id: i64,
    /// User who performed the action (None for system events).
    pub user_id: Option<String>,
    /// Action description (e.g. "login", "delete_media").
    pub action: String,
    /// Resource that was acted on.
    pub resource: String,
    /// When the action occurred (Unix seconds).
    pub timestamp: i64,
    /// Source IP address, if available.
    pub ip: Option<String>,
}

/// Aggregated server statistics for admins.
#[derive(Debug, Serialize)]
pub struct AdminStats {
    /// Total registered users.
    pub total_users: i64,
    /// Total media items.
    pub total_media: i64,
    /// Total storage consumed by media files (bytes).
    pub storage_bytes_used: i64,
    /// Number of pending or processing transcode jobs.
    pub jobs_pending: i64,
}

/// A redacted view of the current server configuration.
#[derive(Debug, Serialize)]
pub struct AdminConfigView {
    /// SQLite database URL (host/path only, no credentials).
    pub database_url: String,
    /// Media storage directory.
    pub media_dir: String,
    /// Thumbnail storage directory.
    pub thumbnail_dir: String,
    /// JWT expiration in seconds.
    pub jwt_expiration: u64,
    /// Maximum upload size in bytes.
    pub max_upload_size: usize,
    /// Maximum concurrent transcode jobs.
    pub max_concurrent_jobs: usize,
    /// Rate limit (requests per minute per user).
    pub rate_limit_per_minute: u32,
    /// HLS segment duration (seconds).
    pub hls_segment_duration: u64,
    /// DASH segment duration (seconds).
    pub dash_segment_duration: u64,
    /// CORS enabled flag.
    pub enable_cors: bool,
}

// ── Query parameter structs ───────────────────────────────────────────────────

/// Query parameters for `GET /admin/users`.
#[derive(Debug, Deserialize)]
pub struct AdminUserListQuery {
    /// Page number, 1-based (default: 1).
    #[serde(default = "default_page")]
    pub page: i64,
    /// Page size (default: 20, max: 200).
    #[serde(default = "default_per_page")]
    pub per_page: i64,
    /// Optional role filter ("admin" | "user" | "guest").
    pub role: Option<String>,
}

/// Query parameters for `GET /admin/audit`.
#[derive(Debug, Deserialize)]
pub struct AuditLogQuery {
    /// Earliest timestamp (Unix seconds, inclusive).
    pub from: Option<i64>,
    /// Latest timestamp (Unix seconds, inclusive).
    pub to: Option<i64>,
    /// Filter by user ID.
    pub user_id: Option<String>,
    /// Filter by action substring.
    pub action: Option<String>,
    /// Page number, 1-based (default: 1).
    #[serde(default = "default_page")]
    pub page: i64,
    /// Page size (default: 50).
    #[serde(default = "default_audit_per_page")]
    pub per_page: i64,
}

/// Request body for `PUT /admin/users/{id}/role`.
#[derive(Debug, Deserialize)]
pub struct ChangeRoleRequest {
    /// New role string ("admin" | "user" | "guest").
    pub role: String,
}

const fn default_page() -> i64 {
    1
}

const fn default_per_page() -> i64 {
    20
}

const fn default_audit_per_page() -> i64 {
    50
}

// ── Helper ────────────────────────────────────────────────────────────────────

/// Returns `ServerError::Forbidden` when the caller is not an admin.
#[inline]
fn require_admin(auth_user: &AuthUser) -> ServerResult<()> {
    if auth_user.is_admin() {
        Ok(())
    } else {
        Err(ServerError::Forbidden(
            "Admin privileges required".to_string(),
        ))
    }
}

// ── Route handlers ────────────────────────────────────────────────────────────

/// `GET /api/v1/admin/users` — list all users with optional role filter and pagination.
pub async fn list_users(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
    Query(params): Query<AdminUserListQuery>,
) -> ServerResult<impl IntoResponse> {
    require_admin(&auth_user)?;

    let per_page = params.per_page.clamp(1, 200);
    let page = params.page.max(1);
    let offset = (page - 1) * per_page;

    // Build dynamic WHERE clause for optional role filter.
    let (where_clause, role_bind) = match &params.role {
        Some(r) => (" WHERE role = ?", Some(r.clone())),
        None => ("", None),
    };

    // `where_clause` is always one of two hardcoded literal fragments (never built
    // from `params.role`'s CONTENT), and the actual role value is always bound below
    // -- audited safe for sqlx 0.9's `SqlSafeStr` gate.
    let count_sql = format!("SELECT COUNT(*) FROM users{}", where_clause);
    let total: i64 = if let Some(ref role) = role_bind {
        crate::db::query_scalar(crate::db::AssertSqlSafe(count_sql))
            .bind(role)
            .fetch_one(state.db.pool())
            .await?
    } else {
        crate::db::query_scalar(crate::db::AssertSqlSafe(count_sql))
            .fetch_one(state.db.pool())
            .await?
    };

    let list_sql = format!(
        "SELECT id, username, role, created_at FROM users{} ORDER BY created_at DESC LIMIT ? OFFSET ?",
        where_clause
    );

    let rows = if let Some(ref role) = role_bind {
        crate::db::query(crate::db::AssertSqlSafe(list_sql))
            .bind(role)
            .bind(per_page)
            .bind(offset)
            .fetch_all(state.db.pool())
            .await?
    } else {
        crate::db::query(crate::db::AssertSqlSafe(list_sql))
            .bind(per_page)
            .bind(offset)
            .fetch_all(state.db.pool())
            .await?
    };

    let users: Vec<UserSummary> = rows
        .iter()
        .map(|row| UserSummary {
            id: row.get("id"),
            username: row.get("username"),
            role: row.get("role"),
            created_at: row.get("created_at"),
        })
        .collect();

    Ok(Json(AdminUserListResponse {
        users,
        total,
        page,
        per_page,
    }))
}

/// `PUT /api/v1/admin/users/{id}/role` — change a user's role.
pub async fn change_user_role(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
    Path(user_id): Path<String>,
    Json(body): Json<ChangeRoleRequest>,
) -> ServerResult<impl IntoResponse> {
    require_admin(&auth_user)?;

    // Validate the role string.
    let role_str = body.role.to_lowercase();
    if !matches!(role_str.as_str(), "admin" | "user" | "guest") {
        return Err(ServerError::BadRequest(format!(
            "Invalid role '{}'. Must be one of: admin, user, guest",
            role_str
        )));
    }

    // Prevent an admin from demoting themselves.
    if user_id == auth_user.user_id && role_str != "admin" {
        return Err(ServerError::BadRequest(
            "Cannot demote your own admin account".to_string(),
        ));
    }

    let now = chrono::Utc::now().timestamp();
    let result = crate::db::query("UPDATE users SET role = ?, updated_at = ? WHERE id = ?")
        .bind(&role_str)
        .bind(now)
        .bind(&user_id)
        .execute(state.db.pool())
        .await?;

    if result.rows_affected() == 0 {
        return Err(ServerError::NotFound(format!(
            "User '{}' not found",
            user_id
        )));
    }

    Ok(Json(serde_json::json!({
        "user_id": user_id,
        "new_role": role_str,
    })))
}

/// `DELETE /api/v1/admin/users/{id}` — delete a user account.
pub async fn delete_user(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
    Path(user_id): Path<String>,
) -> ServerResult<impl IntoResponse> {
    require_admin(&auth_user)?;

    if user_id == auth_user.user_id {
        return Err(ServerError::BadRequest(
            "Cannot delete your own account".to_string(),
        ));
    }

    let result = crate::db::query("DELETE FROM users WHERE id = ?")
        .bind(&user_id)
        .execute(state.db.pool())
        .await?;

    if result.rows_affected() == 0 {
        return Err(ServerError::NotFound(format!(
            "User '{}' not found",
            user_id
        )));
    }

    Ok(StatusCode::NO_CONTENT)
}

/// `GET /api/v1/admin/stats` — server-wide statistics.
pub async fn get_admin_stats(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
) -> ServerResult<impl IntoResponse> {
    require_admin(&auth_user)?;

    let total_users: i64 = crate::db::query_scalar("SELECT COUNT(*) FROM users")
        .fetch_one(state.db.pool())
        .await?;

    let total_media: i64 = crate::db::query_scalar("SELECT COUNT(*) FROM media")
        .fetch_one(state.db.pool())
        .await?;

    let storage_bytes_used: Option<i64> =
        crate::db::query_scalar("SELECT SUM(file_size) FROM media")
            .fetch_one(state.db.pool())
            .await?;

    let jobs_pending: i64 = crate::db::query_scalar(
        "SELECT COUNT(*) FROM transcode_jobs WHERE status IN ('queued', 'processing')",
    )
    .fetch_one(state.db.pool())
    .await?;

    Ok(Json(AdminStats {
        total_users,
        total_media,
        storage_bytes_used: storage_bytes_used.unwrap_or(0),
        jobs_pending,
    }))
}

/// `GET /api/v1/admin/audit` — paginated audit log with optional filters.
///
/// Queries the `audit_log` table (created lazily on first write by the
/// [`log_audit_event`] helper).  Returns an empty list if the table has not
/// yet been created.
pub async fn get_audit_log(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
    Query(params): Query<AuditLogQuery>,
) -> ServerResult<impl IntoResponse> {
    require_admin(&auth_user)?;

    // Ensure the table exists (idempotent).
    ensure_audit_table(state.db.pool()).await?;

    let per_page = params.per_page.clamp(1, 500);
    let page = params.page.max(1);
    let offset = (page - 1) * per_page;

    // Build WHERE clauses dynamically.
    let mut conditions: Vec<String> = Vec::new();
    if params.from.is_some() {
        conditions.push("timestamp >= ?".to_string());
    }
    if params.to.is_some() {
        conditions.push("timestamp <= ?".to_string());
    }
    if params.user_id.is_some() {
        conditions.push("user_id = ?".to_string());
    }
    if params.action.is_some() {
        conditions.push("action LIKE ?".to_string());
    }

    let where_sql = if conditions.is_empty() {
        String::new()
    } else {
        format!(" WHERE {}", conditions.join(" AND "))
    };

    // Helper closure: bind optional parameters in order.
    macro_rules! bind_filters {
        ($query:expr) => {{
            let mut q = $query;
            if let Some(v) = params.from {
                q = q.bind(v);
            }
            if let Some(v) = params.to {
                q = q.bind(v);
            }
            if let Some(ref v) = params.user_id {
                q = q.bind(v.clone());
            }
            if let Some(ref v) = params.action {
                q = q.bind(format!("%{}%", v));
            }
            q
        }};
    }

    // `where_sql` is built entirely from hardcoded literal condition fragments
    // (`"timestamp >= ?"` etc, chosen only by whether each filter is `Some`), and
    // every actual filter value is bound via `bind_filters!` above -- audited safe
    // for sqlx 0.9's `SqlSafeStr` gate.
    let count_sql = format!("SELECT COUNT(*) FROM audit_log{}", where_sql);
    let total: i64 = bind_filters!(crate::db::query_scalar(crate::db::AssertSqlSafe(count_sql)))
        .fetch_one(state.db.pool())
        .await?;

    let list_sql = format!(
        "SELECT id, user_id, action, resource, timestamp, ip \
         FROM audit_log{} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
        where_sql
    );
    let rows = bind_filters!(crate::db::query(crate::db::AssertSqlSafe(list_sql)))
        .bind(per_page)
        .bind(offset)
        .fetch_all(state.db.pool())
        .await?;

    let entries: Vec<AuditEntry> = rows
        .iter()
        .map(|row| AuditEntry {
            id: row.get("id"),
            user_id: row.get("user_id"),
            action: row.get("action"),
            resource: row.get("resource"),
            timestamp: row.get("timestamp"),
            ip: row.get("ip"),
        })
        .collect();

    Ok(Json(serde_json::json!({
        "entries": entries,
        "total": total,
        "page": page,
        "per_page": per_page,
    })))
}

/// `GET /api/v1/admin/config` — current server config (secrets redacted).
pub async fn get_config(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
) -> ServerResult<impl IntoResponse> {
    require_admin(&auth_user)?;

    let cfg = &state.config;
    Ok(Json(AdminConfigView {
        database_url: redact_url(&cfg.database_url),
        media_dir: cfg.media_dir.display().to_string(),
        thumbnail_dir: cfg.thumbnail_dir.display().to_string(),
        jwt_expiration: cfg.jwt_expiration,
        max_upload_size: cfg.max_upload_size,
        max_concurrent_jobs: cfg.max_concurrent_jobs,
        rate_limit_per_minute: cfg.rate_limit_per_minute,
        hls_segment_duration: cfg.hls_segment_duration,
        dash_segment_duration: cfg.dash_segment_duration,
        enable_cors: cfg.enable_cors,
    }))
}

/// `POST /api/v1/admin/maintenance/vacuum` — run SQLite VACUUM.
pub async fn vacuum_db(
    State(state): State<Arc<AppState>>,
    auth_user: AuthUser,
) -> ServerResult<impl IntoResponse> {
    require_admin(&auth_user)?;

    crate::db::query("VACUUM")
        .execute(state.db.pool())
        .await
        .map_err(|e| ServerError::Internal(format!("VACUUM failed: {e}")))?;

    Ok(Json(
        serde_json::json!({ "status": "ok", "message": "VACUUM completed" }),
    ))
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Redacts sensitive parts of a URL (passwords, tokens).
fn redact_url(url: &str) -> String {
    // For SQLite URLs like "sqlite:/path/to/db" there is nothing to redact;
    // for hypothetical postgres URLs we strip the password portion.
    if let Ok(parsed) = url::Url::parse(url) {
        if parsed.password().is_some() {
            let mut out = parsed.clone();
            let _ = out.set_password(Some("***"));
            return out.to_string();
        }
    }
    url.to_string()
}

/// Creates the `audit_log` table if it does not already exist.
pub async fn ensure_audit_table(pool: &crate::db::SqlitePool) -> ServerResult<()> {
    crate::db::query(
        r"
        CREATE TABLE IF NOT EXISTS audit_log (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id   TEXT,
            action    TEXT NOT NULL,
            resource  TEXT NOT NULL DEFAULT '',
            timestamp INTEGER NOT NULL,
            ip        TEXT
        )
        ",
    )
    .execute(pool)
    .await?;

    crate::db::query("CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp)")
        .execute(pool)
        .await?;

    crate::db::query("CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id)")
        .execute(pool)
        .await?;

    Ok(())
}

/// Appends a row to the `audit_log` table.
///
/// Creates the table on first use.  Failures are logged but not propagated —
/// audit writes should never abort a user-facing request.
pub async fn log_audit_event(
    pool: &crate::db::SqlitePool,
    user_id: Option<&str>,
    action: &str,
    resource: &str,
    ip: Option<&str>,
) {
    if let Err(e) = ensure_audit_table(pool).await {
        tracing::error!("audit_log table init failed: {}", e);
        return;
    }

    let now = chrono::Utc::now().timestamp();
    if let Err(e) = crate::db::query(
        "INSERT INTO audit_log (user_id, action, resource, timestamp, ip) VALUES (?, ?, ?, ?, ?)",
    )
    .bind(user_id)
    .bind(action)
    .bind(resource)
    .bind(now)
    .bind(ip)
    .execute(pool)
    .await
    {
        tracing::error!("Failed to write audit log entry: {}", e);
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Unit tests for pure helper functions ──────────────────────────────────

    #[test]
    fn test_redact_url_sqlite_passthrough() {
        let url = "sqlite:oximedia.db";
        assert_eq!(redact_url(url), url);
    }

    #[test]
    fn test_redact_url_postgres_redacts_password() {
        let url = "postgres://user:secret@localhost/db";
        let redacted = redact_url(url);
        assert!(!redacted.contains("secret"), "password must not appear");
        assert!(redacted.contains("***"), "placeholder must appear");
    }

    #[test]
    fn test_redact_url_no_password() {
        let url = "postgres://user@localhost/db";
        let redacted = redact_url(url);
        assert!(!redacted.contains("***"));
    }

    #[test]
    fn test_require_admin_rejects_non_admin() {
        let user = AuthUser {
            user_id: "u1".to_string(),
            username: "alice".to_string(),
            role: crate::models::user::UserRole::User,
        };
        assert!(require_admin(&user).is_err());
    }

    #[test]
    fn test_require_admin_accepts_admin() {
        let user = AuthUser {
            user_id: "u2".to_string(),
            username: "bob".to_string(),
            role: crate::models::user::UserRole::Admin,
        };
        assert!(require_admin(&user).is_ok());
    }

    #[test]
    fn test_user_summary_serializes() {
        let s = UserSummary {
            id: "x".to_string(),
            username: "u".to_string(),
            role: "admin".to_string(),
            created_at: 0,
        };
        let j = serde_json::to_value(&s).expect("serialize");
        assert_eq!(j["role"], "admin");
    }

    #[test]
    fn test_admin_user_list_response_fields() {
        let r = AdminUserListResponse {
            users: vec![],
            total: 100,
            page: 3,
            per_page: 20,
        };
        assert_eq!(r.total, 100);
        assert_eq!(r.page, 3);
    }

    #[test]
    fn test_audit_entry_serializes() {
        let e = AuditEntry {
            id: 1,
            user_id: Some("u1".to_string()),
            action: "login".to_string(),
            resource: "/auth/login".to_string(),
            timestamp: 1_700_000_000,
            ip: Some("127.0.0.1".to_string()),
        };
        let j = serde_json::to_value(&e).expect("serialize");
        assert_eq!(j["action"], "login");
        assert_eq!(j["ip"], "127.0.0.1");
    }

    #[test]
    fn test_admin_stats_serializes() {
        let s = AdminStats {
            total_users: 5,
            total_media: 42,
            storage_bytes_used: 1_000_000,
            jobs_pending: 2,
        };
        let j = serde_json::to_value(&s).expect("serialize");
        assert_eq!(j["total_users"], 5);
        assert_eq!(j["jobs_pending"], 2);
    }

    #[test]
    fn test_admin_config_view_serializes() {
        let v = AdminConfigView {
            database_url: "sqlite:test.db".to_string(),
            media_dir: "/var/media".to_string(),
            thumbnail_dir: "/var/thumbs".to_string(),
            jwt_expiration: 86400,
            max_upload_size: 5_368_709_120,
            max_concurrent_jobs: 4,
            rate_limit_per_minute: 120,
            hls_segment_duration: 6,
            dash_segment_duration: 4,
            enable_cors: true,
        };
        let j = serde_json::to_value(&v).expect("serialize");
        assert_eq!(j["jwt_expiration"], 86400);
        assert_eq!(j["enable_cors"], true);
    }

    #[test]
    fn test_default_pagination_values() {
        // default_page and default_per_page are const fns
        assert_eq!(default_page(), 1);
        assert_eq!(default_per_page(), 20);
        assert_eq!(default_audit_per_page(), 50);
    }

    #[tokio::test]
    async fn test_ensure_audit_table_idempotent() {
        let dir = std::env::temp_dir();
        let db_path = dir.join(format!("oximedia_admin_test_{}.db", uuid::Uuid::new_v4()));
        let url = format!("sqlite:{}", db_path.display());

        let pool = crate::db::SqlitePool::connect(&url).await.expect("connect");

        // Call twice — must not fail on second call.
        ensure_audit_table(&pool).await.expect("first call");
        ensure_audit_table(&pool).await.expect("second call");

        let count: i64 = crate::db::query_scalar("SELECT COUNT(*) FROM audit_log")
            .fetch_one(&pool)
            .await
            .expect("count");
        assert_eq!(count, 0);

        let _ = std::fs::remove_file(&db_path);
    }

    #[tokio::test]
    async fn test_log_audit_event_writes_row() {
        let dir = std::env::temp_dir();
        let db_path = dir.join(format!(
            "oximedia_audit_write_test_{}.db",
            uuid::Uuid::new_v4()
        ));
        let url = format!("sqlite:{}", db_path.display());

        let pool = crate::db::SqlitePool::connect(&url).await.expect("connect");

        log_audit_event(
            &pool,
            Some("user-1"),
            "login",
            "/auth/login",
            Some("10.0.0.1"),
        )
        .await;

        let count: i64 = crate::db::query_scalar("SELECT COUNT(*) FROM audit_log")
            .fetch_one(&pool)
            .await
            .expect("count");
        assert_eq!(count, 1);

        let _ = std::fs::remove_file(&db_path);
    }
}

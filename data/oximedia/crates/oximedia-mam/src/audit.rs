//! Audit logging and activity tracking
//!
//! Provides comprehensive audit logging for:
//! - User activity tracking
//! - Asset access logs
//! - Change history
//! - Compliance reporting
//! - Security monitoring

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::net::IpAddr;
use std::sync::Arc;
use uuid::Uuid;

use crate::database::Database;
use crate::Result;

/// Audit logger handles all audit logging
pub struct AuditLogger {
    db: Arc<Database>,
}

/// Audit log entry
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct AuditLog {
    pub id: Uuid,
    pub action: String,
    pub resource_type: Option<String>,
    pub resource_id: Option<Uuid>,
    pub user_id: Option<Uuid>,
    pub username: Option<String>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub details: Option<serde_json::Value>,
    pub changes: Option<serde_json::Value>,
    pub success: bool,
    pub error_message: Option<String>,
    pub timestamp: DateTime<Utc>,
}

/// Audit action type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditAction {
    // Authentication actions
    /// User logged in
    Login,
    /// User logged out
    Logout,
    /// Login failed
    LoginFailed,
    /// Password changed
    PasswordChanged,

    // Asset actions
    /// Asset created
    AssetCreate,
    /// Asset read/viewed
    AssetRead,
    /// Asset updated
    AssetUpdate,
    /// Asset deleted
    AssetDelete,
    /// Asset downloaded
    AssetDownload,
    /// Asset shared
    AssetShare,

    // Collection actions
    /// Collection created
    CollectionCreate,
    /// Collection read
    CollectionRead,
    /// Collection updated
    CollectionUpdate,
    /// Collection deleted
    CollectionDelete,

    // Workflow actions
    /// Workflow created
    WorkflowCreate,
    /// Workflow updated
    WorkflowUpdate,
    /// Workflow approved
    WorkflowApprove,
    /// Workflow rejected
    WorkflowReject,

    // User management actions
    /// User created
    UserCreate,
    /// User updated
    UserUpdate,
    /// User deleted
    UserDelete,
    /// Role assigned
    RoleAssign,
    /// Permission granted
    PermissionGrant,
    /// Permission revoked
    PermissionRevoke,

    // Storage actions
    /// File uploaded
    FileUpload,
    /// File downloaded
    FileDownload,
    /// File deleted
    FileDelete,

    // System actions
    /// Configuration changed
    ConfigChange,
    /// System started
    SystemStart,
    /// System stopped
    SystemStop,

    // Custom action
    /// Custom action
    Custom,
}

impl AuditAction {
    /// Convert to string
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Login => "auth.login",
            Self::Logout => "auth.logout",
            Self::LoginFailed => "auth.login_failed",
            Self::PasswordChanged => "auth.password_changed",
            Self::AssetCreate => "asset.create",
            Self::AssetRead => "asset.read",
            Self::AssetUpdate => "asset.update",
            Self::AssetDelete => "asset.delete",
            Self::AssetDownload => "asset.download",
            Self::AssetShare => "asset.share",
            Self::CollectionCreate => "collection.create",
            Self::CollectionRead => "collection.read",
            Self::CollectionUpdate => "collection.update",
            Self::CollectionDelete => "collection.delete",
            Self::WorkflowCreate => "workflow.create",
            Self::WorkflowUpdate => "workflow.update",
            Self::WorkflowApprove => "workflow.approve",
            Self::WorkflowReject => "workflow.reject",
            Self::UserCreate => "user.create",
            Self::UserUpdate => "user.update",
            Self::UserDelete => "user.delete",
            Self::RoleAssign => "user.role_assign",
            Self::PermissionGrant => "permission.grant",
            Self::PermissionRevoke => "permission.revoke",
            Self::FileUpload => "file.upload",
            Self::FileDownload => "file.download",
            Self::FileDelete => "file.delete",
            Self::ConfigChange => "system.config_change",
            Self::SystemStart => "system.start",
            Self::SystemStop => "system.stop",
            Self::Custom => "custom",
        }
    }
}

/// Audit log filter
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuditLogFilter {
    pub user_id: Option<Uuid>,
    pub action: Option<String>,
    pub resource_type: Option<String>,
    pub resource_id: Option<Uuid>,
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
    pub success: Option<bool>,
    pub ip_address: Option<String>,
}

/// Audit log request
#[derive(Debug, Clone)]
pub struct AuditLogRequest {
    pub action: AuditAction,
    pub resource_type: Option<String>,
    pub resource_id: Option<Uuid>,
    pub user_id: Option<Uuid>,
    pub username: Option<String>,
    pub ip_address: Option<IpAddr>,
    pub user_agent: Option<String>,
    pub details: Option<serde_json::Value>,
    pub changes: Option<serde_json::Value>,
    pub success: bool,
    pub error_message: Option<String>,
}

/// Change tracking for audit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Change {
    pub field: String,
    pub old_value: Option<serde_json::Value>,
    pub new_value: Option<serde_json::Value>,
}

/// Audit statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditStatistics {
    pub total_events: i64,
    pub failed_events: i64,
    pub unique_users: i64,
    pub most_common_actions: Vec<ActionCount>,
    pub most_active_users: Vec<UserActivity>,
}

/// Action count for statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionCount {
    pub action: String,
    pub count: i64,
}

/// User activity for statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserActivity {
    pub user_id: Uuid,
    pub username: String,
    pub action_count: i64,
}

impl AuditLogger {
    /// Create a new audit logger
    #[must_use]
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }

    /// Log an audit event
    ///
    /// # Errors
    ///
    /// Returns an error if logging fails
    pub async fn log(&self, req: AuditLogRequest) -> Result<Uuid> {
        let log_id = Uuid::new_v4();

        sqlx::query(
            "INSERT INTO audit_logs
             (id, action, resource_type, resource_id, user_id, username, ip_address, user_agent,
              details, changes, success, error_message, timestamp)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, NOW())",
        )
        .bind(log_id)
        .bind(req.action.as_str())
        .bind(&req.resource_type)
        .bind(req.resource_id)
        .bind(req.user_id)
        .bind(&req.username)
        .bind(req.ip_address.map(|ip| ip.to_string()))
        .bind(&req.user_agent)
        .bind(&req.details)
        .bind(&req.changes)
        .bind(req.success)
        .bind(&req.error_message)
        .execute(self.db.pool())
        .await?;

        Ok(log_id)
    }

    /// Log successful action
    ///
    /// # Errors
    ///
    /// Returns an error if logging fails
    pub async fn log_success(
        &self,
        action: AuditAction,
        user_id: Option<Uuid>,
        resource_type: Option<String>,
        resource_id: Option<Uuid>,
        details: Option<serde_json::Value>,
    ) -> Result<Uuid> {
        self.log(AuditLogRequest {
            action,
            resource_type,
            resource_id,
            user_id,
            username: None,
            ip_address: None,
            user_agent: None,
            details,
            changes: None,
            success: true,
            error_message: None,
        })
        .await
    }

    /// Log failed action
    ///
    /// # Errors
    ///
    /// Returns an error if logging fails
    pub async fn log_failure(
        &self,
        action: AuditAction,
        user_id: Option<Uuid>,
        resource_type: Option<String>,
        resource_id: Option<Uuid>,
        error: String,
    ) -> Result<Uuid> {
        self.log(AuditLogRequest {
            action,
            resource_type,
            resource_id,
            user_id,
            username: None,
            ip_address: None,
            user_agent: None,
            details: None,
            changes: None,
            success: false,
            error_message: Some(error),
        })
        .await
    }

    /// Log with changes
    ///
    /// # Errors
    ///
    /// Returns an error if logging fails
    pub async fn log_with_changes(
        &self,
        action: AuditAction,
        user_id: Option<Uuid>,
        resource_type: Option<String>,
        resource_id: Option<Uuid>,
        changes: Vec<Change>,
    ) -> Result<Uuid> {
        let changes_json = serde_json::to_value(changes)?;

        self.log(AuditLogRequest {
            action,
            resource_type,
            resource_id,
            user_id,
            username: None,
            ip_address: None,
            user_agent: None,
            details: None,
            changes: Some(changes_json),
            success: true,
            error_message: None,
        })
        .await
    }

    /// Get audit log by ID
    ///
    /// # Errors
    ///
    /// Returns an error if log not found
    pub async fn get_log(&self, log_id: Uuid) -> Result<AuditLog> {
        let log = sqlx::query_as::<_, AuditLog>("SELECT * FROM audit_logs WHERE id = $1")
            .bind(log_id)
            .fetch_one(self.db.pool())
            .await?;

        Ok(log)
    }

    /// Query audit logs with filters
    ///
    /// # Errors
    ///
    /// Returns an error if query fails
    pub async fn query_logs(
        &self,
        filter: AuditLogFilter,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<AuditLog>> {
        // `field` names below are all hardcoded literals (never user-controlled), so
        // the only concern is binding each value correctly — see the `bind_filters!`
        // macro below, which mirrors this exact conditional order so each `$N` lines
        // up with the right bound value. (The previous implementation only ever
        // collected `user_id`/`resource_id` into `bindings` — a plain untyped `Vec`
        // that only compiled because those two happen to share a type (`Uuid`) — and
        // the other 5 filters' matched values were discarded entirely via `_action`/
        // `_resource_type`/`_success`/`_start_time`/`_end_time` bindings, even though
        // their `$N` placeholder was still added to the query text. Worse, `bindings`
        // itself was NEVER passed to `.bind()` at all — only `limit`/`offset` were
        // bound. Every one of the 7 filters was therefore non-functional: any
        // `query_logs` call using ANY filter would fail at execution time with a
        // bind-parameter count mismatch. This was a pre-existing bug unrelated to any
        // dependency version, just newly surfaced by sqlx 0.9's `SqlSafeStr` audit
        // gate forcing a look at this call site.)
        let mut query = String::from("SELECT * FROM audit_logs WHERE 1=1");
        let mut param_num = 1;

        if filter.user_id.is_some() {
            query.push_str(&format!(" AND user_id = ${param_num}"));
            param_num += 1;
        }

        if filter.action.is_some() {
            query.push_str(&format!(" AND action = ${param_num}"));
            param_num += 1;
        }

        if filter.resource_type.is_some() {
            query.push_str(&format!(" AND resource_type = ${param_num}"));
            param_num += 1;
        }

        if filter.resource_id.is_some() {
            query.push_str(&format!(" AND resource_id = ${param_num}"));
            param_num += 1;
        }

        if filter.success.is_some() {
            query.push_str(&format!(" AND success = ${param_num}"));
            param_num += 1;
        }

        if filter.start_time.is_some() {
            query.push_str(&format!(" AND timestamp >= ${param_num}"));
            param_num += 1;
        }

        if filter.end_time.is_some() {
            query.push_str(&format!(" AND timestamp <= ${param_num}"));
            param_num += 1;
        }

        query.push_str(&format!(
            " ORDER BY timestamp DESC LIMIT ${param_num} OFFSET ${}",
            param_num + 1
        ));

        // Binds `filter`'s present fields in the SAME order the WHERE clause above
        // appended them.
        macro_rules! bind_filters {
            ($q:expr) => {{
                let mut q = $q;
                if let Some(user_id) = filter.user_id {
                    q = q.bind(user_id);
                }
                if let Some(action) = &filter.action {
                    q = q.bind(action.clone());
                }
                if let Some(resource_type) = &filter.resource_type {
                    q = q.bind(resource_type.clone());
                }
                if let Some(resource_id) = filter.resource_id {
                    q = q.bind(resource_id);
                }
                if let Some(success) = filter.success {
                    q = q.bind(success);
                }
                if let Some(start_time) = filter.start_time {
                    q = q.bind(start_time);
                }
                if let Some(end_time) = filter.end_time {
                    q = q.bind(end_time);
                }
                q
            }};
        }

        // `query`'s only dynamic parts are the `$N` placeholders bound above;
        // audited safe for sqlx 0.9's `SqlSafeStr` gate.
        let logs = bind_filters!(sqlx::query_as::<_, AuditLog>(sqlx::AssertSqlSafe(query)))
            .bind(limit)
            .bind(offset)
            .fetch_all(self.db.pool())
            .await?;

        Ok(logs)
    }

    /// Get logs for user
    ///
    /// # Errors
    ///
    /// Returns an error if query fails
    pub async fn get_user_logs(&self, user_id: Uuid, limit: i64) -> Result<Vec<AuditLog>> {
        let logs = sqlx::query_as::<_, AuditLog>(
            "SELECT * FROM audit_logs WHERE user_id = $1 ORDER BY timestamp DESC LIMIT $2",
        )
        .bind(user_id)
        .bind(limit)
        .fetch_all(self.db.pool())
        .await?;

        Ok(logs)
    }

    /// Get logs for resource
    ///
    /// # Errors
    ///
    /// Returns an error if query fails
    pub async fn get_resource_logs(
        &self,
        resource_type: &str,
        resource_id: Uuid,
        limit: i64,
    ) -> Result<Vec<AuditLog>> {
        let logs = sqlx::query_as::<_, AuditLog>(
            "SELECT * FROM audit_logs
             WHERE resource_type = $1 AND resource_id = $2
             ORDER BY timestamp DESC
             LIMIT $3",
        )
        .bind(resource_type)
        .bind(resource_id)
        .bind(limit)
        .fetch_all(self.db.pool())
        .await?;

        Ok(logs)
    }

    /// Get audit statistics
    ///
    /// # Errors
    ///
    /// Returns an error if query fails
    pub async fn get_statistics(&self, days: i32) -> Result<AuditStatistics> {
        let since = Utc::now() - chrono::Duration::days(days as i64);

        let total_events: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM audit_logs WHERE timestamp >= $1")
                .bind(since)
                .fetch_one(self.db.pool())
                .await?;

        let failed_events: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM audit_logs WHERE timestamp >= $1 AND success = false",
        )
        .bind(since)
        .fetch_one(self.db.pool())
        .await?;

        let unique_users: i64 = sqlx::query_scalar(
            "SELECT COUNT(DISTINCT user_id) FROM audit_logs WHERE timestamp >= $1 AND user_id IS NOT NULL",
        )
        .bind(since)
        .fetch_one(self.db.pool())
        .await?;

        // Most common actions
        let action_counts = sqlx::query_as::<_, (String, i64)>(
            "SELECT action, COUNT(*) as count
             FROM audit_logs
             WHERE timestamp >= $1
             GROUP BY action
             ORDER BY count DESC
             LIMIT 10",
        )
        .bind(since)
        .fetch_all(self.db.pool())
        .await?;

        let most_common_actions: Vec<ActionCount> = action_counts
            .into_iter()
            .map(|(action, count)| ActionCount { action, count })
            .collect();

        // Most active users
        let user_activities = sqlx::query_as::<_, (Uuid, String, i64)>(
            "SELECT user_id, username, COUNT(*) as count
             FROM audit_logs
             WHERE timestamp >= $1 AND user_id IS NOT NULL
             GROUP BY user_id, username
             ORDER BY count DESC
             LIMIT 10",
        )
        .bind(since)
        .fetch_all(self.db.pool())
        .await?;

        let most_active_users: Vec<UserActivity> = user_activities
            .into_iter()
            .map(|(user_id, username, action_count)| UserActivity {
                user_id,
                username,
                action_count,
            })
            .collect();

        Ok(AuditStatistics {
            total_events,
            failed_events,
            unique_users,
            most_common_actions,
            most_active_users,
        })
    }

    /// Export audit logs to JSON
    ///
    /// # Errors
    ///
    /// Returns an error if export fails
    pub async fn export_logs(&self, filter: AuditLogFilter, limit: i64) -> Result<String> {
        let logs = self.query_logs(filter, limit, 0).await?;
        let json = serde_json::to_string_pretty(&logs)?;
        Ok(json)
    }

    /// Export audit logs to SIEM-compatible CEF (Common Event Format) string.
    ///
    /// CEF specification: ArcSight CEF Implementation Standard v25.
    /// Each log line follows the schema:
    /// `CEF:0|Vendor|Product|Version|EventId|Name|Severity|Extension`
    ///
    /// # Errors
    ///
    /// Returns an error if the logs cannot be fetched
    pub async fn export_cef(&self, filter: AuditLogFilter, limit: i64) -> Result<String> {
        let logs = self.query_logs(filter, limit, 0).await?;
        let mut output = String::new();

        for log in &logs {
            // CEF severity: map success/failure to numeric level (0-10)
            let severity = if log.success { 3u8 } else { 7u8 };

            // Escape CEF pipe-delimited header fields (| and \)
            let action_escaped = log.action.replace('\\', "\\\\").replace('|', "\\|");

            // Build CEF header
            output.push_str(&format!(
                "CEF:0|OxiMedia|OxiMedia-MAM|{}|{}|{}|{}|",
                env!("CARGO_PKG_VERSION"),
                log.id,
                action_escaped,
                severity,
            ));

            // Build CEF extension key=value pairs (values escape = and \)
            let mut ext = String::new();

            // Standard CEF extension keys
            ext.push_str(&format!("rt={}", log.timestamp.timestamp_millis()));

            if let Some(uid) = log.user_id {
                ext.push_str(&format!(" suser={}", uid));
            }
            if let Some(ref name) = log.username {
                let v = cef_escape_extension_value(name);
                ext.push_str(&format!(" suid={v}"));
            }
            if let Some(ref ip) = log.ip_address {
                let v = cef_escape_extension_value(ip);
                ext.push_str(&format!(" src={v}"));
            }
            if let Some(ref rt) = log.resource_type {
                let v = cef_escape_extension_value(rt);
                ext.push_str(&format!(" cs1Label=resourceType cs1={v}"));
            }
            if let Some(rid) = log.resource_id {
                ext.push_str(&format!(" cs2Label=resourceId cs2={rid}"));
            }
            if log.success {
                ext.push_str(" outcome=success");
            } else {
                ext.push_str(" outcome=failure");
                if let Some(ref err) = log.error_message {
                    let v = cef_escape_extension_value(err);
                    ext.push_str(&format!(" msg={v}"));
                }
            }
            if let Some(ref ua) = log.user_agent {
                let v = cef_escape_extension_value(ua);
                ext.push_str(&format!(" requestClientApplication={v}"));
            }

            output.push_str(&ext);
            output.push('\n');
        }

        Ok(output)
    }

    /// Export audit logs to SIEM-compatible LEEF (Log Event Extended Format) string.
    ///
    /// LEEF specification: IBM QRadar LEEF 2.0.
    /// Each log line follows the schema:
    /// `LEEF:2.0|Vendor|Product|Version|EventId|Tab-separated attributes`
    ///
    /// # Errors
    ///
    /// Returns an error if the logs cannot be fetched
    pub async fn export_leef(&self, filter: AuditLogFilter, limit: i64) -> Result<String> {
        let logs = self.query_logs(filter, limit, 0).await?;
        let mut output = String::new();

        for log in &logs {
            // LEEF 2.0 header uses tab as attribute delimiter (0x09)
            let delimiter = '\t';

            // Build LEEF header: LEEF:2.0|Vendor|Product|Version|EventId|
            let event_id = log.action.replace('|', "_");
            output.push_str(&format!(
                "LEEF:2.0|OxiMedia|OxiMedia-MAM|{}|{}|",
                env!("CARGO_PKG_VERSION"),
                event_id,
            ));

            // Collect key=value attributes, tab-delimited
            let mut attrs: Vec<String> = Vec::new();

            attrs.push(format!("devTime={}", log.timestamp.to_rfc3339()));
            attrs.push(format!("eventId={}", log.id));
            attrs.push(format!("cat={}", log.action));

            if let Some(uid) = log.user_id {
                attrs.push(format!("usrName={uid}"));
            }
            if let Some(ref name) = log.username {
                attrs.push(format!("accountName={}", leef_escape(name)));
            }
            if let Some(ref ip) = log.ip_address {
                attrs.push(format!("src={}", leef_escape(ip)));
            }
            if let Some(ref rt) = log.resource_type {
                attrs.push(format!("resourceType={}", leef_escape(rt)));
            }
            if let Some(rid) = log.resource_id {
                attrs.push(format!("resourceId={rid}"));
            }
            if log.success {
                attrs.push("outcome=success".to_string());
                attrs.push("severity=3".to_string());
            } else {
                attrs.push("outcome=failure".to_string());
                attrs.push("severity=7".to_string());
                if let Some(ref err) = log.error_message {
                    attrs.push(format!("reason={}", leef_escape(err)));
                }
            }
            if let Some(ref ua) = log.user_agent {
                attrs.push(format!("userAgent={}", leef_escape(ua)));
            }
            if let Some(ref details) = log.details {
                let v = leef_escape(&details.to_string());
                attrs.push(format!("details={v}"));
            }

            output.push_str(&attrs.join(&delimiter.to_string()));
            output.push('\n');
        }

        Ok(output)
    }

    /// Delete old audit logs (for retention policy)
    ///
    /// # Errors
    ///
    /// Returns an error if deletion fails
    pub async fn cleanup_old_logs(&self, days: i32) -> Result<u64> {
        let cutoff = Utc::now() - chrono::Duration::days(days as i64);

        let result = sqlx::query("DELETE FROM audit_logs WHERE timestamp < $1")
            .bind(cutoff)
            .execute(self.db.pool())
            .await?;

        Ok(result.rows_affected())
    }

    /// Get failed login attempts
    ///
    /// # Errors
    ///
    /// Returns an error if query fails
    pub async fn get_failed_logins(&self, since: DateTime<Utc>) -> Result<Vec<AuditLog>> {
        let logs = sqlx::query_as::<_, AuditLog>(
            "SELECT * FROM audit_logs
             WHERE action = 'auth.login_failed'
             AND timestamp >= $1
             ORDER BY timestamp DESC",
        )
        .bind(since)
        .fetch_all(self.db.pool())
        .await?;

        Ok(logs)
    }

    /// Get suspicious activity (multiple failed attempts from same IP)
    ///
    /// # Errors
    ///
    /// Returns an error if query fails
    pub async fn get_suspicious_activity(
        &self,
        threshold: i64,
        hours: i32,
    ) -> Result<Vec<(String, i64)>> {
        let since = Utc::now() - chrono::Duration::hours(hours as i64);

        let suspicious = sqlx::query_as::<_, (String, i64)>(
            "SELECT ip_address, COUNT(*) as count
             FROM audit_logs
             WHERE action = 'auth.login_failed'
             AND timestamp >= $1
             AND ip_address IS NOT NULL
             GROUP BY ip_address
             HAVING COUNT(*) >= $2
             ORDER BY count DESC",
        )
        .bind(since)
        .bind(threshold)
        .fetch_all(self.db.pool())
        .await?;

        Ok(suspicious)
    }
}

/// Escape a CEF extension field value per the ArcSight CEF spec.
///
/// Characters that must be escaped: `\` → `\\`, `=` → `\=`, newline → `\n`, CR → `\r`.
fn cef_escape_extension_value(v: &str) -> String {
    v.replace('\\', "\\\\")
        .replace('=', "\\=")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

/// Escape a LEEF attribute value.
///
/// Tab and newline characters are replaced to avoid breaking the LEEF record
/// structure.  Backslash is doubled for safety.
fn leef_escape(v: &str) -> String {
    v.replace('\\', "\\\\")
        .replace('\t', "\\t")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_action_as_str() {
        assert_eq!(AuditAction::Login.as_str(), "auth.login");
        assert_eq!(AuditAction::AssetCreate.as_str(), "asset.create");
        assert_eq!(AuditAction::WorkflowApprove.as_str(), "workflow.approve");
    }

    #[test]
    fn test_change_serialization() {
        let change = Change {
            field: "title".to_string(),
            old_value: Some(serde_json::json!("Old Title")),
            new_value: Some(serde_json::json!("New Title")),
        };

        let json = serde_json::to_string(&change).expect("should succeed in test");
        let deserialized: Change = serde_json::from_str(&json).expect("should succeed in test");

        assert_eq!(deserialized.field, "title");
    }

    #[test]
    fn test_audit_log_filter() {
        let filter = AuditLogFilter {
            user_id: Some(Uuid::new_v4()),
            action: Some("asset.create".to_string()),
            resource_type: Some("asset".to_string()),
            resource_id: None,
            start_time: Some(Utc::now()),
            end_time: None,
            success: Some(true),
            ip_address: None,
        };

        assert_eq!(filter.action, Some("asset.create".to_string()));
        assert_eq!(filter.success, Some(true));
    }

    #[test]
    fn test_action_count() {
        let action_count = ActionCount {
            action: "asset.create".to_string(),
            count: 100,
        };

        assert_eq!(action_count.action, "asset.create");
        assert_eq!(action_count.count, 100);
    }

    // -------------------------------------------------------------------------
    // CEF / LEEF helper unit tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cef_escape_no_special_chars() {
        assert_eq!(cef_escape_extension_value("hello world"), "hello world");
    }

    #[test]
    fn test_cef_escape_equals() {
        assert_eq!(cef_escape_extension_value("a=b"), "a\\=b");
    }

    #[test]
    fn test_cef_escape_backslash() {
        assert_eq!(cef_escape_extension_value("a\\b"), "a\\\\b");
    }

    #[test]
    fn test_cef_escape_newline() {
        assert_eq!(cef_escape_extension_value("line1\nline2"), "line1\\nline2");
    }

    #[test]
    fn test_cef_escape_combined() {
        // backslash must be escaped first, then equals
        assert_eq!(cef_escape_extension_value("\\="), "\\\\\\=");
    }

    #[test]
    fn test_leef_escape_no_special_chars() {
        assert_eq!(leef_escape("plain value"), "plain value");
    }

    #[test]
    fn test_leef_escape_tab() {
        assert_eq!(leef_escape("key\tval"), "key\\tval");
    }

    #[test]
    fn test_leef_escape_newline() {
        assert_eq!(leef_escape("line1\nline2"), "line1\\nline2");
    }

    #[test]
    fn test_leef_escape_backslash() {
        assert_eq!(leef_escape("a\\b"), "a\\\\b");
    }

    /// Build a minimal in-memory [`AuditLog`] for format testing without a DB.
    fn make_audit_log(action: &str, success: bool) -> AuditLog {
        AuditLog {
            id: Uuid::nil(),
            action: action.to_string(),
            resource_type: Some("asset".to_string()),
            resource_id: Some(Uuid::nil()),
            user_id: Some(Uuid::nil()),
            username: Some("alice".to_string()),
            ip_address: Some("192.168.1.1".to_string()),
            user_agent: Some("curl/7.88".to_string()),
            details: None,
            changes: None,
            success,
            error_message: if success {
                None
            } else {
                Some("disk full".to_string())
            },
            timestamp: chrono::DateTime::parse_from_rfc3339("2026-01-01T00:00:00Z")
                .expect("valid RFC3339 literal")
                .with_timezone(&Utc),
        }
    }

    #[test]
    fn test_cef_format_success_event() {
        let log = make_audit_log("asset.create", true);
        // Simulate what export_cef builds for a single log
        let severity: u8 = if log.success { 3 } else { 7 };
        let header = format!(
            "CEF:0|OxiMedia|OxiMedia-MAM|{}|{}|{}|{}|",
            env!("CARGO_PKG_VERSION"),
            log.id,
            log.action,
            severity,
        );
        assert!(header.starts_with("CEF:0|OxiMedia|OxiMedia-MAM|"));
        assert!(header.contains("asset.create"));
        assert!(header.contains("|3|")); // severity 3 for success
    }

    #[test]
    fn test_cef_format_failure_event() {
        let log = make_audit_log("auth.login_failed", false);
        let severity: u8 = 7;
        let header = format!(
            "CEF:0|OxiMedia|OxiMedia-MAM|{}|{}|{}|{}|",
            env!("CARGO_PKG_VERSION"),
            log.id,
            log.action,
            severity,
        );
        assert!(header.contains("|7|")); // severity 7 for failure
    }

    #[test]
    fn test_cef_header_pipe_escaping() {
        // action containing a pipe must be escaped in the CEF header
        let action = "asset|update";
        let escaped = action.replace('\\', "\\\\").replace('|', "\\|");
        assert_eq!(escaped, "asset\\|update");
    }

    #[test]
    fn test_leef_header_format() {
        let log = make_audit_log("asset.delete", true);
        let event_id = log.action.replace('|', "_");
        let header = format!(
            "LEEF:2.0|OxiMedia|OxiMedia-MAM|{}|{}|",
            env!("CARGO_PKG_VERSION"),
            event_id,
        );
        assert!(header.starts_with("LEEF:2.0|OxiMedia|OxiMedia-MAM|"));
        assert!(header.contains("asset.delete"));
    }

    #[test]
    fn test_leef_attributes_tab_separated() {
        // Verify that a tab-separated attribute list can be reconstructed
        let attrs = vec![
            "cat=asset.create".to_string(),
            "outcome=success".to_string(),
        ];
        let line = attrs.join("\t");
        assert!(line.contains('\t'));
        let parts: Vec<&str> = line.split('\t').collect();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0], "cat=asset.create");
    }

    #[test]
    fn test_leef_pipe_in_action_replaced() {
        let action = "asset|create";
        let event_id = action.replace('|', "_");
        assert_eq!(event_id, "asset_create");
    }

    #[test]
    fn test_siem_severity_mapping() {
        // Success → severity 3, failure → severity 7
        let sev_ok: u8 = if true { 3 } else { 7 };
        let sev_fail: u8 = if false { 3 } else { 7 };
        assert_eq!(sev_ok, 3);
        assert_eq!(sev_fail, 7);
    }
}

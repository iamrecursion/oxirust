//! Access control and permissions system
//!
//! Provides comprehensive Role-Based Access Control (RBAC) with:
//! - User roles and permissions
//! - Asset-level permissions
//! - Collection-level permissions
//! - Permission inheritance
//! - Group management
//! - Permission caching for performance

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::database::Database;
use crate::{MamError, Result};

/// Permission manager handles access control
pub struct PermissionManager {
    db: Arc<Database>,
    /// Cache of user permissions for fast lookup
    permission_cache: Arc<RwLock<HashMap<Uuid, UserPermissions>>>,
}

/// User account
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct User {
    pub id: Uuid,
    pub username: String,
    pub email: String,
    pub password_hash: String,
    pub full_name: Option<String>,
    pub role: String,
    pub is_active: bool,
    pub is_admin: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_login: Option<DateTime<Utc>>,
}

/// User role with permissions
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Role {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub permissions: Vec<String>,
    pub is_system: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// System-defined roles
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SystemRole {
    /// Full system access
    Admin,
    /// Can manage assets and workflows
    Editor,
    /// Can view and comment
    Viewer,
    /// Can only view approved content
    Guest,
    /// Custom role
    Custom,
}

impl SystemRole {
    /// Get role name
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Admin => "admin",
            Self::Editor => "editor",
            Self::Viewer => "viewer",
            Self::Guest => "guest",
            Self::Custom => "custom",
        }
    }

    /// Get default permissions for role
    #[must_use]
    pub fn default_permissions(&self) -> Vec<Permission> {
        match self {
            Self::Admin => vec![
                Permission::SystemAdmin,
                Permission::AssetCreate,
                Permission::AssetRead,
                Permission::AssetUpdate,
                Permission::AssetDelete,
                Permission::CollectionCreate,
                Permission::CollectionRead,
                Permission::CollectionUpdate,
                Permission::CollectionDelete,
                Permission::WorkflowCreate,
                Permission::WorkflowRead,
                Permission::WorkflowUpdate,
                Permission::WorkflowDelete,
                Permission::UserManage,
                Permission::RoleManage,
            ],
            Self::Editor => vec![
                Permission::AssetCreate,
                Permission::AssetRead,
                Permission::AssetUpdate,
                Permission::CollectionCreate,
                Permission::CollectionRead,
                Permission::CollectionUpdate,
                Permission::WorkflowCreate,
                Permission::WorkflowRead,
                Permission::WorkflowUpdate,
            ],
            Self::Viewer => vec![
                Permission::AssetRead,
                Permission::CollectionRead,
                Permission::WorkflowRead,
            ],
            Self::Guest => vec![Permission::AssetRead],
            Self::Custom => vec![],
        }
    }
}

/// Permission types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Permission {
    // System permissions
    /// Full system administration
    SystemAdmin,

    // Asset permissions
    /// Create new assets
    AssetCreate,
    /// Read/view assets
    AssetRead,
    /// Update asset metadata
    AssetUpdate,
    /// Delete assets
    AssetDelete,
    /// Download original files
    AssetDownload,
    /// Share assets externally
    AssetShare,

    // Collection permissions
    /// Create collections
    CollectionCreate,
    /// Read collections
    CollectionRead,
    /// Update collections
    CollectionUpdate,
    /// Delete collections
    CollectionDelete,

    // Workflow permissions
    /// Create workflows
    WorkflowCreate,
    /// Read workflows
    WorkflowRead,
    /// Update workflows
    WorkflowUpdate,
    /// Delete workflows
    WorkflowDelete,
    /// Approve workflow items
    WorkflowApprove,

    // User management permissions
    /// Manage users
    UserManage,
    /// Manage roles
    RoleManage,
    /// View audit logs
    AuditView,

    // Proxy permissions
    /// Generate proxies
    ProxyGenerate,
    /// Delete proxies
    ProxyDelete,
}

impl Permission {
    /// Convert to string for database storage
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::SystemAdmin => "system:admin",
            Self::AssetCreate => "asset:create",
            Self::AssetRead => "asset:read",
            Self::AssetUpdate => "asset:update",
            Self::AssetDelete => "asset:delete",
            Self::AssetDownload => "asset:download",
            Self::AssetShare => "asset:share",
            Self::CollectionCreate => "collection:create",
            Self::CollectionRead => "collection:read",
            Self::CollectionUpdate => "collection:update",
            Self::CollectionDelete => "collection:delete",
            Self::WorkflowCreate => "workflow:create",
            Self::WorkflowRead => "workflow:read",
            Self::WorkflowUpdate => "workflow:update",
            Self::WorkflowDelete => "workflow:delete",
            Self::WorkflowApprove => "workflow:approve",
            Self::UserManage => "user:manage",
            Self::RoleManage => "role:manage",
            Self::AuditView => "audit:view",
            Self::ProxyGenerate => "proxy:generate",
            Self::ProxyDelete => "proxy:delete",
        }
    }
}

impl std::str::FromStr for Permission {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "system:admin" => Ok(Self::SystemAdmin),
            "asset:create" => Ok(Self::AssetCreate),
            "asset:read" => Ok(Self::AssetRead),
            "asset:update" => Ok(Self::AssetUpdate),
            "asset:delete" => Ok(Self::AssetDelete),
            "asset:download" => Ok(Self::AssetDownload),
            "asset:share" => Ok(Self::AssetShare),
            "collection:create" => Ok(Self::CollectionCreate),
            "collection:read" => Ok(Self::CollectionRead),
            "collection:update" => Ok(Self::CollectionUpdate),
            "collection:delete" => Ok(Self::CollectionDelete),
            "workflow:create" => Ok(Self::WorkflowCreate),
            "workflow:read" => Ok(Self::WorkflowRead),
            "workflow:update" => Ok(Self::WorkflowUpdate),
            "workflow:delete" => Ok(Self::WorkflowDelete),
            "workflow:approve" => Ok(Self::WorkflowApprove),
            "user:manage" => Ok(Self::UserManage),
            "role:manage" => Ok(Self::RoleManage),
            "audit:view" => Ok(Self::AuditView),
            "proxy:generate" => Ok(Self::ProxyGenerate),
            "proxy:delete" => Ok(Self::ProxyDelete),
            _ => Err(format!("Invalid permission: {s}")),
        }
    }
}

/// User permissions (cached for performance)
#[derive(Debug, Clone)]
pub struct UserPermissions {
    pub user_id: Uuid,
    pub role: String,
    pub is_admin: bool,
    pub global_permissions: HashSet<Permission>,
    pub asset_permissions: HashMap<Uuid, HashSet<Permission>>,
    pub collection_permissions: HashMap<Uuid, HashSet<Permission>>,
    pub cached_at: DateTime<Utc>,
}

/// Asset-specific permission
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct AssetPermission {
    pub id: Uuid,
    pub asset_id: Uuid,
    pub user_id: Option<Uuid>,
    pub group_id: Option<Uuid>,
    pub permission: String,
    pub granted_by: Uuid,
    pub created_at: DateTime<Utc>,
}

/// Collection-specific permission
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct CollectionPermission {
    pub id: Uuid,
    pub collection_id: Uuid,
    pub user_id: Option<Uuid>,
    pub group_id: Option<Uuid>,
    pub permission: String,
    pub granted_by: Uuid,
    pub created_at: DateTime<Utc>,
}

/// User group for permission management
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Group {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Group membership
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct GroupMembership {
    pub group_id: Uuid,
    pub user_id: Uuid,
    pub added_by: Uuid,
    pub created_at: DateTime<Utc>,
}

/// User creation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateUserRequest {
    pub username: String,
    pub email: String,
    pub password: String,
    pub full_name: Option<String>,
    pub role: String,
}

/// User update request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateUserRequest {
    pub email: Option<String>,
    pub full_name: Option<String>,
    pub role: Option<String>,
    pub is_active: Option<bool>,
}

impl PermissionManager {
    /// Create a new permission manager
    #[must_use]
    pub fn new(db: Arc<Database>) -> Self {
        Self {
            db,
            permission_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new user
    ///
    /// # Errors
    ///
    /// Returns an error if user creation fails
    pub async fn create_user(&self, req: CreateUserRequest) -> Result<User> {
        // Hash password
        let password_hash = bcrypt::hash(&req.password, bcrypt::DEFAULT_COST)
            .map_err(|e| MamError::Internal(format!("Password hashing failed: {e}")))?;

        let user = sqlx::query_as::<_, User>(
            "INSERT INTO users
             (id, username, email, password_hash, full_name, role, is_active, is_admin, created_at, updated_at)
             VALUES ($1, $2, $3, $4, $5, $6, true, $7, NOW(), NOW())
             RETURNING *",
        )
        .bind(Uuid::new_v4())
        .bind(&req.username)
        .bind(&req.email)
        .bind(&password_hash)
        .bind(&req.full_name)
        .bind(&req.role)
        .bind(req.role == "admin")
        .fetch_one(self.db.pool())
        .await?;

        Ok(user)
    }

    /// Get user by ID
    ///
    /// # Errors
    ///
    /// Returns an error if user not found
    pub async fn get_user(&self, user_id: Uuid) -> Result<User> {
        let user = sqlx::query_as::<_, User>("SELECT * FROM users WHERE id = $1")
            .bind(user_id)
            .fetch_one(self.db.pool())
            .await?;

        Ok(user)
    }

    /// Get user by username
    ///
    /// # Errors
    ///
    /// Returns an error if user not found
    pub async fn get_user_by_username(&self, username: &str) -> Result<User> {
        let user = sqlx::query_as::<_, User>("SELECT * FROM users WHERE username = $1")
            .bind(username)
            .fetch_one(self.db.pool())
            .await?;

        Ok(user)
    }

    /// Update user
    ///
    /// # Errors
    ///
    /// Returns an error if update fails
    pub async fn update_user(&self, user_id: Uuid, req: UpdateUserRequest) -> Result<User> {
        let user = sqlx::query_as::<_, User>(
            "UPDATE users SET
                email = COALESCE($2, email),
                full_name = COALESCE($3, full_name),
                role = COALESCE($4, role),
                is_active = COALESCE($5, is_active),
                is_admin = CASE WHEN $4 = 'admin' THEN true ELSE is_admin END,
                updated_at = NOW()
             WHERE id = $1
             RETURNING *",
        )
        .bind(user_id)
        .bind(req.email)
        .bind(req.full_name)
        .bind(req.role)
        .bind(req.is_active)
        .fetch_one(self.db.pool())
        .await?;

        // Invalidate cache for this user
        self.permission_cache.write().await.remove(&user_id);

        Ok(user)
    }

    /// Delete user
    ///
    /// # Errors
    ///
    /// Returns an error if deletion fails
    pub async fn delete_user(&self, user_id: Uuid) -> Result<()> {
        sqlx::query("DELETE FROM users WHERE id = $1")
            .bind(user_id)
            .execute(self.db.pool())
            .await?;

        // Remove from cache
        self.permission_cache.write().await.remove(&user_id);

        Ok(())
    }

    /// List all users
    ///
    /// # Errors
    ///
    /// Returns an error if query fails
    pub async fn list_users(&self) -> Result<Vec<User>> {
        let users = sqlx::query_as::<_, User>("SELECT * FROM users ORDER BY username")
            .fetch_all(self.db.pool())
            .await?;

        Ok(users)
    }

    /// Authenticate user
    ///
    /// # Errors
    ///
    /// Returns an error if authentication fails
    pub async fn authenticate(&self, username: &str, password: &str) -> Result<User> {
        let user = self.get_user_by_username(username).await?;

        if !user.is_active {
            return Err(MamError::Authentication(
                "User account is inactive".to_string(),
            ));
        }

        let valid = bcrypt::verify(password, &user.password_hash)
            .map_err(|e| MamError::Internal(format!("Password verification failed: {e}")))?;

        if !valid {
            return Err(MamError::Authentication("Invalid credentials".to_string()));
        }

        // Update last login
        sqlx::query("UPDATE users SET last_login = NOW() WHERE id = $1")
            .bind(user.id)
            .execute(self.db.pool())
            .await?;

        Ok(user)
    }

    /// Check if user has permission
    ///
    /// # Errors
    ///
    /// Returns an error if permission check fails
    pub async fn has_permission(&self, user_id: Uuid, permission: Permission) -> Result<bool> {
        let permissions = self.get_user_permissions(user_id).await?;

        // Admins have all permissions
        if permissions.is_admin {
            return Ok(true);
        }

        Ok(permissions.global_permissions.contains(&permission))
    }

    /// Check if user has asset permission
    ///
    /// # Errors
    ///
    /// Returns an error if permission check fails
    pub async fn has_asset_permission(
        &self,
        user_id: Uuid,
        asset_id: Uuid,
        permission: Permission,
    ) -> Result<bool> {
        let permissions = self.get_user_permissions(user_id).await?;

        // Admins have all permissions
        if permissions.is_admin {
            return Ok(true);
        }

        // Check global permission
        if permissions.global_permissions.contains(&permission) {
            return Ok(true);
        }

        // Check asset-specific permission
        if let Some(asset_perms) = permissions.asset_permissions.get(&asset_id) {
            if asset_perms.contains(&permission) {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Get user permissions (with caching)
    ///
    /// # Errors
    ///
    /// Returns an error if query fails
    pub async fn get_user_permissions(&self, user_id: Uuid) -> Result<UserPermissions> {
        // Check cache first
        {
            let cache = self.permission_cache.read().await;
            if let Some(perms) = cache.get(&user_id) {
                // Cache valid for 5 minutes
                if perms.cached_at + chrono::Duration::minutes(5) > Utc::now() {
                    return Ok(perms.clone());
                }
            }
        }

        // Load from database
        let user = self.get_user(user_id).await?;

        // Get role permissions
        let role = self.get_role_by_name(&user.role).await?;
        let global_permissions: HashSet<Permission> = role
            .permissions
            .iter()
            .filter_map(|p| p.parse().ok())
            .collect();

        // Get asset-specific permissions
        let asset_perms = sqlx::query_as::<_, AssetPermission>(
            "SELECT * FROM asset_permissions WHERE user_id = $1",
        )
        .bind(user_id)
        .fetch_all(self.db.pool())
        .await?;

        let mut asset_permissions: HashMap<Uuid, HashSet<Permission>> = HashMap::new();
        for perm in asset_perms {
            if let Ok(p) = perm.permission.parse() {
                asset_permissions
                    .entry(perm.asset_id)
                    .or_default()
                    .insert(p);
            }
        }

        // Get collection-specific permissions
        let coll_perms = sqlx::query_as::<_, CollectionPermission>(
            "SELECT * FROM collection_permissions WHERE user_id = $1",
        )
        .bind(user_id)
        .fetch_all(self.db.pool())
        .await?;

        let mut collection_permissions: HashMap<Uuid, HashSet<Permission>> = HashMap::new();
        for perm in coll_perms {
            if let Ok(p) = perm.permission.parse() {
                collection_permissions
                    .entry(perm.collection_id)
                    .or_default()
                    .insert(p);
            }
        }

        let permissions = UserPermissions {
            user_id,
            role: user.role.clone(),
            is_admin: user.is_admin,
            global_permissions,
            asset_permissions,
            collection_permissions,
            cached_at: Utc::now(),
        };

        // Update cache
        self.permission_cache
            .write()
            .await
            .insert(user_id, permissions.clone());

        Ok(permissions)
    }

    /// Grant asset permission to user
    ///
    /// # Errors
    ///
    /// Returns an error if grant fails
    pub async fn grant_asset_permission(
        &self,
        asset_id: Uuid,
        user_id: Uuid,
        permission: Permission,
        granted_by: Uuid,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO asset_permissions
             (id, asset_id, user_id, permission, granted_by, created_at)
             VALUES ($1, $2, $3, $4, $5, NOW())
             ON CONFLICT (asset_id, user_id, permission) DO NOTHING",
        )
        .bind(Uuid::new_v4())
        .bind(asset_id)
        .bind(user_id)
        .bind(permission.as_str())
        .bind(granted_by)
        .execute(self.db.pool())
        .await?;

        // Invalidate cache
        self.permission_cache.write().await.remove(&user_id);

        Ok(())
    }

    /// Revoke asset permission from user
    ///
    /// # Errors
    ///
    /// Returns an error if revoke fails
    pub async fn revoke_asset_permission(
        &self,
        asset_id: Uuid,
        user_id: Uuid,
        permission: Permission,
    ) -> Result<()> {
        sqlx::query(
            "DELETE FROM asset_permissions
             WHERE asset_id = $1 AND user_id = $2 AND permission = $3",
        )
        .bind(asset_id)
        .bind(user_id)
        .bind(permission.as_str())
        .execute(self.db.pool())
        .await?;

        // Invalidate cache
        self.permission_cache.write().await.remove(&user_id);

        Ok(())
    }

    /// Create a role
    ///
    /// # Errors
    ///
    /// Returns an error if creation fails
    pub async fn create_role(
        &self,
        name: String,
        description: Option<String>,
        permissions: Vec<Permission>,
    ) -> Result<Role> {
        let perm_strings: Vec<String> =
            permissions.iter().map(|p| p.as_str().to_string()).collect();

        let role = sqlx::query_as::<_, Role>(
            "INSERT INTO roles
             (id, name, description, permissions, is_system, created_at, updated_at)
             VALUES ($1, $2, $3, $4, false, NOW(), NOW())
             RETURNING *",
        )
        .bind(Uuid::new_v4())
        .bind(&name)
        .bind(description)
        .bind(&perm_strings)
        .fetch_one(self.db.pool())
        .await?;

        Ok(role)
    }

    /// Get role by name
    ///
    /// # Errors
    ///
    /// Returns an error if role not found
    pub async fn get_role_by_name(&self, name: &str) -> Result<Role> {
        let role = sqlx::query_as::<_, Role>("SELECT * FROM roles WHERE name = $1")
            .bind(name)
            .fetch_one(self.db.pool())
            .await?;

        Ok(role)
    }

    /// List all roles
    ///
    /// # Errors
    ///
    /// Returns an error if query fails
    pub async fn list_roles(&self) -> Result<Vec<Role>> {
        let roles = sqlx::query_as::<_, Role>("SELECT * FROM roles ORDER BY name")
            .fetch_all(self.db.pool())
            .await?;

        Ok(roles)
    }

    /// Create a group
    ///
    /// # Errors
    ///
    /// Returns an error if creation fails
    pub async fn create_group(&self, name: String, description: Option<String>) -> Result<Group> {
        let group = sqlx::query_as::<_, Group>(
            "INSERT INTO groups
             (id, name, description, created_at, updated_at)
             VALUES ($1, $2, $3, NOW(), NOW())
             RETURNING *",
        )
        .bind(Uuid::new_v4())
        .bind(&name)
        .bind(description)
        .fetch_one(self.db.pool())
        .await?;

        Ok(group)
    }

    /// Add user to group
    ///
    /// # Errors
    ///
    /// Returns an error if operation fails
    pub async fn add_user_to_group(
        &self,
        group_id: Uuid,
        user_id: Uuid,
        added_by: Uuid,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO group_memberships
             (group_id, user_id, added_by, created_at)
             VALUES ($1, $2, $3, NOW())
             ON CONFLICT (group_id, user_id) DO NOTHING",
        )
        .bind(group_id)
        .bind(user_id)
        .bind(added_by)
        .execute(self.db.pool())
        .await?;

        // Invalidate cache
        self.permission_cache.write().await.remove(&user_id);

        Ok(())
    }

    /// Remove user from group
    ///
    /// # Errors
    ///
    /// Returns an error if operation fails
    pub async fn remove_user_from_group(&self, group_id: Uuid, user_id: Uuid) -> Result<()> {
        sqlx::query("DELETE FROM group_memberships WHERE group_id = $1 AND user_id = $2")
            .bind(group_id)
            .bind(user_id)
            .execute(self.db.pool())
            .await?;

        // Invalidate cache
        self.permission_cache.write().await.remove(&user_id);

        Ok(())
    }

    /// Get users in group
    ///
    /// # Errors
    ///
    /// Returns an error if query fails
    pub async fn get_group_users(&self, group_id: Uuid) -> Result<Vec<User>> {
        let users = sqlx::query_as::<_, User>(
            "SELECT u.* FROM users u
             INNER JOIN group_memberships gm ON u.id = gm.user_id
             WHERE gm.group_id = $1
             ORDER BY u.username",
        )
        .bind(group_id)
        .fetch_all(self.db.pool())
        .await?;

        Ok(users)
    }

    /// Clear permission cache
    pub async fn clear_cache(&self) {
        self.permission_cache.write().await.clear();
    }

    /// Clear permission cache for specific user
    pub async fn clear_user_cache(&self, user_id: Uuid) {
        self.permission_cache.write().await.remove(&user_id);
    }
}

// ---------------------------------------------------------------------------
// Attribute-Based Access Control (ABAC)
// ---------------------------------------------------------------------------

/// An attribute value used in ABAC policy conditions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AttributeValue {
    /// A string value (e.g. department name, classification label).
    Str(String),
    /// A boolean flag.
    Bool(bool),
    /// An integer value (e.g. clearance level).
    Int(i64),
    /// A list of string values (e.g. project memberships).
    List(Vec<String>),
}

impl std::fmt::Display for AttributeValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Str(s) => write!(f, "{s}"),
            Self::Bool(b) => write!(f, "{b}"),
            Self::Int(i) => write!(f, "{i}"),
            Self::List(v) => write!(f, "[{}]", v.join(", ")),
        }
    }
}

/// Comparison operator for ABAC conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonOp {
    /// Attribute must equal the expected value.
    Equals,
    /// Attribute must NOT equal the expected value.
    NotEquals,
    /// Attribute (Int) must be greater than or equal to expected (Int).
    GreaterOrEqual,
    /// Attribute (Int) must be less than expected (Int).
    LessThan,
    /// Attribute (List) must contain the expected (Str) value.
    Contains,
    /// Attribute (Str) must be one of the expected (List) values.
    In,
}

/// A single ABAC condition comparing an attribute against an expected value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbacCondition {
    /// The attribute key to evaluate (looked up in subject/resource/env
    /// attribute maps).
    pub attribute_key: String,
    /// Source of the attribute.
    pub attribute_source: AttributeSource,
    /// Comparison operator.
    pub operator: ComparisonOp,
    /// Expected value to compare against.
    pub expected: AttributeValue,
}

/// Where the attribute is drawn from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttributeSource {
    /// The requesting user / subject.
    Subject,
    /// The target resource (asset, collection, …).
    Resource,
    /// Environmental context (time of day, IP range, …).
    Environment,
}

impl AbacCondition {
    /// Evaluate this condition against the given attribute maps. Returns
    /// `true` if the condition is satisfied.
    #[must_use]
    pub fn evaluate(&self, attributes: &AbacContext) -> bool {
        let map = match self.attribute_source {
            AttributeSource::Subject => &attributes.subject,
            AttributeSource::Resource => &attributes.resource,
            AttributeSource::Environment => &attributes.environment,
        };

        let actual = match map.get(&self.attribute_key) {
            Some(v) => v,
            None => return false,
        };

        match self.operator {
            ComparisonOp::Equals => actual == &self.expected,
            ComparisonOp::NotEquals => actual != &self.expected,
            ComparisonOp::GreaterOrEqual => {
                if let (AttributeValue::Int(a), AttributeValue::Int(e)) = (actual, &self.expected) {
                    *a >= *e
                } else {
                    false
                }
            }
            ComparisonOp::LessThan => {
                if let (AttributeValue::Int(a), AttributeValue::Int(e)) = (actual, &self.expected) {
                    *a < *e
                } else {
                    false
                }
            }
            ComparisonOp::Contains => {
                if let (AttributeValue::List(list), AttributeValue::Str(needle)) =
                    (actual, &self.expected)
                {
                    list.contains(needle)
                } else {
                    false
                }
            }
            ComparisonOp::In => {
                if let (AttributeValue::Str(val), AttributeValue::List(allowed)) =
                    (actual, &self.expected)
                {
                    allowed.contains(val)
                } else {
                    false
                }
            }
        }
    }
}

/// The effect of an ABAC policy when all conditions are met.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyEffect {
    /// Allow the action.
    Allow,
    /// Deny the action.
    Deny,
}

/// An ABAC policy: a set of conditions that, when all satisfied, produce an
/// effect (Allow or Deny) for one or more permissions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbacPolicy {
    /// Human-readable name.
    pub name: String,
    /// Description of the policy's purpose.
    pub description: String,
    /// Priority (lower = evaluated first; first matching policy wins).
    pub priority: u32,
    /// Whether this policy is currently enabled.
    pub enabled: bool,
    /// Conditions that must ALL be satisfied (logical AND).
    pub conditions: Vec<AbacCondition>,
    /// Permissions this policy applies to (empty = all permissions).
    pub target_permissions: Vec<Permission>,
    /// The effect if all conditions match.
    pub effect: PolicyEffect,
}

impl AbacPolicy {
    /// Create a new ABAC policy.
    pub fn new(name: impl Into<String>, effect: PolicyEffect) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            priority: 100,
            enabled: true,
            conditions: Vec::new(),
            target_permissions: Vec::new(),
            effect,
        }
    }

    /// Set the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set the priority.
    pub fn with_priority(mut self, p: u32) -> Self {
        self.priority = p;
        self
    }

    /// Add a condition.
    pub fn with_condition(mut self, cond: AbacCondition) -> Self {
        self.conditions.push(cond);
        self
    }

    /// Set target permissions.
    pub fn with_target_permissions(mut self, perms: Vec<Permission>) -> Self {
        self.target_permissions = perms;
        self
    }

    /// Check if all conditions are satisfied by the given context.
    #[must_use]
    pub fn matches(&self, context: &AbacContext) -> bool {
        if !self.enabled {
            return false;
        }
        self.conditions.iter().all(|c| c.evaluate(context))
    }

    /// Check if this policy applies to a specific permission.
    #[must_use]
    pub fn applies_to_permission(&self, permission: &Permission) -> bool {
        self.target_permissions.is_empty() || self.target_permissions.contains(permission)
    }
}

/// Context for ABAC evaluation, containing subject, resource, and
/// environment attributes.
#[derive(Debug, Clone, Default)]
pub struct AbacContext {
    /// Attributes of the requesting subject (user).
    pub subject: HashMap<String, AttributeValue>,
    /// Attributes of the target resource.
    pub resource: HashMap<String, AttributeValue>,
    /// Environmental attributes (time, IP, etc.).
    pub environment: HashMap<String, AttributeValue>,
}

impl AbacContext {
    /// Create an empty context.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a subject attribute.
    pub fn with_subject(mut self, key: impl Into<String>, value: AttributeValue) -> Self {
        self.subject.insert(key.into(), value);
        self
    }

    /// Set a resource attribute.
    pub fn with_resource(mut self, key: impl Into<String>, value: AttributeValue) -> Self {
        self.resource.insert(key.into(), value);
        self
    }

    /// Set an environment attribute.
    pub fn with_environment(mut self, key: impl Into<String>, value: AttributeValue) -> Self {
        self.environment.insert(key.into(), value);
        self
    }
}

/// The ABAC policy engine that evaluates access decisions based on
/// attribute conditions, complementing RBAC.
#[derive(Debug, Default)]
pub struct AbacEngine {
    policies: Vec<AbacPolicy>,
}

impl AbacEngine {
    /// Create a new empty ABAC engine.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a policy.
    pub fn add_policy(&mut self, policy: AbacPolicy) {
        self.policies.push(policy);
        self.policies.sort_by_key(|p| p.priority);
    }

    /// Return the number of policies.
    #[must_use]
    pub fn policy_count(&self) -> usize {
        self.policies.len()
    }

    /// Evaluate all policies for a given permission and context.
    ///
    /// Returns the effect of the first matching policy, or `None` if no
    /// policy matches (in which case the caller should fall back to RBAC).
    #[must_use]
    pub fn evaluate(&self, permission: &Permission, context: &AbacContext) -> Option<PolicyEffect> {
        for policy in &self.policies {
            if policy.applies_to_permission(permission) && policy.matches(context) {
                return Some(policy.effect);
            }
        }
        None
    }

    /// Convenience: returns `true` if ABAC explicitly allows, `false` if
    /// explicitly denies, `None` if no policy matched.
    #[must_use]
    pub fn is_allowed(&self, permission: &Permission, context: &AbacContext) -> Option<bool> {
        self.evaluate(permission, context)
            .map(|e| e == PolicyEffect::Allow)
    }

    /// Return all policies (read-only).
    #[must_use]
    pub fn policies(&self) -> &[AbacPolicy] {
        &self.policies
    }

    /// Remove all policies.
    pub fn clear(&mut self) {
        self.policies.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_role_as_str() {
        assert_eq!(SystemRole::Admin.as_str(), "admin");
        assert_eq!(SystemRole::Editor.as_str(), "editor");
        assert_eq!(SystemRole::Viewer.as_str(), "viewer");
    }

    #[test]
    fn test_permission_as_str() {
        assert_eq!(Permission::AssetCreate.as_str(), "asset:create");
        assert_eq!(Permission::AssetRead.as_str(), "asset:read");
        assert_eq!(Permission::SystemAdmin.as_str(), "system:admin");
    }

    #[test]
    fn test_permission_from_str() {
        use std::str::FromStr;
        assert_eq!(
            Permission::from_str("asset:create").ok(),
            Some(Permission::AssetCreate)
        );
        assert_eq!(
            Permission::from_str("asset:read").ok(),
            Some(Permission::AssetRead)
        );
        assert!(Permission::from_str("invalid").is_err());
    }

    #[test]
    fn test_system_role_default_permissions() {
        let admin_perms = SystemRole::Admin.default_permissions();
        assert!(admin_perms.contains(&Permission::SystemAdmin));
        assert!(admin_perms.contains(&Permission::AssetCreate));

        let viewer_perms = SystemRole::Viewer.default_permissions();
        assert!(viewer_perms.contains(&Permission::AssetRead));
        assert!(!viewer_perms.contains(&Permission::AssetCreate));
    }

    #[test]
    fn test_create_user_request() {
        let req = CreateUserRequest {
            username: "testuser".to_string(),
            email: "test@example.com".to_string(),
            password: "password123".to_string(),
            full_name: Some("Test User".to_string()),
            role: "viewer".to_string(),
        };

        assert_eq!(req.username, "testuser");
        assert_eq!(req.role, "viewer");
    }

    // -----------------------------------------------------------------------
    // ABAC tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_attribute_value_display() {
        assert_eq!(AttributeValue::Str("hello".into()).to_string(), "hello");
        assert_eq!(AttributeValue::Bool(true).to_string(), "true");
        assert_eq!(AttributeValue::Int(42).to_string(), "42");
        assert_eq!(
            AttributeValue::List(vec!["a".into(), "b".into()]).to_string(),
            "[a, b]"
        );
    }

    #[test]
    fn test_abac_condition_equals() {
        let cond = AbacCondition {
            attribute_key: "department".into(),
            attribute_source: AttributeSource::Subject,
            operator: ComparisonOp::Equals,
            expected: AttributeValue::Str("engineering".into()),
        };
        let ctx = AbacContext::new()
            .with_subject("department", AttributeValue::Str("engineering".into()));
        assert!(cond.evaluate(&ctx));

        let ctx2 =
            AbacContext::new().with_subject("department", AttributeValue::Str("marketing".into()));
        assert!(!cond.evaluate(&ctx2));
    }

    #[test]
    fn test_abac_condition_not_equals() {
        let cond = AbacCondition {
            attribute_key: "status".into(),
            attribute_source: AttributeSource::Resource,
            operator: ComparisonOp::NotEquals,
            expected: AttributeValue::Str("archived".into()),
        };
        let ctx = AbacContext::new().with_resource("status", AttributeValue::Str("active".into()));
        assert!(cond.evaluate(&ctx));

        let ctx2 =
            AbacContext::new().with_resource("status", AttributeValue::Str("archived".into()));
        assert!(!cond.evaluate(&ctx2));
    }

    #[test]
    fn test_abac_condition_greater_or_equal() {
        let cond = AbacCondition {
            attribute_key: "clearance".into(),
            attribute_source: AttributeSource::Subject,
            operator: ComparisonOp::GreaterOrEqual,
            expected: AttributeValue::Int(3),
        };
        let ctx_high = AbacContext::new().with_subject("clearance", AttributeValue::Int(5));
        assert!(cond.evaluate(&ctx_high));

        let ctx_exact = AbacContext::new().with_subject("clearance", AttributeValue::Int(3));
        assert!(cond.evaluate(&ctx_exact));

        let ctx_low = AbacContext::new().with_subject("clearance", AttributeValue::Int(2));
        assert!(!cond.evaluate(&ctx_low));
    }

    #[test]
    fn test_abac_condition_less_than() {
        let cond = AbacCondition {
            attribute_key: "hour".into(),
            attribute_source: AttributeSource::Environment,
            operator: ComparisonOp::LessThan,
            expected: AttributeValue::Int(18),
        };
        let ctx = AbacContext::new().with_environment("hour", AttributeValue::Int(9));
        assert!(cond.evaluate(&ctx));

        let ctx2 = AbacContext::new().with_environment("hour", AttributeValue::Int(20));
        assert!(!cond.evaluate(&ctx2));
    }

    #[test]
    fn test_abac_condition_contains() {
        let cond = AbacCondition {
            attribute_key: "projects".into(),
            attribute_source: AttributeSource::Subject,
            operator: ComparisonOp::Contains,
            expected: AttributeValue::Str("alpha".into()),
        };
        let ctx = AbacContext::new().with_subject(
            "projects",
            AttributeValue::List(vec!["alpha".into(), "beta".into()]),
        );
        assert!(cond.evaluate(&ctx));

        let ctx2 =
            AbacContext::new().with_subject("projects", AttributeValue::List(vec!["gamma".into()]));
        assert!(!cond.evaluate(&ctx2));
    }

    #[test]
    fn test_abac_condition_in() {
        let cond = AbacCondition {
            attribute_key: "region".into(),
            attribute_source: AttributeSource::Environment,
            operator: ComparisonOp::In,
            expected: AttributeValue::List(vec!["us".into(), "eu".into()]),
        };
        let ctx = AbacContext::new().with_environment("region", AttributeValue::Str("eu".into()));
        assert!(cond.evaluate(&ctx));

        let ctx2 = AbacContext::new().with_environment("region", AttributeValue::Str("ap".into()));
        assert!(!cond.evaluate(&ctx2));
    }

    #[test]
    fn test_abac_condition_missing_attribute() {
        let cond = AbacCondition {
            attribute_key: "missing".into(),
            attribute_source: AttributeSource::Subject,
            operator: ComparisonOp::Equals,
            expected: AttributeValue::Bool(true),
        };
        let ctx = AbacContext::new();
        assert!(!cond.evaluate(&ctx));
    }

    #[test]
    fn test_abac_condition_type_mismatch() {
        let cond = AbacCondition {
            attribute_key: "level".into(),
            attribute_source: AttributeSource::Subject,
            operator: ComparisonOp::GreaterOrEqual,
            expected: AttributeValue::Int(5),
        };
        // Provide a string instead of an int.
        let ctx = AbacContext::new().with_subject("level", AttributeValue::Str("high".into()));
        assert!(!cond.evaluate(&ctx));
    }

    #[test]
    fn test_abac_policy_matches() {
        let policy =
            AbacPolicy::new("eng_allow", PolicyEffect::Allow).with_condition(AbacCondition {
                attribute_key: "department".into(),
                attribute_source: AttributeSource::Subject,
                operator: ComparisonOp::Equals,
                expected: AttributeValue::Str("engineering".into()),
            });
        let ctx = AbacContext::new()
            .with_subject("department", AttributeValue::Str("engineering".into()));
        assert!(policy.matches(&ctx));
    }

    #[test]
    fn test_abac_policy_disabled() {
        let mut policy = AbacPolicy::new("disabled", PolicyEffect::Deny);
        policy.enabled = false;
        let ctx = AbacContext::new();
        assert!(!policy.matches(&ctx));
    }

    #[test]
    fn test_abac_policy_applies_to_all_permissions() {
        let policy = AbacPolicy::new("blanket", PolicyEffect::Allow);
        assert!(policy.applies_to_permission(&Permission::AssetRead));
        assert!(policy.applies_to_permission(&Permission::AssetDelete));
    }

    #[test]
    fn test_abac_policy_applies_to_specific_permissions() {
        let policy = AbacPolicy::new("read_only", PolicyEffect::Allow)
            .with_target_permissions(vec![Permission::AssetRead]);
        assert!(policy.applies_to_permission(&Permission::AssetRead));
        assert!(!policy.applies_to_permission(&Permission::AssetDelete));
    }

    #[test]
    fn test_abac_engine_empty() {
        let engine = AbacEngine::new();
        assert_eq!(engine.policy_count(), 0);
        let ctx = AbacContext::new();
        assert!(engine.evaluate(&Permission::AssetRead, &ctx).is_none());
    }

    #[test]
    fn test_abac_engine_allow_policy() {
        let mut engine = AbacEngine::new();
        engine.add_policy(
            AbacPolicy::new("allow_eng", PolicyEffect::Allow).with_condition(AbacCondition {
                attribute_key: "department".into(),
                attribute_source: AttributeSource::Subject,
                operator: ComparisonOp::Equals,
                expected: AttributeValue::Str("engineering".into()),
            }),
        );
        let ctx = AbacContext::new()
            .with_subject("department", AttributeValue::Str("engineering".into()));
        assert_eq!(
            engine.evaluate(&Permission::AssetRead, &ctx),
            Some(PolicyEffect::Allow)
        );
        assert_eq!(engine.is_allowed(&Permission::AssetRead, &ctx), Some(true));
    }

    #[test]
    fn test_abac_engine_deny_policy() {
        let mut engine = AbacEngine::new();
        engine.add_policy(
            AbacPolicy::new("deny_external", PolicyEffect::Deny).with_condition(AbacCondition {
                attribute_key: "is_external".into(),
                attribute_source: AttributeSource::Subject,
                operator: ComparisonOp::Equals,
                expected: AttributeValue::Bool(true),
            }),
        );
        let ctx = AbacContext::new().with_subject("is_external", AttributeValue::Bool(true));
        assert_eq!(
            engine.evaluate(&Permission::AssetDelete, &ctx),
            Some(PolicyEffect::Deny)
        );
        assert_eq!(
            engine.is_allowed(&Permission::AssetDelete, &ctx),
            Some(false)
        );
    }

    #[test]
    fn test_abac_engine_priority_ordering() {
        let mut engine = AbacEngine::new();
        // Lower priority number = evaluated first.
        engine.add_policy(AbacPolicy::new("allow_all", PolicyEffect::Allow).with_priority(200));
        engine.add_policy(
            AbacPolicy::new("deny_guests", PolicyEffect::Deny)
                .with_priority(10)
                .with_condition(AbacCondition {
                    attribute_key: "role".into(),
                    attribute_source: AttributeSource::Subject,
                    operator: ComparisonOp::Equals,
                    expected: AttributeValue::Str("guest".into()),
                }),
        );
        // Guest user: deny policy matches first.
        let guest_ctx =
            AbacContext::new().with_subject("role", AttributeValue::Str("guest".into()));
        assert_eq!(
            engine.evaluate(&Permission::AssetRead, &guest_ctx),
            Some(PolicyEffect::Deny)
        );
        // Non-guest: deny doesn't match, allow does.
        let admin_ctx =
            AbacContext::new().with_subject("role", AttributeValue::Str("admin".into()));
        assert_eq!(
            engine.evaluate(&Permission::AssetRead, &admin_ctx),
            Some(PolicyEffect::Allow)
        );
    }

    #[test]
    fn test_abac_engine_no_match_returns_none() {
        let mut engine = AbacEngine::new();
        engine.add_policy(
            AbacPolicy::new("specific", PolicyEffect::Allow).with_condition(AbacCondition {
                attribute_key: "department".into(),
                attribute_source: AttributeSource::Subject,
                operator: ComparisonOp::Equals,
                expected: AttributeValue::Str("finance".into()),
            }),
        );
        let ctx = AbacContext::new().with_subject("department", AttributeValue::Str("hr".into()));
        assert!(engine.evaluate(&Permission::AssetRead, &ctx).is_none());
        assert!(engine.is_allowed(&Permission::AssetRead, &ctx).is_none());
    }

    #[test]
    fn test_abac_engine_multiple_conditions_and() {
        let mut engine = AbacEngine::new();
        engine.add_policy(
            AbacPolicy::new("restricted", PolicyEffect::Allow)
                .with_condition(AbacCondition {
                    attribute_key: "clearance".into(),
                    attribute_source: AttributeSource::Subject,
                    operator: ComparisonOp::GreaterOrEqual,
                    expected: AttributeValue::Int(3),
                })
                .with_condition(AbacCondition {
                    attribute_key: "classification".into(),
                    attribute_source: AttributeSource::Resource,
                    operator: ComparisonOp::Equals,
                    expected: AttributeValue::Str("secret".into()),
                }),
        );

        // Both conditions met.
        let ctx_ok = AbacContext::new()
            .with_subject("clearance", AttributeValue::Int(5))
            .with_resource("classification", AttributeValue::Str("secret".into()));
        assert_eq!(
            engine.evaluate(&Permission::AssetRead, &ctx_ok),
            Some(PolicyEffect::Allow)
        );

        // Only one condition met.
        let ctx_low = AbacContext::new()
            .with_subject("clearance", AttributeValue::Int(1))
            .with_resource("classification", AttributeValue::Str("secret".into()));
        assert!(engine.evaluate(&Permission::AssetRead, &ctx_low).is_none());
    }

    #[test]
    fn test_abac_engine_permission_targeting() {
        let mut engine = AbacEngine::new();
        engine.add_policy(
            AbacPolicy::new("read_only_for_viewers", PolicyEffect::Allow)
                .with_target_permissions(vec![Permission::AssetRead])
                .with_condition(AbacCondition {
                    attribute_key: "role".into(),
                    attribute_source: AttributeSource::Subject,
                    operator: ComparisonOp::Equals,
                    expected: AttributeValue::Str("viewer".into()),
                }),
        );
        let ctx = AbacContext::new().with_subject("role", AttributeValue::Str("viewer".into()));
        assert_eq!(
            engine.evaluate(&Permission::AssetRead, &ctx),
            Some(PolicyEffect::Allow)
        );
        // Policy does not target AssetDelete.
        assert!(engine.evaluate(&Permission::AssetDelete, &ctx).is_none());
    }

    #[test]
    fn test_abac_engine_clear() {
        let mut engine = AbacEngine::new();
        engine.add_policy(AbacPolicy::new("p1", PolicyEffect::Allow));
        engine.add_policy(AbacPolicy::new("p2", PolicyEffect::Deny));
        assert_eq!(engine.policy_count(), 2);
        engine.clear();
        assert_eq!(engine.policy_count(), 0);
    }

    #[test]
    fn test_abac_engine_policies_ref() {
        let mut engine = AbacEngine::new();
        engine.add_policy(AbacPolicy::new("alpha", PolicyEffect::Allow).with_priority(10));
        engine.add_policy(AbacPolicy::new("beta", PolicyEffect::Deny).with_priority(5));
        let policies = engine.policies();
        assert_eq!(policies.len(), 2);
        // Should be sorted by priority (5 before 10).
        assert_eq!(policies[0].name, "beta");
        assert_eq!(policies[1].name, "alpha");
    }

    #[test]
    fn test_abac_context_builder() {
        let ctx = AbacContext::new()
            .with_subject("name", AttributeValue::Str("alice".into()))
            .with_resource("type", AttributeValue::Str("video".into()))
            .with_environment("hour", AttributeValue::Int(14));
        assert_eq!(
            ctx.subject.get("name"),
            Some(&AttributeValue::Str("alice".into()))
        );
        assert_eq!(
            ctx.resource.get("type"),
            Some(&AttributeValue::Str("video".into()))
        );
        assert_eq!(ctx.environment.get("hour"), Some(&AttributeValue::Int(14)));
    }

    #[test]
    fn test_abac_environment_time_restriction() {
        let mut engine = AbacEngine::new();
        // Only allow downloads during business hours (9-17).
        engine.add_policy(
            AbacPolicy::new("business_hours", PolicyEffect::Allow)
                .with_target_permissions(vec![Permission::AssetDownload])
                .with_condition(AbacCondition {
                    attribute_key: "hour".into(),
                    attribute_source: AttributeSource::Environment,
                    operator: ComparisonOp::GreaterOrEqual,
                    expected: AttributeValue::Int(9),
                })
                .with_condition(AbacCondition {
                    attribute_key: "hour".into(),
                    attribute_source: AttributeSource::Environment,
                    operator: ComparisonOp::LessThan,
                    expected: AttributeValue::Int(17),
                }),
        );

        let ctx_ok = AbacContext::new().with_environment("hour", AttributeValue::Int(10));
        assert_eq!(
            engine.evaluate(&Permission::AssetDownload, &ctx_ok),
            Some(PolicyEffect::Allow)
        );

        let ctx_after = AbacContext::new().with_environment("hour", AttributeValue::Int(20));
        assert!(engine
            .evaluate(&Permission::AssetDownload, &ctx_after)
            .is_none());
    }

    #[test]
    fn test_abac_policy_serialization() {
        let policy = AbacPolicy::new("test_ser", PolicyEffect::Deny)
            .with_description("serialization test")
            .with_condition(AbacCondition {
                attribute_key: "k".into(),
                attribute_source: AttributeSource::Subject,
                operator: ComparisonOp::Equals,
                expected: AttributeValue::Str("v".into()),
            });
        let json = serde_json::to_string(&policy).expect("should succeed in test");
        let deser: AbacPolicy = serde_json::from_str(&json).expect("should succeed in test");
        assert_eq!(deser.name, "test_ser");
        assert_eq!(deser.effect, PolicyEffect::Deny);
        assert_eq!(deser.conditions.len(), 1);
    }
}

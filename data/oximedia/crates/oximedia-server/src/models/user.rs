//! User and authentication models.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// User account.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    /// Unique user ID
    pub id: String,
    /// Username (unique)
    pub username: String,
    /// Email address (unique)
    pub email: String,
    /// Password hash (not serialized)
    #[serde(skip_serializing)]
    pub password_hash: String,
    /// User role
    pub role: UserRole,
    /// Account creation timestamp
    pub created_at: i64,
    /// Last update timestamp
    pub updated_at: i64,
    /// Last login timestamp
    pub last_login: Option<i64>,
}

impl User {
    /// Creates a new user.
    #[must_use]
    pub fn new(username: String, email: String, password_hash: String) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            id: Uuid::new_v4().to_string(),
            username,
            email,
            password_hash,
            role: UserRole::User,
            created_at: now,
            updated_at: now,
            last_login: None,
        }
    }
}

/// User role for access control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UserRole {
    /// Administrator with full access
    Admin,
    /// Regular user
    User,
    /// Guest with read-only access
    Guest,
}

impl UserRole {
    /// Checks if this role can perform admin operations.
    #[must_use]
    pub const fn is_admin(self) -> bool {
        matches!(self, Self::Admin)
    }

    /// Checks if this role can write data.
    #[must_use]
    pub const fn can_write(self) -> bool {
        matches!(self, Self::Admin | Self::User)
    }
}

impl std::str::FromStr for UserRole {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "admin" => Ok(Self::Admin),
            "user" => Ok(Self::User),
            "guest" => Ok(Self::Guest),
            _ => Err(format!("Invalid user role: {s}")),
        }
    }
}

impl std::fmt::Display for UserRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Admin => write!(f, "admin"),
            Self::User => write!(f, "user"),
            Self::Guest => write!(f, "guest"),
        }
    }
}

/// API key for programmatic access.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    /// Unique key ID
    pub id: String,
    /// User ID this key belongs to
    pub user_id: String,
    /// Key hash (SHA-256 of the actual key)
    #[serde(skip_serializing)]
    pub key_hash: String,
    /// Human-readable name
    pub name: String,
    /// Creation timestamp
    pub created_at: i64,
    /// Expiration timestamp (optional)
    pub expires_at: Option<i64>,
    /// Last used timestamp
    pub last_used: Option<i64>,
}

impl ApiKey {
    /// Creates a new API key.
    #[must_use]
    pub fn new(user_id: String, key_hash: String, name: String, expires_at: Option<i64>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            user_id,
            key_hash,
            name,
            created_at: chrono::Utc::now().timestamp(),
            expires_at,
            last_used: None,
        }
    }

    /// Checks if the API key is expired.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            chrono::Utc::now().timestamp() > expires_at
        } else {
            false
        }
    }
}

/// JWT claims for authentication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    /// Subject (user ID)
    pub sub: String,
    /// Username
    pub username: String,
    /// User role
    pub role: UserRole,
    /// Issued at timestamp
    pub iat: i64,
    /// Expiration timestamp
    pub exp: i64,
}

impl Claims {
    /// Creates new JWT claims.
    #[must_use]
    pub fn new(user_id: String, username: String, role: UserRole, expiration: i64) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            sub: user_id,
            username,
            role,
            iat: now,
            exp: now + expiration,
        }
    }

    /// Checks if the token is expired.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        chrono::Utc::now().timestamp() > self.exp
    }
}

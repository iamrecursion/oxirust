//! Query builder for rights database

use chrono::{DateTime, Utc};

/// Query builder for searching rights grants
#[derive(Default, Clone)]
pub struct RightsQuery {
    asset_id: Option<String>,
    owner_id: Option<String>,
    license_type: Option<String>,
    active_at: Option<DateTime<Utc>>,
    territory: Option<String>,
    exclusive_only: bool,
}

impl RightsQuery {
    /// Create a new query builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter by asset ID
    pub fn asset_id(mut self, id: impl Into<String>) -> Self {
        self.asset_id = Some(id.into());
        self
    }

    /// Filter by owner ID
    pub fn owner_id(mut self, id: impl Into<String>) -> Self {
        self.owner_id = Some(id.into());
        self
    }

    /// Filter by license type
    pub fn license_type(mut self, license_type: impl Into<String>) -> Self {
        self.license_type = Some(license_type.into());
        self
    }

    /// Filter by active date (rights must be valid at this time)
    pub fn active_at(mut self, date: DateTime<Utc>) -> Self {
        self.active_at = Some(date);
        self
    }

    /// Filter by territory
    pub fn territory(mut self, territory: impl Into<String>) -> Self {
        self.territory = Some(territory.into());
        self
    }

    /// Only include exclusive rights
    pub fn exclusive_only(mut self, exclusive: bool) -> Self {
        self.exclusive_only = exclusive;
        self
    }

    /// Build the SQL WHERE clause and parameters
    pub fn build(&self) -> (String, Vec<String>) {
        let mut conditions = Vec::new();
        let mut params = Vec::new();

        if let Some(ref asset_id) = self.asset_id {
            conditions.push("asset_id = ?");
            params.push(asset_id.clone());
        }

        if let Some(ref owner_id) = self.owner_id {
            conditions.push("owner_id = ?");
            params.push(owner_id.clone());
        }

        if let Some(ref license_type) = self.license_type {
            conditions.push("license_type = ?");
            params.push(license_type.clone());
        }

        if let Some(ref date) = self.active_at {
            conditions.push("start_date <= ? AND (end_date IS NULL OR end_date >= ?)");
            let date_str = date.to_rfc3339();
            params.push(date_str.clone());
            params.push(date_str);
        }

        if let Some(ref territory) = self.territory {
            conditions.push("(territory_json IS NULL OR territory_json LIKE ?)");
            params.push(format!("%{territory}%"));
        }

        if self.exclusive_only {
            conditions.push("is_exclusive = 1");
        }

        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", conditions.join(" AND "))
        };

        (where_clause, params)
    }
}

/// Query builder for usage logs
#[derive(Default, Clone)]
pub struct UsageQuery {
    asset_id: Option<String>,
    grant_id: Option<String>,
    usage_type: Option<String>,
    start_date: Option<DateTime<Utc>>,
    end_date: Option<DateTime<Utc>>,
    territory: Option<String>,
}

impl UsageQuery {
    /// Create a new usage query
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter by asset ID
    pub fn asset_id(mut self, id: impl Into<String>) -> Self {
        self.asset_id = Some(id.into());
        self
    }

    /// Filter by grant ID
    pub fn grant_id(mut self, id: impl Into<String>) -> Self {
        self.grant_id = Some(id.into());
        self
    }

    /// Filter by usage type
    pub fn usage_type(mut self, usage_type: impl Into<String>) -> Self {
        self.usage_type = Some(usage_type.into());
        self
    }

    /// Filter by start date
    pub fn start_date(mut self, date: DateTime<Utc>) -> Self {
        self.start_date = Some(date);
        self
    }

    /// Filter by end date
    pub fn end_date(mut self, date: DateTime<Utc>) -> Self {
        self.end_date = Some(date);
        self
    }

    /// Filter by territory
    pub fn territory(mut self, territory: impl Into<String>) -> Self {
        self.territory = Some(territory.into());
        self
    }

    /// Build the SQL WHERE clause and parameters
    pub fn build(&self) -> (String, Vec<String>) {
        let mut conditions = Vec::new();
        let mut params = Vec::new();

        if let Some(ref asset_id) = self.asset_id {
            conditions.push("asset_id = ?");
            params.push(asset_id.clone());
        }

        if let Some(ref grant_id) = self.grant_id {
            conditions.push("grant_id = ?");
            params.push(grant_id.clone());
        }

        if let Some(ref usage_type) = self.usage_type {
            conditions.push("usage_type = ?");
            params.push(usage_type.clone());
        }

        if let Some(ref start_date) = self.start_date {
            conditions.push("usage_date >= ?");
            params.push(start_date.to_rfc3339());
        }

        if let Some(ref end_date) = self.end_date {
            conditions.push("usage_date <= ?");
            params.push(end_date.to_rfc3339());
        }

        if let Some(ref territory) = self.territory {
            conditions.push("territory = ?");
            params.push(territory.clone());
        }

        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", conditions.join(" AND "))
        };

        (where_clause, params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rights_query_builder() {
        let query = RightsQuery::new()
            .asset_id("asset123")
            .license_type("exclusive")
            .exclusive_only(true);

        let (where_clause, params) = query.build();
        assert!(where_clause.contains("WHERE"));
        assert!(where_clause.contains("asset_id"));
        assert!(where_clause.contains("license_type"));
        assert!(where_clause.contains("is_exclusive"));
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_usage_query_builder() {
        let query = UsageQuery::new()
            .asset_id("asset123")
            .usage_type("commercial")
            .territory("US");

        let (where_clause, params) = query.build();
        assert!(where_clause.contains("WHERE"));
        assert!(where_clause.contains("asset_id"));
        assert!(where_clause.contains("usage_type"));
        assert!(where_clause.contains("territory"));
        assert_eq!(params.len(), 3);
    }

    #[test]
    fn test_empty_query() {
        let query = RightsQuery::new();
        let (where_clause, params) = query.build();
        assert!(where_clause.is_empty());
        assert!(params.is_empty());
    }
}

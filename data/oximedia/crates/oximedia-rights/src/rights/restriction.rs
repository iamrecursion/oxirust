//! Usage restrictions for rights management

use serde::{Deserialize, Serialize};

/// Type of content usage
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum UsageType {
    /// Editorial use (news, documentary)
    Editorial,
    /// Commercial use (advertising, marketing)
    Commercial,
    /// Broadcast (TV, radio)
    Broadcast,
    /// Web distribution
    Web,
    /// Print media
    Print,
    /// Theatrical release
    Theatrical,
    /// Social media
    SocialMedia,
    /// Internal use only
    Internal,
    /// Custom usage type
    Custom(String),
}

impl UsageType {
    /// Convert to string representation
    pub fn as_str(&self) -> &str {
        match self {
            UsageType::Editorial => "editorial",
            UsageType::Commercial => "commercial",
            UsageType::Broadcast => "broadcast",
            UsageType::Web => "web",
            UsageType::Print => "print",
            UsageType::Theatrical => "theatrical",
            UsageType::SocialMedia => "social_media",
            UsageType::Internal => "internal",
            UsageType::Custom(s) => s,
        }
    }

    /// Parse from string
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s {
            "editorial" => UsageType::Editorial,
            "commercial" => UsageType::Commercial,
            "broadcast" => UsageType::Broadcast,
            "web" => UsageType::Web,
            "print" => UsageType::Print,
            "theatrical" => UsageType::Theatrical,
            "social_media" => UsageType::SocialMedia,
            "internal" => UsageType::Internal,
            other => UsageType::Custom(other.to_string()),
        }
    }
}

/// Usage restrictions for content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageRestriction {
    /// Allowed usage types
    pub allowed_uses: Vec<UsageType>,
    /// Prohibited usage types
    pub prohibited_uses: Vec<UsageType>,
    /// Maximum number of uses (None = unlimited)
    pub max_uses: Option<u32>,
    /// Maximum audience size
    pub max_audience: Option<u64>,
    /// Requires attribution
    pub requires_attribution: bool,
    /// Allows modifications
    pub allows_modifications: bool,
    /// Allows sublicensing
    pub allows_sublicensing: bool,
    /// Custom restrictions
    pub custom_restrictions: Vec<String>,
}

impl UsageRestriction {
    /// Create a new usage restriction with no limitations
    pub fn unrestricted() -> Self {
        Self {
            allowed_uses: vec![],
            prohibited_uses: vec![],
            max_uses: None,
            max_audience: None,
            requires_attribution: false,
            allows_modifications: true,
            allows_sublicensing: true,
            custom_restrictions: vec![],
        }
    }

    /// Create a new usage restriction with specific allowed uses
    pub fn with_allowed_uses(uses: Vec<UsageType>) -> Self {
        Self {
            allowed_uses: uses,
            prohibited_uses: vec![],
            max_uses: None,
            max_audience: None,
            requires_attribution: false,
            allows_modifications: true,
            allows_sublicensing: true,
            custom_restrictions: vec![],
        }
    }

    /// Add an allowed usage type
    pub fn allow_use(mut self, usage: UsageType) -> Self {
        if !self.allowed_uses.contains(&usage) {
            self.allowed_uses.push(usage);
        }
        self
    }

    /// Add a prohibited usage type
    pub fn prohibit_use(mut self, usage: UsageType) -> Self {
        if !self.prohibited_uses.contains(&usage) {
            self.prohibited_uses.push(usage);
        }
        self
    }

    /// Set maximum number of uses
    pub fn with_max_uses(mut self, max: u32) -> Self {
        self.max_uses = Some(max);
        self
    }

    /// Set maximum audience size
    pub fn with_max_audience(mut self, max: u64) -> Self {
        self.max_audience = Some(max);
        self
    }

    /// Require attribution
    pub fn require_attribution(mut self) -> Self {
        self.requires_attribution = true;
        self
    }

    /// Disallow modifications
    pub fn disallow_modifications(mut self) -> Self {
        self.allows_modifications = false;
        self
    }

    /// Disallow sublicensing
    pub fn disallow_sublicensing(mut self) -> Self {
        self.allows_sublicensing = false;
        self
    }

    /// Add custom restriction
    pub fn add_restriction(mut self, restriction: impl Into<String>) -> Self {
        self.custom_restrictions.push(restriction.into());
        self
    }

    /// Check if a usage type is allowed
    pub fn is_usage_allowed(&self, usage: &UsageType) -> bool {
        // If prohibited, return false
        if self.prohibited_uses.contains(usage) {
            return false;
        }

        // If allowed list is empty, all uses are allowed (except prohibited)
        if self.allowed_uses.is_empty() {
            return true;
        }

        // Otherwise, check if it's in the allowed list
        self.allowed_uses.contains(usage)
    }

    /// Check if usage count is within limits
    pub fn is_usage_count_allowed(&self, current_uses: u32) -> bool {
        match self.max_uses {
            Some(max) => current_uses < max,
            None => true,
        }
    }

    /// Check if audience size is within limits
    pub fn is_audience_allowed(&self, audience: u64) -> bool {
        match self.max_audience {
            Some(max) => audience <= max,
            None => true,
        }
    }
}

impl Default for UsageRestriction {
    fn default() -> Self {
        Self::unrestricted()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usage_type_conversion() {
        assert_eq!(UsageType::Commercial.as_str(), "commercial");
        assert_eq!(UsageType::from_str("commercial"), UsageType::Commercial);
    }

    #[test]
    fn test_unrestricted_usage() {
        let restriction = UsageRestriction::unrestricted();
        assert!(restriction.is_usage_allowed(&UsageType::Commercial));
        assert!(restriction.is_usage_allowed(&UsageType::Editorial));
        assert!(restriction.allows_modifications);
    }

    #[test]
    fn test_allowed_uses() {
        let restriction =
            UsageRestriction::with_allowed_uses(vec![UsageType::Editorial, UsageType::Web]);

        assert!(restriction.is_usage_allowed(&UsageType::Editorial));
        assert!(restriction.is_usage_allowed(&UsageType::Web));
        assert!(!restriction.is_usage_allowed(&UsageType::Commercial));
    }

    #[test]
    fn test_prohibited_uses() {
        let restriction = UsageRestriction::unrestricted().prohibit_use(UsageType::Commercial);

        assert!(!restriction.is_usage_allowed(&UsageType::Commercial));
        assert!(restriction.is_usage_allowed(&UsageType::Editorial));
    }

    #[test]
    fn test_usage_count_limits() {
        let restriction = UsageRestriction::unrestricted().with_max_uses(5);

        assert!(restriction.is_usage_count_allowed(0));
        assert!(restriction.is_usage_count_allowed(4));
        assert!(!restriction.is_usage_count_allowed(5));
    }

    #[test]
    fn test_audience_limits() {
        let restriction = UsageRestriction::unrestricted().with_max_audience(1000);

        assert!(restriction.is_audience_allowed(500));
        assert!(restriction.is_audience_allowed(1000));
        assert!(!restriction.is_audience_allowed(1001));
    }
}

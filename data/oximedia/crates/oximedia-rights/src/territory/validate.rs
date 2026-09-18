//! Territory validation utilities

use super::TerritoryRestriction;

/// Territory validator
pub struct TerritoryValidator;

impl TerritoryValidator {
    /// Validate a country code (ISO 3166-1 alpha-2)
    pub fn is_valid_country_code(code: &str) -> bool {
        // Check if it's a 2-letter code
        if code.len() != 2 {
            return false;
        }

        // Check if it's all uppercase letters
        code.chars().all(|c| c.is_ascii_uppercase())
    }

    /// Normalize a country code to uppercase
    pub fn normalize_country_code(code: &str) -> String {
        code.to_uppercase()
    }

    /// Check if usage in a territory is allowed under multiple restrictions
    pub fn check_multiple_restrictions(
        restrictions: &[TerritoryRestriction],
        country_code: &str,
    ) -> bool {
        // All restrictions must allow the territory
        restrictions.iter().all(|r| r.is_allowed(country_code))
    }

    /// Get the intersection of allowed territories from multiple restrictions
    pub fn intersect_restrictions(restrictions: &[TerritoryRestriction]) -> TerritoryRestriction {
        if restrictions.is_empty() {
            return TerritoryRestriction::worldwide();
        }

        // If any restriction is prohibitive, we need to handle it specially
        // For simplicity, we'll just return the first restriction
        // In a real implementation, this would compute the actual intersection
        restrictions[0].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_country_code() {
        assert!(TerritoryValidator::is_valid_country_code("US"));
        assert!(TerritoryValidator::is_valid_country_code("GB"));
        assert!(!TerritoryValidator::is_valid_country_code("USA"));
        assert!(!TerritoryValidator::is_valid_country_code("us"));
        assert!(!TerritoryValidator::is_valid_country_code("U"));
    }

    #[test]
    fn test_normalize_country_code() {
        assert_eq!(TerritoryValidator::normalize_country_code("us"), "US");
        assert_eq!(TerritoryValidator::normalize_country_code("GB"), "GB");
    }

    #[test]
    fn test_check_multiple_restrictions() {
        let restrictions = vec![
            TerritoryRestriction::allowed_countries(vec!["US".to_string(), "GB".to_string()]),
            TerritoryRestriction::worldwide(),
        ];

        assert!(TerritoryValidator::check_multiple_restrictions(
            &restrictions,
            "US"
        ));
        assert!(!TerritoryValidator::check_multiple_restrictions(
            &restrictions,
            "JP"
        ));
    }
}

//! Rights contract management.
//!
//! Provides data types for representing, querying, and storing rights
//! contracts between licensors and licensees.

use std::collections::HashMap;

/// Unique identifier for a contract.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ContractId(pub String);

impl ContractId {
    /// Create a new contract ID from any string-like value.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Access the inner string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ContractId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Type of contract governing the rights grant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContractType {
    /// One licensee holds exclusive rights.
    Exclusive,
    /// Multiple licensees may hold rights simultaneously.
    NonExclusive,
    /// All rights transferred; licensor retains no interest.
    BuyOut,
    /// Content created as work-for-hire; rights vest in the commissioning party.
    WorkForHire,
    /// A general licence without exclusivity or buy-out.
    License,
}

impl ContractType {
    /// Returns `true` if this contract type permits the licensee to
    /// sub-license the rights to third parties.
    #[must_use]
    pub fn allows_sublicense(&self) -> bool {
        matches!(self, Self::BuyOut | Self::WorkForHire)
    }
}

/// Role of a party in a contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PartyRole {
    /// The rights holder who grants the licence.
    Licensor,
    /// The party who receives the licence.
    Licensee,
    /// An intermediary representing one of the parties.
    Agent,
    /// A distributor of the licensed content.
    Distributor,
}

/// A party involved in a contract.
#[derive(Debug, Clone)]
pub struct ContractParty {
    /// Full name of the party.
    pub name: String,
    /// Role of this party in the contract.
    pub role: PartyRole,
    /// Contact e-mail address.
    pub contact_email: String,
}

impl ContractParty {
    /// Create a new contract party.
    pub fn new(name: impl Into<String>, role: PartyRole, contact_email: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            role,
            contact_email: contact_email.into(),
        }
    }
}

/// How the licensed content may be used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UsageType {
    /// Free-to-air or pay TV broadcast.
    Broadcast,
    /// Streaming or download via the internet.
    Online,
    /// Cinema / theatrical exhibition.
    Theatrical,
    /// Schools, universities, or non-profit educational use.
    Educational,
    /// In-flight entertainment on airlines.
    AirlineInFlight,
    /// Permanent digital download by the end user.
    DigitalDownload,
}

/// The bundle of rights granted by a contract.
#[derive(Debug, Clone)]
pub struct RightsGrant {
    /// Permitted usage types.
    pub usage_types: Vec<UsageType>,
    /// ISO 3166-1 alpha-2 territory codes (or "WW" for worldwide).
    pub territories: Vec<String>,
    /// Duration of the grant in days; `None` means perpetual.
    pub duration_days: Option<u32>,
    /// Platforms to which the grant applies (e.g. "Netflix", "YouTube").
    pub platforms: Vec<String>,
}

impl RightsGrant {
    /// Create a new rights grant.
    pub fn new(
        usage_types: Vec<UsageType>,
        territories: Vec<String>,
        duration_days: Option<u32>,
        platforms: Vec<String>,
    ) -> Self {
        Self {
            usage_types,
            territories,
            duration_days,
            platforms,
        }
    }
}

/// A rights contract between two or more parties.
#[derive(Debug, Clone)]
pub struct Contract {
    /// Unique identifier for this contract.
    pub id: ContractId,
    /// Type of contract.
    pub contract_type: ContractType,
    /// Parties to this contract.
    pub parties: Vec<ContractParty>,
    /// The rights that are granted.
    pub grant: RightsGrant,
    /// Unix timestamp (ms) when the contract was signed.
    pub signed_at_ms: u64,
    /// Unix timestamp (ms) when the contract expires; `None` = perpetual.
    pub expires_at_ms: Option<u64>,
}

impl Contract {
    /// Create a new contract.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: ContractId,
        contract_type: ContractType,
        parties: Vec<ContractParty>,
        grant: RightsGrant,
        signed_at_ms: u64,
        expires_at_ms: Option<u64>,
    ) -> Self {
        Self {
            id,
            contract_type,
            parties,
            grant,
            signed_at_ms,
            expires_at_ms,
        }
    }

    /// Returns `true` if the contract is currently active at `current_ms`.
    ///
    /// A contract is active if it has been signed (signed_at_ms <= current_ms)
    /// and has not yet expired.
    #[must_use]
    pub fn is_active(&self, current_ms: u64) -> bool {
        if current_ms < self.signed_at_ms {
            return false;
        }
        match self.expires_at_ms {
            None => true,
            Some(exp) => current_ms < exp,
        }
    }

    /// Returns `true` if this contract permits `usage` in `territory` at
    /// `current_ms`.
    #[must_use]
    pub fn covers_usage(&self, usage: &UsageType, territory: &str, current_ms: u64) -> bool {
        if !self.is_active(current_ms) {
            return false;
        }
        if !self.grant.usage_types.contains(usage) {
            return false;
        }
        // "WW" in territories means worldwide
        self.grant
            .territories
            .iter()
            .any(|t| t == "WW" || t == territory)
    }
}

/// In-memory repository of rights contracts.
#[derive(Debug, Default)]
pub struct ContractRepository {
    contracts: HashMap<ContractId, Contract>,
}

impl ContractRepository {
    /// Create a new empty repository.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a contract to the repository.  Replaces any existing contract with
    /// the same ID.
    pub fn add(&mut self, contract: Contract) {
        self.contracts.insert(contract.id.clone(), contract);
    }

    /// Find a contract by its ID.
    #[must_use]
    pub fn find_by_id(&self, id: &ContractId) -> Option<&Contract> {
        self.contracts.get(id)
    }

    /// Return all contracts that are currently active at `current_ms`.
    #[must_use]
    pub fn active_contracts(&self, current_ms: u64) -> Vec<&Contract> {
        self.contracts
            .values()
            .filter(|c| c.is_active(current_ms))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_700_000_000_000; // arbitrary fixed "now"
    const PAST: u64 = 1_600_000_000_000;
    const FUTURE: u64 = 1_800_000_000_000;

    fn simple_grant() -> RightsGrant {
        RightsGrant::new(
            vec![UsageType::Online, UsageType::Broadcast],
            vec!["US".into(), "GB".into()],
            Some(365),
            vec!["Netflix".into()],
        )
    }

    fn make_contract(id: &str, signed: u64, expires: Option<u64>) -> Contract {
        Contract::new(
            ContractId::new(id),
            ContractType::NonExclusive,
            vec![ContractParty::new(
                "Licensor Inc",
                PartyRole::Licensor,
                "a@example.com",
            )],
            simple_grant(),
            signed,
            expires,
        )
    }

    // --- ContractId ---

    #[test]
    fn test_contract_id_new() {
        let id = ContractId::new("c-001");
        assert_eq!(id.as_str(), "c-001");
    }

    #[test]
    fn test_contract_id_display() {
        let id = ContractId::new("c-002");
        assert_eq!(id.to_string(), "c-002");
    }

    #[test]
    fn test_contract_id_equality() {
        assert_eq!(ContractId::new("x"), ContractId::new("x"));
        assert_ne!(ContractId::new("x"), ContractId::new("y"));
    }

    // --- ContractType ---

    #[test]
    fn test_contract_type_allows_sublicense() {
        assert!(ContractType::BuyOut.allows_sublicense());
        assert!(ContractType::WorkForHire.allows_sublicense());
        assert!(!ContractType::Exclusive.allows_sublicense());
        assert!(!ContractType::NonExclusive.allows_sublicense());
        assert!(!ContractType::License.allows_sublicense());
    }

    // --- Contract::is_active ---

    #[test]
    fn test_contract_is_active_perpetual() {
        let c = make_contract("c-1", PAST, None);
        assert!(c.is_active(NOW));
    }

    #[test]
    fn test_contract_is_active_not_yet_signed() {
        let c = make_contract("c-2", FUTURE, None);
        assert!(!c.is_active(NOW));
    }

    #[test]
    fn test_contract_is_active_expired() {
        let c = make_contract("c-3", PAST, Some(PAST + 1000));
        assert!(!c.is_active(NOW));
    }

    #[test]
    fn test_contract_is_active_within_validity() {
        let c = make_contract("c-4", PAST, Some(FUTURE));
        assert!(c.is_active(NOW));
    }

    // --- Contract::covers_usage ---

    #[test]
    fn test_covers_usage_matching() {
        let c = make_contract("c-5", PAST, Some(FUTURE));
        assert!(c.covers_usage(&UsageType::Online, "US", NOW));
    }

    #[test]
    fn test_covers_usage_wrong_territory() {
        let c = make_contract("c-6", PAST, Some(FUTURE));
        assert!(!c.covers_usage(&UsageType::Online, "JP", NOW));
    }

    #[test]
    fn test_covers_usage_wrong_type() {
        let c = make_contract("c-7", PAST, Some(FUTURE));
        assert!(!c.covers_usage(&UsageType::Theatrical, "US", NOW));
    }

    #[test]
    fn test_covers_usage_worldwide() {
        let grant = RightsGrant::new(vec![UsageType::Online], vec!["WW".into()], None, vec![]);
        let c = Contract::new(
            ContractId::new("c-8"),
            ContractType::License,
            vec![],
            grant,
            PAST,
            None,
        );
        assert!(c.covers_usage(&UsageType::Online, "JP", NOW));
    }

    #[test]
    fn test_covers_usage_expired() {
        let c = make_contract("c-9", PAST, Some(PAST + 1000));
        assert!(!c.covers_usage(&UsageType::Online, "US", NOW));
    }

    // --- ContractRepository ---

    #[test]
    fn test_repository_add_and_find() {
        let mut repo = ContractRepository::new();
        let id = ContractId::new("c-10");
        repo.add(make_contract("c-10", PAST, None));
        assert!(repo.find_by_id(&id).is_some());
    }

    #[test]
    fn test_repository_find_missing() {
        let repo = ContractRepository::new();
        assert!(repo.find_by_id(&ContractId::new("missing")).is_none());
    }

    #[test]
    fn test_repository_active_contracts() {
        let mut repo = ContractRepository::new();
        repo.add(make_contract("c-11", PAST, Some(FUTURE)));
        repo.add(make_contract("c-12", PAST, Some(PAST + 1))); // expired
        let active = repo.active_contracts(NOW);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id.as_str(), "c-11");
    }
}

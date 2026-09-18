//! Input bank configuration for broadcast video switchers.
//!
//! Manages groups (banks) of physical inputs, their signal types, and
//! validation rules for a complete switcher input configuration.

#![allow(dead_code)]

/// Physical signal type of a switcher input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BankInputType {
    /// Serial Digital Interface (SDI).
    Sdi,
    /// High-Definition SDI (HD-SDI / 3G-SDI).
    HdSdi,
    /// 12G-SDI for 4K signals.
    Sdi12G,
    /// Network Device Interface (NDI).
    Ndi,
    /// HDMI input.
    Hdmi,
    /// Internally generated color bar or black signal.
    Internal,
    /// Input slot is empty / unused.
    None,
}

impl BankInputType {
    /// Returns `true` if this input type is an SDI variant.
    pub fn is_sdi(&self) -> bool {
        matches!(
            self,
            BankInputType::Sdi | BankInputType::HdSdi | BankInputType::Sdi12G
        )
    }

    /// Returns `true` if this input carries an external signal.
    pub fn is_external(&self) -> bool {
        !matches!(self, BankInputType::Internal | BankInputType::None)
    }

    /// Return a short identifier string.
    pub fn label(&self) -> &'static str {
        match self {
            BankInputType::Sdi => "SDI",
            BankInputType::HdSdi => "HD-SDI",
            BankInputType::Sdi12G => "12G-SDI",
            BankInputType::Ndi => "NDI",
            BankInputType::Hdmi => "HDMI",
            BankInputType::Internal => "INT",
            BankInputType::None => "NONE",
        }
    }
}

/// Configuration for a single switcher input slot.
#[derive(Debug, Clone)]
pub struct BankInputConfig {
    /// 1-based input number.
    pub number: usize,
    /// Human-readable name for this input.
    pub name: String,
    /// Signal type.
    pub input_type: BankInputType,
    /// Whether the input is enabled.
    pub enabled: bool,
}

impl BankInputConfig {
    /// Create a new `BankInputConfig`.
    pub fn new(number: usize, name: impl Into<String>, input_type: BankInputType) -> Self {
        Self {
            number,
            name: name.into(),
            input_type,
            enabled: true,
        }
    }

    /// Returns `true` if the configuration is valid (non-zero number, non-empty name).
    pub fn is_valid(&self) -> bool {
        self.number > 0 && !self.name.trim().is_empty()
    }

    /// Disable this input.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Enable this input.
    pub fn enable(&mut self) {
        self.enabled = true;
    }
}

/// A named bank of input configurations.
#[derive(Debug)]
pub struct InputBankConfig {
    /// Bank identifier (0-based).
    pub bank_id: usize,
    /// Human-readable bank name.
    pub name: String,
    /// Inputs belonging to this bank.
    inputs: Vec<BankInputConfig>,
    /// Maximum number of inputs this bank supports.
    capacity: usize,
}

impl InputBankConfig {
    /// Create a new `InputBankConfig` with the given capacity.
    pub fn new(bank_id: usize, name: impl Into<String>, capacity: usize) -> Self {
        Self {
            bank_id,
            name: name.into(),
            inputs: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Add an input to this bank.
    ///
    /// Returns `false` if the bank is at capacity or the input number is already used.
    pub fn add_input(&mut self, cfg: BankInputConfig) -> bool {
        if self.inputs.len() >= self.capacity {
            return false;
        }
        let duplicate = self.inputs.iter().any(|i| i.number == cfg.number);
        if duplicate {
            return false;
        }
        self.inputs.push(cfg);
        true
    }

    /// Return the number of configured inputs.
    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }

    /// Return the number of enabled inputs.
    pub fn enabled_count(&self) -> usize {
        self.inputs.iter().filter(|i| i.enabled).count()
    }

    /// Look up an input by 1-based number.
    pub fn get_input(&self, number: usize) -> Option<&BankInputConfig> {
        self.inputs.iter().find(|i| i.number == number)
    }

    /// Return `true` if the bank is full.
    pub fn is_full(&self) -> bool {
        self.inputs.len() >= self.capacity
    }

    /// Return the bank capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Validator for a complete set of input bank configurations.
#[derive(Debug, Default)]
pub struct InputConfigValidator;

impl InputConfigValidator {
    /// Create a new validator.
    pub fn new() -> Self {
        Self
    }

    /// Validate a slice of bank configurations.
    ///
    /// Returns a list of human-readable error strings; empty means no errors.
    pub fn validate(&self, banks: &[InputBankConfig]) -> Vec<String> {
        let mut errors = Vec::new();
        let mut all_input_numbers: Vec<usize> = Vec::new();

        for bank in banks {
            if bank.name.trim().is_empty() {
                errors.push(format!("Bank {} has an empty name", bank.bank_id));
            }
            for input in &bank.inputs {
                if !input.is_valid() {
                    errors.push(format!(
                        "Bank {}: input #{} is invalid",
                        bank.bank_id, input.number
                    ));
                }
                if all_input_numbers.contains(&input.number) {
                    errors.push(format!(
                        "Duplicate input number {} across banks",
                        input.number
                    ));
                }
                all_input_numbers.push(input.number);
            }
        }
        errors
    }

    /// Return `true` if all banks pass validation.
    pub fn is_valid(&self, banks: &[InputBankConfig]) -> bool {
        self.validate(banks).is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bank_input_type_is_sdi() {
        assert!(BankInputType::Sdi.is_sdi());
        assert!(BankInputType::HdSdi.is_sdi());
        assert!(BankInputType::Sdi12G.is_sdi());
        assert!(!BankInputType::Ndi.is_sdi());
        assert!(!BankInputType::Hdmi.is_sdi());
    }

    #[test]
    fn test_bank_input_type_is_external() {
        assert!(BankInputType::Sdi.is_external());
        assert!(BankInputType::Ndi.is_external());
        assert!(!BankInputType::Internal.is_external());
        assert!(!BankInputType::None.is_external());
    }

    #[test]
    fn test_bank_input_type_label() {
        assert_eq!(BankInputType::Sdi.label(), "SDI");
        assert_eq!(BankInputType::Internal.label(), "INT");
    }

    #[test]
    fn test_bank_input_config_is_valid() {
        let cfg = BankInputConfig::new(1, "Camera 1", BankInputType::HdSdi);
        assert!(cfg.is_valid());
    }

    #[test]
    fn test_bank_input_config_invalid_zero_number() {
        let cfg = BankInputConfig::new(0, "Camera 0", BankInputType::Sdi);
        assert!(!cfg.is_valid());
    }

    #[test]
    fn test_bank_input_config_invalid_empty_name() {
        let cfg = BankInputConfig::new(1, "   ", BankInputType::Sdi);
        assert!(!cfg.is_valid());
    }

    #[test]
    fn test_bank_input_config_enable_disable() {
        let mut cfg = BankInputConfig::new(1, "Cam", BankInputType::Sdi);
        cfg.disable();
        assert!(!cfg.enabled);
        cfg.enable();
        assert!(cfg.enabled);
    }

    #[test]
    fn test_input_bank_add_input() {
        let mut bank = InputBankConfig::new(0, "Bank A", 4);
        let cfg = BankInputConfig::new(1, "Cam 1", BankInputType::HdSdi);
        assert!(bank.add_input(cfg));
        assert_eq!(bank.input_count(), 1);
    }

    #[test]
    fn test_input_bank_duplicate_number_rejected() {
        let mut bank = InputBankConfig::new(0, "Bank A", 4);
        bank.add_input(BankInputConfig::new(1, "Cam 1", BankInputType::Sdi));
        let dup = BankInputConfig::new(1, "Cam 1 Dup", BankInputType::Sdi);
        assert!(!bank.add_input(dup));
    }

    #[test]
    fn test_input_bank_capacity_limit() {
        let mut bank = InputBankConfig::new(0, "Small", 2);
        bank.add_input(BankInputConfig::new(1, "C1", BankInputType::Sdi));
        bank.add_input(BankInputConfig::new(2, "C2", BankInputType::Sdi));
        let extra = BankInputConfig::new(3, "C3", BankInputType::Sdi);
        assert!(!bank.add_input(extra));
        assert!(bank.is_full());
    }

    #[test]
    fn test_input_bank_get_input() {
        let mut bank = InputBankConfig::new(0, "Bank", 8);
        bank.add_input(BankInputConfig::new(3, "Cam 3", BankInputType::Ndi));
        let found = bank.get_input(3);
        assert!(found.is_some());
        assert_eq!(found.expect("should succeed in test").name, "Cam 3");
    }

    #[test]
    fn test_input_bank_enabled_count() {
        let mut bank = InputBankConfig::new(0, "Bank", 4);
        let mut c1 = BankInputConfig::new(1, "C1", BankInputType::Sdi);
        let c2 = BankInputConfig::new(2, "C2", BankInputType::Sdi);
        c1.disable();
        bank.add_input(c1);
        bank.add_input(c2);
        assert_eq!(bank.enabled_count(), 1);
    }

    #[test]
    fn test_validator_no_errors() {
        let mut bank = InputBankConfig::new(0, "Main", 4);
        bank.add_input(BankInputConfig::new(1, "Cam 1", BankInputType::HdSdi));
        let validator = InputConfigValidator::new();
        assert!(validator.is_valid(&[bank]));
    }

    #[test]
    fn test_validator_duplicate_across_banks() {
        let mut bank_a = InputBankConfig::new(0, "A", 4);
        let mut bank_b = InputBankConfig::new(1, "B", 4);
        bank_a.add_input(BankInputConfig::new(1, "Cam 1", BankInputType::Sdi));
        bank_b.add_input(BankInputConfig::new(1, "Cam 1 Again", BankInputType::Sdi));
        let validator = InputConfigValidator::new();
        let errors = validator.validate(&[bank_a, bank_b]);
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_validator_empty_bank_name() {
        let bank = InputBankConfig::new(0, "   ", 4);
        let validator = InputConfigValidator::new();
        let errors = validator.validate(&[bank]);
        assert!(!errors.is_empty());
    }
}

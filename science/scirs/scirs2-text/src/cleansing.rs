//! Advanced text cleansing utilities
//!
//! This module provides functions for cleaning text data including
//! HTML stripping, URL handling, and various normalization operations.

use crate::error::Result;
use lazy_static::lazy_static;
use regex::Regex;
use std::collections::HashMap;

lazy_static! {
    // HTML/XML tag pattern
    static ref HTML_TAG_PATTERN: Regex = Regex::new(r"<[^>]+>").expect("Operation failed");

    // URL pattern
    static ref URL_PATTERN: Regex = Regex::new(
        r"(?i)https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)"
    ).expect("Operation failed");

    // Email pattern
    static ref EMAIL_PATTERN: Regex = Regex::new(
        r"(?i)[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    ).expect("Operation failed");

    // Phone number pattern (simplified, US-style)
    static ref PHONE_PATTERN: Regex = Regex::new(
        r"(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})"
    ).expect("Operation failed");

    // Common contractions mapping
    static ref CONTRACTIONS: HashMap<&'static str, &'static str> = {
        let mut m = HashMap::new();
        m.insert("can't", "cannot");
        m.insert("won't", "will not");
        m.insert("n't", " not");
        m.insert("'re", " are");
        m.insert("'ve", " have");
        m.insert("'ll", " will");
        m.insert("'d", " would");
        m.insert("'m", " am");
        m.insert("'s", " is");
        m.insert("let's", "let us");
        m.insert("it's", "it is");
        m.insert("that's", "that is");
        m.insert("what's", "what is");
        m.insert("where's", "where is");
        m.insert("who's", "who is");
        m.insert("there's", "there is");
        m.insert("here's", "here is");
        m
    };

    // Emoji pattern
    static ref EMOJI_PATTERN: Regex = Regex::new(
        concat!(
            "[",
            "\u{1F600}-\u{1F64F}", // Emoticons
            "\u{1F300}-\u{1F5FF}", // Symbols & Pictographs
            "\u{1F680}-\u{1F6FF}", // Transport & Map
            "\u{1F700}-\u{1F77F}", // Alchemical Symbols
            "\u{1F780}-\u{1F7FF}", // Geometric Shapes Extended
            "\u{1F800}-\u{1F8FF}", // Supplemental Arrows-C
            "\u{2600}-\u{26FF}",   // Miscellaneous Symbols
            "\u{2700}-\u{27BF}",   // Dingbats
            "]"
        )
    ).expect("Operation failed");

    // Number patterns for normalization
    static ref NUMBER_PATTERN: Regex = Regex::new(
        r"(?i)\b[-+]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?\b"
    ).expect("Operation failed");

    // Currency pattern
    static ref CURRENCY_PATTERN: Regex = Regex::new(
        r"(?i)(?:[$€£¥₹])[ \t]*(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d{1,2})?|(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d{1,2})?[ \t]*(?:dollars?|euros?|pounds?|yen|rupees?|USD|EUR|GBP|JPY|INR)\b"
    ).expect("Operation failed");

    // Percentage pattern
    static ref PERCENTAGE_PATTERN: Regex = Regex::new(
        r"(?i)[-+]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?%"
    ).expect("Operation failed");

    // Ordinal pattern
    static ref ORDINAL_PATTERN: Regex = Regex::new(
        r"(?i)\b(\d+)(?:st|nd|rd|th)\b"
    ).expect("Operation failed");

    // Advanced number patterns for enhanced normalization

    // Date patterns (various formats)
    static ref DATE_PATTERN: Regex = Regex::new(
        r"(?i)\b(?:(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12][0-9]|3[01])[/-](?:19|20)?\d{2})|(?:(?:19|20)\d{2}[/-](?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12][0-9]|3[01]))|(?:(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2},?\s+\d{4})|(?:\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{4})\b"
    ).expect("Operation failed");

    // Time patterns
    static ref TIME_PATTERN: Regex = Regex::new(
        r"(?i)\b(?:[01]?[0-9]|2[0-3]):[0-5][0-9](?::[0-5][0-9])?(?:\s*[aApP][mM])?\b"
    ).expect("Operation failed");

    // Fraction patterns
    static ref FRACTION_PATTERN: Regex = Regex::new(
        r"\b\d+\s*/\s*\d+\b|\b\d+\s+\d+\s*/\s*\d+\b"
    ).expect("Operation failed");

    // Roman numeral patterns
    static ref ROMAN_NUMERAL_PATTERN: Regex = Regex::new(
        r"(?i)\b[MDCLXVI]+\b"
    ).expect("Operation failed");

    // Scientific notation (enhanced)
    static ref SCIENTIFIC_NOTATION_PATTERN: Regex = Regex::new(
        r"(?i)[-+]?(?:\d+\.?\d*|\.\d+)[eE][-+]?\d+"
    ).expect("Operation failed");

    // Temperature patterns
    static ref TEMPERATURE_PATTERN: Regex = Regex::new(
        r"(?i)[-+]?(?:\d+\.?\d*|\.\d+)\s*(?:°[CFK]|[CFK](?:\s+degrees?)?\s*(?:celsius|fahrenheit|kelvin)?|degrees?\s+(?:celsius|fahrenheit|kelvin))\b"
    ).expect("Operation failed");

    // Measurement unit patterns
    static ref MEASUREMENT_PATTERN: Regex = Regex::new(
        r"(?i)[-+]?(?:\d+\.?\d*|\.\d+)\s*(?:mm|cm|m|km|in|ft|yd|mi|g|kg|lb|oz|ml|l|gal|mph|kph|Hz|kHz|MHz|GHz|KB|MB|GB|TB|°|rad|sq|cu)\b"
    ).expect("Operation failed");

    // Enhanced currency pattern with more currencies and formats
    static ref ENHANCED_CURRENCY_PATTERN: Regex = Regex::new(
        r"(?i)(?:[$€£¥₹₽₩¢₪₨₦₴₵₡₲₱₫₭₦₨]\s*\d+(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:,\d{3})*(?:\.\d{1,2})?\s*(?:USD|EUR|GBP|JPY|INR|RUB|KRW|CNY|CAD|AUD|CHF|SGD|dollars?|euros?|pounds?|yen|rupees?|yuan|won|rubles?))\b"
    ).expect("Operation failed");

    // Version numbers
    static ref VERSION_PATTERN: Regex = Regex::new(
        r"\bv?\d+(?:\.\d+){1,3}(?:-[a-zA-Z]+\d*)?\b"
    ).expect("Operation failed");

    // IP addresses
    static ref IP_ADDRESS_PATTERN: Regex = Regex::new(
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    ).expect("Operation failed");

    // Hexadecimal numbers
    static ref HEX_PATTERN: Regex = Regex::new(
        r"(?i)\b0x[0-9a-f]+\b|#[0-9a-f]{3,8}\b"
    ).expect("Operation failed");

    // Binary numbers
    static ref BINARY_PATTERN: Regex = Regex::new(
        r"\b0b[01]+\b"
    ).expect("Operation failed");
}

/// Advanced text cleaner with various cleaning options
#[derive(Debug, Clone)]
pub struct AdvancedTextCleaner {
    strip_html: bool,
    replace_urls: bool,
    replace_emails: bool,
    replace_phone_numbers: bool,
    expand_contractions: bool,
    remove_emojis: bool,
    normalize_unicode: bool,
    preserve_case: bool,
    normalize_numbers: bool,
    normalize_currencies: bool,
    normalize_percentages: bool,
    normalize_ordinals: bool,
    normalize_dates: bool,
    normalize_times: bool,
    normalize_fractions: bool,
    normalize_roman_numerals: bool,
    normalize_scientific_notation: bool,
    normalize_temperatures: bool,
    normalize_measurements: bool,
    normalize_versions: bool,
    normalize_ip_addresses: bool,
    normalize_hex_numbers: bool,
    normalize_binary_numbers: bool,
    url_placeholder: String,
    email_placeholder: String,
    phone_placeholder: String,
    number_placeholder: String,
    currency_placeholder: String,
    percentage_placeholder: String,
    ordinal_placeholder: String,
    date_placeholder: String,
    time_placeholder: String,
    fraction_placeholder: String,
    roman_placeholder: String,
    scientific_placeholder: String,
    temperature_placeholder: String,
    measurement_placeholder: String,
    version_placeholder: String,
    ip_placeholder: String,
    hex_placeholder: String,
    binary_placeholder: String,
}

impl AdvancedTextCleaner {
    /// Create a new advanced text cleaner with default settings
    pub fn new() -> Self {
        Self {
            strip_html: true,
            replace_urls: true,
            replace_emails: true,
            replace_phone_numbers: false,
            expand_contractions: true,
            remove_emojis: false,
            normalize_unicode: true,
            preserve_case: false,
            normalize_numbers: false,
            normalize_currencies: false,
            normalize_percentages: false,
            normalize_ordinals: false,
            normalize_dates: false,
            normalize_times: false,
            normalize_fractions: false,
            normalize_roman_numerals: false,
            normalize_scientific_notation: false,
            normalize_temperatures: false,
            normalize_measurements: false,
            normalize_versions: false,
            normalize_ip_addresses: false,
            normalize_hex_numbers: false,
            normalize_binary_numbers: false,
            url_placeholder: "[URL]".to_string(),
            email_placeholder: "[EMAIL]".to_string(),
            phone_placeholder: "[PHONE]".to_string(),
            number_placeholder: "[NUMBER]".to_string(),
            currency_placeholder: "[CURRENCY]".to_string(),
            percentage_placeholder: "[PERCENT]".to_string(),
            ordinal_placeholder: "[ORDINAL]".to_string(),
            date_placeholder: "[DATE]".to_string(),
            time_placeholder: "[TIME]".to_string(),
            fraction_placeholder: "[FRACTION]".to_string(),
            roman_placeholder: "[ROMAN]".to_string(),
            scientific_placeholder: "[SCIENTIFIC]".to_string(),
            temperature_placeholder: "[TEMP]".to_string(),
            measurement_placeholder: "[MEASURE]".to_string(),
            version_placeholder: "[VERSION]".to_string(),
            ip_placeholder: "[IP]".to_string(),
            hex_placeholder: "[HEX]".to_string(),
            binary_placeholder: "[BINARY]".to_string(),
        }
    }

    /// Create a cleaner for privacy-sensitive text
    pub fn privacy_focused() -> Self {
        Self {
            strip_html: true,
            replace_urls: true,
            replace_emails: true,
            replace_phone_numbers: true,
            expand_contractions: true,
            remove_emojis: false,
            normalize_unicode: true,
            preserve_case: false,
            normalize_numbers: false,
            normalize_currencies: false,
            normalize_percentages: false,
            normalize_ordinals: false,
            normalize_dates: true, // Privacy-focused normalizes dates
            normalize_times: true, // and times
            normalize_fractions: false,
            normalize_roman_numerals: false,
            normalize_scientific_notation: false,
            normalize_temperatures: false,
            normalize_measurements: false,
            normalize_versions: false,
            normalize_ip_addresses: true, // Privacy-focused normalizes IPs
            normalize_hex_numbers: false,
            normalize_binary_numbers: false,
            url_placeholder: "[URL]".to_string(),
            email_placeholder: "[EMAIL]".to_string(),
            phone_placeholder: "[PHONE]".to_string(),
            number_placeholder: "[NUMBER]".to_string(),
            currency_placeholder: "[CURRENCY]".to_string(),
            percentage_placeholder: "[PERCENT]".to_string(),
            ordinal_placeholder: "[ORDINAL]".to_string(),
            date_placeholder: "[DATE]".to_string(),
            time_placeholder: "[TIME]".to_string(),
            fraction_placeholder: "[FRACTION]".to_string(),
            roman_placeholder: "[ROMAN]".to_string(),
            scientific_placeholder: "[SCIENTIFIC]".to_string(),
            temperature_placeholder: "[TEMP]".to_string(),
            measurement_placeholder: "[MEASURE]".to_string(),
            version_placeholder: "[VERSION]".to_string(),
            ip_placeholder: "[IP]".to_string(),
            hex_placeholder: "[HEX]".to_string(),
            binary_placeholder: "[BINARY]".to_string(),
        }
    }

    /// Create a minimal cleaner that preserves most content
    pub fn minimal() -> Self {
        Self {
            strip_html: true,
            replace_urls: false,
            replace_emails: false,
            replace_phone_numbers: false,
            expand_contractions: false,
            remove_emojis: false,
            normalize_unicode: true,
            preserve_case: true,
            normalize_numbers: false,
            normalize_currencies: false,
            normalize_percentages: false,
            normalize_ordinals: false,
            normalize_dates: false,
            normalize_times: false,
            normalize_fractions: false,
            normalize_roman_numerals: false,
            normalize_scientific_notation: false,
            normalize_temperatures: false,
            normalize_measurements: false,
            normalize_versions: false,
            normalize_ip_addresses: false,
            normalize_hex_numbers: false,
            normalize_binary_numbers: false,
            url_placeholder: "[URL]".to_string(),
            email_placeholder: "[EMAIL]".to_string(),
            phone_placeholder: "[PHONE]".to_string(),
            number_placeholder: "[NUMBER]".to_string(),
            currency_placeholder: "[CURRENCY]".to_string(),
            percentage_placeholder: "[PERCENT]".to_string(),
            ordinal_placeholder: "[ORDINAL]".to_string(),
            date_placeholder: "[DATE]".to_string(),
            time_placeholder: "[TIME]".to_string(),
            fraction_placeholder: "[FRACTION]".to_string(),
            roman_placeholder: "[ROMAN]".to_string(),
            scientific_placeholder: "[SCIENTIFIC]".to_string(),
            temperature_placeholder: "[TEMP]".to_string(),
            measurement_placeholder: "[MEASURE]".to_string(),
            version_placeholder: "[VERSION]".to_string(),
            ip_placeholder: "[IP]".to_string(),
            hex_placeholder: "[HEX]".to_string(),
            binary_placeholder: "[BINARY]".to_string(),
        }
    }

    /// Set whether to strip HTML tags
    pub fn set_strip_html(mut self, value: bool) -> Self {
        self.strip_html = value;
        self
    }

    /// Set whether to replace URLs
    pub fn set_replace_urls(mut self, value: bool) -> Self {
        self.replace_urls = value;
        self
    }

    /// Set whether to replace emails
    pub fn set_replace_emails(mut self, value: bool) -> Self {
        self.replace_emails = value;
        self
    }

    /// Set whether to replace phone numbers
    pub fn set_replace_phone_numbers(mut self, value: bool) -> Self {
        self.replace_phone_numbers = value;
        self
    }

    /// Set whether to expand contractions
    pub fn set_expand_contractions(mut self, value: bool) -> Self {
        self.expand_contractions = value;
        self
    }

    /// Set whether to remove emojis
    pub fn set_remove_emojis(mut self, value: bool) -> Self {
        self.remove_emojis = value;
        self
    }

    /// Set whether to normalize numbers
    pub fn set_normalize_numbers(mut self, value: bool) -> Self {
        self.normalize_numbers = value;
        self
    }

    /// Set whether to normalize currencies
    pub fn set_normalize_currencies(mut self, value: bool) -> Self {
        self.normalize_currencies = value;
        self
    }

    /// Set whether to normalize percentages
    pub fn set_normalize_percentages(mut self, value: bool) -> Self {
        self.normalize_percentages = value;
        self
    }

    /// Set whether to normalize ordinals
    pub fn set_normalize_ordinals(mut self, value: bool) -> Self {
        self.normalize_ordinals = value;
        self
    }

    /// Set whether to normalize dates
    pub fn set_normalize_dates(mut self, value: bool) -> Self {
        self.normalize_dates = value;
        self
    }

    /// Set whether to normalize times
    pub fn set_normalize_times(mut self, value: bool) -> Self {
        self.normalize_times = value;
        self
    }

    /// Set whether to normalize fractions
    pub fn set_normalize_fractions(mut self, value: bool) -> Self {
        self.normalize_fractions = value;
        self
    }

    /// Set whether to normalize roman numerals
    pub fn set_normalize_roman_numerals(mut self, value: bool) -> Self {
        self.normalize_roman_numerals = value;
        self
    }

    /// Set whether to normalize scientific notation
    pub fn set_normalize_scientific_notation(mut self, value: bool) -> Self {
        self.normalize_scientific_notation = value;
        self
    }

    /// Set whether to normalize temperatures
    pub fn set_normalize_temperatures(mut self, value: bool) -> Self {
        self.normalize_temperatures = value;
        self
    }

    /// Set whether to normalize measurements
    pub fn set_normalize_measurements(mut self, value: bool) -> Self {
        self.normalize_measurements = value;
        self
    }

    /// Set whether to normalize version numbers
    pub fn set_normalize_versions(mut self, value: bool) -> Self {
        self.normalize_versions = value;
        self
    }

    /// Set whether to normalize IP addresses
    pub fn set_normalize_ip_addresses(mut self, value: bool) -> Self {
        self.normalize_ip_addresses = value;
        self
    }

    /// Set whether to normalize hexadecimal numbers
    pub fn set_normalize_hex_numbers(mut self, value: bool) -> Self {
        self.normalize_hex_numbers = value;
        self
    }

    /// Set whether to normalize binary numbers
    pub fn set_normalize_binary_numbers(mut self, value: bool) -> Self {
        self.normalize_binary_numbers = value;
        self
    }

    /// Set custom placeholders
    pub fn set_placeholders(
        mut self,
        url: Option<String>,
        email: Option<String>,
        phone: Option<String>,
    ) -> Self {
        if let Some(u) = url {
            self.url_placeholder = u;
        }
        if let Some(e) = email {
            self.email_placeholder = e;
        }
        if let Some(p) = phone {
            self.phone_placeholder = p;
        }
        self
    }

    /// Set custom number placeholders
    pub fn set_number_placeholders(
        mut self,
        number: Option<String>,
        currency: Option<String>,
        percentage: Option<String>,
        ordinal: Option<String>,
    ) -> Self {
        if let Some(n) = number {
            self.number_placeholder = n;
        }
        if let Some(c) = currency {
            self.currency_placeholder = c;
        }
        if let Some(p) = percentage {
            self.percentage_placeholder = p;
        }
        if let Some(o) = ordinal {
            self.ordinal_placeholder = o;
        }
        self
    }

    /// Set custom advanced placeholders for new normalization types
    pub fn set_advanced_placeholders(
        mut self,
        date: Option<String>,
        time: Option<String>,
        fraction: Option<String>,
        roman: Option<String>,
        scientific: Option<String>,
        temperature: Option<String>,
        measurement: Option<String>,
        version: Option<String>,
        ip: Option<String>,
        hex: Option<String>,
        binary: Option<String>,
    ) -> Self {
        if let Some(d) = date {
            self.date_placeholder = d;
        }
        if let Some(t) = time {
            self.time_placeholder = t;
        }
        if let Some(f) = fraction {
            self.fraction_placeholder = f;
        }
        if let Some(r) = roman {
            self.roman_placeholder = r;
        }
        if let Some(s) = scientific {
            self.scientific_placeholder = s;
        }
        if let Some(temp) = temperature {
            self.temperature_placeholder = temp;
        }
        if let Some(m) = measurement {
            self.measurement_placeholder = m;
        }
        if let Some(v) = version {
            self.version_placeholder = v;
        }
        if let Some(i) = ip {
            self.ip_placeholder = i;
        }
        if let Some(h) = hex {
            self.hex_placeholder = h;
        }
        if let Some(b) = binary {
            self.binary_placeholder = b;
        }
        self
    }

    /// Clean the text according to the configured options
    pub fn clean(&self, text: &str) -> Result<String> {
        let mut cleaned = text.to_string();

        // Strip HTML tags
        if self.strip_html {
            cleaned = strip_html_tags(&cleaned);
        }

        // Replace URLs
        if self.replace_urls {
            cleaned = URL_PATTERN
                .replace_all(&cleaned, &self.url_placeholder)
                .to_string();
        }

        // Replace emails
        if self.replace_emails {
            cleaned = EMAIL_PATTERN
                .replace_all(&cleaned, &self.email_placeholder)
                .to_string();
        }

        // Replace phone numbers
        if self.replace_phone_numbers {
            cleaned = PHONE_PATTERN
                .replace_all(&cleaned, &self.phone_placeholder)
                .to_string();
        }

        // Expand contractions
        if self.expand_contractions {
            cleaned = expand_contractions(&cleaned);
        }

        // Remove emojis
        if self.remove_emojis {
            cleaned = EMOJI_PATTERN.replace_all(&cleaned, " ").to_string();
        }

        // Normalize specialized patterns first (before general numbers)

        // Normalize dates
        if self.normalize_dates {
            cleaned = DATE_PATTERN
                .replace_all(&cleaned, &self.date_placeholder)
                .to_string();
        }

        // Normalize times
        if self.normalize_times {
            cleaned = TIME_PATTERN
                .replace_all(&cleaned, &self.time_placeholder)
                .to_string();
        }

        // Normalize IP addresses (before general numbers)
        if self.normalize_ip_addresses {
            cleaned = IP_ADDRESS_PATTERN
                .replace_all(&cleaned, &self.ip_placeholder)
                .to_string();
        }

        // Normalize version numbers (before general numbers)
        if self.normalize_versions {
            cleaned = VERSION_PATTERN
                .replace_all(&cleaned, &self.version_placeholder)
                .to_string();
        }

        // Normalize scientific notation (before general numbers)
        if self.normalize_scientific_notation {
            cleaned = SCIENTIFIC_NOTATION_PATTERN
                .replace_all(&cleaned, &self.scientific_placeholder)
                .to_string();
        }

        // Normalize temperatures (before general numbers)
        if self.normalize_temperatures {
            cleaned = TEMPERATURE_PATTERN
                .replace_all(&cleaned, &self.temperature_placeholder)
                .to_string();
        }

        // Normalize measurements (before general numbers)
        if self.normalize_measurements {
            cleaned = MEASUREMENT_PATTERN
                .replace_all(&cleaned, &self.measurement_placeholder)
                .to_string();
        }

        // Normalize hexadecimal numbers (before general numbers)
        if self.normalize_hex_numbers {
            cleaned = HEX_PATTERN
                .replace_all(&cleaned, &self.hex_placeholder)
                .to_string();
        }

        // Normalize binary numbers (before general numbers)
        if self.normalize_binary_numbers {
            cleaned = BINARY_PATTERN
                .replace_all(&cleaned, &self.binary_placeholder)
                .to_string();
        }

        // Normalize enhanced currencies (before general currencies)
        if self.normalize_currencies {
            cleaned = ENHANCED_CURRENCY_PATTERN
                .replace_all(&cleaned, &self.currency_placeholder)
                .to_string();
        }

        // Normalize fractions (before general numbers)
        if self.normalize_fractions {
            cleaned = FRACTION_PATTERN
                .replace_all(&cleaned, &self.fraction_placeholder)
                .to_string();
        }

        // Normalize roman numerals (before general numbers)
        if self.normalize_roman_numerals {
            cleaned = ROMAN_NUMERAL_PATTERN
                .replace_all(&cleaned, &self.roman_placeholder)
                .to_string();
        }

        // Normalize percentages (before general numbers)
        if self.normalize_percentages {
            cleaned = PERCENTAGE_PATTERN
                .replace_all(&cleaned, &self.percentage_placeholder)
                .to_string();
        }

        // Normalize ordinals (before general numbers)
        if self.normalize_ordinals {
            cleaned = ORDINAL_PATTERN
                .replace_all(&cleaned, &self.ordinal_placeholder)
                .to_string();
        }

        // Normalize general numbers
        if self.normalize_numbers {
            cleaned = NUMBER_PATTERN
                .replace_all(&cleaned, &self.number_placeholder)
                .to_string();
        }

        // Normalize unicode
        if self.normalize_unicode {
            cleaned = normalize_unicode(&cleaned)?;
        }

        // Handle case
        if !self.preserve_case {
            cleaned = cleaned.to_lowercase();
        }

        // Normalize whitespace
        cleaned = normalize_whitespace(&cleaned);

        Ok(cleaned)
    }
}

impl Default for AdvancedTextCleaner {
    fn default() -> Self {
        Self::new()
    }
}

/// Strip HTML and XML tags from text
#[allow(dead_code)]
pub fn strip_html_tags(text: &str) -> String {
    HTML_TAG_PATTERN.replace_all(text, " ").to_string()
}

/// Replace URLs with a placeholder
#[allow(dead_code)]
pub fn replace_urls(text: &str, placeholder: &str) -> String {
    URL_PATTERN.replace_all(text, placeholder).to_string()
}

/// Replace email addresses with a placeholder
#[allow(dead_code)]
pub fn replace_emails(text: &str, placeholder: &str) -> String {
    EMAIL_PATTERN.replace_all(text, placeholder).to_string()
}

/// Replace phone numbers with a placeholder
#[allow(dead_code)]
pub fn replace_phone_numbers(text: &str, placeholder: &str) -> String {
    PHONE_PATTERN.replace_all(text, placeholder).to_string()
}

/// Expand common contractions
#[allow(dead_code)]
pub fn expand_contractions(text: &str) -> String {
    let mut result = text.to_string();

    // Sort contractions by length (descending) to avoid partial replacements
    let mut contractions: Vec<_> = CONTRACTIONS.iter().collect();
    contractions.sort_by_key(|(k_, _)| std::cmp::Reverse(k_.len()));

    for (contraction, expansion) in contractions {
        let escaped = regex::escape(contraction);
        let pattern = format!(r"\b{escaped}\b");
        if let Ok(re) = Regex::new(&pattern) {
            result = re.replace_all(&result, *expansion).to_string();
        }
    }

    result
}

/// Normalize Unicode text (NFD -> NFC)
#[allow(dead_code)]
pub fn normalize_unicode(text: &str) -> Result<String> {
    use unicode_normalization::UnicodeNormalization;
    Ok(text.nfc().collect())
}

/// Normalize whitespace (multiple spaces to single space, trim)
#[allow(dead_code)]
pub fn normalize_whitespace(text: &str) -> String {
    #[cfg(feature = "simd")]
    {
        // Use SIMD to find whitespace positions for ASCII text
        if text.is_ascii() && crate::simd_ops::SimdStringOps::is_available() {
            let positions = crate::simd_ops::SimdStringOps::find_whitespace_positions(text);
            if positions.is_empty() {
                return text.trim().to_string();
            }
            // Fall through to regex-based approach for complex whitespace normalization
        }
    }

    lazy_static! {
        static ref WHITESPACE_PATTERN: Regex = Regex::new(r"\s+").expect("Operation failed");
    }

    WHITESPACE_PATTERN.replace_all(text.trim(), " ").to_string()
}

/// Remove accents from text
#[allow(dead_code)]
pub fn remove_accents(text: &str) -> String {
    use unicode_normalization::UnicodeNormalization;

    text.nfd()
        .filter(|c| !unicode_normalization::char::is_combining_mark(*c))
        .collect()
}

/// Convert various dash types to regular hyphen
#[allow(dead_code)]
pub fn normalize_dashes(text: &str) -> String {
    lazy_static! {
        static ref DASH_PATTERN: Regex =
            Regex::new(r"[\u{2010}-\u{2015}\u{2212}]").expect("Operation failed");
    }

    DASH_PATTERN.replace_all(text, "-").to_string()
}

/// Convert various quote types to regular quotes
#[allow(dead_code)]
pub fn normalize_quotes(text: &str) -> String {
    lazy_static! {
        static ref SINGLE_QUOTE_PATTERN: Regex =
            Regex::new(r"[\u{2018}\u{2019}\u{201A}\u{201B}]").expect("Operation failed");
        static ref DOUBLE_QUOTE_PATTERN: Regex =
            Regex::new(r"[\u{201C}\u{201D}\u{201E}\u{201F}]").expect("Operation failed");
    }

    let text = SINGLE_QUOTE_PATTERN.replace_all(text, "'");
    DOUBLE_QUOTE_PATTERN.replace_all(&text, "\"").to_string()
}

/// Normalize numbers in text
#[allow(dead_code)]
pub fn normalize_numbers(text: &str, placeholder: &str) -> String {
    NUMBER_PATTERN.replace_all(text, placeholder).to_string()
}

/// Normalize currency values in text
#[allow(dead_code)]
pub fn normalize_currencies(text: &str, placeholder: &str) -> String {
    CURRENCY_PATTERN.replace_all(text, placeholder).to_string()
}

/// Normalize percentage values in text
#[allow(dead_code)]
pub fn normalize_percentages(text: &str, placeholder: &str) -> String {
    PERCENTAGE_PATTERN
        .replace_all(text, placeholder)
        .to_string()
}

/// Normalize ordinal numbers in text
#[allow(dead_code)]
pub fn normalize_ordinals(text: &str, placeholder: &str) -> String {
    ORDINAL_PATTERN.replace_all(text, placeholder).to_string()
}

/// Normalize dates in text
#[allow(dead_code)]
pub fn normalize_dates(text: &str, placeholder: &str) -> String {
    DATE_PATTERN.replace_all(text, placeholder).to_string()
}

/// Normalize times in text
#[allow(dead_code)]
pub fn normalize_times(text: &str, placeholder: &str) -> String {
    TIME_PATTERN.replace_all(text, placeholder).to_string()
}

/// Normalize fractions in text
#[allow(dead_code)]
pub fn normalize_fractions(text: &str, placeholder: &str) -> String {
    FRACTION_PATTERN.replace_all(text, placeholder).to_string()
}

/// Normalize roman numerals in text
#[allow(dead_code)]
pub fn normalize_roman_numerals(text: &str, placeholder: &str) -> String {
    ROMAN_NUMERAL_PATTERN
        .replace_all(text, placeholder)
        .to_string()
}

/// Normalize scientific notation in text
#[allow(dead_code)]
pub fn normalize_scientific_notation(text: &str, placeholder: &str) -> String {
    SCIENTIFIC_NOTATION_PATTERN
        .replace_all(text, placeholder)
        .to_string()
}

/// Normalize temperatures in text
#[allow(dead_code)]
pub fn normalize_temperatures(text: &str, placeholder: &str) -> String {
    TEMPERATURE_PATTERN
        .replace_all(text, placeholder)
        .to_string()
}

/// Normalize measurements in text
#[allow(dead_code)]
pub fn normalize_measurements(text: &str, placeholder: &str) -> String {
    MEASUREMENT_PATTERN
        .replace_all(text, placeholder)
        .to_string()
}

/// Normalize version numbers in text
#[allow(dead_code)]
pub fn normalize_versions(text: &str, placeholder: &str) -> String {
    VERSION_PATTERN.replace_all(text, placeholder).to_string()
}

/// Normalize IP addresses in text
#[allow(dead_code)]
pub fn normalize_ip_addresses(text: &str, placeholder: &str) -> String {
    IP_ADDRESS_PATTERN
        .replace_all(text, placeholder)
        .to_string()
}

/// Normalize hexadecimal numbers in text
#[allow(dead_code)]
pub fn normalize_hex_numbers(text: &str, placeholder: &str) -> String {
    HEX_PATTERN.replace_all(text, placeholder).to_string()
}

/// Normalize binary numbers in text
#[allow(dead_code)]
pub fn normalize_binary_numbers(text: &str, placeholder: &str) -> String {
    BINARY_PATTERN.replace_all(text, placeholder).to_string()
}

/// Normalize all number formats comprehensively
#[allow(dead_code)]
pub fn normalize_all_numbers(text: &str, placeholder: &str) -> String {
    let mut result = text.to_string();

    // Apply in order of specificity (most specific first)
    result = normalize_scientific_notation(&result, placeholder);
    result = normalize_temperatures(&result, placeholder);
    result = normalize_measurements(&result, placeholder);
    result = normalize_hex_numbers(&result, placeholder);
    result = normalize_binary_numbers(&result, placeholder);
    result = normalize_currencies(&result, placeholder);
    result = normalize_percentages(&result, placeholder);
    result = normalize_fractions(&result, placeholder);
    result = normalize_ordinals(&result, placeholder);
    result = normalize_versions(&result, placeholder);
    result = normalize_ip_addresses(&result, placeholder);
    result = normalize_roman_numerals(&result, placeholder);
    result = normalize_numbers(&result, placeholder);

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_html_tags() {
        let html = "<p>Hello <b>world</b>!</p>";
        let cleaned = strip_html_tags(html);
        assert_eq!(cleaned, " Hello  world ! ");
    }

    #[test]
    fn test_replace_urls() {
        let text = "Check out https://www.example.com for more info";
        let replaced = replace_urls(text, "[URL]");
        assert_eq!(replaced, "Check out [URL] for more info");
    }

    #[test]
    fn test_replace_emails() {
        let text = "Contact us at support@example.com for help";
        let replaced = replace_emails(text, "[EMAIL]");
        assert_eq!(replaced, "Contact us at [EMAIL] for help");
    }

    #[test]
    fn test_expand_contractions() {
        let text = "I can't believe it's working! They'll be happy.";
        let expanded = expand_contractions(text);
        assert_eq!(
            expanded,
            "I cannot believe it is working! They will be happy."
        );
    }

    #[test]
    fn test_normalize_whitespace() {
        let text = "  Hello   world  \n\t  test  ";
        let normalized = normalize_whitespace(text);
        assert_eq!(normalized, "Hello world test");
    }

    #[test]
    fn test_remove_accents() {
        let text = "Héllo wörld café";
        let cleaned = remove_accents(text);
        assert_eq!(cleaned, "Hello world cafe");
    }

    #[test]
    fn test_advanced_cleaner() {
        let cleaner = AdvancedTextCleaner::new();
        let text = "<p>Check out https://example.com! Email: test@example.com</p>";
        let cleaned = cleaner.clean(text).expect("Operation failed");
        assert_eq!(cleaned, "check out [url]! email: [email]");
    }

    #[test]
    fn test_privacy_focused_cleaner() {
        let cleaner = AdvancedTextCleaner::privacy_focused();
        let text = "Call me at (555) 123-4567 or email john@example.com";
        let cleaned = cleaner.clean(text).expect("Operation failed");
        assert_eq!(cleaned, "call me at [phone] or email [email]");
    }

    #[test]
    fn test_normalize_numbers() {
        let text = "The price is 1,234.56 and the quantity is 42";
        let normalized = normalize_numbers(text, "[NUM]");
        assert_eq!(normalized, "The price is [NUM] and the quantity is [NUM]");

        let text_scientific = "The value is 3.14e-10";
        let normalized_sci = normalize_numbers(text_scientific, "[NUM]");
        assert_eq!(normalized_sci, "The value is [NUM]");
    }

    #[test]
    fn test_normalize_currencies() {
        let text = "The cost is $45.99 or €50.00";
        let normalized = normalize_currencies(text, "[MONEY]");
        assert_eq!(normalized, "The cost is [MONEY] or [MONEY]");

        let text_words = "It costs 100 dollars or 85 euros";
        let normalized_words = normalize_currencies(text_words, "[MONEY]");
        assert_eq!(normalized_words, "It costs [MONEY] or [MONEY]");
    }

    #[test]
    fn test_normalize_percentages() {
        let text = "The growth is 25% and the decline is -5.5%";
        let normalized = normalize_percentages(text, "[PCT]");
        assert_eq!(normalized, "The growth is [PCT] and the decline is [PCT]");
    }

    #[test]
    fn test_normalize_ordinals() {
        let text = "He came 1st, she was 2nd, and they were 3rd";
        let normalized = normalize_ordinals(text, "[ORD]");
        assert_eq!(
            normalized,
            "He came [ORD], she was [ORD], and they were [ORD]"
        );
    }

    #[test]
    fn test_number_normalization_in_cleaner() {
        let cleaner = AdvancedTextCleaner::new()
            .set_normalize_numbers(true)
            .set_normalize_currencies(true)
            .set_normalize_percentages(true)
            .set_normalize_ordinals(true);

        let text = "The 1st item costs $99.99 with a 15% discount, total: 84.99";
        let cleaned = cleaner.clean(text).expect("Operation failed");
        assert_eq!(
            cleaned,
            "the [ordinal] item costs [currency] with a [percent] discount, total: [number]"
        );
    }

    #[test]
    fn test_normalize_dates() {
        let text = "Meeting on 12/25/2023, or December 25, 2023, or 25 December 2023";
        let normalized = normalize_dates(text, "[DATE]");
        assert_eq!(normalized, "Meeting on [DATE], or [DATE], or [DATE]");
    }

    #[test]
    fn test_normalize_times() {
        let text = "Meeting at 14:30 or 2:30 PM or 09:00:15";
        let normalized = normalize_times(text, "[TIME]");
        assert_eq!(normalized, "Meeting at [TIME] or [TIME] or [TIME]");
    }

    #[test]
    fn test_normalize_fractions() {
        let text = "Mix 1/2 cup flour with 2 3/4 cups sugar";
        let normalized = normalize_fractions(text, "[FRACTION]");
        assert_eq!(
            normalized,
            "Mix [FRACTION] cup flour with [FRACTION] cups sugar"
        );
    }

    #[test]
    fn test_normalize_roman_numerals() {
        let text = "Chapter IV discusses Section XVII and Part III";
        let normalized = normalize_roman_numerals(text, "[ROMAN]");
        assert_eq!(
            normalized,
            "Chapter [ROMAN] discusses Section [ROMAN] and Part [ROMAN]"
        );
    }

    #[test]
    fn test_normalize_scientific_notation() {
        let text = "Value is 6.022e23 or 1.23E-10";
        let normalized = normalize_scientific_notation(text, "[SCI]");
        assert_eq!(normalized, "Value is [SCI] or [SCI]");
    }

    #[test]
    fn test_normalize_temperatures() {
        let text = "Temperature is 25°C or 77°F or 298K degrees kelvin";
        let normalized = normalize_temperatures(text, "[TEMP]");
        assert_eq!(normalized, "Temperature is [TEMP] or [TEMP] or [TEMP]");
    }

    #[test]
    fn test_normalize_measurements() {
        let text = "Distance: 5km, weight: 2.5kg, speed: 60mph, storage: 1TB";
        let normalized = normalize_measurements(text, "[MEASURE]");
        assert_eq!(
            normalized,
            "Distance: [MEASURE], weight: [MEASURE], speed: [MEASURE], storage: [MEASURE]"
        );
    }

    #[test]
    fn test_normalize_versions() {
        let text = "Using Python v3.9.1 and Node.js 16.14.0-alpha1";
        let normalized = normalize_versions(text, "[VER]");
        assert_eq!(normalized, "Using Python [VER] and Node.js [VER]");
    }

    #[test]
    fn test_normalize_ip_addresses() {
        let text = "Server at 192.168.1.1 and backup at 10.0.0.255";
        let normalized = normalize_ip_addresses(text, "[IP]");
        assert_eq!(normalized, "Server at [IP] and backup at [IP]");
    }

    #[test]
    fn test_normalize_hex_numbers() {
        let text = "Color #FF5733 or address 0x1A2B3C4D";
        let normalized = normalize_hex_numbers(text, "[HEX]");
        assert_eq!(normalized, "Color [HEX] or address [HEX]");
    }

    #[test]
    fn test_normalize_binary_numbers() {
        let text = "Binary values: 0b1010 and 0b11110000";
        let normalized = normalize_binary_numbers(text, "[BIN]");
        assert_eq!(normalized, "Binary values: [BIN] and [BIN]");
    }

    #[test]
    fn test_enhanced_currency_normalization() {
        let text = "Cost is $100.50, €75.25, ¥10000, or 50 USD";
        let cleaner = AdvancedTextCleaner::new().set_normalize_currencies(true);
        let cleaned = cleaner.clean(text).expect("Operation failed");
        assert_eq!(
            cleaned,
            "cost is [currency], [currency], [currency], or [currency]"
        );
    }

    #[test]
    fn test_comprehensive_advanced_normalization() {
        let cleaner = AdvancedTextCleaner::new()
            .set_normalize_dates(true)
            .set_normalize_times(true)
            .set_normalize_fractions(true)
            .set_normalize_scientific_notation(true)
            .set_normalize_temperatures(true)
            .set_normalize_measurements(true)
            .set_normalize_versions(true)
            .set_normalize_ip_addresses(true)
            .set_normalize_hex_numbers(true)
            .set_normalize_binary_numbers(true);

        let text = "On 12/25/2023 at 14:30, server 192.168.1.1 v2.1.0 measured 25°C, processed 0xFF data with 1/2 efficiency, used 6.022e23 molecules";
        let cleaned = cleaner.clean(text).expect("Operation failed");

        // Should normalize all the different number/data types
        assert!(cleaned.contains("[date]"));
        assert!(cleaned.contains("[time]"));
        assert!(cleaned.contains("[ip]"));
        assert!(cleaned.contains("[version]"));
        assert!(cleaned.contains("[temp]"));
        assert!(cleaned.contains("[hex]"));
        assert!(cleaned.contains("[fraction]"));
        assert!(cleaned.contains("[scientific]"));
    }

    #[test]
    fn test_privacy_focused_cleaner_with_advanced_features() {
        let cleaner = AdvancedTextCleaner::privacy_focused();
        let text = "Meeting on 01/15/2024 at 14:30, contact john@example.com or call (555) 123-4567, server: 192.168.1.100";
        let cleaned = cleaner.clean(text).expect("Operation failed");

        // Privacy-focused should normalize sensitive information
        assert_eq!(
            cleaned,
            "meeting on [date] at [time], contact [email] or call [phone], server: [ip]"
        );
    }

    #[test]
    fn test_normalize_all_numbers_function() {
        let text = "Value: 3.14e-10, temp: 25°C, price: $99.99, percent: 15%, ordinal: 1st, fraction: 1/2, hex: 0xFF, binary: 0b1010, IP: 192.168.1.1, version: v1.2.3, roman: IV";
        let normalized = normalize_all_numbers(text, "[NUM]");

        // Should normalize all number types with the same placeholder
        assert_eq!(normalized, "Value: [NUM], temp: [NUM], price: [NUM], percent: [NUM], ordinal: [NUM], fraction: [NUM], hex: [NUM], binary: [NUM], IP: [NUM], version: [NUM], roman: [NUM]");
    }

    #[test]
    fn test_advanced_placeholder_customization() {
        let cleaner = AdvancedTextCleaner::new()
            .set_normalize_dates(true)
            .set_normalize_temperatures(true)
            .set_normalize_hex_numbers(true)
            .set_advanced_placeholders(
                Some("[CUSTOM_DATE]".to_string()),
                None,
                None,
                None,
                None,
                Some("[CUSTOM_TEMP]".to_string()),
                None,
                None,
                None,
                Some("[CUSTOM_HEX]".to_string()),
                None,
            );

        let text = "Date: 12/25/2023, temp: 25°C, color: #FF0000";
        let cleaned = cleaner.clean(text).expect("Operation failed");
        assert_eq!(
            cleaned,
            "date: [custom_date], temp: [custom_temp], color: [custom_hex]"
        );
    }
}

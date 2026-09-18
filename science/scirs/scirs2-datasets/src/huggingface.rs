//! HuggingFace dataset card metadata parsing and generation.
//!
//! This module provides support for HuggingFace dataset cards — the YAML
//! frontmatter found in `README.md` files of HuggingFace Hub datasets.
//!
//! No external YAML crate is required; a minimal subset parser handles the
//! specific fields used by HuggingFace dataset cards.
//!
//! ## Example
//!
//! ```rust
//! use scirs2_datasets::huggingface::{parse_dataset_card, to_hf_card, card_to_readme};
//!
//! let yaml = "dataset_name: my-dataset\ntask_categories:\n  - text-classification\n";
//! let card = parse_dataset_card(yaml).expect("parse ok");
//! assert_eq!(card.dataset_name, "my-dataset");
//!
//! let card2 = to_hf_card("test-ds", 1000, "classification");
//! let readme = card_to_readme(&card2);
//! assert!(readme.contains("test-ds"));
//! ```

use std::io;
use std::path::Path;

// ─────────────────────────────────────────────────────────────────────────────
// Public error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur when working with HuggingFace dataset cards.
#[derive(Debug)]
pub enum HfError {
    /// I/O error while reading a file.
    Io(io::Error),
    /// Parsing error with a descriptive message.
    Parse(String),
    /// Required field is missing from the dataset card.
    MissingField(&'static str),
}

impl std::fmt::Display for HfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HfError::Io(e) => write!(f, "IO error: {e}"),
            HfError::Parse(msg) => write!(f, "parse error: {msg}"),
            HfError::MissingField(field) => write!(f, "missing field: {field}"),
        }
    }
}

impl std::error::Error for HfError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            HfError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for HfError {
    fn from(e: io::Error) -> Self {
        HfError::Io(e)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Information about a single dataset split (train / validation / test …).
#[derive(Debug, Clone, PartialEq)]
pub struct HfSplitInfo {
    /// Split name — typically `"train"`, `"test"`, or `"validation"`.
    pub name: String,
    /// Number of rows / examples in this split.
    pub num_rows: usize,
    /// Approximate size of this split in bytes.
    pub num_bytes: usize,
}

/// HuggingFace dataset card metadata parsed from the YAML frontmatter in
/// a `README.md` file.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct HfDatasetCard {
    /// Dataset identifier / slug, e.g. `"squad"`.
    pub dataset_name: String,
    /// HuggingFace task category strings, e.g. `["text-classification"]`.
    pub task_categories: Vec<String>,
    /// BCP-47 language codes, e.g. `["en", "fr"]`.
    pub language: Vec<String>,
    /// HuggingFace size category tags, e.g. `["1M<n<10M"]`.
    pub size_categories: Vec<String>,
    /// SPDX license identifier, e.g. `"apache-2.0"`.
    pub license: Option<String>,
    /// Human-readable dataset name that may differ from the slug.
    pub pretty_name: Option<String>,
    /// Per-split statistics (train, test, validation, …).
    pub splits: Vec<HfSplitInfo>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimal YAML parser
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a YAML value that appears as the rest of a `key: <rest>` line.
fn parse_scalar(s: &str) -> String {
    let s = s.trim();
    // Strip surrounding quotes.
    if s.len() >= 2
        && ((s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')))
    {
        s[1..s.len() - 1].to_owned()
    } else {
        s.to_owned()
    }
}

/// Return the number of leading ASCII space characters in `line`.
fn indent_of(line: &str) -> usize {
    line.len() - line.trim_start_matches(' ').len()
}

/// Find the first `:` character that is not inside a quoted string.
fn find_colon(s: &str) -> Option<usize> {
    let mut in_single = false;
    let mut in_double = false;
    for (i, c) in s.char_indices() {
        match c {
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            ':' if !in_single && !in_double => return Some(i),
            _ => {}
        }
    }
    None
}

/// Parse just the fields used in HuggingFace dataset cards from raw YAML text.
///
/// Handles:
/// - `key: scalar` top-level entries
/// - `key:\n  - item\n  - item` block lists (depth 1)
/// - `key: [a, b, c]` inline lists
///
/// Returns a list of `(key, values)` pairs — values may be a list even for
/// scalar entries (list of one element).
fn parse_hf_yaml(yaml: &str) -> Vec<(String, Vec<String>)> {
    let mut result: Vec<(String, Vec<String>)> = Vec::new();
    let lines: Vec<&str> = yaml.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();

        // Skip blank lines, comments, and YAML document markers.
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed == "---" {
            i += 1;
            continue;
        }

        // Only process top-level keys (indent == 0).
        if indent_of(line) != 0 {
            i += 1;
            continue;
        }

        if let Some(colon) = find_colon(line) {
            let key = line[..colon].trim().to_owned();
            let rest = line[colon + 1..].trim();

            if rest.is_empty() {
                // Value spans subsequent lines.
                i += 1;
                let mut items: Vec<String> = Vec::new();
                while i < lines.len() {
                    let sub = lines[i];
                    let sub_trimmed = sub.trim();
                    // Back to top-level — stop
                    if !sub_trimmed.is_empty()
                        && !sub_trimmed.starts_with('#')
                        && indent_of(sub) == 0
                    {
                        break;
                    }
                    if let Some(rest) = sub_trimmed.strip_prefix("- ") {
                        items.push(parse_scalar(rest));
                    } else if sub_trimmed == "-" {
                        items.push(String::new());
                    }
                    // Skip sub-key maps (splits, features); only collect list items.
                    i += 1;
                }
                result.push((key, items));
                continue;
            } else if rest.starts_with('[') && rest.ends_with(']') {
                // Inline list.
                let inner = &rest[1..rest.len() - 1];
                let items: Vec<String> = inner.split(',').map(parse_scalar).collect();
                result.push((key, items));
            } else {
                result.push((key, vec![parse_scalar(rest)]));
            }
        }
        i += 1;
    }

    result
}

/// Parse nested split blocks from YAML text.
///
/// Looks for:
/// ```text
/// splits:
///   - name: train
///     num_rows: 1000
///     num_bytes: 8192
/// ```
fn parse_splits_from_yaml(yaml: &str) -> Vec<HfSplitInfo> {
    let mut splits: Vec<HfSplitInfo> = Vec::new();
    let lines: Vec<&str> = yaml.lines().collect();
    let mut i = 0;

    // Find "splits:" at indent 0
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();
        if indent_of(line) == 0 && trimmed.starts_with("splits:") {
            i += 1;
            // Collect the block
            while i < lines.len() {
                let sub = lines[i];
                let sub_trimmed = sub.trim();
                if !sub_trimmed.is_empty() && !sub_trimmed.starts_with('#') && indent_of(sub) == 0 {
                    break;
                }
                // New list item starting with "- name:" or just "-"
                if sub_trimmed.starts_with("- name:") || sub_trimmed == "-" {
                    let name_part = if let Some(rest) = sub_trimmed.strip_prefix("- name:") {
                        parse_scalar(rest)
                    } else {
                        String::new()
                    };
                    let mut num_rows = 0usize;
                    let mut num_bytes = 0usize;
                    // Read sub-keys until next "- " at same indent or lower
                    let item_indent = indent_of(sub);
                    i += 1;
                    while i < lines.len() {
                        let inner = lines[i];
                        let inner_trimmed = inner.trim();
                        if inner_trimmed.is_empty() || inner_trimmed.starts_with('#') {
                            i += 1;
                            continue;
                        }
                        let inner_indent = indent_of(inner);
                        // Back to parent block or next sibling
                        if inner_indent <= item_indent
                            && (inner_trimmed.starts_with('-') || inner_indent == 0)
                        {
                            break;
                        }
                        if let Some(colon) = find_colon(inner_trimmed) {
                            let k = inner_trimmed[..colon].trim();
                            let v = parse_scalar(&inner_trimmed[colon + 1..]);
                            match k {
                                "num_rows" => {
                                    num_rows = v.parse().unwrap_or(0);
                                }
                                "num_bytes" => {
                                    num_bytes = v.parse().unwrap_or(0);
                                }
                                _ => {}
                            }
                        }
                        i += 1;
                    }
                    splits.push(HfSplitInfo {
                        name: name_part,
                        num_rows,
                        num_bytes,
                    });
                } else {
                    i += 1;
                }
            }
            return splits;
        }
        i += 1;
    }
    splits
}

// ─────────────────────────────────────────────────────────────────────────────
// Extract YAML frontmatter
// ─────────────────────────────────────────────────────────────────────────────

/// Extract the content between the first two `---` markers (YAML frontmatter).
///
/// Returns `None` if no frontmatter markers are present, in which case the
/// caller should treat the entire input as raw YAML.
fn extract_frontmatter(input: &str) -> Option<&str> {
    // The input may start with "---\n" or have the front matter at a non-zero offset.
    // Split on the literal "\n---\n" or "---\n" at position 0.
    let input_trimmed = input.trim_start();
    if !input_trimmed.starts_with("---") {
        return None;
    }
    // Find the end of the opening "---" line.
    let after_open = input_trimmed.find('\n').map(|p| p + 1)?;
    let rest = &input_trimmed[after_open..];
    // Find the closing "---" line.
    let close = rest.find("\n---")?;
    Some(&rest[..close])
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a HuggingFace dataset card from a YAML string.
///
/// The string may be either:
/// - Raw YAML (no `---` delimiters), or
/// - A full README.md string with YAML frontmatter between `---` markers.
///
/// Only the fields relevant to `HfDatasetCard` are extracted; unknown keys are
/// silently ignored.
///
/// # Errors
///
/// Returns `HfError::Parse` if a required structural element is malformed.
pub fn parse_dataset_card(yaml_str: &str) -> Result<HfDatasetCard, HfError> {
    // Prefer frontmatter if present; otherwise treat as raw YAML.
    let yaml_body = extract_frontmatter(yaml_str).unwrap_or(yaml_str);

    let pairs = parse_hf_yaml(yaml_body);
    let mut card = HfDatasetCard::default();

    for (key, values) in &pairs {
        match key.as_str() {
            "dataset_name" => {
                card.dataset_name = values.first().cloned().unwrap_or_default();
            }
            "task_categories" => {
                card.task_categories = values.clone();
            }
            "language" => {
                card.language = values.clone();
            }
            "size_categories" => {
                card.size_categories = values.clone();
            }
            "license" => {
                let s = values.first().cloned().unwrap_or_default();
                if !s.is_empty() {
                    card.license = Some(s);
                }
            }
            "pretty_name" => {
                let s = values.first().cloned().unwrap_or_default();
                if !s.is_empty() {
                    card.pretty_name = Some(s);
                }
            }
            _ => {}
        }
    }

    // Parse structured splits block separately (needs nested parsing).
    card.splits = parse_splits_from_yaml(yaml_body);

    Ok(card)
}

/// Discover and parse the dataset card from a local directory.
///
/// Searches for `README.md` in `dir` and parses its YAML frontmatter as an
/// `HfDatasetCard`.
///
/// # Errors
///
/// - `HfError::Io` — directory or `README.md` file is not accessible.
/// - `HfError::Parse` — frontmatter could not be parsed.
/// - `HfError::MissingField` — `README.md` has no YAML frontmatter.
pub fn load_dataset_card(dir: &Path) -> Result<HfDatasetCard, HfError> {
    let readme_path = dir.join("README.md");
    let content = std::fs::read_to_string(&readme_path)?;
    if extract_frontmatter(&content).is_none() {
        return Err(HfError::MissingField("YAML frontmatter (---) in README.md"));
    }
    parse_dataset_card(&content)
}

/// Build an `HfDatasetCard` from basic parameters.
///
/// This is a convenience constructor used when converting a SciRS2 dataset to
/// a HuggingFace-compatible card.
///
/// * `name` — dataset slug
/// * `n_rows` — number of training samples
/// * `task` — HuggingFace task category string (e.g. `"classification"`)
pub fn to_hf_card(name: &str, n_rows: usize, task: &str) -> HfDatasetCard {
    let size_cat = size_category(n_rows);
    HfDatasetCard {
        dataset_name: name.to_owned(),
        task_categories: vec![task.to_owned()],
        language: vec!["en".to_owned()],
        size_categories: vec![size_cat],
        license: None,
        pretty_name: Some(name.to_owned()),
        splits: vec![HfSplitInfo {
            name: "train".to_owned(),
            num_rows: n_rows,
            num_bytes: n_rows * 64, // rough estimate
        }],
    }
}

/// Render an `HfDatasetCard` as minimal HuggingFace `README.md` content.
///
/// The output has YAML frontmatter delimited by `---` markers followed by a
/// brief Markdown body.
pub fn card_to_readme(card: &HfDatasetCard) -> String {
    let mut out = String::from("---\n");

    out.push_str(&format!("dataset_name: {}\n", yaml_str(&card.dataset_name)));

    if !card.task_categories.is_empty() {
        out.push_str("task_categories:\n");
        for tc in &card.task_categories {
            out.push_str(&format!("  - {}\n", yaml_str(tc)));
        }
    }

    if !card.language.is_empty() {
        out.push_str("language:\n");
        for lang in &card.language {
            out.push_str(&format!("  - {}\n", yaml_str(lang)));
        }
    }

    if !card.size_categories.is_empty() {
        out.push_str("size_categories:\n");
        for sc in &card.size_categories {
            out.push_str(&format!("  - {}\n", yaml_str(sc)));
        }
    }

    if let Some(ref lic) = card.license {
        out.push_str(&format!("license: {}\n", yaml_str(lic)));
    }

    if let Some(ref pn) = card.pretty_name {
        out.push_str(&format!("pretty_name: {}\n", yaml_str(pn)));
    }

    if !card.splits.is_empty() {
        out.push_str("splits:\n");
        for split in &card.splits {
            out.push_str(&format!(
                "  - name: {}\n    num_rows: {}\n    num_bytes: {}\n",
                yaml_str(&split.name),
                split.num_rows,
                split.num_bytes,
            ));
        }
    }

    out.push_str("---\n\n");
    out.push_str(&format!("# {}\n\n", card.dataset_name));

    if let Some(ref pn) = card.pretty_name {
        out.push_str(&format!("{}\n\n", pn));
    }

    if !card.task_categories.is_empty() {
        out.push_str(&format!("Tasks: {}\n", card.task_categories.join(", ")));
    }

    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Return the HuggingFace size category tag for a number of rows.
fn size_category(n: usize) -> String {
    match n {
        0..=999 => "n<1K".to_owned(),
        1_000..=9_999 => "1K<n<10K".to_owned(),
        10_000..=99_999 => "10K<n<100K".to_owned(),
        100_000..=999_999 => "100K<n<1M".to_owned(),
        1_000_000..=9_999_999 => "1M<n<10M".to_owned(),
        _ => "10M<n<100M".to_owned(),
    }
}

/// Escape a YAML string value if it contains characters requiring quoting.
fn yaml_str(s: &str) -> String {
    if s.contains(':') || s.contains('#') || s.contains('"') || s.contains('\'') {
        format!("\"{}\"", s.replace('"', "\\\""))
    } else {
        s.to_owned()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Sample YAML string covering all key fields.
    const SAMPLE_YAML: &str = "\
dataset_name: squad
task_categories:
  - question-answering
language:
  - en
size_categories:
  - 100K<n<1M
license: cc-by-4.0
pretty_name: Stanford Question Answering Dataset
splits:
  - name: train
    num_rows: 87599
    num_bytes: 29344551
  - name: validation
    num_rows: 10570
    num_bytes: 3519936
";

    // 1. parse_dataset_card parses a sample YAML string correctly
    #[test]
    fn test_parse_dataset_card_basic() {
        let card = parse_dataset_card(SAMPLE_YAML).expect("should parse");
        assert_eq!(card.dataset_name, "squad");
        assert_eq!(card.task_categories, vec!["question-answering"]);
        assert_eq!(card.language, vec!["en"]);
        assert_eq!(card.size_categories, vec!["100K<n<1M"]);
        assert_eq!(card.license, Some("cc-by-4.0".to_owned()));
        assert_eq!(
            card.pretty_name,
            Some("Stanford Question Answering Dataset".to_owned())
        );
    }

    // 2. parse_dataset_card parses splits correctly
    #[test]
    fn test_parse_splits() {
        let card = parse_dataset_card(SAMPLE_YAML).expect("should parse");
        assert_eq!(card.splits.len(), 2);
        assert_eq!(card.splits[0].name, "train");
        assert_eq!(card.splits[0].num_rows, 87599);
        assert_eq!(card.splits[0].num_bytes, 29344551);
        assert_eq!(card.splits[1].name, "validation");
        assert_eq!(card.splits[1].num_rows, 10570);
    }

    // 3. to_hf_card creates a card with correct n_rows
    #[test]
    fn test_to_hf_card_n_rows() {
        let card = to_hf_card("my-ds", 5000, "classification");
        assert_eq!(card.dataset_name, "my-ds");
        assert_eq!(card.task_categories, vec!["classification"]);
        assert!(!card.splits.is_empty());
        let train_split = card.splits.iter().find(|s| s.name == "train");
        assert!(train_split.is_some(), "should have a train split");
        assert_eq!(train_split.expect("verified above").num_rows, 5000);
    }

    // 4. card_to_readme contains the dataset name
    #[test]
    fn test_card_to_readme_contains_name() {
        let card = to_hf_card("awesome-dataset", 100, "text-classification");
        let readme = card_to_readme(&card);
        assert!(
            readme.contains("awesome-dataset"),
            "README should contain the dataset name"
        );
    }

    // 5. load_dataset_card returns Err for non-existent directory
    #[test]
    fn test_load_dataset_card_nonexistent() {
        let result = load_dataset_card(Path::new("/nonexistent/path/that/does/not/exist"));
        assert!(result.is_err(), "should fail for non-existent path");
    }

    // 6. card_to_readme -> parse_dataset_card round-trip preserves dataset_name
    #[test]
    fn test_roundtrip_dataset_name() {
        let original = to_hf_card("roundtrip-test", 2000, "regression");
        let readme = card_to_readme(&original);
        let parsed = parse_dataset_card(&readme).expect("round-trip parse should succeed");
        assert_eq!(
            parsed.dataset_name, original.dataset_name,
            "dataset_name should survive round-trip"
        );
    }

    // 7. load_dataset_card reads a real README.md from a temp directory
    #[test]
    fn test_load_dataset_card_from_temp_dir() {
        let tmp_dir = std::env::temp_dir().join("scirs2_hf_test_load_card");
        std::fs::create_dir_all(&tmp_dir).expect("create temp dir");

        let yaml_fm = "---\ndataset_name: temp-dataset\ntask_categories:\n  - classification\nlanguage:\n  - en\n---\n# temp-dataset\n";
        let readme_path = tmp_dir.join("README.md");
        let mut f = std::fs::File::create(&readme_path).expect("create README.md");
        f.write_all(yaml_fm.as_bytes()).expect("write");

        let card = load_dataset_card(&tmp_dir).expect("load card");
        assert_eq!(card.dataset_name, "temp-dataset");
        assert_eq!(card.task_categories, vec!["classification"]);

        // Cleanup
        let _ = std::fs::remove_file(&readme_path);
        let _ = std::fs::remove_dir(&tmp_dir);
    }

    // 8. load_dataset_card returns MissingField error when README has no frontmatter
    #[test]
    fn test_load_dataset_card_no_frontmatter() {
        let tmp_dir = std::env::temp_dir().join("scirs2_hf_test_no_fm");
        std::fs::create_dir_all(&tmp_dir).expect("create temp dir");

        let readme_path = tmp_dir.join("README.md");
        let mut f = std::fs::File::create(&readme_path).expect("create README.md");
        f.write_all(b"# Plain README\n\nNo frontmatter here.\n")
            .expect("write");

        let result = load_dataset_card(&tmp_dir);
        assert!(
            matches!(result, Err(HfError::MissingField(_))),
            "expected MissingField, got: {:?}",
            result
        );

        let _ = std::fs::remove_file(&readme_path);
        let _ = std::fs::remove_dir(&tmp_dir);
    }

    // 9. size_category helper returns expected values
    #[test]
    fn test_size_categories() {
        assert_eq!(size_category(500), "n<1K");
        assert_eq!(size_category(5000), "1K<n<10K");
        assert_eq!(size_category(50_000), "10K<n<100K");
        assert_eq!(size_category(500_000), "100K<n<1M");
        assert_eq!(size_category(5_000_000), "1M<n<10M");
        assert_eq!(size_category(50_000_000), "10M<n<100M");
    }

    // 10. parse_dataset_card handles inline list syntax
    #[test]
    fn test_parse_inline_list() {
        let yaml = "dataset_name: inline-test\nlanguage: [en, fr, de]\n";
        let card = parse_dataset_card(yaml).expect("parse");
        assert_eq!(card.dataset_name, "inline-test");
        assert_eq!(card.language, vec!["en", "fr", "de"]);
    }
}

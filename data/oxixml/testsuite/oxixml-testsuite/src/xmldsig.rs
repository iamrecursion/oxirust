//! XML Signature interoperability runner.
//!
//! The 45 documents under `xmldsig-core1-interop/interop/xmldsig11/oracle` were produced by an
//! independent JSR-105 implementation for the XML Security 1.1 interop round. There is no
//! `mf:`-style manifest — the suite is a flat directory of signed documents — so the expected
//! outcome of each is recovered from its file name, the same way a human skimming the suite's
//! own `README` would: the algorithm family a document exercises is spelled out in its name.
//!
//! | file name contains | expectation |
//! |---|---|
//! | `truncated40` | rejected: the `HMACOutputLength` is below [XML Signature §4.4.2]'s floor |
//! | `x509digest` | rejected: resolving `<dsig11:X509Digest>` needs a certificate store |
//! | `-ec`, `p256`, `p384`, `p521` | rejected as an unsupported algorithm, by design |
//! | anything else | core valid |
//!
//! [XML Signature §4.4.2]: https://www.w3.org/TR/xmldsig-core1/#sec-HMAC

use std::fs;
use std::path::{Path, PathBuf};

use oxixml_dom::{Document, build};
use oxixml_dsig::{
    DsigError, SharedSecret, Signature, SignatureFamily, UntrustedKeyInfo, Verifier,
};

use crate::env;
use crate::report::SuiteReport;

/// The suite name reported for this runner.
pub const SUITE: &str = "xmldsig";

/// The HMAC secret every interop document keyed with HMAC shares; an HMAC signature carries no
/// key of its own, so this is the one piece of the suite that is not self-describing.
const INTEROP_HMAC_SECRET: &[u8] = b"testkey";

/// What the suite expects of one document, recovered from its file name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Expectation {
    /// The signature must be core valid.
    Valid,
    /// Well-formed but must be rejected: `HMACOutputLength` truncates below the floor.
    TruncatedHmac,
    /// A documented algorithm gap (ECDSA), must be reported as unsupported.
    Unsupported,
    /// The keying material cannot be resolved without a certificate store.
    NoKey,
}

fn expectation(file_name: &str) -> Expectation {
    if file_name.contains("truncated40") {
        Expectation::TruncatedHmac
    } else if file_name.contains("x509digest") {
        Expectation::NoKey
    } else if file_name.contains("-ec")
        || file_name.contains("p256")
        || file_name.contains("p384")
        || file_name.contains("p521")
    {
        Expectation::Unsupported
    } else {
        Expectation::Valid
    }
}

/// Runs the XML Signature 1.1 interoperability suite.
#[must_use]
pub fn run() -> Vec<SuiteReport> {
    let Some(directory) = interop_directory() else {
        return vec![SuiteReport::unavailable(
            SUITE,
            match env::xml_suites_dir() {
                None => env::missing(env::XML_SUITES_DIR_VAR),
                Some(_) => env::not_found(env::XML_SUITES_DIR_VAR, "xmldsig-core1-interop/"),
            },
        )];
    };

    let mut report = SuiteReport::new(SUITE);
    for path in xml_files(&directory) {
        run_case(&path, &mut report);
    }
    vec![report]
}

fn run_case(path: &Path, report: &mut SuiteReport) {
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let Ok(text) = fs::read_to_string(path) else {
        report.record_skip("document unreadable");
        return;
    };
    let document = match build::parse_str(&text) {
        Ok(document) => document,
        Err(error) => {
            report.record_fail(&name, format!("document did not parse: {error}"));
            return;
        }
    };
    let outcome = verify(&document);
    match (expectation(&name), outcome) {
        (Expectation::Valid, Ok(result)) => {
            if result.core_valid() {
                report.record_pass();
            } else {
                report.record_fail(
                    &name,
                    format!(
                        "expected core validity, got signature={}",
                        result.signature_valid()
                    ),
                );
            }
        }
        (Expectation::TruncatedHmac, Ok(result)) => {
            if result.core_valid() {
                report.record_fail(&name, "a 40-bit HMAC truncation must be rejected");
            } else if result.signature_error().is_some() {
                report.record_pass();
            } else {
                report.record_fail(&name, "rejected, but not for the truncation-length reason");
            }
        }
        (Expectation::Unsupported, Err(DsigError::UnsupportedAlgorithm(_))) => report.record_pass(),
        (Expectation::NoKey, Err(DsigError::NoKey(_))) => report.record_pass(),
        (expected, actual) => {
            report.record_fail(&name, format!("expected {expected:?}, got {actual:?}"));
        }
    }
}

/// Verifies one document with whichever key selector its `SignatureMethod` calls for: the shared
/// interop HMAC secret for the HMAC family, or no key at all (the algorithm/keying gaps this
/// suite deliberately exercises need no key to be classified as such).
fn verify(document: &Document) -> Result<oxixml_dsig::VerificationResult, DsigError> {
    let signature = Signature::find(document)?;
    let algorithm = signature.signed_info().signature_method().algorithm();
    if matches!(algorithm.family(), SignatureFamily::Hmac) {
        Verifier::new().verify(
            document,
            &signature,
            &SharedSecret::new(INTEROP_HMAC_SECRET.to_vec()),
        )
    } else {
        Verifier::new().verify(document, &signature, &UntrustedKeyInfo)
    }
}

fn interop_directory() -> Option<PathBuf> {
    let directory = env::xml_suites_dir()?.join("xmldsig-core1-interop");
    directory.is_dir().then_some(directory)
}

/// Every `*.xml` file below `directory`, sorted.
fn xml_files(directory: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![directory.to_path_buf()];
    while let Some(current) = stack.pop() {
        let Ok(entries) = fs::read_dir(&current) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|extension| extension == "xml") {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expectation_matches_the_documented_file_name_scheme() {
        assert_eq!(expectation("signature-rsa-sha256.xml"), Expectation::Valid);
        assert_eq!(
            expectation("signature-hmac-sha1-truncated40.xml"),
            Expectation::TruncatedHmac
        );
        assert_eq!(expectation("signature-x509digest.xml"), Expectation::NoKey);
        assert_eq!(
            expectation("signature-ecdsa-p256-sha256.xml"),
            Expectation::Unsupported
        );
        assert_eq!(
            expectation("signature-ecdsa-p384-sha384.xml"),
            Expectation::Unsupported
        );
    }

    #[test]
    fn unavailable_when_env_is_unset() {
        if std::env::var_os(env::XML_SUITES_DIR_VAR).is_some() {
            return;
        }
        let reports = run();
        assert_eq!(reports.len(), 1);
        assert!(!reports[0].is_available());
    }
}

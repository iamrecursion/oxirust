//! DE ephemeris file resolution for the `pos` subcommand.
//!
//! Resolution order:
//! 1. `--de PATH` (explicit CLI flag).
//! 2. `$OXIEPH_DE` environment variable.
//! 3. `./data/de440/linux_p1550p2650.440`, resolved **relative to the
//!    current working directory** at run time (never an absolute path
//!    baked into the binary).

use std::path::{Path, PathBuf};

use crate::errors::CliError;

/// The default DE file path, relative to the current working directory.
pub const DEFAULT_DE_PATH: &str = "data/de440/linux_p1550p2650.440";

/// Name of the environment variable consulted after `--de`.
pub const DE_ENV_VAR: &str = "OXIEPH_DE";

/// Resolves the DE file path per the order documented on this module.
#[must_use]
pub fn resolve_de_path(explicit: Option<&Path>) -> PathBuf {
    if let Some(p) = explicit {
        return p.to_path_buf();
    }
    if let Ok(env_path) = std::env::var(DE_ENV_VAR) {
        if !env_path.is_empty() {
            return PathBuf::from(env_path);
        }
    }
    PathBuf::from(DEFAULT_DE_PATH)
}

/// Resolves the DE file path ([`resolve_de_path`]) and reads it into
/// memory, mapping an I/O failure to the friendly
/// [`CliError::DeFileMissing`]. Shared by every subcommand that
/// evaluates a DE ephemeris (`pos`, `pos --all`, `chart`): each call site
/// still owns the returned bytes and parses them with
/// `oxiephemeris_de::DeFile::parse` itself, since [`DeFile`](oxiephemeris_de::DeFile)
/// borrows from the buffer and cannot be returned from here.
///
/// # Errors
///
/// [`CliError::DeFileMissing`] if the file cannot be read.
pub fn read_de_bytes(explicit: Option<&Path>) -> Result<Vec<u8>, CliError> {
    let de_path = resolve_de_path(explicit);
    std::fs::read(&de_path).map_err(|source| CliError::DeFileMissing {
        path: de_path,
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::{resolve_de_path, DEFAULT_DE_PATH, DE_ENV_VAR};
    use std::path::PathBuf;

    #[test]
    fn explicit_path_wins() {
        let p = resolve_de_path(Some(std::path::Path::new("/tmp/whatever.440")));
        assert_eq!(p, PathBuf::from("/tmp/whatever.440"));
    }

    #[test]
    fn falls_back_to_default_when_env_absent() {
        // Guard: only assert the fallback when the env var truly is not
        // set in this test process, to avoid cross-test interference.
        if std::env::var(DE_ENV_VAR).is_err() {
            let p = resolve_de_path(None);
            assert_eq!(p, PathBuf::from(DEFAULT_DE_PATH));
        }
    }
}

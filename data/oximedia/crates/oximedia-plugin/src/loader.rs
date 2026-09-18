//! Dynamic plugin loader.
//!
//! This module is only available when the `dynamic-loading` feature
//! is enabled. It uses `libloading` to load shared libraries and
//! extract the plugin entry points.
//!
//! # Security Considerations — Trust Model
//!
//! **Loading a shared library executes arbitrary, unsandboxed code as
//! part of this process.** `dlopen`/`LoadLibrary` runs the library's own
//! constructors before this module gets a chance to do anything, and the
//! two exported symbols this module then calls
//! (`oximedia_plugin_api_version`, `oximedia_plugin_create`) can do
//! anything a native function can do. The API-version handshake only
//! guards against *accidental* ABI mismatch — it is a compatibility
//! check, not a security boundary. **Only ever point this loader at a
//! `path` you trust as much as your own process image.**
//!
//! Two loading entry points are provided on [`LoadedPlugin`]:
//!
//! - [`LoadedPlugin::load_unchecked`] (also reachable via the original
//!   [`LoadedPlugin::load`] name, kept for backward compatibility) opens
//!   and executes the file with no integrity verification beyond the API
//!   version check above.
//! - [`LoadedPlugin::load_with_digest`] additionally computes the
//!   SHA-256 digest of the file and compares it against a caller-supplied
//!   expected value **before** the file is ever opened as a library,
//!   returning [`PluginError::IntegrityMismatch`] on a mismatch without
//!   executing a single byte of it. This is the **recommended** entry
//!   point whenever the expected plugin contents are known ahead of time
//!   (e.g. pinned in a manifest, fetched from a registry that publishes
//!   checksums, etc). It does not, by itself, establish trust in the
//!   digest's *source* — verify the digest itself came from somewhere
//!   trustworthy.
// Dynamic plugin loading fundamentally requires unsafe code to call into
// foreign shared libraries. The safety invariants are documented inline.
#![allow(unsafe_code)]

use crate::error::{PluginError, PluginResult};
use crate::traits::{CodecPlugin, PluginApiVersionFn, PluginCreateFn, PLUGIN_API_VERSION};
use libloading::{Library, Symbol};
use sha2::{Digest as _, Sha256};
use std::path::Path;
use std::sync::Arc;

/// Compute the SHA-256 digest of the file at `path`, as a lowercase hex string.
///
/// This reads the entire file into memory. It is intended for verifying
/// plugin file integrity before [`LoadedPlugin::load_unchecked`] is ever
/// called on the same path — see [`LoadedPlugin::load_with_digest`].
///
/// # Errors
///
/// Returns [`PluginError::LoadFailed`] if the file cannot be read.
pub fn sha256_hex(path: &Path) -> PluginResult<String> {
    let data = std::fs::read(path).map_err(|e| {
        PluginError::LoadFailed(format!(
            "Failed to read '{}' for integrity check: {e}",
            path.display()
        ))
    })?;
    let mut hasher = Sha256::new();
    hasher.update(&data);
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest.iter() {
        hex.push_str(&format!("{byte:02x}"));
    }
    Ok(hex)
}

/// Verify that the file at `path` has the expected SHA-256 digest.
///
/// `expected_sha256_hex` is a hex-encoded digest, compared
/// case-insensitively; an optional leading `sha256:` prefix is accepted
/// and stripped for convenience.
///
/// # Errors
///
/// Returns [`PluginError::IntegrityMismatch`] if the digests do not
/// match, or [`PluginError::LoadFailed`] if the file cannot be read.
pub fn verify_sha256(path: &Path, expected_sha256_hex: &str) -> PluginResult<()> {
    let expected = expected_sha256_hex
        .strip_prefix("sha256:")
        .unwrap_or(expected_sha256_hex)
        .trim();
    let actual = sha256_hex(path)?;
    if actual.eq_ignore_ascii_case(expected) {
        Ok(())
    } else {
        Err(PluginError::IntegrityMismatch {
            path: path.display().to_string(),
            expected: expected.to_ascii_lowercase(),
            actual,
        })
    }
}

/// A plugin loaded from a shared library.
///
/// This struct keeps the `Library` handle alive for as long as the
/// plugin instance exists. Dropping this struct will unload the
/// shared library (after the plugin Arc's refcount reaches zero).
pub struct LoadedPlugin {
    /// The loaded library handle. Must be kept alive as long as
    /// the plugin is in use.
    _library: Library,
    /// The plugin instance created from the library.
    plugin: Arc<dyn CodecPlugin>,
}

impl std::fmt::Debug for LoadedPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedPlugin")
            .field("plugin", &"<dyn CodecPlugin>")
            .finish_non_exhaustive()
    }
}

impl LoadedPlugin {
    /// Load a plugin from a shared library file.
    ///
    /// This is a backward-compatible alias for
    /// [`LoadedPlugin::load_unchecked`] — it performs **no integrity
    /// verification** of the file before executing its code. Prefer
    /// [`LoadedPlugin::load_with_digest`] wherever the expected plugin
    /// contents are known ahead of time; see the module-level "Security
    /// Considerations" section for the full trust model.
    ///
    /// # Errors
    ///
    /// - [`PluginError::LoadFailed`] if the library cannot be opened
    ///   or required symbols are missing
    /// - [`PluginError::ApiIncompatible`] if the API version does not match
    ///
    /// # Safety
    ///
    /// This function loads and executes code from an external shared library.
    /// The caller must ensure the library is from a trusted source.
    pub fn load(path: &Path) -> PluginResult<Self> {
        Self::load_unchecked(path)
    }

    /// Load a plugin from a shared library file, verifying its SHA-256
    /// digest against `expected_sha256_hex` **before** opening it.
    ///
    /// This is the **recommended** loading entry point: if the file's
    /// contents do not match the expected digest, this returns
    /// [`PluginError::IntegrityMismatch`] without ever calling `dlopen`/
    /// `LoadLibrary` on the path, so no code from a tampered or
    /// substituted file is executed. See [`verify_sha256`] for the
    /// accepted digest formats.
    ///
    /// Note that this only verifies the file matches a digest the caller
    /// supplied — it is the caller's responsibility to ensure that
    /// digest itself came from a trustworthy source (e.g. a signed
    /// manifest), not an attacker who could substitute both the plugin
    /// and the expected digest together.
    ///
    /// # Errors
    ///
    /// - [`PluginError::IntegrityMismatch`] if the computed digest does
    ///   not match `expected_sha256_hex`
    /// - [`PluginError::LoadFailed`] if the file cannot be read/opened
    ///   or required symbols are missing
    /// - [`PluginError::ApiIncompatible`] if the API version does not match
    ///
    /// # Safety
    ///
    /// This function loads and executes code from an external shared
    /// library once the digest check passes. The caller must ensure the
    /// expected digest itself is trustworthy.
    pub fn load_with_digest(path: &Path, expected_sha256_hex: &str) -> PluginResult<Self> {
        verify_sha256(path, expected_sha256_hex)?;
        Self::load_unchecked(path)
    }

    /// Load a plugin from a shared library file, performing **no
    /// integrity verification** of its contents.
    ///
    /// The loader performs the following steps:
    /// 1. Opens the shared library
    /// 2. Looks up `oximedia_plugin_api_version` and validates it
    /// 3. Looks up `oximedia_plugin_create` and calls it
    /// 4. Takes ownership of the returned plugin pointer
    ///
    /// # Errors
    ///
    /// - [`PluginError::LoadFailed`] if the library cannot be opened
    ///   or required symbols are missing
    /// - [`PluginError::ApiIncompatible`] if the API version does not match
    ///
    /// # Safety
    ///
    /// This function loads and executes code from an external shared library.
    /// The caller must ensure the library is from a trusted source. Prefer
    /// [`LoadedPlugin::load_with_digest`] when the expected contents of the
    /// library are known ahead of time.
    pub fn load_unchecked(path: &Path) -> PluginResult<Self> {
        // Safety: Loading external code is inherently unsafe.
        // We validate API version before calling any other functions.
        let library = unsafe {
            Library::new(path).map_err(|e| {
                PluginError::LoadFailed(format!("Failed to open '{}': {e}", path.display()))
            })?
        };

        // Check API version first (least risky call)
        let api_version = {
            let api_version_fn: Symbol<PluginApiVersionFn> = unsafe {
                library.get(b"oximedia_plugin_api_version").map_err(|e| {
                    PluginError::LoadFailed(format!(
                        "Missing 'oximedia_plugin_api_version' symbol in '{}': {e}",
                        path.display()
                    ))
                })?
            };
            unsafe { api_version_fn() }
        };

        if api_version != PLUGIN_API_VERSION {
            return Err(PluginError::ApiIncompatible(format!(
                "Plugin '{}' has API v{api_version}, host expects v{PLUGIN_API_VERSION}",
                path.display()
            )));
        }

        // Create the plugin instance
        let plugin = {
            let create_fn: Symbol<PluginCreateFn> = unsafe {
                library.get(b"oximedia_plugin_create").map_err(|e| {
                    PluginError::LoadFailed(format!(
                        "Missing 'oximedia_plugin_create' symbol in '{}': {e}",
                        path.display()
                    ))
                })?
            };

            let raw_plugin = unsafe { create_fn() };
            if raw_plugin.is_null() {
                return Err(PluginError::InitFailed(format!(
                    "Plugin create function returned null for '{}'",
                    path.display()
                )));
            }

            // Safety: The plugin was created by Box::into_raw in the shared library.
            // We take ownership here. The raw pointer came from PluginCreateFn which
            // is documented to return a Box::into_raw'd pointer.
            unsafe { Arc::from_raw(raw_plugin) }
        };

        // Log plugin info
        let info = plugin.info();
        tracing::info!(
            "Loaded plugin from '{}': {} v{} [{}]{}",
            path.display(),
            info.name,
            info.version,
            info.license,
            if info.patent_encumbered {
                " (patent-encumbered)"
            } else {
                ""
            }
        );

        if info.patent_encumbered {
            tracing::warn!(
                "Plugin '{}' contains patent-encumbered codecs. \
                 Ensure you have appropriate licenses before use.",
                info.name
            );
        }

        let caps = plugin.capabilities();
        tracing::debug!(
            "Plugin '{}' provides {} codec(s): {}",
            info.name,
            caps.len(),
            caps.iter()
                .map(|c| c.codec_name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );

        Ok(Self {
            _library: library,
            plugin,
        })
    }

    /// Get a reference to the loaded plugin.
    pub fn plugin(&self) -> &Arc<dyn CodecPlugin> {
        &self.plugin
    }

    /// Consume this `LoadedPlugin` and return the plugin Arc.
    ///
    /// Note: The caller must ensure the `Library` is kept alive
    /// for as long as the plugin is used. In practice, this is
    /// typically called only by the registry which manages lifetimes.
    pub fn into_plugin(self) -> Arc<dyn CodecPlugin> {
        // We intentionally leak the library handle here because the
        // plugin code references it. The registry owns the plugin Arc
        // and when it drops, the library symbols may still be needed
        // during drop. Leaking is safe and prevents use-after-free.
        let library = self._library;
        std::mem::forget(library);
        self.plugin
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_nonexistent_library() {
        let result = LoadedPlugin::load(Path::new("/nonexistent/plugin.so"));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, PluginError::LoadFailed(_)));
    }

    #[test]
    fn test_load_invalid_library() {
        // Create a temp file that is not a valid shared library
        let dir = std::env::temp_dir().join("oximedia-plugin-test-loader");
        std::fs::create_dir_all(&dir).expect("dir creation should succeed");
        let fake_lib = dir.join("fake_plugin.so");
        std::fs::write(&fake_lib, b"not a real library").expect("write should succeed");

        let result = LoadedPlugin::load(&fake_lib);
        assert!(result.is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_sha256_hex_known_vector() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let dir = std::env::temp_dir().join("oximedia-plugin-test-loader-sha256");
        std::fs::create_dir_all(&dir).expect("dir creation should succeed");
        let empty_file = dir.join("empty.bin");
        std::fs::write(&empty_file, b"").expect("write should succeed");

        let digest = sha256_hex(&empty_file).expect("hashing should succeed");
        assert_eq!(
            digest,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_with_digest_mismatch_is_rejected() {
        // A caller-supplied digest that does not match the file's actual
        // contents must be rejected *before* the loader ever tries to
        // open the file as a shared library.
        let dir = std::env::temp_dir().join("oximedia-plugin-test-loader-digest-mismatch");
        std::fs::create_dir_all(&dir).expect("dir creation should succeed");
        let fake_lib = dir.join("fake_plugin.so");
        std::fs::write(&fake_lib, b"not a real library").expect("write should succeed");

        let wrong_digest = "0".repeat(64);
        let result = LoadedPlugin::load_with_digest(&fake_lib, &wrong_digest);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, PluginError::IntegrityMismatch { .. }),
            "expected IntegrityMismatch, got: {err:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_with_digest_match_proceeds_past_integrity_check() {
        // A correct digest must pass the integrity check and fall through
        // to the normal load path (which then fails because the file
        // isn't a real shared library) rather than short-circuiting into
        // a false "success" or being rejected as a mismatch.
        let dir = std::env::temp_dir().join("oximedia-plugin-test-loader-digest-match");
        std::fs::create_dir_all(&dir).expect("dir creation should succeed");
        let fake_lib = dir.join("fake_plugin.so");
        std::fs::write(&fake_lib, b"not a real library").expect("write should succeed");

        let correct_digest = sha256_hex(&fake_lib).expect("hashing should succeed");
        let result = LoadedPlugin::load_with_digest(&fake_lib, &correct_digest);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, PluginError::LoadFailed(_)),
            "expected LoadFailed (invalid library), got: {err:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_verify_sha256_accepts_prefix_and_case_insensitive() {
        let dir = std::env::temp_dir().join("oximedia-plugin-test-loader-sha256-prefix");
        std::fs::create_dir_all(&dir).expect("dir creation should succeed");
        let file = dir.join("data.bin");
        std::fs::write(&file, b"hello").expect("write should succeed");

        let digest = sha256_hex(&file).expect("hashing should succeed");
        let prefixed = format!("sha256:{}", digest.to_ascii_uppercase());
        assert!(verify_sha256(&file, &prefixed).is_ok());

        let _ = std::fs::remove_dir_all(&dir);
    }
}

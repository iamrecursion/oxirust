//! Process-wide TLS crypto provider bootstrap.
//!
//! The workspace configures `reqwest`/`rustls` with the `rustls-no-provider`
//! feature so that the default build stays 100% Pure Rust (no `aws-lc-sys` /
//! `ring` C or assembly code compiled in). That means `rustls` no longer
//! ships a compiled-in default [`CryptoProvider`], and one **must** be
//! installed at process level before the first TLS connection is opened,
//! otherwise building a TLS-capable client (e.g. `reqwest::Client`) fails at
//! runtime with `no process-level CryptoProvider available`.
//!
//! Every storage/media client constructor in this crate that creates a
//! `reqwest::Client` calls [`install_default_crypto_provider`] first. The
//! call is backed by [`std::sync::Once`], so repeated invocations (from this
//! crate or from other crates that install their own provider earlier) are
//! free and safe.
//!
//! [`CryptoProvider`]: https://docs.rs/rustls/latest/rustls/crypto/struct.CryptoProvider.html

use std::sync::Once;

static INSTALL_CRYPTO_PROVIDER: Once = Once::new();

/// Installs the Pure-Rust [`rustls-rustcrypto`](https://docs.rs/rustls-rustcrypto)
/// crypto provider as the process-wide default `rustls` provider.
///
/// Idempotent and safe to call from multiple entry points (library
/// constructors, CLI `main`, server startup, ...) — the installation itself
/// only happens once per process, guarded by a [`std::sync::Once`].
///
/// `install_default` reports an already-installed provider (e.g. installed
/// first by another crate in the same process) as an `Err`; that is not an
/// error condition for us — any installed provider is sufficient — so the
/// result is deliberately discarded instead of unwrapped.
pub fn install_default_crypto_provider() {
    INSTALL_CRYPTO_PROVIDER.call_once(|| {
        let _ = rustls_rustcrypto::provider().install_default();
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_default_crypto_provider_is_idempotent() {
        // Calling this multiple times (including concurrently with other
        // tests in this process) must never panic.
        install_default_crypto_provider();
        install_default_crypto_provider();
        install_default_crypto_provider();
    }
}

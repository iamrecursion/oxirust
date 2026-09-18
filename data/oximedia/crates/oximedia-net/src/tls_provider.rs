//! Process-wide TLS crypto provider bootstrap.
//!
//! The workspace configures `reqwest`/`rustls` with the `rustls-no-provider`
//! feature so that the default build stays 100 % Pure Rust (no `aws-lc-sys` /
//! `ring` C or assembly code compiled in). That means `rustls` no longer ships
//! a compiled-in default [`rustls::crypto::CryptoProvider`] — one **must** be
//! installed at process start, otherwise the first attempt to build a TLS
//! client (e.g. `reqwest::Client::builder().build()`) panics with
//! `no process-level CryptoProvider available`.
//!
//! Call [`install_default_crypto_provider`] once, as early as possible, from
//! every binary/library entry point that may open a TLS connection (directly
//! or transitively through `reqwest`, `tokio-rustls`, `hyper-rustls`, ...).
//! The call is backed by [`std::sync::Once`] so it is safe (and cheap) to
//! invoke from multiple independent init paths — only the first call does
//! any work.

use std::sync::Once;

static INSTALL_CRYPTO_PROVIDER: Once = Once::new();

/// Installs the Pure-Rust [`rustls-rustcrypto`](https://docs.rs/rustls-rustcrypto)
/// crypto provider as the process-wide default `rustls` [`CryptoProvider`](rustls::crypto::CryptoProvider).
///
/// This is required because the workspace builds `reqwest`/`rustls` with no
/// compiled-in default provider (`rustls-no-provider`), to keep the default
/// build free of `aws-lc-sys`/`ring` C and assembly code.
///
/// Idempotent and safe to call from multiple entry points (CLI `main`,
/// server startup, library constructors that eagerly build a TLS-capable
/// HTTP client, ...) — the installation itself only happens once per
/// process, guarded by a [`std::sync::Once`]. Subsequent calls (from this
/// process or, via `rustls`'s own guard, from any other place that installs
/// a provider first) are silently ignored: `rustls` reports an
/// already-installed provider as an `Err`, which is not a bug and is
/// deliberately not surfaced to the caller.
pub fn install_default_crypto_provider() {
    INSTALL_CRYPTO_PROVIDER.call_once(|| {
        // `install_default` returns `Err(Arc<CryptoProvider>)` if a provider
        // was already installed (e.g. by another crate, or a previous call
        // racing with this one before the `Once` guard closed). That is not
        // an error condition for us — any installed provider is sufficient —
        // so we deliberately discard the `Result` instead of unwrapping it.
        let _ = rustls_rustcrypto::provider().install_default();
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_default_crypto_provider_is_idempotent() {
        // Calling this multiple times (including concurrently with other
        // tests in this process that may install a provider of their own)
        // must never panic.
        install_default_crypto_provider();
        install_default_crypto_provider();
        install_default_crypto_provider();

        // A process-level default provider must now be available.
        assert!(rustls::crypto::CryptoProvider::get_default().is_some());
    }
}

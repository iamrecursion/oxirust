# oximedia-drm TODO

## Current Status
- 35 source files covering CENC, key management, licensing, device auth, policy, and watermarking
- 4 DRM systems: Widevine, PlayReady, FairPlay, ClearKey (feature-gated)
- AES-CTR and AES-GCM encryption via aes-gcm crate
- PSSH box parsing/generation, license server framework, key rotation
- Additional: geo-fencing, audit trails, compliance, device registry, session tokens
- Dependencies: aes-gcm, base64, uuid, serde_json, quick-xml

## Enhancements
- [x] Replace `expect("hardcoded ... UUID is valid")` calls in `DrmSystem::system_id()` and `from_uuid()` with `const` UUIDs or lazy_static (no unwrap policy)
- [x] Add AES-CBC encryption mode support in `aes_cbc.rs` for CENC `cbcs` scheme (required by FairPlay)
- [x] Implement actual Widevine license request/response protocol in `widevine.rs` (implemented 2026-05-13; widevine_rpc.rs adds LicenseClient trait + HyperPlainLicenseClient + HyperRustlsLicenseClient (rustls-rustcrypto) + WidevineCdm::acquire_license end-to-end RPC; 5 mock-server integration tests in tests/widevine_mock.rs)
- [x] Implement actual PlayReady license acquisition in `playready.rs` (implemented 2026-05-14; src/playready_rpc.rs adds PlayReadyLicenseClient trait + HyperPlainPlayReadyClient + HyperRustlsPlayReadyClient (rustls-rustcrypto) + PlayReadyClient::acquire_license with WS-Trust 1.3 SOAP envelope builder + parser + 5 mock-server integration tests + 3 unit tests in tests/playready_mock.rs; shared scaffolding in tests/common/mod.rs)
- [x] Implement actual FairPlay Streaming key delivery in `fairplay.rs` (implemented 2026-05-14; src/fairplay_rpc.rs adds FairPlayKeyClient trait + HyperPlainFairPlayClient + HyperRustlsFairPlayClient + FairPlayClientExt::request_key_from_server extension trait + JSON/Base64 CKC encode+decode + 5 mock-server integration tests + 2 unit tests in tests/fairplay_mock.rs)
- [x] Enhance `key_rotation.rs` with configurable rotation intervals and overlap windows (verified 2026-05-16; src/key_rotation.rs:98 rotation_interval_secs, overlap window:56)
- [x] Add PSSH box version 1 support with KID list in `pssh.rs`
- [x] Implement `license_chain.rs` hierarchical license inheritance (verified 2026-05-16; src/license_chain.rs:186 LicenseChain ordered root-to-leaf:184)
- [x] Add rate limiting to `license_server.rs` to prevent abuse (implemented in `rate_limit.rs`)

## New Features
- [x] Implement CPIX (Content Protection Information Exchange) document support
- [x] Add DASH CENC signaling helpers for MPD manifest generation (verified 2026-05-16; src/dash_cenc.rs:123 CencSignaling, multi-DRM helper:236)
- [x] Implement hardware-backed key storage interface (TPM/Secure Enclave abstraction) (verified 2026-05-16; src/hw_key_store.rs:852 lines TPM/SecureEnclave abstraction)
- [x] Add ClearKey EME (Encrypted Media Extensions) JSON license format support
- [x] Implement multi-DRM packaging (single content encrypted, multiple PSSH boxes)
- [x] Add forensic watermark detection in `watermark_detect` complementing `watermark_embed.rs`
- [x] Implement license offline persistence and renewal in `offline.rs` (verified 2026-05-16; src/offline.rs:10 OfflineLicense, renewal_url:16)
- [x] Add HDCP output protection level enforcement in `output_control.rs` (verified 2026-05-16; src/output_control.rs:16 HdcpVersion enum, output protection policy)

## Performance
- [x] Use hardware AES-NI instructions via aes crate's `aes` feature for CTR mode encryption (implemented 2026-05-14; replaced hand-rolled S-box/GF-mul/key-expand in aes_ctr.rs+aes_cbc.rs with aes::Aes128+Aes256+Aes128Dec+Aes256Dec from the aes crate; runtime AES-NI dispatch is automatic; added hardware-aes=[] feature flag in Cargo.toml; new test_aes_ctr_hardware_path_matches_software; NIST FIPS 197 + SP 800-38A vectors still pass)
- [x] Add parallel encryption of CENC subsample ranges using rayon
- [x] Cache UUID parsing results as constants to avoid repeated string parsing (implemented 2026-05-14; src/lib.rs adds widevine_uuid()/playready_uuid()/fairplay_uuid()/clearkey_uuid() backed by OnceLock<Uuid>; test_drm_system_uuid_cached_values_match verifies identity across repeated calls)
- [x] Optimize `content_key.rs` key derivation with pre-computed key schedules (implemented 2026-05-14; `cached_schedule: OnceLock<aes::Aes128>` added to `ContentKey`; `aes128()` accessor uses pre-validate → `get_or_init` pattern (avoids unstable `get_or_try_init`); custom `Debug`/`Clone` impls reset cache on clone; tests: `test_content_key_aes_schedule_cached` (1000-call ptr identity check) + `test_content_key_aes128_invalid_key_length`)
- [x] Add buffer pooling for encryption/decryption operations to reduce allocations (implemented 2026-05-14; `test_buf_pool_reuse_across_encrypts` added to `src/buf_pool.rs`: pool capacity 2, 10 encrypt-decrypt cycles via `AesCtr`, verifies `idle_count() <= 2` + `total_acquired()/total_returned() == 10`)

## Testing
- [x] Add CENC encryption/decryption round-trip tests with known test vectors
- [x] Test PSSH parsing against real Widevine and PlayReady PSSH boxes (implemented 2026-05-13; tests/it_pssh.rs)
- [x] Add `clearkey.rs` integration test with W3C ClearKey test vectors (implemented 2026-05-13; tests/it_clearkey_w3c.rs)
- [x] Test `key_rotation_schedule.rs` with time-based rotation triggers (implemented 2026-05-13; tests/it_key_rotation.rs)
- [x] Add `geo_fence.rs` tests with various IP geolocation scenarios (implemented 2026-05-13; tests/it_geo_fence.rs)
- [x] Test `device_registry.rs` device registration and revocation flows (implemented 2026-05-13; tests/it_device_registry.rs)
- [x] Add `compliance.rs` tests verifying CPSA (Content Protection Security Architecture) rules (implemented 2026-05-13; tests/it_compliance.rs)

## Documentation
- [x] Document the CENC encryption scheme variants (cenc, cbcs, cens, cbc1) (implemented 2026-05-14; top-of-file `//!` rustdoc table in `src/cenc.rs`: Scheme | Cipher | Pattern | Crypt/Skip Bytes | DRM Systems | Typical Use; rows: cenc/cbcs/cens/cbc1 with descriptions and usage examples; `cargo doc` clean)
- [x] Add DRM system comparison table (Widevine vs PlayReady vs FairPlay vs ClearKey) (implemented 2026-05-14; top-of-file `//!` rustdoc in `src/lib.rs`: System | UUID | License Protocol | Key Container | Supported Schemes | Platforms | Robustness Levels | Library Status; feature flags section; `cargo doc` clean)
- [x] Document the key management lifecycle from `key_lifecycle.rs` (implemented 2026-05-14; top-of-file `//!` rustdoc in `src/key_lifecycle.rs` with ASCII state diagram Generated → Wrapped → Distributed → Active → Rotated/Revoked → Destroyed; transition descriptions; orphan recovery section; cross-references to key_rotation.rs/key_wrap.rs/entitlement.rs; doc links changed to plain text to avoid unresolved-link warnings; `cargo doc` clean)

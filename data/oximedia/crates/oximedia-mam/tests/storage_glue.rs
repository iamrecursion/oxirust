//! Integration tests for `MamStorage` — URI dispatch and round-trip put/get/list/delete.

use oximedia_mam::storage::{MamStorage, MamStorageError, MamStorageScheme};
use std::env;

// ── URI scheme parsing ────────────────────────────────────────────────────────

#[test]
fn test_parse_bare_path_as_local() {
    let scheme = MamStorage::parse_scheme("/var/media").expect("should parse");
    assert!(matches!(scheme, MamStorageScheme::Local(_)));
}

#[test]
fn test_parse_relative_path_as_local() {
    let scheme = MamStorage::parse_scheme("relative/path").expect("should parse");
    assert!(matches!(scheme, MamStorageScheme::Local(_)));
}

#[test]
fn test_parse_file_uri() {
    let scheme = MamStorage::parse_scheme("file:///var/media/assets").expect("should parse");
    match scheme {
        MamStorageScheme::Local(p) => {
            assert_eq!(p.to_string_lossy(), "/var/media/assets");
        }
        _ => panic!("expected Local"),
    }
}

#[test]
fn test_parse_s3_uri() {
    let scheme = MamStorage::parse_scheme("s3://my-bucket/media/raw").expect("should parse");
    match scheme {
        MamStorageScheme::S3 { bucket, prefix } => {
            assert_eq!(bucket, "my-bucket");
            assert_eq!(prefix, "media/raw");
        }
        _ => panic!("expected S3"),
    }
}

#[test]
fn test_parse_s3_uri_no_prefix() {
    let scheme = MamStorage::parse_scheme("s3://my-bucket").expect("should parse");
    match scheme {
        MamStorageScheme::S3 { bucket, prefix } => {
            assert_eq!(bucket, "my-bucket");
            assert_eq!(prefix, "");
        }
        _ => panic!("expected S3"),
    }
}

#[test]
fn test_parse_gcs_uri() {
    let scheme = MamStorage::parse_scheme("gs://gs-bucket/footage").expect("should parse");
    match scheme {
        MamStorageScheme::Gcs { bucket, prefix } => {
            assert_eq!(bucket, "gs-bucket");
            assert_eq!(prefix, "footage");
        }
        _ => panic!("expected Gcs"),
    }
}

#[test]
fn test_parse_azure_uri() {
    let uri = "https://myaccount.blob.core.windows.net/mycontainer/prefix";
    let scheme = MamStorage::parse_scheme(uri).expect("should parse Azure");
    match scheme {
        MamStorageScheme::Azure {
            account,
            container,
            prefix,
        } => {
            assert_eq!(account, "myaccount");
            assert_eq!(container, "mycontainer");
            assert_eq!(prefix, "prefix");
        }
        _ => panic!("expected Azure, got {scheme:?}"),
    }
}

#[test]
fn test_parse_azure_uri_no_prefix() {
    let uri = "https://myaccount.blob.core.windows.net/mycontainer";
    let scheme = MamStorage::parse_scheme(uri).expect("should parse Azure");
    match scheme {
        MamStorageScheme::Azure {
            account,
            container,
            prefix,
        } => {
            assert_eq!(account, "myaccount");
            assert_eq!(container, "mycontainer");
            assert_eq!(prefix, "");
        }
        _ => panic!("expected Azure"),
    }
}

#[test]
fn test_parse_unsupported_scheme_returns_error() {
    let result = MamStorage::parse_scheme("ftp://example.com/data");
    assert!(
        matches!(result, Err(MamStorageError::UnsupportedScheme(_))),
        "expected UnsupportedScheme, got {result:?}"
    );
}

#[test]
fn test_parse_https_non_azure_returns_error() {
    let result = MamStorage::parse_scheme("https://example.com/data");
    assert!(
        matches!(result, Err(MamStorageError::UnsupportedScheme(_))),
        "expected UnsupportedScheme for non-Azure https"
    );
}

// ── Round-trip put / get / list / delete ─────────────────────────────────────

#[tokio::test]
async fn test_local_put_get_round_trip() {
    let root = env::temp_dir().join(format!("mam_storage_test_{}", uuid_str()));
    let storage = MamStorage::local(&root)
        .await
        .expect("create local storage");

    let key = "test/asset.mp4";
    let data = b"hello world media data";

    storage.put(key, data).await.expect("put");
    let retrieved = storage.get(key).await.expect("get");
    assert_eq!(retrieved, data);
}

#[tokio::test]
async fn test_local_exists() {
    let root = env::temp_dir().join(format!("mam_exists_test_{}", uuid_str()));
    let storage = MamStorage::local(&root).await.expect("create storage");

    let key = "exists_check/file.bin";
    assert!(!storage.exists(key).await.expect("exists check false"));
    storage.put(key, b"data").await.expect("put");
    assert!(storage.exists(key).await.expect("exists check true"));
}

#[tokio::test]
async fn test_local_delete() {
    let root = env::temp_dir().join(format!("mam_delete_test_{}", uuid_str()));
    let storage = MamStorage::local(&root).await.expect("create storage");

    let key = "delete_me/file.bin";
    storage.put(key, b"temporary").await.expect("put");
    assert!(storage.exists(key).await.expect("exists before delete"));
    storage.delete(key).await.expect("delete");
    // After deletion the key should be gone — LocalStorage::delete is idempotent
    // so we just check the data is not retrievable.
    let result = storage.get(key).await;
    assert!(result.is_err(), "get after delete should fail");
}

#[tokio::test]
async fn test_local_list() {
    let root = env::temp_dir().join(format!("mam_list_test_{}", uuid_str()));
    let storage = MamStorage::local(&root).await.expect("create storage");

    storage.put("media/a.mp4", b"a").await.expect("put a");
    storage.put("media/b.mp4", b"b").await.expect("put b");

    let keys = storage.list("media").await.expect("list");
    assert_eq!(keys.len(), 2, "expected 2 keys, got {keys:?}");
}

// ── From-URI construction (stub paths) ───────────────────────────────────────

#[tokio::test]
async fn test_from_uri_local() {
    let root = env::temp_dir().join(format!("mam_fromuri_{}", uuid_str()));
    tokio::fs::create_dir_all(&root).await.expect("mkdir");
    let uri = format!("file://{}", root.display());
    let storage = MamStorage::from_uri(&uri).await.expect("from_uri local");
    storage.put("x", b"hello").await.expect("put via from_uri");
}

// When the `s3` feature is enabled the AWS SDK initialises its TLS stack during
// `from_uri`; on macOS the keychain may be unavailable in CI / sandbox / --all-features
// runs, causing a panic from aws-smithy-http-client ("could not load platform certs").
// The live-credential path is covered by the `#[ignore]` test in storage.rs.
// We therefore only run this test in the default configuration (no `s3` feature),
// where it exercises the pure-Rust local-shadow round-trip.
#[cfg(not(feature = "s3"))]
#[tokio::test]
async fn test_from_uri_s3_local_shadow_or_real() {
    // Without the `s3` feature: the handle runs in pure-Rust local-shadow
    // mode and a put/get round-trip succeeds offline.
    let storage = MamStorage::from_uri("s3://test-bucket/pfx")
        .await
        .expect("s3 from_uri yields a handle");

    assert!(
        !storage.has_cloud_backend(),
        "expected local-shadow mode without the s3 feature"
    );

    // Local-shadow degraded mode — full round-trip works offline.
    storage
        .put("k", b"v")
        .await
        .expect("put on s3 local shadow");
    let got = storage.get("k").await.expect("get on s3 local shadow");
    assert_eq!(got, b"v");
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn uuid_str() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    format!("{nanos:x}")
}

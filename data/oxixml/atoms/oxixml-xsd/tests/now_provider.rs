//! Regression tests for the pluggable current-time clock.
//!
//! `DateTime::now()` / `Date::now()` / `Time::now()` used to call
//! [`std::time::SystemTime::now`] unconditionally. That function is not implemented on
//! `wasm32-unknown-unknown`, where it aborts the module with an `unreachable` trap — and since
//! the XPath/XQuery evaluation context timestamps every evaluation, the whole query stack
//! became unusable on that target. The clock is now overridable with
//! [`oxixml_xsd::set_now_provider`], and the fallible `checked_now()` constructors report the
//! absence of a clock instead of trapping.
//!
//! This lives in its own integration test because the provider is process-global and can only
//! be registered once.

use oxixml_xsd::{Date, DateTime, Time, set_now_provider};
use std::time::Duration;

/// 2000-01-01T00:00:00Z, as milliseconds since the Unix epoch.
const MILLENNIUM_MILLIS: u64 = 946_684_800_000;

#[test]
fn a_registered_provider_replaces_the_platform_clock() {
    // Before registration, the platform clock answers on every target that has one.
    assert!(DateTime::checked_now().is_some());

    set_now_provider(|| Duration::from_millis(MILLENNIUM_MILLIS))
        .expect("no provider has been registered yet");

    assert_eq!(
        DateTime::now().to_string(),
        "2000-01-01T00:00:00Z",
        "the registered provider must back fn:current-dateTime"
    );
    assert_eq!(Date::now().to_string(), "2000-01-01Z");
    assert_eq!(Time::now().to_string(), "00:00:00Z");
    assert_eq!(
        DateTime::checked_now().map(|now| now.to_string()),
        Some("2000-01-01T00:00:00Z".to_owned())
    );

    // The clock can only be chosen once, so a single evaluation never sees two of them.
    assert!(set_now_provider(|| Duration::from_millis(0)).is_err());
}

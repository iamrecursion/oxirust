//! [`FnContext`] — the slice of the XPath static and dynamic context that the function library
//! itself consults.
//!
//! A function library is not an evaluator, so this is deliberately *not* the whole context of
//! [XPath 3.1 §2.1](https://www.w3.org/TR/xpath-31/#context): there is no context item, no
//! variable bindings and no function signatures here. What is here is exactly what
//! [F&O 3.1 §2](https://www.w3.org/TR/xpath-functions-31/#context) lists as the context
//! components the functions depend on:
//!
//! * the **default collation** and the set of known collations;
//! * the **static base URI**, used by `fn:resolve-uri` and `fn:parse-xml`;
//! * the **implicit timezone**, used by every date/time comparison and by `fn:adjust-*`;
//! * the **current dateTime**, which is *stable*: `fn:current-dateTime` returns the same
//!   instant every time it is called within one execution, which is why it is captured once
//!   when the context is built;
//! * the **decimal formats** consulted by `fn:format-number`;
//! * the destination of `fn:trace`.
//!
//! One further component is the processor's own, rather than the query's: the **version of XML**
//! the processor implements. [XDM 3.1
//! §2](https://www.w3.org/TR/xpath-datamodel-31/#terminology) defines a *character* as an
//! instance of the `Char` production of whichever XML version the implementation supports, so
//! `fn:codepoints-to-string(8)` is a `FOCH0001` on an XML 1.0 processor and a backspace on an
//! XML 1.1 one. The default is XML 1.0.
//!
//! One slot here belongs to no function in this crate: [`FnContext::host_state`] carries whatever
//! the *host language* needs to travel with a context it hands out. XSLT uses it for the
//! grouping focus a named function reference has to capture; nothing in this crate reads it.

use crate::collation::{Collation, Collations};
use crate::error::{FnError, FnErrorCode, FnResult};
use crate::numeric::DecimalFormat;
use oxixml_infoset::ExpandedName;
use oxixml_unicode::xml::XmlVersion;
use oxixml_xsd::{DateTime, TimezoneOffset};
use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

/// Where `fn:trace` sends its diagnostics.
///
/// The default, [`StderrTrace`], writes one line per item to standard error, which is what most
/// processors do. Tests usually want [`CollectingTrace`] instead.
pub trait TraceSink: fmt::Debug + Send + Sync {
    /// Records one traced item. `label` is `fn:trace`'s second argument (empty when the
    /// one-argument form was used), `value` the item's string representation, `index` its
    /// one-based position in the traced sequence and `total` the sequence's length.
    fn trace(&self, label: &str, value: &str, index: usize, total: usize);
}

/// A [`TraceSink`] that writes `label` and the item to standard error.
#[derive(Debug, Clone, Copy, Default)]
pub struct StderrTrace;

impl TraceSink for StderrTrace {
    fn trace(&self, label: &str, value: &str, index: usize, total: usize) {
        if total == 1 {
            eprintln!("{label}{value}");
        } else {
            eprintln!("{label}[{index}]: {value}");
        }
    }
}

/// A [`TraceSink`] that accumulates the traced lines in memory, for tests.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use oxixml_fn::{diagnostics::trace, CollectingTrace, FnContext};
/// use oxixml_infoset::{AtomicValue, Sequence};
///
/// let sink = Arc::new(CollectingTrace::new());
/// let context = FnContext::new().with_trace_sink(sink.clone());
/// let value = Sequence::from(AtomicValue::integer(42_i64));
/// assert_eq!(trace(&context, &value, "answer: "), value);
/// assert_eq!(sink.messages(), vec!["answer: 42".to_owned()]);
/// ```
#[derive(Debug, Default)]
pub struct CollectingTrace {
    messages: Mutex<Vec<String>>,
}

impl CollectingTrace {
    /// An empty sink.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The lines recorded so far, in order.
    #[must_use]
    pub fn messages(&self) -> Vec<String> {
        self.messages
            .lock()
            .map(|messages| messages.clone())
            .unwrap_or_default()
    }

    /// Discards every recorded line.
    pub fn clear(&self) {
        if let Ok(mut messages) = self.messages.lock() {
            messages.clear();
        }
    }
}

impl TraceSink for CollectingTrace {
    fn trace(&self, label: &str, value: &str, index: usize, total: usize) {
        let line = if total == 1 {
            format!("{label}{value}")
        } else {
            format!("{label}[{index}]: {value}")
        };
        if let Ok(mut messages) = self.messages.lock() {
            messages.push(line);
        }
    }
}

/// The host language's own state, carried by an [`FnContext`] but opaque to it.
///
/// The wrapper exists so that [`FnContext`] can stay `Debug`: a `dyn Any` is not, and printing a
/// context should not depend on what a host happened to put in it.
#[derive(Clone)]
struct HostState(Arc<dyn Any + Send + Sync>);

impl fmt::Debug for HostState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("HostState(..)")
    }
}

/// The context the functions in this crate evaluate against.
///
/// # Examples
///
/// ```
/// use oxixml_fn::{FnContext, CASELESS_COLLATION};
/// use oxixml_xsd::{DateTime, TimezoneOffset};
/// use std::str::FromStr;
///
/// // A deterministic context, which is what a test or a reproducible pipeline wants.
/// let context = FnContext::new()
///     .with_current_date_time(DateTime::from_str("2004-05-12T18:17:15.125Z")?)
///     .with_implicit_timezone(TimezoneOffset::new(-300)?)
///     .with_static_base_uri("http://example.com/data/")
///     .with_default_collation(CASELESS_COLLATION)?;
///
/// assert_eq!(context.current_date_time().to_string(), "2004-05-12T18:17:15.125Z");
/// assert_eq!(context.implicit_timezone().to_string(), "-05:00");
/// assert_eq!(context.default_collation().uri(), CASELESS_COLLATION);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct FnContext {
    collations: Collations,
    default_collation: Arc<dyn Collation>,
    static_base_uri: Option<String>,
    implicit_timezone: TimezoneOffset,
    current_date_time: DateTime,
    default_decimal_format: DecimalFormat,
    decimal_formats: HashMap<ExpandedName, DecimalFormat>,
    trace: Arc<dyn TraceSink>,
    xml_version: XmlVersion,
    host_state: Option<HostState>,
}

impl Default for FnContext {
    fn default() -> Self {
        Self::new()
    }
}

impl FnContext {
    /// A context with the specification defaults: the code point collation, no static base URI,
    /// the UTC implicit timezone, the current instant, the default decimal format, and
    /// `fn:trace` writing to standard error.
    ///
    /// The implicit timezone defaults to UTC rather than to the host's local zone: a Pure-Rust
    /// crate cannot read the platform's zone database, and F&O only requires the value to be
    /// implementation-defined. Call [`FnContext::with_implicit_timezone`] to set it.
    #[must_use]
    pub fn new() -> Self {
        let collations = Collations::new();
        let default_collation = collations.codepoint();
        Self {
            collations,
            default_collation,
            static_base_uri: None,
            implicit_timezone: TimezoneOffset::UTC,
            current_date_time: DateTime::now(),
            default_decimal_format: DecimalFormat::default(),
            decimal_formats: HashMap::new(),
            trace: Arc::new(StderrTrace),
            xml_version: XmlVersion::V10,
            host_state: None,
        }
    }

    /// Declares which version of XML the processor implements.
    ///
    /// This decides what counts as a *character*, which is what
    /// [`fn:codepoints-to-string`](crate::strings::codepoints_to_string_in) admits: XML 1.0
    /// rejects every C0 control other than tab, line feed and carriage return, while XML 1.1
    /// admits all of them but `NUL`. The two W3C suites pin the pair over the same query — QT3
    /// `K2-Serialization-7` requires `codepoints-to-string(1 to 31)` to succeed under 1.1 and
    /// `-7a` requires `FOCH0001` under 1.0 — so a processor answers for exactly one version and
    /// says which here. The default is [`XmlVersion::V10`].
    #[must_use]
    pub const fn with_xml_version(mut self, version: XmlVersion) -> Self {
        self.xml_version = version;
        self
    }

    /// Sets the static base URI, which `fn:resolve-uri` and `fn:parse-xml` resolve against.
    #[must_use]
    pub fn with_static_base_uri(mut self, uri: impl Into<String>) -> Self {
        self.static_base_uri = Some(uri.into());
        self
    }

    /// Sets the implicit timezone.
    #[must_use]
    pub const fn with_implicit_timezone(mut self, timezone: TimezoneOffset) -> Self {
        self.implicit_timezone = timezone;
        self
    }

    /// Pins the value `fn:current-dateTime` reports.
    #[must_use]
    pub const fn with_current_date_time(mut self, current: DateTime) -> Self {
        self.current_date_time = current;
        self
    }

    /// Selects the default collation by URI.
    ///
    /// # Errors
    ///
    /// [`FnErrorCode::Foch0002`] when no collation is registered under `uri`.
    pub fn with_default_collation(mut self, uri: &str) -> FnResult<Self> {
        self.default_collation = self.collations.resolve(uri)?;
        Ok(self)
    }

    /// Adds a collation to the context's registry.
    #[must_use]
    pub fn with_collation(mut self, collation: Arc<dyn Collation>) -> Self {
        self.collations.register(collation);
        self
    }

    /// Replaces the unnamed decimal format `fn:format-number` uses by default.
    #[must_use]
    pub fn with_default_decimal_format(mut self, format: DecimalFormat) -> Self {
        self.default_decimal_format = format;
        self
    }

    /// Adds a named decimal format, as `xsl:decimal-format` does.
    #[must_use]
    pub fn with_decimal_format(mut self, name: ExpandedName, format: DecimalFormat) -> Self {
        self.decimal_formats.insert(name, format);
        self
    }

    /// Redirects `fn:trace`.
    #[must_use]
    pub fn with_trace_sink(mut self, sink: Arc<dyn TraceSink>) -> Self {
        self.trace = sink;
        self
    }

    /// Attaches the host language's own state to the context.
    ///
    /// Nothing in this crate reads the value: the slot exists because a host language evaluates
    /// its expressions against a context it builds here, and some of its state has to travel with
    /// that context even though no F&O function names it. XSLT stores the grouping focus
    /// (`current-group()`, `current-grouping-key()`) here, so that a named function reference
    /// captured inside `xsl:for-each-group` still answers from the group it was written in rather
    /// than from whatever group is current when it is finally called.
    ///
    /// The state is an [`Arc`] because a context is cloned freely — every capture shares one
    /// value rather than copying it — and `Send + Sync` because a context may cross threads.
    /// Read it back with [`FnContext::host_state`], or with
    /// [`FnContext::host_state_as`] when the concrete type is known.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxixml_fn::FnContext;
    /// use std::sync::Arc;
    ///
    /// #[derive(Debug, PartialEq)]
    /// struct GroupingFocus {
    ///     key: String,
    /// }
    ///
    /// let focus = GroupingFocus { key: "a".to_owned() };
    /// let context = FnContext::new().with_host_state(Arc::new(focus));
    ///
    /// // A clone of the context shares the same state, which is what a capture relies on.
    /// let captured = context.clone();
    /// assert_eq!(
    ///     captured.host_state_as::<GroupingFocus>().map(|focus| focus.key.as_str()),
    ///     Some("a")
    /// );
    /// assert!(captured.host_state_as::<String>().is_none());
    /// ```
    #[must_use]
    pub fn with_host_state(mut self, state: Arc<dyn Any + Send + Sync>) -> Self {
        self.host_state = Some(HostState(state));
        self
    }

    /// The host language's state, as set by [`FnContext::with_host_state`], or `None` when the
    /// host set none.
    ///
    /// The [`Arc`] is returned rather than a bare reference so that a caller can keep the state
    /// alive past the context — clone it, or take ownership of the concrete value with
    /// [`Arc::downcast`].
    #[must_use]
    pub fn host_state(&self) -> Option<&Arc<dyn Any + Send + Sync>> {
        self.host_state.as_ref().map(|state| &state.0)
    }

    /// The host language's state downcast to `T`, or `None` when there is none or it is of
    /// another type.
    #[must_use]
    pub fn host_state_as<T: Any>(&self) -> Option<&T> {
        self.host_state()?.downcast_ref::<T>()
    }

    /// The collation registry.
    #[must_use]
    pub const fn collations(&self) -> &Collations {
        &self.collations
    }

    /// The default collation, used whenever a `$collation` argument is absent.
    #[must_use]
    pub fn default_collation(&self) -> &Arc<dyn Collation> {
        &self.default_collation
    }

    /// Resolves a `$collation` argument: `None` selects the default collation.
    ///
    /// F&O 3.1 §5.3.1: "if the specified collation is a relative URI reference, it is resolved
    /// against the static base URI", so `collation/codepoint` under a static base URI of
    /// `http://www.w3.org/2005/xpath-functions/` is the code point collation. The literal URI is
    /// tried first, so a registered collation whose name happens not to be absolute still wins.
    ///
    /// # Errors
    ///
    /// [`FnErrorCode::Foch0002`] when `uri` names no known collation.
    pub fn collation(&self, uri: Option<&str>) -> FnResult<Arc<dyn Collation>> {
        match uri {
            None => Ok(Arc::clone(&self.default_collation)),
            Some(uri) => match self.collations.resolve(uri) {
                Ok(collation) => Ok(collation),
                Err(error) => match self.resolve_collation_uri(uri) {
                    Some(absolute) => self.collations.resolve(&absolute).map_err(|_| error),
                    None => Err(error),
                },
            },
        }
    }

    /// Resolves a relative collation URI reference against the static base URI.
    fn resolve_collation_uri(&self, uri: &str) -> Option<String> {
        let base = oxixml_iri::Iri::parse(self.static_base_uri.as_deref()?).ok()?;
        let reference = oxixml_iri::IriRef::parse(uri).ok()?;
        Some(base.resolve(&reference).ok()?.into_inner())
    }

    /// The static base URI, if the context has one.
    #[must_use]
    pub fn static_base_uri(&self) -> Option<&str> {
        self.static_base_uri.as_deref()
    }

    /// The static base URI, or `FONS0005` when the context has none.
    ///
    /// # Errors
    ///
    /// [`FnErrorCode::Fons0005`] when no static base URI is defined.
    pub fn require_static_base_uri(&self) -> FnResult<&str> {
        self.static_base_uri.as_deref().ok_or_else(|| {
            FnError::new(
                FnErrorCode::Fons0005,
                "the static base URI is not defined in the context",
            )
        })
    }

    /// The implicit timezone.
    #[must_use]
    pub const fn implicit_timezone(&self) -> TimezoneOffset {
        self.implicit_timezone
    }

    /// The version of XML the processor implements; see [`FnContext::with_xml_version`].
    #[must_use]
    pub const fn xml_version(&self) -> XmlVersion {
        self.xml_version
    }

    /// The stable current instant, what `fn:current-dateTime` returns.
    #[must_use]
    pub const fn current_date_time(&self) -> DateTime {
        self.current_date_time
    }

    /// The unnamed decimal format.
    #[must_use]
    pub const fn default_decimal_format(&self) -> &DecimalFormat {
        &self.default_decimal_format
    }

    /// Looks a decimal format up: `None` selects the unnamed one.
    ///
    /// # Errors
    ///
    /// [`FnErrorCode::Fodf1280`] when `name` is not a declared decimal format.
    pub fn decimal_format(&self, name: Option<&ExpandedName>) -> FnResult<&DecimalFormat> {
        match name {
            None => Ok(&self.default_decimal_format),
            Some(name) => self.decimal_formats.get(name).ok_or_else(|| {
                FnError::new(
                    FnErrorCode::Fodf1280,
                    format!("`{name}` is not a declared decimal format"),
                )
            }),
        }
    }

    /// The destination of `fn:trace`.
    #[must_use]
    pub fn trace_sink(&self) -> &Arc<dyn TraceSink> {
        &self.trace
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collation::CASELESS_COLLATION;
    use std::str::FromStr;

    #[test]
    fn defaults_match_the_specification() {
        let context = FnContext::new();
        assert_eq!(
            context.default_collation().uri(),
            crate::collation::CODEPOINT_COLLATION
        );
        assert_eq!(context.implicit_timezone(), TimezoneOffset::UTC);
        assert_eq!(context.static_base_uri(), None);
        assert!(context.require_static_base_uri().is_err());
    }

    #[test]
    fn a_relative_collation_uri_resolves_against_the_static_base_uri() -> FnResult<()> {
        let context =
            FnContext::new().with_static_base_uri("http://www.w3.org/2005/xpath-functions/");
        assert_eq!(
            context.collation(Some("collation/codepoint"))?.uri(),
            crate::collation::CODEPOINT_COLLATION
        );
        // Without a base URI there is nothing to resolve against, and the error is the original
        // `FOCH0002` for the URI as written.
        let error = FnContext::new()
            .collation(Some("collation/codepoint"))
            .expect_err("no static base URI");
        assert!(error.is(FnErrorCode::Foch0002));
        assert!(
            context.collation(Some("collation/nonesuch")).is_err(),
            "resolution does not invent a collation"
        );
        Ok(())
    }

    #[test]
    fn the_current_instant_is_stable() -> Result<(), Box<dyn std::error::Error>> {
        let pinned = DateTime::from_str("2004-05-12T18:17:15.125Z")?;
        let context = FnContext::new().with_current_date_time(pinned);
        assert_eq!(context.current_date_time(), pinned);
        assert_eq!(context.current_date_time(), context.current_date_time());
        Ok(())
    }

    #[test]
    fn collations_are_resolved_through_the_context() -> FnResult<()> {
        let context = FnContext::new().with_default_collation(CASELESS_COLLATION)?;
        assert_eq!(context.collation(None)?.uri(), CASELESS_COLLATION);
        assert_eq!(
            context
                .collation(Some(crate::collation::CODEPOINT_COLLATION))?
                .uri(),
            crate::collation::CODEPOINT_COLLATION
        );
        assert!(context.collation(Some("urn:example:nope")).is_err());
        Ok(())
    }

    #[test]
    fn host_state_survives_a_clone_and_downcasts_to_its_own_type() {
        #[derive(Debug, PartialEq, Eq)]
        struct Grouping {
            key: String,
        }

        let context = FnContext::new();
        assert!(context.host_state().is_none());
        assert!(context.host_state_as::<Grouping>().is_none());

        let context = context.with_host_state(Arc::new(Grouping {
            key: "first".to_owned(),
        }));
        let captured = context.clone();
        // A later context replaces the slot; the earlier capture keeps what it captured, which is
        // the whole point of the `Arc`.
        let context = context.with_host_state(Arc::new(Grouping {
            key: "second".to_owned(),
        }));

        assert_eq!(
            captured.host_state_as::<Grouping>().map(|g| g.key.as_str()),
            Some("first")
        );
        assert_eq!(
            context.host_state_as::<Grouping>().map(|g| g.key.as_str()),
            Some("second")
        );
        assert!(captured.host_state_as::<String>().is_none());
        assert!(captured.host_state().is_some());
        // `Debug` stays available whatever the host stored.
        assert!(format!("{captured:?}").contains("HostState(..)"));
    }

    #[test]
    fn unknown_decimal_formats_raise_fodf1280() {
        let context = FnContext::new();
        let name = ExpandedName::new(None::<&str>, "missing");
        assert!(context.decimal_format(None).is_ok());
        let error = context
            .decimal_format(Some(&name))
            .expect_err("not declared");
        assert!(error.is(FnErrorCode::Fodf1280));
    }
}

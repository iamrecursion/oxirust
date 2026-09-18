//! HTTP response compression as a `tower` layer, built on OxiARC.
//!
//! [`CompressionLayer`] replaces `tower_http::compression::CompressionLayer`,
//! whose `compression-br` / `compression-gzip` features pulled `brotli` and
//! `flate2` (via `async-compression`) into the build. Encoding here goes
//! through `oxiarc-http` (Pure Rust, COOLJAPAN OxiARC).
//!
//! Behaviour:
//!
//! - The coding is negotiated from the request's `Accept-Encoding` header(s)
//!   with `oxiarc_http::negotiate` (RFC 9110 §12.5.3: q-values, `identity`,
//!   `*`). Server preference on ties: gzip, then br, then deflate (each can
//!   be switched off). When the client accepts nothing the server can
//!   produce, the response is sent uncompressed rather than failed.
//! - Responses are left untouched when they already carry a
//!   `Content-Encoding`, are partial (`206` / `Content-Range`), have no body
//!   by definition (`1xx`, `204`, `304`, or the request was `HEAD`), carry
//!   `Cache-Control: no-transform`, declare a `Content-Length` below
//!   [`CompressionLayer::min_size`], or have a content type that is already
//!   compressed or streamed as events (`image/*` except SVG, `audio/*`,
//!   `video/*`, `text/event-stream`, common archive types).
//! - A compressed response gets `Content-Encoding`, loses `Content-Length`
//!   and `Accept-Ranges`, and gets `Vary: Accept-Encoding`; responses that
//!   were eligible but negotiated to identity still get the `Vary` header.
//! - Bodies are encoded **incrementally** with `oxiarc_http::Encoder`: every
//!   input frame is fed to the encoder and whatever compressed bytes it has
//!   produced are forwarded, and when the inner body is pending the encoder
//!   is sync-flushed so streaming responses are not held back. Memory is
//!   bounded by the encoder's own window, not by the body size.

mod body;

#[cfg(test)]
mod tests;

pub use body::CompressionBody;

use http::header::{self, HeaderMap, HeaderValue};
use http::{Method, Request, Response, StatusCode};
use oxiarc_http::ContentCoding;
use pin_project::pin_project;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tower::{Layer, Service};

/// Settings shared by every clone of the layer and its services.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CompressionConfig {
    gzip: bool,
    br: bool,
    deflate: bool,
    min_size: u64,
    level: u8,
    brotli_quality: u32,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            gzip: true,
            br: true,
            deflate: true,
            min_size: 32,
            level: 6,
            brotli_quality: 4,
        }
    }
}

impl CompressionConfig {
    /// The codings this server can produce, in tie-break preference order.
    fn available(&self) -> Vec<ContentCoding> {
        let mut available = Vec::with_capacity(3);
        if self.gzip {
            available.push(ContentCoding::Gzip);
        }
        if self.br {
            available.push(ContentCoding::Brotli);
        }
        if self.deflate {
            available.push(ContentCoding::Deflate);
        }
        available
    }
}

/// `tower::Layer` that compresses response bodies (gzip / br / deflate via
/// OxiARC). See the [module documentation](self) for the exact rules.
#[derive(Debug, Clone, Default)]
pub struct CompressionLayer {
    config: Arc<CompressionConfig>,
}

impl CompressionLayer {
    /// Layer with every coding enabled, a 32-byte minimum size, deflate/gzip
    /// level 6 and Brotli quality 4.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn update(mut self, change: impl FnOnce(&mut CompressionConfig)) -> Self {
        change(Arc::make_mut(&mut self.config));
        self
    }

    /// Enable or disable gzip.
    #[must_use]
    pub fn gzip(self, enabled: bool) -> Self {
        self.update(|c| c.gzip = enabled)
    }

    /// Enable or disable Brotli (`br`).
    #[must_use]
    pub fn br(self, enabled: bool) -> Self {
        self.update(|c| c.br = enabled)
    }

    /// Enable or disable `deflate`.
    #[must_use]
    pub fn deflate(self, enabled: bool) -> Self {
        self.update(|c| c.deflate = enabled)
    }

    /// Do not compress responses whose `Content-Length` is below `bytes`.
    /// (Bodies of unknown length are always eligible.)
    #[must_use]
    pub fn min_size(self, bytes: u64) -> Self {
        self.update(|c| c.min_size = bytes)
    }

    /// gzip / deflate compression level (0-9).
    #[must_use]
    pub fn level(self, level: u8) -> Self {
        self.update(|c| c.level = level.min(9))
    }

    /// Brotli quality (0-11).
    #[must_use]
    pub fn brotli_quality(self, quality: u32) -> Self {
        self.update(|c| c.brotli_quality = quality.min(11))
    }
}

impl<S> Layer<S> for CompressionLayer {
    type Service = Compression<S>;

    fn layer(&self, inner: S) -> Self::Service {
        Compression {
            inner,
            config: Arc::clone(&self.config),
        }
    }
}

/// Service produced by [`CompressionLayer`].
#[derive(Debug, Clone)]
pub struct Compression<S> {
    inner: S,
    config: Arc<CompressionConfig>,
}

impl<S, ReqBody, ResBody> Service<Request<ReqBody>> for Compression<S>
where
    S: Service<Request<ReqBody>, Response = Response<ResBody>>,
    ResBody: http_body::Body,
{
    type Response = Response<CompressionBody<ResBody>>;
    type Error = S::Error;
    type Future = ResponseFuture<S::Future>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: Request<ReqBody>) -> Self::Future {
        let accept_encoding = joined_accept_encoding(request.headers());
        let is_head = request.method() == Method::HEAD;
        ResponseFuture {
            inner: self.inner.call(request),
            accept_encoding,
            is_head,
            config: Arc::clone(&self.config),
        }
    }
}

/// Future returned by [`Compression`].
#[pin_project]
pub struct ResponseFuture<F> {
    #[pin]
    inner: F,
    /// `None` when the request carried no `Accept-Encoding` header at all.
    accept_encoding: Option<String>,
    is_head: bool,
    config: Arc<CompressionConfig>,
}

impl<F, B, E> Future for ResponseFuture<F>
where
    F: Future<Output = Result<Response<B>, E>>,
    B: http_body::Body,
{
    type Output = Result<Response<CompressionBody<B>>, E>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        let response = match this.inner.poll(cx) {
            Poll::Ready(Ok(response)) => response,
            Poll::Ready(Err(error)) => return Poll::Ready(Err(error)),
            Poll::Pending => return Poll::Pending,
        };
        Poll::Ready(Ok(compress_response(
            response,
            this.accept_encoding.as_deref(),
            *this.is_head,
            this.config,
        )))
    }
}

/// All `Accept-Encoding` header lines joined into one list, or `None` when
/// the header is absent (which RFC 9110 treats differently from empty).
fn joined_accept_encoding(headers: &HeaderMap) -> Option<String> {
    let mut values = headers
        .get_all(header::ACCEPT_ENCODING)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .peekable();
    values.peek()?;
    Some(values.collect::<Vec<_>>().join(", "))
}

/// Pick the coding for a response (`None` = identity).
fn choose_coding(
    accept_encoding: Option<&str>,
    config: &CompressionConfig,
) -> Option<ContentCoding> {
    let available = config.available();
    if available.is_empty() {
        return None;
    }
    // `NotAcceptable` (identity refused and nothing else acceptable): send
    // identity anyway instead of failing an otherwise successful response.
    let chosen = oxiarc_http::negotiate(accept_encoding, &available).unwrap_or(None)?;

    // oxiarc-http 0.4.2 falls back to the server's first coding when identity
    // *and* every listed coding carry `q=0` (without a wildcard). A coding the
    // client named with `q=0` must never be sent (RFC 9110 §12.5.3), so treat
    // that case as identity.
    let explicitly_refused = accept_encoding
        .and_then(|raw| oxiarc_http::parse_accept_encoding(raw).ok())
        .is_some_and(|entries| {
            entries.iter().any(|entry| {
                entry.coding.as_ref() == Some(&chosen) && !entry.qvalue.is_acceptable()
            })
        });
    (!explicitly_refused).then_some(chosen)
}

/// Whether a content type is worth compressing.
fn is_compressible_content_type(headers: &HeaderMap) -> bool {
    let Some(content_type) = headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
    else {
        return true;
    };
    let essence = content_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();

    if essence.starts_with("image/") {
        return essence == "image/svg+xml";
    }
    if essence.starts_with("audio/") || essence.starts_with("video/") {
        return false;
    }
    !matches!(
        essence.as_str(),
        "text/event-stream"
            | "application/zip"
            | "application/gzip"
            | "application/x-gzip"
            | "application/zstd"
            | "application/x-bzip2"
            | "application/x-xz"
            | "application/x-7z-compressed"
            | "application/x-rar-compressed"
            | "application/vnd.rar"
            | "application/wasm"
            | "font/woff"
            | "font/woff2"
    )
}

/// Whether the response is eligible for compression at all (independent of
/// what the client accepts).
fn is_eligible<B>(response: &Response<B>, is_head: bool, config: &CompressionConfig) -> bool {
    let status = response.status();
    let headers = response.headers();
    if is_head
        || status.is_informational()
        || status == StatusCode::NO_CONTENT
        || status == StatusCode::NOT_MODIFIED
        || status == StatusCode::PARTIAL_CONTENT
        || headers.contains_key(header::CONTENT_ENCODING)
        || headers.contains_key(header::CONTENT_RANGE)
    {
        return false;
    }
    let no_transform = headers
        .get_all(header::CACHE_CONTROL)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .any(|directive| directive.trim().eq_ignore_ascii_case("no-transform"));
    if no_transform {
        return false;
    }
    let too_small = headers
        .get(header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<u64>().ok())
        .is_some_and(|length| length < config.min_size);
    !too_small && is_compressible_content_type(headers)
}

/// Add `Accept-Encoding` to `Vary` unless it (or `*`) is already there.
fn add_vary_accept_encoding(headers: &mut HeaderMap) {
    let already = headers
        .get_all(header::VARY)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .any(|field| {
            let field = field.trim();
            field == "*" || field.eq_ignore_ascii_case("accept-encoding")
        });
    if !already {
        headers.append(header::VARY, HeaderValue::from_static("accept-encoding"));
    }
}

fn compress_response<B: http_body::Body>(
    response: Response<B>,
    accept_encoding: Option<&str>,
    is_head: bool,
    config: &CompressionConfig,
) -> Response<CompressionBody<B>> {
    if !is_eligible(&response, is_head, config) {
        return response.map(CompressionBody::identity);
    }

    let (mut parts, body) = response.into_parts();
    add_vary_accept_encoding(&mut parts.headers);

    let Some(coding) = choose_coding(accept_encoding, config) else {
        return Response::from_parts(parts, CompressionBody::identity(body));
    };
    let options = oxiarc_http::EncodeOptions::new()
        .with_level(config.level)
        .with_brotli_quality(config.brotli_quality);
    let body = match CompressionBody::encoded(body, &coding, options) {
        Ok(body) => body,
        // The coding came from our own `available` list, so this is not
        // expected; fall back to identity rather than failing the response.
        Err(body) => return Response::from_parts(parts, CompressionBody::identity(body)),
    };

    parts.headers.remove(header::CONTENT_LENGTH);
    parts.headers.remove(header::ACCEPT_RANGES);
    if let Ok(value) = HeaderValue::from_str(coding.as_str()) {
        parts.headers.insert(header::CONTENT_ENCODING, value);
    }
    Response::from_parts(parts, body)
}

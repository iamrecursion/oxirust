//! Tests for the OxiARC-backed response compression layer.

use super::*;
use bytes::Bytes;
use http_body::{Body, Frame};
use http_body_util::{BodyExt, Full};
use std::collections::VecDeque;
use std::convert::Infallible;
use tower::ServiceExt;

type TestResult = std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>;

/// Highly compressible JSON-ish payload well above the minimum size.
fn payload() -> Bytes {
    Bytes::from("{\"text\":\"hello voirs\",\"samples\":[0,0,0,0,0,0,0,0]}\n".repeat(64))
}

async fn respond(
    layer: CompressionLayer,
    request: Request<()>,
    response: Response<Full<Bytes>>,
) -> std::result::Result<Response<CompressionBody<Full<Bytes>>>, Infallible> {
    let service = tower::service_fn(move |_request: Request<()>| {
        let response = response.clone();
        async move { Ok::<_, Infallible>(response) }
    });
    layer.layer(service).oneshot(request).await
}

fn request(accept_encoding: Option<&str>) -> Result<Request<()>, http::Error> {
    let mut builder = Request::builder().uri("/synthesize");
    if let Some(value) = accept_encoding {
        builder = builder.header(header::ACCEPT_ENCODING, value);
    }
    builder.body(())
}

fn json_response(body: Bytes) -> Result<Response<Full<Bytes>>, http::Error> {
    Response::builder()
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::CONTENT_LENGTH, body.len())
        .body(Full::new(body))
}

fn header_str(response: &Response<impl Sized>, name: header::HeaderName) -> Option<&str> {
    response.headers().get(name).and_then(|v| v.to_str().ok())
}

#[test]
fn test_negotiation_matrix() {
    let all = CompressionConfig::default();
    let cases: &[(Option<&str>, Option<ContentCoding>)] = &[
        // Header absent: any coding acceptable, server preference wins.
        (None, Some(ContentCoding::Gzip)),
        // Present but empty: identity only.
        (Some(""), None),
        (Some("gzip"), Some(ContentCoding::Gzip)),
        (Some("br"), Some(ContentCoding::Brotli)),
        (Some("deflate"), Some(ContentCoding::Deflate)),
        (Some("x-gzip"), Some(ContentCoding::Gzip)),
        // Client q-values dominate the server order.
        (Some("gzip;q=0.5, br;q=0.8"), Some(ContentCoding::Brotli)),
        (Some("deflate, gzip;q=0.9"), Some(ContentCoding::Deflate)),
        // Ties fall back to the server order (gzip first).
        (Some("br, gzip"), Some(ContentCoding::Gzip)),
        // Wildcard covers unlisted codings, explicit q=0 still excludes.
        (Some("br;q=0, *"), Some(ContentCoding::Gzip)),
        (Some("gzip;q=0, br;q=0, *"), Some(ContentCoding::Deflate)),
        // `*;q=0` refuses codings but identity stays acceptable.
        (Some("*;q=0"), None),
        (Some("identity"), None),
        // Only unsupported codings: identity.
        (Some("zstd, compress"), None),
        // Identity refused and nothing else addressed: server order.
        (Some("identity;q=0"), Some(ContentCoding::Gzip)),
        // Identity refused and every server coding refused: not acceptable
        // -> sent as identity rather than failing the response.
        (Some("identity;q=0, gzip;q=0, br;q=0, deflate;q=0"), None),
    ];
    for (accept, expected) in cases {
        assert_eq!(
            &choose_coding(*accept, &all),
            expected,
            "Accept-Encoding: {accept:?}"
        );
    }

    let no_gzip = CompressionConfig {
        gzip: false,
        ..CompressionConfig::default()
    };
    assert_eq!(
        choose_coding(Some("gzip, br"), &no_gzip),
        Some(ContentCoding::Brotli)
    );
    let none = CompressionConfig {
        gzip: false,
        br: false,
        deflate: false,
        ..CompressionConfig::default()
    };
    assert_eq!(choose_coding(None, &none), None);
}

/// Every coding round-trips through oxiarc's decoder with correct headers.
#[tokio::test]
async fn test_round_trip_each_coding() -> TestResult {
    for token in ["gzip", "br", "deflate"] {
        let body = payload();
        let response = respond(
            CompressionLayer::new(),
            request(Some(token))?,
            json_response(body.clone())?,
        )
        .await?;

        assert_eq!(header_str(&response, header::CONTENT_ENCODING), Some(token));
        assert_eq!(header_str(&response, header::VARY), Some("accept-encoding"));
        assert!(response.headers().get(header::CONTENT_LENGTH).is_none());
        assert!(response.body().is_encoded());

        let encoded = response.into_body().collect().await?.to_bytes();
        assert!(encoded.len() < body.len(), "{token} should shrink the body");
        let decoded = oxiarc_http::decode_body_from_header(
            token,
            &encoded,
            &oxiarc_http::DecodeLimits::default(),
        )?;
        assert_eq!(decoded, body.as_ref(), "{token} round trip");
    }
    Ok(())
}

#[tokio::test]
async fn test_multiple_accept_encoding_headers_are_joined() -> TestResult {
    let mut req = request(Some("br;q=0.1"))?;
    req.headers_mut()
        .append(header::ACCEPT_ENCODING, HeaderValue::from_static("gzip"));
    let response = respond(CompressionLayer::new(), req, json_response(payload())?).await?;
    assert_eq!(
        header_str(&response, header::CONTENT_ENCODING),
        Some("gzip")
    );
    Ok(())
}

/// Responses that must not be touched keep their body and headers.
#[tokio::test]
async fn test_pass_through_cases() -> TestResult {
    let body = payload();

    let already = Response::builder()
        .header(header::CONTENT_ENCODING, "br")
        .body(Full::new(body.clone()))?;
    let small = json_response(Bytes::from_static(b"{}"))?;
    let png = Response::builder()
        .header(header::CONTENT_TYPE, "image/png")
        .body(Full::new(body.clone()))?;
    let wav = Response::builder()
        .header(header::CONTENT_TYPE, "audio/wav; rate=22050")
        .body(Full::new(body.clone()))?;
    let events = Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .body(Full::new(body.clone()))?;
    let no_transform = Response::builder()
        .header(header::CACHE_CONTROL, "public, no-transform")
        .body(Full::new(body.clone()))?;
    let partial = Response::builder()
        .status(StatusCode::PARTIAL_CONTENT)
        .header(header::CONTENT_RANGE, "bytes 0-9/100")
        .body(Full::new(body.clone()))?;
    let no_content = Response::builder()
        .status(StatusCode::NO_CONTENT)
        .body(Full::new(Bytes::new()))?;

    for (name, response) in [
        ("already encoded", already),
        ("below min size", small),
        ("image/png", png),
        ("audio/wav", wav),
        ("event stream", events),
        ("no-transform", no_transform),
        ("206", partial),
        ("204", no_content),
    ] {
        let original_headers = response.headers().clone();
        let original_body = response.body().clone().collect().await?.to_bytes();
        let result = respond(
            CompressionLayer::new(),
            request(Some("gzip, br"))?,
            response,
        )
        .await?;
        assert!(!result.body().is_encoded(), "{name}");
        assert_eq!(
            result.headers(),
            &original_headers,
            "{name}: headers untouched"
        );
        assert_eq!(
            result.into_body().collect().await?.to_bytes(),
            original_body,
            "{name}"
        );
    }

    let head = Request::builder()
        .method(Method::HEAD)
        .header(header::ACCEPT_ENCODING, "gzip")
        .body(())?;
    let result = respond(CompressionLayer::new(), head, json_response(body)?).await?;
    assert!(!result.body().is_encoded(), "HEAD");
    Ok(())
}

/// An eligible response negotiated to identity is sent as-is but still
/// advertises `Vary`; an existing `Vary` is extended, `*` is left alone.
#[tokio::test]
async fn test_vary_handling() -> TestResult {
    let identity = respond(
        CompressionLayer::new(),
        request(Some("identity"))?,
        json_response(payload())?,
    )
    .await?;
    assert!(!identity.body().is_encoded());
    assert_eq!(header_str(&identity, header::VARY), Some("accept-encoding"));
    assert!(identity.headers().get(header::CONTENT_LENGTH).is_some());

    let mut with_vary = json_response(payload())?;
    with_vary
        .headers_mut()
        .insert(header::VARY, HeaderValue::from_static("origin"));
    let result = respond(CompressionLayer::new(), request(Some("gzip"))?, with_vary).await?;
    let vary: Vec<_> = result.headers().get_all(header::VARY).iter().collect();
    assert_eq!(
        vary,
        [
            HeaderValue::from_static("origin"),
            HeaderValue::from_static("accept-encoding")
        ]
    );

    let mut star = json_response(payload())?;
    star.headers_mut()
        .insert(header::VARY, HeaderValue::from_static("*"));
    let result = respond(CompressionLayer::new(), request(Some("gzip"))?, star).await?;
    assert_eq!(result.headers().get_all(header::VARY).iter().count(), 1);
    Ok(())
}

/// Test body fed through a channel: `Pending` until the next chunk arrives.
struct ChannelBody {
    receiver: tokio::sync::mpsc::UnboundedReceiver<Frame<Bytes>>,
}

impl Body for ChannelBody {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
        self.receiver.poll_recv(cx).map(|frame| frame.map(Ok))
    }
}

/// Streaming: compressed bytes for the first chunk are delivered before the
/// second chunk exists, and the concatenated output decodes to the input.
#[tokio::test]
async fn test_streaming_body_is_flushed_incrementally() -> TestResult {
    let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
    let mut body = CompressionBody::encoded(
        ChannelBody { receiver },
        &ContentCoding::Gzip,
        oxiarc_http::EncodeOptions::new(),
    )
    .map_err(|_| "encoder creation failed")?;

    let first = payload();
    sender.send(Frame::data(first.clone()))?;
    let mut output = Vec::new();
    // Must yield data without the stream ending: the encoder is sync-flushed
    // once the channel is pending.
    let frame = tokio::time::timeout(std::time::Duration::from_secs(5), body.frame())
        .await?
        .ok_or("stream ended early")??;
    let chunk = frame.into_data().map_err(|_| "expected a data frame")?;
    assert!(!chunk.is_empty());
    output.extend_from_slice(&chunk);

    let second = Bytes::from_static(b"second chunk of the stream\n");
    sender.send(Frame::data(second.clone()))?;
    let mut trailers = HeaderMap::new();
    trailers.insert("x-voirs-frames", HeaderValue::from_static("2"));
    sender.send(Frame::trailers(trailers.clone()))?;
    drop(sender);

    let mut received_trailers = None;
    while let Some(frame) = body.frame().await {
        let frame = frame?;
        match frame.into_data() {
            Ok(data) => output.extend_from_slice(&data),
            Err(frame) => received_trailers = frame.into_trailers().ok(),
        }
    }
    assert!(body.is_end_stream());
    assert_eq!(received_trailers, Some(trailers));

    let decoded = oxiarc_http::decode_body_from_header(
        "gzip",
        &output,
        &oxiarc_http::DecodeLimits::default(),
    )?;
    let mut expected = first.to_vec();
    expected.extend_from_slice(&second);
    assert_eq!(decoded, expected);
    Ok(())
}

/// Frames queued up front (no pending gaps) still produce a valid stream.
#[tokio::test]
async fn test_multi_frame_body_round_trip() -> TestResult {
    struct Frames(VecDeque<Bytes>);
    impl Body for Frames {
        type Data = Bytes;
        type Error = Infallible;
        fn poll_frame(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
            Poll::Ready(self.0.pop_front().map(|b| Ok(Frame::data(b))))
        }
    }

    let chunks: VecDeque<Bytes> = (0..50)
        .map(|i| Bytes::from(format!("chunk {i}: {}\n", "ab".repeat(i))))
        .collect();
    let expected: Vec<u8> = chunks.iter().flat_map(|c| c.iter().copied()).collect();
    for coding in [
        ContentCoding::Gzip,
        ContentCoding::Brotli,
        ContentCoding::Deflate,
    ] {
        let body = CompressionBody::encoded(
            Frames(chunks.clone()),
            &coding,
            oxiarc_http::EncodeOptions::new(),
        )
        .map_err(|_| "encoder creation failed")?;
        let encoded = body.collect().await?.to_bytes();
        let decoded = oxiarc_http::decode_body_from_header(
            coding.as_str(),
            &encoded,
            &oxiarc_http::DecodeLimits::default(),
        )?;
        assert_eq!(decoded, expected, "{coding:?}");
    }
    Ok(())
}

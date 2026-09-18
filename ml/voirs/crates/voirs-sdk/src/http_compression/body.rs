//! Response body that is passed through or incrementally content-encoded.

use bytes::{Buf, Bytes};
use http::HeaderMap;
use http_body::{Body, Frame, SizeHint};
use oxiarc_http::{ContentCoding, EncodeOptions, Encoder};
use pin_project::pin_project;
use std::io::Write;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll};

/// Boxed error type produced by [`CompressionBody`].
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// `Write` target shared between the encoder (which owns one handle) and the
/// body (which drains what the encoder has produced so far).
#[derive(Clone, Default)]
struct SharedSink(Arc<Mutex<Vec<u8>>>);

impl SharedSink {
    fn lock(&self) -> MutexGuard<'_, Vec<u8>> {
        // A poisoned lock only means another thread panicked mid-write; the
        // buffered bytes are still the best data available.
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn drain(&self) -> Bytes {
        Bytes::from(std::mem::take(&mut *self.lock()))
    }
}

impl Write for SharedSink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.lock().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// State of an encoding body.
struct EncodeState {
    /// `None` once `finish` has written the final framing.
    encoder: Option<Encoder<SharedSink>>,
    sink: SharedSink,
    /// Input was written since the last sync flush.
    unflushed: bool,
    /// Trailers of the inner body, emitted after the final data frame.
    trailers: Option<HeaderMap>,
    done: bool,
}

/// Response body of [`super::Compression`]: either the inner body unchanged
/// or its content-encoded form.
#[pin_project]
pub struct CompressionBody<B> {
    #[pin]
    kind: Kind<B>,
}

#[pin_project(project = KindProj)]
enum Kind<B> {
    /// Passed through unchanged.
    Identity(#[pin] B),
    /// Encoded with the negotiated coding.
    Encoded {
        #[pin]
        inner: B,
        state: Box<EncodeState>,
    },
}

impl<B> CompressionBody<B> {
    pub(crate) fn identity(body: B) -> Self {
        Self {
            kind: Kind::Identity(body),
        }
    }

    /// Whether this body is being content-encoded.
    #[must_use]
    pub fn is_encoded(&self) -> bool {
        matches!(self.kind, Kind::Encoded { .. })
    }

    /// Encode `body` with `coding`; gives the body back if the encoder cannot
    /// be created.
    pub(crate) fn encoded(
        body: B,
        coding: &ContentCoding,
        options: EncodeOptions<'_>,
    ) -> Result<Self, B> {
        let sink = SharedSink::default();
        match Encoder::new(sink.clone(), coding, options) {
            Ok(encoder) => Ok(Self {
                kind: Kind::Encoded {
                    inner: body,
                    state: Box::new(EncodeState {
                        encoder: Some(encoder),
                        sink,
                        unflushed: false,
                        trailers: None,
                        done: false,
                    }),
                },
            }),
            Err(_) => Err(body),
        }
    }
}

fn io_error(error: std::io::Error) -> BoxError {
    Box::new(error)
}

/// A data frame with whatever the encoder has produced, or `None` if empty.
fn drained_frame(state: &EncodeState) -> Option<Frame<Bytes>> {
    let bytes = state.sink.drain();
    (!bytes.is_empty()).then(|| Frame::data(bytes))
}

impl<B> Body for CompressionBody<B>
where
    B: Body,
    B::Error: Into<BoxError>,
{
    type Data = Bytes;
    type Error = BoxError;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        match self.project().kind.project() {
            KindProj::Identity(inner) => match inner.poll_frame(cx) {
                Poll::Ready(Some(Ok(frame))) => Poll::Ready(Some(Ok(
                    frame.map_data(|mut data| data.copy_to_bytes(data.remaining()))
                ))),
                Poll::Ready(Some(Err(error))) => Poll::Ready(Some(Err(error.into()))),
                Poll::Ready(None) => Poll::Ready(None),
                Poll::Pending => Poll::Pending,
            },
            KindProj::Encoded { mut inner, state } => poll_encoded(inner.as_mut(), state, cx),
        }
    }

    fn is_end_stream(&self) -> bool {
        match &self.kind {
            Kind::Identity(inner) => inner.is_end_stream(),
            Kind::Encoded { state, .. } => state.done && state.trailers.is_none(),
        }
    }

    fn size_hint(&self) -> SizeHint {
        match &self.kind {
            Kind::Identity(inner) => inner.size_hint(),
            // The encoded length is not known up front.
            Kind::Encoded { .. } => SizeHint::default(),
        }
    }
}

fn poll_encoded<B>(
    mut inner: Pin<&mut B>,
    state: &mut EncodeState,
    cx: &mut Context<'_>,
) -> Poll<Option<Result<Frame<Bytes>, BoxError>>>
where
    B: Body,
    B::Error: Into<BoxError>,
{
    loop {
        if state.done {
            return Poll::Ready(state.trailers.take().map(|t| Ok(Frame::trailers(t))));
        }

        match inner.as_mut().poll_frame(cx) {
            Poll::Pending => {
                // Nothing more right now: sync-flush so a streaming client
                // receives what has been produced so far.
                if state.unflushed {
                    state.unflushed = false;
                    if let Some(encoder) = state.encoder.as_mut() {
                        if let Err(error) = encoder.flush() {
                            return Poll::Ready(Some(Err(io_error(error))));
                        }
                    }
                    if let Some(frame) = drained_frame(state) {
                        return Poll::Ready(Some(Ok(frame)));
                    }
                }
                return Poll::Pending;
            }
            Poll::Ready(Some(Err(error))) => return Poll::Ready(Some(Err(error.into()))),
            Poll::Ready(Some(Ok(frame))) => match frame.into_data() {
                Ok(mut data) => {
                    let bytes = data.copy_to_bytes(data.remaining());
                    if bytes.is_empty() {
                        continue;
                    }
                    if let Some(encoder) = state.encoder.as_mut() {
                        if let Err(error) = encoder.write_all(&bytes) {
                            return Poll::Ready(Some(Err(io_error(error))));
                        }
                    }
                    state.unflushed = true;
                    if let Some(frame) = drained_frame(state) {
                        return Poll::Ready(Some(Ok(frame)));
                    }
                }
                Err(frame) => {
                    // Trailers end the body: finish encoding first.
                    state.trailers = frame.into_trailers().ok();
                    if let Some(result) = finish(state) {
                        return Poll::Ready(Some(result));
                    }
                }
            },
            Poll::Ready(None) => {
                if let Some(result) = finish(state) {
                    return Poll::Ready(Some(result));
                }
            }
        }
    }
}

/// Write the final framing; returns the last data frame (or an error), or
/// `None` if the encoder produced nothing more.
fn finish(state: &mut EncodeState) -> Option<Result<Frame<Bytes>, BoxError>> {
    state.done = true;
    if let Some(encoder) = state.encoder.take() {
        if let Err(error) = encoder.finish() {
            return Some(Err(io_error(error)));
        }
    }
    drained_frame(state).map(Ok)
}

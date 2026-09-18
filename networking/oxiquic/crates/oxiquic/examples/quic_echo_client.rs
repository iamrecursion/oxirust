//! Minimal QUIC echo client.
//!
//! Connects to a `quic_echo_server` instance (default `127.0.0.1:4433`,
//! override with the first CLI argument) **without** validating the
//! server's certificate: this demo pairs with the ephemeral self-signed
//! certificate `quic_echo_server` regenerates on every run, so there is no
//! CA-issued chain for a normal client to check. A client talking to a real
//! server should use [`oxiquic::connect`] instead, which validates against
//! the system's WebPKI trust store.
//!
//! ```text
//! cargo run -p oxiquic --example quic_echo_server --features dangerous
//! cargo run -p oxiquic --example quic_echo_client --features dangerous
//! ```

use std::net::SocketAddr;

const DEFAULT_ADDR: &str = "127.0.0.1:4433";
const MESSAGE: &[u8] = b"Hello, OxiQUIC!";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr: SocketAddr = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_ADDR.to_string())
        .parse()?;

    let mut conn = oxiquic::connect_insecure(addr, "localhost").await?;
    println!("quic_echo_client: connected to {addr}");

    let stream = conn.open_bidi()?;
    conn.send(stream, MESSAGE, true).await?;
    println!(
        "quic_echo_client: sent {} bytes on stream {stream:?}",
        MESSAGE.len()
    );

    let (echoed, _fin) = conn.read(stream).await?;
    println!(
        "quic_echo_client: received {} bytes: {:?}",
        echoed.len(),
        String::from_utf8_lossy(&echoed)
    );

    if echoed != MESSAGE {
        return Err(format!(
            "server echoed {} bytes, expected the original {} bytes back unchanged",
            echoed.len(),
            MESSAGE.len()
        )
        .into());
    }
    println!("quic_echo_client: echo verified, payload matched");

    Ok(())
}

# Contributing to OxiMedia Video-over-IP

Thank you for your interest in contributing to the OxiMedia Video-over-IP protocol implementation!

## Code of Conduct

This project adheres to the OxiMedia Code of Conduct. By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Rust 1.75 or later
- Tokio async runtime knowledge
- Understanding of video/audio codecs
- Network programming experience
- Familiarity with UDP and real-time protocols

### Development Setup

1. Clone the repository:
```bash
git clone https://github.com/oximedia/oximedia.git
cd oximedia/crates/oximedia-videoip
```

2. Build the crate:
```bash
cargo build
```

3. Run tests:
```bash
cargo test
```

4. Run examples:
```bash
cargo run --example simple_sender
cargo run --example simple_receiver
```

## Development Guidelines

### Code Style

Follow the Rust standard coding style:
- Use `rustfmt` for formatting (run `cargo fmt`)
- Use `clippy` for linting (run `cargo clippy`)
- Enable all pedantic warnings
- Zero unsafe code (enforced by `#![forbid(unsafe_code)]`)

### Documentation

- All public APIs must have doc comments
- Include examples in doc comments
- Document all parameters and return values
- Explain error conditions

Example:
```rust
/// Sends a video frame with optional audio samples.
///
/// # Arguments
///
/// * `video_frame` - The video frame to send
/// * `audio_samples` - Optional audio samples to send with the frame
///
/// # Errors
///
/// Returns an error if:
/// - Encoding fails
/// - Network transmission fails
/// - The frame is too large for packetization
///
/// # Example
///
/// ```ignore
/// let frame = VideoFrame::new(...);
/// source.send_frame(frame, None).await?;
/// ```
pub async fn send_frame(
    &mut self,
    video_frame: VideoFrame,
    audio_samples: Option<AudioSamples>,
) -> VideoIpResult<()>
```

### Testing

#### Unit Tests

- Every module must have comprehensive unit tests
- Test normal cases and edge cases
- Test error conditions
- Aim for > 80% code coverage

Example:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packet_encoding() {
        let packet = create_test_packet();
        let encoded = packet.encode();
        let decoded = Packet::decode(&encoded).unwrap();
        assert_eq!(packet.header.sequence, decoded.header.sequence);
    }

    #[test]
    fn test_invalid_packet() {
        let result = Packet::decode(&[0u8; 10]);
        assert!(result.is_err());
    }
}
```

#### Integration Tests

- Test complete workflows (sender -> receiver)
- Test protocol interoperability
- Test error recovery mechanisms

#### Benchmarks

- Benchmark critical paths (encoding, packetization, FEC)
- Use Criterion for consistent results
- Target < 1ms for packet operations

### Performance Considerations

1. **Zero-Copy**: Use `bytes::Bytes` for packet payloads
2. **Lock-Free**: Prefer atomic operations over mutexes
3. **Async**: Use async/await for all I/O operations
4. **Memory Pools**: Reuse buffers to reduce allocations
5. **SIMD**: Use SIMD where applicable (e.g., FEC encoding)

### Error Handling

- Use `thiserror` for error types
- All errors must be actionable and descriptive
- Log errors with appropriate severity
- Never panic in production code
- Use `Result` for all fallible operations

Example:
```rust
#[derive(Debug, thiserror::Error)]
pub enum VideoIpError {
    #[error("Packet too large: {size} bytes (max {max})")]
    PacketTooLarge { size: usize, max: usize },

    #[error("FEC encoding failed: {0}")]
    Fec(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
```

## Architecture

### Module Organization

```
oximedia-videoip/
├── src/
│   ├── lib.rs           # Public API
│   ├── error.rs         # Error types
│   ├── types.rs         # Core types
│   ├── packet.rs        # Packet format
│   ├── transport.rs     # UDP transport
│   ├── fec.rs           # Forward error correction
│   ├── jitter.rs        # Jitter buffer
│   ├── codec.rs         # Codec wrappers
│   ├── discovery.rs     # mDNS discovery
│   ├── source.rs        # Source implementation
│   ├── receiver.rs      # Receiver implementation
│   ├── stats.rs         # Statistics tracking
│   ├── metadata.rs      # Metadata handling
│   ├── ptz.rs           # PTZ control
│   ├── tally.rs         # Tally lights
│   └── utils/           # Utility modules
│       ├── bandwidth.rs # Bandwidth adaptation
│       ├── connection.rs# Connection management
│       ├── quality.rs   # Quality monitoring
│       └── ring_buffer.rs# Ring buffer
├── examples/            # Usage examples
├── tests/               # Integration tests
├── benches/             # Benchmarks
└── README.md
```

### Key Abstractions

#### Source
- Manages video/audio encoding
- Packetizes frames
- Handles FEC encoding
- Broadcasts via mDNS
- Sends control messages

#### Receiver
- Discovers sources via mDNS
- Receives and reorders packets
- Handles FEC decoding
- Manages jitter buffer
- Decodes video/audio

#### Transport
- Low-level UDP operations
- Socket tuning (buffers, QoS)
- Multicast support

### Data Flow

```
Source:
Video Frame → Encode → Packetize → FEC → UDP Send

Receiver:
UDP Recv → FEC Decode → Jitter Buffer → Reorder → Decode → Video Frame
```

## Feature Requests

### Proposing New Features

1. Open an issue describing the feature
2. Explain the use case
3. Propose an API design
4. Discuss implementation approach

### Implementing Features

1. Create a feature branch
2. Implement with tests
3. Update documentation
4. Submit pull request

## Pull Request Process

### Before Submitting

1. Run all tests: `cargo test`
2. Check formatting: `cargo fmt --check`
3. Check for warnings: `cargo clippy -- -D warnings`
4. Update CHANGELOG.md
5. Update documentation

### PR Guidelines

- One feature/fix per PR
- Clear, descriptive title
- Detailed description
- Reference related issues
- Include test coverage
- Update documentation

### PR Template

```markdown
## Description

Brief description of changes.

## Motivation

Why is this change needed?

## Changes

- Change 1
- Change 2

## Testing

How was this tested?

## Checklist

- [ ] Tests pass
- [ ] No clippy warnings
- [ ] Documentation updated
- [ ] CHANGELOG updated
```

## Versioning

We use [Semantic Versioning](https://semver.org/):

- MAJOR: Breaking API changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

## Performance Testing

### Benchmarking

Run benchmarks:
```bash
cargo bench
```

### Profiling

Use `perf` on Linux:
```bash
cargo build --release --example simple_sender
perf record -g target/release/examples/simple_sender
perf report
```

### Network Testing

Test with `tc` (traffic control):
```bash
# Simulate 1% packet loss
sudo tc qdisc add dev lo root netem loss 1%

# Simulate 50ms delay
sudo tc qdisc add dev lo root netem delay 50ms

# Remove
sudo tc qdisc del dev lo root
```

## Debugging

### Enable Logging

```rust
RUST_LOG=oximedia_videoip=trace cargo run --example simple_sender
```

### Packet Capture

Use Wireshark to inspect packets:
```bash
wireshark -i lo -f "udp port 5000"
```

### Common Issues

#### Packet Loss
- Check network conditions
- Verify FEC configuration
- Monitor buffer occupancy

#### High Latency
- Reduce jitter buffer size
- Check encoding time
- Verify network path

#### Frame Drops
- Check encoding performance
- Monitor CPU usage
- Verify bitrate settings

## Code Review

### Review Checklist

- [ ] Code follows style guide
- [ ] All tests pass
- [ ] No new warnings
- [ ] Documentation updated
- [ ] Performance impact assessed
- [ ] Security implications considered
- [ ] Error handling complete

### Review Process

1. Automated checks (CI)
2. Code review by maintainer
3. Testing in staging
4. Merge to main

## Release Process

1. Update version in `Cargo.toml`
2. Update CHANGELOG.md
3. Create git tag
4. Build and test release
5. Publish to crates.io
6. Create GitHub release

## Getting Help

- Open an issue for bugs
- Use discussions for questions
- Join our Discord server
- Email: dev@oximedia.org

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

## Acknowledgments

Thank you to all contributors who help make this project better!

Special thanks to:
- The Rust community
- Tokio project
- VP9/AV1 codec developers
- SMPTE for professional media standards

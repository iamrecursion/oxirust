# OxiMedia Video-over-IP Protocol Specification

Version: 1.0
Date: 2024

## Overview

The OxiMedia Video-over-IP protocol is a patent-free alternative to NDI for professional video streaming over IP networks. It provides low-latency, high-quality video and audio transmission with resilience to network conditions.

## Design Goals

1. **Low Latency**: Target < 16ms at 60fps (less than 1 frame)
2. **Network Resilience**: FEC and adaptive bitrate to handle packet loss
3. **Patent-Free**: Only use royalty-free codecs and techniques
4. **Professional Features**: Tally lights, PTZ control, timecode embedding
5. **Scalability**: Support multicast for efficient one-to-many streaming

## Protocol Stack

```
┌────────────────────────────────────┐
│ Application (Video/Audio Frames)   │
├────────────────────────────────────┤
│ Metadata (Timecode, PTZ, Tally)    │
├────────────────────────────────────┤
│ FEC (Reed-Solomon)                 │
├────────────────────────────────────┤
│ Packetization                      │
├────────────────────────────────────┤
│ UDP                                │
├────────────────────────────────────┤
│ IP (IPv4/IPv6)                     │
└────────────────────────────────────┘
```

## Packet Format

### Header Structure (20 bytes)

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                   Magic Number (0x4F585650)                   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|   Version     |     Flags     |          Sequence             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
+                       Timestamp (64-bit)                      +
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| Stream Type   |   Reserved    |        Payload Size           |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

### Field Descriptions

#### Magic Number (4 bytes)
- Value: `0x4F585650` ("OXVP" in ASCII)
- Purpose: Packet identification and validation
- Endianness: Big-endian

#### Version (1 byte)
- Current version: 1
- Purpose: Protocol versioning for future compatibility

#### Flags (1 byte)
Bit flags indicating packet properties:
- Bit 0 (`0x01`): VIDEO - Contains video data
- Bit 1 (`0x02`): AUDIO - Contains audio data
- Bit 2 (`0x04`): METADATA - Contains metadata
- Bit 3 (`0x08`): KEYFRAME - Video keyframe/IDR
- Bit 4 (`0x10`): FEC - FEC parity packet
- Bit 5 (`0x20`): END_OF_FRAME - Last packet in frame
- Bit 6 (`0x40`): START_OF_FRAME - First packet in frame
- Bit 7: Reserved

#### Sequence Number (2 bytes)
- Range: 0-65535 (wraps around)
- Purpose: Packet ordering and loss detection
- Increments by 1 for each packet sent

#### Timestamp (8 bytes)
- Unit: Microseconds since UNIX epoch
- Purpose: Synchronization and latency calculation
- Endianness: Big-endian

#### Stream Type (1 byte)
- 0: Program feed
- 1: Preview feed
- 2: Alpha channel
- 3-254: Custom streams
- 255: Reserved

#### Reserved (1 byte)
- Must be set to 0
- Reserved for future use

#### Payload Size (2 bytes)
- Range: 0-8980 bytes
- Maximum packet size: 9000 bytes (jumbo frame)
- Maximum payload: 8980 bytes (9000 - 20 header)

### Payload

Variable-length payload containing:
- Encoded video data (VP9, AV1, or uncompressed)
- Encoded audio data (Opus or PCM)
- Metadata (timecode, PTZ commands, tally info)
- FEC parity data

## Supported Codecs

### Video Codecs

#### VP9 (Compressed)
- Profile: 0 (8-bit 4:2:0)
- Quality: CQ mode with configurable quality
- Keyframe interval: Configurable (default: 2 seconds)
- Thread count: Auto-detected

#### AV1 (Compressed)
- Profile: Main
- Level: Auto
- Usage: Real-time
- Speed: 5-8 (higher = faster encoding)

#### Uncompressed Video
- **v210**: 10-bit 4:2:2 YUV (packed)
  - 32 bits for 3 pixels (4 bytes)
  - Big-endian bit packing

- **UYVY**: 8-bit 4:2:2 YUV (packed)
  - 2 bytes per pixel (Y, U, V interleaved)

- **YUV420p**: 8-bit 4:2:0 YUV (planar)
  - 1.5 bytes per pixel

- **YUV420p10**: 10-bit 4:2:0 YUV (planar)
  - 3 bytes per pixel

### Audio Codecs

#### Opus (Compressed)
- Sample rates: 48kHz, 96kHz
- Bitrate: 64-512 kbps
- Frame size: 2.5, 5, 10, 20, 40, 60 ms
- Complexity: 0-10

#### PCM (Uncompressed)
- **PCM16**: Signed 16-bit little-endian
- **PCM24**: Signed 24-bit little-endian
- **PCM32F**: 32-bit float little-endian

## Forward Error Correction (FEC)

### Reed-Solomon FEC

The protocol uses Reed-Solomon erasure codes for packet loss recovery:

- **Data Shards**: 10-50 packets (configurable)
- **Parity Shards**: 1-10 packets (configurable)
- **FEC Ratio**: Typical 5-20%
- **Block Size**: Aligned to data shard count

#### FEC Packet Structure

FEC parity packets have the `FEC` flag set and contain:
- Standard packet header with FEC flag
- Parity data computed over a block of data packets
- Same size as the largest data packet in the block

#### Recovery Algorithm

1. Collect packets from a FEC block
2. If `received >= data_shards`, can recover lost packets
3. Use Reed-Solomon decoding to reconstruct missing packets
4. Maximum recoverable loss: `parity_shards` packets per block

### FEC Configuration

| Ratio | Data Shards | Parity Shards | Recovery Capability |
|-------|-------------|---------------|---------------------|
| 5%    | 20          | 1             | 1 lost packet       |
| 10%   | 20          | 2             | 2 lost packets      |
| 20%   | 20          | 4             | 4 lost packets      |

## Service Discovery (mDNS/DNS-SD)

### Service Type

```
_oximedia-videoip._udp.local.
```

### TXT Records

| Key           | Value Type | Description                    |
|---------------|------------|--------------------------------|
| codec         | string     | Video codec (vp9, av1, v210)   |
| width         | integer    | Video width in pixels          |
| height        | integer    | Video height in pixels         |
| fps           | float      | Frame rate (e.g., "29.97")     |
| audio_codec   | string     | Audio codec (opus, pcm16)      |
| sample_rate   | integer    | Audio sample rate (Hz)         |
| channels      | integer    | Audio channel count            |
| bitrate       | integer    | Target bitrate (bps)           |
| description   | string     | Source description (optional)  |

### Example mDNS Advertisement

```
camera1._oximedia-videoip._udp.local. IN PTR camera1.local.
camera1.local. IN SRV 0 0 5000 192.168.1.100.
camera1.local. IN TXT "codec=vp9" "width=1920" "height=1080"
                      "fps=60" "audio_codec=opus"
                      "sample_rate=48000" "channels=2"
```

## Metadata

### Timecode (SMPTE 12M)

4-byte timecode structure:

```
Byte 0: Frames (0-29, BCD) | Drop frame flag (bit 6)
Byte 1: Seconds (0-59, BCD)
Byte 2: Minutes (0-59, BCD)
Byte 3: Hours (0-23, BCD)
```

Example: `01:23:45:12` (non-drop frame)
```
Bytes: 0x0C 0x2D 0x17 0x01
```

### PTZ Control Messages

22-byte PTZ message structure:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|          Source ID            |   Command     | Pan Speed     |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| Tilt Speed    | Zoom Speed    |    Preset     |               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+               +
|                      Pan Position (float)                     |
+               +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|               |             Tilt Position (float)             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                      Zoom Position (float)                    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

Commands:
- 0: Pan Left
- 1: Pan Right
- 2: Tilt Up
- 3: Tilt Down
- 4: Zoom In
- 5: Zoom Out
- 6: Focus Near
- 7: Focus Far
- 8: Auto Focus
- 9: Stop
- 10: Go to Preset
- 11: Save Preset
- 12: Absolute Position
- 13: Home

### Tally Light Messages

4-byte tally message structure:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|          Source ID            |     State     |  Brightness   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

States:
- 0: Off
- 1: Program (red)
- 2: Preview (green)
- 3: Both (red + green)

Brightness: 0-255 (0 = off, 255 = full brightness)

## Quality of Service (QoS)

### DSCP Marking

Packets are marked with DSCP Expedited Forwarding (EF):
- DSCP value: 46 (0x2E)
- Binary: 101110
- IPv4 ToS: 0xB8
- Priority: Highest (real-time traffic)

### Buffer Sizes

- Send buffer: 8 MB
- Receive buffer: 8 MB
- Jitter buffer: 5-100 ms (adaptive)

### Adaptive Bitrate

The protocol adjusts bitrate based on:
- Available bandwidth estimation
- Packet loss rate
- Buffer occupancy
- Round-trip time

Algorithm:
1. If packet loss > 5%, decrease bitrate by 20%
2. If buffer > 70%, decrease bitrate by 10%
3. If buffer < 30% and bandwidth available, increase bitrate by 10%
4. Clamp bitrate to configured min/max bounds

## Network Topology

### Unicast (One-to-One)

```
Source ────UDP───> Receiver
       <───ACK────
```

### Multicast (One-to-Many)

```
                 ┌─> Receiver 1
Source ──┬──UDP──┼─> Receiver 2
         │       └─> Receiver 3
         └────> Multicast Group (239.0.0.0/8)
```

### Recommended Multicast Addresses

- 239.0.0.0/24: Organization-local scope
- 239.255.0.0/16: Administratively scoped

## Performance Characteristics

### Latency Components

| Component         | Typical Latency |
|-------------------|-----------------|
| Encoding          | 1-5 ms          |
| Packetization     | 0.1 ms          |
| Network           | 1-10 ms         |
| Jitter Buffer     | 5-20 ms         |
| Decoding          | 1-5 ms          |
| **Total**         | **8-40 ms**     |

### Bandwidth Requirements

#### 1080p60 Examples

| Codec  | Quality | Bitrate      |
|--------|---------|--------------|
| VP9    | High    | 10-15 Mbps   |
| AV1    | High    | 8-12 Mbps    |
| UYVY   | N/A     | 1.5 Gbps     |
| v210   | N/A     | 1.9 Gbps     |

#### Audio

| Codec  | Channels | Bitrate     |
|--------|----------|-------------|
| Opus   | 2        | 128 kbps    |
| PCM16  | 2        | 1.5 Mbps    |
| PCM16  | 16       | 12.3 Mbps   |

## Security Considerations

### Encryption

The base protocol does not include encryption. For secure transmission:
- Use VPN tunnels (WireGuard, IPsec)
- Implement TLS/DTLS wrapper
- Use encrypted network infrastructure

### Authentication

- mDNS announcements are unauthenticated
- Implement network-level access control
- Use VLANs to segregate video traffic

## Implementation Notes

### Thread Safety

- All packet operations are thread-safe
- Jitter buffer uses atomic operations
- Statistics tracking uses lock-free data structures

### Zero-Copy

- Packets use `bytes::Bytes` for zero-copy sharing
- DMA-capable ring buffers for network I/O
- Memory pools for packet allocation

### Error Handling

- All errors are recoverable
- Packet loss is handled gracefully
- Invalid packets are logged and discarded

## Future Extensions

Planned for version 2.0:
- QUIC transport option
- WebRTC compatibility
- HDR metadata (HDR10, HLG)
- Encryption (DTLS/SRTP)
- Multi-path support
- Better congestion control

## References

- SMPTE ST 2110: Professional Media Over IP
- RFC 6716: Opus Codec
- RFC 3550: RTP Protocol
- RFC 5109: RTP FEC
- RFC 6762: mDNS
- RFC 6763: DNS-SD

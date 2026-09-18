# OxiMedia WASM

WebAssembly bindings for OxiMedia - Patent-free multimedia processing in the browser.

This is the original, monolithic `oximedia-wasm` crate (77+ bound modules
covering demuxing, color management, HDR, LUTs, spatial audio, editing,
QC, and more). If you only need WebCodecs-adjacent color/scopes/scaling/quality
tooling with a strict size budget, see
[`@cooljapan/oximedia-web`](../web/README.md) instead (a separate,
actively-maintained package from a different part of this monorepo -- see
[Installation](#installation) below for how the two relate).

## Features

- **Format Probing**: Detect container formats (WebM, Matroska, Ogg, FLAC, WAV, MP4)
- **Container Demuxing**: Extract compressed packets from media files (does not
  imply the packets' codec is decodable -- see [Supported Formats](#supported-formats))
- **Zero-Copy**: Efficient buffer management using JavaScript `ArrayBuffer`;
  buffer-crossing APIs use `Uint8Array`/`Uint8ClampedArray` (8-bit SDR) or
  `Float32Array` (HDR/linear) exclusively -- never `Float64Array`
- **Patent-Free Decoders**: Opus and FLAC audio decode; AV1 video decode
  (via `WasmMediaPlayer`) -- all royalty-free
- **Browser-Native**: No file system dependencies, works entirely in-memory

## Installation

### From npm

```bash
npm install @cooljapan/oximedia
```

`@cooljapan/oximedia` (the bundler build of *this* crate, for webpack, vite,
rollup, etc.) is the **only** package this crate publishes to npm.

`build.sh` also produces `pkg-web/` and `pkg-node/` directories for local
testing (`wasm-pack build --target web` / `--target nodejs`), but these are
unpublished build artifacts, not installable npm packages -- do not `npm
install` them.

If you need a browser-native, size-budgeted package (no bundler required),
use [`@cooljapan/oximedia-web`](../web/README.md) instead. That name belongs
to a separate, independently-versioned package (`/web` in this monorepo) with
its own four WebCodecs-adjacent modules (`scopes`, `color`, `scale`,
`quality`) -- it is **not** produced by this crate's build scripts.

### From Source

#### Prerequisites

- [Rust](https://rustup.rs/) (1.75 or later)
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)

Install wasm-pack:

```bash
cargo install wasm-pack
```

### Building

Build for different targets:

```bash
# Build for all targets (web, node, bundler)
./build.sh

# Build for web only (development mode with debug symbols)
./build-dev.sh

# Build for specific target
wasm-pack build --target web --out-dir pkg
```

Available targets:
- `web`: For use in browsers via `<script type="module">`
- `nodejs`: For use in Node.js
- `bundler`: For use with webpack, rollup, parcel, etc.

## Usage

### In the Browser

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>OxiMedia WASM Demo</title>
</head>
<body>
    <input type="file" id="fileInput" accept="video/*,audio/*">
    <pre id="output"></pre>

    <script type="module">
        import init, * as oximedia from './pkg-web/oximedia_wasm.js';

        async function run() {
            // Initialize the WASM module
            await init();

            console.log('OxiMedia version:', oximedia.version());

            // Handle file input
            document.getElementById('fileInput').addEventListener('change', async (e) => {
                const file = e.target.files[0];
                const arrayBuffer = await file.arrayBuffer();
                const data = new Uint8Array(arrayBuffer);

                // Probe format
                try {
                    const result = oximedia.probe_format(data);
                    console.log('Format:', result.format());
                    console.log('Confidence:', result.confidence());
                    console.log('Description:', result.description());

                    // Create demuxer
                    const demuxer = new oximedia.WasmDemuxer(data);
                    const probe = demuxer.probe();

                    // Get streams
                    const streams = demuxer.streams();
                    let output = `Format: ${probe.format()}\n`;
                    output += `Streams: ${streams.length}\n\n`;

                    for (const stream of streams) {
                        output += `Stream ${stream.index()}:\n`;
                        output += `  Codec: ${stream.codec()}\n`;
                        output += `  Type: ${stream.media_type()}\n`;
                        const params = stream.codec_params();
                        if (params.has_video_params()) {
                            output += `  Resolution: ${params.width()}x${params.height()}\n`;
                        }
                        if (params.has_audio_params()) {
                            output += `  Sample Rate: ${params.sample_rate()} Hz\n`;
                            output += `  Channels: ${params.channels()}\n`;
                        }
                        output += '\n';
                    }

                    // Read some packets
                    let packetCount = 0;
                    while (packetCount < 10) {
                        const packet = demuxer.read_packet();
                        if (!packet) break;
                        output += `Packet ${packetCount}: `;
                        output += `stream=${packet.stream_index()}, `;
                        output += `size=${packet.size()} bytes, `;
                        output += `keyframe=${packet.is_keyframe()}\n`;
                        packetCount++;
                    }

                    document.getElementById('output').textContent = output;
                } catch (e) {
                    console.error('Error:', e);
                    document.getElementById('output').textContent = 'Error: ' + e;
                }
            });
        }

        run();
    </script>
</body>
</html>
```

### With a Bundler (webpack, vite, etc.)

```javascript
import init, * as oximedia from '@cooljapan/oximedia';

async function processMedia(data) {
    // Initialize WASM module
    await init();

    // Probe format
    const result = oximedia.probe_format(data);
    console.log('Detected format:', result.format());

    // Create demuxer
    const demuxer = new oximedia.WasmDemuxer(data);
    demuxer.probe();

    // Get streams
    const streams = demuxer.streams();
    console.log('Found', streams.length, 'streams');

    // Read packets
    let packet;
    while ((packet = demuxer.read_packet()) !== null) {
        console.log('Packet:', packet.stream_index(), packet.size());
    }
}
```

### In Node.js

```javascript
const fs = require('fs');
const { probe_format, WasmDemuxer } = require('./pkg-node/oximedia_wasm');

// Read file
const data = fs.readFileSync('video.webm');

// Probe format
const result = probe_format(data);
console.log('Format:', result.format());

// Demux
const demuxer = new WasmDemuxer(data);
demuxer.probe();

const streams = demuxer.streams();
console.log('Streams:', streams.length);
```

## API Reference

### Functions

#### `probe_format(data: Uint8Array): WasmProbeResult`

Detects the container format from raw bytes.

**Parameters:**
- `data`: At least the first 12 bytes of the file

**Returns:** `WasmProbeResult` with format and confidence

**Throws:** Exception if format cannot be detected

### Classes

#### `WasmDemuxer`

Synchronous demuxer for extracting packets from containers.

**Constructor:**
- `new WasmDemuxer(data: Uint8Array)`

**Methods:**
- `probe(): WasmProbeResult` - Detect format and parse headers
- `streams(): WasmStreamInfo[]` - Get stream information
- `read_packet(): WasmPacket | null` - Read next packet (null at EOF)
- `size(): number` - Total size in bytes
- `position(): number` - Current position in bytes
- `is_eof(): boolean` - Check if all packets have been read

#### `WasmProbeResult`

Result of format probing.

**Methods:**
- `format(): string` - Container format name
- `confidence(): number` - Confidence score (0.0 to 1.0)
- `description(): string` - Human-readable description
- `is_video_container(): boolean` - Check if video container
- `is_audio_only(): boolean` - Check if audio-only container

#### `WasmStreamInfo`

Information about a media stream.

**Methods:**
- `index(): number` - Stream index
- `codec(): string` - Codec name
- `media_type(): string` - Media type ("Video", "Audio", "Subtitle")
- `is_video(): boolean` - Check if video stream
- `is_audio(): boolean` - Check if audio stream
- `duration_seconds(): number | undefined` - Duration in seconds
- `timebase_num(): number` - Timebase numerator
- `timebase_den(): number` - Timebase denominator
- `codec_params(): WasmCodecParams` - Codec parameters
- `metadata(): WasmMetadata` - Stream metadata

#### `WasmPacket`

Compressed media packet.

**Methods:**
- `stream_index(): number` - Stream index
- `size(): number` - Packet size in bytes
- `data(): Uint8Array` - Packet data
- `is_keyframe(): boolean` - Check if keyframe
- `is_corrupt(): boolean` - Check if potentially corrupt
- `pts(): number` - Presentation timestamp
- `dts(): number | undefined` - Decode timestamp
- `duration(): number | undefined` - Packet duration

## Supported Formats

### Containers (demux only -- packet extraction, no decoding)
- Matroska (.mkv)
- WebM (.webm)
- Ogg (.ogg, .opus, .oga)
- FLAC (.flac)
- WAV (.wav)
- MP4 (.mp4)

Demuxing a container extracts compressed packets and stream metadata
regardless of the codec inside; it does not imply this crate can decode
that codec (see below).

### Decoders (real, verified decode -> PCM/YUV)
- **Audio**: Opus (`WasmOpusDecoder`), FLAC (`WasmFlacDecoder`)
- **Video**: AV1, via `WasmMediaPlayer` only (no standalone `WasmAv1Decoder`
  class -- see [Removed decoders](#removed-decoders))

### Removed decoders

Earlier releases exposed standalone `WasmVp8Decoder`, `WasmAv1Decoder`, and
`WasmVorbisDecoder` classes. They were removed from the WASM surface because
they did not do what their names claimed:
- `WasmVp8Decoder` wrapped a VP8 decoder that returned an error on every
  `decode_frame()` call.
- `WasmAv1Decoder` wrapped an AV1 decoder that returned `Ok` with buffers
  that were never actually populated with decoded pixels.
- `WasmVorbisDecoder` wrapped a codec that only round-trips its own
  synthetic test format, not real Ogg Vorbis bitstreams.

Shipping classes that silently produce garbage or empty output is worse than
not shipping them. `WasmMediaPlayer` retains real AV1 decode internally; VP8
and Vorbis decode are not available anywhere in this crate today. This is a
WASM-surface-only change -- the underlying native codecs in
`crates/oximedia-codec` / `crates/oximedia-audio` are untouched by it and may
be fixed independently of this crate.

## License

Apache-2.0

Version: 0.2.0 — 2026-07-15 — data-plane, decoder-honesty, and dependency-hygiene pass

## See Also

- [OxiMedia](https://github.com/cool-japan/oximedia) - The main Rust library
- [`@cooljapan/oximedia-web`](../web/README.md) - Separate, size-budgeted WebCodecs-adjacent package (scopes/color/scale/quality)
- [wasm-bindgen](https://rustwasm.github.io/wasm-bindgen/) - Rust/WASM interop
- [wasm-pack](https://rustwasm.github.io/wasm-pack/) - WASM build tool

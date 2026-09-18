# OxiMedia Fuzzing Infrastructure

This directory contains fuzzing targets for the OxiMedia media processing library. Fuzzing helps discover bugs, panics, infinite loops, and security vulnerabilities by testing with randomized and malformed inputs.

## Overview

The fuzzing infrastructure tests:

### Container Parsers (7 targets)
- **matroska_parser** - Matroska/WebM demuxer (EBML, clusters, lacing)
- **ogg_parser** - Ogg container demuxer (page parsing, CRC, stream demux)
- **flac_parser** - FLAC container demuxer (metadata, frame headers)
- **mp4_parser** - MP4/ISOBMFF demuxer (boxes, sample tables)
- **wav_parser** - WAV/RIFF demuxer (chunk parsing)
- **y4m_parser** - Y4M/YUV4MPEG2 demuxer (stream header, frame reads)
- **aaf_parser** - AAF reader (CFB structured storage, object model, essence)

### Image Formats (3 targets)
- **tiff_parser** - TIFF reader (IFDs, strips/tiles, BigTIFF, compression)
- **exr_parser** - OpenEXR reader (attributes, scanline/tiled, multi-part)
- **jpegxl_decoder** - JPEG-XL decoder (codestream + ISOBMFF, modular/ANS, animation)

### Subtitle / Caption Parsers (4 targets)
- **srt_parser** - SubRip parser (cue blocks, timestamps, formatting tags)
- **ass_parser** - ASS/SSA parser (sections, format lines, override tags)
- **webvtt_parser** - WebVTT parser (cues, settings, NOTE/STYLE/REGION)
- **ttml_parser** - TTML/IMSC parser (XML, time expressions, TTML2 documents)

### Video Codecs (5 targets)
- **av1_decoder** - AV1 decoder (OBU parsing, tiles, entropy decoding)
- **ffv1_decoder** - FFV1 decoder (config record, range coder, slices)
- **vp9_decoder** - VP9 decoder (superframes, probability updates)
- **vp8_decoder** - VP8 decoder (boolean decoder, partitions)
- **theora_decoder** - Theora decoder (VP3-based, DCT, Huffman coding)

### Audio Codecs (3 targets)
- **opus_decoder** - Opus decoder (SILK, CELT, hybrid modes)
- **vorbis_decoder** - Vorbis decoder (codebooks, floors, residues)
- **flac_decoder** - FLAC audio decoder (subframes, Rice coding)

### Parsers (1 target)
- **ebml_parser** - EBML parser (VINT encoding, element parsing)

### Network Protocols (3 targets)
- **hls_parser** - HLS M3U8 playlist parser (master/media playlists)
- **dash_parser** - DASH MPD parser (XML, periods, representations)
- **rtmp_parser** - RTMP protocol parser (AMF0, chunk headers)

Total: **26 fuzz targets**

## Requirements

Install cargo-fuzz (already done if you're reading this):

```bash
cargo install cargo-fuzz
```

## Running Fuzzers

### Run a specific target

```bash
cargo fuzz run matroska_parser
```

### Run with a specific corpus directory

```bash
cargo fuzz run matroska_parser corpus/matroska
```

### Run with timeout (recommended for CI)

```bash
cargo fuzz run matroska_parser -- -max_total_time=600  # 10 minutes
```

### Run all targets sequentially

```bash
for target in $(cargo fuzz list); do
    echo "Running $target..."
    cargo fuzz run $target -- -max_total_time=300
done
```

### Use the CI fuzzing script

The repository includes a CI-ready script that runs all fuzzers:

```bash
./scripts/ci-fuzz.sh
```

This script will:
- Run all fuzz targets with appropriate dictionaries
- Set time and memory limits
- Report any crashes found
- Exit with error if issues are detected

## Reproducing Crashes

If a fuzzer finds a crash, it will save the input to:
```
fuzz/artifacts/<target>/crash-<hash>
```

To reproduce:

```bash
cargo fuzz run <target> fuzz/artifacts/<target>/crash-<hash>
```

## Corpus Management

### Adding seed files

Place valid media files in the corpus directories:
```
fuzz/corpus/matroska/
fuzz/corpus/ogg/
fuzz/corpus/flac/
fuzz/corpus/mp4/
fuzz/corpus/wav/
fuzz/corpus/av1/
fuzz/corpus/vp9/
fuzz/corpus/vp8/
fuzz/corpus/opus/
fuzz/corpus/vorbis/
```

Good seed files help the fuzzer explore more code paths efficiently.

### Minimizing corpus

After fuzzing, minimize the corpus to remove redundant inputs:

```bash
cargo fuzz cmin matroska_parser
```

This keeps only the inputs that provide unique coverage.

### Merging corpora

If you have multiple corpus directories, merge them:

```bash
cargo fuzz cmin matroska_parser corpus/matroska corpus/matroska-extra
```

## Advanced Options

### Dictionary files

The repository includes pre-built dictionaries for format-aware fuzzing in `dictionaries/`:

- `matroska.dict` - EBML element IDs and patterns
- `ogg.dict` - Ogg page sync patterns and codec headers
- `av1.dict` - AV1 OBU headers and common patterns
- `hls.dict` - HLS M3U8 tags and attributes
- `dash.dict` - DASH MPD XML elements and attributes
- `rtmp.dict` - RTMP AMF markers and chunk patterns

Use them with the `-dict` flag:

```bash
cargo fuzz run matroska_parser -- -dict=dictionaries/matroska.dict
cargo fuzz run hls_parser -- -dict=dictionaries/hls.dict
cargo fuzz run rtmp_parser -- -dict=dictionaries/rtmp.dict
```

The CI script automatically uses the appropriate dictionary for each target.

### Coverage reporting

Generate coverage reports:

```bash
cargo fuzz coverage matroska_parser
```

### Parallel fuzzing

Run multiple fuzzer instances in parallel:

```bash
# Terminal 1
cargo fuzz run matroska_parser -- -jobs=4

# Or use separate processes with shared corpus
cargo fuzz run matroska_parser -- -fork=4
```

### Continuous fuzzing

Use the continuous fuzzing script to run all targets in parallel:

```bash
./scripts/continuous-fuzz.sh
```

This will:
- Start all fuzz targets in parallel with fork mode
- Write logs to `logs/` directory
- Run indefinitely until stopped with Ctrl+C
- Use appropriate dictionaries for each target

For single-target long-term fuzzing with screen/tmux:

```bash
screen -S fuzzing
cargo fuzz run matroska_parser
# Ctrl+A, D to detach
```

## CI Integration

The repository includes CI-ready scripts in the `scripts/` directory:

### CI Fuzzing Script

`scripts/ci-fuzz.sh` runs all fuzz targets for a limited time and reports results:

```bash
# Run with defaults (5 minutes per target, 2GB memory)
./scripts/ci-fuzz.sh

# Customize settings with environment variables
FUZZ_TIME=600 MEMORY_LIMIT=4096 JOBS=2 ./scripts/ci-fuzz.sh
```

### GitHub Actions Example

```yaml
name: Fuzzing

on:
  schedule:
    - cron: '0 0 * * *'  # Daily
  workflow_dispatch:

jobs:
  fuzz:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-toolchain@nightly

      - name: Install cargo-fuzz
        run: cargo install cargo-fuzz

      - name: Run fuzzing suite
        run: |
          cd fuzz
          ./scripts/ci-fuzz.sh
        env:
          FUZZ_TIME: 600
          MEMORY_LIMIT: 4096

      - name: Upload artifacts
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: fuzz-artifacts
          path: fuzz/artifacts/
```

### GitLab CI Example

```yaml
fuzz:
  image: rust:latest
  script:
    - cargo install cargo-fuzz
    - cd fuzz
    - ./scripts/ci-fuzz.sh
  variables:
    FUZZ_TIME: "600"
    MEMORY_LIMIT: "4096"
  artifacts:
    when: on_failure
    paths:
      - fuzz/artifacts/
  only:
    - schedules
```

## Performance Tips

1. **Use release builds** - Fuzzing uses release builds by default for speed

2. **Limit iterations** - The fuzz targets limit internal loops to prevent hanging on malformed inputs

3. **Set memory limits**:
   ```bash
   cargo fuzz run matroska_parser -- -rss_limit_mb=2048
   ```

4. **Use persistent mode** - libfuzzer-sys uses persistent mode by default for better performance

5. **Adjust jobs** - Use `-jobs=N` or `-fork=N` for parallel fuzzing

## Troubleshooting

### Fuzzer hangs

If a fuzzer appears to hang, it may have found an infinite loop:
- Press Ctrl+C
- Check `fuzz/artifacts/<target>/timeout-*` for the input
- Add timeout limits: `-timeout=10` (10 seconds per input)

### Out of memory

Reduce memory per input:
```bash
cargo fuzz run matroska_parser -- -rss_limit_mb=1024
```

### Slow fuzzing

- Ensure seed corpus has valid files
- Check that fuzz target isn't doing excessive work
- Use dictionary files for format-aware fuzzing
- Run with more jobs: `-jobs=4`

## Utility Scripts

The `scripts/` directory contains helpful utilities:

### Coverage Report

Generate code coverage reports to see what's being tested:

```bash
./scripts/coverage-report.sh <target>

# Example
./scripts/coverage-report.sh matroska_parser
```

This generates HTML coverage reports in `coverage/html/`.

### Crash Minimization

Minimize crash inputs to the smallest reproducing case:

```bash
./scripts/minimize-crashes.sh <target>

# Example
./scripts/minimize-crashes.sh av1_decoder
```

Minimized inputs are saved to `minimized/<target>/`.

## Adding New Targets

To add a new fuzz target:

1. Create the target file:
   ```bash
   vi fuzz/fuzz_targets/new_target.rs
   ```

2. Add to `fuzz/Cargo.toml`:
   ```toml
   [[bin]]
   name = "new_target"
   path = "fuzz_targets/new_target.rs"
   test = false
   doc = false
   bench = false
   ```

3. (Optional) Create a dictionary:
   ```bash
   vi fuzz/dictionaries/new_target.dict
   ```

4. Create corpus directory:
   ```bash
   mkdir -p fuzz/corpus/new_target
   ```

5. Run the new target:
   ```bash
   cargo fuzz run new_target
   ```

## Security Policy

All crashes found by fuzzing should be:
1. Reproduced with the crash input
2. Fixed in the codebase
3. Added to regression tests
4. Added to corpus (if valid) to prevent regression

Critical issues (panics, overflows, infinite loops) must be fixed before release.

## Fuzzing Strategies

The OxiMedia fuzzing infrastructure uses multiple strategies:

### Structure-aware fuzzing

- Container parsers use dictionaries with format-specific magic bytes and element IDs
- Network protocol parsers include common tags and attribute patterns
- This helps the fuzzer generate valid-ish inputs that exercise deeper code paths

### Grammar-based fuzzing

- Protocol parsers (HLS, DASH, RTMP) benefit from grammar-aware mutations
- The fuzzer learns the structure through the corpus and dictionaries
- Invalid syntax is still tested to ensure robust error handling

### Mutation-based fuzzing

- All targets use libfuzzer's default mutation strategies
- Bit flips, byte insertions, arithmetic mutations, etc.
- Effective for finding edge cases and unexpected behavior

### Corpus management

- Seed corpus with valid samples for better coverage
- Run `cargo fuzz cmin` regularly to minimize redundancy
- Share corpus across similar targets (e.g., different container formats)

## Best Practices

1. **Run fuzzers regularly** - Set up CI to run fuzzers on every commit or nightly
2. **Monitor coverage** - Use coverage reports to find untested code paths
3. **Minimize crashes** - Always minimize crash inputs before debugging
4. **Update dictionaries** - Add new format patterns as the code evolves
5. **Share corpus** - Contribute interesting inputs back to the corpus
6. **Fix all crashes** - Treat fuzzer-found crashes as critical bugs

## Resources

- [cargo-fuzz documentation](https://rust-fuzz.github.io/book/cargo-fuzz.html)
- [libFuzzer documentation](https://llvm.org/docs/LibFuzzer.html)
- [Fuzzing best practices](https://google.github.io/oss-fuzz/getting-started/new-project-guide/)
- [The Fuzzing Book](https://www.fuzzingbook.org/)
- [AFL++ documentation](https://aflplus.plus/)

## AFL++ Support (Future)

For even better coverage, AFL++ can be integrated:

```bash
cargo install afl
cargo afl build --release
cargo afl fuzz -i corpus/matroska -o findings target/release/matroska_parser
```

This provides different mutation strategies and may find bugs that libfuzzer misses.

## License

The fuzzing infrastructure follows the same license as OxiMedia (Apache-2.0).

# OxiMedia CLI Examples

These scripts demonstrate end-to-end workflow chains using `oximedia` and
`oximedia-cv2`. Each script is self-contained — copy, adapt, and integrate
into your own pipelines.

## Prerequisites

- `oximedia` and `oximedia-cv2` on `PATH`. Install via:
  ```
  cargo install --path /path/to/oximedia/oximedia-cli
  ```
  or symlink `target/release/oximedia` and `target/release/oximedia-cv2`
  into a directory on your `PATH`.
- A POSIX-compatible shell. Scripts use Bash 4-style features
  (`mapfile`, arrays); they are tested with `/usr/bin/env bash`.

## Scripts

| Script                     | What it does                                                                            |
|----------------------------|-----------------------------------------------------------------------------------------|
| `dailies-ingest.sh`        | Camera-to-MAM ingest: probe, metadata, thumbnail, sprite, proxy, register.              |
| `quality-check.sh`         | Transcode -> compare via PSNR/SSIM/MS-SSIM/VMAF; emit JSON report.                      |
| `abr-package.sh`           | Generate 1080p/720p/480p ladder; package as HLS-fMP4 + DASH.                            |
| `loudness-normalize.sh`    | Batch EBU R128 normalize a directory; verify post-normalize compliance.                 |
| `forensics-investigate.sh` | Tamper / dedup-hash / watermark forensic chain on a single asset.                       |
| `restore-degraded.sh`      | Analyze, denoise, stabilize, upscale a degraded asset.                                  |
| `live-broadcast.sh`        | Live switcher session + scheduled playout server.                                       |
| `cv2-pipeline.sh`          | OpenCV cv2 chain: imread -> cvt-color -> gaussian-blur -> canny -> probe.               |

## Running

```
chmod +x examples/*.sh
./examples/dailies-ingest.sh /path/to/file.mxf
```

Each script's header documents the environment variables it honours; pass
them inline:

```
PROXY_QUALITY=high PROXY_RESOLUTION=half ./examples/dailies-ingest.sh raw.mxf
```

## Notes

- All scripts use `set -euo pipefail` for fail-fast behaviour.
- `--json` and `--output-format json` are preferred where the subcommand
  exposes them; the global `--json` flag is used elsewhere.
- Long-running operations stream progress to `stderr`; final status to
  `stdout`. Capture machine-parseable output with shell redirection.
- These scripts are reference material — they are intentionally not part
  of any test target.

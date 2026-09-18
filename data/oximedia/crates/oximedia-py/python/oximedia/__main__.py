"""oximedia CLI entry point.

Usage: ``python -m oximedia <subcommand> [args...]``

Mirrors the most common operations from oximedia-cli, dispatching into the
oximedia Python bindings.  Subcommands:

  - probe     - print metadata for a media file (uses ``oximedia.io.probe``)
  - transcode - transcode input to output (uses ``oximedia.transcode_simple``)
  - quality   - compute PSNR/SSIM on two videos by decoding the first frame
                of each and running the bytes-based assessor.  This is a
                deliberate single-frame approximation — the streaming
                bytes-level assessor is the authoritative API.
  - cv2       - cv2 sub-operations (cvt-color, imread/imwrite passthrough)
  - presets   - list known transcoding presets
  - codecs    - list registered codecs
  - version   - print the installed package version

All subcommands return ``0`` on success, ``1`` on a runtime failure of the
underlying API, and ``2`` when an API surface is missing on this build.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m oximedia",
        description="OxiMedia Python bindings - command-line interface",
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="<subcommand>")

    # probe -----------------------------------------------------------------
    probe = sub.add_parser("probe", help="Print metadata for a media file")
    probe.add_argument("input", type=Path, help="Path to media file")
    probe.add_argument(
        "--json",
        action="store_true",
        help="Emit the probe result as JSON instead of repr text",
    )

    # transcode -------------------------------------------------------------
    transcode = sub.add_parser(
        "transcode",
        help="Transcode an input file using oximedia.transcode_simple",
    )
    transcode.add_argument("input", type=Path)
    transcode.add_argument("output", type=Path)
    transcode.add_argument(
        "--preset",
        default=None,
        help="Quality preset name (e.g. youtube-1080p, netflix-4k)",
    )
    transcode.add_argument(
        "--crf",
        type=int,
        default=None,
        help="CRF value passed through to the encoder",
    )

    # quality ---------------------------------------------------------------
    quality = sub.add_parser(
        "quality",
        help="Compute PSNR/SSIM between two videos (single-frame approximation)",
    )
    quality.add_argument("reference", type=Path)
    quality.add_argument("distorted", type=Path)
    quality.add_argument(
        "--metric",
        choices=["psnr", "ssim", "all"],
        default="all",
    )

    # cv2 -------------------------------------------------------------------
    cv2_p = sub.add_parser("cv2", help="cv2 operations (cvt-color, convert)")
    cv2_sub = cv2_p.add_subparsers(dest="cv2_cmd", required=True)

    cvt = cv2_sub.add_parser(
        "cvt-color",
        help="Read an image, convert colour space, write the result",
    )
    cvt.add_argument("input", type=Path)
    cvt.add_argument("output", type=Path)
    cvt.add_argument(
        "--code",
        required=True,
        help="Conversion code suffix (e.g. BGR2RGB -> COLOR_BGR2RGB)",
    )

    convert = cv2_sub.add_parser(
        "convert",
        help="cv2.imread followed by cv2.imwrite (format conversion)",
    )
    convert.add_argument("input", type=Path)
    convert.add_argument("output", type=Path)

    # presets / codecs ------------------------------------------------------
    sub.add_parser("presets", help="List the known transcoding presets")
    sub.add_parser("codecs", help="List registered codecs")

    # version ---------------------------------------------------------------
    sub.add_parser("version", help="Print the installed package version")

    return parser


# ─── dispatch handlers ─────────────────────────────────────────────────────


def _import_oximedia():
    """Lazy import so ``--help`` works even if the extension is missing."""
    try:
        import oximedia  # noqa: WPS433 — runtime import is intentional
    except ImportError as exc:  # pragma: no cover - exercised in failing builds
        print(
            f"failed to import oximedia: {exc}\n"
            "Hint: install the wheel with `pip install oximedia` or build with "
            "`maturin develop` from crates/oximedia-py.",
            file=sys.stderr,
        )
        return None
    return oximedia


def cmd_probe(args: argparse.Namespace) -> int:
    oximedia = _import_oximedia()
    if oximedia is None:
        return 2
    try:
        info = oximedia.io.probe(str(args.input))
    except Exception as exc:  # noqa: BLE001 — surface every failure
        print(f"probe failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        import json

        try:
            payload = info.to_dict()
        except AttributeError:
            payload = {"repr": repr(info)}
        print(json.dumps(payload, indent=2, default=str))
        return 0

    print(info)
    return 0


def cmd_transcode(args: argparse.Namespace) -> int:
    oximedia = _import_oximedia()
    if oximedia is None:
        return 2
    fn = getattr(oximedia, "transcode_simple", None)
    if fn is None:
        print(
            "transcode API not available: oximedia.transcode_simple missing",
            file=sys.stderr,
        )
        return 2
    try:
        result = fn(
            str(args.input),
            str(args.output),
            preset=args.preset,
            crf=args.crf,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"transcode failed: {exc}", file=sys.stderr)
        return 1
    print(result)
    return 0


def _decode_first_y_plane(oximedia, path: str):
    """Decode one frame from ``path`` and return ``(y_bytes, width, height)``.

    Uses ``oximedia.io.open_video`` (returns a streaming ``MediaReader``).
    """
    reader = oximedia.io.open_video(path, max_frames=1)
    try:
        frame = next(iter(reader))
    except StopIteration as exc:
        raise RuntimeError(f"no frames decoded from {path}") from exc
    finally:
        close = getattr(reader, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 — best effort
                pass

    width = int(frame.width)
    height = int(frame.height)
    y_bytes = frame.plane_data(0)
    return y_bytes, width, height


def cmd_quality(args: argparse.Namespace) -> int:
    oximedia = _import_oximedia()
    if oximedia is None:
        return 2

    try:
        ref_data, ref_w, ref_h = _decode_first_y_plane(oximedia, str(args.reference))
        dist_data, dist_w, dist_h = _decode_first_y_plane(oximedia, str(args.distorted))
    except Exception as exc:  # noqa: BLE001
        print(f"quality decode failed: {exc}", file=sys.stderr)
        return 1

    if (ref_w, ref_h) != (dist_w, dist_h):
        print(
            f"quality dimensions differ: reference={ref_w}x{ref_h} "
            f"distorted={dist_w}x{dist_h}",
            file=sys.stderr,
        )
        return 1

    metric = args.metric
    psnr_fn = getattr(oximedia, "compute_psnr", None)
    ssim_fn = getattr(oximedia, "compute_ssim", None)
    if metric in ("psnr", "all") and psnr_fn is None:
        print("quality API not available: oximedia.compute_psnr missing", file=sys.stderr)
        return 2
    if metric in ("ssim", "all") and ssim_fn is None:
        print("quality API not available: oximedia.compute_ssim missing", file=sys.stderr)
        return 2

    try:
        if metric in ("psnr", "all"):
            psnr_value = psnr_fn(ref_data, dist_data, ref_w, ref_h)
            print(f"psnr={psnr_value:.4f} dB")
        if metric in ("ssim", "all"):
            ssim_value = ssim_fn(ref_data, dist_data, ref_w, ref_h)
            print(f"ssim={ssim_value:.6f}")
    except Exception as exc:  # noqa: BLE001
        print(f"quality computation failed: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_cv2(args: argparse.Namespace) -> int:
    oximedia = _import_oximedia()
    if oximedia is None:
        return 2
    cv2 = getattr(oximedia, "cv2", None)
    if cv2 is None:
        print("cv2 submodule not available on this build", file=sys.stderr)
        return 2

    try:
        if args.cv2_cmd == "cvt-color":
            mat = cv2.imread(str(args.input))
            if mat is None:
                print(f"cv2.imread returned None for {args.input}", file=sys.stderr)
                return 1
            const_name = f"COLOR_{args.code}"
            code = getattr(cv2, const_name, None)
            if code is None:
                print(f"unknown cv2 constant: {const_name}", file=sys.stderr)
                return 2
            converted = cv2.cvtColor(mat, code)
            cv2.imwrite(str(args.output), converted)
        elif args.cv2_cmd == "convert":
            mat = cv2.imread(str(args.input))
            if mat is None:
                print(f"cv2.imread returned None for {args.input}", file=sys.stderr)
                return 1
            cv2.imwrite(str(args.output), mat)
        else:
            print(f"unknown cv2 subcommand: {args.cv2_cmd}", file=sys.stderr)
            return 2
    except Exception as exc:  # noqa: BLE001
        print(f"cv2 operation failed: {exc}", file=sys.stderr)
        return 1

    print(f"OK: {args.input} -> {args.output}")
    return 0


def cmd_presets(args: argparse.Namespace) -> int:
    oximedia = _import_oximedia()
    if oximedia is None:
        return 2
    fn = getattr(oximedia, "list_presets", None)
    if fn is None:
        print("list_presets not available on this build", file=sys.stderr)
        return 2
    try:
        presets = fn()
    except Exception as exc:  # noqa: BLE001
        print(f"list_presets failed: {exc}", file=sys.stderr)
        return 1
    for entry in presets:
        print(entry)
    return 0


def cmd_codecs(args: argparse.Namespace) -> int:
    oximedia = _import_oximedia()
    if oximedia is None:
        return 2
    fn = getattr(oximedia, "list_codecs", None)
    if fn is None:
        # Fall back to the io-level helper, which is documented in io.pyi.
        io_mod = getattr(oximedia, "io", None)
        fn = getattr(io_mod, "list_supported_codecs", None) if io_mod else None
    if fn is None:
        print("codec listing API not available on this build", file=sys.stderr)
        return 2
    try:
        codecs = fn()
    except Exception as exc:  # noqa: BLE001
        print(f"codec listing failed: {exc}", file=sys.stderr)
        return 1
    for entry in codecs:
        print(entry)
    return 0


def cmd_version(args: argparse.Namespace) -> int:
    # Try the installed-distribution metadata first; fall back to a module
    # attribute (some PyO3 builds expose it explicitly) and finally "unknown".
    version: str = "unknown"
    try:
        from importlib.metadata import PackageNotFoundError, version as _md_version

        try:
            version = _md_version("oximedia")
        except PackageNotFoundError:
            version = "unknown"
    except ImportError:  # pragma: no cover - importlib.metadata always present in 3.10+
        pass

    if version == "unknown":
        oximedia = _import_oximedia()
        if oximedia is not None:
            version = getattr(oximedia, "__version__", "unknown")

    print(f"oximedia {version}")
    return 0


_DISPATCH = {
    "probe": cmd_probe,
    "transcode": cmd_transcode,
    "quality": cmd_quality,
    "cv2": cmd_cv2,
    "presets": cmd_presets,
    "codecs": cmd_codecs,
    "version": cmd_version,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = _DISPATCH.get(args.cmd)
    if handler is None:
        parser.error(f"unknown subcommand: {args.cmd}")
        return 2
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())

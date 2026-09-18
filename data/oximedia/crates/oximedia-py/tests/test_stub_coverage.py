"""Verify that ``.pyi`` type stubs are syntactically valid Python and present.

These tests do NOT require a built wheel — they parse the stub files directly
from the source tree.  They guard against accidental syntax breakage and
ensure the expected stub artefacts exist for downstream packaging.
"""
from __future__ import annotations

import ast
import pathlib

import pytest


_STUBS_DIR = (
    pathlib.Path(__file__).resolve().parent.parent / "python" / "oximedia"
)


@pytest.mark.parametrize(
    "pyi_path",
    sorted(_STUBS_DIR.glob("*.pyi")),
    ids=lambda p: p.name if hasattr(p, "name") else str(p),
)
def test_stub_parses(pyi_path: pathlib.Path):
    """Every shipped ``.pyi`` stub must be valid Python syntax."""
    source = pyi_path.read_text()
    # ``ast.parse`` raises ``SyntaxError`` on bad stubs — that's the failure
    # we want to surface here.
    ast.parse(source, filename=str(pyi_path))


def test_stubs_present():
    """Required stubs and the ``py.typed`` PEP 561 marker file exist."""
    expected = [
        "__init__.pyi",
        "cv2.pyi",
        "io.pyi",
        "logging.pyi",
        "utils.pyi",
        "test.pyi",
        "benchmark.pyi",
        "py.typed",
    ]
    missing = [name for name in expected if not (_STUBS_DIR / name).exists()]
    assert not missing, f"missing stub files: {missing}"


def test_init_stub_non_trivial():
    """The top-level ``__init__.pyi`` should contain a substantial API surface.

    Slice D documented this file at ~1259 lines.  Allow a generous floor (200)
    so churn doesn't break the test, but flag if the file shrinks dramatically.
    """
    init = _STUBS_DIR / "__init__.pyi"
    line_count = sum(1 for _ in init.open())
    assert line_count >= 200, f"__init__.pyi shrank to {line_count} lines"


def test_init_stub_declares_core_classes():
    """The ``__init__.pyi`` declares core PyO3 classes via ``class`` nodes."""
    source = (_STUBS_DIR / "__init__.pyi").read_text()
    tree = ast.parse(source)
    class_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }
    # A reasonable smoke set — these are stable PyO3 types.
    expected = {"VideoFrame", "AudioFrame", "PixelFormat"}
    missing = expected - class_names
    assert not missing, f"__init__.pyi missing class declarations: {missing}"


def test_cv2_stub_parses_and_has_constants():
    """cv2.pyi parses and declares at least one of the OpenCV-style constants."""
    source = (_STUBS_DIR / "cv2.pyi").read_text()
    tree = ast.parse(source)
    # Look for module-level ``IMREAD_COLOR`` / ``THRESH_BINARY`` annotations.
    found = False
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in {"IMREAD_COLOR", "THRESH_BINARY", "COLOR_BGR2RGB"}:
                found = True
                break
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "IMREAD_COLOR",
                    "THRESH_BINARY",
                    "COLOR_BGR2RGB",
                }:
                    found = True
                    break
            if found:
                break
    assert found, "cv2.pyi declares no recognisable cv2 constant"


def test_test_stub_declares_synthetic_helpers():
    """``test.pyi`` declares the synthetic-media generators."""
    source = (_STUBS_DIR / "test.pyi").read_text()
    tree = ast.parse(source)
    func_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    expected = {
        "synthetic_video_frame",
        "synthetic_audio_frame",
        "checkerboard_frame",
    }
    missing = expected - func_names
    assert not missing, f"test.pyi missing synthetic helpers: {missing}"


def test_py_typed_marker_exists():
    """PEP 561 marker file ``py.typed`` exists in the stub package.

    The PEP only requires the file's presence; some projects ship comments
    in it for human readers, others use the bare ``partial`` marker.  We
    only assert existence here.
    """
    py_typed = _STUBS_DIR / "py.typed"
    assert py_typed.exists()

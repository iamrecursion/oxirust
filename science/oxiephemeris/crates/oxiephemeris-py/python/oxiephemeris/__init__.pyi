"""Type stubs for the `oxiephemeris` native extension.

The implementation is the compiled Rust module; these signatures document
the API and give editors/type-checkers something to work with.
"""

from typing import Any, Optional, Protocol

__version__: str

class _Person(Protocol):
    date: str
    lat: float
    lon: float

def julday(
    year: int,
    month: int,
    day: int,
    hours: float = 0.0,
    calendar: str = "gregorian",
) -> float:
    """Julian Date of a calendar date/time."""

def revjul(jd: float, calendar: str = "gregorian") -> dict[str, Any]:
    """Calendar date/time of a Julian Date: ``{year, month, day, hours}``."""

def natal(
    de: bytes,
    date: str,
    lat: float,
    lon: float,
    *,
    alt: float = 0.0,
    dut1: float = 0.0,
    cal: str = "gregorian",
    system: str = "placidus",
    sidereal: Optional[str] = None,
    rulership: str = "traditional",
) -> dict[str, Any]:
    """Compute a natal chart; returns the stable JSON view as a dict."""

def natal_rdf(
    de: bytes,
    date: str,
    lat: float,
    lon: float,
    *,
    alt: float = 0.0,
    dut1: float = 0.0,
    cal: str = "gregorian",
    system: str = "placidus",
    sidereal: Optional[str] = None,
    rulership: str = "traditional",
    base_iri: Optional[str] = None,
    ntriples: bool = False,
) -> str:
    """Compute a natal chart; returns RDF Turtle (or N-Triples)."""

def synastry(
    de: bytes,
    a: _Person,
    b: _Person,
    *,
    system: str = "placidus",
    dut1: float = 0.0,
    cal: str = "gregorian",
) -> dict[str, Any]: ...
def synastry_rdf(
    de: bytes,
    a: _Person,
    b: _Person,
    *,
    system: str = "placidus",
    dut1: float = 0.0,
    cal: str = "gregorian",
    base_iri: Optional[str] = None,
    ntriples: bool = False,
) -> str: ...
def transit(
    de: bytes,
    natal: _Person,
    transit: str,
    *,
    system: str = "placidus",
    dut1: float = 0.0,
    cal: str = "gregorian",
) -> dict[str, Any]: ...
def progress(
    de: bytes,
    natal: _Person,
    target: str,
    *,
    system: str = "placidus",
    dut1: float = 0.0,
    cal: str = "gregorian",
) -> dict[str, Any]: ...
def composite(
    de: bytes,
    a: _Person,
    b: _Person,
    *,
    system: str = "placidus",
    dut1: float = 0.0,
    cal: str = "gregorian",
    base_iri: Optional[str] = None,
) -> dict[str, Any]: ...
def composite_rdf(
    de: bytes,
    a: _Person,
    b: _Person,
    *,
    system: str = "placidus",
    dut1: float = 0.0,
    cal: str = "gregorian",
    base_iri: Optional[str] = None,
    ntriples: bool = False,
) -> str: ...

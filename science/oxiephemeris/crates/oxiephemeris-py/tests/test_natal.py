"""Python-side verification of the OxiEphemeris bindings.

Requires the extension to be built (`maturin develop`) and the DE440
fixture at data/de440/linux_p1550p2650.440; skips cleanly if the fixture
is absent.
"""

import os

import pytest

oxi = pytest.importorskip("oxiephemeris")

_DE440 = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "de440", "linux_p1550p2650.440"
)


def _de():
    if not os.path.exists(_DE440):
        pytest.skip(f"DE440 fixture not present at {_DE440}")
    with open(_DE440, "rb") as f:
        return f.read()


class _P:
    """A person object with the date/lat/lon the pair APIs read."""

    def __init__(self, date, lat, lon):
        self.date = date
        self.lat = lat
        self.lon = lon


# The synthetic reference chart used throughout: the Unix epoch at the
# Royal Observatory, Greenwich. A companion chart for the pair APIs is the
# Y2K noon at Paris. Both are public reference points, not personal data.
_EPOCH = "1970-01-01T00:00:00Z"
_GREENWICH = dict(lat=51.4779, lon=0.0)
_Y2K = "2000-01-01T12:00:00Z"
_PARIS = dict(lat=48.8566, lon=2.3522)


def test_version_is_exposed():
    assert isinstance(oxi.__version__, str)
    assert oxi.__version__


def test_julday_revjul_round_trip():
    jd = oxi.julday(1970, 1, 1, 0.0)
    back = oxi.revjul(jd)
    assert back["year"] == 1970
    assert back["month"] == 1
    assert back["day"] == 1
    assert abs(back["hours"] - 0.0) < 1e-6


def test_natal_matches_reference_chart():
    de = _de()
    chart = oxi.natal(de, _EPOCH, **_GREENWICH)

    assert chart["kind"] == "natal"
    assert chart["sect"] == "nocturnal"

    sun = next(b for b in chart["bodies"] if b["body"] == "Sun")
    assert sun["sign"] == "Capricorn"
    assert abs(sun["sign_degrees"] - 10.156) < 0.01
    assert sun["house"] == 4
    assert sun["retrograde"] is False

    mars = next(d for d in chart["dignities"] if d["body"] == "Mars")
    assert mars["score"] == 3
    assert mars["triplicity"] and not mars["domicile"]

    # Part of Fortune ~ 6 deg 40' Capricorn.
    assert 276.0 < chart["lots"]["fortune_deg"] < 277.0

    dist = chart["distribution"]
    assert dist["earth"] == 5
    assert dist["fire"] + dist["earth"] + dist["air"] + dist["water"] == 10


def test_natal_rdf_is_turtle():
    de = _de()
    turtle = oxi.natal_rdf(de, _EPOCH, **_GREENWICH)
    assert "@prefix oxa:" in turtle
    assert "oxa:inSign sign:Capricorn" in turtle
    assert "oxa:dignityScore 3" in turtle
    # N-Triples variant is a different, prefix-free serialization.
    nt = oxi.natal_rdf(de, _EPOCH, **_GREENWICH, ntriples=True)
    assert "@prefix" not in nt
    assert "concept/sign/Capricorn" in nt


def test_sidereal_shifts_the_sign():
    de = _de()
    tropical = oxi.natal(de, _EPOCH, **_GREENWICH)
    lahiri = oxi.natal(de, _EPOCH, **_GREENWICH, sidereal="lahiri")
    t_sun = next(b for b in tropical["bodies"] if b["body"] == "Sun")
    l_sun = next(b for b in lahiri["bodies"] if b["body"] == "Sun")
    # Lahiri ayanamsha ~23 deg in 1970 pulls the Sun back across the
    # sign boundary: 10.2 deg Capricorn -> 16.7 deg Sagittarius.
    assert t_sun["sign"] == "Capricorn"
    assert abs(t_sun["sign_degrees"] - 10.156) < 0.05
    assert l_sun["sign"] == "Sagittarius"
    assert abs(l_sun["sign_degrees"] - 16.71) < 0.05
    assert lahiri["sidereal"] == "lahiri"
    assert 23.0 < lahiri["ayanamsha_deg"] < 24.0


def test_synastry_and_composite():
    de = _de()
    a = _P(_EPOCH, **_GREENWICH)
    b = _P(_Y2K, **_PARIS)

    syn = oxi.synastry(de, a, b)
    assert syn["kind"] == "synastry"
    assert len(syn["chart_a"]) == 12  # ten planets + ASC + MC
    assert len(syn["cross_aspects"]) > 0

    turtle = oxi.synastry_rdf(de, a, b)
    assert "oxa:SynastryComparison" in turtle

    comp = oxi.composite(de, a, b)
    assert len(comp["composite"]) == 12
    assert len(comp["source_charts"]) == 2

    comp_ttl = oxi.composite_rdf(de, a, b)
    assert "oxa:CompositeChart" in comp_ttl
    assert "wasDerivedFrom" in comp_ttl


def test_transit_and_progress():
    de = _de()
    natal = _P(_EPOCH, **_GREENWICH)

    tr = oxi.transit(de, natal, "2026-07-10T00:00:00")
    assert tr["kind"] == "transit"
    assert len(tr["cross_aspects"]) > 0

    pr = oxi.progress(de, natal, "2010-01-01T00:00:00")
    assert pr["kind"] == "progression"
    assert pr["elapsed_years"] > 35.0


def test_bad_request_raises_valueerror():
    de = _de()
    with pytest.raises(ValueError):
        oxi.natal(de, _EPOCH, **_GREENWICH, system="martian")
    with pytest.raises(ValueError):
        oxi.natal(de, "not-a-date", **_GREENWICH)

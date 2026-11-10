"""Exact rational checks for constants in docs/sasa-filter-error-bound.md."""
from fractions import Fraction


def test_error_envelope_constants_have_strict_rounding_headroom():
    u = Fraction(1, 2**24)
    v = Fraction(1, 2**53)
    assert 5*u / (1-5*u) < 6*u
    assert 186*u + 120*v < 200*u
    lower = (1-u) * (1-20*v) * (1-5*v)**2
    assert 256*lower > 255
    assert 512*lower > 510
    assert 200*u < 255*u
    assert 64*v < 510*v
    assert 100*Fraction(1, 2**126) < 4*u

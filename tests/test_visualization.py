import pytest
import numpy as np

from htl_package.visualization import attr_to_rgb


class TestAttrToRgb:
    def test_positive_returns_red_tone(self):
        r, g, b = attr_to_rgb(1.0)
        assert r > 0.8
        assert g < 0.2
        assert b < 0.2

    def test_negative_returns_blue_tone(self):
        r, g, b = attr_to_rgb(-1.0)
        assert r < 0.2
        assert g < 0.5
        assert b > 0.6

    def test_zero_returns_white(self):
        r, g, b = attr_to_rgb(0.0)
        assert abs(r - 1.0) < 1e-6
        assert abs(g - 1.0) < 1e-6
        assert abs(b - 1.0) < 1e-6

    def test_clipping_above_one(self):
        r1, g1, b1 = attr_to_rgb(1.0)
        r2, g2, b2 = attr_to_rgb(5.0)
        assert abs(r1 - r2) < 1e-6
        assert abs(g1 - g2) < 1e-6
        assert abs(b1 - b2) < 1e-6

    def test_clipping_below_minus_one(self):
        r1, g1, b1 = attr_to_rgb(-1.0)
        r2, g2, b2 = attr_to_rgb(-5.0)
        assert abs(r1 - r2) < 1e-6
        assert abs(g1 - g2) < 1e-6
        assert abs(b1 - b2) < 1e-6

    def test_output_is_tuple_of_floats(self):
        result = attr_to_rgb(0.5)
        assert isinstance(result, tuple)
        assert len(result) == 3
        for v in result:
            assert isinstance(v, float)
            assert 0.0 <= v <= 1.0

    def test_symmetry(self):
        rp, gp, bp = attr_to_rgb(0.5)
        rn, gn, bn = attr_to_rgb(-0.5)
        assert abs(rp - rn) > 0.1

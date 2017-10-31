import pytest
import numpy as np


def test_simple_extrapolation():
    s = pd.Series({0:0, 1:1})
    f = ceam_experiments.cvd.components.opportunistic_screening.simple_extrapolation(s)

    assert f(0) == 0, 'should be precise at index values'
    assert f(1) == 1
    assert f(2) == 1, 'should be constant extrapolation outside of input range'
    assert f(-1) == 0

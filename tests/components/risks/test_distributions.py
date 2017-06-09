import numpy as np

from ceam_public_health.components.risks.distributions import _sll_ppf as sll_ppf


def test_sll_ppf():
    assert sll_ppf(0, 0, 1, 1) == -1
    for location in np.random.normal(size=100):
        assert sll_ppf(0.5, location=location, scale=2, shape=2) == location

    for location in np.random.normal(size=10):
        for scale in np.random.exponential(size=10):
            for shape in np.random.normal(size=10):
                assert np.isinf(sll_ppf(1., location, scale, shape))
                median = sll_ppf(0.5, location, scale, shape)
                assert np.allclose(median, location)

                assert median < sll_ppf(0.75, location, scale, shape)

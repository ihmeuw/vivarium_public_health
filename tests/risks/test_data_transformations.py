import pandas as pd
import pytest
from vivarium.interface.interactive import InteractiveContext

from vivarium_public_health.risks.base_risk import Risk
from vivarium_public_health.risks.data_transformations import (
    _rebin_relative_risk_data,
    get_relative_risk_data,
)
from vivarium_public_health.risks.distributions import DichotomousDistribution
from vivarium_public_health.risks.effect import RiskEffect
from vivarium_public_health.utilities import TargetString


@pytest.mark.parametrize(
    "rebin_categories, rebinned_values",
    [
        ({"cat1", "cat2"}, (0.7, 0.3)),
        ({"cat1"}, (0.5, 0.5)),
        ({"cat2"}, (0.2, 0.8)),
        ({"cat2", "cat3"}, (0.5, 0.5)),
        ({"cat1", "cat3"}, (0.8, 0.2)),
    ],
)
def test__rebin_exposure_data(rebin_categories, rebinned_values):
    df = pd.DataFrame(
        {
            "year": [1990, 1990, 1995, 1995] * 3,
            "age": [10, 40, 10, 40] * 3,
            "parameter": ["cat1"] * 4 + ["cat2"] * 4 + ["cat3"] * 4,
            "value": [0.5] * 4 + [0.2] * 4 + [0.3] * 4,
        }
    )
    rebinned_df = DichotomousDistribution._rebin_exposure_data(df, rebin_categories)

    assert rebinned_df.shape == (4, 4)
    assert (rebinned_df.value == rebinned_values[0]).all()
    assert (rebinned_df["parameter"] == "cat1").all()


@pytest.mark.parametrize(
    "rebin_categories, rebinned_values",
    [
        ({"cat1", "cat2"}, (10, 1)),
        ({"cat1"}, (0, 7.3)),
        ({"cat2"}, (10, 1)),
        ({"cat2", "cat3"}, (7.3, 0)),
        ({"cat1", "cat3"}, (1, 10)),
    ],
)
def test__rebin_relative_risk(rebin_categories, rebinned_values):
    exp = pd.DataFrame(
        {
            "year": [1990, 1990, 1995, 1995] * 3,
            "age": [10, 40, 10, 40] * 3,
            "parameter": ["cat1"] * 4 + ["cat2"] * 4 + ["cat3"] * 4,
            "value": [0.0] * 4 + [0.7] * 4 + [0.3] * 4,
        }
    )

    rr = pd.DataFrame(
        {
            "year": [1990, 1990, 1995, 1995] * 3,
            "age": [10, 40, 10, 40] * 3,
            "parameter": ["cat1"] * 4 + ["cat2"] * 4 + ["cat3"] * 4,
            "value": [5] * 4 + [10] * 4 + [1] * 4,
        }
    )

    rebinned_df = _rebin_relative_risk_data(rr, exp, rebin_categories)

    assert rebinned_df.shape == (8, 4)
    assert (rebinned_df[rebinned_df.parameter == "cat1"].value == rebinned_values[0]).all()
    assert (rebinned_df[rebinned_df.parameter == "cat2"].value == rebinned_values[1]).all()


def test__subset_relative_risk_to_empty_dataframe(base_config, base_plugins):
    target = TargetString("cause.test_cause.missing_measure")
    risk = Risk("risk_factor.risk_factor")
    risk_effect = RiskEffect("risk_factor.risk_factor", "cause.test_cause.missing_measure")

    sim = InteractiveContext(
        model_specification=None,
        components=[risk, risk_effect],
        configuration=base_config,
        plugin_configuration=base_plugins,
        setup=False,
    )
    sim.configuration.update(
        {
            f"effect_of_{risk.name}_on_{target.name}": {
                f"{target.measure}": {
                    "relative_risk": None,
                    "mean": None,
                    "se": None,
                    "log_mean": None,
                    "log_se": None,
                    "tau_squared": None,
                },
            },
        }
    )

    error_msg = f"Subsetting {risk_effect.risk} relative risk data to {target.name} {target.measure} returned an empty DataFrame. Check your artifact"
    with pytest.raises(ValueError, match=error_msg):
        get_relative_risk_data(
            sim._builder, risk_effect.risk, target, risk_effect._exposure_distribution_type
        )

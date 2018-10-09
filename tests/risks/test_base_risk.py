
import numpy as np
import pandas as pd
from scipy.stats import norm

from vivarium.testing_utilities import build_table, TestPopulation, metadata
from vivarium.interface.interactive import initialize_simulation


from vivarium_public_health.risks.base_risk import Risk


def make_dummy_column(name, initial_value):
    class _make_dummy_column:
        def setup(self, builder):
            self.population_view = builder.population.get_view([name])
            builder.population.initializes_simulants(self.make_column, creates_columns=[name])

        def make_column(self, pop_data):
            self.population_view.update(pd.Series(initial_value, index=pop_data.index, name=name))
    return _make_dummy_column()


def test_propensity_effect(mocker, base_config, base_plugins):
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year

    risk = "test_risk"

    exposure_data = build_table(
        0.5, year_start, year_end
    ).melt(id_vars=('age', 'year', 'sex'), var_name='parameter', value_name='value')

    affected_causes = ["test_cause_1", "test_cause_2"]

    rr_data = []
    for cause in affected_causes:
        rr_data.append(
            build_table([1.01, 'continuous', cause], year_start, year_end,
                        ('age', 'year', 'sex', 'value', 'parameter', 'cause'))
        )
    rr_data = pd.concat(rr_data)

    tmred = {
            "distribution": 'uniform',
            "min": 110.0,
            "max": 115.0,
            "inverted": False,
    }
    exposure_parameters = {
            "scale": 10.0,
            "max_rr": 200.0,
            "max_val": 300.0,
            "min_val": 50.0,
    }

    class Distribution:
        def __init__(self, *_, **__):
            pass

        def setup(self, builder):
            data = build_table([130, 15], year_start, year_end, ['age', 'year', 'sex', 'mean', 'std'])
            self.parameters = builder.lookup.build_table(data)

        def ppf(self, propensity):
            params = self.parameters(propensity.index)
            return norm(loc=params['mean'], scale=params['std']).ppf(propensity)

    get_distribution_mock = mocker.patch('vivarium_public_health.risks.base_risk.get_distribution')
    get_distribution_mock.side_effect = lambda *args, **kwargs: Distribution(args, kwargs)

    component = Risk("risk_factor", risk)

    base_config.update({'population': {'population_size': 100000}}, **metadata(__file__))
    simulation = initialize_simulation([TestPopulation(), component],
                                       input_config=base_config, plugin_config=base_plugins)
    simulation.data.write("risk_factor.test_risk.tmred", tmred)
    simulation.data.write("risk_factor.test_risk.exposure_parameters", exposure_parameters)
    simulation.data.write("risk_factor.test_risk.exposure", exposure_data)
    simulation.data.write("risk_factor.test_risk.relative_risk", rr_data)
    simulation.data.write("risk_factor.test_risk.population_attributable_fraction", 1)
    simulation.data.write("risk_factor.test_risk.affected_causes", affected_causes)
    simulation.data.write("risk_factor.test_risk.affected_risk_factors", [])
    simulation.data.write("risk_factor.test_risk.distribution", "ensemble")
    simulation.setup()

    propensity_pipeline = mocker.Mock()
    simulation.values.register_value_producer('test_risk.propensity', propensity_pipeline)
    propensity_pipeline.side_effect = lambda index: pd.Series(0.00001, index)

    simulation.step()

    expected_value = norm(loc=130, scale=15).ppf(0.00001)

    assert np.allclose(component.exposure(simulation.population.population.index), expected_value)

    propensity_pipeline.side_effect = lambda index: pd.Series(0.5, index)

    simulation.step()

    expected_value = 130
    assert np.allclose(component.exposure(simulation.population.population.index), expected_value)

    propensity_pipeline.side_effect = lambda index: pd.Series(0.99999, index)
    simulation.step()

    expected_value = norm(loc=130, scale=15).ppf(0.99999)
    assert np.allclose(component.exposure(simulation.population.population.index), expected_value)



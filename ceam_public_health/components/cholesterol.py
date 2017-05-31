import pandas as pd
import numpy as np

from scipy.stats import norm

from ceam.interpolation import Interpolation
from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns

from ceam_inputs import get_exposures

from ceam_public_health.util.risk import continuous_exposure_effect, make_risk_effects

def cholesterol_dists(func=None):
    df = get_exposures(106)
    # NOTE: Cholesterol is not modeled for younger ages so set them equal to the TMRL
    df.loc[df.age < 27.5, 'continuous'] = 3.08
    df = df.set_index(['age', 'sex', 'year'])
    means = df.mean(axis=1)
    means.name = 'mean'
    std = np.sqrt(means)
    std.name = 'std'
    dist = pd.concat([means, std], axis=1).reset_index()
    dist = Interpolation(dist, ['sex'], ['age', 'year'], func=lambda parameters: norm(loc=parameters['mean'], scale=parameters['std']).cdf)
    return dist

class TotalCholesterol:
    """Model total cholesterol

    Population Columns
    ------------------
    cholesterol_percentile
        Position of the simulant in the population's cholesterol distribution
    """

    def setup(self, builder):
        self.cholesterol_distributions = builder.lookup(cholesterol_dists())
        self.randomness = builder.randomness('total_cholesterol')

        effect_function = continuous_exposure_effect('total_cholesterol', tmrl=3.08, scale=1)
        risk_effects = make_risk_effects(106, [
            (493, 'heart_attack'),
            (495, 'ischemic_stroke'),
            ], effect_function, 'total_cholesterol')
        return risk_effects


    @listens_for('initialize_simulants')
    @uses_columns(['cholesterol_percentile', 'total_cholesterol'])
    def initialize(self, event):
        event.population_view.update(pd.DataFrame({
            'cholesterol_percentile': self.randomness.get_draw(event.index)*0.98+0.01,
            'total_cholesterol': np.full(len(event.index), 20.0)
        }))

    @listens_for('time_step__prepare', priority=8)
    @uses_columns(['total_cholesterol', 'cholesterol_percentile'], 'alive')
    def update_body_mass_index(self, event):
        new_cholesterol = self.cholesterol_distributions(event.index)(event.population.cholesterol_percentile)
        event.population_view.update(pd.Series(new_cholesterol, name='total_cholesterol', index=event.index))

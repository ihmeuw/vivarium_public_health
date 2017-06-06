import pandas as pd
import numpy as np

from scipy.stats import norm

from ceam.interpolation import Interpolation

from ceam_inputs import get_exposures, risk_factors


def distribution_loader(builder):
    df = get_exposures(risk_factors.high_total_normcholesterol.gbd_risk)
    # NOTE: Cholesterol is not modeled for younger ages so set them equal to the TMRL
    df.loc[df.age < 27.5, 'continuous'] = 3.08
    df = df.set_index(['age', 'sex', 'year'])
    means = df.mean(axis=1)
    means.name = 'mean'
    std = np.sqrt(means)
    std.name = 'std'
    dist = pd.concat([means, std], axis=1).reset_index()
    dist = Interpolation(dist, ['sex'], ['age', 'year'],
                         func=lambda parameters: norm(loc=parameters['mean'], scale=parameters['std']).ppf)
    return builder.lookup(dist)

from ceam_public_health.util.risk import assign_exposure_categories, assign_relative_risk_value
import pandas as pd
import numpy as np


def test_assign_exposure_categories():
    df = pd.DataFrame({'simulant_id': [x for x in range(0, 100000)], 'cat2': 100000*[.5], 'cat1': 100000*[.1], 'cat3': 100000*[.4] })
    categories = sorted([column for column in list(df) if 'cat' in column],
                        key=lambda s: int(s.split('cat')[1]))
    df = df[categories]
    df = np.cumsum(df, axis=1)
    df['susceptibility_colum'] = np.random.uniform(0, 1, size=100000)
    df = assign_exposure_categories(df, 'susceptibility_colum', categories)
    num_cat1 = len(df.loc[df.exposure_category == 'cat1'])
    num_cat2 = len(df.loc[df.exposure_category == 'cat2'])
    num_cat3 = len(df.loc[df.exposure_category == 'cat3'])
    assert np.allclose(num_cat1/100000, .1 , atol=.01), "assign_exposure_categories needs to assign exposures based on population distribution of the risk factor"
    assert np.allclose(num_cat2/100000, .5 , atol=.01), "assign_exposure_categories needs to assign exposures based on population distribution of the risk factor"
    assert np.allclose(num_cat3/100000, .4 , atol=.01), "assign_exposure_categories needs to assign exposures based on population distribution of the risk factor"

def test_assign_relative_risk_value():
    df = pd.DataFrame({'exposure_category': 5*['cat1', 'cat2'], 'cat1': 10*[1], 'cat2': 10*[2]})
    df = assign_relative_risk_value(df, ['cat1', 'cat2'])
    num_cat1 = len(df.loc[df.relative_risk_value == 1])
    num_cat2 = len(df.loc[df.relative_risk_value == 2])
    assert num_cat1 == 5, "assign_relative_risk value should assign rr based on simulant's exposure"
    assert num_cat2 == 5, "assign_relative_risk value should assign rr based on simulant's exposure"

# End.

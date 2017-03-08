import numpy as np
import pandas as pd

from ceam.framework.population import uses_columns
import re

def natural_key(string_):
    """ Sorts columns with strings and numbers naturally
    Parameters
    ----------
    string_: str
        string with letters and numbers
    """
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def naturally_sort_df(df):
    """
    grabs all of the category columns in a dataframe and then naturally sorts them

    Parameters
    ----------

    df : pd.DataFrame

    """
    col_list = df.columns.tolist()
    categories =  [c for c in col_list if "cat" in c]
    categories = sorted(categories, key=natural_key)

    return df[categories], categories


def assign_exposure_categories(df, susceptibility_column, categories):
    """
    creates an 'exposure_category' column that assigns simulant's exposure based on their susceptibility draw

    Parameters
    ----------
    df : pd.DataFrame

    susceptibility_column : pd.Series

    categories : list
        list of all of the category columns in df 
    """

    bool_list = [c + '_bool' for c in categories]
    
    for col in categories:
        df['{}_bool'.format(col)] = df['{}'.format(col)] < df[susceptibility_column]

    df['exposure_category'] = df[bool_list].sum(axis=1)

    # seems weird, but we need to add 1 to exposure category. e.g. if all values for a row in bool_list are false that simulant will be in exposure cat1, not cat0
    df['exposure_category'] = df['exposure_category'] + 1

    df['exposure_category'] = 'cat' + df['exposure_category'].astype(str)

    return df[['exposure_category']]
    

def assign_relative_risk_value(df, categories):
    """
    creates an 'relative_risk_value' column that assigns simulant's relative risk based on their exposure

    Parameters
    ----------
    df : pd.DataFrame

    susceptibility_column : pd.Series

    categories : list
        list of all of the category columns in df

    """

    for col in categories:
        df.loc[(df['exposure_category'] == col, 'relative_risk_value')] = df[col]

    return df


def continuous_exposure_effect(exposure_column, tmrl, scale):
    """Factory that makes functions which can be used as the exposure_effect
    for standard continuous risks

    Parameters
    ----------
    exposure_column : str
        The name of the column which contains exposure data for this risk
    tmrl : float
        The theoretical minimum risk level of the risk
    scale : float
        The ratio of the effect of one unit change in RR to change in rate
    """
    @uses_columns([exposure_column])
    def inner(rates, rr, population_view):
        return rates * np.maximum(rr.values**((population_view.get(rr.index)[exposure_column] - tmrl) / scale).values, 1)
    return inner


def categorical_exposure_effect(exposure, susceptibility_column):
    """Factory that makes function which can be used as the exposure_effect
    for binary categorical risks

    Parameters
    ----------
    exposure : ceam.framework.lookup.TableView
        A lookup for exposure data
    susceptibility_column : str
        The name of the column which contains susceptibility data
    """
    @uses_columns([susceptibility_column])
    def inner(rates, rr, population_view):
    
        pop = population_view.get(rr.index)

        exp = exposure(pop.index)

        exp, categories = naturally_sort_df(exp)

        # cumulatively sum over exposures
        exp = np.cumsum(exp, axis=1)

        exp = pop.join(exp)
        
        exp = assign_exposure_categories(exp, susceptibility_column, categories)

        df = exp.join(rr)

        df = assign_relative_risk_value(df, categories)

        return rates * (df.relative_risk_value.values)

    return inner


class RiskEffect:
    """
    Parameters
    ----------
    rr_data : pandas.DataFrame
        A dataframe of relative risk data with age, sex, year, and rr columns
    paf_data : pandas.DataFrame
        A dataframe of population attributable fraction data with age, sex, year, and paf columns
    cause : str
        The name of the cause to effect as used in named variables like 'incidence_rate.<cause>'
    exposure_effect : callable
        A function which takes a series of incidence rates and a series of
        relative risks and returns rates modified as appropriate for this risk
    """

    def __init__(self, rr_data, paf_data, cause, exposure_effect):
        self.rr_data = rr_data
        self.paf_data = paf_data
        self.cause_name = cause
        self.exposure_effect = exposure_effect

    def setup(self, builder):
        self.rr_lookup = builder.lookup(self.rr_data)
        builder.modifies_value(self.incidence_rates, 'incidence_rate.{}'.format(self.cause_name))
        builder.modifies_value(builder.lookup(self.paf_data), 'paf.{}'.format(self.cause_name))

        return [self.exposure_effect]

    def incidence_rates(self, index, rates):
        rr = self.rr_lookup(index)

        return self.exposure_effect(rates, rr)


# End.

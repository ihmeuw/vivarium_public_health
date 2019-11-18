"""
=========
Observers
=========

This module contains tools for recording various outputs of interest in
multi-state lifetable simulations.

"""
import pandas as pd


def output_file(config, suffix, sep='_', ext='csv'):
    """
    Determine the output file name for an observer, based on the prefix
    defined in ``config.observer.output_prefix`` and the (optional)
    ``config.input_data.input_draw_number``.

    Parameters
    ----------
    config
        The builder configuration object.
    suffix
        The observer-specific suffix.
    sep
        The separator between prefix, suffix, and draw number.
    ext
        The output file extension.

    """
    if 'observer' not in config:
        raise ValueError('observer.output_prefix not defined')
    if 'output_prefix' not in config.observer:
        raise ValueError('observer.output_prefix not defined')
    prefix = config.observer.output_prefix
    if 'input_draw_number' in config.input_data:
        draw = config.input_data.input_draw_number
    else:
        draw = 0
    out_file = prefix + sep + suffix
    if draw > 0:
        out_file += '{}{}'.format(sep, draw)
    out_file += '.{}'.format(ext)
    return out_file


class MorbidityMortality:
    """
    This class records the all-cause morbidity and mortality rates for each
    cohort at each year of the simulation.

    Parameters
    ----------
    output_suffix
        The suffix for the CSV file in which to record the
        morbidity and mortality data.

    """

    def __init__(self, output_suffix='mm'):
        self.output_suffix = output_suffix

    @property
    def name(self):
        return 'morbidity_mortality_observer'

    def setup(self, builder):
        # Record the key columns from the core multi-state life table.
        columns = ['age', 'sex',
                   'population', 'bau_population',
                   'acmr', 'bau_acmr',
                   'pr_death', 'bau_pr_death',
                   'deaths', 'bau_deaths',
                   'yld_rate', 'bau_yld_rate',
                   'person_years', 'bau_person_years',
                   'HALY', 'bau_HALY']
        self.population_view = builder.population.get_view(columns)
        self.clock = builder.time.clock()
        builder.event.register_listener('collect_metrics', self.on_collect_metrics)
        builder.event.register_listener('simulation_end', self.write_output)
        self.tables = []
        self.table_cols = ['sex', 'age', 'year',
                           'population', 'bau_population',
                           'prev_population', 'bau_prev_population',
                           'acmr', 'bau_acmr',
                           'pr_death', 'bau_pr_death',
                           'deaths', 'bau_deaths',
                           'yld_rate', 'bau_yld_rate',
                           'person_years', 'bau_person_years',
                           'HALY', 'bau_HALY']

        self.output_file = output_file(builder.configuration,
                                       self.output_suffix)

    def on_collect_metrics(self, event):
        pop = self.population_view.get(event.index)
        if len(pop.index) == 0:
            # No tracked population remains.
            return

        pop['year'] = self.clock().year
        # Record the population size prior to the deaths.
        pop['prev_population'] = pop['population'] + pop['deaths']
        pop['bau_prev_population'] = pop['bau_population'] + pop['bau_deaths']
        self.tables.append(pop[self.table_cols])

    def calculate_LE(self, table, py_col, denom_col):
        """Calculate the life expectancy for each cohort at each time-step.

        Parameters
        ----------
        table
            The population life table.
        py_col
            The name of the person-years column.
        denom_col
            The name of the population denominator column.

        Returns
        -------
            The life expectancy for each table row, represented as a
            pandas.Series object.

        """
        # Group the person-years by cohort.
        group_cols = ['year_of_birth', 'sex']
        subset_cols = group_cols + [py_col]
        grouped = table.loc[:, subset_cols].groupby(by=group_cols)[py_col]
        # Calculate the reverse-cumulative sums of the adjusted person-years
        # (i.e., the present and future person-years) by:
        #   (a) reversing the adjusted person-years values in each cohort;
        #   (b) calculating the cumulative sums in each cohort; and
        #   (c) restoring the original order.
        cumsum = grouped.apply(lambda x: pd.Series(x[::-1].cumsum()).iloc[::-1])
        return cumsum / table[denom_col]

    def write_output(self, event):
        data = pd.concat(self.tables, ignore_index=True)
        data['year_of_birth'] = data['year'] - data['age']
        # Sort the table by cohort (i.e., generation and sex), and then by
        # calendar year, so that results are output in the same order as in
        # the spreadsheet models.
        data = data.sort_values(by=['year_of_birth', 'sex', 'age'], axis=0)
        data = data.reset_index(drop=True)
        # Re-order the table columns.
        cols = ['year_of_birth'] + self.table_cols
        data = data[cols]
        # Calculate life expectancy and HALE for the BAU and intervention,
        # with respect to the initial population, not the survivors.
        data['LE'] = self.calculate_LE(data, 'person_years', 'prev_population')
        data['bau_LE'] = self.calculate_LE(data, 'bau_person_years',
                                           'bau_prev_population')
        data['HALE'] = self.calculate_LE(data, 'HALY', 'prev_population')
        data['bau_HALE'] = self.calculate_LE(data, 'bau_HALY',
                                           'bau_prev_population')
        data.to_csv(self.output_file, index=False)


class Disease:
    """
    This class records the disease incidence rate and disease prevalence for
    each cohort at each year of the simulation.

    Parameters
    ----------
    name
        The name of the chronic disease.
    output_suffix
        The suffix for the CSV file in which to record the
        disease data.

    """

    def __init__(self, name, output_suffix=None):
        self._name = name
        if output_suffix is None:
            output_suffix = name.lower()
        self.output_suffix = output_suffix
        
    @property
    def name(self):
        return f'{self._name}_observer'

    def setup(self, builder):
        bau_incidence_value = '{}.incidence'.format(self._name)
        int_incidence_value = '{}_intervention.incidence'.format(self._name)
        self.bau_incidence = builder.value.get_value(bau_incidence_value)
        self.int_incidence = builder.value.get_value(int_incidence_value)

        self.bau_S_col = '{}_S'.format(self._name)
        self.bau_C_col = '{}_C'.format(self._name)
        self.int_S_col = '{}_S_intervention'.format(self._name)
        self.int_C_col = '{}_C_intervention'.format(self._name)

        columns = ['age', 'sex',
                   self.bau_S_col, self.bau_C_col,
                   self.int_S_col, self.int_C_col]
        self.population_view = builder.population.get_view(columns)

        builder.event.register_listener('collect_metrics', self.on_collect_metrics)
        builder.event.register_listener('simulation_end', self.write_output)

        self.tables = []
        self.table_cols = ['sex', 'age', 'year',
                           'bau_incidence', 'int_incidence',
                           'bau_prevalence', 'int_prevalence',
                           'bau_deaths', 'int_deaths']
        self.clock = builder.time.clock()
        self.output_file = output_file(builder.configuration,
                                       self.output_suffix)

    def on_collect_metrics(self, event):
        pop = self.population_view.get(event.index)
        if len(pop.index) == 0:
            # No tracked population remains.
            return

        pop['year'] = self.clock().year
        pop['bau_incidence'] = self.bau_incidence(event.index)
        pop['int_incidence'] = self.int_incidence(event.index)
        pop['bau_prevalence'] = pop[self.bau_C_col] / (pop[self.bau_C_col] + pop[self.bau_S_col])
        pop['int_prevalence'] = pop[self.int_C_col] / (pop[self.bau_C_col] + pop[self.bau_S_col])
        pop['bau_deaths'] = 1000 - pop[self.bau_S_col] - pop[self.bau_C_col]
        pop['int_deaths'] = 1000 - pop[self.int_S_col] - pop[self.int_C_col]
        self.tables.append(pop.loc[:, self.table_cols])

    def write_output(self, event):
        data = pd.concat(self.tables, ignore_index=True)
        data['diff_incidence'] = data['int_incidence'] - data['bau_incidence']
        data['diff_prevalence'] = data['int_prevalence'] - data['bau_prevalence']
        data['year_of_birth'] = data['year'] - data['age']
        data['disease'] = self._name
        # Sort the table by cohort (i.e., generation and sex), and then by
        # calendar year, so that results are output in the same order as in
        # the spreadsheet models.
        data = data.sort_values(by=['year_of_birth', 'sex', 'age'], axis=0)
        data = data.reset_index(drop=True)
        # Re-order the table columns.
        diff_cols = ['diff_incidence', 'diff_prevalence']
        cols = ['disease', 'year_of_birth'] + self.table_cols + diff_cols
        data = data[cols]
        data.to_csv(self.output_file, index=False)


class TobaccoPrevalence:
    """This class records the prevalence of tobacco use in the population.

    Parameters
    ----------
    output_suffix
        The suffix for the CSV file in which to record the
        prevalence data.

    """

    def __init__(self, output_suffix='tobacco'):
        self.output_suffix = output_suffix
    
    @property
    def name(self):
        return 'tobacco_prevalence_observer'

    def setup(self, builder):
        self.config = builder.configuration
        self.clock = builder.time.clock()
        self.bin_years = int(self.config['tobacco']['delay'])

        view_columns = ['age', 'sex', 'bau_population', 'population'] + self.get_bin_names()
        self.population_view = builder.population.get_view(view_columns)

        self.tables = []
        self.table_cols = ['age', 'sex', 'year',
                           'bau_no', 'bau_yes', 'bau_previously', 'bau_population',
                           'int_no', 'int_yes', 'int_previously', 'int_population']

        builder.event.register_listener('collect_metrics',
                                        self.on_collect_metrics)
        builder.event.register_listener('simulation_end',
                                        self.write_output)
        self.output_file = output_file(builder.configuration,
                                       self.output_suffix)

    def get_bin_names(self):
        """Return the bin names for both the BAU and the intervention scenario.

        These names take the following forms:

        ``"name.no"``
            The number of people who have never been exposed.
        ``"name.yes"``
            The number of people currently exposed.
        ``"name.N"``
            The number of people N years post-exposure.

        The final bin is the number of people :math:`\ge N` years
        post-exposure.

        The intervention bin names take the form ``"name_intervention.X"``.

        """
        if self.bin_years == 0:
            delay_bins = [str(0)]
        else:
            delay_bins = [str(s) for s in range(self.bin_years + 2)]
        bins = ['no', 'yes'] + delay_bins
        bau_bins = ['{}.{}'.format('tobacco', bin) for bin in bins]
        int_bins = ['{}_intervention.{}'.format('tobacco', bin) for bin in bins]
        all_bins = bau_bins + int_bins
        return all_bins

    def on_collect_metrics(self, event):
        pop = self.population_view.get(event.index)
        if len(pop.index) == 0:
            # No tracked population remains.
            return

        bau_cols = [c for c in pop.columns.values
                    if c.startswith('{}.'.format('tobacco'))]
        int_cols = [c for c in pop.columns.values
                    if c.startswith('{}_intervention.'.format('tobacco'))]

        bau_denom = pop.reindex(columns=bau_cols).sum(axis=1)
        int_denom = pop.reindex(columns=int_cols).sum(axis=1)

        # Normalise prevalence with respect to the total population.
        pop['bau_no'] = pop['{}.no'.format('tobacco')] / bau_denom
        pop['bau_yes'] = pop['{}.yes'.format('tobacco')] / bau_denom
        pop['bau_previously'] = 1 - pop['bau_no'] - pop['bau_yes']
        pop['int_no'] = pop['{}_intervention.no'.format('tobacco')] / int_denom
        pop['int_yes'] = pop['{}_intervention.yes'.format('tobacco')] / int_denom
        pop['int_previously'] = 1 - pop['int_no'] - pop['int_yes']

        pop = pop.rename(columns={'population': 'int_population'})

        pop['year'] = self.clock().year
        self.tables.append(pop.reindex(columns=self.table_cols).reset_index(drop=True))

    def write_output(self, event):
        data = pd.concat(self.tables, ignore_index=True)
        data['year_of_birth'] = data['year'] - data['age']
        # Sort the table by cohort (i.e., generation and sex), and then by
        # calendar year, so that results are output in the same order as in
        # the spreadsheet models.
        data = data.sort_values(by=['year_of_birth', 'sex', 'age'], axis=0)
        data = data.reset_index(drop=True)
        # Re-order the table columns.
        cols = ['year_of_birth'] + self.table_cols
        data = data.reindex(columns=cols)
        data.to_csv(self.output_file, index=False)

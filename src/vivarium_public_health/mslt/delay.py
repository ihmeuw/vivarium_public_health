"""Provide components to represent delayed effects."""
import pandas as pd

from . import add_year_column


class DelayedRisk:
    """
    A delayed risk represents an exposure whose impact takes time to come into
    effect (e.g., smoking uptake and cessation).

    The data required by this component are:

    - Initial prevalence: the initial level of exposure and post-exposure.
    - Incidence rate: the rate at which people begin to be exposed;

      - This should be specified separately for the BAU and the intervention
        scenario.

    - Remission rate: the rate at which people stop being exposed;

      - This should be specified separately for the BAU and the intervention
        scenario.

    - Relative risk of mortality for:

      - Currently exposed (e.g., currently smoking); and
      - Post-exposure (e.g., stopped smoking :math:`0..N` years ago).

    - Disease-specific relative risks for:

      - Currently exposed (e.g., currently smoking); and
      - Post-exposure (e.g., stopped smoking :math:`0..N` years ago).

    .. note:: The relative risks are defined in relation to the pre-exposure
       group (whose relative risks are therefore defined to be :math:`1`).

    Identify the disease(s) for which this delayed risk will have an effect in
    the simulation configuration. For example, to modify the incidence of CHD
    and stroke, this would look like:

    .. code-block:: yaml

       components:
           mslt_port:
               population:
                   - BasePopulation()
                   - Mortality()
                   - Disability()
               disease:
                   - Disease('chd')
                   - Disease('stroke')
               delay:
                   - DelayedRisk('tobacco')
               ...
       configuration:
           tobacco:
               affects:
                   # This is where the affected diseases should be listed.
                   stroke:
               # For now, apply a constant PIF to incidence and/or remission.
               incidence_pif: 0.95
               remission_pif: 1.05
    """

    def __init__(self, name, bin_years=20):
        """
        :param name: The name of the exposure (e.g., ``"tobacco"``).
        :param bin_years: The number of years over which the risk changes
            (default: 20).
        """
        self.name = name
        self.bin_years = bin_years

    def setup(self, builder):
        """
        Configure the delayed risk component.

        This involves loading the required data tables, registering event
        handlers and rate modifiers, and setting up the population view.
        """
        self.config = builder.configuration

        # NOTE: for now, apply a constant PIF to incidence and remission.
        self.incidence_pif = self.config[self.name].incidence_pif
        self.remission_pif = self.config[self.name].remission_pif

        # Load the initial prevalence.
        prev_data = builder.data.load(f'risk_factor.{self.name}.prevalence')
        self.initial_prevalence = builder.lookup.build_table(prev_data)

        # Load the incidence rates for the BAU and intervention scenarios.
        inc_data = builder.lookup.build_table(
            builder.data.load(f'risk_factor.{self.name}.incidence')
        )
        inc_name = '{}.incidence'.format(self.name)
        inc_int_name = '{}_intervention.incidence'.format(self.name)
        self.incidence = builder.value.register_rate_producer(inc_name, source=inc_data)
        self.int_incidence = builder.value.register_rate_producer(inc_int_name, source=inc_data)

        # Load the remission rates for the BAU and intervention scenarios.
        rem_data = builder.lookup.build_table(
            builder.data.load(f'risk_factor.{self.name}.remission')
        )
        rem_name = '{}.remission'.format(self.name)
        rem_int_name = '{}_intervention.remission'.format(self.name)
        self.remission = builder.value.register_rate_producer(rem_name, source=rem_data)
        self.int_remission = builder.value.register_rate_producer(rem_int_name, source=rem_data)

        # We apply separate mortality rates to the different exposure bins.
        # This requires having access to the life table mortality rate, and
        # also the relative risks associated with each bin.
        self.acm_rate = builder.value.get_value('mortality_rate')
        mort_rr_data = builder.data.load(f'risk_factor.{self.name}.mortality_relative_risk')
        self.mortality_rr = builder.lookup.build_table(mort_rr_data)

        # Register a modifier for each disease affected by this delayed risk.
        diseases = self.config[self.name].affects.keys()
        for ix, disease in enumerate(diseases):
            self.register_modifier(builder, disease)

        # Load the disease-specific relative risks for each exposure bin.
        dis_rr_data = builder.data.load(f'risk_factor.{self.name}.disease_relative_risk')
        key_columns = ['age', 'sex', 'year']
        self.dis_rr = {}
        for disease in diseases:
            dis_columns = [c for c in dis_rr_data.columns
                           if c.startswith(disease)]
            dis_keys = [c for c in dis_rr_data.columns
                        if c in key_columns]
            if not dis_columns or not dis_keys:
                msg = 'No {} relative risks for disease {}'
                raise ValueError(msg.format(self.name, disease))
            rr_data = dis_rr_data[dis_keys + dis_columns]
            dis_prefix = '{}_'.format(disease)
            bau_prefix = '{}.'.format(self.name)
            int_prefix = '{}_intervention.'.format(self.name)
            bau_col = {c: c.replace(dis_prefix, bau_prefix).replace('post_', '')
                       for c in dis_columns}
            int_col = {c: c.replace(dis_prefix, int_prefix).replace('post_', '')
                       for c in dis_columns}
            for column in dis_columns:
                rr_data[int_col[column]] = rr_data[column]
            rr_data = rr_data.rename(columns=bau_col)
            rr_data = add_year_column(builder, rr_data)
            self.dis_rr[disease] = builder.lookup.build_table(rr_data)

        # Add a handler to create the exposure bin columns.
        req_columns = ['age', 'sex', 'population']
        new_columns = self.get_bin_names()
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=new_columns,
            requires_columns=req_columns)

        # Add a handler to move people from one bin to the next.
        builder.event.register_listener('time_step__prepare',
                                        self.on_time_step_prepare)

        # Define the columns that we need to access during the simulation.
        view_columns = req_columns + new_columns
        self.population_view = builder.population.get_view(view_columns)

    def get_bin_names(self):
        """
        Return the bin names for both the BAU and the intervention scenario.

        These names take the following forms:

        - ``"name.no"``: The number of people who have never been exposed.
        - ``"name.yes"``: The number of people currently exposed.
        - ``"name.N"``: The number of people N years post-exposure.

          - The final bin is the number of people :math:`\ge N` years
            post-exposure.

        The intervention bin names take the form ``"name_intervention.X"``.
        """
        bins = ['no', 'yes'] + [str(s) for s in range(self.bin_years + 2)]
        bau_bins = ['{}.{}'.format(self.name, bin) for bin in bins]
        int_bins = ['{}_intervention.{}'.format(self.name, bin) for bin in bins]
        all_bins = bau_bins + int_bins
        return all_bins

    def on_initialize_simulants(self, pop_data):
        """
        Define the initial distribution of the population across the bins, in
        both the BAU and the intervention scenario.
        """
        # Set all bins to zero, in order to create the required columns.
        pop = pd.DataFrame({}, index=pop_data.index)
        for column in self.get_bin_names():
            pop[column] = 0

        # Update the life table, so that we can then obtain a view that
        # includes the population counts.
        self.population_view.update(pop)
        pop = self.population_view.get(pop_data.index)

        # Calculate the absolute prevalence by multiplying the fractional
        # prevalence by the population size for each cohort.
        prev = self.initial_prevalence(pop_data.index).mul(pop['population'], axis=0)
        self.population_view.update(prev)

        # Rename the columns and apply the same initial prevalence for the
        # intervention.
        bau_prefix = '{}.'.format(self.name)
        int_prefix = '{}_intervention.'.format(self.name)
        rename_to = {c: c.replace(bau_prefix, int_prefix)
                     for c in prev.columns if c.startswith(bau_prefix)}
        int_prev = prev.rename(columns=rename_to)
        self.population_view.update(int_prev)

    def on_time_step_prepare(self, event):
        """
        Account for transitions between bins, and for mortality rates.

        These transitions include:

        - New exposures;
        - Cessation of exposure; and
        - Increased duration of time since exposure.
        """
        idx = event.index
        pop = self.population_view.get(idx)
        acmr = self.acm_rate(idx)
        inc_rate = self.incidence(idx)
        rem_rate = self.remission(idx)
        # NOTE: for now, apply a constant PIF to the incidence rate.
        int_inc_rate = self.int_incidence(idx) * self.incidence_pif
        int_rem_rate = self.int_remission(idx) * self.remission_pif

        # Calculate the survival rate for each bin.
        pop = self.population_view.get(idx)
        bin_cols = self.get_bin_names()
        base_surv_rate = (1 - acmr)
        surv_rate = self.mortality_rr(idx).rpow(base_surv_rate, axis=0)

        # Account for mortality in each bin.
        pop[bin_cols] = pop[bin_cols].mul(surv_rate[bin_cols])

        # Account for transitions between bins.
        # Note that the order of evaluation matters.
        suffixes = ['', '_intervention']
        # First, accumulate the final post-exposure bin.
        for suffix in suffixes:
            accum_col = '{}{}.{}'.format(self.name, suffix, self.bin_years + 1)
            from_col = '{}{}.{}'.format(self.name, suffix, self.bin_years)
            pop[accum_col] += pop[from_col]
        # Then increase time since exposure for all other post-exposure bins.
        for n_years in reversed(range(self.bin_years)):
            for suffix in suffixes:
                source_col = '{}{}.{}'.format(self.name, suffix, n_years)
                dest_col = '{}{}.{}'.format(self.name, suffix, n_years + 1)
                pop[dest_col] = pop[source_col]

        # Account for incidence and remission.
        col_no = '{}.no'.format(self.name)
        col_int_no = '{}_intervention.no'.format(self.name)
        col_yes = '{}.yes'.format(self.name)
        col_int_yes = '{}_intervention.yes'.format(self.name)
        col_zero = '{}.0'.format(self.name)
        col_int_zero = '{}_intervention.0'.format(self.name)
        inc = inc_rate * pop[col_no]
        int_inc = int_inc_rate * pop[col_int_no]
        rem = rem_rate * pop[col_yes]
        int_rem = int_rem_rate * pop[col_int_yes]
        pop[col_no] = pop[col_no] - inc
        pop[col_int_no] = pop[col_int_no] - int_inc
        pop[col_yes] = pop[col_yes] + inc - rem
        pop[col_int_yes] = pop[col_int_yes] + int_inc - int_rem
        pop[col_zero] = rem
        pop[col_int_zero] = int_rem

        self.population_view.update(pop)

    def register_modifier(self, builder, disease):
        """
        Register that a disease incidence rate will be modified by this
        delayed risk in the intervention scenario.

        :param builder: The builder object for the simulation, which provides
            access to event handlers and rate modifiers.
        :param disease: The name of the disease whose incidence rate will be
            modified.
        """
        inc_rate = '{}_intervention.incidence'.format(disease)
        modifier = lambda ix, rate: self.incidence_adjustment(disease, ix, rate)
        builder.value.register_value_modifier(inc_rate, modifier)

    def incidence_adjustment(self, disease, index, incidence_rate):
        """
        Modify a disease incidence rate in the intervention scenario.

        :param disease: The name of the disease.
        :param index: The index into the population life table.
        :param incidence_rate: The un-adjusted disease incidence rate.
        """
        # Multiply the population in each bin by the associated relative risk.
        bin_cols = self.get_bin_names()
        incidence_rr = self.dis_rr[disease](index)[bin_cols]
        pop = self.population_view.get(index)
        rr_values = pop[bin_cols] * incidence_rr

        # Calculate the mean relative-risk for the BAU scenario.
        bau_prefix = '{}.'.format(self.name)
        bau_cols = [c for c in bin_cols if c.startswith(bau_prefix)]
        # Sum over all of the bins in each row.
        mean_bau_rr = rr_values[bau_cols].sum(axis=1)

        # Calculate the mean relative-risk for the intervention scenario.
        int_prefix = '{}_intervention.'.format(self.name)
        int_cols = [c for c in bin_cols if c.startswith(int_prefix)]
        # Sum over all of the bins in each row.
        mean_int_rr = rr_values[int_cols].sum(axis=1)

        # Calculate the disease incidence PIF for the intervention scenario.
        pif = mean_int_rr / mean_bau_rr
        return incidence_rate * pif

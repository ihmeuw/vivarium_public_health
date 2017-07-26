"""This module contains several components that  model birth rates."""
import pandas as pd
import numpy as np

from vivarium import config
from vivarium.framework.event import listens_for
from vivarium.framework.population import uses_columns, creates_simulants

from ceam_inputs import get_age_specific_fertility_rates, get_annual_live_births, get_populations

DAYS_PER_YEAR = 365
# TODO: Incorporate GBD estimates into gestational model (probably as a separate component)
PREGNANCY_DURATION = pd.Timedelta(days=9*30.5)


class FertilityDeterministic:
    """Deterministic model of births.

    This model of fertility expects that
    `config.simulation_parameters.number_of_new_simulants_each_year` is
    set in some configuration file.  It adds simulants every time-step
    by scaling this parameter by the time-step size.

    Attributes
    ----------
    fractional_new_births : float
        A rolling record of the fractional part of new births generated
        each time-step that allows us to
    """
    def __init__(self):
        self.fractional_new_births = 0
        # Assume time step comes to us in days
        self.annual_new_simulants = config.simulation_parameters.number_of_new_simulants_each_year

    @listens_for('time_step')
    @creates_simulants
    def add_new_birth_cohort(self, event, creator):
        """Deterministically adds a new set of simulants at every timestep
        based on a parameter in the configuration.

        Parameters
        ----------
        event : vivarium.population.PopulationEvent
            The event that triggered the function call.
        creator : method
            A function or method for creating a population.
        """

        # Assume births are uniformly distributed throughout the year.
        simulants_to_add = self.annual_new_simulants*event.step_size.days/DAYS_PER_YEAR + self.fractional_new_births
        self.fractional_new_births = simulants_to_add % 1
        simulants_to_add = int(simulants_to_add)

        creator(simulants_to_add,
                population_configuration={
                    'initial_age': 0.0,
                })


class FertilityCrudeBirthRate:
    """Population-level model of births using Crude Birth Rate.

    Attributes
    ----------
    randomness : `randomness.RandomStream`
        A named stream of random numbers bound to CEAM's common
        random number framework.

    Notes
    -----
    The OECD definition of Crude Birthrate can be found on their
    website_, while a more thorough discussion of fertility and
    birth rate models can be found on Wikipedia_ or in demography
    textbooks.

    .. _website: https://stats.oecd.org/glossary/detail.asp?ID=490
    .. _Wikipedia: https://en.wikipedia.org/wiki/Birth_rate
    """
    def setup(self, builder):
        self.randomness = builder.randomness('crude_birth_rate')

    @listens_for('time_step')
    @uses_columns([], "alive == 'alive'")
    @creates_simulants
    def add_new_birth_cohort(self, event, creator):
        """Adds new simulants every time step based on the Crude Birth Rate
        and an assumption that birth is a Poisson process

        Parameters
        ----------
        event : vivarium.population.PopulationEvent
            The event that triggered the function call.
        creator : method
            A function or method for creating a population.

        Notes
        -----
        The method for computing the Crude Birth Rate employed here is
        approximate.

        """
        # FIXME: We are pulling data every time here.  Use the value pipeline system.
        birth_rate = self._get_birth_rate(event.time.year)
        population_size = len(event.index)

        mean_births = birth_rate*population_size*event.step_size.days/DAYS_PER_YEAR

        # Assume births occur as a Poisson process
        r = np.random.RandomState(seed=self.randomness.get_seed())
        simulants_to_add = r.poisson(mean_births)

        creator(simulants_to_add,
                population_configuration={
                    'initial_age': 0.0,
                })

    @staticmethod
    def _get_birth_rate(year):
        """Computes a crude birth rate from demographic data in a given year.

        Parameters
        ----------
        year : int
            The year we want the birth rate for.

        Returns
        -------
        float
            The crude birth rate of the population in the given year in
            births per person per year.
        """

        location_id = config.simulation_parameters.location_id
        population_table = get_populations(location_id, year, sex='Both')

        if 'maximum_age' in config.simulation_parameters:
            population = population_table.pop_scaled[
                population_table.age < config.simulation_parameters.maximum_age].sum()
        else:
            population = population_table.pop_scaled.sum()

        births = float(get_annual_live_births(location_id, year))
        return births / population


class FertilityAgeSpecificRates:
    """
    A simulant-specific model for fertility and pregnancies.
    """

    def setup(self, builder):
        """ Setup the common randomness stream and
        age-specific fertility lookup tables.

        Parameters
        ----------
        builder : vivarium.engine.Builder
            Framework coordination object.

        """

        self.randomness = builder.randomness('fertility')
        self.asfr = builder.lookup(get_age_specific_fertility_rates()[['year', 'age', 'rate']],
                                   key_columns=(),
                                   parameter_columns=('year', 'age',))

    @listens_for('initialize_simulants')
    @uses_columns(['last_birth_time', 'sex', 'parent_id'])
    def update_state_table(self, event):
        """ Adds 'last_birth_time' and 'parent' columns to the state table.

        Parameters
        ----------
        event : vivarium.population.PopulationEvent
            Event that triggered this method call.
        """

        women = event.population.sex == 'Female'
        last_birth_time = pd.Series(pd.NaT, name='last_birth_time', index=event.index)

        # Do the naive thing, set so all women can have children
        # and none of them have had a child in the last year.
        last_birth_time[women] = event.time - pd.Timedelta(days=DAYS_PER_YEAR)

        event.population_view.update(last_birth_time)
        event.population_view.update(pd.Series(-1, name='parent_id', index=event.index, dtype=np.int64))

    @listens_for('time_step')
    @uses_columns(['last_birth_time', 'parent_id'], 'alive == "alive" and sex =="Female"')
    @creates_simulants
    def step(self, event, creator):
        """Produces new children and updates parent status on time steps.

        Parameters
        ----------
        event : vivarium.population.PopulationEvent
            The event that triggered the function call.
        creator : method
            A function or method for creating a population.
        """
        # Get a view on all living women who haven't had a child in at least nine months.
        nine_months_ago = pd.Timestamp(event.time - PREGNANCY_DURATION)

        can_have_children = event.population.last_birth_time < nine_months_ago
        eligible_women = event.population[can_have_children]

        rate_series = self.asfr(eligible_women.index)
        had_children = self.randomness.filter_for_rate(eligible_women, rate_series)

        had_children.loc[:, 'last_birth_time'] = event.time
        event.population_view.update(had_children['last_birth_time'])

        # If children were born, add them to the state table and record
        # who their mother was.
        num_babies = len(had_children)
        if num_babies:
            idx = creator(num_babies, population_configuration={'initial_age': 0})
            parents = pd.Series(data=had_children.index, index=idx, name='parent_id')
            event.population_view.update(parents)

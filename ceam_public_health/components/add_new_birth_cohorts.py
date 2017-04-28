from math import ceil
from scipy.stats import poisson
import pandas as pd
import numpy as np


from ceam import config
from ceam_inputs import get_age_specific_fertility_rates
from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam.framework.population import creates_simulants
from ceam_inputs.gbd_ms_auxiliary_functions import get_populations
from ceam.framework.util import rate_to_probability

DAYS_PER_YEAR = 365.
PREGNANCY_DURATION = pd.Timedelta(days=9*30.5)


@listens_for('time_step', priority=9)
@creates_simulants
def add_new_birth_cohort(event, creator):
    """Deterministically adds a new set of simulants at every timestep
    based on a parameter in the configuration.
    
    Parameters
    ----------
    event : ceam.population.PopulationEvent
        The event that triggered the function call.
    creator : method
        A function or method for creating a population.
    """
    # Assume time step comes to us in days
    time_step_size = config.simulation_parameters.time_step
    annual_new_simulants = config.simulation_parameters.number_of_new_simulants_each_year

    # Assume births are uniformly distributed throughout the year.
    # N.B. Tracking things like leap years, etc., seems silly at this level
    # of model detail, so we don't.
    simulants_to_add = int(ceil(
        annual_new_simulants * time_step_size / DAYS_PER_YEAR))

    creator(simulants_to_add,
            population_configuration={
                'initial_age': 0.0,
            })


@listens_for('time_step', priority=9)
@creates_simulants
def add_new_birth_cohort_nondeterministic(event, creator):
    """Adds new simulants every time step based on the Crude Birth Rate
    and an assumption that birth is a Poisson process
    
    Parameters
    ----------
    event : ceam.population.PopulationEvent
        The event that triggered the function call.
    creator : method
        A function or method for creating a population. 
    """
    birth_rate = _get_birth_rate(event.time.year)
    population_size = len(event.index)
    time_step_size = config.simulation_parameters.time_step
    mean_births = birth_rate*population_size*time_step_size/DAYS_PER_YEAR

    # Make the random draws deterministic w/r/t the configuration draw number.
    seed = (config.run_configuration.draw_number + hash(event.time)) % (2**32 - 1)

    # Assume births occur as a Poisson process
    simulants_to_add = poisson.rvs(mean_births, random_state=seed)

    creator(simulants_to_add,
            population_configuration={
                'initial_age': 0.0,
            })


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

    # 3 is the sex_id to pull both males and females
    population_table = get_populations(location_id, year, 3)

    population = population_table.pop_scaled.sum()
    births = population_table.pop_scaled[population_table.age<1].sum()

    return births / population


class Fertility:
    """
    A simulant-specific model for fertility and pregnancies.
    """

    def setup(self, builder):
        """ Setup the common randomness stream and 
        age-specific fertility lookup tables.
        
        Parameters
        ----------
        builder : ceam.engine.Builder
            Framework coordination object.    
         
        """

        self.random = builder.randomness('fertility')
        self.asfr = builder.lookup(get_age_specific_fertility_rates()[['year', 'age', 'mean_value']],
                                   key_columns=(),
                                   parameter_columns=('year', 'age',))

    @listens_for('initialize_simulants')
    @uses_columns(['last_birth_time', 'sex', 'parent'])
    def update_state_table(self, event):
        """ Adds 'last_birth_time' and 'parent' columns to the state table.
        
        Parameters
        ----------
        event : ceam.population.PopulationEvent
            Event that triggered this method call.
        """

        women = event.population.sex == 'Female'
        last_birth_time = pd.Series(pd.NaT, name='last_birth_time', index=event.index)

        # Do the naive thing, set so all women can have children
        # and none of them have had a child in the last year.
        last_birth_time[women] = event.time - pd.Timedelta(days=365)

        event.population_view.update(last_birth_time)
        event.population_view.update(pd.Series(-1, name='parent', index=event.index, dtype=np.int64))

    @listens_for('time_step', priority=8)
    @uses_columns(['last_birth_time', 'parent'], 'alive == True and sex =="Female"')
    @creates_simulants
    def step(self, event, creator):
        """Produces new children and updates parent status on time steps.
        
        Parameters
        ----------
        event : ceam.population.PopulationEvent
            The event that triggered the function call.
        creator : method
            A function or method for creating a population. 
        """
        # Get a view on all living women who haven't had a child
        # in at least nine months.
        nine_months_ago = pd.Timestamp(event.time - PREGNANCY_DURATION)
        can_have_children = event.population.query('last_birth_time < @nine_months_ago')

        # Get the age-specific rates, convert to probabilities
        rate_series = self.asfr(can_have_children.index)
        prob_series = rate_to_probability(rate_series)
        # Determine whether each eligible simulant had a child.
        prob_df = pd.DataFrame({'birth': prob_series, 'no_birth': 1-prob_series})
        prob_df['outcome'] = self.random.choice(prob_df.index, prob_df.columns, prob_df)

        # Get all simulants who had children and record that
        # they just had a child.
        had_children = prob_df.query('outcome == "birth"').copy()
        had_children['last_birth_time'] = event.time
        event.population_view.update(had_children['last_birth_time'])

        # If children were born, add them to the state table and record
        # who their mother was.
        num_babies = len(had_children)
        if num_babies:
            idx = creator(num_babies, population_configuration={'initial_age': 0})
            parents = pd.Series(data=had_children.index, index=idx, name='parent')
            event.population_view.update(parents)





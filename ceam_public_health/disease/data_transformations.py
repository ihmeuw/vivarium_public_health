import pandas as pd
import numpy as np

from ceam.interpolation import Interpolation


def get_cause_level_prevalence(states, year_start):
    prevalence_df = pd.DataFrame()
    for key in states.keys():
        assert set(states[key].columns) == {'year', 'age', 'prevalence', 'sex'}, ("The keys in the dict passed to "
                                                                                  "get_cause_level_prevalence need "
                                                                                  "to be dataframes with columns year, "
                                                                                  "age, prevalence, and sex.")
        states[key] = states[key].query("year == {}".format(year_start))
        states[key] = states[key][['year', 'age', 'prevalence', 'sex']]
        prevalence_df = prevalence_df.append(states[key])

    cause_level_prevalence = prevalence_df.groupby(['year', 'sex', 'age'], as_index=False)[['prevalence']].sum()
    return cause_level_prevalence, states


def determine_if_sim_has_cause(simulants_df, cause_level_prevalence, randomness):
    # TODO: Need to include Interpolation in this function for cause_level_prevalence.
    # There are more age values for simulants df (older ages) than there are for cause_level_prevalence,
    # hence why an interpolation function is needed.
    # TODO: this is weird and not general but I don't think we should be doing this lookup here anyway
    assert len(set(cause_level_prevalence.year)) == 1
    cause_level_prevalence = cause_level_prevalence.copy()
    del cause_level_prevalence['year']
    probability_of_disease = Interpolation(cause_level_prevalence, ['sex'], ['age'])(simulants_df[['age', 'sex']])
    probability_of_not_having_disease = 1 - probability_of_disease
    weights = np.array([probability_of_not_having_disease, probability_of_disease]).T
    results = simulants_df.copy()
    results = results.set_index('simulant_id')
    # Need to sort results so that the simulants are in the same order as the weights
    results['condition_envelope'] = randomness.choice(results.index, [False, True], weights)
    return results


def get_sequela_proportions(cause_level_prevalence, states):
    """Returns a dictionary with keys that are modelable entity ids and values are dataframes with proportion data

    Parameters
    ----------
    cause_level_prevalence: df
        dataframe of 1k prevalence draws

    states : dict
        dict with keys = name of cause, values = dataframe of prevalence draws

    Returns
    -------
    A dictionary of dataframes where each dataframe contains
    proportion of cause prevalence made up by a specific sequela
    """

    sequela_proportions = {}

    for key in states.keys():
        sequela_proportions[key] = pd.merge(states[key], cause_level_prevalence,
                                            on=['age', 'sex', 'year'], suffixes=('_single', '_total'))
        single = sequela_proportions[key]['prevalence_single']
        total = sequela_proportions[key]['prevalence_total']
        sequela_proportions[key]['scaled_prevalence'] = np.divide(single, total)

    return sequela_proportions


def determine_which_seq_diseased_sim_has(sequela_proportions, new_sim_file, randomness):
    """
    Parameters
    ----------
    sequela_proportions: dict
        a dictionary of dataframes where each dataframe contains proportion of
        cause prevalence made up by a specific sequela
    new_sim_file: df
        dataframe of simulants

    Returns
    -------
    dataframe of simulants with new column condition_state that indicates if simulant which
    sequela simulant has or indicates that they are healthy (i.e. they do not have the disease)
    """
    sequela_proportions = [(key, Interpolation(data[['sex', 'age', 'scaled_prevalence']], ['sex'], ['age']))
                           for key, data in sequela_proportions.items()]
    sub_pop = new_sim_file.query('condition_envelope == 1')
    list_of_keys, list_of_weights = zip(*[(key, data(sub_pop)) for key, data in sequela_proportions])
    results = randomness.choice(sub_pop.index, list_of_keys, np.array(list_of_weights).T)
    new_sim_file.loc[sub_pop.index, 'condition_state'] = results
    return new_sim_file


def assign_cause_at_beginning_of_simulation(simulants_df, year_start, states, randomness):
    """Function that assigns chronic ihd status to starting population of simulants

    Parameters
    ----------
    simulants_df : dataframe
        dataframe of simulants that is made by generate_ceam_population
    year_start : int, year
        year_start is the year in which you want to start the simulation
    states : dict
        dict with keys = name of cause, values = modelable entity id of cause

    Returns
    -------
    Creates a new column for a df of simulants with a column called chronic_ihd
        chronic_ihd takes values 0 or 1
            1 indicates that the simulant has chronic ihd
            0 indicates that the simulant does not have chronic ihd
    """
    simulants_df = simulants_df.reset_index()
    simulants_df['simulant_id'] = simulants_df['index']
    simulants_df = simulants_df[['simulant_id', 'age', 'sex']]

    cause_level_prevalence, prevalence_draws_dictionary = get_cause_level_prevalence(states, year_start)
    # TODO: Should we be using groupby for these loops to ensure that
    # we're not looping over an age/sex combo that does not exist
    post_cause_assignment_population = determine_if_sim_has_cause(simulants_df, cause_level_prevalence, randomness)
    sequela_proportions = get_sequela_proportions(cause_level_prevalence, states)
    post_sequela_assignmnet_population = determine_which_seq_diseased_sim_has(sequela_proportions,
                                                                              post_cause_assignment_population,
                                                                              randomness)
    post_sequela_assignmnet_population.condition_state = post_sequela_assignmnet_population.condition_state.fillna(
        'healthy')

    return post_sequela_assignmnet_population

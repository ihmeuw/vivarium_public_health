import pandas as pd
import numpy as np

from vivarium.interpolation import Interpolation


def get_cause_level_prevalence(states, year_start):
    prevalence_df = pd.DataFrame()
    initial_states = states
    states = dict(states)
    for key in states.keys():
        assert set(states[key].columns) == {'year', 'age', 'value', 'sex'}, ("The keys in the dict passed to "
                                                                                  "get_cause_level_prevalence need "
                                                                                  "to be dataframes with columns year, "
                                                                                  "age, value, and sex.")
        states[key] = states[key].rename(columns={"value":"prevalence"})

        # round to nearest gbd year
        years = list(states[key].year.unique())
        deltas = [abs(year_start - y) for y in years]
        y = years[deltas.index(min(deltas))]
        states[key] = states[key].query(f"year == {y}")
        states[key] = states[key][['year', 'age', 'prevalence', 'sex']]
        prevalence_df = prevalence_df.append(states[key])

    cause_level_prevalence = prevalence_df.groupby(['year', 'sex', 'age'], as_index=False)[['prevalence']].sum()
    return cause_level_prevalence, states


def determine_if_sim_has_cause(simulants_df, cause_level_prevalence, randomness, interpolation_order):
    # TODO: Need to include Interpolation in this function for cause_level_prevalence.
    # There are more age values for simulants df (older ages) than there are for cause_level_prevalence,
    # hence why an interpolation function is needed.
    # TODO: this is weird and not general but I don't think we should be doing this lookup here anyway
    assert len(set(cause_level_prevalence.year)) == 1
    cause_level_prevalence = cause_level_prevalence.copy()
    del cause_level_prevalence['year']
    probability_of_disease = Interpolation(
        cause_level_prevalence, ['sex'], ['age'], order=interpolation_order)(simulants_df[['age', 'sex']])
    probability_of_not_having_disease = 1 - probability_of_disease
    weights = np.array([probability_of_not_having_disease.values, probability_of_disease.values]).T
    results = simulants_df.copy()
    # Need to sort results so that the simulants are in the same order as the weights
    results['condition_envelope'] = randomness.choice(results.index, [False, True], weights)
    return results


def get_sequela_proportions(cause_level_prevalence, states):
    sequela_proportions = {}

    for key in states.keys():
        sequela_proportions[key] = pd.merge(states[key].rename(columns={"value":"prevalence"}), cause_level_prevalence,
                                            on=['age', 'sex', 'year'], suffixes=('_single', '_total'))
        single = sequela_proportions[key]['prevalence_single']
        total = sequela_proportions[key]['prevalence_total']
        with np.errstate(invalid='ignore'):
            sequela_proportions[key]['scaled_prevalence'] = np.nan_to_num(np.divide(single, total))

    return sequela_proportions


def determine_which_seq_diseased_sim_has(sequela_proportions, new_sim_file, randomness, interpolation_order):
    sequela_proportion_interpolations = [
        (key, Interpolation(data[['sex', 'age', 'scaled_prevalence']], ['sex'], ['age'], order=interpolation_order))
        for key, data in sequela_proportions.items()
    ]
    sub_pop = new_sim_file.query('condition_envelope == 1')
    list_of_keys, list_of_weights = zip(*[(key, data(sub_pop)) for key, data in sequela_proportion_interpolations])
    # Hack around a bug in pandas.
    if isinstance(list_of_weights[0], pd.Series):
        list_of_weights = [weights.values for weights in list_of_weights]

    results = randomness.choice(sub_pop.index, list_of_keys, np.array(list_of_weights).T)
    new_sim_file.loc[sub_pop.index, 'condition_state'] = results
    return new_sim_file


def assign_cause_at_beginning_of_simulation(simulants_df, year_start, states, randomness, initial_state, interpolation_order):
    simulants_df = simulants_df[['age', 'sex']]

    cause_level_prevalence, prevalence_draws_dictionary = get_cause_level_prevalence(states, year_start)
    # TODO: Should we be using groupby for these loops to ensure that
    # we're not looping over an age/sex combo that does not exist
    post_cause_assignment_population = determine_if_sim_has_cause(simulants_df, cause_level_prevalence,
                                                                  randomness, interpolation_order)
    sequela_proportions = get_sequela_proportions(cause_level_prevalence, states)
    post_sequela_assignment_population = determine_which_seq_diseased_sim_has(sequela_proportions,
                                                                              post_cause_assignment_population,
                                                                              randomness,
                                                                              interpolation_order)
    post_sequela_assignment_population.condition_state = post_sequela_assignment_population.condition_state.fillna(
        initial_state)

    return post_sequela_assignment_population

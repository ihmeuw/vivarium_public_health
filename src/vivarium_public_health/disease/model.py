"""
=================
The Disease Model
=================

This module contains a state machine driver for disease models.  Its primary
function is to provide coordination across a set of disease states and
transitions at simulation initialization and during transitions.

"""
import pandas as pd
import numpy as np

from vivarium.exceptions import VivariumError
from vivarium.framework.state_machine import Machine

from vivarium_public_health.disease import SusceptibleState


class DiseaseModelError(VivariumError):
    pass


class DiseaseModel(Machine):
    def __init__(self, cause, initial_state=None, get_data_functions=None, cause_type="cause", **kwargs):
        super().__init__(cause, **kwargs)
        self.cause = cause
        self.cause_type = cause_type

        if initial_state is not None:
            self.initial_state = initial_state.state_id
        else:
            self.initial_state = self._get_default_initial_state()

        self._get_data_functions = get_data_functions if get_data_functions is not None else {}

    @property
    def name(self):
        return f"disease_model.{self.cause}"

    def setup(self, builder):
        super().setup(builder)

        self.configuration_age_start = builder.configuration.population.age_start
        self.configuration_age_end = builder.configuration.population.age_end

        cause_specific_mortality_rate = self.load_cause_specific_mortality_rate_data(builder)
        self.cause_specific_mortality_rate = builder.lookup.build_table(cause_specific_mortality_rate,
                                                                        key_columns=['sex'],
                                                                        parameter_columns=['age', 'year'])
        builder.value.register_value_modifier('cause_specific_mortality_rate',
                                              self.adjust_cause_specific_mortality_rate,
                                              requires_columns=['age', 'sex'])

        self.population_view = builder.population.get_view(['age', 'sex', self.state_column])
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=[self.state_column],
                                                 requires_columns=['age', 'sex'],
                                                 requires_streams=[f'{self.state_column}_initial_states'])
        self.randomness = builder.randomness.get_stream(f'{self.state_column}_initial_states')

        builder.event.register_listener('time_step', self.on_time_step)
        builder.event.register_listener('time_step__cleanup', self.on_time_step_cleanup)

    def on_initialize_simulants(self, pop_data):
        population = self.population_view.subview(['age', 'sex']).get(pop_data.index)

        assert self.initial_state in {s.state_id for s in self.states}

        # FIXME: this is a hack to figure out whether or not we're at the simulation start based on the fact that the
        #  fertility components create this user data
        if pop_data.user_data['sim_state'] == 'setup':  # simulation start
            if self.configuration_age_start != self.configuration_age_end != 0:
                state_names, weights_bins = self.get_state_weights(pop_data.index, "prevalence")
            else:
                raise NotImplementedError('We do not currently support an age 0 cohort. '
                                          'configuration.population.age_start and configuration.population.age_end '
                                          'cannot both be 0.')

        else:  # on time step
            if pop_data.user_data['age_start'] == pop_data.user_data['age_end'] == 0:
                state_names, weights_bins = self.get_state_weights(pop_data.index, "birth_prevalence")
            else:
                state_names, weights_bins = self.get_state_weights(pop_data.index, "prevalence")

        if state_names and not population.empty:
            # only do this if there are states in the model that supply prevalence data
            population['sex_id'] = population.sex.apply({'Male': 1, 'Female': 2}.get)

            condition_column = self.assign_initial_status_to_simulants(population, state_names, weights_bins,
                                                                       self.randomness.get_draw(population.index))

            condition_column = condition_column.rename(columns={'condition_state': self.state_column})
        else:
            condition_column = pd.Series(self.initial_state, index=population.index, name=self.state_column)
        self.population_view.update(condition_column)

    def on_time_step(self, event):
        self.transition(event.index, event.time)

    def on_time_step_cleanup(self, event):
        self.cleanup(event.index, event.time)

    def load_cause_specific_mortality_rate_data(self, builder):
        if 'cause_specific_mortality_rate' not in self._get_data_functions:
            only_morbid = builder.data.load(f'cause.{self.cause}.restrictions')['yld_only']
            if only_morbid:
                csmr_data = 0
            else:
                csmr_data = builder.data.load(f"{self.cause_type}.{self.cause}.cause_specific_mortality_rate")
        else:
            csmr_data = self._get_data_functions['cause_specific_mortality_rate'](self.cause, builder)
        return csmr_data

    def adjust_cause_specific_mortality_rate(self, index, rate):
        return rate + self.cause_specific_mortality_rate(index)

    def _get_default_initial_state(self):
        susceptible_states = [s for s in self.states if isinstance(s, SusceptibleState)]
        if len(susceptible_states) != 1:
            raise DiseaseModelError("Disease model must have exactly one SusceptibleState.")
        return susceptible_states[0].state_id

    def get_state_weights(self, pop_index, prevalence_type):
        states = [s for s in self.states
                  if hasattr(s, f'{prevalence_type}') and getattr(s, f'{prevalence_type}') is not None]

        if not states:
            return states, None

        weights = [getattr(s, f'{prevalence_type}')(pop_index) for s in states]
        for w in weights:
            w.reset_index(inplace=True, drop=True)
        weights += ((1 - np.sum(weights, axis=0)), )

        weights = np.array(weights).T
        weights_bins = np.cumsum(weights, axis=1)

        state_names = [s.state_id for s in states] + [self.initial_state]

        return state_names, weights_bins

    @staticmethod
    def assign_initial_status_to_simulants(simulants_df, state_names, weights_bins, propensities):
        simulants = simulants_df[['age', 'sex']].copy()

        choice_index = (propensities.values[np.newaxis].T > weights_bins).sum(axis=1)
        initial_states = pd.Series(np.array(state_names)[choice_index], index=simulants.index)

        simulants.loc[:, 'condition_state'] = initial_states
        return simulants

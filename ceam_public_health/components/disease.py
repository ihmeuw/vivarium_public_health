# ~/ceam/ceam/framework/disease.py

import os.path
from datetime import timedelta
from functools import partial

import pandas as pd
import numpy as np

from ceam import config

from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam.framework.values import modifies_value, produces_value
from ceam.framework.util import rate_to_probability
from ceam.framework.state_machine import Machine, State, Transition, TransitionSet
import numbers

from collections import defaultdict
from ceam_inputs import get_excess_mortality, get_incidence, get_disease_states, get_proportion, get_etiology_probability


class DiseaseState(State):
    def __init__(self, state_id, disability_weight, dwell_time=0, event_time_column=None, event_count_column=None, condition=None):
        State.__init__(self, state_id)

        self.condition = condition
        self._disability_weight = disability_weight
        self.dwell_time = dwell_time
        if isinstance(self.dwell_time, timedelta):
            self.dwell_time = self.dwell_time.total_seconds()

        if event_time_column:
            self.event_time_column = event_time_column
        else:
            self.event_time_column = self.state_id + '_event_time'

        if event_count_column:
            self.event_count_column = event_count_column
        else:
            self.event_count_column = self.state_id + '_event_count'

    def setup(self, builder):
        columns = [self.condition]
        if self.dwell_time > 0:
            columns += [self.event_time_column]
        if self.event_count_column:
            columns += [self.event_count_column]
        self.population_view = builder.population_view(columns, 'alive')
        self.clock = builder.clock()

    @listens_for('initialize_simulants')
    def load_population_columns(self, event):
        population_size = len(event.index)
        if self.dwell_time > 0:
            self.population_view.update(pd.DataFrame({self.event_time_column: np.zeros(population_size)}, index=event.index))
        self.population_view.update(pd.DataFrame({self.event_count_column: np.zeros(population_size)}, index=event.index))

    def next_state(self, index, population_view):
        if self.dwell_time > 0:
            population = self.population_view.get(index)
            eligible_index = population.loc[population[self.event_time_column] < self.clock().timestamp() - self.dwell_time].index
        else:
            eligible_index = index
        return super(DiseaseState, self).next_state(eligible_index, population_view)

    def _transition_side_effect(self, index):
        pop = self.population_view.get(index)
        
        if self.dwell_time > 0:
            pop[self.event_time_column] = self.clock().timestamp()
        
        pop[self.event_count_column] += 1

        self.population_view.update(pop)

    @modifies_value('metrics')
    def metrics(self, index, metrics):
        population = self.population_view.get(index)
        metrics[self.event_count_column] = population[self.event_count_column].sum()
        return metrics

    @modifies_value('disability_weight')
    def disability_weight(self, index):
        population = self.population_view.get(index)
        return self._disability_weight * (population[self.condition] == self.state_id)


# TODO: Make ExcessMortalityState code more flexible so that it only accepts dataframes and not modelable entity ids
class ExcessMortalityState(DiseaseState):
    def __init__(self, state_id, modelable_entity_id, prevalence_meid=None, prevalence_df=None, **kwargs):
        DiseaseState.__init__(self, state_id, **kwargs)

        self.modelable_entity_id = modelable_entity_id
        if prevalence_meid:
            # We may be calculating initial prevalence based on a different
            # modelable_entity_id than we use for the mortality rate
            self.prevalence_meid = prevalence_meid
        else:
            self.prevalence_meid = modelable_entity_id

        if not prevalence_df.empty:
            # FIXME: What to do with the prevalence rate df from here? EM 11/22
            self.prevalence_df = prevalence_df

    def setup(self, builder):
        self.mortality = builder.rate('excess_mortality.{}'.format(self.state_id))
        self.mortality.source = builder.lookup(get_excess_mortality(self.modelable_entity_id))

        return super(ExcessMortalityState, self).setup(builder)

    @modifies_value('mortality_rate')
    def mortality_rates(self, index, rates):
        population = self.population_view.get(index)
        return rates + self.mortality(population.index) * (population[self.condition] == self.state_id)

    @modifies_value('modelable_entity_ids.mortality')
    def mmeids(self):
        return self.modelable_entity_id

    def name(self):
        if not self.prevalence_df.empty:
            return '{} ({}, {})'.format(self.state_id, self.modelable_entity_id, self.prevalence_df)
        else:
            return '{} ({}, {})'.format(self.state_id, self.modelable_entity_id, self.prevalence_meid)

    def __str__(self):
        return 'ExcessMortalityState("{}", "{}" ...)'.format(self.state_id, self.modelable_entity_id)


class DiarrheaState(ExcessMortalityState):
    def setup(self, builder):

        self.eti_dict = dict()
        self.count_dict = defaultdict(lambda: 0)

        # TODO: Move Chris T's file to somewhere central to cost effectiveness
        self.diarrhea_and_lri_etiologies = pd.read_csv("/home/j/temp/ctroeger/GEMS/eti_rr_me_ids.csv")
        self.diarrhea_only_etiologies = self.diarrhea_and_lri_etiologies.query("cause_id == 302")

        # Clostiridium has its own DisMod full model, so we will not be modeling it as a proportion
        self.diarrhea_only_etiologies = self.diarrhea_only_etiologies.query("modelable_entity != 'diarrhea_clostridium'")

        # Line below removes "diarrhea_" from the string, since I'd rather be able to fee in just the etiology name (e.g. "rotavirus" instead of "diarrhea_rotavirus")
        self.diarrhea_only_etiologies['modelable_entity'] = self.diarrhea_only_etiologies['modelable_entity'].map(lambda x: x.split('_', -1)[1])

        for eti in self.diarrhea_only_etiologies.modelable_entity.values:
            self.eti_dict[eti] = builder.lookup(get_etiology_probability(eti))

        super(DiarrheaState, self).setup(builder)
        self.random = builder.randomness("diarrhea")

    @listens_for('initialize_simulants')
    @uses_columns(['cholera', 'salmonella', 'shigellosis', 'epec', 'etec', 'campylobac', 'amoebiasis', 'cryptospor', 'rotavirus', 'aeromonas', 'norovirus', 'adenovirus'])
    def _create_etiology_columns(self, event):
        length = len(event.index)
        falses = np.zeros((length, 12), dtype=bool)
        df = pd.DataFrame(falses, columns=['cholera', 'salmonella', 'shigellosis', 'epec', 'etec', 'campylobac', 'amoebiasis', 'cryptospor', 'rotavirus', 'aeromonas', 'norovirus', 'adenovirus'], index=event.index)
 
        event.population_view.update(df)

    @uses_columns(['cholera', 'salmonella', 'shigellosis', 'epec', 'etec', 'campylobac', 'amoebiasis', 'cryptospor', 'rotavirus', 'aeromonas', 'norovirus', 'adenovirus'])
    def _transition_side_effect(self, index, population_view):
        etiology_cols = pd.DataFrame()

        for eti in self.diarrhea_only_etiologies.modelable_entity.values:
            self.eti_dict[eti](index)
            draw = self.random.get_draw(index)
            etiology = draw < self.eti_dict[eti](index)
            etiology_cols[eti] = etiology
            self.count_dict[eti] += etiology.sum()

        population_view.update(etiology_cols)

    @modifies_value('metrics')
    def metrics(self, index, metrics):
        # TODO: Better way to get counts of each etiology? Since they are a series of bools, figured summing works
        for eti in self.diarrhea_only_etiologies.modelable_entity.values:
            metrics['{}_count'.format(eti)] = self.count_dict[eti]
        return metrics


class RateTransition(Transition):
    def __init__(self, output, rate_label, rate_data, name_prefix='incidence_rate'):
        Transition.__init__(self, output, self.probability)

        self.rate_label = rate_label
        self.rate_data = rate_data
        self.name_prefix = name_prefix

    def setup(self, builder):
        self.incidence_rates = produces_value('{}.{}'.format(self.name_prefix, self.rate_label))(self.incidence_rates)
        self.effective_incidence = builder.rate('{}.{}'.format(self.name_prefix, self.rate_label))
        self.effective_incidence.source = self.incidence_rates
        self.joint_paf = builder.value('paf.{}'.format(self.rate_label))
        self.base_incidence = builder.lookup(self.rate_data)

    def probability(self, index):
        return rate_to_probability(self.effective_incidence(index))

    def incidence_rates(self, index):
        base_rates = self.base_incidence(index)
        joint_mediated_paf = self.joint_paf(index)

        return pd.Series(base_rates.values * joint_mediated_paf.values, index=index)

    def __str__(self):
        return 'RateTransition("{0}", "{1}")'.format(self.output.state_id if hasattr(self.output, 'state_id') else [str(x) for x in self.output], self.rate_label)


class RemissionRateTransition(Transition):
    def __init__(self, output, rate_label, modelable_entity_id):
        Transition.__init__(self, output, self.probability)

        self.rate_label = rate_label
        self.modelable_entity_id = modelable_entity_id

    # TODO: Think about how risks and remission works. Is there mediation? 
    def setup(self, builder):
        self.remission_rates = produces_value('remission_rate.{}'.format(self.rate_label))(self.remission_rates)
        self.effective_remission = builder.rate('remission_rate.{}'.format(self.rate_label))
        self.effective_remission.source = self.remission_rates
        self.base_remission = builder.lookup(get_remission(self.modelable_entity_id))

    def probability(self, index):
        return rate_to_probability(self.effective_remission(index))

    def remission_rates(self, index):
        base_rates = self.base_remission(index)

        return base_rates

    def __str__(self):
        return 'RemissionRateTransition("{0}", "{1}", "{2}")'.format(self.output.state_id if hasattr(self.output, 'state_id') else [str(x) for x in self.output], self.rate_label, self.modelable_entity_id)


class ProportionTransition(Transition):
    def __init__(self, output, modelable_entity_id=None, proportion=None):
        Transition.__init__(self, output, self.probability)

        if modelable_entity_id and proportion:
            raise ValueError("Must supply modelable_entity_id or proportion (proportion can be an int or df) but not both")

        # @alecwd: had to change line below since it was erroring out when proportion is a dataframe. might be a cleaner way to do this that I don't know of
        if modelable_entity_id is None and proportion is None:
           raise ValueError("Must supply either modelable_entity_id or proportion (proportion can be int or df)")

        self.modelable_entity_id = modelable_entity_id
        self.proportion = proportion

    def setup(self, builder):
        if self.modelable_entity_id:
            self.proportion = builder.lookup(get_proportion(self.modelable_entity_id))
        elif not isinstance(self.proportion, numbers.Number):
            self.proportion = builder.lookup(self.proportion)

    def probability(self, index):
        if callable(self.proportion):
            return self.proportion(index)
        else:
            return pd.Series(self.proportion, index=index)

    def label(self):
        if self.modelable_entity_id:
            return str(self.modelable_entity_id)
        else:
            return str(self.proportion)

    def __str__(self):
        return 'ProportionTransition("{}", "{}", "{}")'.format(self.output.state_id if hasattr(self.output, 'state_id') else [str(x) for x in self.output], self.modelable_entity_id, self.proportion)


class DiseaseModel(Machine):
    def __init__(self, condition):
        Machine.__init__(self, condition)

    def module_id(self):
        return str((self.__class__, self.state_column))

    @property    
    def condition(self):
        return self.state_column

    def setup(self, builder):
        self.population_view = builder.population_view([self.condition], 'alive')

        sub_components = set()
        for state in self.states:
            state.condition = self.condition
            sub_components.add(state)
            sub_components.add(state.transition_set)
            for transition in state.transition_set:
                sub_components.add(transition)
                if isinstance(transition.output, TransitionSet):
                    sub_components.add(transition.output)
        return sub_components

    @listens_for('time_step')
    def time_step_handler(self, event):
        self.transition(event.index)


    @listens_for('initialize_simulants')
    @uses_columns(['age', 'sex'])
    def load_population_columns(self, event):
        population = event.population

        # TODO: figure out what "s" is in context below
        # TODO: figure out how to pass a prevalence dataframe into this function
        state_map = {s.state_id:s.prevalence_df for s in self.states if hasattr(s, 'prevalence_df')}

        population['sex_id'] = population.sex.apply({'Male':1, 'Female':2}.get)
        condition_column = get_disease_states(population, state_map)
        condition_column = condition_column.rename(columns={'condition_state': self.condition})

        self.population_view.update(condition_column)

    # @modifies_value('metrics')
    # def metrics(self, index, metrics):
    #    population = self.population_view.get(index)
    #    metrics[self.condition + '_count'] = (population[self.condition] != 'healthy').sum()
    #    return metrics
# End.

"""
========================
The Core Emigration Model
========================

This module contains tools modeling Emigration

"""
import pandas as pd

from vivarium.framework.utilities import rate_to_probability


class Emigration:

    @property
    def name(self):
        return 'emigration'

    def setup(self, builder):
        emigration_data = builder.data.load("covariate.age_specific_migration_rate.estimate")
        self.all_cause_emigration_rate = builder.lookup.build_table(emigration_data, key_columns=['sex', 'location', 'ethnicity'],
                                                                    parameter_columns=['age', 'year'])


        self.emigration_rate = builder.value.register_rate_producer('emigration_rate',
                                                                    source=self.calculate_emigration_rate,
                                                                    requires_columns=['sex','location','ethnicity'])


        self.random = builder.randomness.get_stream('emigration_handler')
        self.clock = builder.time.clock()

        columns_created = ['emigrated']
        view_columns = columns_created + ['alive', 'exit_time', 'age', 'sex', 'location','ethnicity']
        self.population_view = builder.population.get_view(view_columns)
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=columns_created)

        builder.event.register_listener('time_step', self.on_time_step, priority=0)

    def on_initialize_simulants(self, pop_data):
        pop_update = pd.DataFrame({'emigrated': 'no_emigration'},
                                  index=pop_data.index)
        self.population_view.update(pop_update)

    def on_time_step(self, event):
        pop = self.population_view.get(event.index, query="alive =='alive' and sex != 'nan'")
        prob_df = rate_to_probability(pd.DataFrame(self.emigration_rate(pop.index)))
        prob_df['no_emigration'] = 1-prob_df.sum(axis=1)
        prob_df['emigrated'] = self.random.choice(prob_df.index, prob_df.columns, prob_df)
        emigrated_pop = prob_df.query('emigrated != "no_emigration"').copy()

        if not emigrated_pop.empty:
            emigrated_pop['alive'] = pd.Series('emigrated', index=emigrated_pop.index)
            emigrated_pop['emigrated'] = pd.Series('Yes', index=emigrated_pop.index)
            emigrated_pop['exit_time'] = event.time
            self.population_view.update(emigrated_pop[['alive', 'exit_time', 'emigrated']])

    def calculate_emigration_rate(self, index):
        emigration_rate = self.all_cause_emigration_rate(index)
        return pd.DataFrame({'emigrated': emigration_rate})

    def __repr__(self):
        return "Emigration()"

"""
========================
The Core IntegralOutMigration Model
========================

This module contains tools modeling IntegralOutMigration

"""
import pandas as pd

from vivarium.framework.utilities import rate_to_probability


class IntegralOutMigration:

    @property
    def name(self):
        return 'integraloutmigration'

    def setup(self, builder):
        int_outmigration_data = builder.data.load("cause.age_specific_internal_outmigration_rate")
        self.int_out_migration_rate = builder.lookup.build_table(int_outmigration_data, key_columns=['sex', 'location', 'ethnicity'],
                                                                 parameter_columns=['age', 'year'])


        self.int_outmigration_rate = builder.value.register_rate_producer('int_outmigration_rate',
                                                                          source=self.calculate_outmigration_rate,
                                                                          requires_columns=['sex','location','ethnicity'])


        self.random = builder.randomness.get_stream('outmigtation_handler')
        self.clock = builder.time.clock()

        columns_created = ['internal_outmigration','last_outmigration_time']
        view_columns = columns_created + ['alive', 'age', 'sex', 'location','ethnicity']
        self.population_view = builder.population.get_view(view_columns)
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=columns_created)

        builder.event.register_listener('time_step', self.on_time_step, priority=0)

    def on_initialize_simulants(self, pop_data):
        pop_update = pd.DataFrame({'internal_outmigration': 'No',
                                  'last_outmigration_time': pd.NaT},
                                  index=pop_data.index)
        self.population_view.update(pop_update)

    def on_time_step(self, event):
        pop = self.population_view.get(event.index, query="alive =='alive' and sex != 'nan' and internal_outmigration=='No'")
        prob_df = rate_to_probability(pd.DataFrame(self.int_outmigration_rate(pop.index)))
        prob_df['No'] = 1-prob_df.sum(axis=1)
        prob_df['internal_outmigration'] = self.random.choice(prob_df.index, prob_df.columns, prob_df)
        int_outmigrated_pop = prob_df.query('internal_outmigration != "No"').copy()

        if not int_outmigrated_pop.empty:
            int_outmigrated_pop['internal_outmigration'] = pd.Series('Yes', index=int_outmigrated_pop.index)
            int_outmigrated_pop['last_outmigration_time'] = event.time
            self.population_view.update(int_outmigrated_pop[['last_outmigration_time', 'internal_outmigration']])

    def calculate_outmigration_rate(self, index):
        int_out_migration = self.int_out_migration_rate(index)
        return pd.DataFrame({'internal_outmigration': int_out_migration})

    def __repr__(self):
        return "IntegralOutMigration()"

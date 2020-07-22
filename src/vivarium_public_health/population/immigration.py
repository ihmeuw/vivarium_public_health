"""
==========================
The Core Immigration Model
==========================

Currently, we have a deterministic immigration component in which:
- the total number of immigrants is read from a file
- the characteristics of the immigrants are sampled from the migration rate file

"""
import pandas as pd

from vivarium_public_health import utilities


class ImmigrationDeterministic:

    @property
    def name(self):
        return "deterministic_immigration"

    def setup(self, builder):
        self.fractional_new_immigrations = 0
        # read rates and total number of immigrants
        self.asfr_data_immigration = builder.data.load("cause.all_causes.cause_specific_immigration_rate") 
        self.simulants_per_year = builder.data.load("cause.all_causes.cause_specific_total_immigrants_per_year") 

        self.simulant_creator = builder.population.get_simulant_creator()
        self.population_view = builder.population.get_view(['immigrated', 'sex', 'ethnicity', 'location', 'age'])
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=["immigrated"])
        builder.event.register_listener('time_step', self.on_time_step)

    def on_initialize_simulants(self, pop_data):
        if pop_data.user_data['sim_state'] == 'time_step_imm':
            pop_update = pd.DataFrame({'immigrated': 'Yes'},
                                    index=pop_data.index)
        else:
            pop_update = pd.DataFrame({'immigrated': 'no_immigration'},
                                    index=pop_data.index)


        self.population_view.update(pop_update)

    def on_time_step(self, event):
        """Adds a set number of simulants to the population each time step.

        Parameters
        ----------
        event
            The event that triggered the function call.
        """
        # Assume immigrants are uniformly distributed throughout the year.
        step_size = utilities.to_years(event.step_size)
        simulants_to_add = self.simulants_per_year*step_size + self.fractional_new_immigrations

        self.fractional_new_immigrations = simulants_to_add % 1
        simulants_to_add = int(simulants_to_add)

        if simulants_to_add > 0:
            self.simulant_creator(simulants_to_add,
                                  population_configuration={
                                      'age_start': 0,
                                      'age_end': 100,
                                      'sim_state': 'time_step_imm',
                                      'immigrated': "Yes"
                                  })
        
        # XXX make sure this does not conflict with fertility XXX
        new_residents = self.population_view.get(event.index, query='sex == "nan"')
        
        new_residents = new_residents.query('immigrated != "no_immigration"').copy()
        if len(new_residents) > 0:
            # sample residents using the immigration rates
            sample_resident = self.asfr_data_immigration.sample(len(new_residents), weights="mean_value", replace=True)
            new_residents["sex"] = sample_resident["sex"].values.astype(float)
            new_residents["ethnicity"] = sample_resident["ethnicity"].values
            new_residents["location"] = sample_resident["location"].values
            new_residents["age"] = sample_resident["age_start"].values.astype(float)
            new_residents["immigrated"] = "Yes"

            self.population_view.update(new_residents[['immigrated', 'location', 'ethnicity', 'sex', 'age']])


    def __repr__(self):
        return "ImmigrationDeterministic()"
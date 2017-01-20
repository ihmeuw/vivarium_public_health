import pandas as pd
import numpy as np
from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam.framework.values import modifies_value

class RotaVaccine():
    def __init__(self, active=True):
        self.active = active
        self.etiology = 'rotaviral_entiritis'
        self.etiology_column = 'diarrhea_due_to_' + self.etiology
        self.vaccine_column = self.etiology + "_vaccine"

    def setup(self, builder):
        columns = [self.vaccine_column, self.etiology_column]
        self.population_view = builder.population_view(columns, query='alive')

    @listens_for('initialize_simulants')
    def load_population_columns(self, event):
        self.population_view.update(pd.DataFrame({self.vaccine_column: np.zeros(len(event.index), dtype=int)}))

    # TODO: Figure out which priority to make intervention. Using 9 for now since that's what Alec used for the ORS intervention
    @listens_for('time_step__prepare', priority=9)
    def _determine_who_gets_vaccine(self, event):
        population = self.population_view.get(event.index)
        population[self.vaccine_column] = 1

    @modifies_value('incidence_rate.incidence_rate.diarrhea_due_to_rotaviral_entiritis')
    @uses_columns(['diarrhea_due_to_rotaviral_entiritis'], 'alive')
    def mortality_rates(self, index, rates, population_view):
        population = self.population_view.get(index)

        if self.active == True:
            return rates * 0 * (population[self.vaccine_column] == 1)

        else:
            return rates
# End.


         
    

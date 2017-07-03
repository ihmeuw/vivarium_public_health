import pandas as pd

from ceam import config
from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns

from ceam_inputs import get_diarrhea_costs, get_ors_costs


class HealthcareAccess:

    def setup(self, _):
        if config.ors.run_intervention:
            self.costs = get_ors_costs()
        else:
            self.costs = get_diarrhea_costs()

    @listens_for('initialize_simulants')
    @uses_columns(['healthcare_access_cost'])
    def load_population_columns(self, event):
        event.population_view.update(pd.Series(0., index=event.index, name='healthcare_access_cost'))

    @listens_for('time_step', priority=6)
    @uses_columns(['care_sought_event_time', 'healthcare_access_cost'], "alive == 'alive")
    def time_step(self, event):
        patients = event.population[event.population['care_sought_event_time'] == event.time]
        patients['healthcare_access_cost'] += float(self.costs[self.costs.year == event.time.year])
        event.population_view.update(patients)

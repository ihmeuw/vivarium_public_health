import pandas as pd

from vivarium import config
from vivarium.framework.event import listens_for
from vivarium.framework.values import modifies_value
from vivarium.framework.population import uses_columns

from ceam_inputs import get_diarrhea_costs, get_ors_costs


class HealthcareAccess:

    def setup(self, builder):
        if config.ors.run_intervention:
            self.costs = get_ors_costs()
        else:
            self.costs = get_diarrhea_costs()

    @listens_for('initialize_simulants')
    @uses_columns(['healthcare_access_cost'])
    def load_population_columns(self, event):
        event.population_view.update(pd.Series(0., index=event.index, name='healthcare_access_cost'))

    @listens_for('time_step', priority=6)
    @uses_columns(['care_sought_event_time', 'healthcare_access_cost'], "alive == 'alive'")
    def time_step(self, event):
        patients = event.population[event.population['care_sought_event_time'] == event.time]
        patients.loc[:, 'healthcare_access_cost'] += float(self.costs[self.costs.year == event.time.year].cost)
        event.population_view.update(patients)

    @modifies_value('metrics')
    @uses_columns(['healthcare_access_cost'])
    def metrics(self, index, metrics, population_view):
        pop = population_view.get(index)
        metrics['healthcare_access_cost'] = pop['healthcare_access_cost'].sum()
        return metrics

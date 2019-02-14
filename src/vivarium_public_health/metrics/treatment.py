import pandas as pd


class TreatmentObserver:
    """ An observer for treaement counts by each dose and each simulation year.
    This component by default have 4 different doses for a given treatment.
    To change it, configuration for the case of 'shigellosis_vaccine'
    should be given as e.g.,

    metrics:
        shigellosis_vaccine_observer:
            doses: ['first', 'second']

    """

    configuration_defaults = {
        'metrics':{
            'treatment_observer':{
                'doses': ['first', 'second', 'booster', 'catchup']
            }
        }
    }

    def __init__(self, treatment: str):
        self.treatment = treatment
        self.name = f'treatment_observer'
        self.configuration_defaults = {'metrics': {
            f'{self.treatment}_observer': TreatmentObserver.configuration_defaults['metrics']['treatment_observer']
        }}

    def setup(self, builder):
        columns_required = ['tracked', 'alive', f'{self.treatment}_current_dose_event_time',
                            f'{self.treatment}_current_dose' ]
        self.population_view = builder.population.get_view(columns_required)
        self.doses = self.configuration_defaults['metrics'][f'{self.treatment}_observer']['doses']
        years = range(builder.configuration.time.start.year, builder.configuration.time.end.year + 1)
        self.data = pd.DataFrame({f'{dose}_counts': 0 for dose in self.doses}, index=years)
        builder.value.register_value_modifier('metrics', self.metrics)
        builder.event.register_listener('collect_metrics', self.on_collect_metrics)

    def on_collect_metrics(self, event):
        pop = self.population_view.get(event.index)
        for dose in self.doses:
            pop_received_dose = pop[(pop[f'{self.treatment}_current_dose'] == dose) &
                                    (pop[f'{self.treatment}_current_dose_event_time'] == event.time)]
            self.data.loc[event.time.year, f'{dose}_counts'] += len(pop_received_dose)

    def metrics(self, index, metrics):
        for label, counts in self.data.iteritems():
            for year in counts.index:
                metrics[f'{self.treatment}_{label}_in_{year}'] = counts.loc[year]
        return metrics

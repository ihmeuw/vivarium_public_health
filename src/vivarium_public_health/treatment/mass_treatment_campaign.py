import pandas as pd

from .base_treatment import Treatment
from .schedule import TreatmentSchedule


class MassTreatmentCampaign:

    configuration_defaults = {
        'treatment': {
            'doses': ['first', 'second', 'booster', 'catchup'],
            'dose_response': {
                'onset_delay': 14,  # Days
                'duration': 720,  # Days
                'waning_rate': 0.038  # Percent/Day
            },
            'protection': {
                'efficacy': {
                    'mean': 0.5,
                    'standard_error': 0.11,
                },
                'dose_protection': {
                    'first': 0.7,
                    'second': 1.0,
                    'booster': 1.0,
                    'catchup': 0.7,
                },
            },
            'dose_age_range': {
                'first': {
                    'start': 270,
                    'end': 360,
                },
                'second': {
                    'start': 450,
                    'end': 513,
                },
                'booster': {
                    'start': 1080,
                    'end': 1440,
                },
                'catchup': {
                    'start': 1080,
                    'end': 1440,
                },
            },
            'coverage_proportion': {
                'second': 0.8,
                'booster': 0.9,
                'catchup': 0.1,
            },
        }
    }

    def __init__(self, treatment_name, etiology):
        self.treatment_name = treatment_name
        self.configuration_defaults = {treatment_name: MassTreatmentCampaign.configuration_defaults['treatment']}
        self.treatment = Treatment(treatment_name, etiology)
        self.schedule = TreatmentSchedule(treatment_name)

    def setup(self, builder):
        builder.components.add_components([self.treatment, self.schedule])
        self.config = builder.configuration[self.treatment_name]
        self.clock = builder.time.clock()

        columns = [f'{self.treatment.name}_current_dose',
                   f'{self.treatment.name}_current_dose_event_time',
                   f'{self.treatment.name}_previous_dose',
                   f'{self.treatment.name}_previous_dose_event_time']
        self.population_view = builder.population.get_view(['age', 'alive']+columns)
        builder.population.initializes_simulants(self.load_population_columns, creates_columns=columns)
        builder.value.register_value_modifier('metrics', modifier=self.metrics)
        builder.event.register_listener('time_step', self.administer_treatment)

        return

    def load_population_columns(self, event):
        """Adds this components columns to the simulation state table.

        Parameters
        ----------
        event : vivarium.framework.builder.Builder.event
            An event signaling the creation of new simulants.
            - unless there's fertility or migration, it only happens once at the beginning of simulation
        """
        self.population_view.update(pd.DataFrame({
            f'{self.treatment.name}_current_dose': None,
            f'{self.treatment.name}_current_dose_event_time': pd.NaT,
            f'{self.treatment.name}_previous_dose': None,
            f'{self.treatment.name}_previous_dose_event_time': pd.NaT,
        }, index=event.index))

    def administer_treatment(self, event):
        population = self.population_view.get(event.index, 'alive' == True)
        for dose in self.schedule.doses:
            dosed_population = self.schedule.get_newly_dosed_simulants(dose, population, event.step_size)

            dosed_population[f'{self.treatment.name}_previous_dose'] = dosed_population[
                f'{self.treatment.name}_current_dose']
            dosed_population[f'{self.treatment.name}_previous_dose_event_time'] = dosed_population[
                f'{self.treatment.name}_current_dose_event_time']

            dosed_population[f'{self.treatment.name}_current_dose'] = dose
            dosed_population[f'{self.treatment.name}_current_dose_event_time'] = event.time

            self.population_view.update(dosed_population)

    def metrics(self, index, metrics):
        population = self.population_view.get(index)
        current_dose = population[f'{self.treatment.name}_current_dose'].value_counts().to_dict()
        previous_dose = population[f'{self.treatment.name}_previous_dose'].value_counts().to_dict()
        for dose in self.schedule.doses:
            metrics[f'{self.treatment.name}_{dose}_dose_current_count'] = current_dose.get(dose)
            metrics[f'{self.treatment.name}_{dose}_dose_previous_count'] = previous_dose.get(dose)
        return metrics

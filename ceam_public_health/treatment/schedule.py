import numpy as np
import pandas as pd
from scipy import stats


class TreatmentSchedule:

    def __init__(self, name):
        self.name = name
        self._schedule = pd.DataFrame()

    def setup(self, builder):
        coverages = self._get_coverage(builder)
        self.dose_coverages = {dose: builder.lookup(dose_coverage, key_columns=(), parameter_columns=('year',))
                               for dose, dose_coverage in coverages.items()}
        self.doses = builder.configuration[self.name].doses
        self.dose_ages = builder.configuration[self.name].dose_age_range.to_dict()

        self.randomness = builder.randomness.get_stream(f'{self.name}_dosing')
        self.population_view = builder.population.get_view(['age'])
        builder.population.initializes_simulants(self.add_simulants)

    def _get_coverage(self, builder):
        return self.get_coverage(builder)

    def _determine_dose_eligibility(self, current_schedule, dose, index):
        return self.determine_dose_eligibility(current_schedule, dose, index)

    @staticmethod
    def get_coverage(builder):
        raise NotImplementedError("No coverage function supplied.")

    @staticmethod
    def determine_dose_eligibility(current_schedule, dose, index):
        raise NotImplementedError

    def add_simulants(self, index):
        self._schedule.append(self.determine_who_should_receive_dose_and_when(index))

    def determine_who_should_receive_dose_and_when(self, index):
        """Determine who/when will get each dose and record it in the self.vaccination DataFrame
        Parameters:
        vaccination: data frame having the same index as newly added simulants. By default filled with False/NaT

        Returns
        -------
        pd.DataFrame : vaccination
            which includes all the information about who will get each dose and at which age it
            will be given

        """
        schedule = {dose: False for dose in self.doses}
        schedule.update({f'{dose}_age': np.NaN for dose in self.doses})
        schedule = pd.DataFrame(schedule, index=index)

        population = self.population_view.get(index)
        for dose in self.doses:
            coverage = self.dose_coverages[dose](population)
            eligible_index = self._determine_dose_eligibility(schedule, dose, index)

            coverage_draw = self.randomness.get_draw(index, additional_key=f'{dose}_covered')
            dosed_index = eligible_index[coverage_draw < coverage]

            age_draw = self.randomness.get_draw(index, additional_key=f'{dose}_age')
            min_age, max_age = self.dose_ages[dose]
            mean_age = (min_age + max_age) / 2
            age_std_dev = (mean_age - min_age) / 3
            age_at_dose = stats.norm(mean_age, age_std_dev**2).ppf(age_draw)
            age_at_dose[age_at_dose > max_age] = max_age
            age_at_dose[age_at_dose < min_age] = min_age

            schedule.loc[dosed_index, dose] = True
            schedule.loc[dosed_index, f'{dose}_age'] = age_at_dose

        return schedule

    def get_newly_dosed_simulants(self, dose, population, step_size):
        eligible_pop = population[self._schedule[dose]]
        dose_age = self._schedule.loc[eligible_pop.index, f'{dose}_age']

        time_to_dose = eligible_pop.age * 365 + step_size.days - dose_age
        correct_age = np.abs(time_to_dose) < step_size.days / 2
        return eligible_pop[correct_age]

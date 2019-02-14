import pandas as pd
from .utilities import get_age_bins


class DiseaseObserver:
    """ An observer for disease cases and person time susceptible to
    the specific disease during simulation. This component by default
    observes the total number of disease cases and total sum of susceptible
    person time by each demographic group (age and sex) by each simulation
    year.

    """
    def __init__(self, disease: str):
        self.disease = disease
        self.name = 'disease_observer'

    def setup(self, builder):
        self.age_bins = get_age_bins(builder)
        columns_required = ['tracked', 'alive', 'age', 'sex', f'{self.disease}', f'{self.disease}_event_time']
        self.population_view = builder.population.get_view(columns_required)

        years = range(builder.configuration.time.start.year, builder.configuration.time.end.year+1)
        frame_index = pd.MultiIndex.from_product([self.age_bins.index, years, ['Male', 'Female']],
                                                 names=['age_group', 'year', 'sex'])
        self.frame = pd.DataFrame({f'{self.disease}_cases': 0, 'susceptible_person_time': 0}, index=frame_index)

        builder.value.register_value_modifier('metrics', self.metrics)
        builder.event.register_listener('collect_metrics', self.on_collect_metrics)

    def on_collect_metrics(self, event):
        pop = self.population_view.get(event.index)
        pop = pop[pop.alive == 'alive']
        for group, age_group in self.age_bins.iterrows():
            start, end = age_group.age_group_start, age_group.age_group_end
            for sex in ['Male', 'Female']:
                in_age_group = pop[(pop.age >= start) & (pop.age < end) & (pop.sex == sex)]
                new_disease_cases = in_age_group[in_age_group[f'{self.disease}_event_time'] == event.time]
                susceptible_pop = in_age_group[in_age_group[f'{self.disease}'] == f'susceptible_to_{self.disease}']
                self.frame.loc[(group, event.time.year, sex), f'{self.disease}_cases'] += len(new_disease_cases)
                self.frame.loc[(group, event.time.year, sex), 'susceptible_person_time'] += \
                    (event.step_size / pd.Timedelta(days=365.25)) * len(susceptible_pop)

    def metrics(self, index, metrics):
        data = self.frame.to_dict(orient='index')
        for key, value in data.items():
            age_id, year, sex = key
            age_group_name = self.age_bins.age_group_name.loc[age_id].replace(" ", "_").lower()
            case_label = f'{self.disease}_cases_for_{sex}_in_age_group_{age_group_name}_in_year_{year}'
            person_time_label = f'{self.disease}_susceptible_person_time_{sex}_in_age_group_{age_group_name}_in_{year}'
            metrics[case_label] = value[f'{self.disease}_cases']
            metrics[person_time_label] = value['susceptible_person_time']
        return metrics

from vivarium_public_health.risks.data_transformations import TargetString


class AbsoluteShift:
    configuration_defaults = {
        'intervention': {
            'target_value': 'baseline',
            'age_start': 0,
            'age_end': 125,
        }
    }

    def __init__(self, target):
        self.target = TargetString(target)
        self.configuration_defaults = {
            f'intervention_on_{self.target.name}': AbsoluteShift.configuration_defaults['intervention']
        }

    def setup(self, builder):
        self.config = builder.configuration[f'intervention_on_{self.target.name}']
        builder.value.register_value_modifier(f'{self.target.name}.{self.target.measure}',
                                              modifier=self.intervention_effect)
        self.population_view = builder.population.get_view(['age'])

    def intervention_effect(self, index, value):
        if self.config['target_value'] != 'baseline':
            pop = self.population_view.get(index)
            affected_group = pop[pop.age.between(self.config['age_start'], self.config['age_end'])]
            value.loc[affected_group.index] = float(self.config['target_value'])
        return value

import pandas as pd
import scipy.stats


class TherapeuticInertia:

    configuration_defaults = {
        'therapeutic_inertia': {
            'triangle_min': 0.65,
            'triangle_max': 0.9,
            'triangle_mode': 0.875
        }
    }

    def setup(self, builder):
        self.name = 'therapeutic_inertia'
        self.therapeutic_inertia_parameters = builder.configuration.therapeutic_inertia

        self.randomness = builder.randomness.get_stream(self.name)

        self._therapeutic_inertia = pd.Series()
        self.therapeutic_inertia = builder.value.register_value_producer('therapeutic_inertia',
                                                                         source=lambda index:
                                                                         self._therapeutic_inertia.loc[index])

        builder.population.initializes_simulants(self.on_initialize_simulants)

    def on_initialize_simulants(self, pop_data):
        self._therapeutic_inertia = self._therapeutic_inertia.append(pd.Series(self.initialize_therapeutic_inertia(pop_data.index),
                                                                               index=pop_data.index))

    def initialize_therapeutic_inertia(self, index):
        triangle_min = self.therapeutic_inertia_parameters.triangle_min
        triangle_max = self.therapeutic_inertia_parameters.triangle_max
        triangle_mode = self.therapeutic_inertia_parameters.triangle_mode

        # convert to scipy params
        loc = triangle_min
        scale = triangle_max - triangle_min
        c = (triangle_mode - loc) / scale

        draw = self.randomness.get_draw(index, additional_key='individual_draw')
        therapeutic_inertia = scipy.stats.triang(c, loc=loc, scale=scale).ppf(draw)

        return therapeutic_inertia



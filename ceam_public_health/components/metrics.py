import os.path

import pandas as pd
import numpy as np

from ceam import config

from ceam.framework.values import modifies_value
from ceam.framework.event import listens_for
from ceam.framework.values import list_combiner, joint_value_post_processor, rescale_post_processor

class Metrics:
    """
    Accumulate various metrics as the simulation runs.
    """
    def setup(self, builder):
        self.years_lived_with_disability = None
        self.disability_weight = builder.value('disability_weight', list_combiner, lambda a: rescale_post_processor(joint_value_post_processor(a)))
        self.disability_weight.source = lambda index: [pd.Series(0.0, index=index)]

        metrics = builder.value('metrics')
        metrics.source = lambda index: {}

    @listens_for('time_step__cleanup')
    def calculate_ylds(self, event):
        time_step = config.simulation_parameters.time_step
        if self.years_lived_with_disability is None:
            self.years_lived_with_disability = pd.Series(0, index=event.index)
        self.years_lived_with_disability += self.disability_weight(event.index)

    @modifies_value('metrics')
    def metrics(self, index, metrics):
        if self.years_lived_with_disability is not None:
            metrics['years_lived_with_disability'] = self.years_lived_with_disability.sum()
        else:
            metrics['years_lived_with_disability'] = 0
        return metrics

import os
from ceam import config
try:
    config.reset_layer('override', preserve_keys=['input_data.intermediary_data_cache_path', 'input_data.auxiliary_data_folder'])
except KeyError:
    pass
config.simulation_parameters.set_with_metadata('year_start', 2005, layer='override', source=os.path.realpath(__file__))
config.simulation_parameters.set_with_metadata('year_end', 2010, layer='override', source=os.path.realpath(__file__))
config.simulation_parameters.set_with_metadata('time_step', 1, layer='override', source=os.path.realpath(__file__))
config.simulation_parameters.set_with_metadata('initial_age', None, layer='override', source=os.path.realpath(__file__))
config.simulation_parameters.set_with_metadata('num_simulants', 1000, layer='override', source=os.path.realpath(__file__))



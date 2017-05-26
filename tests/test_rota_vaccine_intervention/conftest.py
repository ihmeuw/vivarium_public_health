import os
from ceam import config
# Remove user overrides but keep custom cache locations if any
try:
    config.reset_layer('override', preserve_keys=['input_data.intermediary_data_cache_path', 'input_data.auxiliary_data_folder'])
except KeyError:
    pass

config.simulation_parameters.set_with_metadata('year_start', 2005, layer='override', source=os.path.realpath(__file__))
config.simulation_parameters.set_with_metadata('year_end', 2010, layer='override', source=os.path.realpath(__file__))
config.simulation_parameters.set_with_metadata('time_step', 1, layer='override', source=os.path.realpath(__file__))
config.simulation_parameters.set_with_metadata('initial_age', 0, layer='override', source=os.path.realpath(__file__))

config.rota_vaccine.set_with_metadata('age_at_first_dose', 6, layer='override', source=os.path.realpath(__file__))
config.rota_vaccine.set_with_metadata('age_at_second_dose', 12, layer='override', source=os.path.realpath(__file__))
config.rota_vaccine.set_with_metadata('age_at_third_dose', 18, layer='override', source=os.path.realpath(__file__))
config.rota_vaccine.set_with_metadata('time_after_dose_at_which_immunity_is_conferred', 1,
                                      layer='override', source=os.path.realpath(__file__))
config.rota_vaccine.set_with_metadata('vaccine_full_immunity_duration', 20,
                                      layer='override', source=os.path.realpath(__file__))
config.rota_vaccine.set_with_metadata('waning_immunity_time', 20, layer='override', source=os.path.realpath(__file__))
config.rota_vaccine.set_with_metadata('vaccination_proportion_increase', 0.1, layer='override', source=os.path.realpath(__file__))
config.rota_vaccine.set_with_metadata('second_dose_retention', 1, layer='override', source=os.path.realpath(__file__))
config.rota_vaccine.set_with_metadata('third_dose_retention', 1, layer='override', source=os.path.realpath(__file__))


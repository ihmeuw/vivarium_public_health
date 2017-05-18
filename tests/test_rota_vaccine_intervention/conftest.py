from ceam import config

# Remove user overrides but keep custom cache locations if any
config.reset_layer('override', preserve_keys=['input_data.intermediary_data_cache_path', 'input_data.auxiliary_data_folder'])

config.rota_vaccine.age_at_first_dose= 6
config.rota_vaccine.age_at_second_dose = 12
config.rota_vaccine.age_at_third_dose = 18
config.rota_vaccine.time_after_dose_at_which_immunity_is_conferred = 1
config.rota_vaccine.vaccine_full_immunity_duration = 20
config.rota_vaccine.waning_immunity_time = 20
config.simulation_parameters.time_step = 1
config.simulation_parameters.year_start = 2005
config.simulation_parameters.initial_age = 0
config.rota_vaccine.vaccination_proportion_increase = .1
config.rota_vaccine.third_dose_retention = 1
config.rota_vaccine.second_dose_retention = 1

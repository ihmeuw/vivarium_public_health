
def get_age_bins(builder):
    age_bins = builder.data.load('population.age_bins')
    exit_age = builder.configuration.population.exit_age
    if exit_age:
        age_bins = age_bins[age_bins.age_group_start < exit_age]
        age_bins.loc[age_bins.age_group_end > exit_age, 'age_group_end'] = exit_age
    return age_bins

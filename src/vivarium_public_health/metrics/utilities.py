from string import Template

def get_age_bins(builder):
    age_bins = builder.data.load('population.age_bins')
    exit_age = builder.configuration.population.exit_age
    if exit_age:
        age_bins = age_bins[age_bins.age_group_start < exit_age]
        age_bins.loc[age_bins.age_group_end > exit_age, 'age_group_end'] = exit_age
    return age_bins


def get_output_template(by_age, by_sex, by_year):
    template = '{measure}'
    if by_year:
        template += '_in_{year}'
    if by_sex:
        template += '_among_{sex}'
    if by_age:
        template += 'in_age_group_{age_group}'
    return Template(template)

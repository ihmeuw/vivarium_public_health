from ceam_inputs import get_fpg_distributions


def distribution_loader(builder):
    return builder.lookup(get_fpg_distributions(), key_columns=('sex', 'location'))

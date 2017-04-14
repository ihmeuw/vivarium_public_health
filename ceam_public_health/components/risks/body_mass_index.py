from ceam_inputs import get_bmi_distributions

def distribution_loader(builder):
    return builder.lookup(get_bmi_distributions())

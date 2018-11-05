"""Provide components to represent common forms of intervention."""


class ModifyDiseaseRate:
    """Interventions that modify a rate associated with a chronic disease."""
    def __init__(self, name, disease, rate):
        self.name = name
        self.disease = disease
        self.rate = rate

    def setup(self, builder):
        self.config = builder.configuration
        # NOTE: this will be replaced by an (age, sex, year) lookup-table.
        scale_name = "{}_{}_scale".format(self.disease, self.rate)
        self.scale = self.config.intervention[self.name][scale_name]
        if self.scale < 0:
            raise ValueError('Invalid scale: {}'.format(self.scale))
        rate_name = '{}_intervention.{}'.format(self.disease, self.rate)
        builder.value.register_value_modifier(rate_name, self.adjust_rate)

    def adjust_rate(self, index, rates):
        return rates * self.scale


class ModifyDiseaseIncidence(ModifyDiseaseRate):
    """
    Interventions that modify a disease incidence rate, based on a PIF lookup
    table.
    """

    def __init__(self, name, disease):
        super().__init__(name=name, disease=disease, rate='incidence')


class ModifyDiseaseMortality(ModifyDiseaseRate):
    """
    Interventions that modify a disease fatality rate, based on a PIF lookup
    table.
    """

    def __init__(self, name, disease):
        super().__init__(name=name, disease=disease, rate='excess_mortality')


class ModifyDiseaseMorbidity(ModifyDiseaseRate):
    """
    Interventions that modify a disease disability rate, based on a PIF lookup
    table.
    """

    def __init__(self, name, disease):
        super().__init__(name=name, disease=disease, rate='yld_rate')


class ModifyAcuteDiseaseIncidence:
    """
    Interventions that modify an acute disease incidence rate.

    Note that this intervention will simply modify both the disability rate
    and the mortality rate for the chosen acute disease.
    """

    def __init__(self, name):
        self.name = name

    def setup(self, builder):
        self.config = builder.configuration
        self.scale = self.config.intervention[self.name].incidence_scale
        if self.scale < 0:
            raise ValueError('Invalid incidence scale: {}'.format(self.scale))
        yld_rate = '{}_intervention.yld_rate'.format(self.name)
        builder.value.register_value_modifier(yld_rate, self.rate_adjustment)
        mort_rate = '{}_intervention.excess_mortality'.format(self.name)
        builder.value.register_value_modifier(mort_rate, self.rate_adjustment)

    def rate_adjustment(self, index, rates):
        return rates * self.scale


class ModifyAcuteDiseaseMorbidity:
    """Interventions that modify an acute disease disability rate."""

    def __init__(self, name):
        self.name = name

    def setup(self, builder):
        self.config = builder.configuration
        self.scale = self.config.intervention[self.name].yld_scale
        if self.scale < 0:
            raise ValueError('Invalid YLD scale: {}'.format(self.scale))
        rate = '{}_intervention.yld_rate'.format(self.name)
        builder.value.register_value_modifier(rate, self.disability_adjustment)

    def disability_adjustment(self, index, rates):
        return rates * self.scale


class ModifyAcuteDiseaseMortality:
    """Interventions that modify an acute disease fatality rate."""

    def __init__(self, name):
        self.name = name

    def setup(self, builder):
        self.config = builder.configuration
        self.scale = self.config.intervention[self.name].mortality_scale
        if self.scale < 0:
            raise ValueError('Invalid mortality scale: {}'.format(self.scale))
        rate = '{}_intervention.excess_mortality'.format(self.name)
        builder.value.register_value_modifier(rate, self.mortality_adjustment)

    def mortality_adjustment(self, index, rates):
        return rates * self.scale

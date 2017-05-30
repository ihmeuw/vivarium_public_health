import pandas as pd

from ceam.framework.event import listens_for

from ceam_inputs import get_exposures

from ceam_public_health.util.risk import categorical_exposure_effect, make_risk_effects

# TODO: Change 'susceptibility' to propensity

class CategoricalRiskHandler:
    """
    Model a categorical risk. Simulants will be in a specific exposure category based on their `categorical_risk_susceptibility`.

    Population Columns
    ------------------
    categorical_risk_susceptibility        
    """
    def __init__(self, risk_id, risk_name):
        self.risk_id = risk_id
        self.risk_name = risk_name

    def setup(self, builder):

        # get the population table to add the susceptibility column
        column = [self.risk_name + '_susceptibility']
        self.population_view = builder.population_view(column)

        self.exposure = builder.value('{}.exposure'.format(self.risk_name))

        self.exposure.source = builder.lookup(get_exposures(risk_id=self.risk_id))

        self.randomness = builder.randomness(self.risk_name)

        list_of_etiologies = ['diarrhea_due_to_shigellosis',
                              'diarrhea_due_to_cholera',
                              'diarrhea_due_to_other_salmonella',
                              'diarrhea_due_to_EPEC',
                              'diarrhea_due_to_ETEC',
                              'diarrhea_due_to_campylobacter',
                              'diarrhea_due_to_amoebiasis',
                              'diarrhea_due_to_cryptosporidiosis',
                              'diarrhea_due_to_rotaviral_entiritis',
                              'diarrhea_due_to_aeromonas',
                              'diarrhea_due_to_clostridium_difficile',
                              'diarrhea_due_to_norovirus',
                              'diarrhea_due_to_adenovirus']

        list_of_tuples = [(302, i) for i in list_of_etiologies]

        effect_function = categorical_exposure_effect(self.exposure, '{}_susceptibility'.format(self.risk_name))
        risk_effects = make_risk_effects(self.risk_id, list_of_tuples, effect_function, self.risk_name)

        return risk_effects

    @listens_for('initialize_simulants')
    def load_population_columns(self, event):
        self.population_view.update(pd.Series(self.randomness.get_draw(event.index), name='{}_susceptibility'.format(self.risk_name)))

# End.


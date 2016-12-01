from ceam.framework.state_machine import State

from ceam_inputs import get_disability_weight, get_incidence, get_remission
from ceam_public_health.components.disease import DiseaseModel, ExcessMortalityState, RateTransition


def ckd_factory():
    component = DiseaseModel('ckd')

    healthy = State('healthy', key='ckd')

    # NOTE: According to Carrie Purcell, stage three and four only have
    # disability weight in the presence of anemia, which we are not
    # currently modeling. Stage five has it's own weight
    # TODO: The only stage referenced in the weights CSV file is stage four
    # so I'm giving that weight to both stages four and five until I get clarity
    stage_three = ExcessMortalityState('stage_three_ckd', disability_weight=0, modelable_entity_id=2018)
    stage_four = ExcessMortalityState('stage_four_ckd', disability_weight=get_disability_weight(healthstate_id=391), modelable_entity_id=2019)
    stage_five = ExcessMortalityState('stage_five_ckd', disability_weight=get_disability_weight(healthstate_id=391), modelable_entity_id=2022)

    stage_three_transition = RateTransition(stage_three, 'incidence_rate.stage_three_ckd', get_incidence(2018))
    # NOTE: These "incidence rates" come from the remission measure because the
    # CKD model uses the remission from one stage to store the incidence rate
    # for the successor stage. Stage three is different because it's the first
    # stage in our model and people enter it from the general population not
    # from (the non-existent) stage two.
    stage_four_transition = RateTransition(stage_four, 'incidence_rate.stage_four_ckd', get_remission(2018))
    stage_five_transition = RateTransition(stage_five, 'incidence_rate.stage_five_ckd', get_remission(2019))

    healthy.transition_set.extend([stage_three_transition, stage_four_transition, stage_five_transition])
    stage_three.transition_set.extend([stage_four_transition])
    stage_four.transition_set.extend([stage_five_transition])

    component.states.extend([healthy, stage_three, stage_four, stage_five])

    return component

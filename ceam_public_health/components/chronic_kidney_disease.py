from ceam.framework.state_machine import State

from ceam_inputs import get_disability_weight
from ceam_public_health.components.disease import DiseaseModel, ExcessMortalityState, IncidenceRateTransition


def ckd_factory():
    component = DiseaseModel('ihd')

    healthy = State('healthy', key='ckd')

    # NOTE: According to Carrie Purcell, stage three and four only have
    # disability weight in the presence of anemia, which we are not
    # currently modeling. Stage five has it's own weight
    # TODO: The only stage referenced in the weights CSV file is stage four
    # so I'm giving that weight to both stages four and five until I get clarity
    stage_three = ExcessMortalityState('stage_three_ckd', disability_weight=0, modelable_entity_id=2018)
    stage_four = ExcessMortalityState('stage_four_ckd', disability_weight=get_disability_weight(healthstate_id=391), modelable_entity_id=2019)
    stage_five = ExcessMortalityState('stage_five_ckd', disability_weight=get_disability_weight(healthstate_id=391), modelable_entity_id=2022)

    stage_three_transition = IncidenceRateTransition(stage_three, 'stage_three_ckd', modelable_entity_id=2019)
    stage_four_transition = IncidenceRateTransition(stage_three, 'stage_four_ckd', modelable_entity_id=2019)
    stage_five_transition = IncidenceRateTransition(stage_three, 'stage_five_ckd', modelable_entity_id=2022)

    healthy.transition_set.extend([stage_three_transition, stage_four_transition, stage_five_transition])
    stage_three.transition_set.extend([stage_four_transition, stage_five_transition])
    stage_four.transition_set.extend([stage_three_transition, stage_five_transition])
    stage_five.transition_set.extend([stage_three_transition, stage_four_transition])

    component.states.extend([healthy, stage_three, stage_four, stage_five])

    return component

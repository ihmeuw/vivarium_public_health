from ceam import config
from ceam.framework.state_machine import Transition, State, TransitionSet
from ceam_public_health.components.disease import DiseaseModel, DiseaseState, ExcessMortalityState, IncidenceRateTransition, ProportionTransition, RemissionRateTransition, DiarrheaState


def diarrhea_factory():
    """Hello world for diarrhea cost effectiveness analysis"""
    module = DiseaseModel('diarrhea')

    # initialize an object of the State class. object has 2 attributes, state_id and transition_set
    healthy = State('healthy', key='diarrhea')

    # TODO: Need to determine where to put code to aeteological split
    # TODO: Need to employ severity splits (mild, moderate, and severe diarrhea) in the future 
    # FIXME: Figure out what to use for the disability weight
    diarrhea = DiarrheaState('diarrhea', disability_weight=0.1, modelable_entity_id=1181, prevalence_me_id = 1181) 

    diarrhea_transition = IncidenceRateTransition(diarrhea, 'diarrhea', modelable_entity_id=1181)

    healthy.transition_set.extend([diarrhea_transition])
  
    # TODO: After the MVS is finished, include transitions to non-fully healthy states (e.g. malnourished and stunted health states)
    remission_transition = RemissionRateTransition(healthy, 'healthy', modelable_entity_id=1181)

    diarrhea.transition_set.append(Transition(healthy))

    return module



class DiarrheaState(ExcessMortalityState):
    def setup(self, builder):

        # TODO: Move Chris T's file to somewhere central to cost effectiveness
        diarrhea_and_lri_etiologies = pd.read_csv("/home/j/temp/ctroeger/GEMS/eti_rr_me_ids.csv")
        diarrhea_only_etiologies = diarrhea_and_lri_etiologies.query("cause_id == 302")
        
        # Line below removes "diarrhea_" from the string, since I'd rather be able to fee in just the etiology name (e.g. "rotavirus" instead of "diarrhea_rotavirus")
        diarrhea_only_etiologies['modelable_entity'] = diarrhea_only_etiologies['modelable_entity'].map(lambda x: x.split('_', -1)[1])

        for eti in diarrhea_only_etiologies.modelable_entity.values:
            setattr(self, eti, builder.lookup(get_etiology_probability(eti))

        super(DiarrheaState, self).setup(builder)
        self.random = builder.randomness("diarrhea")

    @uses_columns([diarrhea_only_etiologies.modelable_entity.values.tolist()])
    def _transition_side_effect(self, index, population_view):
        etiology_cols = pd.DataFrame()

        for eti in diarrhea_only_etiologies.modelable_entity.values:
            self.eti(index)
            etiology = self.random.choice(index, [True, False], p=self.eti(index))   
            etiology_cols[eti] = etiology

        self.population_view.update(etiology_cols)


# End.

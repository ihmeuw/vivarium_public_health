from ceam import config
from ceam.framework.state_machine import Transition, State, TransitionSet
from ceam_public_health.components.disease import DiseaseModel, DiseaseState, ExcessMortalityState, IncidenceRateTransition, ProportionTransition, RemissionRateTransition


def diarrhea_factory():
    """Hello world for diarrhea cost effectiveness analysis"""
    module = DiseaseModel('diarrhea')

    # initialize an object of the State class. object has 2 attributes, state_id and transition_set
    healthy = State('healthy', key='diarrhea')

    # TODO: Need to determine where to put code to aeteological split
    # TODO: Need to employ severity splits (mild, moderate, and severe diarrhea) in the future 
    # FIXME: Figure out what to use for the disability weight
    diarrhea = ExcessMortalityState('diarrhea', disability_weight=0.1, modelable_entity_id=1181, prevalence_me_id = 1181) 

    diarrhea_transition = IncidenceRateTransition(diarrhea, 'diarrhea', modelable_entity_id=1181)

    healthy.transition_set.extend([diarrhea_transition])
  
    # TODO: What's the best way to assign etiology?
    population_view = assign_diarrhea_etiology(population_view, "rotavirus")
    
    # TODO: After the MVS is finished, include transitions to non-fully healthy states (e.g. malnourished and stunted health states)
    remission_transition = RemissionRateTransition(healthy, 'healthy', modelable_entity_id=1181)

    diarrhea.transition_set.append(Transition(healthy))

    return module



class DiarrheaState(ExcessMortalityState):
    def setup(self, builder):
        self.rotavirus_probability = builder.lookup(get_etiology_proportion("rotavirus")) # get_etiology_proportion should return one draw
        super(DiarrheaState, self).setup(builder)

    @uses_columns(['rotavirus_probability'])
    def _transition_side_effect(self, index):
            self.rotavirus_probability(index)
            self.randomness.choice
   

        for sex_id in pop_with_diarrhea.sex.unique():
        for age in pop_with_diarrhea.age.unique():
            elements = [0, 1]
            probability_of_etiology = etiology_df.\
                query("age=={a} and sex_id=={s}".format(a=age, s=sex_id))[
                    'draw_{}'.format(config.getint('run_configuration', 'draw_number'))]
            probability_of_NOT_etiology = 1 - probability_of_etiology
            weights = [float(probability_of_NOT_etiology),
                       float(probability_of_etiology)]

            one_age = pop_with_diarrhea.query(
                "age=={a} and sex_id=={s}".format(a=age, s=sex_id)).copy()
            one_age['{}'.format(etiology_name)] = one_age['age'].map(
                lambda x: np.random.choice(elements, p=weights))
            new_sim_file = new_sim_file.append(one_age)

    new_sim_file = new_sim_file.append(population_without_diarrhea)

    return new_sim_file.sort_values(by=["simulant_id"])
 


# End.

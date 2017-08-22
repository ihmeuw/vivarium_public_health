from ceam_inputs.gbd_mapping import risk_factors, causes, healthcare_entities
import ceam_inputs as inputs

from ceam_public_health.risks import get_distribution

class PassthroughDatasetManager:
    def __init__(self):
        self.datasets_loaded = set()

    def construct_data_container(self, entity_path):
        self.datasets_loaded.add(entity_path)
        entity_type, entity_name = entity_path.split('.')
        if entity_type == 'risk_factor':
            return RiskDataContainer(risk_factors[entity_name])
        if entity_type == 'cause':
            return CauseDataContainer(causes[entity_name])
        if entity_type == 'auxiliary':
            return AuxiliaryDataContainer(entity_name)
        if entity_type == 'healthcare_entity':
            return AuxiliaryDataContainer(healthcare_entities[entity_name])
        else:
            raise ValueError('Unknown entity type: {}'.format(entity_type))

class _DataContainer:
    def __init__(self, entity):
        self.entity = entity
        self.name = entity if isinstance(entity, str) else entity.name
        self.type = None

class RiskDataContainer(_DataContainer):
    def __init__(self, entity):
        super(RiskDataContainer, self).__init__(entity)
        self.type = 'risk_factor'
        self.tmred = entity.tmred
        self.scale = entity.scale
        self.affected_causes = entity.affected_causes
        self.distribution = entity.distribution

    def get_distribution(self):
        return get_distribution(self.entity)

    def exposure_means(self):
        return inputs.get_exposure_means(risk=self.entity)

    def pafs(self, cause):
        return inputs.get_pafs(risk=self.entity, cause=cause)

    def relative_risks(self, cause):
        return inputs.get_relative_risks(risk=self.entity, cause=cause)

    def mediation_factors(self, cause):
        return inputs.get_mediation_factors(risk=self.entity, cause=cause)

class AuxiliaryDataContainer(_DataContainer):
    def __init__(self, entity):
        super(AuxiliaryDataContainer, self).__init__(entity)
        self.type = 'auxiliary'
        self.data_function = {
                'risk_factor_exposure_correlation_matrices': inputs.load_risk_correlation_matrices,
                'hypertension_drug_costs': inputs.get_hypertension_drug_costs,
                'inpatient_costs': inputs.get_inpatient_visit_costs,
                'outpatient_costs': inputs.get_outpatient_visit_costs,
                'outpatient_visits': lambda: inputs.get_proportion(self.entity),
                'population': inputs.get_populations,
                'subregions': inputs.get_subregions,
                'life_table': inputs.get_life_table,
                'annual_live_births': inputs.get_annual_live_births,
                'age_specific_fertility_rates': inputs.get_age_specific_fertility_rates,
                'age_bins': inputs.get_age_bins,
        }[self.name]

    def data(self, *args, **kwargs):
        return self.data_function(*args, **kwargs)

class CauseDataContainer(_DataContainer):
    def __init__(self, entity):
        super(CauseDataContainer, self).__init__(entity)
        self.type = 'cause'

    def disability_weight(self):
        return inputs.get_disability_weight(self.entity)

    def prevalence(self):
        return inputs.get_prevalence(self.entity)

    def incidence(self):
        return inputs.get_incidence(self.entity)

    def excess_mortality(self):
        return inputs.get_excess_mortality(self.entity)

    def duration(self):
        return inputs.get_duration(self.entity)

    def cause_specific_mortality(self):
        return inputs.get_cause_specific_mortality(self.entity)

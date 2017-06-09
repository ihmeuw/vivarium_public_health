from ceam_public_health.risks import (CategoricalRiskComponent, ContinuousRiskComponent,
                                      get_distribution, get_exposure_function)
from ceam_inputs.gbd_mapping import risk_factors


def factory(cause_list):
    risks = []
    for name, risk in risk_factors.items():
        for cause in risk.effected_causes:
            if cause in cause_list:
                if risk.type == 'continuous':
                    risks.append(ContinuousRiskComponent(risk,
                                                         distribution_loader=get_distribution(name),
                                                         exposure_function=get_exposure_function(name)))
                elif risk.type == 'categorical':
                    risks.append(CategoricalRiskComponent(risk))
                else:
                    raise NotImplementedError("The risk type {} has not been implemented yet".format(risk.type))
                break

    return risks

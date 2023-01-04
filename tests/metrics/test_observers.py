from vivarium_public_health.disease import DiseaseState, RiskAttributableDisease
from vivarium_public_health.metrics.disability import DisabilityObserver


def test_disability_observer_setup(mocker):
    observer = DisabilityObserver()
    builder = mocker.Mock()
    builder.results.register_observation = mocker.Mock()
    builder.components.get_components_by_type = lambda n: []

    builder.results.register_observation.assert_not_called()
    observer.setup(builder)
    builder.results.register_observation.assert_called_once_with(
        name="ylds_due_to_all_causes",
        pop_filter='tracked == True and alive == "alive"',
        aggregator_sources=[str(observer.disability_weight)],
        aggregator=observer._disability_weight_aggregator,
        requires_columns=["alive"],
        requires_values=["disability_weight"],
        additional_stratifications=observer.config.include,
        excluded_stratifications=observer.config.exclude,
        when="time_step__prepare",
    )
    assert DiseaseState in observer.disease_classes
    assert RiskAttributableDisease in observer.disease_classes

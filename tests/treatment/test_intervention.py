import pytest

from vivarium_public_health.treatment.intervention import Intervention, InterventionEffect


def test_intervention_validate_entity_type():
    """Test that Intervention only accepts valid entity types."""
    # Valid entity type should not raise
    Intervention("intervention.test_intervention")

    # Invalid entity types should raise ValueError
    with pytest.raises(ValueError, match="Entity type must be one of"):
        Intervention("risk_factor.test_risk")

    with pytest.raises(ValueError, match="Entity type must be one of"):
        Intervention("cause.some_cause")


def test_intervention_effect_exposure_class():
    """Test that InterventionEffect.EXPOSURE_CLASS is Intervention."""
    assert InterventionEffect.EXPOSURE_CLASS is Intervention

import pytest

from vivarium_public_health.dataset_manager import get_location_term


def test_location_term():
    assert get_location_term("Cote d'Ivoire") == 'location == "Cote d\'Ivoire" | location == "Global"'
    assert get_location_term("Kenya") == "location == 'Kenya' | location == 'Global'"
    with pytest.raises(NotImplementedError):
        get_location_term("W'eird \"location\"")

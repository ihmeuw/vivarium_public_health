import pytest

from vivarium_public_health.risks.implementations.low_birth_weight_and_short_gestation import (
    LBWSGDistribution,
)


@pytest.mark.parametrize(
    "description, expected_age_values, expected_weight_values",
    [
        (
            "Neonatal preterm and LBWSG (estimation years) - [0, 24) wks, [0, 500) g",
            (0.0, 24.0),
            (0.0, 500.0),
        ),
        (
            "Neonatal preterm and LBWSG (estimation years) - [40, 42+] wks, [2000, 2500) g",
            (40.0, 42.0),
            (2000.0, 2500.0),
        ),
        (
            "Neonatal preterm and LBWSG (estimation years) - [34, 36) wks, [4000, 9999] g",
            (34.0, 36.0),
            (4000.0, 9999.0),
        ),
    ],
)
def test_parsing_lbwsg_descriptions(description, expected_weight_values, expected_age_values):
    weight_interval = LBWSGDistribution._parse_description("birth_weight", description)
    age_interval = LBWSGDistribution._parse_description("gestational_age", description)
    assert weight_interval.left == expected_weight_values[0]
    assert weight_interval.right == expected_weight_values[1]
    assert age_interval.left == expected_age_values[0]
    assert age_interval.right == expected_age_values[1]

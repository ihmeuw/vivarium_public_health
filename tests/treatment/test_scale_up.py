from vivarium_public_health.treatment import LinearScaleUp


def test_linear_scale_up_instantiation():
    scale_up = LinearScaleUp("treatment.sqlns")

    assert scale_up.treatment == "treatment.sqlns"

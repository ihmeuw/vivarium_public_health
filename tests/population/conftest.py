import pytest


@pytest.fixture(params=["Male", "Female", "Both"])
def include_sex(request):
    return request.param

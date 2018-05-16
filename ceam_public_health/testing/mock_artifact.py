from ..dataset_manager import Artifact, ArtifactException
from .utils import make_uniform_pop_data

MOCKERS = {
        'cause': {
            'prevalence': lambda *args: 0,
            'cause_specific_mortality': lambda *args: 0,
            'excess_mortality': lambda *args: 0,
            'remission': lambda *args: 0,
            'incidence': lambda *args: 0,
        },
        'sequela': {
            'prevalence': lambda *args: 0,
            'cause_specific_mortality': lambda *args: 0,
            'excess_mortality': lambda *args: 0,
            'remission': lambda *args: 0,
            'incidence': lambda *args: 0,
        },
        'population': {
            'structure': lambda: make_uniform_pop_data(),
        },
}

class MockArtifact(Artifact):
    def __init__(self):
        super(MockArtifact, self).__init__()
        self._is_open = False
        self._overrides = {}

    def load(self, entity_path, keep_age_group_edges=False, **column_filters):
        if entity_path in self._overrides:
            return self._overrides[entity_path]

        entity_type, *tail = entity_path.split('.')
        assert entity_type in MOCKERS
        assert tail[-1] in MOCKERS[entity_type]

        return MOCKERS[entity_type][tail[-1]]()

    def set(self, entity_path, value):
        self._overrides[entity_path] = value

    def open(self, *args, **kwargs):
        if not self._is_open:
            self._is_open = True
        else:
            raise ArtifactException("Opening already open artifact")

    def close(self):
        if self._is_open:
            self._is_open = False
        else:
            raise ArtifactException("Closing already closed artifact")

    def summary(self):
        return "Mock Artifact"

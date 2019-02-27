
class EntityString(str):
    """Convenience class for representing entities as strings."""

    def __init__(self, entity):
        super().__init__()
        self._type, self._name = self.split_entity()

    @property
    def type(self):
        return self._type

    @property
    def name(self):
        return self._name

    def split_entity(self):
        split = self.split('.')
        if len(split) != 2:
            raise ValueError(f'You must specify the entity as "entity_type.entity". You specified {self}.')
        return split[0], split[1]

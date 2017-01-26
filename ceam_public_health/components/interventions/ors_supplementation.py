from ceam.framework.event import listens_for
from ceam.framework.values import modifies_value

from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam.framework.values import modifies_value
from ceam import config
from ceam.framework.randomness import choice

# There are 2 ways to increase exposure due to a risk factor -- change people's draw number or change the exposure proportion
@modifies_value('')
@uses_columns([''], 'alive')

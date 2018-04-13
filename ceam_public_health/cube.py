from warnings import warn

import pandas as pd

from ceam_inputs import (get_excess_mortality, get_prevalence, get_cause_specific_mortality,
                         get_incidence, get_disability_weight, causes, sequelae, etiologies)


def make_measure_cube_from_gbd(year_start, year_end, locations, draws, measures, config):
    """ Build a DataFrame which contains GBD data for each of the measure/cause
    pairs listed in `measures`.
    """
    # Map from each measure name to the function which gets that measure's data


    # TODO: I'm always complaining about how other people don't include
    # metadata with their data. This should afford the attachment of
    # metadata like meid or guidance on how to interpret distribution
    # parameters.

    # TODO: This fiddling of the config is awkward but it's necessary
    # unless we re-architect the existing ceam_input functions.
    old_year_start = config.time.start.year
    old_year_end = config.time.end.year
    old_location = config.input_data.location_id
    old_draw = config.input_data.input_draw_number
    config.time.start.year = year_start
    config.time.end.year = year_end

    cube = pd.DataFrame(columns=['year', 'age', 'sex', 'measure', 'cause', 'draw', 'value'])
    for location in locations:
        config.input_data.location_id = location
        for draw in draws:
            config.input_data.input_draw_number = draw
            for cause, measure in measures:
                data = _get_data(cause, measure, config)
                if data is None:
                    warn("Trying to load input for {}.{} but no mapping was present".format(cause, measure))
                    continue

                # TODO: This assumes a single value for each point but that won't
                # be valid for categorical risks data or distribution data.
                # To support those we'll need to allow for multiple value columns.
                value_column = [c for c in data.columns if c not in ['age', 'sex', 'year']]
                assert len(value_column) == 1
                value_column = value_column[0]
                data = data.rename(columns={value_column: 'value'})

                data['draw'] = draw
                data['measure'] = measure
                data['cause'] = cause
                data['location'] = location

                cube = cube.append(data)

    config.time.start.year = old_year_start
    config.time.end.year = old_year_end
    config.input_data.location_id = old_location
    config.input_data.input_draw_number = old_draw

    return cube.set_index(['year', 'age', 'sex', 'measure', 'cause', 'draw', 'location'])


def _get_data(cause_name, measure_name, config):
    function_map = {
        'excess_mortality': get_excess_mortality,
        'prevalence': get_prevalence,
        'csmr': get_cause_specific_mortality,
        'disability_weight': get_disability_weight,
        'incidence': get_incidence,
    }
    cause = _get_cause_from_name(cause_name)
    return function_map[measure_name](cause, config)



def _get_cause_from_name(cause_name):
    if cause_name in causes:
        cause = causes[cause_name]
    elif cause_name in sequelae:
        cause = sequelae[cause_name]
    elif cause_name in etiologies:
        cause = etiologies[cause_name]
    else:
        prefix, name = cause_name.split('_', maxsplit=1)
        if prefix in ['mild', 'moderate', 'severe', 'asymptomatic']:
            parent_cause = _get_cause_from_name(name)
            cause = parent_cause.severity_splits[prefix]
        else:
            raise ValueError('Invalid cause name {}'.format(cause_name))
    return cause

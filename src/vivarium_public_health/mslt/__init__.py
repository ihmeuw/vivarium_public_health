import pandas as pd

def add_year_column(builder, data):
    """
    Ensure that the table has a 'year_start' column.

    If the 'year_start' column does not exist, this will concatenate a copy of
    the original table for each year of the simulation, defining the
    'year_start' and 'year_end' columns appropriately.
    """
    if 'year_start' in data.columns:
        return data

    tables = []
    for year in range(builder.configuration.time.start.year,
                      builder.configuration.time.end.year):
        data['year_start'] = year
        data['year_end'] = year + 1
        tables.append(data.copy())
    return pd.concat(tables).reset_index(drop=True)

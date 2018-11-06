import pandas as pd

def add_year_column(builder, data):
    """
    Ensure that the table has a 'year' column.

    If the 'year' column does not exist, this will concatenate a copy of the
    original table for each year of the simulation.
    """
    if 'year' in data.columns:
        return data

    tables = []
    for year in range(builder.configuration.time.start.year,
                      builder.configuration.time.end.year):
        data['year'] = year
        tables.append(data.copy())
    return pd.concat(tables).reset_index(drop=True)

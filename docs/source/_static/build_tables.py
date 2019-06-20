#!/usr/bin/env python3

import glob
import numpy as np
import pandas as pd
import sys


def column_precision():
    return {
        'population': 1,
        'bau_population': 1,
        'prev_population': 1,
        'bau_prev_population': 1,
        'acmr': 4,
        'bau_acmr': 4,
        'pr_death': 4,
        'bau_pr_death': 4,
        'deaths': 1,
        'bau_deaths': 1,
        'yld_rate': 4,
        'bau_yld_rate': 4,
        'person_years': 1,
        'bau_person_years': 1,
        'HALY': 1,
        'bau_HALY': 1,
        'LE': 1,
        'bau_LE': 1,
        'HALE': 1,
        'bau_HALE': 1,
        'bau_incidence': 6,
        'int_incidence': 6,
        'bau_prevalence': 6,
        'int_prevalence': 6,
        'int_deaths': 1,
        'diff_incidence': 6,
        'diff_prevalence': 6,
    }


def column_names():
    return {
        'disease': 'Disease',
        'year_of_birth': 'Year of birth',
        'year': 'Year',
        'age': 'Age',
        'sex': 'Sex',
        'population': 'Survivors',
        'bau_population': 'BAU Survivors',
        'prev_population': 'Population',
        'bau_prev_population': 'BAU Population',
        'acmr': 'ACMR',
        'bau_acmr': 'BAU ACMR',
        'pr_death': 'Probability of death',
        'bau_pr_death': 'BAU Probability of death',
        'deaths': 'Deaths',
        'bau_deaths': 'BAU Deaths',
        'yld_rate': 'YLD rate',
        'bau_yld_rate': 'BAU YLD rate',
        'person_years': 'Person years',
        'bau_person_years': 'BAU Person years',
        'HALY': 'HALYs',
        'bau_HALY': 'BAU HALYs',
        'LE': 'LE',
        'bau_LE': 'BAU LE',
        'HALE': 'HALE',
        'bau_HALE': 'BAU HALE',
        'bau_incidence': 'BAU Incidence',
        'int_incidence': 'Incidence',
        'bau_prevalence': 'BAU Prevalence',
        'int_prevalence': 'Prevalence',
        'int_deaths': 'Deaths',
        'diff_incidence': 'Change in incidence',
        'diff_prevalence': 'Change in prevalence',
    }


def display_large_numbers_with_commas(df):
    numeric_types = [np.dtype(float)]
    for column in df.columns.values:
        if df[column].dtype in numeric_types:
            if abs(df[column].max()) >= 1000:
                df[column] = df[column].apply(lambda x: '{:,.1f}'.format(x))


def build_table(df, csv_file):
    # Filter rows, format and rename columns.
    df = df.loc[(df['sex'] == 'male') & (df['year_of_birth'] == 1959)]
    df = df.round(column_precision())
    df = df.rename(columns=column_names())
    display_large_numbers_with_commas(df)

    # Display the first few and last few rows for this cohort.
    df_head = df.iloc[:3]
    df_tail = df.iloc[-3:]
    with open(csv_file, 'w') as f:
        f.write(','.join(df.columns.values))
        f.write('\n')
        f.write('...\n')
        for line in df_head.to_csv(index=False, header=False).split('\n'):
            if line:
                f.write(line)
                f.write('\n')
        f.write('...\n')
        for line in df_tail.to_csv(index=False, header=False).split('\n'):
            if line:
                f.write(line)
                f.write('\n')
        f.write('...\n')


def main(args=None):
    for input_file in glob.glob('mslt_*.csv'):
        df = pd.read_csv(input_file)
        output_file = 'table_{}'.format(input_file)
        build_table(df, output_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())

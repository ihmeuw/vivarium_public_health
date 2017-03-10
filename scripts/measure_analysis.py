import os.path
import argparse
import warnings
import math

from joblib import Memory

import pandas as pd
import numpy as np

import seaborn
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from transmogrifier.draw_ops import get_draws

from ceam_inputs import get_age_bins
from ceam_inputs.gbd_ms_functions import get_model_versions
from ceam_inputs.gbd_mapping import causes, meid
from ceam_inputs.util import round_to_gbd_year

memory = Memory(cachedir='~/.ceam_analysis', verbose=0)

def get_age_group_ids(data):
    groups = get_age_bins()[['age_group_years_start', 'age_group_years_end', 'age_group_id']].set_index(['age_group_years_start', 'age_group_years_end']).age_group_id
    ids = []
    for low, high in data[['age_low', 'age_high']].values.tolist():
        ids.append(int(groups.ix[low, high]))
    return ids

def load_validation_data(cause_list, measure):
    for parent, cause in cause_list:
        if cause not in causes:
            warnings.warn('Cause "{}" not in gbd mapping. Skipping'.format(cause))
            continue

def plot_convergence(ax, groups, title):
    ax.set_title(title)
    ax.plot(ax.get_xlim(), ax.get_ylim(),'w-', zorder=1, lw=0.5)
    for gbd, simulation, color, shape in groups:
        ax.scatter(gbd, simulation, color=color, zorder=2, marker=shape)

@memory.cache
def cached_draw(*args, **kwargs):
    return get_draws(*args, **kwargs)

def graph_convergence(data, measure, output_directory):
    data['age_group_id'] = get_age_group_ids(data)
    age_group_ids = data.age_group_id.unique()
    data['sex_id'] = np.where(data.sex == 'Male', 1, 2)
    sex_ids = data.sex_id.unique()
    data['location_id'] = data.location
    location_ids = data.location.unique()
    original_years = data.year.unique()
    years = [round_to_gbd_year(year) for year in original_years]
    if original_years != years:
        warnings.warn("Some years don't align with GBD years, rounding to nearest GBD year instead. Raw years: {}".format(original_years))
    data['year_id'] = data.year.apply(round_to_gbd_year)
    draw_ids = data.draw.unique()

    data['cause'] = data.cause.apply(lambda x: x[1])
    meids = {}
    for cause_name in data.cause.unique():
        if cause_name in causes:
            cause = causes[cause_name]
            if 'prevalence' in cause and isinstance(cause.prevalence, meid):
                meids[cause_name] = cause.prevalence

    data = data[data.cause.apply(lambda c: c in meids)]
    data['modelable_entity_id'] = data.cause.apply(lambda c: meids[c])

    groups = get_age_bins()

    gbd_data = pd.DataFrame()
    meid_version_map = get_model_versions()
    for me_id in set(meids.values()):
        model_version = meid_version_map[me_id]
        draws = cached_draw('modelable_entity_id', me_id, location_ids=location_ids, source='dismod', sex_ids=sex_ids, age_group_ids=age_group_ids, year_ids=years, model_version_id=model_version).query('measure_id == 5')[['location_id', 'year_id', 'age_group_id', 'sex_id', 'modelable_entity_id'] + ['draw_{}'.format(draw) for draw in draw_ids]]
        draws = draws.set_index(['location_id', 'year_id', 'age_group_id', 'sex_id', 'modelable_entity_id'])
        gbd_data = gbd_data.append(draws, verify_integrity=True)

    gbd_data = gbd_data.reset_index()
    data = data.pivot_table(columns='draw', values='value', index=[c for c in data.columns if c not in ['draw', 'value']])
    data.columns = ['draw_{}'.format(c) for c in data.columns]
    data = data.reset_index()

    comparison = pd.merge(gbd_data, data, on=['location_id', 'year_id', 'age_group_id', 'sex_id', 'modelable_entity_id'], suffixes=('_gbd', '_simulation'))

    cause_num = {n:i for i,n in enumerate(sorted(np.unique(comparison.cause)))}
    mc = np.max(list(cause_num.values()))
    cause_num = {c:n/mc for c,n in cause_num.items()}
    nums = [cause_num[cause] for cause in comparison.cause]
    comparison['color'] = [tuple(r) for r in cm.rainbow(nums).tolist()]

    patches = []
    for name, color in sorted(comparison[['cause', 'color']].drop_duplicates().values.tolist()):
        if name not in patches:
            patches.append(mpatches.Patch(color=color, label=str(name)))

    row_count = math.sqrt(len(age_group_ids))
    if row_count != int(row_count):
        row_count = int(row_count)
        column_count = row_count + 1
    else:
        row_count = int(row_count)
        column_count = row_count
    fig, rows = plt.subplots(row_count, column_count, sharex='col', sharey='row')
    labels = [
        fig.text(0.5, 0.0, 'Simulation', ha='center', va='center'),
        fig.text(0.0, 0.5, 'GBD', ha='center', va='center', rotation='vertical'),
        fig.suptitle(measure),
    ]
    i = 0
    shapes = {'Male': 's', 'Female': '^'}
    for sex, shape in shapes.items():
        patches.append(Line2D([],[], marker=shape, label=sex, markeredgecolor='k', markeredgewidth=1, fillstyle='none', linestyle='None'))
    for row in rows:
        for ax in row:
            if i >= len(age_group_ids):
                break
            ax.set_xlim([0, max(comparison.draw_0_gbd.max(), comparison.draw_0_simulation.max())])
            ax.set_ylim([0, max(comparison.draw_0_gbd.max(), comparison.draw_0_simulation.max())])
            age_group_id = age_group_ids[i]
            filtered = comparison.query('age_group_id == @age_group_id')
            title = '{} - {}'.format(filtered.age_low.iloc[0], filtered.age_high.iloc[0])
            groups = []
            for sex, shape in shapes.items():
                groups.append(list(zip(*filtered.query('sex==@sex')[['draw_0_gbd', 'draw_0_simulation', 'color']].values.tolist())) + [shape])
            plot_convergence(ax, groups, title)
            i += 1
    lgd = rows[0][-1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    fig.set_size_inches(18.5, 10.5)
    fig.savefig(os.path.join(output_directory, '{}.png'.format(measure)), dpi=100, bbox_extra_artists=[lgd]+labels, bbox_inches='tight')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('measure_data_path', type=str)
    parser.add_argument('output_directory', type=str)
    args = parser.parse_args()

    data = pd.read_hdf(args.measure_data_path)

    for measure in data.measure.unique():
        graph_convergence(data.query('measure == @measure'), measure, args.output_directory)


if __name__ == '__main__':
    main()

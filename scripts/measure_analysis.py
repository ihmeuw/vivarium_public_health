import matplotlib as mpl
mpl.use('Agg')

import os.path
import argparse
import math

import pandas as pd
import numpy as np

import seaborn
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from ceam_inputs.cube import make_measure_cube_from_gbd

def graph_measure(data, measure, output_directory):
    """ Save the convergence graph for a particular measure
    """
    data = data.reset_index().query('measure == @measure')
    age_groups = sorted(data.age.unique())

    # Setup the legend with color swatches and marker shape examples
    patches = []
    for name, color in sorted(data[['cause', 'color']].drop_duplicates().values.tolist()):
        if name not in patches:
            patches.append(mpatches.Patch(color=color, label=str(name)))
    shapes = {'Male': 's', 'Female': '^'}
    for sex, shape in sorted(shapes.items(), key=lambda x:x[0]):
        patches.append(Line2D([],[], marker=shape, label=sex, markeredgecolor='k', markeredgewidth=1, fillstyle='none', linestyle='None'))

    # Determine the shape of our sub-figure matrix
    row_count = max(2, math.sqrt(len(age_groups)))
    if row_count != int(row_count):
        row_count = int(row_count)
        column_count = row_count + 1
    else:
        row_count = int(row_count)
        column_count = row_count

    # Create the sub-figure matrix and labels
    fig, rows = plt.subplots(row_count, column_count)
    labels = [
        fig.text(0.5, 0.0, 'GBD', ha='center', va='center'),
        fig.text(0.0, 0.5, 'Simulation', ha='center', va='center', rotation='vertical'),
        fig.suptitle(measure),
    ]

    # Walk through the age groups graphing one into each sub-figure until we run out
    # and then hide the remaining figures
    for row in rows:
        for ax in row:
            if not age_groups:
                ax.axis('off')
                continue

            age = age_groups.pop(0)
            filtered = data.query('age == @age')

            # TODO: This may be better represented as mark size
            mean_sample_size = filtered.sample_size.mean()

            ax.set_xlim([0, max(filtered.gbd.max(), filtered.simulation.max())])
            ax.set_ylim([0, max(filtered.gbd.max(), filtered.simulation.max())])

            title = '{:.1f} ({})'.format(age, int(mean_sample_size))
            ax.set_title(title)

            # Draw the equivalence line
            ax.plot(ax.get_xlim(), ax.get_ylim(),'w-', zorder=1, lw=1)

            for sex, shape in shapes.items():
                # Draw the actual points with different shapes for each sex
                gbd, simulation, color = zip(*filtered.query('sex==@sex')[['gbd', 'simulation', 'color']].values.tolist())
                ax.scatter(gbd, simulation, color=color, zorder=2, marker=shape)

            # The graphs tend to be pretty tight so rotate the x axis labels to make better use of space
            for tick in ax.get_xticklabels():
                tick.set_rotation(30)

    # Attach the legend in the upper right corner of the figure
    lgd = rows[0][-1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    fig.set_size_inches(18.5, 10.5)
    plt.tight_layout()
    # Make room for the main title
    plt.subplots_adjust(top=0.94)
    fig.savefig(os.path.join(output_directory, '{}.png'.format(measure)), dpi=100, bbox_extra_artists=[lgd]+labels, bbox_inches='tight')

def prepare_comparison(data):
    """Combines the simulation output with the corresponding GBD estimate for each
    sample so they can be graphed together.
    """
    measures = data[['cause','measure']].drop_duplicates().values.tolist()
    year_min = data.year.min()
    year_max = data.year.max()
    measure_cube = make_measure_cube_from_gbd(int(year_min), int(year_max), data.location.unique(), data.draw.unique(), measures)

    # Resolve age ranges down to age group midpoints
    # NOTE: If this midpoint doesn't exactly align with the one from GBD then the comparison
    # won't work. It may be worth figuring out a less fragile approach which (ideally) doesn't
    # involve leaning on age_group_ids
    data['age'] = (data.age_low + data.age_high) / 2
    del data['age_low']
    del data['age_high']

    # NOTE: This averages the draws without capturing uncertainty. May want to improve at some point.
    measure_cube = measure_cube.reset_index().groupby(['year', 'age', 'sex', 'measure', 'cause', 'location']).mean()
    del measure_cube['draw']
    data = data.groupby(['year', 'age', 'sex', 'measure', 'cause', 'location']).mean().reset_index()
    del data['draw']

    # Calculate RGB triples for each cause for use in coloring marks on the graphs
    cmap = plt.get_cmap('jet')
    # This sort and shuffle looks a bit odd but what it accomplishes is to deterministically
    # spread the causes out across the color space which makes it easier to assign visually
    # distinct colors to them that don't change from run to run
    causes = sorted(data.cause.unique())
    np.random.RandomState(1001).shuffle(causes)
    color_map = {cause:tuple(color) for cause, color in zip(causes, cmap(np.linspace(0, 1, len(causes))))}
    data['color'] = data.cause.apply(color_map.get)

    data = data.set_index(['year', 'age', 'sex', 'measure', 'cause', 'location'])

    # Give our value columns descriptive names so we know which is which
    data = data.rename(columns={'value': 'simulation'})
    measure_cube = measure_cube.rename(columns={'value': 'gbd'})
    return data.merge(measure_cube, left_index=True, right_index=True)

def graph_comparison(data, output_directory):
    data = prepare_comparison(data)

    # Save a graph for each measure
    for measure in data.reset_index().measure.unique():
        graph_measure(data, measure, output_directory)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('measure_data_path', type=str)
    parser.add_argument('output_directory', type=str)
    parser.add_argument('--year', '-y', default='last', type=str)
    parser.add_argument('--draw', '-d', default='all', type=str)
    args = parser.parse_args()

    data = pd.read_hdf(args.measure_data_path, format='t')

    # TODO: right now this can only do one year per run.
    # If we want to do multiple years, that's certainly possible
    # it would just be a matter of deciding how to represent time.
    # Could be separate graphs or some sort of timeseries thingy
    if args.year == 'last':
        year = data.year.max()
    else:
        year = int(args.year)

    data = data.query('year == @year')

    if args.draw != 'all':
        draw = int(args.draw)
        data = data.query('draw == @draw')

    graph_comparison(data, args.output_directory)

if __name__ == '__main__':
    main()

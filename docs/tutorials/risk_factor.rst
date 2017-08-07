Implementing a Simple Risk Factor
=================================

First things first
------------------

Build out a development environment by following these instructions: :doc:`setup`

If you were writing a risk factor that was intended for use in the main model you would work out of a branch in ``ceam_public_health`` but for this tutorial put the code in a separate place. If you haven't already, create a tutorial package by following the instructions `here <http://rancher-host-d02.ihme.washington.edu:8080/tutorials/0_Setup.html#setup-a-working-directory>`_.

Intro to the interesting bit
----------------------------

We're going to write a model of body mass index and its effect on the the causes we model. To do that we need three things: a plan for how to model it, the data to drive that model and (last and probably least) an actual implementation of the model in code.

Part one: The plan
------------------

We'll pattern our BMI model CEAM's systolic blood pressure model since they're both continuous risk factors with known population distributions. The method will be to randomly assign each simulant a BMI percentile at birth and to use that percentile to select a BMI from the population distribution for the simulants at each time step. Based on that BMI we can use the GBD's relative risk numbers for BMI to calculate it's effect on incidence of the causes.

Part two: The data
------------------

CEAM already has good tools for getting PAFs and RRs so all we'll need to worry about here is the BMI distribution itself. Like some of the other similar data that CEAM uses, BMI distributions aren't really a first class citizen in GBD so they require some hunting down. I went to the metabolism modelers and got the location and structure of the distribution data from them so you won't need to do that part, but normally it's an important part of the process. In this case they use beta distributions stored in a large number of files, one for each parameter/location/year/sex combination. Like so many things, those files are all on the J drive.

We don't want to depend on the modelers to not delete or change these kind of files so we want to copy them to a place which we control. That place is a subdirectory under ``/home/j/Project/Cost_Effectiveness/CEAM/Auxiliary_Data/GBD_2015``. I've already created the directories and populated them with the data for Kenya but, again, that's something you would need to do if you were starting from scratch.

One last bit before we move on to actual implementation. We don't really want to have hard coded paths for files like this built into the code. Until we come up with a better solution, what we are doing is keeping a mapping from names to file paths in ``ceam_inputs/ceam_inputs/aukiliary_files.py``. If you look in that file, you'll see a dictionary of named datasets with information about each dataset. There's an entry for the BMI distributions under 'Body Mass Index Distributions' that describes the directory where the data was originally stored, which researcher owns the originals and a pattern that describes the path to each of the files that make up the data in our copy. You use the mapping by calling ``open_auxiliary_file`` with the name of the dataset ('Body Mass Index Distributions') and the values necessary to fill in the file path pattern (in this case parameter, location_id, year_id and sex_id). A simpler dataset, like the disability weights table, doesn't require any additional parameters, just the dataset name.

Part three: The details
-----------------------

Ok, now we know what our BMI model will look like and how to find the data we need to run it.

.. code-block:: python

    import pandas as pd
    import numpy as np

    from scipy.stats import beta

    from ceam.interpolation import Interpolation
    from ceam_inputs.auxiliary_files import open_auxiliary_file

Having imported the tools we need, the first order of business is to load in the distribution tables and turn them into objects in python that the simulation can use. I'll walk you through a function for doing that which is typical of how CEAM handles input data. It goes through three phases, loading CSVs from disk, filtering and transforming their contents and then turning them into a group of interpolation functions which can produce data for any simulante.

.. code-block:: python

    def get_bmi_distributions(location_id, year_start, year_end, draw):
        a = pd.DataFrame()
        b = pd.DataFrame()
        loc = pd.DataFrame()
        scale = pd.DataFrame()

We need to know what location, time range and draw number we should be loading data for so we take those as arguments. Then we create temporary dataframes for each of the distribution parameters which will hold the raw data as we load in the CSVs.

.. code-block:: python

        for sex_id in [1,2]:
            for year_id in np.arange(year_start, year_end + 1, 5):

The input files are broken up by location, five year chunk and sex. We only do one location at a time so there's no need to loop over that but we'll need to load files for both sexes and each five year chunk in our time range so there are loops for both of those.

.. code-block:: python

                    with open_auxiliary_file('Body Mass Index Distributions',
                                             parameter='bshape1',
                                             location_id=location_id,
                                             year_id=year_id,
                                             sex_id=sex_id) as f:
                        a = a.append(pd.read_csv(f))
                    with open_auxiliary_file('Body Mass Index Distributions',
                                             parameter='bshape2',
                                             location_id=location_id,
                                             year_id=year_id,
                                             sex_id=sex_id) as f:
                        b = b.append(pd.read_csv(f))
                    with open_auxiliary_file('Body Mass Index Distributions',
                                             parameter='mm',
                                             location_id=location_id,
                                             year_id=year_id,
                                             sex_id=sex_id) as f:
                        loc = loc.append(pd.read_csv(f))
                    with open_auxiliary_file('Body Mass Index Distributions',
                                             parameter='scale',
                                             location_id=location_id,
                                             year_id=year_id,
                                             sex_id=sex_id) as f:
                        scale = scale.append(pd.read_csv(f))


We use ``open_auxiliary_file`` to open the file for each distribution parameter and read the data int pandas DataFrames which we append into our accumulators. After the loops each accumulator will contain all the rows for all the age-sex-year permutations of that parameter.

.. code-block:: python

            a = a.set_index(['age_group_id', 'sex_id', 'year_id'])
            b = b.set_index(['age_group_id', 'sex_id', 'year_id'])
            loc = loc.set_index(['age_group_id', 'sex_id', 'year_id'])
            scale = scale.set_index(['age_group_id', 'sex_id', 'year_id'])

Once the loops are done we reindex by multiindexes which will make some of the combining we have to do next easier.

.. code-block:: python

            distributions = pd.DataFrame()
            distributions['a'] = a['draw_{}'.format(draw)]
            distributions['b'] = b['draw_{}'.format(draw)]
            distributions['loc'] = loc['draw_{}'.format(draw)]
            distributions['scale'] = scale['draw_{}'.format(draw)]

Combine the separate DataFrames into a single one with a column for each distribution parameter and a row for each age-sex-year combination. Notice that we filter the data down to a single draw. The raw files contain all 1000 draws.

.. code-block:: python

            distributions = distributions.reset_index()
            distributions = get_age_from_age_group_id(distributions)
            distributions['year'] = distributions.year_id
            distributions.loc[distributions.sex_id == 1, 'sex'] = 'Male'
            distributions.loc[distributions.sex_id == 2, 'sex'] = 'Female'
            distributions = distributions[['age', 'year', 'sex', 'a', 'b', 'scale', 'loc']]

Now that we have the columns all in one place and aligned we reset the index which makes the columns that were in the index easier to work with. Then we do a series of standard transformations which turn raw GBD data into a form that makes sense outside of the context of GBD. ``age_group_ids`` are converted into real ages. ``sex_id`` is converted into meaningful 'Male' and 'Female' strings. ``year_id`` which is already just the year is renamed to be ``year``. Then we strip of all the other columns so we only have the ones we care about in our final result.

.. code-block:: python

            return Interpolation(
                    distributions[['age', 'year', 'sex', 'a', 'b', 'scale', 'loc']],
                    categorical_parameters=('sex',),
                    continuous_parameters=('age', 'year'),
                    func=_bmi_ppf
                    )

Finally we convert the cleaned up data into a collection of percent point functions of the beta distributions defined by the interpolated parameters. What the Interpolation object represents is a group of spline interpolations, one for each parameter ('a', 'b', 'scale' and 'loc') and each sex. They interpolate between the distribution parameters over age and year. The output of the interpolations is fed through ``_bmi_ppd`` which calculates the percent point function.

.. code-block:: python

        def _bmi_ppf(parameters):
           return beta(a=parameters['a'], b=parameters['b'], scale=parameters['scale'], loc=parameters['loc']).ppf

All together the output of ``get_bmi_distributions`` is a function which takes a DataFrame with 'age', 'sex' and 'year' columns and returns a Series of PPF functions.

In the production code I put ``get_bmi_distributions`` into the ``ceam_inputs`` repository. For this tutorial you can stick it in with the rest of the code but when you write production models it's important to think about organization. Code that interacts directly with GBD (and related datasets) by reading and preprocessing data should be in ``ceam_inputs``. This is code that is unlikely to be useful to anyone outside of the IHME. Code that uses data from ``ceam_inputs`` but could potentially use data from other sources should be in ``ceam_public_health``. These are models are specific to the microsimulation of health related things and could be useful other modelers within IHME or even researchers at other organizations. ``ceam_public_health`` is where the BMI model goes in production. The ``ceam`` repository contains code that is useful for any microsimulation, even one that isn't health related.

Now onto the model itself. BMI isn't a super complicated model but it still will need to group together quite a bit of data and have a couple of behaviors so it makes sense to make it an object:

.. code-block:: python

    class BodyMassIndex:
        """Model BMI"""

During the setup phase of the simulation, we'll need to load the BMI distributions as well as some other data.

.. code-block:: python

            def setup(self, builder):
                location_id = config.getint('simulation_parameters', 'location_id')
                year_start = config.getint('simulation_parameters', 'year_start')
                year_end = config.getint('simulation_parameters', 'year_end')
                draw = config.getint('run_configuration', 'draw_number')
                self.bmi_distributions = builder.lookup(get_bmi_distributions(location_id, year_start, year_end, draw))

We look up the location, time range and current draw from the simulation's configuration system and use those to evoke ``get_bmi_distributions``. The interesting part here is ``builder.lookup`` which is a utility that the simulation provides for turning DataFrames indexed by simulant attributes or interpolation functions that take simulant attributes as arguments into a function that converts lists of simulant_ids (which is the form in which models normally interact with the population) into interpolated output. Or, in the case of ``get_bmi_distributions`` which is more complicated than usual, interpolated output processed through an additional function like the one that produces the PPFs.

.. code-block:: python

                self.ihd_rr = builder.lookup(get_relative_risks(risk_id=108, cause_id=493))
                self.hemorrhagic_stroke_rr = builder.lookup(get_relative_risks(risk_id=108, cause_id=496))
                self.ischemic_stroke_rr = builder.lookup(get_relative_risks(risk_id=108, cause_id=495))

                self.ihd_paf = builder.lookup(get_pafs(risk_id=108, cause_id=493))
                self.hemorrhagic_stroke_paf = builder.lookup(get_pafs(risk_id=108, cause_id=496))
                self.ischemic_stroke_paf = builder.lookup(get_pafs(risk_id=108, cause_id=495))

Here we do something very similar and build interpolated lookup functions for RR and PAF data using standard ``ceam_inputs`` lookup functions.

.. code-block:: python

                builder.modifies_value(self.ihd_paf, 'heart_attack.paf')
                builder.modifies_value(self.hemorrhagic_stroke_paf, 'hemorrhagic_stroke.paf')
                builder.modifies_value(self.ischemic_stroke_paf, 'ischemic_stroke.paf')

We then take PAF lookups and attach them to dynamic values so that they can be used for risk deletion.

.. code-block:: python

                builder.modifies_value(partial(self.incidence_rates, rr_lookup=self.ihd_rr), 'heart_attack.incidence_rate')
                builder.modifies_value(partial(self.incidence_rates, rr_lookup=self.hemorrhagic_stroke_rr), 'hemorrhagic_stroke.incidence_rate')
                builder.modifies_value(partial(self.incidence_rates, rr_lookup=self.ischemic_stroke_rr), 'ischemic_stroke.incidence_rate')



These are a bit more complicated. At a high level what we are doing is setting up the plumbing that connects this risk to the causes it effects so that we can adjust incidence rates. ``builder.modifies_values`` associates it's first argument, a function which modifies RRs (or other similar values), with it's second argument, a named value.

We construct the mutation functions using a trick called partial application which lets us avoid duplicating some code. ``partial`` takes a function with some subset of the arguments which that function accepts and returns a new function which is the equivalent with the value of those arguments fixed so you only need to pass it the remaining arguments when you call it. In this case we have a generic: ``self.incidence_rates``, which handles incidence rates for any cause. We then use partial application to to configure them with the RR lookup function for each cause.

.. code-block:: python

            @listens_for('initialize_simulants')
            @uses_columns(['bmi_percentile', 'bmi'])
            def initialize(self, event):
                event.population_view.update(pd.DataFrame({
                    'bmi_percentile': self.randomness.get_draw(event.index)*0.98+0.01,
                    'bmi': np.full(len(event.index), 20)
                }))

Here we create the columns to store BMI related values in the simulation's state table. We give each simulant a susceptibility which we'll use later to determine their actual BMI. We also create a column to store current BMI which starts filled with dummy data but it the first timestep that will be replaced with real numbers.

.. code-block:: python

            @uses_columns(['bmi'])
            def incidence_rates(self, index, rates, population_view, rr_lookup):
                population = population_view.get(index)
                rr = rr_lookup(index)

                rates *= np.maximum(rr.values**((population.bmi - 21) / 5).values, 1)
                return rates

This is the generic function that applies an RR to any incidence rate. Remember from above that we configured ``rr_lookup`` to be the lookup function for the RR of BMI on the relevant cause. We calculate an modified rate based on the base rate we get as an argument, the RR and the current BMI for each simulant. The formula here shifts the simulant's BMI by 21, the TMR, and scales it by 5, which is the number of units of BMI needed to change the incidence rate by one unit. We then trim the result so that it's at least 1 which prevents values under the minimum risk level from suppressing the rate. You can find these parameters here: /snfs1/Project/Cost_Effectiveness/dev/data/gbd/risk_data/risk_variables.xlsx

.. code-block:: python

            @listens_for('time_step__prepare', priority=8)
            @uses_columns(['bmi', 'bmi_percentile'], 'alive')
            def update_body_mass_index(self, event):
                new_bmi = self.bmi_distributions(event.index)(event.population.bmi_percentile)
                event.population_view.update(pd.Series(new_bmi, name='bmi', index=event.index))

This is arguably the heart of the model, where we actually track each simulant's current BMI. It's also probably the simplest part. We just feed the simulants' susceptibility percentile into the PPF functions we created earlier and write the result into the simulation's state table.

If you open up ``ceam_public_health/configurations/opportunistic_sbp_screening.json`` and replace the entry for ``ceam_public_health.components.body_mass_index.BodyMassIndex`` with the path to your tutorial implementation and run the simulation you should see it work.

========
Artifact
========

The :class:`~vivarium_public_health.dataset_manager.artifact.Artifact` in vivarium_public_health provides an interface
to interact with the data artifacts that can be used as the underlying storage for input data into a simulation. We'll
go through how to view, delete, and write data to an artifact using the tools provided by
:class:`~vivarium_public_health.dataset_manager.artifact.Artifact`. You'll access data in the artifact through keys,
mirroring the underlying hdf storage of artifacts.

.. contents::
   :depth: 1
   :local:
   :backlinks: none

Creating an artifact
---------------------
To view an existing hdf file via the :class:`~vivarium_public_health.dataset_manager.artifact.Artifact` tools, we'll
create a new artifact. We can print the resulting artifact to view the tree structure of the keys
in our artifact. We'll use our test artifact to illustrate:

.. code-block:: python

    from vivarium_public_health.dataset_manager import Artifact

    art = Artifact('test_artifact.hdf')
    print(art)
::

    Artifact containing the following keys:
    metadata
            keyspace
            locations
            versions
    population
            age_bins
            structure
            theoretical_minimum_risk_life_expectancy

Now we have an :class:`~vivarium_public_health.dataset_manager.artifact.Artifact` object, which we can use to interact
with the data stored in the hdf file with which we created it.

Optionally, we can specify filter terms on the artifact when we create it, which will be applied to filter all data
that we load out of the artifact. For example, if we were only interested in data for years after 2005, we could add a
filter term to our artifact that would filter all data with a ``year_start`` column. Any data in our artifact that does
not contain this column will not be filtered. If we also wanted to filter to ages under 5, we could add another filter
term to the list as below:

.. code-block:: python

    from vivarium_public_health.dataset_manager import Artifact

    art = Artifact('test_artifact.hdf', filter_terms=['year_start > 2005', 'age_group_start <= 5'])
    print(art)

::

    Artifact containing the following keys:
    metadata
            keyspace
            locations
            versions
    population
            age_bins
            structure
            theoretical_minimum_risk_life_expectancy

Note that the keys in the artifact are unchanged. The filter terms only affect data when it is loaded out of the artifact.

.. testcode::
    :hide:

    from vivarium_public_health.dataset_manager import Artifact

    art = Artifact('../../../tests/dataset_manager/artifact.hdf', filter_terms=['year_start > 2005'])
    print(art)

Artifacts store data under keys. Each key is of the form ``<type>.<name>.<measure>``, e.g.,
"cause.all_causes.restrictions" or ``<type>.<measure>``, e.g., "population.structure." To view all keys in an
artifact, use the `~vivarium_public_health.dataset_manager.artifact.Artifact.keys` attribute of the
artifact:

.. code-block:: python

    art.keys

::

    [EntityKey(metadata.keyspace), EntityKey(metadata.locations), EntityKey(metadata.versions), EntityKey(population.age_bins),
     EntityKey(population.structure), EntityKey(population.theoretical_minimum_risk_life_expectancy)]

What we get back is a list of :class:`~vivarium_public_health.dataset_manager.artifact.EntityKey` objects. We can
access the individual components of each key via attributes, like so:

.. code-block:: python

    key = art.keys[4]
    print(key.type)
    print(key.name)
    print(key.measure)
::

    population

    structure

Because we're looking at the 'population.structure' key, we only have a type and measure.

.. testcode::
    :hide:

    from vivarium_public_health.dataset_manager import Artifact

    art = Artifact('../../../tests/dataset_manager/artifact.hdf')
    key = art.keys[0]
    print(key.type)
    print(key.name)
    print(key.measure)


Reading data
-------------
Now that we've seen how to create an :class:`~vivarium_public_health.dataset_manager.artifact.Artifact` object and
view the underlying storage structure, let's cover how to actually retrieve data from that artifact. We'll use the
:func:`~vivarium_public_health.dataset_manager.artifact.Artifact.load` method. We saw the key names in our artifact
in the previous step, and we'll use those names to load data. For example, if we want to load the population structure
data from our Artifact we do:

.. code-block:: python

    art = Artifact('test_artifact.hdf')
    pop = art.load('population.structure')
    print(pop.head()))
::

       location  year_start  population     sex  age_group_start  age_group_end  year_end
    1  Ethiopia        2006    25610.50  Female              0.0       0.019178      2007
    3  Ethiopia        2011    29136.66    Male              0.0       0.019178      2012
    4  Ethiopia        2008    27492.91    Male              0.0       0.019178      2009
    6  Ethiopia        1999    22157.50  Female              0.0       0.019178      2000
    7  Ethiopia        1992    19066.45  Female              0.0       0.019178      1993

Notice that if we construct our artifact with filter terms as we touched on above, we'll filter the data
that gets loaded out of it:

.. code-block:: python

    art = Artifact('test_artifact.hdf', filter_terms=['age > 5'])
    pop = art.load('population.structure')
    print(pop.head()))
::

.. testcode::
    :hide:

    from vivarium_public_health.dataset_manager import Artifact

    art = Artifact('../../../tests/dataset_manager/artifact.hdf')
    art.load(str(art.keys[0]))

Writing data
------------
To write new data to an artifact, use the :func:`~vivarium_public_health.dataset_manager.artifact.Artifact.write` method,
passing the full key (in the string representation we saw above of type.name.measure or type.measure) and the data you wish
to store.

.. code-block:: python

    new_data = ['United States', 'Washington', 'California']

    art.write('locations.names', new_data)

    if 'locations.names' in art:
        print('Successfully Added!')


.. testcode::
    :hide:

    art.remove('locations.names')


Removing data
-------------

Like :func:`~vivarium_public_health.dataset_manager.artifact.Artifact.load` and :func:`~vivarium_public_health.dataset_manager.artifact.Artifact.write`,
:func:`~vivarium_public_health.dataset_manager.artifact.Artifact.remove` is based on keys. Pass the name of the key
you wish to remove, and it will be deleted from the artifact and the underlying hdf file.

.. code-block:: python

    art.remove('locations.names')

    if not 'locations.names' in art:
        print('Successfully Deleted!')
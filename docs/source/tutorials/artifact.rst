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

    art = Artifact('../../../tests/dataset_manager/artifact.hdf')
    print(art)

Now we have an :class:`~vivarium_public_health.dataset_manager.artifact.Artifact` object, which we can use to interact
with the data stored in the hdf file with which we created it.

Optionally, we can specify filter terms on the artifact when we create it, which will be applied to filter all data
that we load out of the artifact. For example, if we were only interested in data for ages over 5, we could add a filter
term to our artifact that would filter all data with an ``age`` column.

.. code-block:: python

    from vivarium_public_health.dataset_manager import Artifact

    art = Artifact('../../../tests/dataset_manager/artifact.hdf', filter_terms=['age > 5'])
    print(art)

.. testcode::
    :hide:

    from vivarium_public_health.dataset_manager import Artifact

    art = Artifact('../../../tests/dataset_manager/artifact.hdf', filter_terms=['age > 5'])
    print(art)

Artifacts store data under keys. Each key is of the form ``<type>.<name>.<measure>``, e.g.,
"cause.all_causes.restrictions" or ``<type>.<measure>``, e.g., "population.structure." To view all keys in an
artifact, use the :attribute:`~vivarium_public_health.dataset_manager.artifact.Artifact.keys` attribute of the
artifact:

.. code-block:: python

    from vivarium_public_health.dataset_manager import Artifact

    art = Artifact('../../../tests/dataset_manager/artifact.hdf')
    art.keys

What we get back is a list of :class:`~vivarium_public_health.dataset_manager.artifact.EntityKey` objects. We can
access the individual components of each key via attributes, like so:

.. code-block:: python

    from vivarium_public_health.dataset_manager import Artifact

    art = Artifact('../../../tests/dataset_manager/artifact.hdf')
    key = art.keys[0]
    print(key.type)
    print(key.name)
    print(key.measure)

.. testcode::
    :hide:

    from vivarium_public_health.dataset_manager import Artifact

    art = Artifact('../../../tests/dataset_manager/artifact.hdf')
    key = art.keys[0]
    print(key.type)
    print(key.name)
    print(key.measure)

And the string representation of the key (helpful for writing/removing data as we'll see in a bit):

.. code-block:: python

    from vivarium_public_health.dataset_manager import Artifact

    art = Artifact('../../../tests/dataset_manager/artifact.hdf')
    key = art.keys[0]
    str(key)

Reading data
-------------
Now that we've seen how to create an :class:`~vivarium_public_health.dataset_manager.artifact.Artifact` object and
view the underlying storage structure, let's cover how to actually retrieve data from that artifact. We'll use the
:func:`~vivarium_public_health.dataset_manager.artifact.Artifact.load` method:

.. code-block:: python

    from vivarium_public_health.dataset_manager import Artifact

    art = Artifact('../../../tests/dataset_manager/artifact.hdf')
    art.load(str(art.keys[0]))

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
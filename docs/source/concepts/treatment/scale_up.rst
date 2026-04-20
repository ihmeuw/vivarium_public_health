.. _scale_up_concept:

================
Linear Scale-Up
================

.. contents::
   :depth: 2
   :local:
   :backlinks: none

The :class:`~vivarium_public_health.treatment.scale_up.LinearScaleUp` component
models a gradual, linear change in intervention coverage over a specified time
period. This is useful when an intervention is not introduced all at once but
is instead rolled out progressively.

How It Works
------------

The component linearly interpolates between a start value and an end value
over a configured date range. Before the start date the start value is used;
after the end date the end value is used.

At each time step the progress through the scale-up period is calculated as:

.. math::

   \text{progress} = \frac{\text{clock} - \text{start\_date}}{\text{end\_date} - \text{start\_date}}

The adjustment applied to the target exposure parameter is then:

.. math::

   \text{adjustment} = \text{progress} \times (\text{end\_value} - \text{start\_value})

Configuration
-------------

Scale-up dates and values are configured under a key derived from the
treatment name:

.. code-block:: yaml

    configuration:
        my_treatment_scale_up:
            date:
                start: "2025-01-01"
                end: "2030-12-31"
            value:
                start: 0.0
                end: 0.9

When the value ``"data"`` is used instead of a numeric value, the endpoint
value is loaded from the artifact. Start values are loaded from
``{treatment}.exposure`` and end values from
``alternate_{treatment}.exposure``.

The scale-up modifier is only applied when the simulation is configured as
an intervention scenario (i.e. ``intervention.scenario`` is not
``"baseline"``).

See Also
--------

- :ref:`vph_treatment_concept`
- :ref:`therapeutic_inertia_concept`
- :mod:`vivarium_public_health.treatment.scale_up`

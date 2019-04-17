Chronic heart disease
=====================

.. _mslt_reduce_chd:

Intervention: a reduction in CHD incidence
------------------------------------------

.. note:: In this example, we will also use components from the
   :mod:`vivarium_public_health.mslt.disease` module.

Compared to the :ref:`previous simulation <mslt_reduce_acmr>`, we will now add
a chronic disease component, and replace the all-cause mortality rate
intervention with an intervention that affects CHD incidence.

To add CHD as a separate cause of morbidity and mortality, we use the
:class:`~vivarium_public_health.mslt.disease.Disease` component:

.. literalinclude:: /_static/mslt_reduce_chd.yaml
   :language: yaml
   :lines: 7-9,14-15
   :caption: Add a chronic disease.

.. py:currentmodule:: vivarium_public_health.mslt.intervention

We then replace the :class:`ModifyAllCauseMortality` intervention with the
:class:`ModifyDiseaseIncidence` intervention.
We give this intervention a name (``reduce_chd``) and identify the disease
that it affects (``CHD``).
In the configuration settings, we identify this intervention by name
(``reduce_chd``) and specify the scaling factor for CHD incidence
(``CHD_incidence_scale``).

.. literalinclude:: /_static/mslt_reduce_chd.yaml
   :language: yaml
   :lines: 7-9,16-17,22,34-37
   :caption: Add an intervention that reduces CHD incidence.

.. py:currentmodule:: vivarium_public_health.mslt.observer

Finally, we add an observer to record CHD incidence, prevalence, and deaths,
in both the BAU scenario and the intervention scenario.
We use the :class:`Disease` observer, identify the disease of interest by name
(``CHD``), and specify the prefix for output files (``mslt_reduce_chd``).

.. literalinclude:: /_static/mslt_reduce_chd.yaml
   :language: yaml
   :lines: 7-9,18,20,21-22,38-39
   :caption: Record CHD incidence, prevalence, and deaths.

Putting all of these pieces together, we obtain the following simulation
definition:

.. literalinclude:: /_static/mslt_reduce_chd.yaml
   :language: yaml
   :caption: The simulation definition for the BAU scenario and the
      intervention.

Running the model simulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The above simulation is already defined in ``mslt_reduce_chd.yaml``. Run this
simulation with the following command:

.. code-block:: console

   simulate run mslt_reduce_chd.yaml

When this has completed, the output recorded by the
:class:`MorbidityMortality` observer will be saved in the file
``mslt_reduce_chd_mm.csv``.

We can now plot the survival of this cohort in both the BAU and intervention
scenarios, relative to the starting population, and see how the survival rate
has increased as a result of this intervention.

.. _mslt_reduce_chd_fig:

.. figure:: /_static/mslt_reduce_chd_survival.png
   :alt: The survival rates in the BAU and intervention scenarios, and the
      difference between these two rates.

   The impact of reducing the CHD incidence rate by 5% on survival rate.
   Results are shown for the cohort of males aged 50-54 in 2010.
   Compare this to the impact of
   :ref:`reducing all-cause mortality rate by 5% <mslt_reduce_acmr_fig>`.

The output recorded by the :class:`Disease` observer will be saved in the file
``reduce_chd_disease.csv``.
The contents of this file will look like:

.. csv-table:: An extract of the CHD statistics, showing a subset of rows for
   the cohort of males aged 50-54 in 2010.

   **Disease**,**Year of birth**,**Sex**,**Age**,**Year**,**BAU incidence**,**Incidence**,**BAU prevalence**,**Prevalence**,**BAU deaths**,**Deaths**,**Change in incidence**,**Change in prevalence**
   ...
   CHD,1958,male,53,2011,0.005339172657680636,0.005072214024796604,0.040773746292951045,0.04054116957472282,0.58533431153149,0.583569340293451,-0.00026695863288403194,-0.00023257671822822512
   CHD,1958,male,54,2012,0.005698168146464383,0.005413259739141163,0.04517666366247726,0.044700445256575384,1.2175752762775431,1.2105903765985175,-0.0002849084073232198,-0.0004762184059018751
   CHD,1958,male,55,2013,0.006208586286987784,0.005898156972638394,0.04994138950730491,0.04920560358832287,1.911997775650292,1.8961582714695382,-0.0003104293143493895,-0.000735785918982039
   ...
   CHD,1958,male,107,2065,0.039465892849735555,0.037492598207248776,0.19437015273682678,0.1938969881187649,673.2415425866858,655.7541923605287,-0.0019732946424867795,-0.0004731646180618776
   CHD,1958,male,108,2066,0.039465892849735555,0.037492598207248776,0.18918308469826292,0.18921952179569373,687.8956907787912,670.391700818296,-0.0019732946424867795,3.643709743081369e-05
   CHD,1958,male,109,2067,0.039465892849735555,0.037492598207248776,0.1848858097143421,0.1854028718371867,701.5550104751002,684.0712684559271,-0.0019732946424867795,0.0005170621228446082
   ...

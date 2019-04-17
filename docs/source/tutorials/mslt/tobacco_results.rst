Tobacco smoking: effect of interventions
========================================

Each chronic and acute disease that is affected by tobacco smoking is modelled
as a separate component, so that interventions on tobacco smoking can affect
the morbidity and mortality of these diseases.
We also need to inform the tobacco component which diseases it should affect;
this is done in the configuration section.
The resulting simulation definition is quite long, simply because there are
many diseases to include.

.. literalinclude:: /_static/mslt_tobacco_bau.yaml
   :language: yaml
   :caption: The simulation definition for the BAU scenario.
   :name: tobacco_bau_yaml

Tobacco eradication
-------------------

.. py:currentmodule:: vivarium_public_health.mslt.intervention

We add the :class:`TobaccoEradication` component, and specify at what year it
comes into effect.
Shown below are the new lines that are added to the simulation definition
:ref:`for the BAU scenario <tobacco_bau_yaml>`.

.. code:: yaml

   components:
       vivarium_public_health:
           mslt:
               # Other components ...
               intervention:
                   TobaccoEradication()

   configuration:
       # Other configuration settings ...
       tobacco_eradication:
           year: 2011

These simulations are already defined in the following files:

+ ``mslt_tobacco_maori_20-years_decreasing_erad.yaml``
+ ``mslt_tobacco_non-maori_20-years_decreasing_erad.yaml``

Tobacco-free generation
-----------------------

We add the :class:`TobaccoFreeGeneration` component, and specify at what year
it comes into effect.
Shown below are the new lines that are added to the simulation definition
:ref:`for the BAU scenario <tobacco_bau_yaml>`.

.. code:: yaml

   components:
       vivarium_public_health:
           mslt:
               # Other components ...
               intervention:
                   TobaccoFreeGeneration()

   configuration:
       # Other configuration settings ...
       tobacco_free_generation:
           year: 2011

These simulations are already defined in the following files:

+ ``mslt_tobacco_maori_20-years_decreasing_tfg.yaml``
+ ``mslt_tobacco_non-maori_20-years_decreasing_tfg.yaml``

Tobacco tax
-----------

.. py:currentmodule:: vivarium_public_health.mslt.delay

We enable the ``tobacco_tax`` option of the tobacco risk factor
(:class:`DelayedRisk`).
Shown below are the new lines that are added to the simulation definition
:ref:`for the BAU scenario <tobacco_bau_yaml>`.

.. code:: yaml

   configuration:
       # Other configuration settings ...
       tobacco:
           tobacco_tax: True

These simulations are already defined in the following files:

+ ``mslt_tobacco_maori_20-years_decreasing_tax.yaml``
+ ``mslt_tobacco_non-maori_20-years_decreasing_tax.yaml``

Intervention comparison
-----------------------

If you run all of these simulations, you can then compare them by the gains
that they provide in LYs and HALYs, and the reductions that they provide in
ACMR and YLDR, using the data analysis software of your choice.

As an example, here is such a comparison of these 3 interventions:

.. csv-table:: A comparison of the 3 tobacco interventions, showing the gains
   in LYs, HALYs, ACMR, and YLDR.

   "Output","Demographic","Age","Calendar Year","BAU","Tobacco Eradication","Tobacco Eradication CIs","Tobacco Tax","Tobacco Tax CIs","Tobacco-Free Generation","Tobacco-Free Generation CIs"
   "LYs","Total","","","1,041,057,424","6,863,937 (0.66%)","976,461-1,818,974 (0.47-0.87%)","1,316,628 (0.13%)","134,311-395,617 (0.06-0.19%)","2,133,304 (0.20%)","230,529-622,888 (0.11-0.30%)"
   "","Maori female","","","90,220,488","1,739,549 (1.93%)","245,227-458,804 (1.36-2.54%)","416,001 (0.46%)","41,374-126,706 (0.23-0.70%)","714,446 (0.79%)","80,199-202,823 (0.44-1.12%)"
   "","Non-Maori male","","","421,176,895","2,181,064 (0.52%)","313,823-  578,526 (0.37-0.69%)","343,172 (0.08%)","33,959-104,313 (0.04-0.12%)","511,148 (0.12%)","52,758-153,014 (0.06-0.18%)"
   "HALYs","Total","","","846,310,798","7,360,855 (0.87%)","1,044,120-1,957,475 (0.62-1.16%)","1,412,896 (0.17%)","142,510-423,565 (0.09-0.25%)","2,391,733 (0.28%)","261,356-682,703 (0.16-0.40%)"
   "","Maori female","","","70,552,505","1,605,854 (2.28%)","228,578-  418,157 (1.63-2.97%)","389,223 (0.55%)","39,395-117,545 (0.28-0.83%)","690,034 (0.98%)","78,793-190,528 (0.56-1.35%)"
   "","Non-Maori male","","","349,619,482","2,563,709 (0.73%)","364,734-  685,206 (0.53-0.98%)","416,924 (0.12%)","41,830-124,684 (0.06-0.18%)","668,184 (0.19%)","70,963-194,477 (0.10-0.28%)"
   "ACMR","Maori female","63","2041","1157","-266 (-22.96%)","-314 - -214 (-37.36 - -22.64%)","-40 (-3.45%)","-54 - -20 ( -4.91 - -1.81%)","0 (  0.00%)","0 -    0 (  0.00 -   0.00%)"
   "","Maori female","63","2061","1110","-194 (-17.45%)","-271 - -117 (-32.22 - -11.78%)","-77 (-6.95%)","-120 - -40 (-12.12 - -3.71%)","-194 (-17.45%)","-271 - -117 (-32.22 - -11.78%)"
   "","Non-Maori male","63","2041","606","-38 ( -6.33%)","-46 -  -30 ( -8.18 -  -5.29%)","-5 (-0.77%)","-6 -  -2 ( -1.01 - -0.41%)","0 (  0.00%)","0 -    0 (  0.00 -   0.00%)"
   "","Non-Maori male","63","2061","653","-25 ( -3.78%)","-36 -  -15 ( -5.89 -  -2.29%)","-9 (-1.30%)","-14 -  -4 ( -2.11 - -0.67%)","-25 ( -3.78%)","-36 -  -15 ( -5.89 -  -2.29%)"
   "YLDR","Maori female","63","2041","0.228255","-0.010779 (-4.72%)","-0.013300 - -0.008300 (-7.22 - -3.33%)","-0.001398 (-0.61%)","-0.001900 - -0.000700 (-0.90 - -0.28%)","0.000000 ( 0.00%)","0.000000 -  0.000000 ( 0.00 -  0.00%)"
   "","Maori female","63","2061","0.228255","-0.009201 (-4.03%)","-0.012600 - -0.005900 (-6.51 - -2.45%)","-0.003570 (-1.56%)","-0.005400 - -0.001900 (-2.63 - -0.79%)","-0.009201 (-4.03%)","-0.012600 - -0.005900 (-6.51 - -2.45%)"
   "","Non-Maori male","63","2041","0.158340","-0.003632 (-2.29%)","-0.004600 - -0.002700 (-3.62 - -1.48%)","-0.000398 (-0.25%)","-0.000500 - -0.000200 (-0.37 - -0.11%)","0.000000 ( 0.00%)","0.000000 -  0.000000 ( 0.00 -  0.00%)"
   "","Non-Maori male","63","2061","0.158340","-0.002818 (-1.78%)","-0.004100 - -0.001700 (-3.06 - -0.98%)","-0.000947 (-0.60%)","-0.001500 - -0.000500 (-1.05 - -0.29%)","-0.002818 (-1.78%)","-0.004100 - -0.001700 (-3.06 - -0.98%)"

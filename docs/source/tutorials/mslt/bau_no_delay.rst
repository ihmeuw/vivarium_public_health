Immediate recovery upon cessation
=================================

We now consider the case where cessation of smoking results in **immediate**
recovery, rather than taking 20 years for the tobacco-associated relative
risks to decrease back to 1.0.
The purpose here is to highlight how our assumptions about the BAU scenario
can affect the predicted impact of an intervention.

The only changes that we need to make to the simulation definition are:

1. To use a different data artifact for these simulations, where the initial
   prevalence of tobacco use is only defined for 3 exposure levels: never
   smoked, current smoker, and former smoker; and

2. Set the recovery delay to 0 years.

.. note:: We could have used the same data artifact as in previous
   simulations, but then the tobacco component would have to manipulate the
   input data into the appropriate form. We instead choose to perform all
   input data manipulation *before* generating the data artifacts.

.. code:: yaml

   configuration:
       input_data:
           # Change this to "mslt_tobacco_maori_data_0-years.hdf" for the Maori
           # population.
           artifact_path: mslt_tobacco_non-maori_0-years.hdf
       # Other configuration settings ...
       tobacco:
           delay: 0

These simulations are already defined in the following files:

+ Tobacco eradication:

  + ``mslt_tobacco_maori_0-years_decreasing_erad.yaml``
  + ``mslt_tobacco_non-maori_0-years_decreasing_erad.yaml``

+ Tobacco tax:

  + ``mslt_tobacco_maori_0-years_decreasing_tax.yaml``
  + ``mslt_tobacco_non-maori_0-years_decreasing_tax.yaml``

+ Tobacco-free generation:

  + ``mslt_tobacco_maori_0-years_decreasing_tfg.yaml``
  + ``mslt_tobacco_non-maori_0-years_decreasing_tfg.yaml``

Intervention comparison
-----------------------

If you run all of these simulations, you can then compare their effects (and
how these differ to those obtained with the original BAU scenario), using the
data analysis software of your choice.

As an example, here is such a comparison of these 3 interventions:

.. csv-table:: A comparison of the 3 tobacco interventions, showing the gains
   in LYs, HALYs, ACMR, and YLDR.

   "Output","Demographic","Age","Calendar Year","BAU","Tobacco Eradication","Tobacco Eradication CIs","Tobacco Tax","Tobacco Tax CIs","Tobacco-Free Generation","Tobacco-Free Generation CIs"
   "LYs","Total","","","1,041,057,424","7,761,221 (0.75%)","1,056,046-2,130,171 (0.51-1.02%)","1,333,735 (0.13%)","131,064-417,120 (0.06-0.20%)","1,717,623 (0.16%)","159,561-563,966 (0.08-0.27%)"
   "","Maori female","","","90,220,488","2,071,967 (2.30%)","286,663-  556,286 (1.59-3.08%)","436,440 (0.48%)","42,592-138,888 (0.24-0.77%)","619,399 (0.69%)","58,303-197,900 (0.32-1.10%)"
   "","Non-Maori male","","","421,176,895","2,438,781 (0.58%)","330,564-  677,641 (0.39-0.80%)","339,963 (0.08%)","32,089-108,083 (0.04-0.13%)","377,025 (0.09%)","33,468-128,170 (0.04-0.15%)"
   "HALYs","Total","","","846,310,798","8,373,333 (0.99%)","1,152,146-2,286,166 (0.69-1.35%)","1,447,379 (0.17%)","142,262-444,979 (0.08-0.26%)","1,984,274 (0.23%)","191,537-626,052 (0.11-0.37%)"
   "","Maori female","","","70,552,505","1,918,462 (2.72%)","267,972-  506,148 (1.93-3.60%)","411,494 (0.58%)","41,394-128,610 (0.29-0.91%)","614,011 (0.87%)","60,730-188,901 (0.43-1.33%)"
   "","Non-Maori male","","","349,619,482","2,884,740 (0.83%)","392,734-  798,699 (0.57-1.14%)","419,002 (0.12%)","40,911-129,438 (0.06-0.19%)","521,127 (0.15%)","48,459-168,960 (0.07-0.24%)"
   "ACMR","Maori female","63","2041","1157","-318 (-27.54%)","-375 - -243 (-47.99 - -26.63%)","-46 (-4.01%)","-62 - -24 ( -5.64 - -2.10%)","0 (  0.00%)","0 -   0 (  0.00 -  0.00%)"
   "","Maori female","63","2061","1110","-174 (-15.71%)","-272 -  -85 (-32.51 -  -8.26%)","-77 (-6.92%)","-132 - -34 (-13.47 - -3.13%)","-174 (-15.71%)","-272 - -85 (-32.51 - -8.26%)"
   "","Non-Maori male","63","2041","606","-39 ( -6.35%)","-49 -  -27 ( -8.83 -  -4.68%)","-5 (-0.85%)","-7 -  -3 ( -1.17 - -0.44%)","0 (  0.00%)","0 -   0 (  0.00 -  0.00%)"
   "","Non-Maori male","63","2061","653","-19 ( -2.96%)","-32 -  -10 ( -5.19 -  -1.49%)","-7 (-1.07%)","-12 -  -3 ( -1.90 - -0.48%)","-19 ( -2.96%)","-32 - -10 ( -5.19 - -1.49%)"
   "YLDR","Maori female","63","2041","0.228255","-0.013865 (-6.07%)","-0.016800 - -0.010800 (-9.36 - -4.31%)","-0.001654 (-0.72%)","-0.002200 - -0.000800 (-1.07 - -0.35%)","0.000000 ( 0.00%)","0.000000 -  0.000000 ( 0.00 -  0.00%)"
   "","Maori female","63","2061","0.228255","-0.008945 (-3.92%)","-0.012800 - -0.005100 (-6.62 - -2.18%)","-0.003817 (-1.67%)","-0.006000 - -0.001900 (-2.94 - -0.79%)","-0.008945 (-3.92%)","-0.012800 - -0.005100 (-6.62 - -2.18%)"
   "","Non-Maori male","63","2041","0.158340","-0.004255 (-2.69%)","-0.005400 - -0.003200 (-4.27 - -1.72%)","-0.000490 (-0.31%)","-0.000600 - -0.000300 (-0.46 - -0.14%)","0.000000 ( 0.00%)","0.000000 -  0.000000 ( 0.00 -  0.00%)"
   "","Non-Maori male","63","2061","0.158340","-0.002506 (-1.58%)","-0.003900 - -0.001400 (-2.83 - -0.80%)","-0.000889 (-0.56%)","-0.001400 - -0.000400 (-1.02 - -0.25%)","-0.002506 (-1.58%)","-0.003900 - -0.001400 (-2.83 - -0.80%)"

Constant tobacco prevalence
===========================

We now consider the case where the prevalence of tobacco use in each cohort
remains constant over time --- in other words, the cessation rate is zero.
As per the previous tutorial, the purpose here is to highlight how our
assumptions about the BAU scenario can affect the predicted impact of an
intervention.

The only change that we need to make to the simulation definition is:

+ Set the remission rate to zero.

This is done by adding the following configuration option:

.. code:: yaml

   configuration:
       tobacco:
           constant_prevalence: True

These simulations are already defined in the following files:

+ Tobacco eradication:

  + ``mslt_tobacco_maori_20-years_constant_erad.yaml``
  + ``mslt_tobacco_non-maori_20-years_constant_erad.yaml``

+ Tobacco tax:

  + ``mslt_tobacco_maori_20-years_constant_tax.yaml``
  + ``mslt_tobacco_non-maori_20-years_constant_tax.yaml``

+ Tobacco-free generation:

  + ``mslt_tobacco_maori_20-years_constant_tfg.yaml``
  + ``mslt_tobacco_non-maori_20-years_constant_tfg.yaml``

Intervention comparison
-----------------------

If you run all of these simulations, you can then compare their effects (and
how these differ to those obtained with the original BAU scenario), using the
data analysis software of your choice.

As an example, here is such a comparison of these 3 interventions:

.. csv-table:: A comparison of the 3 tobacco interventions, showing the gains
   in LYs, HALYs, ACMR, and YLDR.

   "Output","Demographic","Age","Calendar Year","BAU","Tobacco Eradication","Tobacco Eradication CIs","Tobacco Tax","Tobacco Tax CIs","Tobacco-Free Generation","Tobacco-Free Generation CIs"
   "LYs","Total","","","1,041,057,424","15,814,909 (1.52%)","2,693,518-3,465,246 (1.29-1.66%)","3,494,904 (0.34%)","405,957-900,978 (0.20-0.43%)","6,181,205 (0.59%)","877,025-1,407,409 (0.42-0.68%)"
   "","Maori female","","","90,220,488","3,713,120 (4.12%)","633,573-  822,051 (3.51-4.56%)","999,741 (1.11%)","115,333-265,768 (0.64-1.47%)","1,838,253 (2.04%)","273,556-  411,650 (1.52-2.28%)"
   "","Non-Maori male","","","421,176,895","5,136,703 (1.22%)","887,830-1,122,062 (1.05-1.33%)","968,489 (0.23%)","112,764-248,251 (0.13-0.29%)","1,628,756 (0.39%)","220,864-  377,375 (0.26-0.45%)"
   "HALYs","Total","","","846,310,798","16,031,699 (1.89%)","2,679,605-3,613,018 (1.61-2.11%)","3,495,943 (0.41%)","403,506-911,019 (0.24-0.54%)","6,348,215 (0.75%)","892,419-1,457,169 (0.54-0.86%)"
   "","Maori female","","","70,552,505","3,266,124 (4.63%)","538,690-  739,974 (3.95-5.17%)","875,807 (1.24%)","101,089-235,236 (0.72-1.66%)","1,642,624 (2.33%)","241,152-  372,587 (1.75-2.61%)"
   "","Non-Maori male","","","349,619,482","5,637,442 (1.61%)","951,450-1,264,699 (1.38-1.80%)","1,071,834 (0.31%)","123,514-278,973 (0.18-0.40%)","1,875,759 (0.54%)","254,863-  434,530 (0.37-0.62%)"
   "ACMR","Maori female","63","2041","1157","-388 (-33.59%)","-420 - -354 (-57.01 - -44.07%)","-61 ( -5.28%)","-79 - -33 ( -7.36 - -2.91%)","0 (  0.00%)","0 -    0 (  0.00 -   0.00%)"
   "","Maori female","63","2061","1110","-381 (-34.28%)","-433 - -315 (-63.99 - -39.62%)","-151 (-13.57%)","-203 - -91 (-22.33 - -8.94%)","-381 (-34.28%)","-433 - -315 (-63.99 - -39.62%)"
   "","Non-Maori male","63","2041","606","-61 (-10.08%)","-66 -  -56 (-12.20 - -10.15%)","-8 ( -1.27%)","-10 -  -4 ( -1.60 - -0.71%)","0 (  0.00%)","0 -    0 (  0.00 -   0.00%)"
   "","Non-Maori male","63","2061","653","-56 ( -8.53%)","-69 -  -42 (-11.74 -  -6.91%)","-19 ( -2.91%)","-26 - -11 ( -4.20 - -1.75%)","-56 ( -8.53%)","-69 -  -42 (-11.74 -  -6.91%)"
   "YLDR","Maori female","63","2041","0.228255","-0.014666 (-6.43%)","-0.017100 - -0.012000 ( -9.60 - -4.69%)","-0.002004 (-0.88%)","-0.002600 - -0.001000 (-1.26 - -0.41%)","0.000000 ( 0.00%)","0.000000 -  0.000000 (  0.00 -  0.00%)"
   "","Maori female","63","2061","0.228255","-0.015292 (-6.70%)","-0.018400 - -0.011900 (-10.14 - -4.74%)","-0.005896 (-2.58%)","-0.008200 - -0.003500 (-4.06 - -1.42%)","-0.015292 (-6.70%)","-0.018400 - -0.011900 (-10.14 - -4.74%)"
   "","Non-Maori male","63","2041","0.158340","-0.005186 (-3.28%)","-0.006100 - -0.004200 ( -5.00 - -2.19%)","-0.000596 (-0.38%)","-0.000700 - -0.000300 (-0.54 - -0.18%)","0.000000 ( 0.00%)","0.000000 -  0.000000 (  0.00 -  0.00%)"
   "","Non-Maori male","63","2061","0.158340","-0.005171 (-3.27%)","-0.006500 - -0.003800 ( -5.12 - -2.07%)","-0.001728 (-1.09%)","-0.002400 - -0.001000 (-1.76 - -0.58%)","-0.005171 (-3.27%)","-0.006500 - -0.003800 ( -5.12 - -2.07%)"

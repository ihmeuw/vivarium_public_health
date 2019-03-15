.. _concept_observer:

Recording life table outputs
============================

.. py:currentmodule:: vivarium_public_health.mslt.observer

The multi-state life table contains a vast amount of information for each
population cohort at each time-step of a model simulation.
Since the primary objective of MSLT models is to predict the impact of
preventative interventions on population morbidity and mortality, only some
of these data are relevant and worth recording.

The core concepts are:

1. MSLT components, such as diseases, risk factors, and interventions, will
   record quantities of interest as columns in the population table;

2. Observers will record the values of these columns (and also those of
   columns that identify each cohort, such as their age and sex) at each
   time-step; and

3. At the end of the simulation, observers will concatenate the values
   observed at each time-step into a single table, calculate summary
   statistics (if required), and save the resulting table to disk.

The MSLT framework provides a number of "observers" that record tailored
summary statistics during a model simulation.
We now introduce each of the provided observers in turn.

.. note:: Typically, each observer will record summary statistics for the
   "business-as-usual" (BAU) scenario **and** for the intervention scenario.

Population morbidity and mortality
----------------------------------

The :class:`~MorbidityMortality` observer records the core life table
quantities (as shown in the :ref:`example table <example_mslt_table>`) at each
year of the simulation.
This includes calculating quantities such as the life expectancy and
health-adjusted life expectancy (HALE) for each cohort at each time-step.

Chronic disease incidence, prevalence, and mortality
----------------------------------------------------

The :class:`~Disease` observer records the chronic disease incidence and
prevalence, and the number of deaths caused by this disease, at each year of
the simulation.
For example, with an intervention that :ref:`reduces the incidence of chronic
heart disease (CHD) by 5% <mslt_reduce_chd>` for all cohorts at all
time-steps, it will produce the following output:

.. csv-table::

   **disease**,**year**,**age**,**sex**,**BAU incidence**,**Incidence**,**BAU prevalence**,**Prevalence**,**BAU deaths**,**Deaths**,**Change in incidence**,**Change in prevalence**
   ...,...,...,...,...,...,...,...,...,...,...,...
   CHD,2011,53,male,0.005339172657680636,0.005072214024796604,0.040773746292951045,0.04054116957472282,0.58533431153149,0.583569340293451,-0.00026695863288403194,-0.00023257671822822512
   CHD,2012,54,male,0.005698168146464383,0.005413259739141163,0.04517666366247726,0.044700445256575384,1.2175752762775431,1.2105903765985175,-0.0002849084073232198,-0.0004762184059018751
   ...,...,...,...,...,...,...,...,...,...,...,...
   CHD,2066,108,male,0.039465892849735555,0.037492598207248776,0.18918308469826292,0.18921952179569373,687.8956907787912,670.391700818296,-0.0019732946424867795,3.643709743081369e-05
   CHD,2067,109,male,0.039465892849735555,0.037492598207248776,0.1848858097143421,0.1854028718371867,701.5550104751002,684.0712684559271,-0.0019732946424867795,0.0005170621228446082
   ...,...,...,...,...,...,...,...,...,...,...,...

Risk factor prevalence
----------------------

The :class:`TobaccoPrevalence` observer records the smoking status of each
cohort at each time-step.
Note that all of the post-cessation exposure categories are summed together.

.. csv-table::

   **year**,**age**,**sex**,**BAU never smoked**,**BAU currently smoking**,**BAU previously smoked**,**BAU population**,**Never smoked**,**Currently smoking**,**Previously smoked**,**Population**
   ...,...,...,...,...,...,...,...,...,...,...
   2011,53,male,0.5613260600324522,0.15550808606224473,0.283165853905303,129435.28592207265,0.5613260600324521,0.0,0.4386739399675479,129435.55873403646
   2012,54,male,0.5614856404922235,0.1493582211992655,0.289156138308511,128994.49027457157,0.5614797822892069,0.0,0.4385202177107931,128995.65724459664
   ...,...,...,...,...,...,...,...,...,...,...
   2066,108,male,0.5890673671092423,5.160204075837396e-05,0.4108810308499993,136.85271732801016,0.5650606908555851,0.0,0.4349393091444149,150.1088283279755
   2067,109,male,0.5890897533263759,3.947771215412827e-05,0.41087076896146996,84.60051616850757,0.565060690855585,0.0,0.43493930914441503,92.95319866896016
   ...,...,...,...,...,...,...,...,...,...,...

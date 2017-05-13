In GBD, whenever someone suffers a myocardial infarction (mi), also known as a heart attack, 
they transition into one of three buckets. We're calling this transition the post-mi transitition. 
Once someone has a heart attack, they will have ischemic heart disease for the rest of their 
life (i.e. there is no remission).

The three buckets are as follows:
        - angina
        - asymptomatic ischemic heart disease
        - heart failure due to ischemic heart disease

In CEAM, we need to determine how many people go into each mutually exclusive bucket following a heart attack. 

To determine how many people get angina following an mi, we have a spreadsheet with proportions
of people that get angina following a heart attack. This spreadsheet was used by Catherine Johnson 
for GBD 2015 and is located here: 
/snfs1/WORK/04_epi/01_database/02_data/cvd_ihd/04_models/02_misc_data/angina_prop_postMI.csv 
(I cleaned up the file a bit and put the clean file here: 
/snfs1/Project/Cost_Effectiveness/dev/data_processed/angina_props.csv). 

After determining the proportion of people that will get angina following an mi, we determine 
the proportion of people that get heart failure due to ischemic heart disease. We do not have 
proportion estimates like we did for angina, so we need to use GBD data as a proxy. We first 
calculate the incidence of heart failure due to ischemic heart disease by taking the incidence 
of all heart failure (i.e. the heart failure envelope) and then multiply the incidence of the 
envelope by the proportion of all heart failure due to heart failure due to ischemic heart disease. 
We then convert the incidence rate to a proportion and use that proportion to determine the proportion 
of simulants that should get heart failure.

Finally, everyone that is left gets asymptomatic ihd. We subtract the proportion of people that get 
hf due to ihd and angina from 1, and then use the remaining value as the proportion of people that 
get asymptomatic ihd.

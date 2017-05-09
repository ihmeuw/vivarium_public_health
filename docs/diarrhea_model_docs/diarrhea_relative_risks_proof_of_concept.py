
# coding: utf-8

# # Proof of concept for implementing pathogen specific diarrhea relative risks
# 
# #### See JIRA ticket CE-346 for more information
# #### Do for one draw (draw_0), one sex (Male), one year (1990), and one location (Kenya national) as proof of concept before implementing in CEAM

# In[149]:

import numpy as np
from ceam_inputs import get_relative_risks, get_etiology_specific_prevalence, get_incidence, get_pafs, get_exposures
from ceam_inputs.gbd_ms_auxiliary_functions import get_populations


# #### Optimal breastfeeding practices protect against:
# - norovirus
# - campylobacter
# - salmonella
# - cholera 
# - EPEC
# - shigella
# - rotavirus
# - cryptosporidium
# - amoebiasis
# - ETEC
# - Clostridium difficile (as discussed last meeting, would need to ensure that only people who have accessed a healthcare system + taken antibiotics can become infected)
# 
# #### Optimal breastfeeding does not protect against:
# - Aeromonas
# - Adenovirus
# 
# #### Therefore, we assume that the incidence rate for diarrhea due to aeromonas and diarrhea due to adenovirus is the same in the population that is exposed to sub-optimal breastfeeding and the population that is not exposed to sub-optimal breastfeeding
# 
# #### Biological reasons for protective effect of breast milk
# - Oligosaccharides in breast milk protect against norovirus, campylobacter<sup>1</sup>, salmonella fyris<sup>1</sup>, vibrio cholera<sup>1</sup>, ETEC<sup>1</sup>, and amoebiasis<sup>1</sup>
# - Antibodies in breast milk protect against cholera, campylobacter, EPEC, shigella, salmonella, crypto and has differing results on its protection for rotavirus
# - Lactoferrin in breast milk in vitro protects against shigella, salmonella, EPEC, amoebiasis
# - Hyaluronan protects against salmonella
# - K-casein protects against pathogens adhering to gastrointestinal tract (does not specify which pathogens)
# - Mucins protect norovirus
# - Lactadherin protects against rota
# - Probiotics protect against infection from all pathogens
# - Immunoglobin IgA/sIgA protects against c. difficile
# 
# <sup>1</sup>in vitro
# - source: Turin, Current Tropical Medicine Reports. http://pubmedcentralcanada.ca/pmcc/articles/PMC4036098/#R7

# #### We need to solve the following equation for the RR path specific variable:
# 
# $$ RR = \frac{ \frac{RR_{path \ specific} * num \ exposed \ cases_{due \ to \ affected pathogen_1} + ... + RR_{path \ specific} * Num \ exposed \ cases_{due \ to \ affected pathogen_z} + Num \ exposed \ cases_{due \ to \ aeromonas \ or \ adenovirus}} {Person \ years \ at \ risk_{exposed \ to \ suboptimal \ breastfeeding}}} {\frac {Num \ unexoposed \ cases} {Person \ years \ at \ risk_{unexposed \ to \ suboptimal \ breastfeeding}}} $$
# 
# ##### We can solve to the RR_path_specific variable if we make 2 assumptions
# - The pathogen-specific relative risk is the same for each affected pathogen
# - The incidence rate for diarrhea due to aeromonas and adenovirus is the same in the unexposed/exposed populations

# #### Find incidence rate of diarrhea among those unexposed to suboptimal breastfeeding
# 
# #### We know that: 
# $$ Incidence \ rate \ of \ diarrhea \ among \ unexposed \ to \ unsafe \ handwashing \ = \ Population \ incidence \ rate * \ (1-PAF_{suboptimal breastfeeding}) $$ 

# In[143]:

# Get inicdence rate in the population
population_level_diarrhea_incidence = get_incidence(modelable_entity_id=1181) #modelable_entity_id=diarrhea envelope


# In[144]:

population_level_diarrhea_incidence.head()


# In[150]:

# Get PAF of no handwashing with soap on diarrhea
pafs = get_pafs(risk_id=293, cause_id=302) #no handwashing with soap, cause_id=diarrhea


# In[158]:

pafs.head()

# TODO: Email PJ about why PAFs are 0 for neonates. doesn't seem to make right


# In[159]:

# Get the suboptimal breastfeeding-deleted incidence rate of diarrhea (i.e. incidence rate of diarrhea among the unexposed)
unexposed_inc = pafs.merge(population_level_diarrhea_incidence, on=['year', 'age', 'sex'])
unexposed_inc['1-PAF'] = 1 - unexposed_inc['PAF']
unexposed_inc['unexposed_rate'] = np.multiply(unexposed_inc['rate'], 
                                              unexposed_inc['1-PAF'])
unexposed_inc[['year', 'age', 'sex', 'unexposed_rate']].head()


# #### We can use the incidence rate of diarrhea among the unexposed and the relative risk to solve for the incidence rate among those exposed to suboptimal breastfeeding. We know that:
# $$ RR \ = \frac {Incidence \ rate \ of \ diarrhea \ among \ exposed} {Incidence \ rate \ of \ diarrhea \ among \ unexposed} $$
# 
# ##### Which can be rearranged to:
# 
# $$ Incidence \ rate \ of \ diarrhea \ among \ exposed \ = \ RR \ * \ Incidence \ rate \ of \ diarrhea \ among \ unexposed $$

# In[164]:

# get the relative risk of suboptimal breastfeeding on diarrhea
diarrhea_rr = get_relative_risks(risk_id=238, cause_id=302) #risk_id=no handwashing with soap, cause_id=diarrhea
diarrhea_rr.head()

# TODO: Email PJ about why there are no relative risks for suboptimal breastfeeding


# In[165]:

# now get the incidence rate among the exposed
exposed_inc = unexposed_inc.merge(diarrhea_rr, on=['age', 'sex', 'year'])
exposed_inc['exposed_rate'] = np.multiply(exposed_inc['rr'],
                                          exposed_inc['unexposed_rate'])
exposed_inc[['year', 'age', 'sex', 'unexposed_rate']].head()


# #### Now that we have the incidence rate among the (un)exposed, we need to calculate the number of people that are (un)exposed: 
# $$ Number \ of \ Exposed \ Cases \ = \ Number \ of \ people \ that \ are \ exposed \ * \ Incidence \ Rate_{among \ those \ exposed \ to \ unsafe \ handwashing} $$
# 
# #### Number of people that are (un)exposed can be found using the following formula:
# $$ Number \ of \ (un)exposed \ = \ percent \ (un)exposed \ * \ population$$

# In[166]:

# Get number exposed 
exposure = get_exposures(risk_id=93) # risk_id=suboptimal breastfeeding


# In[167]:

exposure.head()


# In[136]:

# Get population
population = get_populations(180, 1990, 1)
population['sex'] = population.sex_id.map({1: "Male", 2: "Female"})
population.drop('sex_id', axis=1, inplace=True)
population.rename(columns={"year_id" : "year"}, inplace=True)


# In[139]:

# Merge together and calculate population exposed and unexposed
exposure_pop = exposure.merge(population, on=['age', 'sex', 'year'])
exposure_pop['exposed_pop'] = np.multiply(exposure_pop['pop_scaled'], 
                                          exposure_pop['exposure'])
exposure_pop['1-exposure'] = 1 - exposure_pop['exposure']
exposure_pop['unexposed_pop'] = np.multiply(exposure_pop['pop_scaled'], 
                                            exposure_pop['1-exposure'])
exposure_pop[['year', 'age', 'sex', 'exposed_pop', 'unexposed_pop']].head()


# #### Now we calculate the number of people that are (un)exposed, we can calculate number of cases among the (un)exposed:
# $$ Number \ of \ (un)exposed \ cases \ = \ Number \ of \ people \ that \ are \ (un)exposed \ * \ Incidence \ Rate_{among \ those \ (un)exposed \ to \ unsafe \ handwashing} $$

# In[97]:

num_cases_df = exposure_pop.merge(exposed_inc, on=['age', 'sex', 'year'])
num_cases_df['number_unexposed_cases'] = num_cases_df['unexposed_pop'] * num_cases_df['unexposed_rate']
num_cases_df['number_exposed_cases'] = num_cases_df['exposed_pop'] * num_cases_df['exposed_rate']
num_cases_df.head()


# #### After getting the pathogen-specific diarrhea prevalence, we can calculate the number of cases due to specific pathogens in the exposed and unexposed groups
# 
# $$ Number \ of \ Exposed \ Cases \ due \ to \ pathogen = \ Number \ of \ Exposed \ Cases \ * \ Prevalence \ of \ diarrhea \ due \ to \ pathogen$$
# 
# #### Key assumption here is that prevalence of different pathogens is the same in the unexposed and exposed groups

# In[98]:

rota_specific_prevalence = get_etiology_specific_prevalence(eti_risk_id=181, cause_id=302).rename(columns={'eti_prev': 'rota_prev'}) # risk_id=rota, cause_id=diarrhea
rota_specific_prevalence.head()


# In[99]:

shig_specific_prevalence =  get_etiology_specific_prevalence(eti_risk_id=175, cause_id=302).rename(columns={'eti_prev': 'shig_prev'}) # risk_id=shig, cause_id=diarrhea
shig_specific_prevalence.head()


# In[100]:

# merge all of the dataframes together
list_of_dfs = [rota_specific_prevalence, shig_specific_prevalence, num_cases_df]
get_adjusted_rr = list_of_dfs.pop()
for other_df in list_of_dfs:
    get_adjusted_rr = get_adjusted_rr.merge(other_df, on=['age', 'sex', 'year'])
get_adjusted_rr.head()


# In[101]:

# get number of cases in exposed and unexposed groups
for pathogen in ('rota', 'shig'):
    print(pathogen)
    for status in ('exposed', 'unexposed'):
        print(status)
        get_adjusted_rr['{s}_cases_due_to_{p}'.format(s=status, p=pathogen)] =             np.multiply(get_adjusted_rr['{}_pop'.format(status)],
                        get_adjusted_rr['{}_prev'.format(pathogen)])
get_adjusted_rr['number_exposed_cases_not_due_to_shig_or_rota'] = get_adjusted_rr['number_exposed_cases'] -                  get_adjusted_rr['exposed_cases_due_to_rota'] - get_adjusted_rr['exposed_cases_due_to_shig']
get_adjusted_rr.head()


# #### Rearranging the equation above yields:
# $$ RR_{path \ specific} \ = \ Person \ years \ at \ risk_{exposed} * \frac{RR * \frac {Num \ unexoposed \ cases} {Person \ years \ at \ risk_{unexposed}} - Num \ exposed \ cases_{due \ to \ aeromonas \ or \ adenovirus}} {num \ exposed \ cases_{due \ to \ affected pathogen_1} + ... + num \ exposed \ cases_{due \ to \ affected pathogen_z}}$$

# #### Need to solve for RR due to rota/shig
# #### Assume RR due to rota = RR due to Shigella
# #### Outstanding Question: What to do about person years? Are person years the same in the exposed/unexposed group?

# In[108]:

get_adjusted_rr['RR_path_specific'] = get_adjusted_rr['unexposed_pop'] * (get_adjusted_rr['rr'] *                                       get_adjusted_rr['number_unexposed_cases'] / get_adjusted_rr['unexposed_pop'] -                                       get_adjusted_rr['number_unexposed_cases'] - get_adjusted_rr['unexposed_cases_due_to_rota'] - get_adjusted_rr['unexposed_cases_due_to_shig']) / (
                                      get_adjusted_rr['unexposed_cases_due_to_rota'] + get_adjusted_rr['unexposed_cases_due_to_shig'])


# In[109]:

get_adjusted_rr


# does not specify protection against aeromonas or adenovirus

# In[140]:

###### Other Risk Factor Notes
##### C Diff primarily caused in older people who have been exposed to antibiotics
##### Aeromonas primarily caused by exposure to unsafe water
##### TODO: Write up actual documentation on the different pathogens
##### TODO: Match up pathogens and risk factors in a DAG
##### Handwashing can increase risk for all types of diarrhea
##### Sub-optimal breastfeeding can only affect so many risk factors


# ##### Questions
# - What are 'other salmonella infections'?

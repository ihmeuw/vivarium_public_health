
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt, seaborn as sns
get_ipython().magic(u'matplotlib inline')
sns.set_context('paper')
sns.set_style('darkgrid')
pd.set_option('max_rows',5)
from matplotlib.backends.backend_pdf import PdfPages  


# In[3]:

from ceam_inputs import generate_ceam_population, assign_cause_at_beginning_of_simulation, get_prevalence


# In[4]:

pop = generate_ceam_population(180, 1990, 100000)


# In[5]:

number_of_distinct_pops = 50


# In[6]:

dict_of_disease_states = {'severe_heart_failure' : 1823, 'moderate_heart_failure' : 1822, 'mild_heart_failure' : 1821,
                                  'asymptomatic_angina' : 3102, 'mild_angina' : 1818, 'moderate_angina' : 1819, 'severe_angina' : 1820,
                                  'asymptomatic_ihd' : 3233,
                                  'heart_attack' : 1814}


# In[ ]:

pop_w_causes = {}
for i in range(0, number_of_distinct_pops):
    pop_w_causes['{}'.format(i)] = assign_cause_at_beginning_of_simulation(
        pop, 180, 1990, states = dict_of_disease_states)


# In[ ]:

def prevalence_plot(me_id, disease, sex_id):
    
    # specify an outpath for the graphs
    out_path = '/share/costeffectiveness/CEAM/gbd_ms_function_output_plots/{d}_prevalence_among_sex_id{s}.pdf'.    format(d=disease, s=sex_id)
    
    pp = PdfPages(out_path)
    
    prev_dict = {}
    
    for i in range(0, number_of_distinct_pops):
        
        onesex = pop_w_causes['{}'.format(i)].query("sex_id == {}".format(sex_id))

        onedis = onesex.loc[onesex.condition_state == disease]

        # TODO: This is a weird way of getting counts of all of the people
        # with a given disease at a given age. Need to rethink a better way
        # to write the three lines of code below
        
        sick_df = pd.DataFrame()
        sick_df = sick_df.append(pd.value_counts(onedis.age))
        melted_sick = pd.melt(sick_df, var_name = 'age', value_name = 'count_sick')

        totalpop = pop_w_causes['{}'.format(i)].query("sex_id == {}".format(sex_id))
        total_df = pd.DataFrame()
        total_df = total_df.append(pd.value_counts(totalpop.age))

        melted_total = pd.melt(total_df, var_name = 'age', value_name = 'count_total')

        prev_dict['{s}_{i}'.format(s=sex_id, i=i)] = pd.merge(melted_sick, melted_total, on =['age'])

        prev_dict['{s}_{i}'.format(s=sex_id, i=i)]['sex_id'] = sex_id

        prev_dict['{s}_{i}'.format(s=sex_id, i=i)]['prevalence_{}'.format(i)] = prev_dict['{s}_{i}'.format(s=sex_id, i=i)]['count_sick'] /         prev_dict['{s}_{i}'.format(s=sex_id, i=i)]['count_total']

    merged = pd.merge(prev_dict['{}_0'.format(sex_id)], prev_dict['{}_1'.format(sex_id)], on =['age', 'sex_id'], how='outer')
    for i in range(2, number_of_distinct_pops):
        merged = pd.merge(merged, prev_dict['{s}_{i}'.format(s = sex_id, i = i)], on=['age', 'sex_id'], how='outer')

    # This block creates a list of all of the prevalence columns (e.g. ['prevalence_0', 'prevalence_1', ...])
    # Need this list to calculate the mean, upper, and lower
    prevalence_columns = []
    for i in range(0, number_of_distinct_pops):
        prevalence_columns.append('prevalence_{i}'.format(i=i))

    # Fill in the nulls with 0 since null at this point means no one at a given age has the disease of interest
    merged = merged.fillna(0)    

    merged['mean_prev'] = merged[prevalence_columns].mean(axis=1)
    merged['upper'] = merged[prevalence_columns].quantile(0.975, axis=1)
    merged['lower'] = merged[prevalence_columns].quantile(0.025, axis=1)

    merged = merged[['age', 'sex_id'] + prevalence_columns + ['mean_prev', 'upper', 'lower']]

    draw_prev = get_prevalence(me_id)
    draw_prev = draw_prev[['age', 'sex_id', 'draw_0']]
    draw_prev = draw_prev.query("sex_id == {}".format(sex_id))
    draw_prev = draw_prev.loc[draw_prev.age.isin(np.arange(30, 81, 5))]

    plt.figure()

    plt.plot(draw_prev.age.values, draw_prev.draw_0.values, 's', label ='prev from database (draw_0)', color='red')
    plt.errorbar(merged.age.values, merged.mean_prev.values, yerr = [merged.mean_prev.values - merged.lower.values, 
                                                                   merged.upper.values - merged.mean_prev.values], fmt='o', 
                label = 'mean prevalence in simulation w/ error bars')
    
    plt.xlim((30,81))
    plt.title("{d} prevalence among sex_id = {s}".format(d = disease, s = sex_id))
    plt.xlabel('age')
    plt.ylabel('prevalence')
    plt.legend()
    
    pp.savefig()
    plt.clf()

    pp.close()


# In[ ]:

for sex_id in [1, 2]:
    for key, value in states.items():
        prevalence_plot(value, key, sex_id)


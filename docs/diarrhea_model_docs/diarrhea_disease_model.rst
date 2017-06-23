Diarrhea Disease Model Documentation
====================================
Purpose
*******
The purpose of the CEAM diarrhea model is to determine the cost-effectiveness of interventions aimed at reducing the burden of diarrheal diseases. The results from the model can be used to influence policy.

The purpose of the diarrhea component specifically is to accurately simulate the burden of diarrhea in a population of simulants by applying GBD measures of incidence, prevalence, mortality, severity, disability, and remission.

Entities, state variables, and scales
*************************************
Entities are individual simulants. Though diarrhea can be spread between people in truth, there are no infectious disease dynamics or interactions between simulants in CEAM. 

All models in CEAM define simulants by demographic characteristics (e.g. age and sex). In addition to the general states associated with any given CEAM model, the diarrhea model includes a diarrhea state and states that define exposure to the 10 risk factors associated with diarrhea in GBD. The diarrhea state defines whether or not a simulant is susceptible or infected. A simulant can be susceptible (currently free of diarrhea) or have mild, moderate, or severe diarrhea. Simulants are also be exposed or unexposed to the 10 categorical risk factors associated with diarrhea in GBD at a given time.

Process overview and scheduling
*******************************










- We start with a population of healthy people
- We set up a state for each diarrhea pathogen as well as a state for "unattributed" diarrhea (i.e. diarrhea that is not associated with any of the 13 pathogens in GBD). Simulants can remain healthy or get diarrhea (due to one or multiple pathogens) each time step.
    - The incidence rate for diarrhea due to each pathogen is the diarrhea envelope incidence x etiology PAF. 
- Once a simulant gets diarrhea due to a pathogen (or multiple pathogens and/or unattributed), we say that they are in the diarrhea state. We use the severity splits to apportion out all of the diarrhea cases into mild, moderate, and severe diarrhea.
    - One of our key assumptions is that simulants are not susceptible to reinfection during a bout of diarrhea. That is, if a simulant got diarrhea yesterday, they can't get a new infection today. They can become reinfected after the current bout clears up.
    - We get duration of each bout of diarrhea from GBD. Currently, there is no difference in duration between the severity levels. GBD provides us with a mean duration for each age, sex, year, and location. We don't currently have distribution surrounding duration (that is, each simulant that is of a certain age/sex will have the same duration as other simulants of that age/sex) but may want to add some more complexity in the future.
- Only simulants in the severe diarrhea state are subject to an elevated mortality. We calculate the severe diarrhea excess mortality by taking the diarrhea excess mortality rate (that is, the excess mortality among all cases of diarrhea) and dividing by the proportion of severe cases. 
- Each severity level of diarrhea is associated with its own disability weight.


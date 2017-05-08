Diarrhea Disease Model Documentation
====================================
- We start with a population of healthy people
- We set up a state for each diarrhea pathogen as well as a state for "unattributed" diarrhea (i.e. diarrhea that is not associated with any of the 13 pathogens in GBD). Simulants can remain healthy or get diarrhea (due to one or multiple pathogens) each time step.
    - The incidence rate for diarrhea due to each pathogen is the diarrhea envelope incidence x etiology PAF. 
- Once a simulant gets diarrhea due to a pathogen (or multiple pathogens), we say that they are in the diarrhea state. We use the severity splits to apportion out all of the diarrhea cases into mild, moderate, and severe diarrhea.
    - One of our key assumptions is that simulants are not susceptible to reinfection during a bout of diarrhea. That is, if a simulant got diarrhea yesterday, they can't get a new infection today. They can become reinfected after the current bout clears up.
    - We get duration of each bout of diarrhea from GBD. Currently, there is no difference in duration between the severity levels. GBD provides us with a mean duration for each age, sex, year, and location. We don't currently have distribution surrounding duration (that is, each simulant that is of a certain age/sex will have the same duration as other simulants of that age/sex) but may want to add some more complexity in the future.
- Only simulants in the severe diarrhea state are subject to an elevated mortality. We calculate the severe diarrhea excess mortality by taking the diarrhea excess mortality rate (that is, the excess mortality among all cases of diarrhea) and dividing by the proportion of severe cases. 
- Each severity level of diarrhea is associated with its own disability weight.

Questions
*********
1. Are we applying the severity splits correctly?

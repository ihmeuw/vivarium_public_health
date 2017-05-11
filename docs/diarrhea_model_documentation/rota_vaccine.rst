Rotavirus Vaccine Intervention
==============================
- The rota vaccine component handles everything that we care about with regards to the rotavirus vaccine, including bringning estimates of the real world coverage and scaling up coverage as part of potential interventions.
- The first thing that we do is determine who should receive the vaccine. We get age/sex/year/location specific coverage estimates from GBD. If we're running an intervention, we increase the probability of each individual being vaccinated by the probability specified by the intervention (that is, if the intervention specifies that each individual's probability of vaccination should increase by 50%, each in individual's probability of being vaccinated increases by .5).
- For those that do get vaccinated, we accrue vaccine costs and counts
- The vaccine affects an individual's probability of getting diarrhea due to rotavirus. We specify the effectiveness of 3 doses of the vaccine at .39. That is, vaccinated simulants will see a 39% decrease in their (rate or probability, need to confirm) of diarrhea due to rotavirus.
    - It's important to note two things
        - We're well positioned to include onset of immunity (i.e. effect of first/second doses of the vaccine) but have not discussed what the effect (if any) should be
        - In truth, the rota vaccine reduces diarrhea due to multiple pathogens, but we're only allowing it to affect diarrhea due to rota presently
- The vaccine has a shelf life for how long it will last. Currently, we're saying that the vaccine will have an effect after only 14 days after 3 doses have been administered, and that the effect will last for 2 years after the vaccine kicks in
    - We're well positioned to include waning immunity, but haven't discussed what that should look like in the simulation


Questions
*********
1. Should different doses have different durations? Or time in between dosage and immunity being conferred?
2. Should vaccine lose effect 2 years after its administered? Or 2 years after it starts to take effect (i.e. 2 years and 2 weeks after its administered)
3. Confirm whether the rota vaccine coverage is people that receive all vaccines or just the first

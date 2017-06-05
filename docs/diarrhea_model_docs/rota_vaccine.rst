Rotavirus Vaccine Intervention
==============================
- The rota vaccine component handles everything that we care about with regards to the rotavirus vaccine.
- The first thing that we do is determine who should receive the vaccine. Each time step, we check to see if any of our simulants are at an age where they should receive a dose of the vaccine. Simulants that are at an age where they should receive a dose of the vaccine will be given the vaccine or not, depending the probability of being vaccinated. We get age/sex/year/location specific coverage estimates from GBD. If we're running an intervention, we increase the probability of each individual being vaccinated by the probability specified by the intervention (that is, if the intervention specifies that each individual's probability of vaccination should increase by 50%, each in individual's probability of being vaccinated increases by .5). Simulants can only receive the 2nd dose of the vaccine if they received the first dose of the vaccine, just as they can only receive the third dose of the vaccine if they received the 2nd dose.
    - There are several parameters that we can change here. We can change the probability that a simulant should receive the first dose of the vaccine, the probability that a simulant who receives the first vaccine will come back for the second, the probability that a simulant who receives the second vaccine will come back for the third, and the age at which a simulant should receive each dose of the vaccine.
    - For those that do get vaccinated, we accrue vaccine costs and counts
    - An important caveat of our current approach is that the rota vaccine coverage estimate from GBD yields the proportion of people in a given age/sex/year/location that received THE COMPLETE VACCINE SERIES. We're currently using the estimate from GBD to determine the probability that a simulant receives the first dose of the vaccine.
- After determining who should receive the vaccine, we set up the vaccine immunity start/end time. Each dose of the vaccine takes a little while to start conferrring any protection (we can play with this parameter). We say the vaccine immunity start time is the date at which the vaccine was administered + some amount of time before the vaccine will confer any benefit (we've been using 14 days as our estimate currently).
    - We say that the vaccine only confers any benefit during a specific window. Currently, we're saying that the vaccine will confer 
      a benefit for 2 years after the immunity start time and that there is no waning immunity time (that is, the vaccine confers its full benefit for two years and then confers no benefit). We're well positioned to include waning immunity
- The vaccine affects an individual's probability of getting diarrhea due to rotavirus. We specify the effectiveness of 3 doses of the vaccine at .39. That is, vaccinated simulants will see a 39% decrease in their (rate or probability, need to confirm) of diarrhea due to rotavirus.
    - It's important to note two things
        - We're well positioned to include onset of immunity (i.e. effect of first/second doses of the vaccine) but have not discussed what the effect (if any) should be
        - In truth, the rota vaccine reduces diarrhea due to multiple pathogens, but we're only allowing it to affect diarrhea due to rota presently
- The vaccine has a shelf life for how long it will last. Currently, we're saying that the vaccine will have an effect after only 14 days after 3 doses have been administered, and that the effect will last for 2 years after the vaccine's protective effect kicks in
    - We're well positioned to include waning immunity, but haven't discussed what that should look like in the simulation


Questions
*********
1. Should different doses have different durations? Or time in between dosage and immunity being conferred?
2. Should vaccine lose effect 2 years after its administered? Or 2 years after it starts to take effect (i.e. 2 years and 2 weeks after its administered)
3. Confirm whether the rota vaccine coverage is people that receive all vaccines or just the first
4. Right now we've only modeled RotaTeq. Do Ethiopia, Bangladesh, and Nigeria all use RotaTeq?

Future Potential Improvements
*****************************
1. We could potentially use the code I've written to make a more general vaccine component

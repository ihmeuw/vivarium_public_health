import pandas as pd
import numpy as np

# define observer function
# FIXME: Make this flexible enough for any disease state
def count_susceptible_person_time(population_view, disease_col, condition):
    pop = population_view.query("@disease_col != @condition").copy()
    

def count_person_time(population_view):
    # count overall person time in the simulation
    
    


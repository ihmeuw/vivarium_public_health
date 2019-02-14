import pandas as pd


class TreatmentObserver:
    def __init__(self, treatment: str):
        self.treatment = treatment

    def setup(self, builder):

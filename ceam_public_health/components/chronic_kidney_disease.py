"""
Cause 589 is the envelope for all etiologies

stage 3+ is defined by GFR and is currently modeled
stages 1 and 2 are defined by albuminuria which is not currently modeled but these stages seem to be relevant to CVD

The etiologies are probably not relevant to us. They are just proportional splits of the envelope and don't effect mortality.

We do not have GFR distributions internally. GFR is modeled categorically and that model is represented by the prevalence data in the MEs for each CKD stage.
"""

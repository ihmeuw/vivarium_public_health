"""Model for diarrhea with etiologies."""

from collections import namedtuple

from vivarium.framework.population import uses_columns
from vivarium.framework.state_machine import Trigger
from ceam_inputs import (get_severity_splits, get_disability_weight,
                         get_cause_specific_mortality, causes)

from ceam_public_health.disease import DiseaseState, TransientDiseaseState, ExcessMortalityState, DiseaseModel

from .data_transformations import get_etiology_incidence, get_duration_in_days, get_severe_diarrhea_excess_mortality


Etiology = namedtuple('Etiology', ['name', 'model', 'sick_transition', 'recovery_transition', 'pre_trigger_state'])
Etiology.__doc__ = """A container for information about a diarrhea etiology.

Attributes
----------
name : str
    The name of the etiology.
model : `ceam_public_health.disease.DiseaseModel`
    The state-machine model of the etiology.
sick_transition : `vivarium.framework.state_machine.Transition`
    A handle to the transition from this etiology's susceptible state to this etiology's sick state.
recovery_transition : `vivarium.framework.state_machine.Transition`
    A handle to the transition from this etiology's sick state to this etiology's susceptible state.
pre_trigger_state : `ceam_public_health.disease.DiseaseState`
    This etiology's sick state.
"""


def build_etiology_model(etiology_name, infection_side_effect=None):
    """Builds a diarrhea etiology model.

    Parameters
    ----------
    etiology_name : str
        The name of the etiology to build the model for.
    infection_side_effect : callable
        A function to be called when the produced etiology transitions from healthy to sick.

    Returns
    -------
    `Etiology` :
        A container for the etiology model produced.
    """
    healthy = DiseaseState('healthy', track_events=False, key=etiology_name)
    sick = DiseaseState(etiology_name,
                        disability_weight=0,
                        side_effect_function=infection_side_effect)

    sick_transition = healthy.add_transition(sick, rates=get_etiology_incidence(etiology_name),
                                             triggered=Trigger.START_ACTIVE)
    healthy.allow_self_transitions()
    recovery_transition = sick.add_transition(healthy, triggered=Trigger.START_INACTIVE)

    return Etiology(name=etiology_name,
                    model=DiseaseModel(etiology_name, states=[healthy, sick]),
                    sick_transition=sick_transition,
                    recovery_transition=recovery_transition,
                    pre_trigger_state=sick)


def build_diarrhea_model():
    """Builds a model of diarrhea as a symptom of various etiologies.

    Returns
    -------
    list of `ceam_public_health.disease.DiseaseModel` :
        A list of length n where the first n-1 components are the etiology
        `DiseaseModel`s and the final component is the diarrhea `DiseaseModel`"""

    # First we build the states that make up the diarrhea disease model.
    healthy = DiseaseState('healthy', track_events=False, key='diarrhea')
    diarrhea = TransientDiseaseState('diarrhea')
    mild_diarrhea = DiseaseState(causes.mild_diarrhea.name,
                                 disability_weight=get_disability_weight(
                                     healthstate_id=causes.mild_diarrhea.disability_weight),
                                 dwell_time=get_duration_in_days(causes.mild_diarrhea.duration))
    moderate_diarrhea = DiseaseState(causes.moderate_diarrhea.name,
                                     disability_weight=get_disability_weight(
                                         healthstate_id=causes.moderate_diarrhea.disability_weight),
                                     dwell_time=get_duration_in_days(causes.moderate_diarrhea.duration))
    severe_diarrhea = ExcessMortalityState(causes.severe_diarrhea.name,
                                           excess_mortality_data=get_severe_diarrhea_excess_mortality(),
                                           disability_weight=get_disability_weight(
                                               healthstate_id=causes.severe_diarrhea.disability_weight),
                                           dwell_time=get_duration_in_days(causes.severe_diarrhea.duration))

    # Allow healthy to transition into the transient state diarrhea when triggered,
    # otherwise allow it to transition back to itself each time step.
    diarrhea_transition = healthy.add_transition(diarrhea, triggered=Trigger.START_INACTIVE)
    healthy.allow_self_transitions()

    # As diarrhea is a transient state, it immediately moves into one of its sequela
    # based on severity splits.
    diarrhea.add_transition(mild_diarrhea, proportion=get_severity_splits(
            causes.diarrhea.incidence, causes.mild_diarrhea.incidence))
    diarrhea.add_transition(moderate_diarrhea, proportion=get_severity_splits(
            causes.diarrhea.incidence, causes.moderate_diarrhea.incidence))
    diarrhea.add_transition(severe_diarrhea, proportion=get_severity_splits(
            causes.diarrhea.incidence, causes.severe_diarrhea.incidence))

    # Each sequela state will transition back to healthy once the dwell time
    # associated with the state has passed.
    mild_diarrhea.add_transition(healthy)
    moderate_diarrhea.add_transition(healthy)
    severe_diarrhea.add_transition(healthy)

    # As a side effect of getting an etiology, a simulant acquires diarrhea as a symptom
    @uses_columns(['diarrhea'], "alive == 'alive'")
    def cause_diarrhea(index, population_view):
        """Causes a population to move into the diarrheal state.

        Parameters
        ----------
        index : iterable of ints
            An iterable of integer labels for the simulants.
        population_view : `pandas.DataFrame`
            A view of the internal state of the simulation with
            only the `diarrhea` column and only living simulants.
        """
        if not index.empty:
            diarrhea_transition.set_active(index)
            healthy.next_state(index, population_view)
            diarrhea_transition.set_inactive(index)

    # Get a list of the names of all etiologies that cause diarrhea.  See gbd_mapping for more information.
    etiology_names = ['{}'.format(name) for name, etiology in causes.items() if 'gbd_parent_cause' in etiology and
                      etiology.gbd_parent_cause == causes.diarrhea.gbd_cause] + ['unattributed_diarrhea']

    # Build an `Etiology` container for each etiology we're modelling.
    etiologies = [build_etiology_model(name, cause_diarrhea) for name in etiology_names]

    # Our model has simulants recover from their etiologies when they recover from their bout
    # of diarrhea.  Here we set up a factory to generate the functions that will cause each
    # etiology to remit as a side effect of diarrhea remission.
    def etiology_recovery_factory(etiology):
        """Produces a callable that causes remission of an etiology to the susceptible state.

        Parameters
        ----------
        etiology : `Etiology`
            A container for information and objects related to the modelled etiology.

        Returns
        -------
        callable :
            A function that will cause all simulants in the etiology's sick state to remit
            to the etiology's susceptible state.
        """
        @uses_columns([etiology.name], "alive == 'alive'")
        def reset_etiology(index, population_view):
            """Cause all simulants in the etiology's sick state to remit to the etiology's susceptible state.

            Parameters
            ----------
            index : iterable of ints
                An iterable of integer labels for the simulants.
            population_view : `pandas.DataFrame`
                A view of the internal state of the simulation with
                only the etiology column and only living simulants.
            """
            if not index.empty:
                etiology.recovery_transition.set_active(index)
                etiology.pre_trigger_state.next_state(index, population_view)
                etiology.recovery_transition.set_inactive(index)
                # Allow them to get sick again since they're no longer diarrheal.
                etiology.sick_transition.set_active(index)
        return reset_etiology

    recovery_side_effects = [etiology_recovery_factory(etiology) for etiology in etiologies]

    # Build a single callable that will reset all etiologies for the simulants that have
    # remitted from diarrhea and set it as a side effect of people returning to the susceptible diarrhea state.
    def reset_etiologies(index):
        for side_effect in recovery_side_effects:
            side_effect(index)
    healthy.side_effect_function = reset_etiologies

    # Our model allows simulants to get multiple etiologies on a time step, but doesn't allow
    # people who already have diarrhea to get infected with new etiologies. We therefore need a way to
    # make simulants immune to new etiologies while they're currently experiencing diarrhea.
    def immunity_factory(etiology):
        """Produces a callable that causes immunity to an etiology.

        Parameters
        ----------
        etiology : `Etiology`
            A container for information and objects related to the modelled etiology.

        Returns
        -------
        callable :
            A function that will cause all simulants in the etiology's healthy state to become immune to the etiology.
        """
        @uses_columns(['diarrhea_event_time', etiology.name], "alive == 'alive'")
        def make_immune(index, current_time, population_view):
            """Cause all simulants in the etiology's healthy state to become immune to the etiology.

            Parameters
            ----------
            index : iterable of ints
                An iterable of integer labels for the simulants.
            current_time :
                The current simulation time.
            population_view : `pandas.DataFrame`
                A view of the internal state of the simulation with  only the etiology and
                diarrhea_event_time columns and only living simulants.
            """
            if not index.empty:
                pop = population_view.get(index)
                affected_index = pop[pop['diarrhea_event_time'] == current_time].index
                etiology.sick_transition.set_inactive(affected_index)
        return make_immune

    immunity_effects = [immunity_factory(etiology) for etiology in etiologies]

    # Build a single callable that will freeze all transitions to etiology sick states
    def freeze_etiologies(index, current_time):
        for effect in immunity_effects:
            effect(index, current_time)

    # Set it so at the end of a time step, everyone who got diarrhea this time step
    # becomes immune to etiologies they already have.
    mild_diarrhea.cleanup_function = freeze_etiologies
    moderate_diarrhea.cleanup_function = freeze_etiologies
    severe_diarrhea.cleanup_function = freeze_etiologies

    return [etiology.model for etiology in etiologies] + [
        DiseaseModel('diarrhea',
                     states=[healthy, diarrhea, mild_diarrhea, moderate_diarrhea, severe_diarrhea],
                     csmr_data=get_cause_specific_mortality(causes.diarrhea.mortality))
    ]

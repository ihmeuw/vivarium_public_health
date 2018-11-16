import numpy as np
import pandas as pd

from . import add_year_column


class AcuteDisease:
    """
    An acute disease has a sufficiently short duration, relative to the
    time-step size, that it is not meaningful to talk about prevalence.
    Instead, it simply contributes an excess mortality rate, and/or a
    disability rate.

    Interventions may affect these rates:

    - `<disease>_intervention.excess_mortality`
    - `<disease>_intervention.yld_rate`

    where `<disease>` is the name given to an acute disease object.
    """

    def __init__(self, name):
        self.name = name

    def setup(self, builder):
        mty_data = builder.data.load(f'acute_disease.{self.name}.mortality')
        mty_data = add_year_column(builder, mty_data)
        mty_rate = builder.lookup.build_table(mty_data)
        yld_data = builder.data.load(f'acute_disease.{self.name}.morbidity')
        yld_data = add_year_column(builder, yld_data)
        yld_rate = builder.lookup.build_table(yld_data)
        self.excess_mortality = builder.value.register_rate_producer(
            f'{self.name}.excess_mortality',
            source=mty_rate)
        self.int_excess_mortality = builder.value.register_rate_producer(
            f'{self.name}_intervention.excess_mortality',
            source=mty_rate)
        self.disability_rate = builder.value.register_rate_producer(
            f'{self.name}.yld_rate',
            source=yld_rate)
        self.int_disability_rate = builder.value.register_rate_producer(
            f'{self.name}_intervention.yld_rate',
            source=yld_rate)
        builder.value.register_value_modifier('mortality_rate', self.mortality_adjustment)
        builder.value.register_value_modifier('yld_rate', self.disability_adjustment)

    def mortality_adjustment(self, index, mortality_rate):
        delta = self.int_excess_mortality(index) - self.excess_mortality(index)
        return mortality_rate + delta

    def disability_adjustment(self, index, yld_rate):
        delta = self.int_disability_rate(index) - self.disability_rate(index)
        return yld_rate + delta


class Disease:

    def __init__(self, name, simple_eqns=False):
        self.name = name
        self.simple_eqns = simple_eqns

    def setup(self, builder):
        inc_data = builder.data.load(f'chronic_disease.{self.name}.incidence')
        inc_data = add_year_column(builder, inc_data)
        rem_data = builder.data.load(f'chronic_disease.{self.name}.remission')
        rem_data = add_year_column(builder, rem_data)
        mty_data = builder.data.load(f'chronic_disease.{self.name}.mortality')
        mty_data = add_year_column(builder, mty_data)
        yld_data = builder.data.load(f'chronic_disease.{self.name}.morbidity')
        yld_data = add_year_column(builder, yld_data)
        i = builder.lookup.build_table(inc_data)
        r = builder.lookup.build_table(rem_data)
        f = builder.lookup.build_table(mty_data)
        yld_rate = builder.lookup.build_table(yld_data)

        prev_data = builder.data.load(f'chronic_disease.{self.name}.prevalence')
        self.initial_prevalence = builder.lookup.build_table(prev_data)

        self.incidence = builder.value.register_rate_producer(f'{self.name}.incidence', source=i)
        self.incidence_intervention = builder.value.register_rate_producer(f'{self.name}_intervention.incidence',
                                                                           source=i)
        self.remission = builder.value.register_rate_producer(f'{self.name}.remission', source=r)
        self.excess_mortality = builder.value.register_rate_producer(f'{self.name}.excess_mortality', source=f)
        self.disability_rate = builder.value.register_rate_producer(f'{self.name}.yld_rate', source=yld_rate)

        builder.value.register_value_modifier('mortality_rate', self.mortality_adjustment)
        builder.value.register_value_modifier('yld_rate', self.disability_adjustment)

        columns = [f'{self.name}_S', f'{self.name}_S_previous',
                   f'{self.name}_C', f'{self.name}_C_previous',
                   f'{self.name}_S_intervention', f'{self.name}_S_intervention_previous',
                   f'{self.name}_C_intervention', f'{self.name}_C_intervention_previous']
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=columns,
                                                 requires_columns=['age', 'sex'])
        self.population_view = builder.population.get_view(columns)

        builder.event.register_listener('time_step__prepare', self.on_time_step_prepare)

    def on_initialize_simulants(self, pop_data):
        C = 1000 * self.initial_prevalence(pop_data.index)
        S = 1000 - C

        pop = pd.DataFrame({f'{self.name}_S': S,
                            f'{self.name}_C': C,
                            f'{self.name}_S_previous': S,
                            f'{self.name}_C_previous': C,
                            f'{self.name}_S_intervention': S,
                            f'{self.name}_C_intervention': C,
                            f'{self.name}_S_intervention_previous': S,
                            f'{self.name}_C_intervention_previous': C},
                           index=pop_data.index)

        self.population_view.update(pop)

    def on_time_step_prepare(self, event):
        idx = event.index
        pop = self.population_view.get(idx)
        S, C = pop[f'{self.name}_S'], pop[f'{self.name}_C']
        S_int, C_int = pop[f'{self.name}_S_intervention'], pop[f'{self.name}_C_intervention']

        new_S = self.update_S(
            S, C, self.incidence(idx), self.remission(idx), self.excess_mortality(idx),
            self.l(idx), self.q(idx), self.w(idx), self.v(idx), self.simple_eqns
        )
        new_C = self.update_C(
            S, C, self.incidence(idx), self.remission(idx), self.excess_mortality(idx),
            self.l(idx), self.q(idx), self.w(idx), self.v(idx), self.simple_eqns
        )

        new_S_intervention = self.update_S(
            S_int, C_int, self.incidence_intervention(idx), self.remission(idx), self.excess_mortality(idx),
            self.l_intervention(idx), self.q_intervention(idx), self.w_intervention(idx), self.v_intervention(idx)
        )
        new_C_intervention = self.update_C(
            S_int, C_int, self.incidence_intervention(idx), self.remission(idx), self.excess_mortality(idx),
            self.l_intervention(idx), self.q_intervention(idx), self.w_intervention(idx), self.v_intervention(idx)
        )
        pop_update = pd.DataFrame({f'{self.name}_S': new_S,
                                   f'{self.name}_C': new_C,
                                   f'{self.name}_S_previous': S,
                                   f'{self.name}_C_previous': C,
                                   f'{self.name}_S_intervention': new_S_intervention,
                                   f'{self.name}_C_intervention': new_C_intervention,
                                   f'{self.name}_S_intervention_previous': S_int,
                                   f'{self.name}_C_intervention_previous': C_int},
                                  index=pop.index)
        self.population_view.update(pop_update)

    def mortality_adjustment(self, index, mortality_rate):
        pop = self.population_view.get(index)

        S, C = pop[f'{self.name}_S'], pop[f'{self.name}_C']
        S_prev, C_prev = pop[f'{self.name}_S_previous'], pop[f'{self.name}_C_previous']
        D, D_prev = 1000 - S - C, 1000 - S_prev - C_prev

        S_int, C_int = pop[f'{self.name}_S_intervention'], pop[f'{self.name}_C_intervention']
        S_int_prev, C_int_prev = pop[f'{self.name}_S_intervention_previous'], pop[f'{self.name}_C_intervention_previous']
        D_int, D_int_prev = 1000 - S_int - C_int, 1000 - S_int_prev - C_int_prev

        mortality_risk = (D - D_prev) / (S + C)
        mortality_risk_int = (D_int - D_int_prev) / (S_int + C_int)

        cause_mortality_rate = -np.log(1 - mortality_risk)
        cause_mortality_rate_int = -np.log(1 - mortality_risk_int)

        delta = cause_mortality_rate_int - cause_mortality_rate
        return mortality_rate + delta

    def disability_adjustment(self, index, yld_rate):
        pop = self.population_view.get(index)

        S, S_prev = pop[f'{self.name}_S'], pop[f'{self.name}_S_previous']
        C, C_prev = pop[f'{self.name}_C'], pop[f'{self.name}_C_previous']
        S_int, S_int_prev = pop[f'{self.name}_S_intervention'], pop[f'{self.name}_S_intervention_previous']
        C_int, C_int_prev = pop[f'{self.name}_C_intervention'], pop[f'{self.name}_C_intervention_previous']

        # The prevalence rate is the mean number of diseased people over the
        # year, divided by the mean number of alive people over the year.
        # The 0.5 multipliers in the numerator and denominator therefore cancel
        # each other out, and can be removed.
        prevalence_rate = (C + C_prev) / (S + C + S_prev + C_prev)
        prevalence_rate_int = (C_int + C_int_prev) / (S_int + C_int + S_int_prev + C_int_prev)

        delta = prevalence_rate_int - prevalence_rate
        return yld_rate + self.disability_rate(index) * delta

    def l(self, index):
        i = self.incidence(index)
        r = self.remission(index)
        f = self.excess_mortality(index)

        return i + r + f

    def l_intervention(self, index):
        i = self.incidence_intervention(index)
        r = self.remission(index)
        f = self.excess_mortality(index)

        return i + r + f

    def q(self, index):
        i = self.incidence(index)
        r = self.remission(index)
        f = self.excess_mortality(index)

        return np.sqrt(i**2 + r**2 + f**2 + i*r + f*r - i*f)

    def q_intervention(self, index):
        i = self.incidence_intervention(index)
        r = self.remission(index)
        f = self.excess_mortality(index)

        return np.sqrt(i ** 2 + r ** 2 + f ** 2 + i * r + f * r - i * f)

    def w(self, index):
        l = self.l(index)
        q = self.q(index)

        return np.exp(-(l + q) / 2)

    def w_intervention(self, index):
        l = self.l_intervention(index)
        q = self.q_intervention(index)

        return np.exp(-(l + q) / 2)

    def v(self, index):
        l = self.l(index)
        q = self.q(index)

        return np.exp(-(l - q) / 2)

    def v_intervention(self, index):
        l = self.l_intervention(index)
        q = self.q_intervention(index)

        return np.exp(-(l - q) / 2)

    @staticmethod
    def update_S(S, C, i, r, f, l, q, w, v, simple_eqns=False):
        # NOTE: try using simpler no-remission equations.
        if simple_eqns and all(r == 0):
            return S * np.exp(-i)
        new_S = (2*(v - w)*(S*(f + r) + C*r) + S*(v*(q - l) + w*(q + l))) / (2 * q)
        new_S[q == 0] = S[q == 0]
        return new_S

    @staticmethod
    def update_C(S, C, i, r, f, l, q, w, v, simple_eqns=False):
        # NOTE: try using simpler no-remission equations.
        if simple_eqns and all(r == 0):
            return S * S * (1 - np.exp(-i)) + C * np.exp(-f)
        new_C = -((v - w)*(2*((f + r)*(S + C) - l*S) - l*C) - (v + w)*q*C) / (2 * q)
        new_C[q == 0] = C[q == 0]
        return new_C

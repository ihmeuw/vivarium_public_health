from vivarium.framework.engine import Builder
from vivarium.framework.results import Observer


class HealthObserver(Observer):
    def register_adding_observation(
        self,
        builder: Builder,
        name,
        pop_filter,
        when,
        requires_columns,
        additional_stratifications,
        excluded_stratifications,
        aggregator,
    ):
        builder.results.register_adding_observation(
            name=name,
            pop_filter=pop_filter,
            when=when,
            requires_columns=requires_columns,
            results_formatter=self.formatter,
            additional_stratifications=additional_stratifications,
            excluded_stratifications=excluded_stratifications,
            aggregator=aggregator,
        )

    def formatter(self, measure, df):
        _format(...)
        df["measure"] = get_measure_col(...)
        df["entity_type"] = get_entity_type_col(...)
        df["entity"] = get_entity_col(...)
        df["sub_entity"] = get_sub_entity_col(...)
        return df

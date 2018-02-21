from code import interact

from vivarium.framework.event import listens_for


class Inspector:
    def setup(self, builder):
        self.population_view = builder.population_view()

    @listens_for('collect_metrics')
    def inspect(self, event):
        interact(banner="""
            Inspecting population at {}.
            Population table in 'population'.
            To stop the simulation use 'exit()'.
            To continue to the next time step use 'Ctrl-D' ('Ctrl-Z' on Windows).
    """.format(event.time), local={'population': self.population_view.get(event.index)})

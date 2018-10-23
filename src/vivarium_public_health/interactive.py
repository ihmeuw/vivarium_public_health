from vivarium.interface import interactive
from vivarium.framework.configuration import build_simulation_configuration, build_model_specification
from vivarium.framework.plugins import PluginManager
from vivarium.config_tree import ConfigTree

ARTIFACT_PLUGIN = {
    'optional': {
        'data': {
            'controller': 'vivarium_public_health.dataset_manager.ArtifactManager',
            'builder_interface': 'vivarium_public_health.dataset_manager.ArtifactManagerInterface'
        }
    }
}



def initialize_simulation(components, input_config=None, plugin_config=None):
    plugin_config = ARTIFACT_PLUGIN.update(plugin_config) if plugin_config else ARTIFACT_PLUGIN
    return interactive.initialize_simulation(components, input_config, plugin_config)


def setup_simulation(components, input_config=None, plugin_config=None):
    simulation = initialize_simulation(components, input_config, plugin_config)
    simulation.setup()

    return simulation


def initialize_simulation_from_model_specification(model_specification_file):
    model_specification = build_model_specification(model_specification_file)

    plugin_config = ConfigTree(ARTIFACT_PLUGIN)
    plugin_config.update(model_specification.plugins)
    component_config = model_specification.components
    simulation_config = model_specification.configuration

    plugin_manager = PluginManager(plugin_config)
    component_config_parser = plugin_manager.get_plugin('component_configuration_parser')
    components = component_config_parser.get_components(component_config)

    return interactive.InteractiveContext(simulation_config, components, plugin_manager)


def setup_simulation_from_model_specification(model_specification_file):
    simulation = initialize_simulation_from_model_specification(model_specification_file)
    simulation.setup()

    return simulation




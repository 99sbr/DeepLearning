import os
import json
config_file = os.path.normpath(os.path.join(os.path.dirname(__file__), 'config.json'))
class Configurations(object):

    def __init__(self):
        config = json.load(open(config_file))
        self.IM_WIDTH = config.IM_WIDTH
        self.IM_HEIGHT = config.IM_HEIGHT
        self.NB_IV3_LAYERS_TO_FREEZE = config.NB_IV3_LAYERS_TO_FREEZE
        self.learning_rate = config.learning_rate
        self.requirements_directory = config.requirements_directory
        self.logs_directory = config.logs_directory
        self.published_output_directory = config.published_output_directory

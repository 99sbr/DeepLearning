import logging
import sys
import os
from logging.handlers import TimedRotatingFileHandler
from load_config import Configurations

config = Configurations()

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")


class LogHandler(object):
    def __init__(self):
        self.LOG_FILE = os.path.join(config.logs_directory,'ImageClassification.log')

    def get_console_handler(self):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(FORMATTER)
        return console_handler

    def get_file_handler(self):
        file_handler = TimedRotatingFileHandler(self.LOG_FILE, when='D')
        file_handler.setFormatter(FORMATTER)
        return file_handler

    def get_logger(self, logger_name):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)  # better to have too much log than not enough
        logger.addHandler(self.get_console_handler())
        logger.addHandler(self.get_file_handler())
        # with this pattern, it's rarely necessary to propagate the error up to parent
        logger.propagate = False
        return logger

# my_logger = LogHandler().get_logger("my module name")
# my_logger.info("a debug message")

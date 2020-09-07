import ruamel.yaml
from pathlib import Path

from utils.load_config import LoadConfig



class PrintConfig:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def _print_dict_value(values, key_name, level=0, tab_size=2):
        tab = level * tab_size * ' '
        print(tab + '-', key_name, ':', values)

    def _print_dict(self, dictionary, recursion_level=0):
        for key in dictionary.keys():
            if isinstance(key, dict):
                recursion_level += 1
                self._print_dict(dictionary[key], recursion_level)
            else:
                self._print_dict_value(dictionary[key], key_name=key, level=recursion_level)


    def print_config(self):
        print('\nCheck Configure')
        self._print_dict(self.config)


if __name__ == '__main__':
    config = LoadConfig('conf').load_config()
    pconf = PrintConfig(config).print_config()
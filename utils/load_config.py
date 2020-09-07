import ruamel.yaml
from pathlib import Path


class LoadConfig:
    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.yaml = ruamel.yaml.YAML()

    def _load_config(self):
        with open(str(self.config_path / 'text_config.yaml'), 'rb') as config_file:
            config_dict = self.yaml.load(config_file)

        full_config_dict  = {}
        full_config_dict.update(config_dict)

        return full_config_dict

    def load_config(self):
        config = self._load_config()
        return config
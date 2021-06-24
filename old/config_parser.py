import yaml


class ArgDict():
    def __init__(self, yaml_path):
        with open(yaml_path, 'r') as f:
            data = yaml.load(f)

        self._dict = {
            k: ArgDict(v) if isinstance(v, dict) else v for k, v in data.items()
        }

    def __getattr__(self, x):
        if x in self._dict.keys():
            return self._dict[x]
        else:
            raise ValueError(f'{x} is not a member')

import yaml


class Struct:
    def __init__(self, **entries):
        for k,v in entries.items():
            if isinstance(v, dict):
                setattr(self, k, Struct(**v))
            else:
                try:
                    v = eval(v)
                except:
                    pass
                setattr(self, k, v)


def load_yaml(path):
    with open(path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def load_as_object(path):
    dictionary = load_yaml(path)
    return Struct(**dictionary)
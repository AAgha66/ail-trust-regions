import yaml


def get_args_dict(config):
    with open(config) as info:
        args_dict = yaml.safe_load(info)
    return args_dict

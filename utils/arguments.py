import yaml


def get_args_dict(config):
    with open(config) as info:
        args_dict = yaml.safe_load(info)
        assert args_dict['algo'] in ['ppo', 'trpo']
        args_dict['cuda'] = not args_dict['no_cuda']
        if args_dict['eval_interval'] == 'None':
            args_dict['eval_interval'] = None
    return args_dict

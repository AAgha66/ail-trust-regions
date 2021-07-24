from cw2 import experiment, cw_error
from cw2 import cluster_work
from cw2.cw_data import cw_logging
from utils.arguments import get_args_dict
from main import main

class MyExperiment(experiment.AbstractExperiment):
    
    def initialize(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        pass

    def run(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        args_dict = get_args_dict(config=config.params['config_file'])
        args_dict['num_env_steps'] = config.params['num_env_steps']
        args_dict['env_name'] = config.params['env_name']

        args_dict['logging'] = config.params['logging']
        args_dict['summary'] = config.params['summary']
        args_dict['gradient_penalty'] = config.params['gradient_penalty']

        args_dict['lr_disc'] = config.params['lr_disc']
        args_dict['lr_policy'] = config.params['lr_policy']
        args_dict['lr_value'] = config.params['lr_value']
        
        args_dict['lambda_gae'] = config.params['lambda_gae']
        args_dict['gamma'] = config.params['gamma']
        args_dict['seed'] = config.params['seed']

        if config.params['use_proj']:
            args_dict['proj_type'] = config.params['proj_type']
            args_dict['cov_bound'] = config.params['cov_bound']
            args_dict['mean_bound'] = config.params['mean_bound']
            args_dict['trust_region_coeff'] = config.params['trust_region_coeff']
        main(config=None, args_dict=args_dict)
    
    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass

if __name__ == "__main__":
    # Give the MyExperiment Class, not MyExperiment() Object!!
    cw = cluster_work.ClusterWork(MyExperiment)
    # RUN!
    cw.run()
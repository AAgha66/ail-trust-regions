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

        args_dict['logging'] = config.params['logging']
        args_dict['summary'] = config.params['summary']
        args_dict['save_model'] = config.params['save_model']
        args_dict['track_vf'] = config.params['track_vf']
        args_dict['track_grad_kurtosis'] = config.params['track_grad_kurtosis']
        args_dict['gail_experts_dir'] = config.params['gail_experts_dir']
        args_dict['log_dir'] = config.params['log_dir']
        args_dict['num_trajectories'] = config.params['num_trajectories']
        args_dict['num_env_steps'] = config.params['num_env_steps']

        args_dict['logging_dir'] = config.params['logging_dir']
        args_dict['seed'] = config.params['seed']
        args_dict['env_name'] = config.params['env_name']

        args_dict['clip_importance_ratio'] = config.params['clip_importance_ratio']
        args_dict['use_kl_penalty'] = config.params['use_kl_penalty']
        args_dict['use_rollback'] = config.params['use_rollback']
        args_dict['use_tr_ppo'] = config.params['use_tr_ppo']
        args_dict['use_truly_ppo'] = config.params['use_truly_ppo']
        args_dict['gradient_penalty'] = config.params['gradient_penalty']

        args_dict['lr_disc'] = config.params['lr_disc']
        args_dict['lr_policy'] = config.params['lr_policy']
        args_dict['lr_value'] = config.params['lr_value']
        args_dict['gae_lambda'] = config.params['gae_lambda']
        args_dict['gamma'] = config.params['gamma']
        args_dict['entropy_coef'] = config.params['entropy_coef']

        if config.params['use_proj']:
            args_dict['proj_type'] = config.params['proj_type']
            args_dict['entropy_schedule'] = config.params['entropy_schedule']

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
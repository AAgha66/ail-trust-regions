import optuna
from utils.arguments import get_args_dict
import numpy as np
from main import main
import yaml
from joblib import Parallel, delayed
import optuna


def get_args(trial, config):
    args_dict = get_args_dict(config=config.params['config_file'])
    args_dict['num_env_steps'] = config.params['num_env_steps']
    args_dict['env_name'] = config.params['env_name']

    args_dict['logging'] = config.params['logging']
    args_dict['summary'] = config.params['summary']
    args_dict['save_model'] = config.params['save_model']
    args_dict['track_vf'] = config.params['track_vf']
    args_dict['track_grad_kurtosis'] = config.params['track_grad_kurtosis']

    args_dict['gail_experts_dir'] = config.params['gail_experts_dir']
    args_dict['logging_dir'] = config.params['logging_dir'] + str(trial.number) + '/'
    args_dict['log_dir'] = config.params['log_dir']

    args_dict['clip_importance_ratio'] = config.params['clip_importance_ratio']
    args_dict['gradient_penalty'] = config.params['gradient_penalty']
    args_dict['spectral_norm'] = config.params['spectral_norm']
    args_dict['airl_reward'] = config.params['airl_reward']

    args_dict['lr_disc'] = trial.suggest_categorical("lr_disc", [3e-6, 1.0e-5, 3.0e-5, 1e-4, 3e-4, 1e-3])
    args_dict['lr_policy'] = trial.suggest_categorical("lr_policy", [3e-5, 1.0e-4, 3.0e-4, 1.0e-3])
    args_dict['lr_value'] = trial.suggest_categorical("lr_value", [3e-5, 1.0e-4, 3.0e-4, 1.0e-3])

    args_dict['gae_lambda'] = trial.suggest_categorical("gae_lambda", [0.95, 0.96, 0.97, 0.98])
    args_dict['gamma'] = trial.suggest_categorical("gamma", [0.97, 0.99, 0.997])

    if config.params['use_entropy_pen']:
        args_dict['entropy_coef'] = trial.suggest_float("entropy_coef", 1e-4, 1e-2, log=True)
    else:
        args_dict['entropy_coef'] = 0

    if config.params['use_proj']:
        args_dict['proj_type'] = config.params['proj_type']
        args_dict['entropy_schedule'] = config.params['entropy_schedule']
        args_dict['cov_bound'] = trial.suggest_float("cov_bound", 1e-5, 1e-2, log=True)
        args_dict['mean_bound'] = trial.suggest_float("mean_bound", 1e-4, 1e-1, log=True)
        args_dict['trust_region_coeff'] = trial.suggest_int("trust_region_coeff", 4, 16, step=2)

    return args_dict


def objective_wrapper(trial, config):
    args_dict = get_args(trial, config)
    seeds = [0, 1, 2]
    dicts = []

    for seed in seeds:
        tmp = args_dict.copy()
        tmp['seed'] = seed
        dicts.append(tmp)

    rewards_moving_avgs = Parallel(n_jobs=3)(delayed(main)(None, dict) for dict in dicts)
    return np.mean(rewards_moving_avgs)  # Aggregate results and determine the score.


def run_study(config):
    study_name = config.params['study_name'] + config.params['env_name'] 
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage_name, load_if_exists=config.params['load_if_exists'])
    study.optimize(lambda trial: objective_wrapper(trial, config), n_trials=config.params['n_trials'])

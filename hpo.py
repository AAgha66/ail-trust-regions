import optuna
from utils.arguments import get_args_dict
import numpy as np
from main import main
import yaml
from joblib import Parallel, delayed


def get_args(trial, config):
    args_dict = get_args_dict(config=config.params['config_file'])
    args_dict['num_env_steps'] = config.params['num_env_steps']
    args_dict['env_name'] = config.params['env_name']

    args_dict['logging'] = config.params['logging']
    args_dict['summary'] = config.params['summary']
    args_dict['gradient_penalty'] = config.params['gradient_penalty']

    args_dict['lr_disc'] = trial.suggest_categorical("lr_disc", [3e-6, 1.0e-5, 3.0e-5, 1e-4])    
    args_dict['lr_policy'] = trial.suggest_categorical("lr_policy", [3e-5, 1.0e-4, 3.0e-4])
    args_dict['lr_value'] = trial.suggest_categorical("lr_value", [3e-5, 1.0e-4, 3.0e-4])
    args_dict['lambda_gae'] = trial.suggest_categorical("lambda_gae", [0.95, 0.96, 0.97, 0.98])
    args_dict['gamma'] = trial.suggest_categorical("gamma", [0.97, 0.99, 0.997])

    if config.params['use_proj']:
        args_dict['proj_type'] = config.params['proj_type']
        args_dict['cov_bound'] = trial.suggest_float("cov_bound", 1e-5, 1e-2, log=True)
        args_dict['mean_bound'] = trial.suggest_float("mean_bound", 1e-4, 1e-1, log=True)
        args_dict['trust_region_coeff'] = trial.suggest_int("trust_region_coeff", 4, 16, step=4)

    return args_dict


def objective_wrapper(trial, config):
    args_dict = get_args(trial, config)
    seeds = [0, 1, 2]
    dicts = []

    for seed in seeds:
        tmp = args_dict.copy()
        tmp['seed'] = seed
        dicts.append(tmp)

    rewards_moving_avgs = Parallel(n_jobs=4)(delayed(main)(None, dict) for dict in dicts)
    return -np.mean(rewards_moving_avgs)  # Aggregate results and determine the score.


def run_study(config):
    study = optuna.create_study()
    study.optimize(lambda trial: objective_wrapper(trial, config), n_trials=config.params['n_trials'])
    print(study.best_trial)


if __name__ == "__main__":
    with open('test.yml') as f:
        dataMap = yaml.safe_load(f)
        print(dataMap)
        run_study(dataMap)

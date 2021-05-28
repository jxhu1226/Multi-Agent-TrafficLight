import argparse
import configparser

from envs.test_grid_env import TestGridController, TestGridEnv
from utils import init_dir, init_log, copy_file, init_test_mode, Counter, Trainer
import logging
from agents.model import IQL
import tensorflow as tf

def parse_args():
    default_base_dir = '/home/jackson/Desktop/FinalProject/traffic_project'
    default_env_name = 'test'
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, required=True, default=default_base_dir, help='experiment base dir')
    subparsers = parser.add_subparsers(dest='option', help='train or evaluate')
    # train arguments
    sp = subparsers.add_parser('train', help='train a single agent under base dir')
    sp.add_argument('--test-mode', type=str, required=False, default='no_test', help='test mode during training',
                    choices=['no_test', 'in_train_test', 'after_train_test', 'all_test'])
    sp.add_argument('--env-name', type=str, required=False, default=default_env_name,
                    help='experiment environment path')
    # evaluate arguments
    sp = subparsers.add_parser('evaluate', help='evaluate and compare agents under base dir')
    sp.add_argument('--agents', type=str, required=False, default='naive',
                    help="agent folder names for evaluation, split by ,")
    sp.add_argument('--evaluation-policy-type', type=str, required=False, default='default',
                    help="inference policy type in evaluation: default, stochastic, or deterministic")
    sp.add_argument('--evaluation-seeds', type=str, required=False,
                    default=','.join([str(i) for i in range(10000, 100001, 10000)]),
                    help="random seeds for evaluation, split by ,")
    sp.add_argument('--demo', action='store_true', help="shows SUMO gui")
    args = parser.parse_args()
    if not args.option:
        parser.print_help()
        exit(1)
    return args


def init_env(config, port=0, naive_policy=False):
    if config.get('scenario') == 'test':
        if not naive_policy:
            return TestGridEnv(config, port=port)
        else:
            env = TestGridEnv(config, port=port)
            policy = TestGridController(env.node_names)
            return env, policy
    else:
        if not naive_policy:
            return None
        else:
            return None, None


def train(args):
    base_dir = 'experiments/' + args.base_dir
    dirs = init_dir(base_dir)
    init_log(dirs['log'])
    config_path = 'config/' + args.env_name +'/config.ini'
    copy_file(config_path, dirs['data'])
    config = configparser.ConfigParser()
    config.read(config_path)
    in_test, post_test = init_test_mode(args.test_mode)

    env = init_env(config['ENV_CONFIG'])
    logging.info('Training: s dim: %d, a dim %d, s dim ls: %r, a dim ls: %r' %
                 (env.n_s, env.n_a, env.n_s_ls, env.n_a_ls))

    # init step counter
    total_step = int(config.getfloat('TRAIN_CONFIG', 'total_step'))
    test_step = int(config.getfloat('TRAIN_CONFIG', 'test_interval'))
    log_step = int(config.getfloat('TRAIN_CONFIG', 'log_interval'))
    global_counter = Counter(total_step, test_step, log_step)

    # init centralized or multi agent
    seed = config.getint('ENV_CONFIG', 'seed')
    model = None
    if env.agent == 'ia2c':
        # model = IA2C(env.n_s_ls, env.n_a_ls, env.n_w_ls, total_step,
        #              config['MODEL_CONFIG'], seed=seed)
        pass
    elif env.agent == 'ma2c':
        # model = MA2C(env.n_s_ls, env.n_a_ls, env.n_w_ls, env.n_f_ls, total_step,
        #              config['MODEL_CONFIG'], seed=seed)
        pass
    elif env.agent == 'iqld':
        model = IQL(env.n_s_ls, env.n_a_ls, env.n_w_ls, total_step, config['MODEL_CONFIG'],
                    seed=0, model_type='dqn')
    else:
        # model = IQL(env.n_s_ls, env.n_a_ls, env.n_w_ls, total_step, config['MODEL_CONFIG'],
        #             seed=0, model_type='lr')
        pass
    summary_writer = tf.summary.FileWriter(dirs['log'])
    trainer = Trainer(env, model, global_counter, summary_writer, in_test, output_path=dirs['data'])
    trainer.run()


def evaluate(args):
    pass


if __name__ == '__main__':
    args = parse_args()
    if args.option == 'train':
        train(args)
    elif args.option == 'evaluate':
        evaluate(args)
    else:
        print('please specify an available option.')

import tensorflow as tf
from agents.policy import DeepQPolicy
import numpy as np
from utils import Scheduler, ReplayBuffer

class IQL:
    def __init__(self, num_state_ls, num_action_ls, num_wait_ls,
                 total_step, model_config, seed=0, model_type='dqn'):
        self.name = 'iql'
        self.model_type = model_type
        self.model_type = model_type
        self.agents = []
        self.n_agent = len(num_state_ls)
        '''
            这俩是干嘛用的？
        '''
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        self.num_state_ls = num_state_ls
        self.num_action_ls = num_action_ls
        self.num_wait_ls = num_wait_ls
        self.n_step = model_config.getint('batch_size')
        # init tf
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        '''
            tf.ConfigProto 用来给 tensorflow 的session 配置信息
            tf.ConfigProto(allow_soft_placement=True): 如果是true，则允许tensorflow自动分配设备
        '''
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.policy_ls = []
        for i, (n_state, n_action, n_wait) in enumerate(
                zip(self.num_state_ls, self.num_action_ls, self.num_action_ls)):
            self.policy_ls.append(self._init_policy(n_state, n_action,
                                                    n_wait, model_config, agent_name='{:d}a'.format(i)))
        self.saver = tf.train.Saver(max_to_keep=5)
        if total_step:
            # training
            self.total_step = total_step
            self._init_scheduler(model_config)
            self._init_train(model_config)
        self.cur_step = 0
        self.sess.run(tf.global_variables_initializer())

    def _init_policy(self, n_state, n_action, n_wait, model_conifg, agent_name=None):
        policy = None
        if self.model_type == 'dqn':
            n_h = model_conifg.getint('num_h')
            n_fc = model_conifg.getint('num_fc')
            policy = DeepQPolicy(n_state - n_wait, n_action, n_wait, self.n_step, n_fc0=n_fc, n_fc=n_h, name=agent_name)
        return policy

    def _init_scheduler(self, model_config):
        lr_init = model_config.getfloat('lr_init')
        lr_decay = model_config.get('lr_decay')
        eps_init = model_config.getfloat('epsilon_init')
        eps_decay = model_config.get('epsilon_decay')
        if lr_decay == 'constant':
            self.lr_scheduler = Scheduler(lr_init, decay=lr_decay)
        else:
            lr_min = model_config.getfloat('lr_min')
            self.lr_scheduler = Scheduler(lr_init, lr_min, self.total_step, decay=lr_decay)
        if eps_decay == 'constant':
            self.eps_scheduler = Scheduler(eps_init, decay=eps_decay)
        else:
            eps_min = model_config.getfloat('epsilon_min')
            eps_ratio = model_config.getfloat('epsilon_ratio')
            self.eps_scheduler = Scheduler(eps_init, eps_min, self.total_step * eps_ratio,
                                           decay=eps_decay)

    def _init_train(self, model_config):
        # init loss
        max_grad_norm = model_config.getfloat('max_grad_norm')
        gamma = model_config.getfloat('gamma')
        buffer_size = model_config.getfloat('buffer_size')
        self.trans_buffer_ls = []
        for i in range(self.n_agent):
            self.policy_ls[i].prepare_loss(max_grad_norm, gamma)
            self.trans_buffer_ls.append(ReplayBuffer(buffer_size, self.n_step))

    def backward(self, summary_writer=None, global_step=None):
        cur_lr = self.lr_scheduler.get(self.n_step)
        if self.trans_buffer_ls[0].size < self.trans_buffer_ls[0].batch_size:
            return
        for i in range(self.n_agent):
            for k in range(10):
                obs, acts, next_obs, rs, dones = self.trans_buffer_ls[i].sample_transition()
                if i == 0:
                    self.policy_ls[i].backward(self.sess, obs, acts, next_obs, dones, rs, cur_lr,
                                               summary_writer=summary_writer,
                                               global_step=global_step + k)
                else:
                    self.policy_ls[i].backward(self.sess, obs, acts, next_obs, dones, rs, cur_lr)

    def forward(self, obs, mode='act', stochastic=False):
        eps = None
        if mode == 'explore':
            eps = self.eps_scheduler.get(1)
        action = []
        qs_ls = []
        for i in range(self.n_agent):
            qs = self.policy_ls[i].forward(self.sess, obs[i])
            if (mode == 'explore') and (np.random.random() < eps):
                action.append(np.random.randint(self.num_action_ls[i]))
            else:
                if not stochastic:
                    action.append(np.argmax(qs))
                else:
                    qs = qs / np.sum(qs)
                    action.append(np.random.choice(np.arange(len(qs)), p=qs))
            qs_ls.append(qs)
        return action, qs_ls

    def reset(self):
        # do nothing
        return

    def add_transition(self, obs, actions, rewards, next_obs, done):
        if (self.reward_norm):
            rewards = rewards / self.reward_norm
        if self.reward_clip:
            rewards = np.clip(rewards, -self.reward_clip, self.reward_clip)
        for i in range(self.n_agent):
            self.trans_buffer_ls[i].add_transition(obs[i], actions[i],
                                                   rewards[i], next_obs[i], done)